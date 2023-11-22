from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from weaviate.exceptions import UnexpectedStatusCodeException
from weaviate.util import generate_uuid5

from airflow.exceptions import AirflowException
from airflow.providers.weaviate.hooks.weaviate import WeaviateHook


class _WeaviateHook(WeaviateHook):
    """Extends the WeaviateHook to include specific methods for handling Ask-Astro."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("airflow.task")
        self.client = self.get_client()

    def compare_schema_subset(self, class_object: Any, class_schema: Any) -> bool:
        """
        Recursively check if requested schema/object is a subset of the current schema.

        :param class_object: The class object to check against current schema
        :param class_schema: The current schema class object
        """

        # Direct equality check
        if class_object == class_schema:
            return True

        # Type mismatch early return
        if type(class_object) != type(class_schema):
            return False

        # Dictionary comparison
        if isinstance(class_object, dict):
            return all(
                k in class_schema and self.compare_schema_subset(v, class_schema[k]) for k, v in class_object.items()
            )

        # List or Tuple comparison
        if isinstance(class_object, (list, tuple)):
            return len(class_object) == len(class_schema) and all(
                self.compare_schema_subset(obj, sch) for obj, sch in zip(class_object, class_schema)
            )

        # Default case for non-matching types or unsupported types
        return False

    def is_class_missing(self, class_object: dict) -> bool:
        """
        Checks if a class is missing from the schema.

        :param class_object: Class object to be checked against the current schema.
        """
        try:
            class_schema = self.client.schema.get(class_object.get("class", ""))
            return self.compare_schema_subset(class_object=class_object, class_schema=class_schema)
        except UnexpectedStatusCodeException as e:
            return e.status_code == 404 and "with response body: None." in e.message
        except Exception as e:
            self.logger.error(f"Error checking schema: {e}")
            raise ValueError(f"Error during schema check {e}")

    def check_schema(self, class_objects: list) -> bool:
        """
        Verifies if the current schema includes the requested schema.

        :param class_objects: Class objects to be checked against the current schema.
        """
        try:
            missing_objects = [obj["class"] for obj in class_objects if self.is_class_missing(obj)]

            if missing_objects:
                self.logger.warning(f"Classes {missing_objects} are not in the current schema.")
                return False
            else:
                self.logger.info("All classes are present in the current schema.")
                return True
        except Exception as e:
            self.logger.error(f"Error during schema check: {e}")
            raise ValueError(f"Error during schema check {e}")

    def create_schema(self, class_objects: list, existing: str = "ignore") -> None:
        """
        Creates or updates the schema in Weaviate based on the given class objects.

        :param class_objects: A list of class objects for schema creation or update.
        :param existing: Strategy to handle existing classes ('ignore' or 'replace'). Defaults to 'ignore'.
        """
        for class_object in class_objects:
            class_name = class_object.get("class", "")
            self.logger.info(f"Processing schema for class: {class_name}")

            try:
                current_class = self.client.schema.get(class_name=class_name)
            except Exception as e:
                self.logger.error(f"Error retrieving current class schema: {e}")
                current_class = None
            if current_class is not None and existing == "replace":
                self.logger.info(f"Replacing existing class {class_name}")
                self.client.schema.delete_class(class_name=class_name)

            if current_class is None or existing == "replace":
                self.client.schema.create_class(class_object)
                self.logger.info(f"Created/updated class {class_name}")

    def generate_uuids(
        self, 
        df: pd.DataFrame, 
        class_name: str, 
        column_subset: list[str] | None = None,
        vector_column: str | None = None,
        uuid_column: str | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """
        Adds a uuid to a dataframe.  This is useful if performing upsert as the UUID must be known before ingest.
        By default weaviate calls uuid4() if a uuid_column is not specified.  This can ingest the same data 
        multiple times with different UUIDs.

        :param df: A dataframe with data to generate a UUID from.
        :param class_name: The name of the class use as part of the uuid namespace.
        :param uuid_column: Name of the column to create. Default is 'id'.
        :param column_subset: A list of columns to use for UUID generation. By default all columns except 
            vector_column will be used.
        :param vector_column: Name of the column containing the vector data.  If specified the vector will be 
            removed prior to generating the uuid.
        :return: A DataFrame with a generated UUID and the name of the uuid column.
        """
        
        column_names = df.columns.to_list()

        column_subset = column_subset or column_names        
        column_subset.sort()

        if uuid_column is None:
            self.logger.info(f"No uuid_column provided. Generating UUIDs as column name {uuid_column}.")
            uuid_column = 'id'

        if uuid_column in column_names:
            raise AirflowException(
                f"Property {uuid_column} already in dataset. Consider renaming or specify a different 'uuid_column'."
                )

        df[uuid_column] = df[column_subset].drop(
            columns=[vector_column], 
            inplace=False, 
            errors="ignore").apply(
                lambda row: generate_uuid5(identifier=row.to_dict(), namespace=class_name), axis=1
            )

        return df, uuid_column

    def identify_upsert_targets(self, df: pd.DataFrame, class_name: str, doc_key: str, uuid_column: str) -> pd.DataFrame:
        """
        Handles the 'upsert' operation for data ingestion.

        :param df: The DataFrame containing the data to be upserted.
        :param class_name: The name of the class to import data.
        :param doc_key: The document key used for upsert operation. This is a property of the data that 
            uniquely identifies all chunks associated with one document.
        :param uuid_column: The column name containing the UUID.
        :return: A dataframe of objects to insert, delete, or leave unchanged as well as their doc_key
        """
        if doc_key is None or not doc_key in df.columns:
            raise AirflowException("Specified doc_key is not specified or not in the dataset.")
        
        if uuid_column is None or not uuid_column in df.columns:
            raise AirflowException("Specified uuid_column is not specified or not in the dataset.")

        if df[[doc_key, uuid_column]].duplicated().any():
            raise AirflowException("Duplicate rows found. Remove duplicates before upsert.")

        current_schema = self.client.schema.get(class_name=class_name)
        doc_key_schema = [prop for prop in current_schema["properties"] if prop["name"] == doc_key]

        if len(doc_key_schema) < 1:
            raise AirflowException("doc_key does not exist in current schema.")
        elif doc_key_schema[0]["tokenization"] != "field":
            raise AirflowException("Tokenization for provided doc_key is not set to 'field'. Cannot upsert safely.")

        ids_df = df.groupby(doc_key)[uuid_column].apply(set).reset_index(name="new_ids")
        ids_df["existing_ids"] = ids_df[doc_key].apply(
            lambda x: self._query_objects(value=x, doc_key=doc_key, uuid_column=uuid_column, class_name=class_name)
        )

        ids_df["objects_to_insert"] = ids_df.apply(lambda x: list(x.new_ids.difference(x.existing_ids)), axis=1)
        ids_df["objects_to_delete"] = ids_df.apply(lambda x: list(x.existing_ids.difference(x.new_ids)), axis=1)
        ids_df["unchanged_objects"] = ids_df.apply(lambda x: x.new_ids.intersection(x.existing_ids), axis=1)

        return ids_df[[doc_key, "objects_to_insert", "objects_to_delete", "unchanged_objects"]]

    def batch_ingest(
        self,
        df: pd.DataFrame,
        class_name: str,
        uuid_column: str,
        existing: str,
        vector_column: str | None = None,
        batch_params: dict = {},
        verbose: bool = False,
    ) -> (list, Any):
        """
        Processes the DataFrame and batches the data for ingestion into Weaviate.

        :param df: DataFrame containing the data to be ingested.
        :param class_name: The name of the class in Weaviate to which data will be ingested.
        :param uuid_column: Name of the column containing the UUID.
        :param vector_column: Name of the column containing the vector data.
        :param batch_params: Parameters for batch configuration.
        :param existing: Strategy to handle existing data ('skip', 'replace', 'upsert').
        :param verbose: Whether to print verbose output.
        :return: List of any objects that failed to be added to the batch.
        """
        self.client.batch.configure(**batch_params)
        batch_errors = []

        with self.client.batch as batch:
            for row_id, row in df.iterrows():
                data_object = row.to_dict()
                uuid = data_object.pop(uuid_column)
                vector = data_object.pop(vector_column, None)

                # Check if the uuid exists and handle accordingly
                if self.client.data_object.exists(uuid=uuid, class_name=class_name):
                    if existing == "skip":
                        if verbose:
                            self.logger.warning(f"UUID {uuid} exists. Skipping.")
                        continue
                    elif existing == "replace":
                        if verbose:
                            self.logger.warning(f"UUID {uuid} exists. Overwriting.")

                try:
                    batch.add_data_object(class_name=class_name, uuid=uuid, data_object=data_object, vector=vector)
                    if verbose:
                        self.logger.info(f"Added row {row_id} with UUID {uuid} for batch import.")
                except Exception as e:
                    if verbose:
                        self.logger.error(f"Failed to add row {row_id} with UUID {uuid}. Error: {e}")
                    batch_errors.append({"row_id": row_id, "uuid": uuid, "error": str(e)})

        return batch_errors

    def process_batch_errors(self, results: list, verbose: bool) -> list:
        """
        Processes the results from batch operation and collects any errors.

        :param results: Results from the batch operation.
        :param verbose: Flag to enable verbose logging.
        :return: List of error messages.
        """
        batch_errors = []
        for item in results:
            if "errors" in item["result"]:
                item_error = {"id": item["id"], "errors": item["result"]["errors"]}
                if verbose:
                    self.logger.info(item_error)
                batch_errors.append(item_error)
        return batch_errors

    def handle_upsert_rollback(self, objects_to_upsert: pd.DataFrame, class_name: str, verbose: bool):
        """
        Handles rollback of inserts in case of errors during upsert operation.

        :param objects_to_upsert: Dictionary of objects to upsert.
        :param class_name: Name of the class in Weaviate.
        :param verbose: Flag to enable verbose logging.
        """
        for each object in errors identify the doc_key
                for each doc_key identify all objects from objects_to_insert
                remove only those associated with this doc_key


        # objects_to_delete = {item for sublist in objects_to_upsert.objects_to_delete.tolist() for item in sublist}


        for uuid in objects_to_upsert["objects_to_insert"]:
            self.logger.info(f"Removing id {uuid} for rollback.")
            if self.client.data_object.exists(uuid=uuid, class_name=class_name):
                self.client.data_object.delete(uuid=uuid, class_name=class_name, consistency_level="ALL")
            elif verbose:
                self.logger.info(f"UUID {uuid} does not exist. Skipping deletion.")

        for uuid in objects_to_upsert["objects_to_delete"]:
            if verbose:
                self.logger.info(f"Deleting id {uuid} for successful upsert.")
            if self.client.data_object.exists(uuid=uuid, class_name=class_name):
                self.client.data_object.delete(uuid=uuid, class_name=class_name)
            elif verbose:
                self.logger.info(f"UUID {uuid} does not exist. Skipping deletion.")

    def ingest_data(
        self,
        df: pd.DataFrame,
        class_name: str,
        existing: str = "skip",
        doc_key: str = None,
        uuid_column: str = None,
        vector_column: str = None,
        batch_params: dict = None,
        verbose: bool = True,
    ) -> list:
        """
        This function ingests df to Weaviate and returns a list of any objects that failed to import.

        Upsert logic relies on a 'doc_key' which is a uniue representation of the document.  Because documents can
        be represented as multiple chunks (each with a UUID which is unique in the DB) the doc_key is a way to represent
        all chunks associated with an ingested document.  Rollback is performed if errors are encountered on ingest.

        Note: Upsert is partially atomic at the document level. If all objects (UUIDs) associated with a document are 
        ingested it will not be rolled-back.  This allows airflow task retries to pickukp where it left off without 
        starting the entire ingest from the beginning and incurring additional costs for documents that ingested without
        error.

        :param df: A pandas dataframes
        :param class_name: The name of the class to import data.  Class should be created with weaviate schema.
        :param existing: Whether to 'upsert', 'skip' or 'replace' any existing documents.  Default is 'skip'.
        :param doc_key: If using upsert you must specify a doc_key as a column of the dataframe which uniquely 
        identifies a document which may or may not consist of multiple (unique) chunks.
        :param vector_column: For pre-embedded data specify the name of the column containing the embedding vector
        :param uuid_column: For data with pre-generated UUID specify the name of the column containing the UUID. If a
            uuid_column is not specified the method will attempt to create and generate UUIDs.
        :param batch_params: Additional parameters to pass to the weaviate batch configuration
        :param verbose: Whether to print verbose output
        :return: A list of object insert errors, if any.
        """

        if existing not in ["skip", "replace", "upsert"]:
            raise AirflowException("Invalid parameter for 'existing'. Choices are 'skip', 'replace', 'upsert'")

        if existing == "upsert":
            objects_to_upsert = self.identify_upsert_targets(df, class_name, doc_key, uuid_column)

            objects_to_insert = {item for sublist in objects_to_upsert.objects_to_insert.tolist() for item in sublist}

            #subset df with only objects that need to be inserted
            df = df[df[uuid_column].isin(objects_to_insert)]

        self.logger.info(f"Passing {len(df)} objects for ingest.")

        batch_errors = self.batch_ingest(
            df, class_name, uuid_column, vector_column, batch_params, existing, verbose
        )

        batch_errors += self.process_batch_errors(batch_errors, verbose)

        if existing == "upsert" and batch_errors:
            self.logger.warning("Error during upsert. Rolling back all inserts for docs with errors.")
            self.handle_upsert_rollback(objects_to_upsert, class_name, verbose)

        return batch_errors

    def _query_objects(self, value: Any, doc_key: str, class_name: str, uuid_column: str) -> set:
        """
        Check for existence of a data_object as a property of a data class and return all object ids.

        :param value: The value of the property to query.
        :param doc_key: The name of the property to query.
        :param class_name: The name of the class to query.
        :param uuid_column: The name of the column containing the UUID.
        """
        existing_uuids = (
            self.client.query.get(properties=[doc_key], class_name=class_name)
            .with_additional([uuid_column])
            .with_where({"path": doc_key, "operator": "Equal", "valueText": value})
            .do()["data"]["Get"][class_name]
        )

        return {additional["_additional"]["id"] for additional in existing_uuids}
