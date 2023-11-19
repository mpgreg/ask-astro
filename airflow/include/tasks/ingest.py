from __future__ import annotations

import warnings

import pandas as pd
import requests
from include.tasks.utils.ingest import _objects_to_upsert
from weaviate.util import generate_uuid5

from airflow.exceptions import AirflowException
from airflow.providers.weaviate.hooks.weaviate import WeaviateHook


def import_data(
    weaviate_conn_id: str,
    dfs: list[pd.DataFrame],
    class_name: str,
    existing: str = "skip",
    doc_key: str = None,
    uuid_column: str = None,
    vector_column: str = None,
    batch_params: dict = None,
    verbose: bool = True,
) -> list:
    """
    This task concatenates multiple dataframes from upstream dynamic tasks and vectorizes with import to weaviate.
    The operator returns a list of any objects that failed to import.

    A 'uuid' is generated based on the content and metadata (the git sha, document url, the document source and a
    concatenation of the headers) and Weaviate will create the vectors.

    Upsert and logic relies on a 'doc_key' which is a uniue representation of the document.  Because documents can
    be represented as multiple chunks (each with a UUID which is unique in the DB) the doc_key is a way to represent
    all chunks associated with an ingested document.

    param dfs: A list of dataframes from downstream dynamic tasks
    type dfs: list[pd.DataFrame]

    param class_name: The name of the class to import data.  Class should be created with weaviate schema.
    type class_name: str

    param existing: Whether to 'upsert', 'skip' or 'replace' any existing documents.  Default is 'skip'.
    type existing: str

    param doc_key: If using upsert you must specify a doc_key which uniquely identifies a document which may or
    may not include multiple (unique) chunks.
    param doc_key: str

    :param vector_column: For pre-embedded data specify the name of the column containing the embedding vector
    :type vector_column: str

    :param uuid_column: For data with pre-genenerated UUID specify the name of the column containing the UUID
    :type uuid_column: str
    """

    if existing not in ["skip", "replace", "upsert"]:
        raise AirflowException("Invalid parameter for 'existing'.  Choices are 'skip', 'replace', 'upsert'")

    weaviate_hook = WeaviateHook(weaviate_conn_id)
    weaviate_client = weaviate_hook.get_client()

    df = pd.concat(dfs, ignore_index=True)

    # Without a pre-geneerated UUID weaviate ingest just creates one with uuid.uuid4()
    # This will lead to duplicates in vector db.
    if uuid_column is None:
        # reorder columns alphabetically for consistent uuid mapping
        column_names = df.columns.to_list()
        column_names.sort()
        df = df[column_names]

        print("No uuid_column provided Generating UUIDs for ingest.")
        if "id" in column_names:
            raise AirflowException("Property 'id' already in dataset. Consider renaming or specify 'uuid_column'.")
        else:
            uuid_column = "id"

        df[uuid_column] = df.drop(columns=[vector_column], inplace=False, errors="ignore").apply(
            lambda row: generate_uuid5(identifier=row.to_dict(), namespace=class_name), axis=1
        )

        if df[uuid_column].duplicated().any():
            raise AirflowException("Duplicate rows found.  Remove duplicates before ingest.")

    if existing == "upsert":
        if doc_key is None:
            raise AirflowException("Must specify 'doc_key' if 'existing=upsert'.")
        else:
            if df[[doc_key, uuid_column]].duplicated().any():
                raise AirflowException("Duplicate rows found.  Remove duplicates before ingest.")

            current_schema = weaviate_client.schema.get(class_name=class_name)
            doc_key_schema = [prop for prop in current_schema["properties"] if prop["name"] == doc_key]

            if len(doc_key_schema) < 1:
                raise AirflowException("doc_key does not exist in current schema.")
            elif doc_key_schema[0]["tokenization"] != "field":
                raise AirflowException(
                    "Tokenization for provided doc_key is not set to 'field'.  Cannot upsert safely."
                )

        # get a list of any UUIDs which need to be removed later
        objects_to_upsert = _objects_to_upsert(
            weaviate_client=weaviate_client, df=df, class_name=class_name, doc_key=doc_key, uuid_column=uuid_column
        )

        df = df[df[uuid_column].isin(objects_to_upsert["objects_to_insert"])]

    print(f"Passing {len(df)} objects for ingest.")

    batch = weaviate_client.batch.configure(**batch_params)

    for row_id, row in df.iterrows():
        data_object = row.to_dict()
        uuid = data_object[uuid_column]

        # if the uuid exists we know that the properties are the same
        if weaviate_client.data_object.exists(uuid=uuid, class_name=class_name) is True:
            if existing == "skip":
                if verbose is True:
                    print(f"UUID {uuid} exists.  Skipping.")
                continue
            elif existing == "replace":
                # Default for weaviate is replace existing
                if verbose is True:
                    print(f"UUID {uuid} exists.  Overwriting.")

        vector = data_object.pop(vector_column, None)
        uuid = data_object.pop(uuid_column)

        added_row = batch.add_data_object(class_name=class_name, uuid=uuid, data_object=data_object, vector=vector)
        if verbose is True:
            print(f"Added row {row_id} with UUID {added_row} for batch import.")

    results = batch.create_objects()

    batch_errors = []
    for item in results:
        if "errors" in item["result"]:
            item_error = {"id": item["id"], "errors": item["result"]["errors"]}
            if verbose:
                print(item_error)
            batch_errors.append(item_error)

    # check errors from callback
    if existing == "upsert":
        if len(batch_errors) > 0:
            warnings.warn("Error during upsert.  Rollling back all inserts.")
            # rollback inserts
            for uuid in objects_to_upsert["objects_to_insert"]:
                print(f"Removing id {uuid} for rollback.")
                weaviate_client.data_object.delete(uuid=uuid, class_name=class_name, consistency_level="ALL")

        elif len(objects_to_upsert["objects_to_delete"]) > 0:
            for uuid in objects_to_upsert["objects_to_delete"]:
                if verbose:
                    print(f"Deleting id {uuid} for successful upsert.")
                weaviate_client.data_object.delete(uuid=uuid, class_name=class_name)

    return batch_errors


def import_baseline(
    weaviate_conn_id: str,
    seed_baseline_url: str,
    class_name: str,
    existing: str = "skip",
    doc_key: str = None,
    uuid_column: str = None,
    vector_column: str = None,
    batch_params: dict = None,
    verbose: bool = True,
) -> list:
    """
    This task ingests data from a baseline of pre-embedded data. This is useful for evaluation and baselining changes
    over time.  This function is used as a python_callable with the weaviate_import decorator.  The returned
    dictionary is passed to the WeaviateImportDataOperator for ingest. The operator returns a list of any objects
    that failed to import.

    seed_baseline_url is a URI for a parquet file of pre-embedded data.

    Any existing documents are replaced.  The assumption is that this is a first import of data and older data
    should be removed.

    param class_name: The name of the class to import data.  Class should be created with weaviate schema.
    type class_name: str

    param seed_baseline_url: The url of a parquet file containing baseline data to ingest.
    param seed_baseline_url: str

    :param vector_column: For pre-embedded data specify the name of the column containing the embedding vector
    :type vector_column: str

    :param uuid_column: For data with pre-genenerated UUID specify the name of the column containing the UUID
    :type uuid_column: str
    """

    seed_filename = f"include/data/{seed_baseline_url.split('/')[-1]}"

    try:
        df = pd.read_parquet(seed_filename)

    except Exception:
        with open(seed_filename, "wb") as fh:
            response = requests.get(seed_baseline_url, stream=True)
            fh.writelines(response.iter_content(1024))

        df = pd.read_parquet(seed_filename)

    return import_data(
        weaviate_conn_id=weaviate_conn_id,
        dfs=[df],
        class_name=class_name,
        existing=existing,
        doc_key=doc_key,
        uuid_column=uuid_column,
        vector_column=vector_column,
        verbose=verbose,
        batch_params=batch_params,
    )
