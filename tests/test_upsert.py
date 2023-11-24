import os
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from include.tasks import ingest, split

from airflow.decorators import dag, task
from airflow.exceptions import AirflowException
# from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
from include.utils.weaviate.hooks.weaviate import _WeaviateHook

ask_astro_env = os.environ.get("ASK_ASTRO_ENV", "")

WEAVIATE_CLASS = os.environ.get("WEAVIATE_CLASS", "DocsLocal")

_WEAVIATE_CONN_ID = os.environ.get("WEAVIATE_CONN_ID", f"weaviate_{ask_astro_env}")

weaviate_hook = _WeaviateHook(_WEAVIATE_CONN_ID)
weaviate_client = weaviate_hook.get_client()

test_question_template_path = Path("include/data/test_questions_template.csv")

test_docLink = "https://registry.astronomer.io/providers/apache-airflow/versions/2.7.3/modules/SmoothOperator"

test_doc_chunk1 = "# TEST TITLE\n## TEST SECTION\n" + "".join(["TEST " for a in range(0, 400)])
test_doc_chunk2 = "".join(["TEST " for a in range(0, 1000)])
test_doc_content = "\n\n".join([test_doc_chunk1, test_doc_chunk2])


@dag(schedule_interval=None, start_date=datetime(2023, 9, 27), catchup=False, is_paused_upon_creation=True)
def test_upsert():
    """
    This DAG performs a test of the upsert logic in the WeaviateHook. An incremental load of a test 
    document is ingested, changed for upsert and checked. 
    """

    @task
    def get_schema(schema_file: str = "include/data/schema.json") -> list:
        """
        Get the schema object for this DAG.
        """

        class_objects = json.loads(Path(schema_file).read_text())
        class_objects["classes"][0].update({"class": WEAVIATE_CLASS})

        if "classes" not in class_objects:
            class_objects = [class_objects]
        else:
            class_objects = class_objects["classes"]

        return class_objects

    @task.branch
    def check_schema(class_objects: dict) -> str:
        """
        Check if the current schema includes the requested schema.  The current schema could be a superset
        so check_schema_subset is used recursively to check that all objects in the requested schema are
        represented in the current schema.
        """
        
        if weaviate_hook.check_schema(class_objects=class_objects):
            return {"get_existing_doc"}
        else:
            raise AirflowException(f"""
                Class {class_objects.get('class')} does not exist in current schema with 
                'weaviate_hook.create_schema(class_objects=class_objects, existing="error")'
                """)
    
    @task()
    def get_existing_doc(docLink:str) -> pd.DataFrame:
        """
        Import an existing document that was added from the baseline ingest.
        """

        existing_doc = (
            weaviate_client.query.get(properties=["content", "docLink"], 
                                    class_name=WEAVIATE_CLASS)
            .with_limit(10)
            .with_additional(["id", "vector"])
            .with_where({"path": ["docLink"], "operator": "Equal", "valueText": docLink})
            .do()
        )

        if existing_doc.get("data"):
            existing_doc = pd.DataFrame(existing_doc["data"]["Get"][WEAVIATE_CLASS])
            existing_doc = pd.concat([
                existing_doc.drop("_additional", axis=1),
                pd.json_normalize(existing_doc["_additional"])
                ], axis=1)
        
        assert len(existing_doc) == 1

        return existing_doc

    @task()
    def create_test_object(existing_doc: pd.DataFrame) -> pd.DataFrame:
        """
        Create a test object with known data with sufficient size to be split into two chunks.
        """
        test_object = existing_doc[["content", "docLink"]]
        test_object["content"] = test_doc_content

        return test_object
    
    @task()
    def create_large_test_object(existing_doc: pd.DataFrame) -> pd.DataFrame:
        """
        Create a large test object which should fail during ingest and trigger the rollback.
        """
        large_test_object = existing_doc[["content", "docLink"]]
        large_test_object["content"] = test_doc_content * 50

        return large_test_object
    
    @task()
    def create_bad_test_object(existing_doc: pd.DataFrame) -> pd.DataFrame:
        """
        Create a test object with a bad UUID which should fail during ingest and trigger the rollback.
        """
        bad_test_object = existing_doc[["content", "docLink"]]
        bad_test_object["id"] = 'NON-WORKING-UUID'

        return bad_test_object

    @task()
    def upsert_large_object(large_test_object:pd.DataFrame):

        expected_error = "update vector: connection to: OpenAI API failed with status: 400 error: This model's maximum context length is 8192 tokens, however you requested 70477 tokens (70477 in your prompt; 0 for the completion). Please reduce your prompt; or completion length."

        upsert_errors = weaviate_hook.ingest_data(
                df=large_test_object,
                class_name=WEAVIATE_CLASS,
                existing="upsert",
                doc_key="docLink",
                verbose=True,
                error_threshold=1
            )
        
        assert upsert_errors[0]['errors']['error'][0]['message'] == expected_error
        
    @task()
    def upsert_bad_object(bad_test_object:pd.DataFrame):

        expected_error = "Not valid 'uuid' or 'uuid' can not be extracted from value"

        upsert_errors = weaviate_hook.ingest_data(
                df=bad_test_object,
                class_name=WEAVIATE_CLASS,
                uuid_column='id',
                existing="upsert",
                doc_key="docLink",
                verbose=True,
                error_threshold=1
            )
        
        assert upsert_errors[0]['errors'][0] == expected_error
    
    @task()
    def upsert_object(test_object:pd.DataFrame):

        df = split.split_markdown(dfs=[test_object])

        upsert_errors = weaviate_hook.ingest_data(
                df=df,
                class_name=WEAVIATE_CLASS,
                existing="upsert",
                doc_key="docLink",
                verbose=True,
                error_threshold=0
            )
        
        assert upsert_errors == None
    
    @task()
    def check_test_objects(existing_doc: pd.DataFrame, docLink:str):
        """
        Check the upserted doc against expected.
        """

        new_docs = (
            weaviate_client.query.get(properties=["content", "docLink"], 
                                      class_name=WEAVIATE_CLASS)
            .with_limit(10)
            .with_additional(["id", "vector"])
            .with_where({"path": ["docLink"], "operator": "Equal", "valueText": docLink})
            .do()
        )
        new_docs = pd.DataFrame(new_docs["data"]["Get"][WEAVIATE_CLASS])
        new_docs = pd.concat([
            new_docs.drop("_additional", axis=1),
            pd.json_normalize(new_docs["_additional"])
            ], axis=1)

        assert len(new_docs) == 3

        assert "TEST TEST TEST" in new_docs.content[0]
        assert "TEST TEST TEST" in new_docs.content[1]
        assert "TEST TEST TEST" in new_docs.content[2]

        assert new_docs.docLink[0] == existing_doc.docLink[0]

    @task()
    def re_upsert_original(existing_doc:pd.DataFrame):

        upsert_errors = weaviate_hook.ingest_data(
                df=existing_doc,
                class_name=WEAVIATE_CLASS,
                existing="upsert",
                doc_key="docLink",
                uuid_column="id",
                vector_column="vector",
                verbose=True,
                error_threshold=0
            )
        
        assert upsert_errors == None
    
    @task()
    def check_original_object(existing_doc:pd.DataFrame, docLink:str):
        """
        Check the re-upserted doc against original.
        """

        new_docs = (
            weaviate_client.query.get(properties=["content", "docLink"], 
                                      class_name=WEAVIATE_CLASS)
            .with_limit(10)
            .with_additional(["id", "vector"])
            .with_where({"path": ["docLink"], "operator": "Equal", "valueText": docLink})
            .do()
        )
        new_docs = pd.DataFrame(new_docs["data"]["Get"][WEAVIATE_CLASS])
        new_docs = pd.concat([
            new_docs.drop("_additional", axis=1),
            pd.json_normalize(new_docs["_additional"])
            ], axis=1)

        assert new_docs.content[0] == existing_doc.content[0]
        assert new_docs.docLink[0] == existing_doc.docLink[0]
        assert new_docs.id[0] == existing_doc.id[0]

    _get_schema = get_schema()
    _check_schema = check_schema(class_objects=_get_schema)
    original_doc = get_existing_doc(test_docLink)
    test_doc = create_test_object(original_doc)
    large_doc = create_large_test_object(original_doc)
    bad_doc = create_bad_test_object(original_doc)
    upsert_large_object(large_doc)
    upsert_bad_object(bad_doc)
    _upserted_doc = upsert_object(test_doc)
    _check_test_objects = check_test_objects(original_doc, test_docLink)
    _re_upsert_original = re_upsert_original(original_doc)
    _recheck_original = check_original_object(original_doc, test_docLink)

    _check_schema >> original_doc
    _upserted_doc >> _check_test_objects >> _re_upsert_original >> _recheck_original

test_upsert()
