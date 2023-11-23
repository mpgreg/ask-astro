import os
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from include.tasks import ingest, split
from include.tasks.utils.retrieval_tests import weaviate_qna, generate_crc, generate_hybrid_crc
from include.tasks.utils.schema import check_schema_subset

from airflow.decorators import dag, task
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.suite.operators.sheets import GoogleSheetsCreateSpreadsheetOperator
# from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
from include.utils.weaviate.hooks.weaviate import _WeaviateHook
from airflow.operators.python import get_current_context
from weaviate.exceptions import UnexpectedStatusCodeException

ask_astro_env = os.environ.get("ASK_ASTRO_ENV", "")

WEAVIATE_CLASS = os.environ.get("WEAVIATE_CLASS", "DocsProd")

_WEAVIATE_CONN_ID = os.environ.get("WEAVIATE_CONN_ID", f"weaviate_{ask_astro_env}")
_GCP_CONN_ID = 'google_cloud_default'

weaviate_hook = _WeaviateHook(_WEAVIATE_CONN_ID)
weaviate_client = weaviate_hook.get_client()

test_question_template_path = Path("include/data/test_questions_template.csv")

results_bucket = 'ask-astro-test-results'
results_bucket_prefix = 'test_pipeline/'

seed_baseline_url = (
    "https://astronomer-demos-public-readonly.s3.us-west-2.amazonaws.com/ask-astro/baseline_data_v3.parquet"
)

test_doc_link = "https://registry.astronomer.io/providers/apache-airflow/versions/2.7.3/modules/SmoothOperator"

test_doc_chunk1 = "# TEST TITLE\n## TEST SECTION\n" + "".join(["TEST " for a in range(0, 400)])
test_doc_chunk2 = "".join(["TEST " for a in range(0, 400)])
test_doc_content = "\n\n".join([test_doc_chunk1, test_doc_chunk2])


@dag(schedule_interval=None, start_date=datetime(2023, 9, 27), catchup=False, is_paused_upon_creation=True)
def test_ask_astro_load_baseline():
    """
    This DAG performs a test of the initial load of data from sources from a seed baseline.  

    It downloads a set of test questions as a CSV file from Google Cloud Storage and generates 
    answers based on a set of tests specified in include.tasks.utils.retrieval_tests. 
    Retrieved answers and references are saved as CSV and uploaded to Google Cloud Storage. 
    
    It then forces  
    a upsert of an existing document, checks the upserted values and then re-upserts the original document.

    The function also includes tasks to test document upsert. After a seed baseline is ingested, an 
    incremental load of a test document is ingested, changed for upsert and checked. These are currently 
    commented out and, in the future, these should be moved to another DAG.
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

        return (
            ["get_existing_doc", "create_test_object"]
            if weaviate_hook.check_schema(class_objects=class_objects)
            else None
        )
    
    @task()
    def get_existing_doc(doc_link:str) -> list[pd.DataFrame]:
        """
        Import an existing document that was added from the baseline ingest.
        """

        existing_doc = (
            weaviate_client.query.get(properties=["docSource", "sha", "content", "docLink"], 
                                      class_name=WEAVIATE_CLASS)
            .with_limit(1)
            .with_additional(["id", "vector"])
            .with_where({"path": ["docLink"], "operator": "Equal", "valueText": doc_link})
            .do()
        )

        existing_doc = pd.DataFrame(existing_doc["data"]["Get"][WEAVIATE_CLASS])
        existing_doc.drop("_additional", axis=1, inplace=True)

        return [existing_doc]

    @task()
    def create_test_object(original_doc: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        Create a test object with known data with sufficient size to be split into two chunks.
        """
        new_doc = original_doc[0][["docSource", "sha", "content", "docLink"]]
        new_doc["content"] = test_doc_content

        return [new_doc]

    @task()
    def check_test_objects(original_doc: list[pd.DataFrame], doc_link:str) -> None:
        """
        Check the upserted doc against expected.
        """

        new_docs = (
            weaviate_client.query.get(properties=["docSource", "sha", "content", "docLink"], 
                                      class_name=WEAVIATE_CLASS)
            .with_limit(10)
            .with_additional(["id", "vector"])
            .with_where({"path": ["docLink"], "operator": "Equal", "valueText": doc_link})
            .do()
        )

        new_docs = new_docs["data"]["Get"][WEAVIATE_CLASS]

        assert len(new_docs) == 2

        assert new_docs[0]["content"] + " " == test_doc_chunk1 or new_docs[0]["content"] + " " == test_doc_chunk2

        assert new_docs[1]["content"] + " " == test_doc_chunk1 or new_docs[1]["content"] + " " == test_doc_chunk2

        assert new_docs[0]["docLink"] == original_doc[0].docLink[0]
        assert new_docs[0]["docSource"] == original_doc[0].docSource[0]

    @task()
    def check_original_object(original_doc: list[pd.DataFrame], doc_link:str) -> None:
        """
        Check the re-upserted doc against original.
        """

        new_docs = (
            weaviate_client.query.get(properties=["docSource", "sha", "content", "docLink"], 
                                      class_name=WEAVIATE_CLASS)
            .with_limit(10)
            .with_additional(["id", "vector"])
            .with_where({"path": ["docLink"], "operator": "Equal", "valueText": doc_link})
            .do()
        )

        new_docs = new_docs["data"]["Get"][WEAVIATE_CLASS]

        assert len(new_docs) == 1

        print(original_doc[0].to_json())

    _download_test_questions = GCSToLocalFilesystemOperator(
        task_id="download_test_questions",
        gcp_conn_id=_GCP_CONN_ID,
        object_name=results_bucket_prefix + test_question_template_path.parts[-1],
        filename=test_question_template_path,
        bucket=results_bucket
    )

    @task()
    def generate_test_answers(test_question_template_path:Path, ts_nodash=None):
        """
        Given a set test questions (csv) run the weaviate_qna, generate_crc tests in 
        include.tasks.utils.retrieval_tests.  Saves results in a csv file name with 
        the DAG run timestamp. 
        """
    
        results_file = f'include/data/test_questions_{ts_nodash}.csv'

        csv_columns = [
            'test_number',
            'question',
            'expected_references',
            'vectordb_answer',
            'vectordb_references',
            'crc_answer',
            'crc_references',
            'langsmith_link',
            'hybrid_crc_answer',
            'hybrid_crc_references',
            'hybrid_langsmith_link'
        ]

        questions_df=pd.read_csv(test_question_template_path)

        questions_df[["vectordb_answer", "vectordb_references"]] = questions_df\
            .question.apply(lambda x: weaviate_qna(
                weaviate_client=weaviate_client, 
                question=x, 
                class_name=WEAVIATE_CLASS))

        questions_df[['crc_answer', 'crc_references', 'langsmith_link']] = questions_df\
            .question.apply(lambda x: generate_crc(
                weaviate_client=weaviate_client,
                question=x,
                class_name=WEAVIATE_CLASS,
                ts_nodash=ts_nodash,
                send_feedback=False))
        
        questions_df[['hybrid_crc_answer', 'hybrid_crc_references', 'hybrid_langsmith_link']] = questions_df\
            .question.apply(lambda x: generate_hybrid_crc(
                weaviate_client=weaviate_client,
                question=x,
                class_name=WEAVIATE_CLASS,
                ts_nodash=ts_nodash,
                send_feedback=False))
        
        questions_df[csv_columns].to_csv(results_file, index=False)

        return results_file

    _results_file = generate_test_answers(test_question_template_path=test_question_template_path)

    LocalFilesystemToGCSOperator(
        task_id='upload_results',
        gcp_conn_id=_GCP_CONN_ID,
        src=_results_file,
        dst='test_pipeline/',
        bucket=results_bucket
    )

    # GoogleSheetsCreateSpreadsheetOperator(
    #     task_id='create_sheet',
    #     gcp_conn_id=_GCP_CONN_ID,
    #     spreadsheet={}
    # )

    _get_schema = get_schema()
    _check_schema = check_schema(class_objects=_get_schema)
    original_doc = get_existing_doc(doc_link=test_doc_link)
    test_doc = create_test_object(original_doc)

    split_test_doc = task(split.split_markdown).expand(dfs=[test_doc])

    _upsert_test_doc = (
        task(ingest.import_data, retries=10)
        .partial(
            weaviate_conn_id=_WEAVIATE_CONN_ID,
            class_name=WEAVIATE_CLASS,
            existing="upsert",
            doc_key="docLink",
            batch_params={"batch_size": 1000},
            verbose=True,
        )
        .expand(dfs=[split_test_doc])
    )

    _check_test_objects = check_test_objects(
      original_doc=original_doc, 
      doc_link=test_doc_link
    )

    split_original_doc = task(split.split_markdown).expand(dfs=[original_doc])

    _reupsert_original_doc = (
        task(ingest.import_data, retries=10)
        .partial(
            weaviate_conn_id=_WEAVIATE_CONN_ID,
            class_name=WEAVIATE_CLASS,
            existing="upsert",
            doc_key="docLink",
            batch_params={"batch_size": 1000},
            verbose=True,
        )
        .expand(dfs=[split_original_doc])
    )

    _check_original_object = check_original_object(
      original_doc=original_doc, 
      doc_link=test_doc_link
    )

    _check_schema >> _results_file
    _check_schema >> original_doc \
        >> test_doc \
            >> _upsert_test_doc \
                >> _check_test_objects \
                        >> _reupsert_original_doc \
                            >> _check_original_object
    
    _download_test_questions >> _results_file

test_ask_astro_load_baseline()
