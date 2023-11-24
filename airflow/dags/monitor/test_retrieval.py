import os
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from include.tasks import ingest
from include.tasks.utils.retrieval_tests import weaviate_qna, generate_crc, generate_hybrid_crc

from airflow.decorators import dag, task
from airflow.exceptions import AirflowException
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.suite.operators.sheets import GoogleSheetsCreateSpreadsheetOperator
# from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
from include.utils.weaviate.hooks.weaviate import _WeaviateHook

ask_astro_env = os.environ.get("ASK_ASTRO_ENV", "")

WEAVIATE_CLASS = os.environ.get("WEAVIATE_CLASS", "DocsLocal")

_WEAVIATE_CONN_ID = os.environ.get("WEAVIATE_CONN_ID", f"weaviate_{ask_astro_env}")
_GCP_CONN_ID = 'google_cloud_default'

weaviate_hook = _WeaviateHook(_WEAVIATE_CONN_ID)
weaviate_client = weaviate_hook.get_client()

test_question_template_path = Path("include/data/test_questions_template.csv")

results_bucket = 'ask-astro-test-results'
results_bucket_prefix = 'test_pipeline/'

seed_baseline_url = (
    "https://astronomer-demos-public-readonly.s3.us-west-2.amazonaws.com/ask-astro/baseline_data_v4.parquet"
)
@dag(schedule_interval=None, start_date=datetime(2023, 9, 27), catchup=False, is_paused_upon_creation=True)
def test_retrieval():
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

    @task.branch(trigger_rule="none_failed")
    def check_schema(class_objects: dict) -> str:
        """
        Check if the current schema includes the requested schema.  The current schema could be a superset
        so check_schema_subset is used recursively to check that all objects in the requested schema are
        represented in the current schema.
        """
        
        if weaviate_hook.check_schema(class_objects=class_objects):
            return {"check_doc_count"}
        else:
            raise AirflowException(f"""
                Class does not exist in current schema. Create it with 
                'weaviate_hook.create_schema(class_objects=class_objects, existing="error")'
                """)
    
    @task.branch(trigger_rule="none_failed")
    def check_doc_count(class_objects: dict, expected_count: int) -> str:
        """
        Check if the vectordb has AT LEAST expected_count objects.
        """
        
        count = weaviate_hook.client.query.aggregate(WEAVIATE_CLASS).with_meta_count().do()
        doc_count = count['data']['Aggregate'][WEAVIATE_CLASS][0]['meta']['count']

        if doc_count >= expected_count:
            return {"download_test_questions"}
        elif doc_count == 0:
            return {"import_baseline"}
        else:
            raise AirflowException("Unknown vectordb state. Ingest baseline or change expected_count.")
    
    _download_test_questions = GCSToLocalFilesystemOperator(
        trigger_rule="none_failed",
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

    _upload_results = LocalFilesystemToGCSOperator(
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

    _import_baseline = (
        task(ingest.import_baseline, trigger_rule="none_failed")(
            weaviate_conn_id=_WEAVIATE_CONN_ID,
            seed_baseline_url=seed_baseline_url,
            class_name=WEAVIATE_CLASS,
            existing="error",
            uuid_column="id",
            vector_column="vector",
            batch_params={"batch_size": 1000},
            verbose=True,
        )
    )

    _get_schema = get_schema()
    _check_schema = check_schema(class_objects=_get_schema)
    _check_doc_count = check_doc_count(class_objects=_get_schema, expected_count=36860)

    _check_schema >> _check_doc_count >> [_import_baseline, _download_test_questions]
    
    _import_baseline >> _download_test_questions >> _results_file >> _upload_results

test_retrieval()
