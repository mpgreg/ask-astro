import os
from datetime import datetime

from include.tasks import ingest, split
from include.tasks.extract import html

from airflow.decorators import dag, task

ask_astro_env = os.environ.get("ASK_ASTRO_ENV", "local")

_WEAVIATE_CONN_ID = os.environ.get("WEAVIATE_CONN_ID", f"weaviate_{ask_astro_env}")
WEAVIATE_CLASS = os.environ.get("WEAVIATE_CLASS", "DocsLocal")

html_docs_sources = [
    {
        "base_url": "https://astronomer-providers.readthedocs.io/en/stable/", 
        "exclude_docs": [r"/changelog.html"], 
        "container_class": "body"
    },
    {
        "base_url": "https://astro-sdk-python.readthedocs.io/en/stable/", 
        "exclude_docs": [r"/changelog.html"], 
        "container_class": "body"
    },
    {
        "base_url": "https://www.astronomer.io/blog/", 
        "exclude_docs": [r"/\d+/", r"/\d+$"], 
        "container_class": "prose"
    },
    {
        "base_url": "https://docs.astronomer.io/astro/", 
        "exclude_docs": [], 
        "container_class": "theme-doc-markdown markdown"
    },
    {
        "base_url": "https://docs.astronomer.io/learn/", 
        "exclude_docs": [r'learn/category', r'learn/tags'], 
        "container_class": "theme-doc-markdown markdown"
    },
    {
        "base_url": "https://airflow.apache.org/docs/",
        "exclude_docs": [
            r"/changelog.html",
            r"/commits.html",
            r"_api",
            r"_modules",
            r"apache-airflow/1.",
            r"apache-airflow/2.",
            r"example",
        ],
        "container_class": "body"
    }
]

default_args = {"retries": 3, "retry_delay": 30}

schedule_interval = "0 5 * * *" if ask_astro_env == "prod" else None


@dag(
    schedule_interval=schedule_interval,
    start_date=datetime(2023, 9, 27),
    catchup=False,
    is_paused_upon_creation=True,
    default_args=default_args,
)
def ask_astro_load_html():
    """
    This DAG performs incremental load for html source docs.  Initial load via ask_astro_load_bulk imported
    data from a point-in-time data capture.  By using the upsert logic of the weaviate provider
    any existing documents that have been updated will be removed and re-added.
    """

    extracted_html_docs = task(html.extract_html).expand(source=html_docs_sources)

    split_html_docs = task(split.split_html).expand(dfs=extracted_html_docs)

    _import_data = (
        task(ingest.import_data, retries=10)
        .partial(
            weaviate_conn_id=_WEAVIATE_CONN_ID,
            class_name=WEAVIATE_CLASS,
            existing="upsert",
            doc_key="docLink",
            batch_params={"batch_size": 1000},
            verbose=True,
        )
        .expand(dfs=[split_html_docs])
    )


ask_astro_load_html()
