import datetime
import json
import os
from pathlib import Path
import urllib

import pandas as pd
from include.tasks import ingest, split
from include.tasks.extract import html, github, registry, stack_overflow, slack

from airflow.decorators import dag, task
# from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
from include.utils.weaviate.hooks.weaviate import _WeaviateHook

seed_baseline_url = None #'https://astronomer-demos-public-readonly.s3.us-west-2.amazonaws.com/ask-astro/baseline_data_v5.parquet'

ask_astro_env = os.environ.get("ASK_ASTRO_ENV", "local")

_WEAVIATE_CONN_ID = os.environ.get("WEAVIATE_CONN_ID", f"weaviate_{ask_astro_env}")
_GITHUB_CONN_ID = "github_ro"
WEAVIATE_CLASS = os.environ.get("WEAVIATE_CLASS", "DocsLocal")

weaviate_hook = _WeaviateHook(_WEAVIATE_CONN_ID)
weaviate_client = weaviate_hook.get_client()

markdown_docs_sources = [
    {"doc_dir": "", "repo_base": "OpenLineage/docs"},
    {"doc_dir": "", "repo_base": "OpenLineage/OpenLineage"},
]
code_samples_sources = [
    {"doc_dir": "code-samples", "repo_base": "astronomer/docs"},
]
issues_docs_sources = [
    {"repo_base": "apache/airflow", "cutoff_date": datetime.date(2020, 1, 1), "cutoff_issue_number": 30000}
]
slack_channel_sources = [
    {
        "channel_name": "troubleshooting",
        "channel_id": "CCQ7EGB1P",
        "team_id": "TCQ18L22Z",
        "team_name": "Airflow Slack Community",
        "slack_api_conn_id": "slack_api_ro",
    }
]
html_docs_sources = [
    {
        "base_url": "https://astronomer-providers.readthedocs.io/en/stable/", 
        "exclude_docs": [r"/changelog.html", r"/_sources/", r"/_modules/", r".txt$"], 
        "container_class": "body"
    },
    {
        "base_url": "https://astro-sdk-python.readthedocs.io/en/stable/", 
        "exclude_docs": [r"/changelog.html", r"_api", r"_modules"], 
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
stackoverflow_tags = [
    {
        "name": "airflow", 
        "cutoff_date": 1630454400,  # "2021-09-01"
        "archive_posts": [
            "https://astronomer-demos-public-readonly.s3.us-west-2.amazonaws.com/ask-astro/extract/stackoverflow_archive/posts.parquet",
        ],
        "archive_comments": [
            "https://astronomer-demos-public-readonly.s3.us-west-2.amazonaws.com/ask-astro/extract/stackoverflow_archive/comments_0.parquet",
            "https://astronomer-demos-public-readonly.s3.us-west-2.amazonaws.com/ask-astro/extract/stackoverflow_archive/comments_1.parquet"
            ]
    }
]

default_args = {"retries": 3, "retry_delay": 30}


@dag(
    schedule_interval=None,
    start_date=datetime.datetime(2023, 9, 27),
    catchup=False,
    is_paused_upon_creation=True,
    default_args=default_args,
)
def ask_astro_load_bulk():
    """
    This DAG performs the initial load of data from sources.

    If seed_baseline_url (set above) points to a parquet file with pre-embedded data it will be
    ingested.  Otherwise new data is extracted, split, embedded and ingested.

    The first time this DAG runs (without seeded baseline) it will take at lease 90 minutes to
    extract data from all sources. Extracted data is then serialized to disk in the project
    directory in order to simplify later iterations of ingest with different chunking strategies,
    vector databases or embedding models.

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
            ["check_seed_baseline"]
            if weaviate_hook.check_schema(class_objects=class_objects)
            else ["create_schema"]
        )

    @task(trigger_rule="none_failed")
    def create_schema(class_objects: dict, existing: str = "ignore"):
        weaviate_hook.create_schema(class_objects=class_objects, existing=existing)

    @task.branch(trigger_rule="none_failed")
    def check_seed_baseline(seed_baseline_url: str = None) -> str:
        """
        Check if we will ingest from pre-embedded baseline or extract each source.
        """

        if seed_baseline_url is not None:
            return "import_baseline"
        else:
            return {
                "extract_github_markdown",
                "extract_html",
                "extract_stack_overflow_archive",
                "extract_slack_archive",
                "extract_astro_registry_cell_types",
                "extract_github_issues",
                "extract_github_python",
                "extract_astro_registry_dags",
            }

    @task(trigger_rule="none_failed")
    def extract_github_markdown(source: dict):

        parquet_file = Path(f"include/data/github/{source['repo_base']}/{source['doc_dir']}/docs.parquet")
        parquet_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_parquet(parquet_file)
        except Exception:
            df = github.extract_github_markdown(source, github_conn_id=_GITHUB_CONN_ID)
            df.to_parquet(parquet_file)

        return df

    @task(trigger_rule="none_failed")
    def extract_github_python(source: dict):

        parquet_file = Path(f"include/data/github/{source['repo_base']}/{source['doc_dir']}/code.parquet")
        parquet_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_parquet(parquet_file)
        except Exception:
            df = github.extract_github_python(source, _GITHUB_CONN_ID)
            df.to_parquet(parquet_file)

        return df

    @task(trigger_rule="none_failed")
    def extract_html(source: dict):
        
        url_parts = urllib.parse.urlparse(source['base_url'])
        
        parquet_file = Path(f"include/data/html/{url_parts.netloc}"+url_parts.path+"docs.parquet")
        parquet_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_parquet(parquet_file)
        except Exception:
            df = html.extract_html(source=source)[0]
            df.to_parquet(parquet_file)

        return df

    @task(trigger_rule="none_failed")
    def extract_stack_overflow_archive(tag: dict):

        parquet_file = Path(f"include/data/stack_overflow/archive/{tag['name']}.parquet")
        parquet_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_parquet(parquet_file)
        except Exception:
            df = stack_overflow.extract_stack_overflow_archive(tag=tag)
            df.to_parquet(parquet_file)

        return df

    @task(trigger_rule="none_failed")
    def extract_slack_archive(source: dict):

        parquet_file = Path(f"include/data/slack/archive/{source['channel_name']}.parquet")
        parquet_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_parquet(parquet_file)
        except Exception:
            df = slack.extract_slack_archive(source)
            df.to_parquet(parquet_file)
    
        return df

    @task(trigger_rule="none_failed")
    def extract_github_issues(source: dict):

        parquet_file = Path(f"include/data/github/{source['repo_base']}/issues.parquet")
        parquet_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_parquet(parquet_file)
        except Exception:
            df = github.extract_github_issues(source, _GITHUB_CONN_ID)
            df.to_parquet(parquet_file)

        return df

    @task(trigger_rule="none_failed")
    def extract_astro_registry_cell_types():

        parquet_file = Path(f"include/data/html/api.astronomer.io/registry/modules.parquet")
        parquet_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_parquet(parquet_file)
        except Exception:
            df = registry.extract_astro_registry_cell_types()[0]
            df.to_parquet(parquet_file)

        return [df]

    @task(trigger_rule="none_failed")
    def extract_astro_registry_dags():

        parquet_file = Path(f"include/data/html/api.astronomer.io/registry/dags.parquet")
        parquet_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_parquet(parquet_file)
        except Exception:
            df = registry.extract_astro_registry_dags()[0]
            df.to_parquet(parquet_file)

        return [df]

    md_docs = extract_github_markdown.expand(source=markdown_docs_sources)
    issues_docs = extract_github_issues.expand(source=issues_docs_sources)
    stackoverflow_docs = extract_stack_overflow_archive.expand(tag=stackoverflow_tags)
    slack_docs = extract_slack_archive.expand(source=slack_channel_sources)
    registry_cells_docs = extract_astro_registry_cell_types()
    html_docs = extract_html.expand(source=html_docs_sources)
    registry_dags_docs = extract_astro_registry_dags()
    code_samples = extract_github_python.expand(source=code_samples_sources)

    _get_schema = get_schema()
    _check_schema = check_schema(class_objects=_get_schema)
    _create_schema = create_schema(class_objects=_get_schema, existing="ignore")
    _check_seed_baseline = check_seed_baseline(seed_baseline_url=seed_baseline_url)

    markdown_tasks = [
        md_docs,
        issues_docs,
        stackoverflow_docs,
        slack_docs,
        registry_cells_docs
    ]

    html_tasks = [html_docs]

    python_code_tasks = [registry_dags_docs, code_samples]

    split_md_docs = task(split.split_markdown).expand(dfs=markdown_tasks)

    split_code_docs = task(split.split_python).expand(dfs=python_code_tasks)

    split_html_docs = task(split.split_html).expand(dfs=html_tasks)

    _import_data = (
        task(ingest.import_data, retries=10)
        .partial(
            weaviate_conn_id=_WEAVIATE_CONN_ID,
            class_name=WEAVIATE_CLASS,
            existing="skip",
            batch_params={"batch_size": 1000},
            verbose=True,
        )
        .expand(dfs=[split_md_docs, split_code_docs, split_html_docs])
    )

    _import_baseline = (
        task(ingest.import_baseline, trigger_rule="none_failed")(
            weaviate_conn_id=_WEAVIATE_CONN_ID,
            seed_baseline_url=seed_baseline_url,
            class_name=WEAVIATE_CLASS,
            existing="skip",
            uuid_column="id",
            vector_column="vector",
            batch_params={"batch_size": 1000},
            verbose=True,
        )
    )

    _check_schema >> [_check_seed_baseline, _create_schema]

    _create_schema >> markdown_tasks + python_code_tasks + html_tasks + [_check_seed_baseline]

    _check_seed_baseline >> markdown_tasks + python_code_tasks + html_tasks + [_import_baseline]


ask_astro_load_bulk()
