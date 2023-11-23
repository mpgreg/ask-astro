from __future__ import annotations

from include.utils.weaviate.hooks.weaviate import _WeaviateHook
import logging
import pandas as pd

logger = logging.getLogger("airflow.task")

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
):
    """
    This task concatenates multiple dataframes from upstream dynamic tasks and vectorizes with import to weaviate.

    Upsert logic relies on a 'doc_key' which is a uniue representation of the document.  Because documents can
    be represented as multiple chunks (each with a UUID which is unique in the DB) the doc_key is a way to represent
    all chunks associated with an ingested document.

    :param dfs: A list of dataframes from downstream dynamic tasks
    :param class_name: The name of the class to import data.  Class should be created with weaviate schema.
        type class_name: str
    :param existing: Whether to 'upsert', 'skip' or 'replace' any existing documents.  Default is 'skip'.
    :param doc_key: If using upsert you must specify a doc_key which uniquely identifies a document which may or
        may not include multiple (unique) chunks.
    :param vector_column: For pre-embedded data specify the name of the column containing the embedding vector
    :param uuid_column: For data with pre-genenerated UUID specify the name of the column containing the UUID
    """

    weaviate_hook = _WeaviateHook(weaviate_conn_id)

    df = pd.concat(dfs, ignore_index=True)

    df, uuid_column = weaviate_hook.generate_uuids(
        df=df, class_name=class_name, uuid_column=uuid_column, vector_column=vector_column
        )

    duplicates = df[df[uuid_column].duplicated()]
    if len(duplicates) > 0:
        logger.error(f"Duplicate rows found. {duplicates}")

    weaviate_hook.ingest_data(
        df=df, 
        class_name=class_name, 
        existing=existing,
        doc_key=doc_key,
        uuid_column=uuid_column,
        vector_column=vector_column,
        batch_params=batch_params,
        verbose=verbose
    )

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
):
    """
    This task ingests data from a baseline of pre-embedded data. This is useful for evaluation and baselining changes
    over time. 

    Any existing documents are replaced unless otherwise specified.  The assumption is that this is a first import of 
    data and older data should be removed.

    :param seed_baseline_url: The uri of a parquet file containing baseline data to ingest.
    :param class_name: The name of the class to import data.  Class should be created with weaviate schema.
        type class_name: str
    :param existing: Whether to 'upsert', 'skip' or 'replace' any existing documents.  Default is 'skip'.
    :param doc_key: If using upsert you must specify a doc_key which uniquely identifies a document which may or
        may not include multiple (unique) chunks.
    :param vector_column: For pre-embedded data specify the name of the column containing the embedding vector
    :param uuid_column: For data with pre-genenerated UUID specify the name of the column containing the UUID
    """
    
    weaviate_hook = _WeaviateHook(weaviate_conn_id)

    seed_filename = f"include/data/{seed_baseline_url.split('/')[-1]}"

    try:
        df = pd.read_parquet(seed_filename)

    except Exception:
        
        df = pd.read_parquet(seed_baseline_url)
        df.to_parquet(seed_filename)

    return import_data(
                weaviate_conn_id=weaviate_conn_id,
                dfs=[df],
                class_name=class_name,
                existing=existing,
                doc_key=doc_key,
                uuid_column=uuid_column,
                vector_column=vector_column,
                batch_params=batch_params,
                verbose=verbose
            )