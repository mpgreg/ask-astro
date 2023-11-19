from __future__ import annotations

from typing import Any

import pandas as pd
from weaviate.client import Client as WeaviateClient


def _query_objects(
    weaviate_client: WeaviateClient, value: Any, doc_key: str, class_name: str, uuid_column: str
) -> list:
    """
    Check for existence of a data_object as a property of a data class and return all object ids.
    """
    existing_uuids = (
        weaviate_client.query.get(properties=[doc_key], class_name=class_name)
        .with_additional([uuid_column])
        .with_where({"path": doc_key, "operator": "Equal", "valueText": value})
        .do()["data"]["Get"][class_name]
    )

    return {additional["_additional"]["id"] for additional in existing_uuids}


def _objects_to_upsert(
    weaviate_client: WeaviateClient, df: pd.DataFrame, class_name: str, doc_key: str, uuid_column: str
) -> dict:
    ids_df = df.groupby(doc_key)[uuid_column].apply(set).reset_index(name="new_ids")
    ids_df["existing_ids"] = ids_df[doc_key].apply(
        lambda x: _query_objects(
            weaviate_client=weaviate_client, value=x, doc_key=doc_key, uuid_column=uuid_column, class_name=class_name
        )
    )

    ids_df["objects_to_insert"] = ids_df.apply(lambda x: list(x.new_ids.difference(x.existing_ids)), axis=1)
    ids_df["objects_to_delete"] = ids_df.apply(lambda x: list(x.existing_ids.difference(x.new_ids)), axis=1)
    ids_df["unchanged_objects"] = ids_df.apply(lambda x: x.new_ids.intersection(x.existing_ids), axis=1)

    objects_to_insert = [item for sublist in ids_df.objects_to_insert.tolist() for item in sublist]
    objects_to_delete = [item for sublist in ids_df.objects_to_delete.tolist() for item in sublist]
    unchanged_objects = [item for sublist in ids_df.unchanged_objects.tolist() for item in sublist]

    return {
        "objects_to_insert": objects_to_insert,
        "objects_to_delete": objects_to_delete,
        "unchanged_objects": unchanged_objects,
    }
