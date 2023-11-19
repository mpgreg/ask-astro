from __future__ import annotations

from typing import Any


def check_schema_subset(class_object: Any, class_schema: Any):
    """
    Recursively check if requested schema/object is a subset of the current schema.

    param class_object: The class object to check against current schema
    type class_object: Any

    param class_schema: The current schema class object
    type class_schema: Any
    """

    if class_object == class_schema:
        return True

    if isinstance(class_object, dict) and isinstance(class_schema, dict):
        if class_object.items() <= class_schema.items():
            return True
        else:
            checks = [check_schema_subset(class_object[k], class_schema[k]) for k in class_object.keys()]
            return all(checks)

    elif isinstance(class_object, (list, tuple)) and type(class_object) == type(class_schema):
        checks = [check_schema_subset(item[0], item[1]) for item in zip(class_object, class_schema)]
        return all(checks)

    else:
        return False
