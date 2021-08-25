from typing import Any, Dict, List


def flatten_nested_dict(nested_dict: Dict[Any, Dict[Any, Any]], key_separator="."):
    """Return a flattened version of a dict of dicts"""
    stack = list(nested_dict.items())
    flattened_dict = {}
    while stack:
        key, val = stack.pop()
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                stack.append((f"{key}{key_separator}{sub_key}", sub_val))
        else:
            flattened_dict[key] = val
    return flattened_dict


def list_of_dict_to_dict_of_list(ld: List[Dict[Any, Any]]):
    dl = {key: [item[key] for item in ld] for key in ld[0].keys()}
    return dl