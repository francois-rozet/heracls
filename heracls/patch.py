"""Monkey patching."""

import dacite

from collections.abc import Mapping
from dacite.core import Config, _build_value, extract_generic, is_subclass
from typing import Any


def _build_value_for_collection(
    collection: type,
    data: Any,  # noqa: ANN401
    config: Config,
) -> Any:  # noqa: ANN401
    data_type = type(data)
    if isinstance(data, dict) and is_subclass(collection, Mapping):
        types = extract_generic(collection, defaults=(Any, Any))
        return data_type(
            (key, _build_value(type_=types[1], data=value, config=config))
            for key, value in data.items()
        )
    elif isinstance(data, (list, tuple)) and is_subclass(collection, tuple):
        types = extract_generic(collection)
        if Ellipsis in types:
            return data_type(
                _build_value(type_=types[0], data=item, config=config) for item in data
            )
        elif len(data) == len(types):
            return data_type(
                _build_value(type_=type_, data=item, config=config)
                for item, type_ in zip(data, types, strict=True)
            )
    elif isinstance(data, (list, tuple)):
        types = extract_generic(collection, defaults=(Any,))
        return data_type(_build_value(type_=types[0], data=item, config=config) for item in data)
    return data


dacite.core._build_value_for_collection = _build_value_for_collection
