import json
import dataclasses
from typing import Any, Optional, TypeVar, overload
from collections.abc import Sequence, Mapping

import dacite


T = TypeVar('T')


def is_dataclass_type(obj: Any) -> bool:
    return dataclasses.is_dataclass(obj) and isinstance(obj, type)


def is_dataclass_instance(obj: Any) -> bool:
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


@overload
def from_data(data_class: type[T], data: Mapping[str, Any]) -> Optional[T]:
    ...


@overload
def from_data(data_class: type[T], data: Sequence[Mapping[str, Any]]) -> list[T]:
    ...


def from_data(data_class, data):
    if isinstance(data, Sequence):
        results = []
        for d in data:
            if obj := from_data(data_class, d):
                results.append(obj)
        return results

    if data:
        return dacite.from_dict(data_class, data)


def from_json(data_class: type[T], s: str) -> T | list[T]:
    return from_data(data_class, json.loads(s))


def to_json(
        obj: Any,
        *,
        ensure_ascii: bool = False,
        indent: Optional[int] = None,
        separators: Optional[tuple[str, str]] = None,
        sort_keys: bool = False,
) -> str:
    return json.dumps(
        dataclasses.asdict(obj),
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )
