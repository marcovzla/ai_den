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
def from_data(
        data_class: type[T],
        data: Mapping[str, Any],
        *,
        check_types: bool = False,
) -> Optional[T]:
    ...


@overload
def from_data(
        data_class: type[T],
        data: Sequence[Mapping[str, Any]],
        *,
        check_types: bool = False,
) -> list[T]:
    ...


def from_data(data_class, data, *, check_types: bool = False):
    if isinstance(data, Sequence):
        return [
            obj
            for datum in data
            if (obj := from_data(data_class, datum))
        ]

    if data:
        return dacite.from_dict(data_class, data, dacite.Config(check_types=check_types))


def from_json(
        data_class: type[T],
        s: str,
        *,
        check_types: bool = False,
) -> T | list[T]:
    return from_data(data_class, json.loads(s), check_types=check_types)


def to_json(
        obj: Any,
        *,
        ensure_ascii: bool = False,
        allow_nan: bool = True,
        indent: Optional[int | str] = None,
        separators: Optional[tuple[str, str]] = None,
        sort_keys: bool = False,
) -> str:
    if isinstance(obj, Sequence):
        obj = [dataclasses.asdict(x) for x in obj]
    else:
        obj = dataclasses.asdict(obj)
    return json.dumps(
        obj,
        ensure_ascii=ensure_ascii,
        allow_nan=allow_nan,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )
