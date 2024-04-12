import re
import json
import inspect
import dataclasses
from types import GenericAlias
from typing import Any, Optional, get_args, get_origin
from collections.abc import Callable, Sequence

import jsonref
import docstring_parser
from pydantic import BaseModel, TypeAdapter, create_model
from pydantic.fields import Field

from ai_den.utils.dataclasses import is_dataclass_type


def json_schema(
        obj: Any,
        *,
        name: Optional[str] = None,
        replace_refs: bool = True,
        keep_titles: bool = False,
        flatten: bool = True,
        ensure_ascii: bool = False,
        allow_nan: bool = True,
        indent: Optional[int | str] = None,
        separators: Optional[tuple[str, str]] = None,
        sort_keys: bool = False,
) -> str:
    schema = create_schema(
        obj,
        name=name,
        replace_refs=replace_refs,
        keep_titles=keep_titles,
        flatten=flatten,
    )
    return json.dumps(
        schema,
        ensure_ascii=ensure_ascii,
        allow_nan=allow_nan,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def create_schema(
        obj: Any,
        *,
        name: Optional[str] = None,
        replace_refs: bool = True,
        keep_titles: bool = False,
        flatten: bool = True,
) -> dict[str, Any]:
    if isinstance(obj, type) or isinstance(obj, GenericAlias):
        return create_schema_for_type(
            obj,
            replace_refs=replace_refs,
            keep_titles=keep_titles,
            flatten=flatten,
        )

    if callable(obj):
        return create_schema_for_callable(
            obj,
            name=name,
            replace_refs=replace_refs,
            keep_titles=keep_titles,
            flatten=flatten,
        )

    if isinstance(obj, Sequence):
        return [
            create_schema(
                elem,
                name=name,
                replace_refs=replace_refs,
                keep_titles=keep_titles,
                flatten=flatten,
            )
            for elem in obj
        ]

    raise ValueError(f'Unsupported: {obj!r}')


def create_schema_for_type(
        t: type[Any],
        replace_refs: bool = True,
        keep_titles: bool = False,
        flatten: bool = True,
) -> dict[str, Any]:
    # prepare type
    t = prepare_type_for_pydantic_compatibility(t)

    # maybe wrap with a TypeAdapter
    if not isinstance(t, type) or not issubclass(t, BaseModel):
        t = TypeAdapter(t)

    # return schema
    return process_schema(
        t,
        replace_refs=replace_refs,
        keep_titles=keep_titles,
        flatten=flatten,
    )


def create_schema_for_callable(
        f: Callable,
        name: Optional[str] = None,
        replace_refs: bool = True,
        keep_titles: bool = False,
        flatten: bool = True,
) -> dict[str, Any]:
    model = new_pydantic_model_from_callable(f, name)

    parameters = process_schema(
        model,
        replace_refs=replace_refs,
        keep_titles=keep_titles,
        flatten=flatten,
    )

    # assemble schema
    schema = {'type': 'function', 'function': {}}
    schema['function']['name'] = model.__name__
    if 'description' in parameters:
        schema['function']['description'] = parameters.pop('description')
    schema['function']['parameters'] = parameters

    return schema


def process_schema(
        schema: dict[str, Any] | TypeAdapter | type[BaseModel],
        *,
        replace_refs: bool = True,
        keep_titles: bool = False,
        flatten: bool = True,
) -> dict[str, Any]:
    # ensure schema dictionary
    if isinstance(schema, TypeAdapter):
        schema = schema.json_schema()
    elif isinstance(schema, type) and issubclass(schema, BaseModel):
        schema = schema.model_json_schema()
    elif not isinstance(schema, dict):
        raise TypeError('Invalid schema')

    # resolve JSON references
    if replace_refs:
        schema = jsonref.replace_refs(schema, proxies=False)
        delete_entry(schema, '$defs')

    # remove title entries
    if not keep_titles:
        delete_entry(schema, 'title')

    # flatten allOf when possible
    if flatten:
        flatten_entry(schema, 'allOf')

    return schema


def delete_entry(obj: Any, name: str) -> None:
    if isinstance(obj, dict):
        if name in obj:
            del obj[name]
        for k, v in obj.items():
            # don't delete entries from properties dictionary
            if k == 'properties':
                for x in v.values():
                    delete_entry(x, name)
            else:
                delete_entry(v, name)
    elif isinstance(obj, list):
        for x in obj:
            delete_entry(x, name)


def flatten_entry(obj: Any, name: str):
    if isinstance(obj, dict):
        clauses = obj.get(name, [])
        if len(clauses) == 1:
            obj.pop(name)
            obj.update(clauses[0])
        for x in obj.values():
            flatten_entry(x, name)
    elif isinstance(obj, list):
        for x in obj:
            flatten_entry(x, name)


def new_pydantic_model_from_dataclass(
        t: type[Any],
        name: Optional[str] = None,
) -> type[BaseModel]:
    if not is_dataclass_type(t):
        raise ValueError('The provided class is not a dataclass')

    if name is None:
        name = t.__name__

    description, param_descriptions = parse_docstring(t)

    fields = {
        f.name: {
            'annotation': prepare_type_for_pydantic_compatibility(f.type),
            'default': determine_default_dataclass_field_value(f.default, f.default_factory),
            'description': param_descriptions.get(f.name),
        }
        for f in dataclasses.fields(t)
    }

    return new_pydantic_model_from_fields(
        fields=fields,
        model_name=name,
        model_description=description,
    )


def new_pydantic_model_from_callable(
        f: Callable,
        name: Optional[str] = None,
) -> type[BaseModel]:
    if not callable(f):
        raise TypeError('The provided object is not a Callable')

    if name is None:
        name = f.__name__

    description, param_descriptions = parse_docstring(f)

    fields = {
        p.name: {
            'annotation': prepare_type_for_pydantic_compatibility(p.annotation),
            'default': p.default if p.default != inspect.Parameter.empty else Ellipsis,
            'description': param_descriptions.get(p.name),
        }
        for p in inspect.signature(f).parameters.values()
    }

    return new_pydantic_model_from_fields(
        fields=fields,
        model_name=name,
        model_description=description,
    )


def new_pydantic_model_from_fields(
        fields: list[str] | dict[str, Optional[dict[str, Any]]],
        model_name: str,
        model_description: Optional[str] = None,
        default_annotation: type[Any] = str,
        default_value: Any = Ellipsis,
) -> type[BaseModel]:
    field_definitions = {}

    for name, meta in (fields.items() if isinstance(fields, dict) else zip(fields, [None] * len(fields))):
        field_args = meta if isinstance(meta, dict) else {}
        annotation = field_args.pop('annotation', default_annotation)
        default = field_args.pop('default', default_value)
        field_definitions[name] = (annotation, Field(default, **field_args))

    return create_model(
        model_name,
        __doc__=model_description,
        **field_definitions,
    )


def prepare_type_for_pydantic_compatibility(t: type[Any]) -> type[Any]:
    if is_dataclass_type(t):
        return new_pydantic_model_from_dataclass(t)

    if isinstance(t, GenericAlias):
        args = tuple(prepare_type_for_pydantic_compatibility(arg) for arg in get_args(t))
        t = get_origin(t)
        return t[args]

    return t


def determine_default_dataclass_field_value(
        default: Any,
        default_factory: Callable,
) -> Any:
    if default != dataclasses.MISSING:
        return default

    if default_factory != dataclasses.MISSING:
        return default_factory()

    return Ellipsis


def parse_docstring(
        obj: Any,
        long_description: bool = False,
) -> tuple[Optional[str], dict[str, str]]:
    doc = docstring_parser.parse(inspect.getdoc(obj))

    if long_description and doc.long_description:
        description = re.sub(r'\s+', ' ', doc.long_description)
    elif doc.short_description:
        description = re.sub(r'\s+', ' ', doc.short_description)
    else:
        description = None

    param_descriptions = {
        p.arg_name: re.sub(r'\s+', ' ', p.description)
        for p in doc.params
        if p.description
    }

    return description, param_descriptions
