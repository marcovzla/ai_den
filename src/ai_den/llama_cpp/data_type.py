import re
import json
from dataclasses import is_dataclass
from typing import Any, Generic, Optional, TypeVar, assert_never
from pydantic import TypeAdapter
from llama_cpp import LlamaGrammar


T = TypeVar('T')
Schema = dict[str, Any]


ARRAY = 'array'
BOOLEAN = 'boolean'
INTEGER = 'integer'
KV_PAIR = 'kv-pair'
NULL = 'null'
NUMBER = 'number'
OBJECT = 'object'
STRING = 'string'
VALUE = 'value'
SPACE = 'space'


PRODUCTIONS = {
    KV_PAIR: rf'{STRING} {SPACE} ":" {SPACE} {VALUE}',
    OBJECT: f'"{{" {SPACE} ({KV_PAIR} {SPACE} ("," {SPACE} {KV_PAIR} {SPACE})*)? "}}"',
    ARRAY: f'"[" {SPACE} ( {VALUE} {SPACE} ("," {SPACE} {VALUE} {SPACE})*)? "]"',
    STRING: r'"\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""',
    NUMBER: f'"-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?',
    INTEGER: '"-"? ([0-9] | [1-9] [0-9]*)',
    BOOLEAN: '"true" | "false"',
    NULL: '"null"',
    VALUE: f'{NULL} | {BOOLEAN} | {INTEGER} | {NUMBER} | {STRING} | {ARRAY} | {OBJECT}',
}


PRIMITIVE_TYPES = {NULL, BOOLEAN, INTEGER, NUMBER, STRING}


class DataType(Generic[T]):
    def __init__(self, data_type: type[T]):
        self.data_type = data_type
        self.type_adapter = TypeAdapter(data_type)
        self.init_grammar()

    def schema(self) -> Schema:
        return self.type_adapter.json_schema()

    def json_schema(
            self,
            *,
            ensure_ascii: bool = False,
            indent: Optional[int | str] = None,
            separators: Optional[tuple[str, str]] = None,
            sort_keys: bool = False,
    ) -> str:
        return json.dumps(
            self.schema(),
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    def parse_json(self, data: str, *, strict: bool = False) -> T:
        return self.type_adapter.validate_json(data, strict=strict)

    def llama_grammar(self, *, verbose: bool = False) -> LlamaGrammar:
        return LlamaGrammar.from_string(self.gbnf(), verbose=verbose)

    def gbnf(self) -> str:
        return '\n'.join(
            f'{name} ::= {rule}'
            for name, rule in reversed(self.productions.items())
        )

    def init_grammar(self):
        self.references: dict[str, str] = {}
        self.productions: dict[str, str] = {SPACE: '" "?'}
        self.production_names: dict[str, str] = {}
        self.prefix_counter: dict[str, int] = {}
        name = self.data_type.__name__ if is_dataclass(self.data_type) else None
        production_name = self.add_schema_to_grammar(self.schema(), name=name)
        self.references['#'] = production_name
        self.productions['root'] = f'{SPACE} {production_name}'

    def add_schema_to_grammar(
            self,
            schema: Schema,
            *,
            name: Optional[str] = None,
    ) -> str:
        name = self.get_production_name(schema=schema, name=name)

        # check if there already is a production with that name
        if name in self.productions:
            return name

        # an empty schema matches any json value
        if not schema:
            self.productions.update(PRODUCTIONS)
            return VALUE

        # follow reference
        if ref := schema.get('$ref'):
            return self.get_production_name(reference=ref)

        # add definitions to grammar
        if defs := schema.get('$defs'):
            self.add_defs_to_grammar(defs, root=ref)

        # create production for union of schemas
        if anyOf := schema.get('anyOf'):
            self.productions[name] = ' | '.join(self.add_schema_to_grammar(clause) for clause in anyOf)
            return name

        # handle enums
        if enum := schema.get('enum'):
            self.productions[name] = ' | '.join(make_string_literal(s) for s in enum)
            return name
        
        # create production based on schema type
        match schema.get('type'):
            case 'array':
                return self.add_array_to_grammar(name, schema)

            case 'object':
                return self.add_object_to_grammar(name, schema)

            case t if t in PRIMITIVE_TYPES:
                if t not in self.productions:
                    self.productions[t] = PRODUCTIONS[t]
                return t

            case t:
                assert_never(t)

    def add_defs_to_grammar(self, defs: dict[str, Schema], root: str):
        for name, schema in defs.items():
            self.add_schema_to_grammar(
                schema=schema,
                name=self.get_production_name(schema=schema, name=name, reference=f'#/$defs/{name}'),
            )

    def add_array_to_grammar(self, name: str, schema: Schema) -> str:
        # check if schema has restrictions
        if not schema.get('prefixItems') and not schema.get('items'):
            self.productions.update(PRODUCTIONS)
            return ARRAY

        # open array
        rule = '"["'

        # prefixItems is a list of schemas that occurs when handling tuples
        if prefixItems := schema.get('prefixItems'):
            rule += '","'.join(
                f' {SPACE} {self.add_schema_to_grammar(item)} {SPACE} '
                for item in prefixItems
            )

        # items is a single schema for the list items
        if items := schema.get('items'):
            if prefixItems:
                rule += f'","'
            prod_name = self.add_schema_to_grammar(items)
            rule += f' {SPACE} ({prod_name} {SPACE} ("," {SPACE} {prod_name} {SPACE})*)? '

        # close array
        rule += '"]"'

        # add production rule
        self.productions[name] = rule

        return name

    def add_key_value_pair_to_grammar(self, object_name: str, property_name: str, property_schema: Schema) -> str:
        # get production name for property value
        value = self.add_schema_to_grammar(property_schema)
        # get production name for key-value pair
        production_name = self.get_production_name(name=f'{object_name}-{property_name}')
        # make production rule for key-value pair
        self.productions[production_name] = f'{SPACE} {make_string_literal(property_name)} {SPACE} ":" {SPACE} {value} {SPACE}'
        # return production name for key-value pair
        return production_name

    def add_object_to_grammar(self, name: str, schema: Schema) -> str:
        required, optional = [], []

        if properties := schema.get('properties'):
            # make productions for key-value pairs and organize them into required and optional
            for property_name, property_schema in reversed(properties.items()):
                if 'default' in property_schema:
                    property_schema.pop('default')
                    production_name = self.add_key_value_pair_to_grammar(name, property_name, property_schema)
                    optional.insert(0, production_name)
                else:
                    production_name = self.add_key_value_pair_to_grammar(name, property_name, property_schema)
                    required.insert(0, production_name)

        if additionalProperties := schema.get('additionalProperties'):
            # ensure string primitive is in productions
            if STRING not in self.productions:
                self.productions[STRING] = PRODUCTIONS[STRING]
            production_name = self.add_schema_to_grammar(additionalProperties)
            kv_pair = f'{SPACE} {STRING} {SPACE} ":" {SPACE} {production_name} {SPACE}'
            optional.append(f'{kv_pair} ("," {kv_pair})*')

        # check if schema has restrictions
        if not required and not optional:
            self.productions.update(PRODUCTIONS)
            return OBJECT

        # construct object rule, handling indentation for grammar readability
        rule = '"{"'

        if required:
            # add all required subrules separated by commas
            rule += ' '
            rule += ' "," '.join(required)

            if optional:
                # if there are optional rules following the required ones, each one must be preceded by a comma
                rule += ' '
                rule += ' '.join(f'("," {component})?' for component in optional)

            rule += ' '

        elif optional:
            # if all subrules are optional, expand them recursively
            rule += ' ( '
            rule += ' | '.join(recursively_expand_optionals(optional))
            rule += ' ) '

        # close object
        rule += '"}"'

        # add production rule
        self.productions[name] = rule

        return name

    def get_production_name(
            self,
            *,
            schema: Optional[Schema] = None,
            name: Optional[str] = None,
            reference: Optional[str] = None,
    ) -> str:

        # if name is a primitive type, there is nothing to do
        if name in PRIMITIVE_TYPES:
            return name

        # get string representation of schema used to check if it has been seen already
        schema_repr = None if schema is None else json.dumps(strip_schema(schema), sort_keys=True)

        # if schema already has a production name, use it
        if production_name := self.production_names.get(schema_repr):
            return production_name

        # check if we have a schema that hasn't been seen before
        if schema is not None:
            # check if schema is empty
            if len(schema) == 0:
                self.production_names[schema_repr] = VALUE
                return VALUE

            # check if schema represents a primitive type
            if len(schema) == 1 and schema.get('type') in PRIMITIVE_TYPES:
                name = schema['type']
                self.production_names[schema_repr] = name
                return name
            
            # check if schema is a reference
            if len(schema) == 1 and (ref := schema.get('$ref')):
                if ref not in self.references:
                    raise ValueError(f'unknown reference: {ref}')
                name = self.references[ref]
                self.production_names[schema_repr] = name
                return name

        # if we got a reference, try to resolve it
        if reference:
            if production_name := self.references.get(reference):
                if schema_repr is not None:
                    self.production_names[schema_repr] = production_name
                return production_name
            prefix = reference.rsplit('/', maxsplit=1)[-1]

        # get production name prefix
        if name:
            prefix = name
        elif 'enum' in schema:
            prefix = 'enum'
        elif 'anyOf' in schema:
            prefix = 'union'
        elif schema.get('type') == 'array':
            prefix = 'array'
        elif schema.get('type') == 'object':
            prefix = 'object'
        else:
            prefix = 'production'

        # maybe append count to prefix to ensure uniqueness
        if prefix in {'enum', 'union', 'array', 'object', 'production'}:
            # append prefix count to production name
            n = 1 + self.prefix_counter.get(prefix, 0)
            self.prefix_counter[prefix] = n
            production_name = f'{prefix}-{n}'
        else:
            production_name = prefix

        # store schema's production name for next time
        if schema_repr is not None:
            self.production_names[schema_repr] = production_name

        # store reference's production name for next time
        if reference is not None:
            self.references[reference] = production_name

        return production_name


def make_string_literal(string: str) -> str:
    escapes = {'"': '\\"', '\\': '\\\\', '\n': '\\n', '\r': '\\r'}
    pattern = '|'.join(re.escape(c) for c in escapes.keys())
    repl = lambda m: escapes[m[0]]
    return '"\\"' + re.sub(pattern, repl, string) + '\\""'


def recursively_expand_optionals(chunks: list[str]) -> list[str]:
    if not chunks:
        return []

    if len(chunks) == 1:
        return [f'({chunks[0]})?']

    clause = f'({chunks[0]})'
    for c in chunks[1:]:
        clause += f' ("," ({c}))?'
    clauses = [clause]

    clauses += recursively_expand_optionals(chunks[1:])

    return clauses


def strip_schema(schema: Schema) -> Schema:
    new_schema = {}
    for key, value in schema.items():
        match key:
            case 'type' | 'default' | 'enum' | '$ref':
                new_schema[key] = value
            case 'anyOf' | 'prefixItems':
                new_schema[key] = [strip_schema(clause) for clause in value]
            case 'items' | 'additionalProperties':
                new_schema[key] = strip_schema(value)
            case 'properties' | '$defs':
                new_schema[key] = {name: strip_schema(prop) for name, prop in value.items()}
            case _:
                pass
    return new_schema
