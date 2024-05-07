import re
import json
import uuid
from typing import Any, Self, assert_never

from ai_den.utils.json_schema import create_schema


Schema = dict[str, Any]

ARRAY = 'array'
BOOLEAN = 'boolean'
INTEGER = 'integer'
KV_PAIR = 'kv_pair'
NULL = 'null'
NUMBER = 'number'
OBJECT = 'object'
STRING = 'string'
VALUE = 'value'

WS = 'ws'

PRIMITIVE_TYPES = {BOOLEAN, INTEGER, NULL, NUMBER, STRING}


def make_string_literal(s: str) -> str:
    return '"\\"' + re.sub(r'("|\\)', r'\\\1', s) + '\\""'


def strip_schema(schema: Schema) -> Schema:
    new_schema = {}
    for key, value in schema.items():
        match key:
            case 'type' | 'default':
                new_schema[key] = value
            case 'anyOf':
                new_schema[key] = [strip_schema(clause) for clause in value]
            case 'items':
                new_schema[key] = strip_schema(value)
            case 'properties':
                new_schema[key] = {name: strip_schema(prop) for name, prop in value.items()}
            case _:
                pass
    return new_schema


def expand_optionals(chunks: list[str]) -> list[str]:
    if not chunks:
        return []

    if len(chunks) == 1:
        return [f'({chunks[0]})?']

    clause = f'({chunks[0]})'
    for c in chunks[1:]:
        clause += f' ("," ({c}))?'
    clauses = [clause]

    clauses += expand_optionals(chunks[1:])

    return clauses


class JsonSchemaGrammar:
    def __init__(self, schema: Schema):
        self.productions = {
            WS: r'[ \t\n]*',
            KV_PAIR: rf'{STRING} {WS} ":" {WS} {VALUE}',
            OBJECT: f'"{{" {WS} ({KV_PAIR} {WS} ("," {WS} {KV_PAIR} {WS})*)? "}}"',
            ARRAY: f'"[" {WS} ( {VALUE} {WS} ("," {WS} {VALUE} {WS})*)? "]"',
            STRING: r'"\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""',
            NUMBER: f'{INTEGER} ("." [0-9]+)? ([eE] [-+]? [0-9]+)?',
            INTEGER: '"-"? ([0-9] | [1-9] [0-9]*)',
            BOOLEAN: '"true" | "false"',
            NULL: '"null"',
            VALUE: f'{NULL} | {BOOLEAN} | {NUMBER} | {STRING} | {ARRAY} | {OBJECT}',
        }

        self.production_names = {}

        self.root = self.add_production(schema)

    def __str__(self) -> str:
        return self.make_gbnf()

    @classmethod
    def from_dataclass(cls, data_class: type) -> Self:
        return cls(create_schema(data_class))

    def make_gbnf(self) -> str:
        productions = [f'root ::= " "? {self.root}'] + [
            f'{k} ::= {v}'
            for k, v in reversed(self.productions.items())
        ]
        return '\n\n'.join(productions)

    def get_production_name(self, schema: Schema) -> str:
        # strip schema of superfluous fields
        schema = strip_schema(schema)

        # if schema is empty, then any value is valid
        if not schema:
            return VALUE

        # get schema string representations
        schema_repr = json.dumps(schema, sort_keys=True)

        # if schema hasn't been seen before, make a new production name
        if schema_repr not in self.production_names:
            self.production_names[schema_repr] = f'p_{uuid.uuid4().time_low:08x}'

        # return production name corresponding to the schema
        return self.production_names[schema_repr]

    def add_production(self, schema: Schema) -> str:
        # get a production name for the schema
        name = self.get_production_name(schema)

        # check if there already is a production with that name
        if name in self.productions:
            return name

        # create production for union of schemas
        if 'anyOf' in schema:
            self.productions[name] = ' | '.join(self.add_production(clause) for clause in schema['anyOf'])
            return name

        # create production based on schema type
        match schema.get('type'):
            case 'array':
                items_production_name = self.add_production(schema['items'])
                self.productions[name] = f'"[" {WS} ({items_production_name} {WS} ("," {WS} {items_production_name} {WS})*)? "]"'
                return name

            case 'object':
                required, optional = [], []
                properties: dict[str, Schema] = schema['properties']
                for prop_name, prop_schema in properties.items():
                    prop_name = make_string_literal(prop_name)
                    if 'default' in prop_schema:
                        prop_schema.pop('default')
                        optional.append((prop_name, self.add_production(prop_schema)))
                    else:
                        required.append((prop_name, self.add_production(prop_schema)))

                required_components = [
                    f'{WS} {prop_name} {WS} ":" {WS} {prod_name} {WS}'
                    for prop_name, prod_name in required
                ]

                optional_components = [
                    f'{WS} {prop_name} {WS} ":" {WS} {prod_name} {WS}'
                    for prop_name, prod_name in optional
                ]

                prod_value = '"{"'

                if required_components:
                    prod_value += '\n\t'
                    prod_value += ' ","\n\t'.join(required_components)
                    if optional_components:
                        prod_value += '\n\t'
                        prod_value += '\n\t'.join(
                            f'({WS} "," {component})?'
                            for component in optional_components
                        )
                    prod_value += '\n'
                elif optional_components:
                    prod_value += '\n\t'
                    prod_value += '\n\t| '.join(expand_optionals(optional_components))
                    prod_value += '\n'
                else:
                    prod_value += ' '

                prod_value += '"}"'

                self.productions[name] = prod_value
                return name

            case 'string' if 'enum' in schema:
                self.productions[name] = ' | '.join(make_string_literal(val) for val in schema['enum'])
                return name

            case t if t in PRIMITIVE_TYPES:
                return t

            case t:
                assert_never(t)
