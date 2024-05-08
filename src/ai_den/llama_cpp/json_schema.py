import re
import json
from typing import Any, Self, assert_never

from llama_cpp import LlamaGrammar

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

PRODUCTION_NAMES = {
    '{}': VALUE,
    f'{{"type": "{NULL}"}}': NULL,
    f'{{"type": "{BOOLEAN}"}}': BOOLEAN,
    f'{{"type": "{INTEGER}"}}': INTEGER,
    f'{{"type": "{NUMBER}"}}': NUMBER,
    f'{{"type": "{STRING}"}}': STRING,
    f'{{"type": "{ARRAY}"}}': ARRAY,
    f'{{"type": "{OBJECT}"}}': OBJECT,
}


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
            WS: '" "?',
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

        self.production_names = PRODUCTION_NAMES.copy()

        self.num_default_productions = len(self.production_names)

        self.root = self.add_production(schema)

    def __str__(self) -> str:
        return self.gbnf()

    @classmethod
    def from_data_class(cls, data_class: type) -> Self:
        return cls(create_schema(data_class))

    @classmethod
    def from_json_schema(cls, json_schema: str) -> Self:
        return cls(json.loads(json_schema))

    def grammar(self, verbose: bool = False) -> LlamaGrammar:
        return LlamaGrammar.from_string(self.gbnf(), verbose=verbose)

    def gbnf(self) -> str:
        productions = [f'root ::= {WS} {self.root}'] + [
            f'{name} ::= {rule}'
            for name, rule in reversed(self.productions.items())
        ]
        return '\n\n'.join(productions)

    def get_production_name(self, schema: Schema) -> str:
        # strip schema of superfluous fields
        schema = strip_schema(schema)

        # get schema string representations
        schema_repr = json.dumps(schema, sort_keys=True)

        # if schema hasn't been seen before, make a new production name
        if schema_repr not in self.production_names:
            n = len(self.production_names) - self.num_default_productions + 1
            self.production_names[schema_repr] = f'prod_{n}'

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

            case t:
                assert_never(t)
