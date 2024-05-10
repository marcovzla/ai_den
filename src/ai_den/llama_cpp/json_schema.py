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

SPACE = 'space'

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
            case 'type' | 'default' | 'enum':
                new_schema[key] = value
            case 'anyOf' | 'prefixItems':
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
            SPACE: '" "?',
            KV_PAIR: rf'{STRING} {SPACE} ":" {SPACE} {VALUE}',
            OBJECT: f'"{{" {SPACE} ({KV_PAIR} {SPACE} ("," {SPACE} {KV_PAIR} {SPACE})*)? "}}"',
            ARRAY: f'"[" {SPACE} ( {VALUE} {SPACE} ("," {SPACE} {VALUE} {SPACE})*)? "]"',
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
        productions = [f'root ::= {SPACE} {self.root}'] + [
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
            if 'enum' in schema:
                prefix = 'enum'
            elif 'anyOf' in schema:
                prefix = 'union'
            elif schema.get('type') == 'array':
                prefix = 'array'
            elif schema.get('type') == 'object':
                prefix = 'object'
            else:
                prefix = 'prod'
            n = 1 + sum(1 for name in self.production_names.values() if name.startswith(f'{prefix}_'))
            self.production_names[schema_repr] = f'{prefix}_{n}'

        # return production name corresponding to the schema
        return self.production_names[schema_repr]

    def add_production(self, schema: Schema) -> str:
        # get a production name for the schema
        name = self.get_production_name(schema)

        # check if there already is a production with that name
        if name in self.productions:
            return name

        # create production for union of schemas
        if anyOf := schema.get('anyOf'):
            self.productions[name] = ' | '.join(self.add_production(clause) for clause in anyOf)
            return name

        # handle enums
        if enum := schema.get('enum'):
            self.productions[name] = ' | '.join(make_string_literal(val) for val in enum)
            return name

        # create production based on schema type
        match schema.get('type'):
            case 'array':
                rule = '"["'

                if prefixItems := schema.get('prefixItems'):
                    rule += '","'.join(
                        f' {SPACE} {self.add_production(item)} {SPACE} '
                        for item in prefixItems
                    )

                if items := schema.get('items'):
                    prod_name = self.add_production(items)
                    rule += f' {SPACE} ({prod_name} {SPACE} ("," {SPACE} {prod_name} {SPACE})*)? '

                rule += '"]"'

                self.productions[name] = rule

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
                    f'{SPACE} {prop_name} {SPACE} ":" {SPACE} {prod_name} {SPACE}'
                    for prop_name, prod_name in required
                ]

                optional_components = [
                    f'{SPACE} {prop_name} {SPACE} ":" {SPACE} {prod_name} {SPACE}'
                    for prop_name, prod_name in optional
                ]

                rule = '"{"'

                if required_components:
                    rule += '\n\t'
                    rule += ' ","\n\t'.join(required_components)
                    if optional_components:
                        rule += '\n\t'
                        rule += '\n\t'.join(
                            f'({SPACE} "," {component})?'
                            for component in optional_components
                        )
                    rule += '\n'
                elif optional_components:
                    rule += '\n\t'
                    rule += '\n\t| '.join(expand_optionals(optional_components))
                    rule += '\n'
                else:
                    rule += ' '

                rule += '"}"'

                self.productions[name] = rule
                return name

            case t:
                assert_never(t)
