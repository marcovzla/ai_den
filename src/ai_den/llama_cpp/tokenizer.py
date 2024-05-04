import re
from typing import overload
from collections.abc import Iterable
import llama_cpp


ENCODING = 'utf-8'

SENTENCEPIECE_WHITESPACE = '\N{LOWER ONE EIGHTH BLOCK}'


def unescape_whitespace(s: str) -> str:
    return s.replace(SENTENCEPIECE_WHITESPACE, ' ')

def escape_whitespace(s: str) -> str:
    return s.replace(' ', SENTENCEPIECE_WHITESPACE)

def parse_byte_token(token: str) -> str:
    return chr(int(token[3:-1], base=16))

def is_byte_token(token: str) -> bool:
    return bool(re.match(r'<0x[0-9a-fA-F]{2}>', token))


class LlamaCppTokenizer:
    def __init__(self, llama: llama_cpp.Llama):
        self.llama = llama
        self.vocab_type = llama_cpp.llama_vocab_type(self.llama.model)
        self.vocab_size = llama_cpp.llama_n_vocab(self.llama.model)
        self.bos_token_id = llama_cpp.llama_token_bos(self.llama.model)
        self.eos_token_id = llama_cpp.llama_token_eos(self.llama.model)
        self.bos_token = self._id_to_token(self.bos_token_id)
        self.eos_token = self._id_to_token(self.eos_token_id)

        self.token_ids = {
            self._id_to_token(i): i
            for i in range(self.vocab_size)
        }

    @property
    def is_sentencepiece(self) -> bool:
        return self.vocab_type == llama_cpp.LLAMA_VOCAB_TYPE_SPM

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self.llama.tokenize(text.encode(ENCODING), add_bos=add_special_tokens, special=True)

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        return ''.join(self._token_to_piece(id, skip_special_tokens) for id in ids)

    @overload
    def convert_ids_to_tokens(self, id: int) -> str:
        ...

    @overload
    def convert_ids_to_tokens(self, ids: Iterable[int]) -> list[str]:
        ...

    def convert_ids_to_tokens(self, x):
        if isinstance(x, int):
            return self._id_to_token(x)
        else:
            return [self._id_to_token(id) for id in x]

    @overload
    def convert_tokens_to_ids(self, token: str) -> int:
        ...

    @overload
    def convert_tokens_to_ids(self, tokens: Iterable[str]) -> list[int]:
        ...

    def convert_tokens_to_ids(self, x):
        if isinstance(x, str):
            return self._token_to_id(x)
        else:
            return [self._token_to_id(t) for t in x]

    @overload
    def escape_tokens(self, token: str) -> str:
        ...

    @overload
    def escape_tokens(self, tokens: Iterable[str]) -> list[str]:
        ...

    def escape_tokens(self, x):
        if isinstance(x, str):
            return self._escape_token(x)
        else:
            return [self._escape_token(t) for t in x]

    @overload
    def unescape_tokens(self, token: str) -> str:
        ...

    @overload
    def unescape_tokens(self, tokens: Iterable[str]) -> list[str]:
        ...

    def unescape_tokens(self, x):
        if isinstance(x, str):
            return self._unescape_token(x)
        else:
            return [self._unescape_token(t) for t in x]

    def convert_tokens_to_string(self, tokens: Iterable[str], skip_special_tokens: bool = False) -> str:
        return ''.join(self._token_to_piece(t, skip_special_tokens) for t in tokens)

    def _id_to_token(self, id: int) -> str:
        return llama_cpp.llama_token_get_text(self.llama.model, id).decode(ENCODING)

    def _id_to_token_type(self, id: int) -> int:
        return llama_cpp.llama_token_get_type(self.llama.model, id)

    def _token_to_id(self, token: str) -> int:
        token = self._escape_token(token)
        return self.token_ids[token]

    def _escape_token(self, token: str) -> str:
        if self.is_sentencepiece:
            token = escape_whitespace(token)
        if token not in self.token_ids and len(token) == 1:
            byte_token = f'<0x{ord(token):02X}>'
            if byte_token in self.token_ids:
                return byte_token
        return token

    def _unescape_token(self, token: str) -> str:
        if self.is_sentencepiece:
            token = unescape_whitespace(token)
        if is_byte_token(token):
            token = parse_byte_token(token)
        return token

    def _token_to_piece(self, token: str | int, skip_special_tokens: bool) -> str:
        # get token and token id
        if isinstance(token, str):
            token_id = self._token_to_id(token)
        else:
            token_id = token
            token = self._id_to_token(token_id)
        # convert token to piece based on its type
        match self._id_to_token_type(token_id):
            case llama_cpp.LLAMA_TOKEN_TYPE_NORMAL:
                return unescape_whitespace(token) if self.is_sentencepiece else token
            case llama_cpp.LLAMA_TOKEN_TYPE_UNKNOWN:
                return '' if skip_special_tokens else token
            case llama_cpp.LLAMA_TOKEN_TYPE_CONTROL:
                return '' if skip_special_tokens else token
            case llama_cpp.LLAMA_TOKEN_TYPE_USER_DEFINED:
                return '' if skip_special_tokens else token
            case llama_cpp.LLAMA_TOKEN_TYPE_BYTE:
                return parse_byte_token(token)
            case _:
                return ''
