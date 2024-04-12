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

        special_tokens_patterns = [
            re.escape(self._id_to_token(i))
            for i in range(self.vocab_size)
            if self._id_to_token_type(i) in (llama_cpp.LLAMA_TOKEN_TYPE_CONTROL, llama_cpp.LLAMA_TOKEN_TYPE_UNKNOWN)
        ]

        # note the capturing parenthesis, they are needed so that split() returns the separators,
        # which correspond to special tokens
        self.special_token_pattern = re.compile('(' + '|'.join(special_tokens_patterns) + ')')

    @property
    def is_sentencepiece(self) -> bool:
        return self.vocab_type == llama_cpp.LLAMA_VOCAB_TYPE_SPM

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        token_ids = [self.bos_token_id] if add_special_tokens else []
        # iterate over chunks of text and special tokens
        # even position: chunk of text
        # odd position: special token
        for i, chunk_or_special_token in enumerate(self.special_token_pattern.split(text)):
            if i % 2 == 0:
                # tokenize the chunk of text
                chunk_token_ids = self.llama.tokenize(chunk_or_special_token.encode(ENCODING), add_bos=False)
                if i != 0 and chunk_token_ids and self.is_sentencepiece:
                    # if sentencepiece prepended a space to the first token of the chunk
                    # and this is not the first chunk
                    # then we need to remove the space
                    first_token = self.convert_ids_to_tokens(chunk_token_ids[0])
                    if first_token.startswith(SENTENCEPIECE_WHITESPACE) and len(first_token) > 1:
                        chunk_token_ids[0] = self.convert_tokens_to_ids(first_token[1:])
                token_ids += chunk_token_ids
            else:
                # get the id of the special token
                token_ids.append(self.convert_tokens_to_ids(chunk_or_special_token))
        return token_ids

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

    def convert_tokens_to_string(self, tokens: Iterable[str], skip_special_tokens: bool = False) -> str:
        return ''.join(self._token_to_piece(t, skip_special_tokens) for t in tokens)

    def _id_to_token(self, id: int) -> str:
        return llama_cpp.llama_token_get_text(self.llama.model, id).decode(ENCODING)

    def _id_to_token_type(self, id: int) -> int:
        return llama_cpp.llama_token_get_type(self.llama.model, id)

    def _token_to_id(self, token: str) -> int:
        if self.is_sentencepiece:
            token = escape_whitespace(token)
        return self.token_ids[token]

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
            case llama_cpp.LLAMA_TOKEN_TYPE_BYTE:
                return parse_byte_token(token)
            case _:
                return ''
