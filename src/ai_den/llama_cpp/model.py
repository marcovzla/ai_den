import json
from pathlib import Path
from typing import Optional, TypeVar, overload
from collections.abc import Iterable

import numpy as np
from llama_cpp import Llama, LlamaGrammar, LogitsProcessor, ChatCompletionRequestMessage
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
from transformers.utils import is_in_notebook

from ai_den.utils.paths import PathLike
from ai_den.llama_cpp.tokenizer import LlamaCppTokenizer
from ai_den.llama_cpp.data_type import DataType


T = TypeVar('T')


GRAMMARS_DIR = Path(__file__).parent / 'grammars'


class LlamaCpp:
    def __init__(
            self,
            model_path: PathLike,
            system_prompt: Optional[str] = None,
            n_ctx: int = 8192,
            n_threads: int = 8,
            n_gpu_layers: int = -1,
            logits_all: bool = True,
            verbose: bool = False,
            grammars_dir: PathLike = GRAMMARS_DIR,
    ):
        self.model_path = Path(model_path)
        self.system_prompt = system_prompt
        self.grammars_dir = Path(grammars_dir)

        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            logits_all=logits_all,
            verbose=verbose,
        )

        self.tokenizer = LlamaCppTokenizer(self.llm)

    def __call__(
            self,
            prompt: str,
            *,
            verbose: bool = False,
            json_mode: bool = False,
            chat_mode: bool = True,
            data_type: Optional[type[T]] = None,
            strict: bool = False,
            **kwargs,
    ) -> str:
        kwargs['stream'] = verbose

        data_class = None

        if data_type:
            json_mode = True
            data_class = DataType(data_type)
            kwargs['grammar'] = data_class.llama_grammar()
        elif json_mode and 'grammar' not in kwargs:
            kwargs['grammar'] = self.load_grammar('json')

        if chat_mode:
            messages = self.prompt_to_messages(prompt)
            resp = self.create_chat_completion(messages, **kwargs)
        else:
            resp = self.create_completion(prompt, **kwargs)

        generated_text = ''

        if verbose:
            if is_in_notebook():
                from IPython.display import Markdown, display

                handle = display(Markdown(generated_text), display_id=True)

                for chunk in resp:
                    if content := chunk['choices'][0]['delta'].get('content'):
                        generated_text += content
                        markdown = f'```json\n{generated_text}\n```' if json_mode else generated_text
                        handle.update(Markdown(markdown))

            else:
                for chunk in resp:
                    if content := chunk['choices'][0]['delta'].get('content'):
                        generated_text += content
                        print(content, end='')
                print()

        else:
            generated_text = resp['choices'][0]['message']['content']

        if data_class:
            return data_class.parse_json(generated_text, strict=strict)
        elif json_mode:
            return json.loads(generated_text)
        else:
            return generated_text

    def set_chat_template(self, template: str):
        self.llm.metadata['tokenizer.chat_template'] = template
        self.llm.chat_handler = self.chat_formatter(add_generation_prompt=True).to_chat_handler()

    def chat_formatter(
            self,
            *,
            template: Optional[str] = None,
            bos_token: Optional[str] = None,
            eos_token: Optional[str] = None,
            add_generation_prompt: bool = False,
            stop_token_ids: Optional[list[int]] = None,
    ) -> Jinja2ChatFormatter:
        return Jinja2ChatFormatter(
            template=template or self.llm.metadata['tokenizer.chat_template'],
            bos_token=bos_token or self.tokenizer.bos_token,
            eos_token=eos_token or self.tokenizer.eos_token,
            add_generation_prompt=add_generation_prompt,
            stop_token_ids=[self.tokenizer.eos_token_id] if stop_token_ids is None else stop_token_ids,
        )

    def prompt_to_messages(
            self,
            prompt: str,
            *,
            system_prompt: Optional[str] = None,
    ) -> list[ChatCompletionRequestMessage]:
        messages = []

        if system_prompt is None:
            system_prompt = self.system_prompt

        if system_prompt is not None:
            messages.append({'role': 'system', 'content': system_prompt})

        messages.append({'role': 'user', 'content': prompt})

        return messages

    def messages_to_prompt(
            self,
            messages: list[ChatCompletionRequestMessage],
            *,
            add_generation_prompt: bool = False,
    ) -> str:
        formatter = self.chat_formatter(add_generation_prompt=add_generation_prompt)
        return formatter(messages=messages).prompt

    def load_grammar(
            self,
            name: str,
            *,
            verbose: bool = False,
    ) -> LlamaGrammar:
        return LlamaGrammar.from_file(self.grammars_dir / f'{name}.gbnf', verbose=verbose)

    def create_grammar(
            self,
            data_type: type,
            *,
            verbose: bool = False,
    ) -> LlamaGrammar:
        return DataType(data_type).llama_grammar(verbose=verbose)

    def create_completion(
            self,
            prompt: str | list[int],
            *,
            stream: bool = False,
            temperature: float = 0.0,
            max_tokens: Optional[int] = None,
            grammar: Optional[LlamaGrammar] = None,
            logprobs: Optional[int] = None,
            logits_processor: Optional[LogitsProcessor] = None,
            data_type: Optional[type[T]] = None,
    ):
        return self.llm.create_completion(
            prompt=prompt,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            grammar=self.create_grammar(data_type) if data_type else grammar,
            logprobs=logprobs,
            logits_processor=logits_processor,
        )

    def create_chat_completion(
            self,
            messages: list[ChatCompletionRequestMessage],
            *,
            stream: bool = False,
            temperature: float = 0.0,
            max_tokens: Optional[int] = None,
            grammar: Optional[LlamaGrammar] = None,
            logprobs: Optional[int] = None,
            logits_processor: Optional[LogitsProcessor] = None,
            data_type: Optional[type[T]] = None,
    ):
        return self.llm.create_chat_completion(
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            grammar=self.create_grammar(data_type) if data_type else grammar,
            logprobs=logprobs is not None,
            top_logprobs=logprobs,
            logits_processor=logits_processor,
        )

    def logprob(self, text: str, start: Optional[int] = None, stop: Optional[int] = None) -> float:
        """Computes the log-probability of the given string."""
        if not self.llm.context_params.logits_all:
            raise RuntimeError('Must set logits_all to True to use logprob()')
        # tokenize string
        token_ids = self.tokenize(text)
        # predict one token to compute logits
        self.create_completion(token_ids, max_tokens=1)
        # compute log-probabilities
        logprobs = Llama.logits_to_logprobs(self.llm._scores)
        # drop first token_id (the first prediction is the second token)
        token_ids = token_ids[1:]
        # get logprob for each output token
        positions = np.arange(len(token_ids))
        token_logprobs = logprobs[positions, token_ids]
        # return total log-probability
        return token_logprobs[start:stop].sum().item()

    def tokenize(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens)

    def detokenize(self, tokens: Iterable[int] | Iterable[str], skip_special_tokens: bool = False) -> str:
        if len(tokens) == 0:
            return ''
        if isinstance(tokens[0], int):
            return self.tokenizer.decode(tokens, skip_special_tokens)
        if isinstance(tokens[0], str):
            return self.tokenizer.convert_tokens_to_string(tokens, skip_special_tokens)
        raise TypeError('Invalid tokens.')

    @overload
    def tokens_to_ids(self, token: str) -> int:
        ...

    @overload
    def tokens_to_ids(self, tokens: Iterable[str]) -> list[int]:
        ...

    def tokens_to_ids(self, x):
        return self.tokenizer.convert_tokens_to_ids(x)

    @overload
    def ids_to_tokens(self, id: int) -> str:
        ...

    @overload
    def ids_to_tokens(self, ids: Iterable[int]) -> list[str]:
        ...

    def ids_to_tokens(self, x):
        return self.tokenizer.convert_ids_to_tokens(x)
