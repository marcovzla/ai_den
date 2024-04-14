from pathlib import Path
from typing import Optional, TypeVar, overload
from collections.abc import Iterable
import numpy as np
from llama_cpp import Llama, LlamaGrammar
from transformers.utils import is_in_notebook
from ai_den.utils.paths import PathLike
from ai_den.llama_cpp.tokenizer import LlamaCppTokenizer


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
            **kwargs,
    ) -> str:
        kwargs['stream'] = verbose

        if json_mode and 'grammar' not in kwargs:
            kwargs['grammar'] = self.load_grammar('json')

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

        return generated_text

    def load_grammar(self, name: str, *, verbose: bool = False) -> LlamaGrammar:
        return LlamaGrammar.from_file(self.grammars_dir / f'{name}.gbnf', verbose=verbose)

    def create_completion(
            self,
            prompt: str,
            *,
            stream: bool = False,
            temperature: float = 0.0,
            max_tokens: Optional[int] = None,
            grammar: Optional[LlamaGrammar] = None,
    ):
        messages = []

        if self.system_prompt:
            messages.append({
                'role': 'system',
                'content': self.system_prompt,
            })

        messages.append({
            'role': 'user',
            'content': prompt,
        })

        return self.llm.create_chat_completion(
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            grammar=grammar,
        )

    def logprob(self, text: str) -> float:
        """Computes the log-probability of the given string."""
        if not self.llm.context_params.logits_all:
            raise RuntimeError('Must set logits_all to True to use logprob()')
        # tokenize string
        token_ids = self.tokenize(text)
        # predict one token to compute logits
        self.llm.create_completion(token_ids, max_tokens=1)
        # compute log-probabilities
        logprobs = Llama.logits_to_logprobs(self.llm._scores)
        # drop first token_id (the first prediction is the second token)
        token_ids = token_ids[1:]
        # compute log-probability for the full string
        positions = np.arange(len(token_ids))
        return logprobs[positions, token_ids].sum().item()

    def rank(self, items: Iterable[str], *, prefix: str = '', suffix: str = '') -> list[tuple[float, str]]:
        return sorted(
            ((self.logprob(f'{prefix}{item}{suffix}'), item) for item in items),
            reverse=True,
        )

    def select(self, items: Iterable[str], *, prefix: str = '', suffix: str = '') -> str:
        return self.rank(items, prefix=prefix, suffix=suffix)[0][1]

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
