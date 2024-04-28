import re
from pathlib import Path
from typing import Optional
from collections.abc import Callable
from natsort import natsorted
from datasets import Dataset
from ai_den.utils.paths import PathLike


def read_conll_file(
        path: PathLike,
        columns: dict[str, int],
        comment_symbol: Optional[str] = '#',
        field_separator: str = '\t',
        sequence_separator: str = '\n\\s*\n',
        document_separator: Optional[str] = None,
        document_separator_field: str = 'text',
        filename_field: Optional[str] = None,
        preprocess_text: Optional[Callable[[str], str]] = None,
        encoding: Optional[str] = 'utf-8',
        errors: Optional[str] = 'replace',
):
    return Dataset.from_generator(
        generator=gen_conll_file,
        gen_kwargs=dict(
            path=path,
            columns=columns,
            comment_symbol=comment_symbol,
            field_separator=field_separator,
            sequence_separator=sequence_separator,
            document_separator=document_separator,
            document_separator_field=document_separator_field,
            filename_field=filename_field,
            preprocess_text=preprocess_text,
            encoding=encoding,
            errors=errors,
        ),
    )


def read_conll_directory(
        path: PathLike,
        columns: dict[str, int],
        glob: str = '*',
        comment_symbol: Optional[str] = '#',
        field_separator: str = '\t',
        sequence_separator: str = '\n\\s*\n',
        document_separator: Optional[str] = None,
        document_separator_field: str = 'text',
        filename_field: Optional[str] = None,
        preprocess_text: Optional[Callable[[str], str]] = None,
        encoding: Optional[str] = 'utf-8',
        errors: Optional[str] = 'replace',
):
    return Dataset.from_generator(
        generator=gen_conll_directory,
        gen_kwargs=dict(
            path=path,
            columns=columns,
            glob=glob,
            comment_symbol=comment_symbol,
            field_separator=field_separator,
            sequence_separator=sequence_separator,
            document_separator=document_separator,
            document_separator_field=document_separator_field,
            filename_field=filename_field,
            preprocess_text=preprocess_text,
            encoding=encoding,
            errors=errors,
        ),
    )


def gen_conll_directory(
        path: PathLike,
        columns: dict[str, int],
        glob: str = '*',
        comment_symbol: Optional[str] = '#',
        field_separator: str = '\t',
        sequence_separator: str = '\n\\s*\n',
        document_separator: Optional[str] = None,
        document_separator_field: str = 'text',
        filename_field: Optional[str] = None,
        preprocess_text: Optional[Callable[[str], str]] = None,
        encoding: Optional[str] = 'utf-8',
        errors: Optional[str] = 'replace',
):
    path = Path(path)
    for f in natsorted(path.glob(glob)):
        if f.is_file():
            yield from gen_conll_file(
                path=f,
                columns=columns,
                comment_symbol=comment_symbol,
                field_separator=field_separator,
                sequence_separator=sequence_separator,
                document_separator=document_separator,
                document_separator_field=document_separator_field,
                filename_field=filename_field,
                preprocess_text=preprocess_text,
                encoding=encoding,
                errors=errors,
            )


def gen_conll_file(
        path: PathLike,
        columns: dict[str, int],
        comment_symbol: Optional[str] = '#',
        field_separator: str = '\t',
        sequence_separator: str = '\n\\s*\n',
        document_separator: Optional[str] = None,
        document_separator_field: str = 'text',
        filename_field: Optional[str] = None,
        preprocess_text: Optional[Callable[[str], str]] = None,
        encoding: Optional[str] = 'utf-8',
        errors: Optional[str] = 'replace',
):
    path = Path(path)
    text = path.read_text(encoding, errors)
    if preprocess_text is not None:
        text = preprocess_text(text)
    text = text.strip()
    for entry in re.split(sequence_separator, text):
        fields = dict()
        if filename_field is not None:
            fields[filename_field] = path.name
        for line in entry.splitlines():
            if comment_symbol is not None and line.startswith(comment_symbol):
                continue
            line_entries = re.split(field_separator, line)
            for name, i in columns.items():
                if name not in fields:
                    fields[name] = []
                fields[name].append(line_entries[i])
        if any(entry == document_separator for entry in fields[document_separator_field]):
            continue
        yield fields
