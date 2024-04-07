import os
from pathlib import Path


PathLike = str | os.PathLike[str]


def replace_subpath(path: PathLike, old: PathLike, new: PathLike) -> Path:
    path, old, new = Path(path), Path(old), Path(new)
    return new / path.relative_to(old if old != path else old.parent)
