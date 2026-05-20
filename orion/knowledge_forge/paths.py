from __future__ import annotations

import os
from pathlib import Path


def resolve_corpus_root() -> Path:
    env = os.environ.get("ORION_KNOWLEDGE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # Walk up from cwd looking for orion-knowledge/
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        root = candidate / "orion-knowledge"
        if root.is_dir():
            return root.resolve()
    return (cwd / "orion-knowledge").resolve()
