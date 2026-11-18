"""Deterministic text normalization for linkage. No word lists."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Set

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s']+")
_SNAKE_KEY = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_JINJA_BLOCK = re.compile(r"(\{\{.*?\}\}|\{%.*?%\})", re.S)
_IDENT = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def fold(text: str) -> str:
    """Lowercase, strip accents, '_'/'-'/'/' to spaces, drop punctuation except apostrophes."""
    t = strip_accents(str(text)).lower().replace("’", "'")
    t = re.sub(r"[_\-/]+", " ", t)
    t = _NON_WORD.sub(" ", t)
    return _WS.sub(" ", t).strip()


def tokens(text: str) -> List[str]:
    """Folded tokens with possessive 's removed ("amy's taste" -> amy, taste)."""
    out = []
    for tok in fold(text).split(" "):
        tok = re.sub(r"'s$", "", tok).strip("'")
        if tok:
            out.append(tok)
    return out


def is_snake_key(label: str) -> bool:
    return bool(_SNAKE_KEY.match(str(label).strip()))


def possessive_head(label: str) -> str | None:
    """Possessive head of a label (amy's taste -> amy); None when there is none."""
    m = re.match(r"^\s*([^\s']+)'s\s+\S", strip_accents(str(label)).lower().replace("’", "'"))
    return m.group(1) if m else None


def prompt_context_keys(prompt_dirs: Iterable[Path]) -> Set[str]:
    """snake_case identifiers used inside Jinja blocks of the prompt templates.

    Harvested from the templates themselves (not a hand list), so a label like
    'x_relationship_summary' is recognized as a leaked prompt field for any
    person whose summary key a template renders.
    """
    keys: Set[str] = set()
    for d in prompt_dirs:
        for p in sorted(Path(d).glob("*.j2")):
            for block in _JINJA_BLOCK.findall(p.read_text(encoding="utf-8", errors="replace")):
                keys.update(_IDENT.findall(block))
    return keys


def as_key(label: str) -> str:
    return "_".join(tokens(label))
