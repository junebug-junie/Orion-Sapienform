"""The hashed personal-terms matcher, shared by the CI gate and Orion's dig writes.

The repo is public. ``config/privacy/personal_terms.sha256`` holds SHA-256 hashes of lowercased
single words (or emails) that must never be written: names of people close to Juniper, her home
place, her contact details. ``scripts/check_personal_terms.py`` runs this on every tracked file;
``orion.dig field-note write`` runs it on a note before the note is saved, because notes live
outside the repo where the CI gate cannot see them (spec 2026-11-04-pr-field-notes-design.md).

Matching: text is split into lowercase letter-only word tokens (camelCase splits); each token and
each email address (and every dotted suffix of its local part) is hashed and looked up. Whole
words only.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

DENYLIST_REL = Path("config") / "privacy" / "personal_terms.sha256"

_EMAIL = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}")
# Word tokens on the original case, so camelCase splits for free:
# "nameMemory" -> name, Memory; "HTMLName" -> HTML, Name; "NAME_X" -> NAME, X.
_TOKEN = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")
_HASH_CACHE: dict[str, str] = {}


def load_denylist(path: Path) -> tuple[set[str], list[tuple[str, str]]]:
    """Returns (hashes, allow) where allow is [(hash, path_glob)]. Raises ValueError on a bad line."""
    hashes: set[str] = set()
    allow: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "allow" and len(parts) == 3:
            allow.append((parts[1], parts[2]))
        elif re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            hashes.add(parts[0])
        else:
            raise ValueError(f"{path}: bad line: {raw!r}")
    return hashes, allow


def _h(c: str) -> str:
    h = _HASH_CACHE.get(c)
    if h is None:
        h = _HASH_CACHE[c] = hashlib.sha256(c.encode("utf-8")).hexdigest()
    return h


def _ascii_fold(text: str) -> str:
    if text.isascii():
        return text
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def _email_variants(addr: str) -> set[str]:
    """The address plus every dotted suffix of its local part (``x.a.b@d`` -> ``a.b@d``, ``b@d``)."""
    local, _, domain = addr.partition("@")
    parts = local.split(".")
    return {".".join(parts[i:]) + "@" + domain for i in range(len(parts))}


def line_hits(line: str, hashes: set[str]) -> set[str]:
    text = _ascii_fold(line)
    cands = {t.lower() for t in set(_TOKEN.findall(text))}
    low = text.lower()
    for addr in _EMAIL.findall(low.replace("%40", "@")) if "@" in low or "%40" in low else ():
        cands |= _email_variants(addr)
    return {h for h in map(_h, cands) if h in hashes}
