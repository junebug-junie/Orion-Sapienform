"""Alias text rules for referents (memory Stage 2 spec 1.3). Pure functions, no I/O.

What KIND of name a name is (a proper name for one thing, or a descriptor that can
point at different things over time) is the distiller's judgment, carried as
``alias_kind`` (orion/schemas/memory_episode.py). Code never classifies a name by its
words. Code only checks that a name is in Juniper's verified words, and normalizes text
the same way for key slugs and aliases.
"""

from __future__ import annotations

import re
import unicodedata

# Juniper's decision (2026-10-06): descriptors work as soon as she says them and lapse
# 90 days after their last use.
DESCRIPTOR_TTL_DAYS = 90

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def normalize_alias(text: str) -> str:
    """NFKC, casefolded, every run of punctuation or space becomes one space.

    One rule for key slugs and aliases, so "Fairview, Ohio", "fairview-ohio" and the slug
    of "place:fairview-ohio" all normalize to "fairview ohio".
    """
    folded = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return _NON_WORD.sub(" ", folded).strip()


def slug_text(key: str) -> str:
    """The name a key spells: "project:orion-camera" -> "orion camera"."""
    return normalize_alias(str(key).partition(":")[2])


def alias_in_text(alias_norm: str, text: str) -> bool:
    """Word-bounded containment, after the same normalization on both sides."""
    if not alias_norm:
        return False
    return f" {alias_norm} " in f" {normalize_alias(text)} "
