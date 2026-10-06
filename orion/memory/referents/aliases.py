"""Alias text rules for referents (memory Stage 2 spec 1.3). Pure functions, no I/O.

No word lists. Whether a name is relative to the speaker ("my boss", "the Wade",
"camera") is read from how often its FIRST word appears in Juniper's own chat
prompts: function words and possessives are among the most frequent words anyone
writes (Luhn 1958; Spärck Jones 1972 idf), so a name that starts with a frequent
word is a description, not a proper name. The cut is #2413's rare-term fraction
(0.5% of documents). Live check 2026-10-06 over 576 prompts: "my" 46, "the" 155,
"a" 133, "camera" 7 (common); "inspur" 1, "jackalope" 1, "boss" 2, "joker" 2 (rare).

The rule applies to the writer's EXTRA aliases only: a thing's own key name ("circe" for
project:circe) is always a proper name (resolve.candidate_aliases), because a frequently
mentioned proper name is frequent for being important, not for being relative.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

# #2413's rare-term cut (RECALL_RARE_TERM_MAX_DF_FRAC): a token in more than this share
# of documents is common.
RARE_DF_FRAC = 0.005
# Juniper's decision (2026-10-06): relative names work as soon as she says them and lapse
# 90 days after their last use.
DESCRIPTOR_TTL_DAYS = 90

_WS = re.compile(r"\s+")
_EDGE_PUNCT = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
_TOKEN = re.compile(r"[\w][\w\-']*", re.UNICODE)


def normalize_alias(text: str) -> str:
    """Lowercased, NFKC, whitespace-collapsed, edge punctuation trimmed. '' if nothing is left."""
    folded = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return _EDGE_PUNCT.sub("", _WS.sub(" ", folded).strip())


def slug_text(key: str) -> str:
    """The name a key spells: "project:orion-camera" -> "orion camera"."""
    return normalize_alias(str(key).partition(":")[2].replace("-", " "))


def first_token(alias_norm: str) -> Optional[str]:
    match = _TOKEN.search(alias_norm)
    return match.group(0) if match else None


def alias_in_text(alias_norm: str, text: str) -> bool:
    """Word-bounded, case-insensitive containment (after the same normalization)."""
    if not alias_norm:
        return False
    hay = normalize_alias(text)
    return re.search(rf"(?<![\w]){re.escape(alias_norm)}(?![\w])", hay) is not None


@dataclass(frozen=True)
class TokenFrequency:
    """How many of ``total`` documents (Juniper's chat prompts) contain the token."""

    documents: int
    total: int

    @property
    def common(self) -> bool:
        return self.total > 0 and self.documents / self.total > RARE_DF_FRAC


def alias_class(alias_norm: str, first_token_frequency: Optional[TokenFrequency]) -> str:
    """'descriptor' when the alias starts with a word common in Juniper's prompts, else 'name'.

    With no frequency data (empty corpus) every alias is a name: the stricter lifecycle
    (expiry) is applied only on evidence, never by default.
    """
    if first_token_frequency is not None and first_token_frequency.common:
        return "descriptor"
    return "name"
