"""Reverie's machinery-words check: does a text narrate attention / substrate mechanism?

Kept from the retired semantic-lift path (concern cards over coalition pointers, removed 2026-11-24 with the
narration chain, reveries that ask Patch 1). orion-thought's question check
(services/orion-thought/app/reverie.py ``check_question``) refuses a reverie question that names this
machinery: a question about the machinery is not a question about the material.
"""

from __future__ import annotations

import re

# Structural mechanism terms only -- not user-content keyword cathedral.
_INFRA_VOCAB_RE = re.compile(
    r"\b("
    r"coalition|open\s+loop|harness_closure|substrate|stability\s+score|"
    r"dwell\s+tick|attended\s+node|projection_id"
    r")\b",
    re.IGNORECASE,
)


def infra_vocabulary_hit(text: str) -> bool:
    return bool(_INFRA_VOCAB_RE.search(text or ""))
