"""Bus payload schema for the doc-semantic-drift domain's
``orion:substrate:doc_semantic_drift`` channel
(docs/superpowers/specs/2026-07-30-doc-semantic-drift-design.md).

One event per real ``*.md`` file changed at a given commit. Carries the
diff-scoped embedding-diff score -- confirmed by real replay
(docs/superpowers/pr-reports/2026-08-11-doc-semantic-drift-diff-scoped-
embedding.md) to separate trivial doc edits from real ones, unlike a
whole-document embedding-diff, which that same replay found structurally
broken for this repo's real (5KB-74KB) docs.

Aggregate scalar only -- the actual hunk text (what changed) is never
persisted here, same privacy discipline as every other signal in this
program. The hunk text *does* get embedded via the real
``orion:embedding:generate`` bus contract, which -- unlike this schema --
does persist a vector-store document (a deliberate choice, not an
oversight: see the producer module's own docstring for why, and note it's
scoped to its own ``doc_semantic_drift`` collection, not commingled with
chat/social memory).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class DocSemanticDriftV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["doc_semantic_drift.v1"] = "doc_semantic_drift.v1"
    observed_at: datetime
    sha: str
    path: str
    commit_prefix: str | None = None
    # None when either side's embedding request failed (e.g. the bus/
    # embedding host was unavailable this tick) -- a real "couldn't measure
    # this one" fact, not a fabricated 0.0.
    diff_scoped_embedding_diff: float | None = None
    # Real replay found 2 of 5 real hunks still exceed the embedding
    # model's token limit even scoped to just the diff (a single large
    # hunk can itself be long) -- named `possibly_truncated`, matching
    # `PrLifecycleDeltaPayloadV1.possibly_truncated`'s existing naming
    # convention in this same codebase for "real, disclosed, not a
    # guaranteed fact" rather than inventing new vocabulary for the same
    # shape of uncertainty. Best-effort (character-length heuristic against
    # this repo's real corpus, not a live token count from the embedding
    # host -- that introspection isn't available over the real bus
    # contract, only via the offline calibration script's direct container
    # access) -- may under- or over-flag relative to the model's real
    # tokenizer.
    possibly_truncated: bool = False
    hunk_removed_len_chars: int = 0
    hunk_added_len_chars: int = 0
