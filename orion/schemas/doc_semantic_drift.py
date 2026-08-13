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

from pydantic import BaseModel, ConfigDict, model_validator


class DocSemanticDriftV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["doc_semantic_drift.v1"] = "doc_semantic_drift.v1"
    # Idempotency key / SQL primary key. Deliberately DERIVED from
    # (sha, path) rather than a uuid4 like DevEconomicsLedgerV1's own
    # event_id: those two fully determine the event (one score per file per
    # scored commit), so a redelivered or re-scored change upserts onto the
    # same row instead of writing a duplicate with a fresh id. Filled
    # automatically below so no caller can forget it or invent a different
    # convention.
    event_id: str = ""
    observed_at: datetime
    sha: str
    path: str
    commit_prefix: str | None = None
    # None here means a real embedding failure -- and only that.
    #
    # It used to have two causes a consumer could not tell apart: a genuine
    # failure, and a structurally-undefined comparison (a new file, a pure
    # append, a pure deletion -- one side has no text, so cosine does not
    # exist). A `change_kind` field was added 2026-08-12 to disambiguate
    # them. The live batch on 2026-08-13 showed why that was the wrong fix:
    # 6 of 8 real events were the structural case, all new PR reports and
    # specs, and that ratio is structural rather than a small sample --
    # this repo's docs are mostly created once and never revised.
    #
    # So the producer now skips those changes entirely instead of publishing
    # a labelled null (`_is_unscoreable()`), and `change_kind` was removed
    # with them: every published event is a modified file with real text on
    # both sides, which made the field a constant, and a constant is not a
    # signal -- the same reason `possibly_truncated` was removed before it.
    #
    # A consumer can therefore treat any None as an alarm.
    diff_scoped_embedding_diff: float | None = None
    # How many <=CHUNK_CHAR_SIZE windows each side's hunk was split into
    # before embedding. Replaces the former `possibly_truncated` bool,
    # which was accurate but had become uninformative: the embedding model
    # (BAAI/bge-large-en-v1.5, read 2026-08-12 from the running
    # orion-vector-host container's own tokenizer_config.json /
    # config.json) has a hard 512-token ceiling, so it fired True on
    # *every* real event this repo produced -- a constant is not a signal.
    # The producer chunks below that ceiling instead of letting the model
    # silently clip; what a consumer needs to know is how much aggregation
    # stands between the raw text and the score, plus (with the field
    # above) whether a None is structural or a real failure.
    #
    # BOTH counts == 1 is the comparability condition: only then does the
    # score reduce exactly to the pre-chunking `1 - cos(a, b)`, directly
    # comparable to earlier events and to the 0.3996 threshold from the
    # original calibration replay. Filtering on one count alone is not
    # enough -- a 1-removed/2-added event is already a max-chunk-pair score,
    # not a plain cosine.
    #
    # Otherwise the score is max chunk-pair drift, which is NOT
    # length-invariant: being a max over 2*(N+M) terms makes it an
    # extreme-value statistic, so more chunks biases it HIGH even with no
    # extra real change (measured over 30 real hunk pairs: median 0.1684 at
    # N=1, 0.2237 at N=2-3, 0.2861 at N=4+). Applying the single-window
    # 0.3996 cutoff across chunk counts will over-flag long docs; a
    # multi-chunk threshold has to be derived separately. See
    # `_max_pair_drift()` for the full note, and for why mean-pooling was
    # rejected outright (it diluted an identical real edit 54x between a
    # 1-chunk and an 8-chunk doc -- a far larger error, in the direction
    # that hides change rather than over-reports it).
    #
    # Honest residual: CHUNK_CHAR_SIZE is a measured chars-per-token proxy,
    # not a live token count. A chunk denser than the observed 2.81
    # chars/token minimum could still exceed 512 tokens and be clipped
    # inside the model without this schema knowing.
    chunk_count_removed: int = 0
    chunk_count_added: int = 0
    hunk_removed_len_chars: int = 0
    hunk_added_len_chars: int = 0

    @model_validator(mode="after")
    def _fill_event_id(self) -> "DocSemanticDriftV1":
        if not self.event_id:
            # object.__setattr__ not needed -- the model is mutable by
            # default; assigning in an "after" validator does not re-trigger
            # it, so this cannot recurse.
            self.event_id = f"doc-semantic-drift:{self.sha}:{self.path}"
        return self
