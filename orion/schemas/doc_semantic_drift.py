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
    # Real git diff status for this file in the scored range ("added" /
    # "modified"), read from `git diff --name-status` rather than inferred
    # from an empty hunk. Exists because `diff_scoped_embedding_diff=None`
    # had two structurally different causes that no consumer could tell
    # apart (confirmed live 2026-08-12 on real events): a newly-added file
    # has no "before" text at all, so no removed-side embedding is requested
    # and the cosine is genuinely undefined -- correct-by-definition, not a
    # failure.
    #
    # Note what this field alone does NOT tell you. `change_kind` is not a
    # sufficient alarm condition on its own: a pure-append edit (adding a
    # section to an existing doc -- the most common doc edit in this repo)
    # has git status "M" but an empty removed side under --unified=0, so it
    # also produces a legitimate None. Measured over this repo's last 300
    # commits: 56 of 161 modified-doc changes (34.8%) have no removed side.
    #
    # Use the chunk counts to disambiguate, not this field:
    #   chunk_count_removed == 0 or chunk_count_added == 0
    #       -> structurally undefined, expected, not a failure
    #   both > 0 and diff is None
    #       -> a real embedding request failed; this is the alarm
    #
    # "deleted" is declared here but never produced today: the producer's
    # `changed_doc_files_with_status()` filters to --diff-filter=ACMR, deliberately
    # excluding deletions (a deleted doc has no "after" state to score
    # drift against). Declared so a consumer's match is exhaustive if that
    # filter is ever widened.
    change_kind: Literal["added", "modified", "deleted"] | None = None
    # None when either side's embedding request failed (e.g. the bus/
    # embedding host was unavailable this tick), or -- expected, not a
    # failure -- when one side has no text at all (see `change_kind`).
    # A real "couldn't measure this one" fact, not a fabricated 0.0.
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
    # 1 = the hunk fit in a single embedding window, and the score reduces
    # exactly to the pre-chunking `1 - cos(a, b)` -- directly comparable to
    # earlier events and to the 0.3996 threshold proposed in the original
    # calibration replay. >1 = a max-chunk-pair score over that many
    # windows, which is length-invariant by construction (see
    # `_max_pair_drift()` for why a mean-pooled score was rejected: it
    # diluted an identical real edit 54x between a 1-chunk and an 8-chunk
    # doc).
    #
    # Honest residual: CHUNK_CHAR_SIZE is a measured chars-per-token proxy,
    # not a live token count. A chunk denser than the observed 2.81
    # chars/token minimum could still exceed 512 tokens and be clipped
    # inside the model without this schema knowing.
    chunk_count_removed: int = 0
    chunk_count_added: int = 0
    hunk_removed_len_chars: int = 0
    hunk_added_len_chars: int = 0
