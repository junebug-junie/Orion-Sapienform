from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from app.db import Base


class DocSemanticDriftSQL(Base):
    """Durable record of orion-cocreation-signals' doc_semantic_drift
    producer -- one row per real ``*.md`` file scored at a given commit,
    published on ``orion:substrate:doc_semantic_drift``.

    First real consumer of that channel, which was ``consumer_services: []``
    from 2026-08-11 until this patch. That mattered more than "shadow write"
    made it sound: ``OrionBusAsync.publish()`` is Redis pub/sub
    (``async_service.py:271``), not a stream, so with no subscriber every
    event was dropped the moment it was published. The only record was the
    producer container's stdout, which every redeploy wiped -- and this
    service redeploys often. Roughly two days of real doc-drift scores exist
    nowhere.

    That is also why this exists *before* any threshold or consumer logic:
    the open question for this signal is what a meaningful drift cutoff is,
    the original 0.3996 calibration only applies to single-window events
    (see ``DocSemanticDriftV1.chunk_count_removed``), and no multi-chunk
    threshold can be derived without accumulated real samples. This table is
    the accumulation. Deliberately no Hub panel and no consumer behavior in
    the same patch -- just stop throwing the data away.

    ``event_id`` is derived from (sha, path) rather than a uuid, so a
    redelivered event or a re-scored range upserts onto the same row instead
    of inflating the sample count with duplicates -- which would quietly
    corrupt exactly the distribution this table is being kept for.
    """

    __tablename__ = "doc_semantic_drift_log"

    event_id = Column(String, primary_key=True)
    observed_at = Column(DateTime(timezone=True), nullable=False)

    sha = Column(String, nullable=False, index=True)
    # Range start (exclusive) -- the score covers `(base_sha, sha]`, which
    # is often but not always a single commit. Nullable because events
    # published before 2026-08-13 carry no such field. Without this column
    # a multi-commit row is indistinguishable from a single-commit one, and
    # overlapping re-scores after a partial publish failure are undetectable.
    base_sha = Column(String, nullable=True, index=True)
    path = Column(String, nullable=False, index=True)
    # Self-declared conventional-commit type ("docs", "fix", ...) or NULL --
    # a weak feature by the design doc's own admission (a `docs:` prefix is
    # claimed, not verified against what changed).
    commit_prefix = Column(String, nullable=True)

    # Max chunk-pair drift. Nullable, and after the 2026-08-13 skip patch a
    # NULL here means exactly one thing: a real embedding failure.
    # Structurally-unscoreable changes (a new file, a pure append -- one
    # hunk side empty) are no longer published at all, so a NULL in this
    # column is an alarm, not an expected state.
    diff_scoped_embedding_diff = Column(Float, nullable=True)

    # Both == 1 is the comparability condition -- only then does the score
    # reduce exactly to the pre-chunking `1 - cos(a, b)` and stay comparable
    # to the 0.3996 single-window threshold. Indexed because that filter is
    # the first query anyone deriving a threshold will run.
    chunk_count_removed = Column(Integer, nullable=False, index=True)
    chunk_count_added = Column(Integer, nullable=False, index=True)

    hunk_removed_len_chars = Column(Integer, nullable=False)
    hunk_added_len_chars = Column(Integer, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
