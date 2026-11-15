"""``page:anchor``: a few fixed lines Juniper wrote, shown first in RELATIONSHIP, above page:juniper.

Issue #3460. The nightly fold (``orion/memory/pages/juniper.py``, run by orion-durable-runs) writes
only ``page:juniper`` rows. These lines are different: Juniper wrote them herself (who Orion is,
who she is and her pronouns, her family), and nothing Orion folds may change them.

Where the text lives: one hand-written ``self_concept_history`` row on the host database,
``concept_id='page:anchor'``, ``produced_by='juniper_anchor'``. Not in git: the repo is public and
the lines name her family. Two things keep the fold away from it:

1. The fold only reads and writes ``concept_id='page:juniper'`` (its SQL is bound to that id).
2. ``juniper_anchor`` is not a value ``SelfConceptHistoryV1.produced_by`` accepts, so no bus
   producer can publish an anchor row: orion-sql-writer would reject it. The row is written by
   hand (see ``WRITE_ANCHOR_SQL_TEMPLATE``) and changed only the same way.

One gap stays: orion-sql-writer upserts ``self_concept_history`` by ``entry_id``, so a valid bus row
that reused the anchor's random ``entry_id`` would replace it. The reader lane never puts that id in
turn ctx, so nothing downstream can learn it.

The felt-state reader lane ``orion_core_page_anchor`` hydrates the newest such row; cortex-exec puts
it on the grounding capsule as a ``CorePageV1`` ahead of page:juniper.
"""
from __future__ import annotations

from typing import Any, Mapping

PAGE_ANCHOR = "page:anchor"
ANCHOR_PRODUCED_BY = "juniper_anchor"
# The felt-state lane key cortex-exec hydrates (orion/substrate/felt_state_reader.py).
CTX_KEY_ANCHOR = "orion_core_page_anchor"

# How the live row is written (psql, by hand, on the host). :'anchor' is a psql variable so the
# text never lands in shell history as part of the SQL. Version = previous max + 1, so a rewrite
# is a new row and the old one stays (append-only, like every self_concept_history row).
WRITE_ANCHOR_SQL_TEMPLATE = (
    "INSERT INTO self_concept_history (entry_id, created_at, concept_id, version, content, "
    "evidence_refs, produced_by) SELECT gen_random_uuid()::text, now(), 'page:anchor', "
    "coalesce(max(version), 0) + 1, :'anchor', '[]'::json, 'juniper_anchor' "
    "FROM self_concept_history WHERE concept_id = 'page:anchor';"
)


def anchor_page_from_payload(payload: Any):
    """The felt-state lane payload (``CTX_KEY_ANCHOR``) -> ``CorePageV1``, or None when there is
    no usable anchor. The text is shown exactly as written (whitespace trimmed)."""
    from orion.schemas.thought import CorePageV1

    if not isinstance(payload, Mapping):
        return None
    text = str(payload.get("content") or "").strip()
    if not text:
        return None
    try:
        version = int(payload.get("version") or 0)
    except (TypeError, ValueError):
        version = 0
    return CorePageV1(
        concept_id=PAGE_ANCHOR, version=version, text=text,
        written_at=str(payload.get("created_at") or "")[:10] or None, evidence_count=0)
