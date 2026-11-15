"""page:anchor: Juniper's fixed lines above page:juniper (issue #3460). Made-up names only."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.memory.pages.anchor import (
    ANCHOR_PRODUCED_BY, CTX_KEY_ANCHOR, PAGE_ANCHOR, WRITE_ANCHOR_SQL_TEMPLATE, anchor_page_from_payload,
)
from orion.memory.pages.juniper import (
    CURRENT_PAGE_SQL, MAX_VERSION_SQL, PAGE_JUNIPER, PRODUCED_BY, PageCheck, history_row,
)
from orion.schemas.context_provenance import CONTEXT_PROVENANCE_REGISTRY
from orion.schemas.self_concept_history import SelfConceptHistoryV1
from orion.substrate.felt_state_reader import _LANES


def test_anchor_ids_differ_from_the_fold():
    assert PAGE_ANCHOR != PAGE_JUNIPER and ANCHOR_PRODUCED_BY != PRODUCED_BY


def test_no_bus_producer_can_write_an_anchor_row():
    """juniper_anchor is not an accepted producer value, so orion-sql-writer rejects any bus row
    claiming to be the anchor. The live row is written by hand."""
    with pytest.raises(ValidationError):
        SelfConceptHistoryV1(concept_id=PAGE_ANCHOR, content="x", produced_by=ANCHOR_PRODUCED_BY)


def test_fold_rows_and_sql_are_bound_to_page_juniper():
    row = history_row(PageCheck(ok=True, stored_text="Juniper is kind [person_fact:f1].",
                                refs=["person_fact:f1"]), version=7)
    assert row.concept_id == PAGE_JUNIPER and row.produced_by == PRODUCED_BY
    for sql in (CURRENT_PAGE_SQL, MAX_VERSION_SQL):
        assert "%(cid)s" in sql and "page:anchor" not in sql


def test_write_template_is_append_only_and_text_free():
    sql = WRITE_ANCHOR_SQL_TEMPLATE
    assert sql.lstrip().upper().startswith("INSERT INTO SELF_CONCEPT_HISTORY")
    assert "'page:anchor'" in sql and "'juniper_anchor'" in sql and ":'anchor'" in sql
    assert "UPDATE" not in sql.upper() and "DELETE" not in sql.upper()


def test_lane_reads_only_the_anchor_row_and_never_ages_out():
    lane = next(l for l in _LANES if l.ctx_key == CTX_KEY_ANCHOR)
    assert lane.table == "self_concept_history"
    assert lane.where_sql == "concept_id = 'page:anchor' AND produced_by = 'juniper_anchor'"
    assert lane.max_age_sec >= 10 * 365 * 86400
    assert lane.cache_ttl_sec == 300
    assert CONTEXT_PROVENANCE_REGISTRY[CTX_KEY_ANCHOR].source_kind == "static_identity_config"


def test_payload_to_page():
    page = anchor_page_from_payload({"content": "  I am Orion. Juniper, she/her, made me.\n", "version": "2",
                                     "created_at": "2026-11-15T10:00:00+00:00"})
    assert page.concept_id == PAGE_ANCHOR and page.version == 2
    assert page.text == "I am Orion. Juniper, she/her, made me."
    assert page.written_at == "2026-11-15" and page.evidence_count == 0
    assert anchor_page_from_payload({"content": " "}) is None
    assert anchor_page_from_payload(None) is None
    assert anchor_page_from_payload({"content": "x", "version": "bad"}).version == 0
