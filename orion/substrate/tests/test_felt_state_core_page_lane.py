"""The `orion_core_page_juniper` felt-state lane: Orion's page:juniper (core memory pages Phase 1a)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.memory.pages.juniper import CTX_KEY_JUNIPER, PAGE_JUNIPER, PRODUCED_BY
from orion.substrate.felt_state_reader import _LANES, SubstrateFeltStateReader


def _lane():
    return next(l for l in _LANES if l.ctx_key == CTX_KEY_JUNIPER)


def test_lane_selects_only_the_folded_page_row():
    lane = _lane()
    assert lane.table == "self_concept_history"
    assert f"concept_id = '{PAGE_JUNIPER}'" in (lane.where_sql or "")
    assert f"produced_by = '{PRODUCED_BY}'" in (lane.where_sql or "")
    assert "'concept_id', concept_id" in lane.payload_col
    # A page stays current until the next fold replaces it; the cache still refreshes in minutes.
    assert lane.max_age_sec == 365 * 86400
    assert lane.cache_ttl_sec == 300


class _Engine:
    def __init__(self, row):
        self.queries: list = []
        self._row = row

    def connect(self):
        engine = self

        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, query, params):
                engine.queries.append(str(query))

                class _R:
                    def mappings(self):
                        return self

                    def first(self):
                        return engine._row

                return _R()

        return _Conn()


def test_subset_hydrate_reads_one_query_and_keeps_a_month_old_page():
    reader = SubstrateFeltStateReader(enabled=False, database_url="postgresql://x", max_age_sec=120)
    reader._enabled = True
    payload = {"concept_id": PAGE_JUNIPER, "content": "Juniper lives in Millbrook.", "version": 1}
    reader._engine = _Engine({"payload": payload, "ts": datetime.now(timezone.utc) - timedelta(days=40)})
    ctx: dict = {}
    reader.hydrate(ctx, lanes=(CTX_KEY_JUNIPER,))
    assert ctx[CTX_KEY_JUNIPER] == payload
    assert len(reader._engine.queries) == 1
    assert "page:juniper" in reader._engine.queries[0]


def test_origin_lane_selects_only_the_folded_origin_row():
    from orion.memory.pages.origin import CTX_KEY_ORIGIN, PAGE_ORIGIN

    lane = next(l for l in _LANES if l.ctx_key == CTX_KEY_ORIGIN)
    assert lane.table == "self_concept_history"
    assert lane.where_sql == f"concept_id = '{PAGE_ORIGIN}' AND produced_by = '{PRODUCED_BY}'"
    assert "'concept_id', concept_id" in lane.payload_col and "'evidence_refs', evidence_refs" in lane.payload_col
    assert lane.max_age_sec == 365 * 86400 and lane.cache_ttl_sec == 300


def test_lately_lane_selects_only_the_folded_lately_row_and_goes_stale_in_three_weeks():
    from orion.memory.pages.lately import CTX_KEY_LATELY, PAGE_LATELY

    lane = next(l for l in _LANES if l.ctx_key == CTX_KEY_LATELY)
    assert lane.table == "self_concept_history"
    assert lane.where_sql == f"concept_id = '{PAGE_LATELY}' AND produced_by = '{PRODUCED_BY}'"
    assert "'concept_id', concept_id" in lane.payload_col and "'evidence_refs', evidence_refs" in lane.payload_col
    assert lane.max_age_sec == 21 * 86400 and lane.cache_ttl_sec == 300
