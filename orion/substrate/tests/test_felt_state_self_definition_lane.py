"""The `orion_self_definition` felt-state lane (orion/substrate/felt_state_reader.py).

Pins the two things the new LaneSpec fields exist for: a lane can select its
row with a constant WHERE instead of `projection_id`, and its cache can
refresh faster than its row-age limit -- a self-definition written last week
is still valid, but a new one must reach chat within minutes, not days.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from orion.substrate.felt_state_reader import _LANES, LaneSpec, SubstrateFeltStateReader


def _lane() -> LaneSpec:
    return next(l for l in _LANES if l.ctx_key == "orion_self_definition")


def test_lane_selects_the_curiosity_self_definition_row_only() -> None:
    lane = _lane()
    assert lane.table == "self_concept_history"
    assert lane.projection_id is None
    assert "concept_id = 'self:definition'" in (lane.where_sql or "")
    assert "produced_by = 'curiosity_self_inquiry'" in (lane.where_sql or "")
    assert lane.max_age_sec == 30 * 86400
    assert lane.cache_ttl_sec == 300


class _Conn:
    def __init__(self, sink: list, row):
        self._sink = sink
        self._row = row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params):
        self._sink.append((str(query), dict(params)))

        class _R:
            def __init__(self, row):
                self._row = row

            def mappings(self):
                return self

            def first(self):
                return self._row

        return _R(self._row)


class _Engine:
    def __init__(self, row):
        self.queries: list = []
        self._row = row

    def connect(self):
        return _Conn(self.queries, self._row)


def _reader(row) -> SubstrateFeltStateReader:
    reader = SubstrateFeltStateReader(enabled=False, database_url="postgresql://x", max_age_sec=600)
    reader._enabled = True
    reader._engine = _Engine(row)
    return reader


def test_where_clause_and_aliases_are_rendered_into_the_query() -> None:
    lane = _lane()
    payload = {"content": "I am", "version": 1}
    reader = _reader({"payload": payload, "ts": datetime.now(timezone.utc)})
    ctx: dict = {}
    reader.hydrate(ctx)
    sql = next(q for q, _ in reader._engine.queries if "self_concept_history" in q)
    assert "WHERE concept_id = 'self:definition' AND produced_by = 'curiosity_self_inquiry'" in sql
    assert "AS payload" in sql and "AS ts" in sql
    assert "ORDER BY created_at DESC LIMIT 1" in sql
    assert ctx["orion_self_definition"] == payload


def test_a_week_old_definition_is_still_served() -> None:
    reader = _reader({"payload": {"content": "old but mine"}, "ts": datetime.now(timezone.utc) - timedelta(days=7)})
    ctx: dict = {}
    reader.hydrate(ctx)
    assert ctx["orion_self_definition"] == {"content": "old but mine"}


def test_cache_ttl_is_the_short_one_not_the_row_age() -> None:
    lane = _lane()
    reader = _reader({"payload": {"content": "v1"}, "ts": datetime.now(timezone.utc)})
    ctx: dict = {}
    reader.hydrate(ctx)
    n_after_first = sum(1 for q, _ in reader._engine.queries if "self_concept_history" in q)
    # Age the cache entry past the cache TTL but well inside the row age.
    payload, _ = reader._cache[lane.ctx_key]
    reader._cache[lane.ctx_key] = (payload, time.monotonic() - (lane.cache_ttl_sec + 1))
    reader._engine._row = {"payload": {"content": "v2"}, "ts": datetime.now(timezone.utc)}
    ctx2: dict = {}
    reader.hydrate(ctx2)
    n_after_second = sum(1 for q, _ in reader._engine.queries if "self_concept_history" in q)
    assert n_after_second == n_after_first + 1, "expired cache re-queried"
    assert ctx2["orion_self_definition"] == {"content": "v2"}


def test_projection_id_lanes_still_query_by_pid() -> None:
    reader = _reader({"payload": {"k": 1}, "ts": datetime.now(timezone.utc)})
    ctx: dict = {}
    reader.hydrate(ctx)
    pid_queries = [(q, p) for q, p in reader._engine.queries if "projection_id = :pid" in q]
    assert pid_queries, "projection lanes unchanged"
    assert all("pid" in p for _, p in pid_queries)


def test_a_miss_is_remembered_for_the_cache_ttl_and_does_not_leak_into_ctx() -> None:
    """Review finding 2026-09-08: until the first self-inquiry run lands the
    table has no row, and that steady-state miss used to re-query on every
    chat turn and gate tick."""
    lane = _lane()
    reader = _reader(None)
    ctx: dict = {}
    reader.hydrate(ctx)
    n1 = sum(1 for q, _ in reader._engine.queries if "self_concept_history" in q)
    reader.hydrate({})
    n2 = sum(1 for q, _ in reader._engine.queries if "self_concept_history" in q)
    assert n1 == 1 and n2 == 1, "second hydrate inside the TTL did not re-query"
    assert "orion_self_definition" not in ctx
    # After the TTL the lane looks again and picks up a new row.
    reader._cache[lane.ctx_key] = (None, time.monotonic() - (lane.cache_ttl_sec + 1))
    reader._engine._row = {"payload": {"content": "now I exist"}, "ts": datetime.now(timezone.utc)}
    ctx3: dict = {}
    reader.hydrate(ctx3)
    assert ctx3["orion_self_definition"] == {"content": "now I exist"}


def test_lanes_without_an_explicit_cache_ttl_still_requery_on_a_miss() -> None:
    """The negative cache is opt-in per lane: `curiosity_signals` (120s row
    age, no cache_ttl_sec) must not delay a fresh candidate by its own max age."""
    reader = _reader(None)
    reader.hydrate({})
    reader.hydrate({})
    n = sum(1 for q, _ in reader._engine.queries if "substrate_endogenous_curiosity_candidates" in q)
    assert n == 2


def test_a_failing_query_is_also_remembered_as_a_miss() -> None:
    class _Boom(_Engine):
        def connect(self):
            self.queries.append(("boom", {}))
            raise RuntimeError("db down")

    reader = _reader(None)
    reader._engine = _Boom(None)
    reader.hydrate({})
    reader.hydrate({})
    # Every lane errors on connect; the opted-in lane is asked once, the others
    # (no cache_ttl_sec) keep retrying -- so the count is 2*(lanes) - 1.
    assert len(reader._engine.queries) == 2 * len(_LANES) - 1
    assert reader._cache["orion_self_definition"][0] is None


def test_hydrate_can_be_limited_to_one_lane() -> None:
    reader = _reader({"payload": {"content": "I am"}, "ts": datetime.now(timezone.utc)})
    ctx: dict = {}
    reader.hydrate(ctx, lanes=("orion_self_definition",))
    tables = {q.split("FROM ")[1].split()[0] for q, _ in reader._engine.queries}
    assert tables == {"self_concept_history"}
    assert ctx["orion_self_definition"] == {"content": "I am"}
