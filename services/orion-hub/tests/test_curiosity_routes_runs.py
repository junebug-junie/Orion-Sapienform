"""`/curiosity/api/runs` and `/curiosity/api/run/{id}`: the store orchestration
in `scripts/curiosity_run_store.py` and the route glue in
`scripts/curiosity_routes.py`.

Fakes stand in for the asyncpg pool and the graph reader so the thing under
test is the join of what each store returned, the never-500 contract, and
the two constants this package cannot import from the loop and therefore
pins by value instead.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

import scripts.curiosity_routes as cr
import scripts.curiosity_run_store as store
from orion.curiosity.atlas import RunNodeRows
from orion.curiosity.worldview import WorldviewReader, WorldviewUnavailable

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


class _Conn:
    def __init__(self, answers, fail=False):
        self.answers = answers
        self.fail = fail
        self.calls: list[tuple[str, tuple]] = []

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        if self.fail:
            raise RuntimeError("pg down")
        for needle, rows in self.answers.items():
            if needle in sql:
                return rows
        return []


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *a):
        return False


class _Pool:
    def __init__(self, answers=None, fail=False):
        self.conn = _Conn(answers or {}, fail=fail)

    def acquire(self):
        return _Acquire(self.conn)


class _Reader(WorldviewReader):
    def __init__(self, answers=None, raises=False):
        super().__init__(host="x", port=1, graph_name="g", client=object())
        self.answers = answers or {}
        self.raises = raises
        self.queries: list[str] = []

    def query(self, cypher):
        self.queries.append(cypher)
        if self.raises:
            raise WorldviewUnavailable("ConnectionError: nope")
        hits = [rows for needle, rows in self.answers.items() if needle in cypher]
        assert len(hits) <= 1, cypher
        return hits[0] if hits else []


def _lifecycle(run_id, status, when, line="investigate", workflow="curiosity.investigate", **detail):
    d = {"line": line, "attempts": 1, **detail} if status == "completed" else detail
    return {"run_id": run_id, "workflow": workflow, "node": "finish", "next_node": "", "status": status,
            "resumed_from_node": None, "correlation_id": "c", "created_at": when, "detail": json.dumps(d)}


def _graph(run_id, start):
    return {
        "RETURN DISTINCT n.run_id": [{"run_id": run_id}],
        "MATCH (n:InvestigationRole)": [{"run_id": run_id, "choice": "local_crawl", "why": "w", "written_at": _ms(start)}],
        "MATCH (n:Hop)": [{"run_id": run_id, "n": 1, "note": "h1", "written_at": _ms(start + timedelta(minutes=5))}],
        "MATCH (n:TurnOutcome)": [{"run_id": run_id, "continue_line": True, "continue_note": "n",
                                   "reach_out": True, "reach_out_why": "because", "written_at": _ms(start + timedelta(minutes=9))}],
    }


def test_runs_payload_joins_both_stores_and_excludes_reflect() -> None:
    start = NOW - timedelta(hours=2)
    pool = _Pool({
        "FROM substrate_durable_run_state": [
            _lifecycle("r1", "completed", NOW - timedelta(hours=1), reach_out=True, reach_out_why="because"),
            _lifecycle("selfy", "completed", NOW - timedelta(hours=3), line="self_inquiry"),
            # The store's SQL filters this out; a fake that returns it anyway
            # proves the join filters too.
            _lifecycle("refl", "completed", NOW - timedelta(hours=4), line="reflect", workflow="self_study.reflect"),
        ],
        "FROM journal_entries": [{"entry_id": "j1", "source_ref": "curiosity:r1", "title": "Curiosity",
                                  "body": "prose", "created_at": NOW - timedelta(hours=1)}],
    })
    reader = _Reader(_graph("r1", start))
    payload = asyncio.run(store.read_runs_payload(pool=pool, reader=reader, days=14, now=NOW))

    assert payload["available"] is True
    assert payload["stores"] == {"postgres": "ok", "graph": "ok"}
    assert [r["run_id"] for r in payload["runs"]] == ["r1", "selfy"], "reflect must not leak"
    r1 = payload["runs"][0]
    assert r1["plain_line_label"] == "World question"
    assert r1["started_at"] == _ms(start) and r1["started_from"] == "graph"
    assert r1["reach_out"] == {"wanted": True, "why": "because", "decision": "not_recorded", "gate": None,
                               "decided_at": None, "sent_at": None, "composed_text": "", "reply": None}
    assert payload["reach_outs"] == {"wanted": 1, "sent": 0, "blocked_by": {}, "top_block_reason": None, "not_recorded": 1}
    assert payload["totals"] == {"investigate": 1, "self_inquiry": 1, "self_sense_eval": 0}
    assert {r["plain_line_label"] for r in payload["runs"]} <= {"World question", "Self question", "Self-sense check"}

    # The window reached every store as a bound, not a row cap.
    lifecycle_call = next(c for c in pool.conn.calls if "substrate_durable_run_state" in c[0])
    assert lifecycle_call[1] == (["curiosity.investigate", "self_sense_eval"], NOW - timedelta(days=14))
    assert "LIMIT" not in lifecycle_call[0]
    assert any("CYPHER since=" in q for q in reader.queries)
    # Secondary reads were keyed by the run's derived outreach id.
    outreach_call = next(c for c in pool.conn.calls if "endogenous_outreach_decisions" in c[0])
    assert str(uuid5(NAMESPACE_URL, "curiosity_outreach:r1")) in outreach_call[1][0]


def test_runs_payload_reads_the_admission_path_and_counts_its_anomalies() -> None:
    t0 = NOW - timedelta(hours=8)
    pool = _Pool({
        "FROM durable_admission_runs": [{"run_id": "adm", "request": json.dumps({"workflow": "self_sense_eval", "brief": {"line": "self_sense_eval"}}),
                                         "created_at": t0, "control": None, "terminal": "completed", "updated_at": NOW - timedelta(hours=1)}],
        "FROM durable_resource_events WHERE run_id = ANY($1::text[]) AND event = ANY($2::text[]) ORDER BY": [
            {"entry_id": "accepted:adm", "run_id": "adm", "event": "run.accepted", "generated_at": t0, "payload": "{}"},
            {"entry_id": "a1", "run_id": "adm", "event": "run.lane_assigned", "generated_at": t0 + timedelta(hours=7),
             "payload": json.dumps({"detail": {"lease": {"lane": "agent"}}})},
            {"entry_id": "adm:terminal:completed", "run_id": "adm", "event": "run.completed", "generated_at": NOW - timedelta(hours=1),
             "payload": json.dumps({"detail": {"finding_text": "scored"}})},
        ],
        "GROUP BY run_id, event": [{"run_id": "adm", "event": "run.checkpoint_resume_failed", "n": 4}],
        # The bridge row for the same run, mislabelled and line-less.
        "FROM substrate_durable_run_state": [_lifecycle("adm", "completed", NOW - timedelta(hours=1), line=None)],
    })
    payload = asyncio.run(store.read_runs_payload(pool=pool, reader=None, now=NOW))
    assert [r["run_id"] for r in payload["runs"]] == ["adm"]
    r = payload["runs"][0]
    assert r["plain_line_label"] == "Self-sense check" and r["line_known"] is True
    assert r["started_from"] == "admission" and r["started_at"] == _ms(t0)
    assert r["lane"] == "agent" and r["lane_wait_sec"] == 7 * 3600.0
    assert r["anomalies"] == {"run.checkpoint_resume_failed": 4}
    events_call = next(c for c in pool.conn.calls if "durable_resource_events" in c[0] and "ORDER BY" in c[0])
    assert events_call[1][0] == ["adm"]
    assert "run.checkpoint_resume_failed" not in events_call[1][1], "noisy events are counted, not fetched"
    assert "run.lane_swap_suppressed" not in events_call[1][1]


def test_runs_payload_filters_by_line_but_tallies_across_all() -> None:
    pool = _Pool({"FROM substrate_durable_run_state": [
        _lifecycle("r1", "completed", NOW - timedelta(hours=1), reach_out=True, reach_out_why="w"),
        _lifecycle("selfy", "completed", NOW - timedelta(hours=3), line="self_inquiry"),
    ]})
    payload = asyncio.run(store.read_runs_payload(pool=pool, reader=None, line="self_inquiry", now=NOW))
    assert [r["run_id"] for r in payload["runs"]] == ["selfy"]
    assert payload["reach_outs"]["wanted"] == 1, "the tile is for the whole window, not the filter"
    assert payload["stores"]["graph"] == "graph_not_configured"


def test_a_dead_postgres_still_yields_graph_only_stories() -> None:
    start = NOW - timedelta(hours=2)
    payload = asyncio.run(store.read_runs_payload(pool=_Pool(fail=True), reader=_Reader(_graph("r1", start)), now=NOW))
    assert payload["available"] is True
    assert payload["stores"]["postgres"].startswith("RuntimeError")
    assert payload["runs"][0]["run_id"] == "r1"
    assert payload["runs"][0]["line_known"] is False


def test_both_stores_dead_is_unavailable_not_an_empty_fortnight() -> None:
    payload = asyncio.run(store.read_runs_payload(pool=_Pool(fail=True), reader=_Reader(raises=True), now=NOW))
    assert payload["available"] is False
    assert "runs" not in payload
    assert payload["stores"]["graph"].startswith("ConnectionError")


def test_no_pool_and_no_reader_is_unavailable() -> None:
    payload = asyncio.run(store.read_runs_payload(pool=None, reader=None, now=NOW))
    assert payload == {"available": False, "reason": "no_pool",
                       "stores": {"postgres": "no_pool", "graph": "graph_not_configured"},
                       "window_days": 14, "line": "all"}


def test_days_and_line_are_clamped() -> None:
    assert store.clamp_days(400) == 90
    assert store.clamp_days(0) == 1
    assert store.clamp_days("x") == 14
    assert store.clamp_days(None) == 14
    assert store.clamp_line("self_sense_eval") == "self_sense_eval"
    assert store.clamp_line("everything") == "all"
    assert store.clamp_line(None) == "all"


def test_run_payload_reads_one_run_by_id_and_validates_it() -> None:
    start = NOW - timedelta(hours=2)
    pool = _Pool({
        "FROM substrate_durable_run_state": [_lifecycle("r1", "completed", NOW - timedelta(hours=1))],
        "FROM journal_entries": [{"entry_id": "j1", "source_ref": "curiosity:r1", "title": "Curiosity",
                                  "body": "prose", "created_at": NOW - timedelta(hours=1)}],
    })
    payload = asyncio.run(store.read_run_payload(pool=pool, reader=_Reader(_graph("r1", start)), run_id="r1"))
    assert payload["found"] is True and payload["available"] is True
    assert payload["run"]["run_id"] == "r1"
    assert [it["kind"] for it in payload["timeline"]][:2] == ["role_choice", "hop"]
    assert payload["journal_body"] == "prose"
    assert payload["readings_available"] is False
    lifecycle_call = next(c for c in pool.conn.calls if "substrate_durable_run_state" in c[0])
    assert lifecycle_call[1] == ("r1", ["curiosity.investigate", "self_sense_eval"])

    bad = asyncio.run(store.read_run_payload(pool=pool, reader=None, run_id="r1' OR 1=1"))
    assert bad == {"available": True, "found": False, "reason": "bad_run_id"}
    missing = asyncio.run(store.read_run_payload(pool=_Pool(), reader=_Reader(), run_id="nope"))
    assert missing["found"] is False and missing["available"] is True


# --- route glue --------------------------------------------------------------


def _install_fake_main(monkeypatch, *, pool=None):
    fake_main = types.SimpleNamespace(
        app=types.SimpleNamespace(state=types.SimpleNamespace(memory_pg_pool=pool)),
        bus=None,
    )
    monkeypatch.setitem(sys.modules, "scripts.main", fake_main)
    import scripts as scripts_pkg

    monkeypatch.setattr(scripts_pkg, "main", fake_main, raising=False)
    return fake_main


def test_runs_route_never_500s_and_carries_the_three_line_schedule(monkeypatch) -> None:
    _install_fake_main(monkeypatch, pool=None)
    monkeypatch.setattr(cr, "_build_reader", lambda: None)

    async def fake_schedule():
        return {"available": True, "local_date": "2026-09-22", "tz": "America/Denver",
                "lines": {ln: {"runs_today": 1, "daily_cap": 3} for ln in ("investigate", "self_inquiry", "self_sense_eval")}}

    monkeypatch.setattr(cr, "_read_schedule", fake_schedule)
    response = asyncio.run(cr.curiosity_runs_api(days=400, line="bogus"))
    body = json.loads(response.body)
    assert response.status_code == 200
    assert response.headers["cache-control"].startswith("no-store")
    assert body["available"] is False
    assert body["window_days"] == 90 and body["line"] == "all"
    assert set(body["schedule"]["lines"]) == {"investigate", "self_inquiry", "self_sense_eval"}
    assert body["schedule"]["runs_seen_today"] == {}


def test_runs_route_swallows_a_store_exception(monkeypatch) -> None:
    _install_fake_main(monkeypatch, pool=None)

    async def boom(**kw):
        raise RuntimeError("kaboom")

    fake_store = types.SimpleNamespace(read_runs_payload=boom, clamp_days=store.clamp_days, clamp_line=store.clamp_line)
    monkeypatch.setattr(cr, "_run_store", lambda: fake_store)

    async def fake_schedule():
        return {"available": False, "lines": {}}

    monkeypatch.setattr(cr, "_read_schedule", fake_schedule)
    response = asyncio.run(cr.curiosity_runs_api())
    body = json.loads(response.body)
    assert response.status_code == 200
    assert body["available"] is False and "kaboom" in body["reason"]


def test_run_route_never_500s(monkeypatch) -> None:
    _install_fake_main(monkeypatch, pool=None)
    monkeypatch.setattr(cr, "_build_reader", lambda: None)
    response = asyncio.run(cr.curiosity_run_api("446ddd7165d5"))
    body = json.loads(response.body)
    assert response.status_code == 200
    assert body["available"] is False
    assert body["stores"] == {"postgres": "no_pool", "graph": "graph_not_configured"}


def test_the_read_endpoints_are_registered_as_gets_only() -> None:
    paths = {r.path: sorted(r.methods) for r in cr.router.routes}
    assert paths["/curiosity/api/runs"] == ["GET"]
    assert paths["/curiosity/api/run/{run_id}"] == ["GET"]
    assert "/curiosity/api/self-questions/{question_id}/park" not in paths


# --- constants this package cannot import from the loop --------------------


def test_the_self_sense_line_name_matches_the_loop() -> None:
    from orion.curiosity.run_story import LINE_SELF_SENSE_EVAL
    from scripts.curiosity_investigation import LINE_SELF_SENSE_EVAL as LOOP_LINE

    assert LINE_SELF_SENSE_EVAL == LOOP_LINE


def test_the_outreach_query_requires_the_curiosity_source_tag() -> None:
    """PR #2290 (open, not yet merged) will stamp
    `result_json.source = 'curiosity_outreach'` on every decision it records.
    The read side requires it too, now -- explicit, not merely implied by
    the correlation_id match -- so a decision row this query returns is
    self-evidently a curiosity decision, and no read-side change is needed
    when #2290 lands."""
    assert "result_json->>'source' = 'curiosity_outreach'" in store.OUTREACH_SQL


def test_the_outreach_key_matches_the_loops_derivation() -> None:
    from orion.curiosity.journal import OUTREACH_TAG
    from orion.curiosity.run_story import outreach_key

    # curiosity_investigation.py:2817 -- `uuid5(NAMESPACE_URL, f"{OUTREACH_TAG}:{run_id}")`
    assert outreach_key("abc") == str(uuid5(NAMESPACE_URL, f"{OUTREACH_TAG}:abc"))


def test_the_budget_tiles_read_every_line_the_loop_keys(monkeypatch) -> None:
    """Acceptance check 7: the three tiles match the Redis counters the loop
    itself reads -- same key constants, same local-date rule."""
    from scripts import curiosity_investigation as ci

    class _Redis:
        def __init__(self):
            self.gets: list[str] = []

        async def get(self, key):
            self.gets.append(key)
            if key.endswith("2026-09-22") or key.endswith(datetime.now(timezone.utc).astimezone(
                    __import__("zoneinfo").ZoneInfo("UTC")).date().isoformat()):
                return b"2"
            if "last" in key:
                return b"2026-09-22T10:00:00+00:00"
            return None

    redis = _Redis()
    fake_main = types.SimpleNamespace(bus=types.SimpleNamespace(redis=redis), app=None)
    monkeypatch.setitem(sys.modules, "scripts.main", fake_main)
    import scripts as scripts_pkg

    monkeypatch.setattr(scripts_pkg, "main", fake_main, raising=False)
    cfg = types.SimpleNamespace(
        HUB_CURIOSITY_INVESTIGATION_ENABLED=True, HUB_CURIOSITY_INVESTIGATION_DAILY_CAP=3,
        HUB_CURIOSITY_INVESTIGATION_MIN_COOLDOWN_SEC=14400.0,
        HUB_CURIOSITY_SELF_INQUIRY_ENABLED=True, HUB_CURIOSITY_SELF_INQUIRY_DAILY_CAP=3,
        HUB_CURIOSITY_SELF_INQUIRY_MIN_COOLDOWN_SEC=7200.0,
        HUB_CURIOSITY_SELF_SENSE_EVAL_ENABLED=True, HUB_CURIOSITY_SELF_SENSE_EVAL_DAILY_CAP=7,
        HUB_CURIOSITY_SELF_SENSE_EVAL_MIN_COOLDOWN_SEC=10800.0,
        HUB_ENDOGENOUS_OUTREACH_TZ="UTC",
    )
    fake_settings = types.SimpleNamespace(get_settings=lambda: cfg)
    monkeypatch.setitem(sys.modules, "app.settings", fake_settings)

    out = asyncio.run(cr._read_schedule())
    assert out["available"] is True
    lines = out["lines"]
    assert lines["investigate"]["daily_cap"] == 3 and lines["self_sense_eval"]["daily_cap"] == 7
    assert lines["investigate"]["runs_today"] == 2
    assert lines["investigate"]["next_eligible_at"] == "2026-09-22T14:00:00+00:00"
    assert lines["self_inquiry"]["next_eligible_at"] == "2026-09-22T12:00:00+00:00"
    keys = set(redis.gets)
    assert ci._COOLDOWN_KEY in keys and ci._SELF_COOLDOWN_KEY in keys and ci._SENSE_EVAL_COOLDOWN_KEY in keys
    assert any(k.startswith(ci._DAILY_COUNT_KEY_PREFIX) for k in keys)
    assert any(k.startswith(ci._SELF_DAILY_COUNT_KEY_PREFIX) for k in keys)
    assert any(k.startswith(ci._SENSE_EVAL_DAILY_COUNT_KEY_PREFIX) for k in keys)
    # Back-compat top-level fields describe the investigate line.
    assert out["runs_today"] == 2 and out["daily_cap"] == 3
