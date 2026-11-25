"""Inner Sky feed (scripts/inner_sky_routes.py): rows in, the page's lights out, nothing invented."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(HUB_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts import inner_sky_routes as isr  # noqa: E402
from scripts import inner_sky_verdicts as isv  # noqa: E402

ORGAN_OF = {"vision_percept": "camera", "gpu_wait": "body", "curiosity_run": "curiosity", "prime_thought": "prime"}
T = datetime(2026, 11, 25, 1, 0, tzinfo=timezone.utc)


def test_a_blank_label_gets_the_same_words_orion_is_shown():
    row = {"event_id": "vision_percept:1", "occurred_at": T, "source_kind": "vision_percept", "label": None,
           "verdict": None, "payload_json": {"entities": ["chair", "desk", "door"]}, "salience": None}
    out = isr.shape_event(row, ORGAN_OF)
    assert out == {"id": "vision_percept:1", "t": T.timestamp(), "organ": "camera", "kind": "vision_percept",
                   "label": "saw chair, desk, door", "salience": None, "verdict": None, "severity": None}


def test_payload_as_json_text_and_unknown_kind_are_handled():
    row = {"event_id": "x:1", "occurred_at": T, "source_kind": "mystery_kind", "label": "", "verdict": None,
           "payload_json": '{"a": 1}', "salience": 0.4}
    out = isr.shape_event(row, ORGAN_OF)
    assert out["organ"] == isr.UNREGISTERED
    assert out["label"] == "mystery kind"
    assert out["salience"] == 0.4


def test_a_cite_names_both_organs_and_falls_back_to_the_id_prefix_when_the_event_left_the_table():
    joined = {"event_id": "curiosity_run:abc", "citing_organ": "prime", "cited_at": T, "source_kind": "curiosity_run",
              "label": None, "verdict": None, "payload_json": {"n_tested": 1, "n_moved": 0, "n_formed": 2}}
    out = isr.shape_cite(joined, ORGAN_OF)
    assert (out["from"], out["by"], out["kind"]) == ("curiosity", "prime", "curiosity_run")
    assert out["label"].startswith("curiosity run: tested 1")
    orphan = {"event_id": "gpu_wait:zz", "citing_organ": "curiosity", "cited_at": T, "source_kind": None}
    out = isr.shape_cite(orphan, ORGAN_OF)
    assert (out["from"], out["kind"], out["label"]) == ("body", "gpu_wait", "")


def _client(monkeypatch, rows, cites) -> TestClient:
    calls: dict[str, Any] = {}

    def fake_rows(since, limit):
        calls["since"], calls["limit"] = since, limit
        return rows

    def fake_cites(since, limit):
        calls["cite_since"] = since
        return cites

    monkeypatch.setattr(isr, "_fetch_event_rows", fake_rows)
    monkeypatch.setattr(isr, "_fetch_cite_rows", fake_cites)
    app = FastAPI()
    app.include_router(isr.router)
    c = TestClient(app)
    c.calls = calls  # type: ignore[attr-defined]
    return c


def test_events_cursor_moves_on_store_time_and_stays_put_when_nothing_new(monkeypatch):
    stored = datetime(2026, 11, 25, 2, 0, tzinfo=timezone.utc)
    row = {"event_id": "gpu_wait:1", "occurred_at": T, "source_kind": "gpu_wait", "label": None, "verdict": None,
           "payload_json": {"holder": "durable-runs:x", "waited_ms": 900}, "salience": None, "ingested_at": stored}
    c = _client(monkeypatch, [row], [])
    body = c.get("/api/inner-sky/events", params={"since": 5, "cite_since": 7}).json()
    assert c.calls == {"since": 5.0, "limit": isr.EVENT_PAGE, "cite_since": 7.0}
    assert body["cursor"] == stored.timestamp()
    assert body["cite_cursor"] == 7.0
    assert body["more"] is False
    assert body["events"][0]["label"] == "durable-runs:x waited 900 ms for a GPU"
    assert "prime" in body["organs"]

    empty = _client(monkeypatch, [], []).get("/api/inner-sky/events", params={"since": 42}).json()
    assert (empty["cursor"], empty["events"]) == (42.0, [])


class FakeGraph:
    def __init__(self):
        self.params = None

    def graph_query(self, q, params=None):
        if "PUBLISHES" in q:
            return [{"id": "cortex-exec", "n": 10, "last": 100.0}, {"id": "lonely", "n": None, "last": None}, {"id": None}]
        self.params = params
        return [{"a": "cortex-exec", "b": "hub", "n": 5, "last": 99.0, "z": 9.0},  # too little history to trust
                {"a": "hub", "b": "hub", "n": 1, "last": 99.0, "z": None},
                {"a": "bus", "b": "hub", "n": 50, "last": 99.0, "z": 3.5},
                {"a": "bus", "b": "exec", "n": 50, "last": 99.0, "z": -8.0},  # faster than usual: not trouble
                {"a": "old", "b": "hub", "n": 50, "last": -500.0, "z": 7.0}]  # slow, but its last hop was long ago


def test_body_keeps_never_published_services_and_drops_self_links():
    g = FakeGraph()
    out = isr.read_body(g, now=600.0)
    assert out["services"] == [{"id": "cortex-exec", "n": 10, "last": 100.0}, {"id": "lonely", "n": 0, "last": None}]
    assert out["links"] == [
        {"a": "cortex-exec", "b": "hub", "n": 5, "last": 99.0, "slow": False, "z": 9.0},
        {"a": "bus", "b": "hub", "n": 50, "last": 99.0, "slow": True, "z": 3.5},
        {"a": "bus", "b": "exec", "n": 50, "last": 99.0, "slow": False, "z": -8.0},
        {"a": "old", "b": "hub", "n": 50, "last": -500.0, "slow": False, "z": 7.0},
    ]
    assert g.params == {"cut": 600.0 - isr.LINK_WINDOW_SEC}


def test_focus_reads_the_open_attention_arc_verbatim():
    frame = {"day_phase": "dusk", "as_of": "x", "active_by_kind": {"attention": {
        "subject_label": "Prediction body prediction error", "subject_ref": "node:prediction.body", "began_at": "b"}}}
    assert isr.focus_from_frame(frame) == {"attention": "Prediction body prediction error", "attention_since": "b",
                                           "day_phase": "dusk", "as_of": "x"}
    assert isr.focus_from_frame({})["attention"] is None
    other_kind_only = {"active_arc": {"kind": "curiosity", "subject_label": "a curiosity run"},
                       "active_by_kind": {"curiosity": {"subject_label": "a curiosity run"}}}
    assert isr.focus_from_frame(other_kind_only)["attention"] is None


def test_state_reports_each_failing_part_instead_of_a_blank_shell(monkeypatch):
    def boom():
        raise RuntimeError("falkordb_uri_not_configured")

    monkeypatch.setattr(isr, "_graph_client", boom)
    monkeypatch.setattr(isr.settings, "HUB_CURIOSITY_DURABLE_RUNS_URL", "", raising=False)
    app = FastAPI()
    app.include_router(isr.router)
    body = TestClient(app).get("/api/inner-sky/state").json()
    assert body["body"] is None and body["body_error"] == "falkordb_uri_not_configured"
    assert body["focus"] is None and body["focus_error"] == "durable_runs_url_not_configured"


def test_the_page_never_reads_through_the_now_endpoint():
    """GET /temporal-self/now stores an experience_shown row; a debug page must not register as Orion seeing."""
    src = (HUB_ROOT / "scripts" / "inner_sky_routes.py").read_text()
    js = (HUB_ROOT / "static" / "js" / "inner-sky.js").read_text()
    code = "\n".join(line for line in src.splitlines() if "temporal-self/now" in line and not line.lstrip().startswith(("#", "*")))
    assert "httpx.get" not in code
    assert "temporal-self/now" not in js


def test_feed_eval_fails_an_empty_shell_and_passes_real_rows():
    sys.path.insert(0, str(HUB_ROOT / "evals"))
    from run_inner_sky_feed_eval import evaluate

    state = {"now": 1000.0, "body": {"services": [{"id": "a", "last": 990.0}, {"id": "b", "last": None}], "links": []},
             "focus": {"attention": None}, "body_error": None, "focus_error": None}
    good = {"organs": ["camera", "prime"], "events": [{"organ": "camera", "kind": "vision_percept", "label": "saw door"}], "cites": []}
    report = evaluate(good, state)
    assert report["verdict"] == "PASS" and report["services_live_last_hour"] == 1
    assert report["organs_with_no_experience"] == ["prime"]
    shell = {"organs": ["camera"], "events": [{"organ": "unregistered", "kind": "x", "label": ""}], "cites": []}
    report = evaluate(shell, dict(state, body_error="down"))
    assert report["verdict"] == "FAIL" and len(report["failures"]) == 3


class _FakeConn:
    def __init__(self, sink):
        self.sink = sink

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params):
        self.sink.append((str(sql), dict(params)))
        return []


class _FakeEngine:
    def __init__(self):
        self.calls: list = []

    def connect(self):
        return _FakeConn(self.calls)


def test_cursors_reread_an_overlap_so_late_committing_transactions_are_not_lost(monkeypatch):
    eng = _FakeEngine()
    monkeypatch.setattr(isr, "_pg_engine", lambda: eng)
    isr._fetch_event_rows(1000.0, 10)
    isr._fetch_cite_rows(2000.0, 10)
    isr._fetch_event_rows(0.0, 10)  # first read: no overlap needed, the whole window comes back
    (sql_e, p_e), (sql_c, p_c), (_, p_first) = eng.calls
    assert "- make_interval(secs => :overlap)" in sql_e and "- make_interval(secs => :overlap)" in sql_c
    assert p_e["overlap"] == p_c["overlap"] == isr.CURSOR_OVERLAP_SEC
    assert p_first["overlap"] == 0
    assert "u.use_id" in sql_c


def test_cites_carry_a_stable_id_for_dedupe_and_the_cursor_never_moves_back(monkeypatch):
    cite = {"use_id": "u-1", "event_id": "gpu_wait:1", "citing_organ": "curiosity", "cited_at": T, "source_kind": None}
    assert isr.shape_cite(cite, ORGAN_OF)["id"] == "u-1"
    # an overlap read can return only rows older than the cursor: the cursor must hold
    old = {"event_id": "gpu_wait:1", "occurred_at": T, "source_kind": "gpu_wait", "label": "x", "verdict": None,
           "payload_json": {}, "salience": None, "ingested_at": T}
    c = _client(monkeypatch, [old], [dict(cite, cited_at=T)])
    body = c.get("/api/inner-sky/events", params={"since": T.timestamp() + 30, "cite_since": T.timestamp() + 30}).json()
    assert body["cursor"] == T.timestamp() + 30 and body["cite_cursor"] == T.timestamp() + 30


@pytest.mark.parametrize("verdict,severity", [
    ("critical", "red"), ("broken", "red"), ("degraded", "amber"), ("claim_upheld=false", "amber"),
    ("refused", "amber"), ("no_read_evidence:http_error", "amber"),
    ("no producer close within horizon + 120s grace", "amber"), ("no_read_evidence", "amber"),
    ("late:717s stage1_completed", "amber"), ("not_produced", "amber"),
    ("confirmed", None), ("disconfirmed", None), ("unscored", None), ("no_lasting_trace", None),
    # a guard that did not fire, a gate doing its job, a choice not to act: never trouble
    ("not_crossed", None), ("claim_upheld=true", None), ("granted", None), ("decayed_unattended", None), (None, None),
    ("produced", None), ("chose", None), ("declined_all", None),
])
def test_severity_comes_only_from_the_rows_own_verdict(verdict, severity):
    assert isv.severity_of(verdict) == severity
    assert isv.verdict_is_classified(verdict)


def test_an_unknown_verdict_is_neutral_and_reported_not_guessed():
    assert isv.severity_of("total_meltdown") is None
    assert not isv.verdict_is_classified("total_meltdown")
    sys.path.insert(0, str(HUB_ROOT / "evals"))
    from run_inner_sky_feed_eval import evaluate

    state = {"now": 1.0, "body": {"services": [], "links": [{"slow": True}]}, "focus": {}, "body_error": None, "focus_error": None}
    ev = {"organs": ["metacognition"], "cites": [], "events": [
        {"organ": "metacognition", "kind": "metacog_observation", "label": "x", "verdict": "critical", "severity": "red"},
        {"organ": "metacognition", "kind": "metacog_observation", "label": "y", "verdict": "total_meltdown", "severity": None}]}
    report = evaluate(ev, state)
    assert report["trouble"] == {"red": 1}
    assert report["unsorted_verdicts"] == {"total_meltdown": 1}
    assert report["slow_links"] == 1 and report["verdict"] == "PASS"
