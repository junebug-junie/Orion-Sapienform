"""Tests for the drive_state_compact facet on orion-thought's independent
"light Mind" path (mirrors orion-cortex-orch's mind_runtime facet, adapted to
orion-thought's sync psycopg2/SQLAlchemy DB seam)."""
from __future__ import annotations

import importlib
import time
from types import SimpleNamespace

import pytest

from app.mind_enrichment import build_light_mind_request
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1


def _fresh_mind_enrichment():
    """Reimport app.mind_enrichment through the module object, following the
    conftest isolation fixture's reset. Needed so monkeypatch.setattr on the
    module used by the test targets the same module object the awaited
    function actually resolves its globals through (see
    test_mind_enrichment_fail_open.py for the same reload pattern)."""
    import app.mind_enrichment as me

    importlib.reload(me)
    return me


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="corr-1",
        session_id="sess-1",
        user_message="how's the drive state looking?",
        association=HubAssociationBundleV1(
            correlation_id="corr-1",
            broadcast=None,
            broadcast_stale=True,
            read_source="hub_sql_fallback",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "how's the drive state looking?"},
    )


def _settings(timeout_sec: float = 0.4) -> SimpleNamespace:
    return SimpleNamespace(mind_drive_state_fetch_timeout_sec=timeout_sec)


# --- build_light_mind_request: drive_state_compact facet wiring ---


def test_drive_state_compact_included_when_present() -> None:
    compact = {"dominant_drive": "curiosity", "active_drives": ["curiosity"], "summary": "s"}
    req = build_light_mind_request(
        _request(), wall_time_ms=12000, router_profile="default", drive_state_compact=compact
    )
    assert req.snapshot_inputs["facets"]["drive_state_compact"] == compact


def test_drive_state_compact_omitted_when_none() -> None:
    req = build_light_mind_request(
        _request(), wall_time_ms=12000, router_profile="default", drive_state_compact=None
    )
    assert "drive_state_compact" not in req.snapshot_inputs.get("facets", {})


def test_drive_state_compact_omitted_when_empty_dict() -> None:
    req = build_light_mind_request(
        _request(), wall_time_ms=12000, router_profile="default", drive_state_compact={}
    )
    assert "drive_state_compact" not in req.snapshot_inputs.get("facets", {})


def test_drive_state_compact_default_param_is_none() -> None:
    req = build_light_mind_request(_request(), wall_time_ms=12000, router_profile="default")
    assert "drive_state_compact" not in req.snapshot_inputs.get("facets", {})


# --- fetch_drive_state_facet_for_thought: bounded, fail-open fetch ---


@pytest.mark.asyncio
async def test_fetch_degrades_on_timeout(monkeypatch) -> None:
    me = _fresh_mind_enrichment()

    def _slow_query():
        time.sleep(0.2)
        return {"dominant_drive": "curiosity", "active_drives": [], "drive_pressures": {}, "summary": "s", "observed_at": None}

    monkeypatch.setattr(me, "_query_latest_drive_audit_row_sync", _slow_query)

    compact, diagnostics = await me.fetch_drive_state_facet_for_thought(
        "corr-1", settings=_settings(timeout_sec=0.01)
    )
    assert compact is None
    assert diagnostics["timed_out"] is True
    assert diagnostics["ok"] is False
    assert diagnostics["reason"] == "timeout"
    assert diagnostics["correlation_id"] == "corr-1"


@pytest.mark.asyncio
async def test_fetch_degrades_on_exception(monkeypatch) -> None:
    me = _fresh_mind_enrichment()

    def _boom():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(me, "_query_latest_drive_audit_row_sync", _boom)

    compact, diagnostics = await me.fetch_drive_state_facet_for_thought(
        "corr-1", settings=_settings()
    )
    assert compact is None
    assert diagnostics["ok"] is False
    assert diagnostics["reason"] == "exception"
    assert diagnostics["exception_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_fetch_never_raises_on_exception(monkeypatch) -> None:
    """Never raise to the caller — a query failure must not break the turn."""
    me = _fresh_mind_enrichment()

    def _boom():
        raise RuntimeError("table missing")

    monkeypatch.setattr(me, "_query_latest_drive_audit_row_sync", _boom)

    # Must not raise.
    compact, _diag = await me.fetch_drive_state_facet_for_thought("corr-1", settings=_settings())
    assert compact is None


@pytest.mark.asyncio
async def test_fetch_returns_none_on_no_rows(monkeypatch) -> None:
    me = _fresh_mind_enrichment()

    monkeypatch.setattr(me, "_query_latest_drive_audit_row_sync", lambda: None)

    compact, diagnostics = await me.fetch_drive_state_facet_for_thought(
        "corr-1", settings=_settings()
    )
    assert compact is None
    assert diagnostics["ok"] is True
    assert diagnostics["reason"] == "no_rows"


@pytest.mark.asyncio
async def test_fetch_returns_none_on_quiet_tick_no_meaningful_content(monkeypatch) -> None:
    me = _fresh_mind_enrichment()

    monkeypatch.setattr(
        me,
        "_query_latest_drive_audit_row_sync",
        lambda: {
            "dominant_drive": None,
            "active_drives": [],
            "drive_pressures": {},
            "summary": None,
            "observed_at": None,
        },
    )

    compact, diagnostics = await me.fetch_drive_state_facet_for_thought(
        "corr-1", settings=_settings()
    )
    assert compact is None
    assert diagnostics["ok"] is True
    assert diagnostics["reason"] == "no_meaningful_content"


@pytest.mark.asyncio
async def test_fetch_returns_compact_on_success(monkeypatch) -> None:
    me = _fresh_mind_enrichment()

    monkeypatch.setattr(
        me,
        "_query_latest_drive_audit_row_sync",
        lambda: {
            "dominant_drive": "curiosity",
            "active_drives": ["curiosity", "connection"],
            "drive_pressures": {"curiosity": 0.8},
            "summary": "Curiosity is dominant.",
            "observed_at": None,
        },
    )

    compact, diagnostics = await me.fetch_drive_state_facet_for_thought(
        "corr-1", settings=_settings()
    )
    assert compact == {
        "dominant_drive": "curiosity",
        "active_drives": ["curiosity", "connection"],
        "drive_pressures": {"curiosity": 0.8},
        "summary": "Curiosity is dominant.",
        "observed_at": None,
    }
    assert diagnostics["ok"] is True
    assert diagnostics["reason"] == "success"


@pytest.mark.asyncio
async def test_fetch_coerces_jsonb_string_columns(monkeypatch) -> None:
    """Guard path: if the driver ever returns JSONB as a raw string, it's decoded."""
    me = _fresh_mind_enrichment()

    monkeypatch.setattr(
        me,
        "_query_latest_drive_audit_row_sync",
        lambda: {
            "dominant_drive": "curiosity",
            "active_drives": '["curiosity"]',
            "drive_pressures": '{"curiosity": 0.5}',
            "summary": "s",
            "observed_at": None,
        },
    )

    compact, _diag = await me.fetch_drive_state_facet_for_thought("corr-1", settings=_settings())
    assert compact["active_drives"] == ["curiosity"]
    assert compact["drive_pressures"] == {"curiosity": 0.5}


# --- bus_listener wiring: fetch result threads into the Mind request ---


class _FakeCortexClient:
    def __init__(self, exec_result: dict) -> None:
        self._exec_result = exec_result
        self.captured_context = None

    async def execute_plan(self, *, req, **_kwargs) -> dict:
        self.captured_context = req.context
        return self._exec_result


def _stance_json() -> str:
    return (
        '{"imperative":"Stay present with Juniper.","tone":"warm",'
        '"strain_refs":["hub:turn:corr-1"],"evidence_refs":["hub:turn:corr-1"],'
        '"stance_harness_slice":{"task_mode":"reflective_dialogue",'
        '"conversation_frame":"reflective","answer_strategy":"companion"}}'
    )


@pytest.mark.asyncio
async def test_bus_listener_threads_drive_state_into_mind_request(monkeypatch) -> None:
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "true")
    import app.settings as s

    importlib.reload(s)
    import app.mind_enrichment as me

    importlib.reload(me)
    import app.bus_listener as bl

    importlib.reload(bl)

    captured: dict = {}

    async def _fake_fetch(correlation_id, *, settings):
        return {"dominant_drive": "curiosity", "summary": "s"}, {
            "ok": True,
            "reason": "success",
            "elapsed_ms": 3,
            "timed_out": False,
        }

    async def _fake_run_mind(mind_req, **_kwargs):
        captured["mind_req"] = mind_req
        return None  # fail-open past this point; only the request shape matters here

    monkeypatch.setattr(bl, "fetch_drive_state_facet_for_thought", _fake_fetch)
    monkeypatch.setattr(bl, "run_mind_for_thought", _fake_run_mind)

    client = _FakeCortexClient({"final_text": _stance_json(), "metadata": {}})
    await bl.run_stance_react(_request(), bus=None, cortex_client=client)

    facets = captured["mind_req"].snapshot_inputs.get("facets", {})
    assert facets["drive_state_compact"] == {"dominant_drive": "curiosity", "summary": "s"}


@pytest.mark.asyncio
async def test_bus_listener_omits_drive_state_when_fetch_fails_open(monkeypatch) -> None:
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "true")
    import app.settings as s

    importlib.reload(s)
    import app.mind_enrichment as me

    importlib.reload(me)
    import app.bus_listener as bl

    importlib.reload(bl)

    captured: dict = {}

    async def _fake_fetch(correlation_id, *, settings):
        return None, {"ok": False, "reason": "timeout", "elapsed_ms": 400, "timed_out": True}

    async def _fake_run_mind(mind_req, **_kwargs):
        captured["mind_req"] = mind_req
        return None

    monkeypatch.setattr(bl, "fetch_drive_state_facet_for_thought", _fake_fetch)
    monkeypatch.setattr(bl, "run_mind_for_thought", _fake_run_mind)

    client = _FakeCortexClient({"final_text": _stance_json(), "metadata": {}})
    thought = await bl.run_stance_react(_request(), bus=None, cortex_client=client)

    facets = captured["mind_req"].snapshot_inputs.get("facets", {})
    assert "drive_state_compact" not in facets
    assert thought.imperative == "Stay present with Juniper."
