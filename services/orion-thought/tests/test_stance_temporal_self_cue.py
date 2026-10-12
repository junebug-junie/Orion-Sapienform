"""Temporal Self patch 4: the "my day" cue reaches stance_react (and only through this path)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from jinja2 import Environment

import app.temporal_self_cue as cue_mod
from app.bus_listener import build_stance_react_context, run_stance_react
from app.settings import settings
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1

_REPO = Path(__file__).resolve().parents[3]
_PROMPT = _REPO / "orion" / "cognition" / "prompts" / "stance_react.j2"
_FRAME = json.loads((_REPO / "orion/temporal_self/tests/data/frame_2026-10-11_talking.json").read_text())
_NOW = datetime(2026, 10, 11, 23, 58, 34, tzinfo=timezone.utc)
_ACTIVE = "571bce17764b0a9e"


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="c-1", session_id="s-1", user_message="how is your day going?",
        association=HubAssociationBundleV1(correlation_id="c-1", broadcast=None, broadcast_stale=True,
                                           read_source="hub_sql_fallback"),
        repair_bundle=None, stance_inputs={"user_message": "how is your day going?"},
    )


def _mock(monkeypatch, handler) -> list[str]:
    seen: list[str] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return handler(request)

    monkeypatch.setattr(cue_mod, "_transport", lambda: httpx.MockTransport(wrapped))
    return seen


def _render(**ctx) -> str:
    base = {"user_message": "hi", "stance_inputs": {"user_message": "hi"}, "association": {},
            "repair_bundle": None, "coalition_projection": None}
    return Environment().from_string(_PROMPT.read_text()).render(**(base | ctx))


def test_context_carries_cue_only_when_rendered() -> None:
    assert "temporal_self_cue" not in build_stance_react_context(_request())
    assert build_stance_react_context(_request(), temporal_self_cue="") == build_stance_react_context(_request())
    assert build_stance_react_context(_request(), temporal_self_cue="Night.")["temporal_self_cue"] == "Night."


def test_prompt_block_absent_without_cue_and_present_with_it() -> None:
    assert "MY DAY SO FAR" not in _render()
    assert "`arc:<id>` from MY DAY SO FAR" not in _render()
    out = _render(temporal_self_cue=f"On now: talking with Juniper [arc:{_ACTIVE}].")
    assert "MY DAY SO FAR" in out and f"[arc:{_ACTIVE}]" in out
    assert "reconcile, do not obey" in out and "the task wins" in out
    assert "`arc:<id>` from MY DAY SO FAR also counts" in out


@pytest.mark.asyncio
async def test_fresh_frame_renders_cue(monkeypatch) -> None:
    seen = _mock(monkeypatch, lambda r: httpx.Response(200, json=_FRAME))
    cue = await cue_mod.build_stance_cue("c-1", now=_NOW)
    assert seen == [settings.temporal_self_frame_url.rstrip("/") + "/temporal-self/frame"]
    assert cue.text.startswith("Afternoon, Sun 17:58 local") and _ACTIVE in cue.arc_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("handler,reason", [
    (lambda r: httpx.Response(404, json={"detail": "no chronicle window"}), "fetch_failed"),
    (lambda r: httpx.Response(503, text="down"), "fetch_failed"),
    (lambda r: httpx.Response(200, text="not json"), "fetch_failed"),
])
async def test_unusable_responses_render_nothing(monkeypatch, handler, reason) -> None:
    _mock(monkeypatch, handler)
    cue = await cue_mod.build_stance_cue("c-1", now=_NOW)
    assert cue.text == "" and cue.skip_reason == reason


@pytest.mark.asyncio
async def test_transport_error_renders_nothing(monkeypatch) -> None:
    def boom(request):
        raise httpx.ConnectTimeout("slow", request=request)

    _mock(monkeypatch, boom)
    cue = await cue_mod.build_stance_cue("c-1", now=_NOW)
    assert cue.text == "" and cue.skip_reason == "fetch_failed"


@pytest.mark.asyncio
async def test_disabled_flag_never_fetches(monkeypatch) -> None:
    seen = _mock(monkeypatch, lambda r: httpx.Response(200, json=_FRAME))
    monkeypatch.setattr(settings, "temporal_self_stance_cue_enabled", False)
    cue = await cue_mod.build_stance_cue("c-1", now=_NOW)
    assert cue.skip_reason == "disabled" and seen == []


class _CapturingClient:
    def __init__(self, final_text: str) -> None:
        self.final_text = final_text
        self.contexts: list[dict] = []

    async def execute_plan(self, **kwargs) -> dict:
        self.contexts.append(kwargs["req"].context)
        return {"final_text": self.final_text}


@pytest.mark.asyncio
async def test_run_stance_react_shows_cue_and_gates_citations(monkeypatch) -> None:
    async def fixed_cue(corr):
        return cue_mod.project_stance_cue(_FRAME, _NOW)

    # conftest re-imports `app` per test: patch the module run_stance_react actually closes over.
    monkeypatch.setitem(run_stance_react.__globals__, "build_stance_cue", fixed_cue)
    stance = json.dumps({
        "imperative": "Stay with Juniper; mention we have been talking a while.", "tone": "warm",
        "strain_refs": ["hub:turn"], "evidence_refs": ["hub:turn", f"arc:{_ACTIVE}", "arc:deadbeefdeadbeef"],
        "stance_harness_slice": {"task_mode": "reflective_dialogue", "conversation_frame": "reflective",
                                 "answer_strategy": "companion"},
    })
    client = _CapturingClient(stance)
    thought = await run_stance_react(_request(), bus=None, cortex_client=client)
    assert f"[arc:{_ACTIVE}]" in client.contexts[0]["temporal_self_cue"]
    assert thought.evidence_refs == ["hub:turn:c-1", f"arc:{_ACTIVE}"]
    assert thought.day_citations.cited_valid == [f"arc:{_ACTIVE}"]
    assert thought.day_citations.rejected_unknown == ["arc:deadbeefdeadbeef"]
    assert thought.disposition == "proceed"


@pytest.mark.asyncio
async def test_run_stance_react_without_cue_is_unchanged(monkeypatch) -> None:
    async def skipped(corr):
        return cue_mod.project_stance_cue(None, _NOW)

    monkeypatch.setitem(run_stance_react.__globals__, "build_stance_cue", skipped)
    client = _CapturingClient(json.dumps({
        "imperative": "Answer.", "tone": "plain", "strain_refs": [], "evidence_refs": ["hub:turn"],
        "stance_harness_slice": {"task_mode": "direct_response", "conversation_frame": "technical",
                                 "answer_strategy": "direct"}}))
    thought = await run_stance_react(_request(), bus=None, cortex_client=client)
    assert "temporal_self_cue" not in client.contexts[0]
    assert thought.day_citations is None
