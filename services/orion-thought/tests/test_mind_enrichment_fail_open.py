from __future__ import annotations

import importlib
from uuid import uuid4

import pytest

from orion.mind.v1 import MindHandoffBriefV1, MindRunResultV1
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1


class _FakeCortexClient:
    def __init__(self, exec_result: dict) -> None:
        self._exec_result = exec_result
        self.captured_context = None

    async def execute_plan(self, *, req, **_kwargs) -> dict:
        self.captured_context = req.context
        return self._exec_result


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="corr-1",
        session_id="sess-1",
        user_message="how are you today?",
        association=HubAssociationBundleV1(
            correlation_id="corr-1",
            broadcast=None,
            broadcast_stale=True,
            read_source="hub_sql_fallback",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "how are you today?"},
    )


def _stance_json() -> str:
    return (
        '{"imperative":"Stay present with Juniper.","tone":"warm",'
        '"strain_refs":["hub:turn:corr-1"],"evidence_refs":["hub:turn:corr-1"],'
        '"stance_harness_slice":{"task_mode":"reflective_dialogue",'
        '"conversation_frame":"reflective","answer_strategy":"companion"}}'
    )


@pytest.mark.asyncio
async def test_enrichment_disabled_is_baseline(monkeypatch):
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "false")
    import app.settings as s
    importlib.reload(s)
    import app.bus_listener as bl
    importlib.reload(bl)

    client = _FakeCortexClient({"final_text": _stance_json(), "metadata": {}})
    thought = await bl.run_stance_react(_request(), bus=None, cortex_client=client)
    assert thought.imperative == "Stay present with Juniper."
    assert "mind_coloring" not in client.captured_context


@pytest.mark.asyncio
async def test_enrichment_enabled_but_mind_fails_open(monkeypatch):
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "true")
    import app.settings as s
    importlib.reload(s)
    import app.mind_enrichment as me
    importlib.reload(me)
    import app.bus_listener as bl
    importlib.reload(bl)

    async def _boom(*_a, **_k):
        return None  # simulate Mind timeout/error → fail-open

    monkeypatch.setattr(bl, "run_mind_for_thought", _boom)

    client = _FakeCortexClient({"final_text": _stance_json(), "metadata": {}})
    thought = await bl.run_stance_react(_request(), bus=None, cortex_client=client)
    assert thought.imperative == "Stay present with Juniper."
    assert "mind_coloring" not in client.captured_context


@pytest.mark.asyncio
async def test_enrichment_enabled_meaningful_injects_coloring(monkeypatch):
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "true")
    import app.settings as s
    importlib.reload(s)
    import app.mind_enrichment as me
    importlib.reload(me)
    import app.bus_listener as bl
    importlib.reload(bl)

    async def _mind(*_a, **_k):
        return MindRunResultV1(
            mind_run_id=uuid4(),
            ok=True,
            snapshot_hash="hash-1",
            brief=MindHandoffBriefV1(mind_quality="meaningful_synthesis"),
            mind_quality="meaningful_synthesis",
        )

    monkeypatch.setattr(bl, "run_mind_for_thought", _mind)
    monkeypatch.setattr(
        bl,
        "select_mind_coloring",
        lambda *_a, **_k: {"attention_frontier": [], "reflective_themes": ["continuity"]},
    )

    client = _FakeCortexClient({"final_text": _stance_json(), "metadata": {}})
    thought = await bl.run_stance_react(_request(), bus=None, cortex_client=client)
    assert client.captured_context["mind_coloring"] == {
        "attention_frontier": [],
        "reflective_themes": ["continuity"],
    }
    assert thought.imperative == "Stay present with Juniper."


@pytest.mark.asyncio
async def test_enrichment_selector_raises_fails_open(monkeypatch):
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "true")
    import app.settings as s
    importlib.reload(s)
    import app.mind_enrichment as me
    importlib.reload(me)
    import app.bus_listener as bl
    importlib.reload(bl)

    async def _mind(*_a, **_k):
        return object()  # non-None so the selector is reached

    def _boom_selector(*_a, **_k):
        raise RuntimeError("selector blew up")

    monkeypatch.setattr(bl, "run_mind_for_thought", _mind)
    monkeypatch.setattr(bl, "select_mind_coloring", _boom_selector)

    client = _FakeCortexClient({"final_text": _stance_json(), "metadata": {}})
    thought = await bl.run_stance_react(_request(), bus=None, cortex_client=client)
    assert thought.imperative == "Stay present with Juniper."
    assert "mind_coloring" not in client.captured_context
