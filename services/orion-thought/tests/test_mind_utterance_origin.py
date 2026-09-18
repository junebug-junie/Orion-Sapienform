from __future__ import annotations

import importlib
from uuid import uuid4

import pytest

from orion.mind.v1 import MindHandoffBriefV1, MindRunResultV1
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1
from app.mind_enrichment import build_light_mind_request


def _stance(user_message: str, *, origin: str | None) -> StanceReactRequestV1:
    inputs: dict = {"user_message": user_message}
    if origin is not None:
        inputs["utterance_origin"] = origin
    # Mirror services/orion-thought/tests/test_mind_light_snapshot.py::_request
    return StanceReactRequestV1(
        correlation_id="corr-origin-1",
        session_id="sess-1",
        user_message=user_message,
        association=HubAssociationBundleV1(
            correlation_id="corr-origin-1",
            broadcast=None,
            broadcast_stale=False,
            read_source="felt_state_reader",
        ),
        repair_bundle=None,
        stance_inputs=inputs,
    )


def test_build_light_mind_request_sets_utterance_origin_field() -> None:
    req = build_light_mind_request(
        _stance("hello from juniper", origin="juniper"),
        wall_time_ms=1000,
        router_profile="default",
    )
    assert req.utterance_origin == "juniper"


def test_situation_compact_includes_origin_prose_for_orion() -> None:
    req = build_light_mind_request(
        _stance("investigate claim X", origin="orion"),
        wall_time_ms=1000,
        router_profile="default",
    )
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    situation = facets.get("situation_compact") or {}
    note = str(situation.get("utterance_origin_note") or "")
    assert "orion" in note.lower()
    assert req.utterance_origin == "orion"


def test_missing_origin_stays_none_and_omits_origin_note() -> None:
    req = build_light_mind_request(
        _stance("legacy caller", origin=None),
        wall_time_ms=1000,
        router_profile="default",
    )
    assert req.utterance_origin is None
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    situation = facets.get("situation_compact") or {}
    assert "utterance_origin_note" not in situation


@pytest.mark.asyncio
async def test_coloring_selector_receives_utterance_origin_from_mind_request(monkeypatch) -> None:
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "true")
    import app.settings as s
    importlib.reload(s)
    import app.mind_enrichment as me
    importlib.reload(me)
    import app.bus_listener as bl
    importlib.reload(bl)

    captured: dict = {}

    async def _mind(*_a, **_k):
        return MindRunResultV1(
            mind_run_id=uuid4(),
            ok=True,
            snapshot_hash="hash-origin",
            brief=MindHandoffBriefV1(mind_quality="meaningful_synthesis"),
            mind_quality="meaningful_synthesis",
        )

    def _sel(result, *, max_items, utterance_origin=None):
        captured["utterance_origin"] = utterance_origin
        return {"reflective_themes": ["continuity"]}

    monkeypatch.setattr(bl, "run_mind_for_thought", _mind)
    monkeypatch.setattr(bl, "select_mind_coloring", _sel)

    class _FakeCortexClient:
        def __init__(self) -> None:
            self.captured_context = None

        async def execute_plan(self, *, req, **_kwargs) -> dict:
            self.captured_context = req.context
            return {
                "final_text": (
                    '{"imperative":"Stay present.","tone":"warm",'
                    '"strain_refs":["hub:turn:corr-origin-1"],'
                    '"evidence_refs":["hub:turn:corr-origin-1"]}'
                ),
                "metadata": {},
            }

    client = _FakeCortexClient()
    await bl.run_stance_react(
        _stance("investigate claim X", origin="orion"),
        bus=None,
        cortex_client=client,
    )
    assert captured.get("utterance_origin") == "orion"
