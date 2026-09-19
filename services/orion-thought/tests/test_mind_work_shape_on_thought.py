"""Mind work_shape attachment on ThoughtEvent.

``app.*`` imports are done inside tests / fixtures (not module scope) so a
combined pytest run after Hub tests cannot poison collection via a stale
``app`` package on ``sys.path``.
"""
from __future__ import annotations

import importlib
from uuid import uuid4

import pytest

from orion.mind.v1 import MindHandoffBriefV1, MindRunResultV1
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1


@pytest.fixture
def work_shape_from_coloring():
    from app.mind_enrichment import work_shape_from_coloring as _fn

    return _fn


class _FakeCortexClient:
    def __init__(self, exec_result: dict) -> None:
        self._exec_result = exec_result
        self.captured_context = None

    async def execute_plan(self, *, req, **_kwargs) -> dict:
        self.captured_context = req.context
        return self._exec_result


def _request(*, origin: str | None = None) -> StanceReactRequestV1:
    inputs: dict = {"user_message": "investigate claim X"}
    if origin is not None:
        inputs["utterance_origin"] = origin
    return StanceReactRequestV1(
        correlation_id="corr-work-shape-1",
        session_id="sess-1",
        user_message="investigate claim X",
        association=HubAssociationBundleV1(
            correlation_id="corr-work-shape-1",
            broadcast=None,
            broadcast_stale=False,
            read_source="felt_state_reader",
        ),
        repair_bundle=None,
        stance_inputs=inputs,
    )


def _stance_json() -> str:
    return (
        '{"imperative":"Stay present with the subject.","tone":"curious",'
        '"strain_refs":["hub:turn:corr-work-shape-1"],'
        '"evidence_refs":["hub:turn:corr-work-shape-1"],'
        '"stance_harness_slice":{"task_mode":"reflective_dialogue",'
        '"conversation_frame":"reflective","answer_strategy":"companion"}}'
    )


def test_work_shape_from_coloring_extracts_allowlisted_strings(
    work_shape_from_coloring,
) -> None:
    coloring = {
        "reflective_themes": ["continuity"],
        "expected_depth": "deep",
        "cross_cutting": "yes",
        "foresight_note": "May need peer help on synthesis.",
        "user_intent": "map the claim graph",
        "attention_frontier": [{"label": "x"}],
        "empty_depth": "",
    }
    assert work_shape_from_coloring(coloring) == {
        "expected_depth": "deep",
        "cross_cutting": "yes",
        "foresight_note": "May need peer help on synthesis.",
        "user_intent": "map the claim graph",
    }


def test_work_shape_from_coloring_none_and_empty(work_shape_from_coloring) -> None:
    assert work_shape_from_coloring(None) is None
    assert work_shape_from_coloring({"reflective_themes": ["continuity"]}) is None
    assert work_shape_from_coloring({"expected_depth": "  "}) is None


@pytest.mark.asyncio
async def test_run_stance_react_attaches_mind_work_shape(monkeypatch) -> None:
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
            snapshot_hash="hash-work-shape",
            brief=MindHandoffBriefV1(mind_quality="meaningful_synthesis"),
            mind_quality="meaningful_synthesis",
        )

    def _sel(*_a, **_k):
        return {
            "reflective_themes": ["continuity"],
            "expected_depth": "deep",
            "cross_cutting": "yes",
            "foresight_note": "May need peer help on synthesis.",
            "user_intent": "map the claim graph",
        }

    monkeypatch.setattr(bl, "run_mind_for_thought", _mind)
    monkeypatch.setattr(bl, "select_mind_coloring", _sel)

    client = _FakeCortexClient({"final_text": _stance_json(), "metadata": {}})
    thought = await bl.run_stance_react(
        _request(origin="orion"), bus=None, cortex_client=client
    )
    assert thought.mind_work_shape == {
        "expected_depth": "deep",
        "cross_cutting": "yes",
        "foresight_note": "May need peer help on synthesis.",
        "user_intent": "map the claim graph",
    }


@pytest.mark.asyncio
async def test_run_stance_react_no_coloring_leaves_mind_work_shape_none(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ORION_THOUGHT_MIND_ENRICHMENT_ENABLED", "false")
    import app.settings as s
    importlib.reload(s)
    import app.bus_listener as bl
    importlib.reload(bl)

    client = _FakeCortexClient({"final_text": _stance_json(), "metadata": {}})
    thought = await bl.run_stance_react(_request(), bus=None, cortex_client=client)
    assert thought.mind_work_shape is None
