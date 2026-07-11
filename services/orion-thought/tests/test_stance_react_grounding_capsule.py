from __future__ import annotations

import pytest

from app.bus_listener import run_stance_react
from orion.schemas.thought import (
    HubAssociationBundleV1,
    StanceReactRequestV1,
)


class _FakeCortexClient:
    def __init__(self, exec_result: dict) -> None:
        self._exec_result = exec_result

    async def execute_plan(self, **_kwargs) -> dict:
        return self._exec_result


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="c-1",
        session_id="s-1",
        user_message="how are you?",
        association=HubAssociationBundleV1(
            correlation_id="c-1",
            broadcast=None,
            broadcast_stale=True,
            read_source="hub_sql_fallback",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "how are you?"},
    )


def _stance_json() -> str:
    return (
        '{"imperative":"Stay present with Juniper.","tone":"warm",'
        '"strain_refs":["hub:turn:c-1"],"evidence_refs":["hub:turn:c-1"],'
        '"stance_harness_slice":{"task_mode":"reflective_dialogue",'
        '"conversation_frame":"reflective","answer_strategy":"companion"}}'
    )


@pytest.mark.asyncio
async def test_run_stance_react_attaches_grounding_capsule() -> None:
    exec_result = {
        "final_text": _stance_json(),
        "metadata": {
            "grounding_capsule": {
                "schema_version": "grounding.capsule.v1",
                "identity_summary": ["I am Oríon."],
                "relationship_summary": ["Juniper is my collaborator."],
                "response_policy_summary": ["Speak plainly."],
                "continuity_digest": "We were mid-refactor.",
                "belief_digest": "Orion values continuity.",
                "memory_digest": "We were mid-refactor.",
                "provenance": {"identity_source": "configured_yaml", "pcr_ran": True},
            }
        },
    }
    thought = await run_stance_react(
        _request(), bus=None, cortex_client=_FakeCortexClient(exec_result)
    )
    assert thought.grounding_capsule is not None
    assert thought.grounding_capsule.identity_summary == ["I am Oríon."]
    assert thought.grounding_capsule.provenance["pcr_ran"] is True


@pytest.mark.asyncio
async def test_run_stance_react_no_capsule_when_metadata_absent() -> None:
    exec_result = {"final_text": _stance_json(), "metadata": {}}
    thought = await run_stance_react(
        _request(), bus=None, cortex_client=_FakeCortexClient(exec_result)
    )
    assert thought.grounding_capsule is None


@pytest.mark.asyncio
async def test_run_stance_react_no_capsule_when_metadata_malformed() -> None:
    exec_result = {
        "final_text": _stance_json(),
        "metadata": {"grounding_capsule": {"identity_summary": "not-a-list"}},
        "request_id": "c-1",
    }
    thought = await run_stance_react(
        _request(), bus=None, cortex_client=_FakeCortexClient(exec_result)
    )
    assert thought.grounding_capsule is None


@pytest.mark.asyncio
async def test_run_stance_react_attaches_autonomy_slice() -> None:
    exec_result = {
        "final_text": _stance_json(),
        "metadata": {
            "autonomy_slice": {
                "schema_version": "autonomy.slice.v1",
                "dominant_drive": "curiosity",
                "active_tensions": ["novelty_vs_stability"],
                "pressure_trend": "rising",
                "confidence": 0.7,
            }
        },
    }
    thought = await run_stance_react(
        _request(), bus=None, cortex_client=_FakeCortexClient(exec_result)
    )
    assert thought.autonomy_slice is not None
    assert thought.autonomy_slice.dominant_drive == "curiosity"
    assert thought.autonomy_slice.active_tensions == ["novelty_vs_stability"]
    assert thought.autonomy_slice.pressure_trend == "rising"


@pytest.mark.asyncio
async def test_run_stance_react_no_autonomy_slice_when_metadata_absent() -> None:
    exec_result = {"final_text": _stance_json(), "metadata": {}}
    thought = await run_stance_react(
        _request(), bus=None, cortex_client=_FakeCortexClient(exec_result)
    )
    assert thought.autonomy_slice is None


@pytest.mark.asyncio
async def test_run_stance_react_no_autonomy_slice_when_metadata_malformed() -> None:
    exec_result = {
        "final_text": _stance_json(),
        "metadata": {"autonomy_slice": {"active_tensions": "not-a-list"}},
        "request_id": "c-1",
    }
    thought = await run_stance_react(
        _request(), bus=None, cortex_client=_FakeCortexClient(exec_result)
    )
    assert thought.autonomy_slice is None


@pytest.mark.asyncio
async def test_run_stance_react_attaches_both_capsule_and_autonomy_slice() -> None:
    exec_result = {
        "final_text": _stance_json(),
        "metadata": {
            "grounding_capsule": {
                "schema_version": "grounding.capsule.v1",
                "identity_summary": ["I am Oríon."],
                "relationship_summary": ["Juniper is my collaborator."],
                "response_policy_summary": ["Speak plainly."],
                "continuity_digest": "We were mid-refactor.",
                "belief_digest": "Orion values continuity.",
                "memory_digest": "We were mid-refactor.",
                "provenance": {"identity_source": "configured_yaml", "pcr_ran": True},
            },
            "autonomy_slice": {
                "schema_version": "autonomy.slice.v1",
                "dominant_drive": "curiosity",
                "active_tensions": ["novelty_vs_stability"],
                "pressure_trend": "rising",
                "confidence": 0.7,
            },
        },
    }
    thought = await run_stance_react(
        _request(), bus=None, cortex_client=_FakeCortexClient(exec_result)
    )
    assert thought.grounding_capsule is not None
    assert thought.grounding_capsule.identity_summary == ["I am Oríon."]
    assert thought.autonomy_slice is not None
    assert thought.autonomy_slice.dominant_drive == "curiosity"
