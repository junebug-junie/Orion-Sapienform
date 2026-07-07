"""Eval: Mind coloring is mode-agnostic and never leaks task-control into the
unified stance. Distinct from unit tests — this scores a labeled mix of
relational / technical / agent Mind results for the anti-contradiction and
aliveness guarantees the design calls load-bearing.

Run: pytest services/orion-thought/evals -q
"""
from __future__ import annotations

from uuid import uuid4

from app.bus_listener import build_stance_react_context
from app.mind_enrichment import select_mind_coloring
from orion.mind.synthesis_v1 import ActiveCognitiveFrontierV1, SelectedFrontierMatterV1
from orion.mind.v1 import MindControlDecisionV1, MindHandoffBriefV1, MindRunResultV1
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1

_TASK_CONTROL_TOKENS = (
    "task_mode", "answer_strategy", "conversation_frame", "response_priorities",
    "response_hazards", "route_kind", "mode_binding", "allowed_verbs", "mode_suggestion",
)


def _frontier(summaries: list[str]) -> ActiveCognitiveFrontierV1:
    return ActiveCognitiveFrontierV1(
        selected=[
            SelectedFrontierMatterV1(
                matter_id=f"m{i}", source_claim_id=f"c{i}", label=f"l{i}",
                summary=s, matter_kind="curiosity_affordance", score=0.8 - i * 0.1,
            )
            for i, s in enumerate(summaries)
        ]
    )


def _result(*, frame: str, task_mode: str, themes: list[str], summaries: list[str]) -> MindRunResultV1:
    payload = {
        "conversation_frame": frame,
        "task_mode": task_mode,
        "identity_salience": "medium",
        "user_intent": "x",
        "self_relevance": "matters to me",
        "juniper_relevance": "matters to Juniper",
        "reflective_themes": themes,
        "response_priorities": ["p1"],
        "response_hazards": ["h1"],
        "answer_strategy": "strat",
        "stance_summary": "s",
    }
    return MindRunResultV1(
        mind_run_id=uuid4(), ok=True, snapshot_hash="h",
        decision=MindControlDecisionV1(route_kind="chat", allowed_verbs=["speak"], mode_suggestion="brain"),
        brief=MindHandoffBriefV1(
            mind_quality="meaningful_synthesis",
            active_frontier=_frontier(summaries),
            stance_payload=payload,
        ),
        mind_quality="meaningful_synthesis",
    )


# (label, mind_result)
CASES = [
    ("relational", _result(frame="reflective", task_mode="reflective_dialogue",
                            themes=["continuity", "trust"], summaries=["what changed since we spoke"])),
    ("technical", _result(frame="technical", task_mode="technical_collaboration",
                          themes=["the deploy risk"], summaries=["the failing migration"])),
    ("agent", _result(frame="planning", task_mode="triage",
                      themes=["the blocked task"], summaries=["which tool to call next"])),
]


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="corr-1", session_id="sess-1", user_message="…",
        association=HubAssociationBundleV1(
            correlation_id="corr-1", broadcast=None, broadcast_stale=True,
            read_source="hub_sql_fallback",
        ),
        repair_bundle=None, stance_inputs={"user_message": "…"},
    )


def test_no_task_control_leaks_across_all_modes() -> None:
    import json
    for label, result in CASES:
        coloring = select_mind_coloring(result, max_items=3)
        assert coloring is not None, f"{label}: expected non-empty coloring"
        blob = json.dumps(coloring)
        for token in _TASK_CONTROL_TOKENS:
            assert token not in blob, f"{label}: task-control token {token!r} leaked"


def test_context_injection_does_not_touch_stance_authoring() -> None:
    # The coloring lands only under the mind_coloring key; the authoritative
    # stance fields are authored by the LLM downstream, never by this pipeline.
    _, result = CASES[0]
    coloring = select_mind_coloring(result, max_items=3)
    ctx = build_stance_react_context(_request(), mind_coloring=coloring)
    assert set(ctx["mind_coloring"].keys()) == set(coloring.keys())
    # No stance-authoring key is fabricated by context assembly.
    assert "stance_harness_slice" not in ctx
    assert "imperative" not in ctx


def test_relational_turn_carries_self_and_curiosity_signal() -> None:
    _, result = CASES[0]
    coloring = select_mind_coloring(result, max_items=3)
    assert coloring is not None
    assert coloring["self_relevance"]
    assert coloring["curiosity_threads"]
    assert coloring["reflective_themes"]
