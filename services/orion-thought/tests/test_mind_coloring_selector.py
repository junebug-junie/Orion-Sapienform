from __future__ import annotations

from uuid import uuid4

from app.mind_enrichment import MIND_COLORING_ALLOWED_KEYS, select_mind_coloring
from orion.mind.synthesis_v1 import ActiveCognitiveFrontierV1, SelectedFrontierMatterV1
from orion.mind.v1 import MindControlDecisionV1, MindHandoffBriefV1, MindRunResultV1


def _selected(label: str, summary: str, score: float) -> SelectedFrontierMatterV1:
    return SelectedFrontierMatterV1(
        matter_id=f"m-{label}",
        source_claim_id=f"c-{label}",
        label=label,
        summary=summary,
        matter_kind="curiosity_affordance",
        score=score,
    )


def _stance_payload() -> dict:
    # A full ChatStanceBrief dump — carries BOTH self-relational and task-control fields.
    return {
        "conversation_frame": "reflective",
        "task_mode": "reflective_dialogue",
        "identity_salience": "high",
        "user_intent": "connect",
        "self_relevance": "This touches my continuity with Juniper.",
        "juniper_relevance": "Juniper is checking in on me.",
        "reflective_themes": ["continuity", "trust", "the shape of our work"],
        "response_priorities": ["companion_presence"],
        "response_hazards": ["avoid_task_tracking"],
        "answer_strategy": "companion",
        "stance_summary": "warm, present",
    }


def _result(*, ok: bool, quality: str, with_frontier: bool = True) -> MindRunResultV1:
    frontier = None
    if with_frontier:
        frontier = ActiveCognitiveFrontierV1(
            selected=[
                _selected("continuity", "the unresolved thread about our last session", 0.91),
                _selected("trust", "whether Juniper felt heard last time", 0.77),
                _selected("curiosity", "what changed since we last spoke", 0.62),
                _selected("overflow", "a fourth item that must be truncated away", 0.40),
            ]
        )
    brief = MindHandoffBriefV1(
        mind_quality=quality,  # type: ignore[arg-type]
        active_frontier=frontier,
        stance_payload=_stance_payload(),
        shadow_synthesis=None,  # proves no projection dependency
    )
    return MindRunResultV1(
        mind_run_id=uuid4(),
        ok=ok,
        snapshot_hash="deadbeef",
        decision=MindControlDecisionV1(route_kind="chat", mode_binding="advisory", allowed_verbs=["speak"]),
        brief=brief,
        mind_quality=quality,  # type: ignore[arg-type]
    )


def test_meaningful_synthesis_key_set_equals_allow_list() -> None:
    coloring = select_mind_coloring(_result(ok=True, quality="meaningful_synthesis"), max_items=3)
    assert coloring is not None
    assert set(coloring.keys()) == MIND_COLORING_ALLOWED_KEYS


def test_task_control_fields_never_cross() -> None:
    coloring = select_mind_coloring(_result(ok=True, quality="meaningful_synthesis"), max_items=3)
    assert coloring is not None
    forbidden = {
        "task_mode", "answer_strategy", "conversation_frame", "response_priorities",
        "response_hazards", "route_kind", "mode_binding", "allowed_verbs", "mode_suggestion",
    }
    assert forbidden.isdisjoint(coloring.keys())
    # And none leak into nested values as dict keys.
    import json
    blob = json.dumps(coloring)
    for token in ("task_mode", "answer_strategy", "response_hazards", "mode_binding"):
        assert token not in blob


def test_themes_and_curiosity_survive_without_shadow_synthesis() -> None:
    coloring = select_mind_coloring(_result(ok=True, quality="meaningful_synthesis"), max_items=3)
    assert coloring is not None
    assert coloring["reflective_themes"] == ["continuity", "trust", "the shape of our work"]
    # curiosity_threads derive from active_frontier.selected[].summary
    assert coloring["curiosity_threads"] == [
        "the unresolved thread about our last session",
        "whether Juniper felt heard last time",
        "what changed since we last spoke",
    ]


def test_attention_frontier_shape_and_truncation() -> None:
    coloring = select_mind_coloring(_result(ok=True, quality="meaningful_synthesis"), max_items=3)
    assert coloring is not None
    af = coloring["attention_frontier"]
    assert len(af) == 3  # 4th item truncated
    assert af[0] == {"label": "continuity", "summary": "the unresolved thread about our last session", "score": 0.91}


def test_provenance_present() -> None:
    result = _result(ok=True, quality="meaningful_synthesis")
    coloring = select_mind_coloring(result, max_items=3)
    assert coloring is not None
    assert coloring["mind_quality"] == "meaningful_synthesis"
    assert coloring["mind_run_id"] == str(result.mind_run_id)
    assert coloring["snapshot_hash"] == "deadbeef"


def test_non_meaningful_returns_none() -> None:
    assert select_mind_coloring(_result(ok=True, quality="shadow_synthesis"), max_items=3) is None
    assert select_mind_coloring(_result(ok=True, quality="fallback_contract_only"), max_items=3) is None


def test_not_ok_returns_none() -> None:
    assert select_mind_coloring(_result(ok=False, quality="meaningful_synthesis"), max_items=3) is None


def test_empty_substance_returns_none() -> None:
    # meaningful_synthesis but no frontier and empty stance_payload -> no substance -> skip.
    empty = MindRunResultV1(
        mind_run_id=uuid4(),
        ok=True,
        brief=MindHandoffBriefV1(mind_quality="meaningful_synthesis", active_frontier=None, stance_payload={}),
        mind_quality="meaningful_synthesis",
    )
    assert select_mind_coloring(empty, max_items=3) is None
