from __future__ import annotations

from uuid import uuid4

from app.mind_enrichment import MIND_COLORING_ALLOWED_KEYS, select_mind_coloring
from orion.mind.synthesis_v1 import (
    ActiveCognitiveFrontierV1,
    AppraisalFeatureVectorV1,
    SelectedFrontierMatterV1,
)
from orion.mind.v1 import MindControlDecisionV1, MindHandoffBriefV1, MindRunResultV1
from orion.schemas.chat_stance import ChatStanceBrief


def _selected(label: str, summary: str, score: float) -> SelectedFrontierMatterV1:
    return SelectedFrontierMatterV1(
        matter_id=f"m-{label}",
        source_claim_id=f"c-{label}",
        label=label,
        summary=summary,
        matter_kind="curiosity_affordance",
        score=score,
        features=AppraisalFeatureVectorV1(confidence=score),
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
    from app.mind_enrichment import MIND_COLORING_BASE_KEYS, MIND_COLORING_ORION_WORK_SHAPE_KEYS

    coloring = select_mind_coloring(_result(ok=True, quality="meaningful_synthesis"), max_items=3)
    assert coloring is not None
    assert MIND_COLORING_ALLOWED_KEYS == (
        MIND_COLORING_BASE_KEYS | MIND_COLORING_ORION_WORK_SHAPE_KEYS
    )
    assert set(coloring.keys()) <= MIND_COLORING_ALLOWED_KEYS
    # Original self/attention keys still appear; work-shape keys stay origin-gated.
    assert {
        "attention_frontier",
        "reflective_themes",
        "curiosity_threads",
        "self_relevance",
        "identity_salience",
        "juniper_relevance",
        "mind_quality",
        "mind_run_id",
        "snapshot_hash",
    } <= set(coloring.keys())
    assert MIND_COLORING_ORION_WORK_SHAPE_KEYS.isdisjoint(coloring.keys())


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


def _result_with_payload(payload: dict) -> MindRunResultV1:
    return MindRunResultV1(
        mind_run_id=uuid4(),
        ok=True,
        snapshot_hash="deadbeef",
        brief=MindHandoffBriefV1(
            mind_quality="meaningful_synthesis",
            active_frontier=None,
            stance_payload=payload,
        ),
        mind_quality="meaningful_synthesis",
    )


def test_scalar_fields_clipped_and_no_nested_leak() -> None:
    # Defense-in-depth: scalar fields are length-bounded; a non-scalar value
    # (e.g. a dict carrying task-control keys) is DROPPED, never stringified,
    # so nested keys can never leak into the coloring blob.
    import json

    payload = {
        "self_relevance": "x" * 1000,
        "identity_salience": {"task_mode": "reflective_dialogue", "leak": "z" * 1000},
        "juniper_relevance": "Juniper matters",
    }
    coloring = select_mind_coloring(_result_with_payload(payload), max_items=3)
    assert coloring is not None
    assert isinstance(coloring["self_relevance"], str)
    assert len(coloring["self_relevance"]) <= 240
    # Non-scalar identity_salience is dropped entirely, not stringified.
    assert coloring["identity_salience"] is None
    blob = json.dumps(coloring)
    assert "task_mode" not in blob


def test_whitespace_only_scalar_normalizes_to_none() -> None:
    payload = {
        "self_relevance": "   ",
        "reflective_themes": ["continuity"],  # provide real substance so coloring fires
    }
    coloring = select_mind_coloring(_result_with_payload(payload), max_items=3)
    assert coloring is not None
    assert coloring["self_relevance"] is None


def test_user_intent_passes_for_juniper_origin() -> None:
    coloring = select_mind_coloring(
        _result(ok=True, quality="meaningful_synthesis"),
        max_items=3,
        utterance_origin="juniper",
    )
    assert coloring is not None
    assert coloring.get("user_intent") == "connect"
    assert "conversation_frame" not in coloring
    assert "task_mode" not in coloring
    assert "expected_depth" not in coloring  # orion-only soft label


def test_orion_origin_passes_soft_work_shape_labels() -> None:
    payload = _stance_payload()
    payload["expected_depth"] = "deep"
    payload["cross_cutting"] = "yes"
    payload["foresight_note"] = "Likely multi-service archaeology."
    coloring = select_mind_coloring(
        _result_with_payload(payload),
        max_items=3,
        utterance_origin="orion",
    )
    assert coloring is not None
    assert coloring["expected_depth"] == "deep"
    assert coloring["cross_cutting"] == "yes"
    assert "multi-service" in coloring["foresight_note"]


def test_soft_labels_blocked_for_juniper_even_if_payload_has_them() -> None:
    payload = _stance_payload()
    payload["expected_depth"] = "deep"
    coloring = select_mind_coloring(
        _result_with_payload(payload),
        max_items=3,
        utterance_origin="juniper",
    )
    assert coloring is not None
    assert "expected_depth" not in coloring


def test_uncertainty_summary_from_frontier_features() -> None:
    coloring = select_mind_coloring(
        _result(ok=True, quality="meaningful_synthesis"),
        max_items=3,
        utterance_origin="juniper",
    )
    assert coloring is not None
    summary = coloring.get("uncertainty_summary")
    assert summary is not None
    assert "continuity:0.91" in summary
    assert "trust:0.77" in summary
    assert "overflow" not in summary
    assert len(summary) <= 240


def test_invalid_soft_labels_dropped_even_for_orion() -> None:
    payload = _stance_payload()
    payload["expected_depth"] = "enormous"
    payload["cross_cutting"] = "maybe"
    coloring = select_mind_coloring(
        _result_with_payload(payload),
        max_items=3,
        utterance_origin="orion",
    )
    assert coloring is not None
    assert "expected_depth" not in coloring
    assert "cross_cutting" not in coloring


def test_chat_stance_brief_soft_work_shape_fields_roundtrip() -> None:
    brief = ChatStanceBrief(
        conversation_frame="technical",
        user_intent="Investigate a cross-service gap.",
        self_relevance="This is my own investigation.",
        juniper_relevance="Juniper asked for hire determination.",
        answer_strategy="DirectAnswer",
        stance_summary="Orion-authored investigation.",
        expected_depth="deep",
        cross_cutting="yes",
        foresight_note="Likely multi-service archaeology.",
    )
    dumped = brief.model_dump(mode="json")
    reparsed = ChatStanceBrief.model_validate(dumped)
    assert reparsed.expected_depth == "deep"
    assert reparsed.cross_cutting == "yes"
    assert "multi-service" in (reparsed.foresight_note or "")
