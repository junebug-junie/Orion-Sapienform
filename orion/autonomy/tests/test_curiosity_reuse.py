from orion.autonomy.curiosity_reuse import outcome_from_followup, select_reusable_followup
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.schemas.world_pulse import CuriosityFindingV1, CuriosityFollowupV1


def _gap_signal(section: str) -> FrontierInvocationSignalV1:
    # NOTE: anchor_scope / target_zone / task_type_candidate are Literal type
    # aliases (plain strings), not classes, so they take string values here.
    return FrontierInvocationSignalV1(
        signal_type="world_coverage_gap",
        anchor_scope="world",
        target_zone="world_ontology",
        task_type_candidate="world_fact_hypothesis",
        focal_node_refs=[f"section:{section}"],
        signal_strength=0.9,
        confidence=0.8,
    )


def _followup(section: str, articles=None) -> CuriosityFollowupV1:
    # Fall back only when articles is unset; an explicit [] must stay empty so
    # test_empty_articles_not_reused actually exercises the empty-articles path.
    if articles is None:
        articles = [
            CuriosityFindingV1(url="https://ex/1", title="t", description="d", salience=0.6)
        ]
    return CuriosityFollowupV1(
        section=section,
        driving_gap="missing",
        query=f"{section.replace('_', ' ')} recent news coverage",
        articles=articles,
        action_id="fetch-abc",
        correlation_id="run-1",
    )


def test_selects_matching_section():
    followups = [_followup("ai_technology"), _followup("hardware_compute_gpu")]
    signals = [_gap_signal("hardware_compute_gpu")]
    chosen = select_reusable_followup(followups, signals)
    assert chosen is not None
    assert chosen.section == "hardware_compute_gpu"


def test_no_match_returns_none():
    followups = [_followup("ai_technology")]
    signals = [_gap_signal("hardware_compute_gpu")]
    assert select_reusable_followup(followups, signals) is None


def test_empty_articles_not_reused():
    followups = [_followup("hardware_compute_gpu", articles=[])]
    signals = [_gap_signal("hardware_compute_gpu")]
    assert select_reusable_followup(followups, signals) is None


def test_outcome_from_followup_maps_fields():
    outcome = outcome_from_followup(_followup("hardware_compute_gpu"), run_id="run-1")
    assert outcome.kind == "web.fetch.readonly"
    assert outcome.success is True
    assert outcome.action_id == "fetch-abc"
    assert outcome.query == "hardware compute gpu recent news coverage"
    assert len(outcome.articles) == 1
    assert outcome.articles[0].url == "https://ex/1"
    assert round(outcome.salience, 2) == 0.6
