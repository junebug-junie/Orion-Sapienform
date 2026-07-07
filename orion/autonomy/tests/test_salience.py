from __future__ import annotations

from orion.autonomy.salience import gap_terms_from_signals, score_article_salience
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1


def _gap_signal(section: str = "section:hardware_compute_gpu") -> FrontierInvocationSignalV1:
    return FrontierInvocationSignalV1(
        signal_type="world_coverage_gap",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="concept_graph",
        task_type_candidate="concept_expand",
        focal_node_refs=[section],
        signal_strength=0.65,
        evidence_summary="gap",
        confidence=0.65,
    )


def test_gap_terms_from_section_refs() -> None:
    terms = gap_terms_from_signals([_gap_signal()])
    assert terms == {"hardware", "compute", "gpu"}


def test_gap_terms_ignores_non_gap_signals() -> None:
    sig = _gap_signal()
    other = sig.model_copy(update={"signal_type": "curiosity_candidate"})
    assert gap_terms_from_signals([other]) == set()


def test_gap_terms_falls_back_to_query_when_no_section() -> None:
    sig = _gap_signal(section="node:not_a_section")
    terms = gap_terms_from_signals([sig], fallback_query="GPU supply chain news")
    assert terms == {"gpu", "supply", "chain", "news"}


def test_score_article_salience_fraction() -> None:
    gap = {"hardware", "compute", "gpu"}
    # 2 of 3 gap terms present.
    score = score_article_salience("New GPU compute cluster launches", gap)
    assert abs(score - (2 / 3)) < 1e-9


def test_score_article_salience_empty_gap_is_zero() -> None:
    assert score_article_salience("anything at all", set()) == 0.0


def test_score_article_salience_empty_text_is_zero() -> None:
    assert score_article_salience("", {"gpu"}) == 0.0


def test_score_article_salience_clamped_to_one() -> None:
    gap = {"gpu"}
    assert score_article_salience("gpu gpu gpu", gap) == 1.0
