from __future__ import annotations

import pytest

from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1, TurnWindowMessageV1
from orion.substrate.appraisal.paradigms.repair_pressure_v2 import (
    RepairPressureV2Paradigm,
    build_repair_probe_prompt,
    reduce_repair_level,
)


def test_build_repair_probe_prompt_lists_seven_kinds() -> None:
    window = [
        TurnWindowMessageV1(role="user", content="you gave garbage directions"),
        TurnWindowMessageV1(role="assistant", content="try this plan"),
    ]
    prompt = build_repair_probe_prompt(window)
    for kind in (
        "specificity_demand",
        "trust_rupture",
        "coherence_gap",
        "repetition_failure",
        "operational_block",
        "explicit_repair_command",
        "assistant_accountability_demand",
    ):
        assert kind in prompt


def test_reduce_repair_level_uses_yaml_weights() -> None:
    kind_scores = {k: 0.9 for k in (
        "specificity_demand", "trust_rupture", "coherence_gap",
        "repetition_failure", "operational_block", "explicit_repair_command",
        "assistant_accountability_demand",
    )}
    weights = {k: 0.14 for k in kind_scores}  # sums ~0.98
    level, confidence = reduce_repair_level(kind_scores, confidences={k: 0.8 for k in kind_scores}, weights=weights)
    assert level >= 0.75
    assert confidence >= 0.60


def test_weak_but_consistent_evidence_no_longer_reads_as_zero_confidence() -> None:
    # 2026-07-30 bug fix regression test. The old formula only counted a kind
    # toward confidence if its own score > 0.5 -- so several real, weak
    # (<=0.5) scores could sum into a genuinely nonzero `level` while
    # `confidence` was forced to exactly 0.0, indistinguishable from true
    # "no evidence at all". Real repair_pressure_appraisal_log data showed
    # 40/52 rows at one repeated level with confidence pinned at 0.0. All
    # three scores here are <=0.5, so this exercises the NEW weighted-average
    # branch specifically (the old min()-preserving branch never activates).
    kind_scores = {
        "specificity_demand": 0.3,
        "trust_rupture": 0.25,
        "coherence_gap": 0.35,
    }
    weights = {"specificity_demand": 0.22, "trust_rupture": 0.20, "coherence_gap": 0.16}
    confidences = {"specificity_demand": 0.7, "trust_rupture": 0.6, "coherence_gap": 0.65}

    level, confidence = reduce_repair_level(kind_scores, confidences=confidences, weights=weights)

    # Exact values, not loose bounds -- pins the actual formula
    # (contribution-weighted average), not just its qualitative direction.
    expected_level = 0.22 * 0.3 + 0.20 * 0.25 + 0.16 * 0.35
    contributions = {"specificity_demand": 0.22 * 0.3, "trust_rupture": 0.20 * 0.25, "coherence_gap": 0.16 * 0.35}
    expected_confidence = (
        contributions["specificity_demand"] * 0.7
        + contributions["trust_rupture"] * 0.6
        + contributions["coherence_gap"] * 0.65
    ) / sum(contributions.values())

    assert level == pytest.approx(expected_level)
    assert confidence == pytest.approx(expected_confidence)
    assert confidence > 0.0  # must NOT read as the paradigm's "no evidence" signal


def test_zero_score_evidence_still_reads_zero_confidence() -> None:
    # Degenerate-input defensive check, not the actual production no-evidence
    # path (that's handled entirely upstream by run()'s `if not evidence`/
    # `if not weights` early returns, which hardcode level=confidence=0.0
    # directly and never call this function -- score_binary_logprob is a
    # sigmoid, so a real per-kind score can get arbitrarily close to 0.0 but
    # is never exactly 0.0 in production). This only confirms the function
    # itself doesn't divide-by-zero or misbehave if every score is exactly 0.
    kind_scores = {"specificity_demand": 0.0, "trust_rupture": 0.0}
    weights = {"specificity_demand": 0.22, "trust_rupture": 0.20}
    confidences = {"specificity_demand": 0.9, "trust_rupture": 0.9}

    level, confidence = reduce_repair_level(kind_scores, confidences=confidences, weights=weights)

    assert level == 0.0
    assert confidence == 0.0


def test_confidence_is_proportional_to_evidence_strength_when_all_kinds_are_weak() -> None:
    # All three scores <=0.5 -- exercises the weighted-average branch. A
    # stronger-scoring kind should pull the blended confidence toward its own
    # confidence more than a weaker-scoring kind does, proportional to actual
    # contribution, not a flat average of the three confidences.
    kind_scores = {
        "specificity_demand": 0.45,  # strongest of the three (still <=0.5)
        "trust_rupture": 0.2,
        "coherence_gap": 0.2,
    }
    weights = {"specificity_demand": 0.22, "trust_rupture": 0.20, "coherence_gap": 0.16}
    confidences = {"specificity_demand": 0.95, "trust_rupture": 0.4, "coherence_gap": 0.4}

    level, confidence = reduce_repair_level(kind_scores, confidences=confidences, weights=weights)

    flat_average = (0.95 + 0.4 + 0.4) / 3.0  # what a naive unweighted average would give
    assert confidence > flat_average  # the higher-contribution kind pulls it up, not a flat mean
    assert 0.4 < confidence < 0.95


def test_multi_strong_kind_case_unchanged_from_old_min_formula() -> None:
    # Code-review regression test (2026-07-30): a naive "always
    # weighted-average" version of this fix could substantially RAISE
    # confidence for cases where multiple kinds each already scored > 0.5
    # with divergent per-kind confidence -- a live-behavior change beyond
    # the bug being fixed, since apply_repair_pressure_contract()'s
    # confidence>=0.60 gate could then fire where it previously wouldn't.
    # This case must be BYTE-IDENTICAL to the original min()-based formula:
    # at least one kind's score already exceeds 0.5, so the fix's
    # min()-preserving branch must be the one that runs.
    kind_scores = {"a": 0.9, "b": 0.9}
    confidences = {"a": 0.95, "b": 0.3}
    weights = {"a": 0.5, "b": 0.5}

    level, confidence = reduce_repair_level(kind_scores, confidences=confidences, weights=weights)

    assert level == pytest.approx(0.9)
    assert confidence == pytest.approx(0.3)  # min(0.95, 0.3), old formula's exact answer


def test_recovers_real_text_fallback_confidence_from_live_floor_row() -> None:
    # Root-cause regression test. Pulled directly from a real
    # repair_pressure_appraisal_log row's persisted `evidence` JSONB: all 7
    # kinds at the identical score/confidence from this module's own
    # _TEXT_FALLBACK_NO/_TEXT_FALLBACK_CONFIDENCE constants (the text-only
    # fallback path, not a hardcoded default elsewhere). The old formula
    # discarded the real 0.65 fallback confidence because 0.087 never
    # crosses 0.5 -- this must recover it exactly.
    kinds = (
        "specificity_demand", "trust_rupture", "coherence_gap",
        "repetition_failure", "operational_block",
        "explicit_repair_command", "assistant_accountability_demand",
    )
    fallback_score = 0.08706577244027125
    kind_scores = {k: fallback_score for k in kinds}
    confidences = {k: 0.65 for k in kinds}
    weights = {k: 1.0 / 7 for k in kinds}

    level, confidence = reduce_repair_level(kind_scores, confidences=confidences, weights=weights)

    assert level == pytest.approx(fallback_score)
    assert confidence == pytest.approx(0.65)  # was 0.0 under the old formula


@pytest.mark.asyncio
async def test_paradigm_fail_closed_on_missing_weights_file() -> None:
    async def _llm(_prompt: str) -> dict:
        return {
            "text": "specificity_demand: YES\n",
            "llm_uncertainty": {
                "available": True,
                "content": [
                    {
                        "token": "YES",
                        "logprob": -0.12,
                        "top_logprobs": [
                            {"token": "YES", "logprob": -0.12},
                            {"token": "NO", "logprob": -2.4},
                        ],
                    }
                ],
            },
        }

    paradigm = RepairPressureV2Paradigm(
        llm_caller=_llm,
        weights_path="/nonexistent/repair_pressure_weights.v2.yaml",
    )
    req = PreTurnAppraisalRequestV1(
        correlation_id="c2",
        session_id="s1",
        turn_window=[TurnWindowMessageV1(role="user", content="be specific")],
    )
    slice_ = await paradigm.run(req)
    assert slice_.level == 0.0
    assert "weights_file_missing" in slice_.notes


@pytest.mark.asyncio
async def test_paradigm_fail_closed_on_empty_llm() -> None:
    paradigm = RepairPressureV2Paradigm(llm_caller=lambda _prompt: {"text": "", "llm_uncertainty": {"available": False}})
    req = PreTurnAppraisalRequestV1(
        correlation_id="c1",
        session_id="s1",
        turn_window=[TurnWindowMessageV1(role="user", content="be specific")],
    )
    slice_ = await paradigm.run(req)
    assert slice_.level == 0.0
    assert "no_repair_evidence" in slice_.notes


@pytest.mark.asyncio
async def test_paradigm_text_fallback_when_llm_uncertainty_unavailable() -> None:
    kinds_text = "\n".join(
        f"{kind}: YES"
        for kind in (
            "specificity_demand",
            "trust_rupture",
            "coherence_gap",
            "repetition_failure",
            "operational_block",
            "explicit_repair_command",
            "assistant_accountability_demand",
        )
    )

    async def _llm(_prompt: str) -> dict:
        return {
            "content": kinds_text,
            "meta": {"llm_uncertainty": {"available": False, "token_count_observed": 0}},
            "raw": {},
        }

    paradigm = RepairPressureV2Paradigm(llm_caller=_llm)
    req = PreTurnAppraisalRequestV1(
        correlation_id="c4",
        session_id="s1",
        turn_window=[TurnWindowMessageV1(role="user", content="give me files and tests")],
    )
    slice_ = await paradigm.run(req)
    assert slice_.level >= 0.45
    assert "no_repair_evidence" not in slice_.notes


@pytest.mark.asyncio
async def test_paradigm_text_fallback_when_logprob_tokens_do_not_align() -> None:
    kinds_text = "\n".join(f"{kind}: YES" for kind in (
        "specificity_demand", "trust_rupture", "coherence_gap",
        "repetition_failure", "operational_block", "explicit_repair_command",
        "assistant_accountability_demand",
    ))

    async def _llm(_prompt: str) -> dict:
        return {
            "content": kinds_text,
            "meta": {"llm_uncertainty": {"available": True, "token_count_observed": 40}},
            "raw": {
                "probs": [
                    {
                        "token": "Hello",
                        "logprob": -0.2,
                        "top_logprobs": [
                            {"token": "Hello", "logprob": -0.2},
                            {"token": "Hi", "logprob": -1.5},
                        ],
                    }
                ]
            },
        }

    paradigm = RepairPressureV2Paradigm(llm_caller=_llm)
    req = PreTurnAppraisalRequestV1(
        correlation_id="c3",
        session_id="s1",
        turn_window=[
            TurnWindowMessageV1(role="user", content="stop hand waving"),
            TurnWindowMessageV1(role="assistant", content="here is another overview"),
            TurnWindowMessageV1(role="user", content="give me files and tests"),
        ],
    )
    slice_ = await paradigm.run(req)
    assert slice_.level >= 0.45
    assert slice_.contract_delta.get("mode") in {"repair_concrete", "concrete_bias"}
