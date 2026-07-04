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
