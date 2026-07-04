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
