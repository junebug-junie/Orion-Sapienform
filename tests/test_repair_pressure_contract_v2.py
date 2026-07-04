from __future__ import annotations

from orion.substrate.appraisal.contract import assemble_repair_contract_delta


def test_kind_active_rules_union_when_score_high() -> None:
    kind_scores = {
        "specificity_demand": 0.91,
        "trust_rupture": 0.80,
        "coherence_gap": 0.30,
        "repetition_failure": 0.0,
        "operational_block": 0.70,
        "explicit_repair_command": 0.0,
        "assistant_accountability_demand": 0.0,
    }
    delta = assemble_repair_contract_delta(
        contract_before={"mode": "default"},
        level=0.82,
        confidence=0.71,
        kind_scores=kind_scores,
    )
    assert delta["mode"] == "repair_concrete"
    rules = delta["rules"]
    assert any("file/module boundaries" in r for r in rules)
    assert any("acknowledge correction briefly" in r for r in rules)
    assert not any("one concrete operational path" in r for r in rules)  # coherence_gap below 0.65


def test_mid_level_yields_concrete_bias() -> None:
    delta = assemble_repair_contract_delta(
        contract_before={"mode": "default"},
        level=0.55,
        confidence=0.70,
        kind_scores={"specificity_demand": 0.80},
    )
    assert delta["mode"] == "concrete_bias"


def test_low_level_returns_unchanged_contract() -> None:
    before = {"mode": "default", "rules": ["keep"]}
    delta = assemble_repair_contract_delta(
        contract_before=before,
        level=0.20,
        confidence=0.90,
        kind_scores={},
    )
    assert delta["mode"] == "default"
    assert delta.get("rules") == ["keep"]
