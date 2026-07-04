from __future__ import annotations

from orion.substrate.appraisal.paradigms.registry import PARADIGM_REGISTRY, ParadigmBuildContext


def test_paradigm_registry_includes_repair_pressure() -> None:
    assert "repair_pressure" in PARADIGM_REGISTRY
    ctx = ParadigmBuildContext(llm_caller=lambda _prompt: {}, weights_path="config/substrate/repair_pressure_weights.v2.yaml")
    paradigm = PARADIGM_REGISTRY["repair_pressure"](ctx)
    assert paradigm.name == "repair_pressure"
