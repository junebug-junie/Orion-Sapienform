from __future__ import annotations

from datetime import datetime, timezone

from orion.harness.prefix import compile_harness_prefix
from orion.harness.repair import map_repair_pressure_contract
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1


def _thought(**overrides: object) -> ThoughtEventV1:
    base = {
        "event_id": "t-1",
        "correlation_id": "c-1",
        "session_id": None,
        "created_at": datetime.now(timezone.utc),
        "imperative": "Give file paths and commands.",
        "tone": "direct",
        "strain_refs": ["n-1"],
        "evidence_refs": ["n-1"],
        "repair_pressure_level": 0.9,
        "stance_harness_slice": StanceHarnessSliceV1(
            task_mode="triage",
            conversation_frame="mixed",
            answer_strategy="direct",
        ),
    }
    base.update(overrides)
    return ThoughtEventV1.model_validate(base)


def test_repair_overlay_changes_harness_prefix() -> None:
    thought = _thought()
    overlay: HarnessRepairOverlayV1 = map_repair_pressure_contract(
        {"mode": "repair_concrete", "rules": ["include file/module boundaries"]}
    )
    default_prefix = compile_harness_prefix(thought, repair_overlay=HarnessRepairOverlayV1())
    repair_prefix = compile_harness_prefix(thought, repair_overlay=overlay)
    assert repair_prefix != default_prefix
    assert "repair_concrete" in repair_prefix or "file/module" in repair_prefix
    assert "Give file paths and commands." in repair_prefix
    assert "direct" in repair_prefix
    assert "n-1" in repair_prefix


def test_default_mode_overlay_maps_to_empty_harness_overlay() -> None:
    overlay = map_repair_pressure_contract({"mode": "default"})
    assert overlay.mode == "default"
    assert overlay.rule_lines == []
    assert overlay.prefix_overlay == ""
    assert overlay.finalize_overlay == ""


def test_default_overlay_does_not_change_harness_prefix() -> None:
    thought = _thought()
    base = compile_harness_prefix(thought, repair_overlay=HarnessRepairOverlayV1())
    default_mapped = compile_harness_prefix(
        thought,
        repair_overlay=map_repair_pressure_contract({"mode": "default"}),
    )
    assert base == default_mapped
    assert "Give file paths and commands." in base
    assert "direct" in base
    assert "n-1" in base
    assert "repair_concrete" not in base
    assert "concrete_bias" not in base
