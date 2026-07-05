from __future__ import annotations

from typing import Any, Literal, cast

from orion.schemas.harness_finalize import HarnessRepairOverlayV1

_VALID_MODES = frozenset({"default", "concrete_bias", "repair_concrete"})

_PREFIX_INTRO: dict[str, str] = {
    "repair_concrete": "Repair turn: answer concretely and operationally.",
    "concrete_bias": "Add concrete specificity this turn.",
}


def _compile_finalize_overlay(mode: str, rules: list[str]) -> str:
    intro = _PREFIX_INTRO[mode]
    return intro + " " + "; ".join(rules) + "."


def map_repair_pressure_contract(contract: dict[str, Any] | None) -> HarnessRepairOverlayV1:
    """Map ingress repair_pressure_contract metadata to harness overlay fields.

    Uses the same mode/rules signals as Brain TURN CONTRACT repair wiring, but
    produces a HarnessRepairOverlayV1 for fcc prefix and finalize overlays.
    """
    if not isinstance(contract, dict):
        return HarnessRepairOverlayV1()

    mode = str(contract.get("mode") or "default")
    if mode not in _VALID_MODES:
        mode = "default"

    raw_rules = contract.get("rules") or []
    rule_lines = [str(rule).strip() for rule in raw_rules if str(rule).strip()]

    if mode == "default" or not rule_lines:
        return HarnessRepairOverlayV1(mode="default")

    prefix_overlay = _PREFIX_INTRO[mode]
    finalize_overlay = _compile_finalize_overlay(mode, rule_lines)
    return HarnessRepairOverlayV1(
        mode=cast(Literal["default", "concrete_bias", "repair_concrete"], mode),
        rule_lines=rule_lines,
        prefix_overlay=prefix_overlay,
        finalize_overlay=finalize_overlay,
    )
