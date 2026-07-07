from __future__ import annotations

from dataclasses import dataclass, field

from orion.memory.low_info_social import is_low_info_social

SIGNIFICANT_SHIFT_KINDS = frozenset({"TOPIC", "STANCE", "REPAIR"})


@dataclass
class RecallSkipGateResult:
    skip: bool
    reasons: list[str] = field(default_factory=list)


def recall_skip_gate(
    user_message: str,
    appraisal: dict | None,
    has_repair_grammar_signal: bool,
    max_novelty: float = 0.25,
    shift_novelty_floor: float = 0.35,
) -> RecallSkipGateResult:
    """Deterministic Phase 0 gate: skip recall on low-info social turns."""
    reasons: list[str] = []

    if has_repair_grammar_signal:
        return RecallSkipGateResult(skip=False, reasons=reasons)

    if not is_low_info_social(user_message):
        return RecallSkipGateResult(skip=False, reasons=reasons)

    reasons.append("low_info_social")

    novelty_score = appraisal.get("novelty_score") if appraisal else None
    shift_kind = str((appraisal or {}).get("shift_kind") or "NONE").upper()

    novelty_below_floor = novelty_score is None or novelty_score < max_novelty
    if novelty_below_floor:
        reasons.append("novelty_below_floor")

    significant_shift = (
        shift_kind in SIGNIFICANT_SHIFT_KINDS
        and novelty_score is not None
        and novelty_score >= shift_novelty_floor
    )
    if significant_shift:
        return RecallSkipGateResult(skip=False, reasons=reasons)

    skip = novelty_below_floor
    return RecallSkipGateResult(skip=skip, reasons=reasons)
