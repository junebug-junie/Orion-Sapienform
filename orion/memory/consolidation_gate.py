from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from orion.memory.low_info_social import is_low_info_social

_SHIFT_FORCE = frozenset({"TOPIC", "STANCE", "REPAIR"})


@dataclass
class ConsolidationGateResult:
    action: Literal["skip", "propose"]
    reasons: list[str] = field(default_factory=list)
    dominant_shift: str | None = None
    grammar_event_ids: list[str] = field(default_factory=list)
    window_novelty_max: float = 0.0
    window_significance_max: float = 0.0


def _appraisal(turn: dict[str, Any]) -> dict[str, Any]:
    spark = turn.get("spark_meta") if isinstance(turn.get("spark_meta"), dict) else {}
    appraisal = spark.get("turn_change_appraisal")
    return appraisal if isinstance(appraisal, dict) else {}


def _significance(turn: dict[str, Any]) -> float:
    spark = turn.get("spark_meta") if isinstance(turn.get("spark_meta"), dict) else {}
    val = spark.get("memory_significance_score", turn.get("memory_significance_score"))
    return float(val) if isinstance(val, (int, float)) else 0.0


def consolidation_memory_gate(
    *,
    turns: list[dict[str, Any]],
    grammar_repair_signal: bool,
    grammar_event_ids: list[str] | None = None,
    min_novelty: float = 0.35,
    min_significance: float = 0.40,
) -> ConsolidationGateResult:
    reasons: list[str] = []
    novelty_max = 0.0
    significance_max = 0.0
    dominant_shift: str | None = None
    best_novelty = -1.0

    for turn in turns:
        appraisal = _appraisal(turn)
        novelty = appraisal.get("novelty_score")
        n = float(novelty) if isinstance(novelty, (int, float)) else 0.0
        novelty_max = max(novelty_max, n)
        significance_max = max(significance_max, _significance(turn))
        shift = str(appraisal.get("shift_kind") or "NONE").upper()
        if n > best_novelty and shift in _SHIFT_FORCE:
            best_novelty = n
            dominant_shift = shift

    if grammar_repair_signal:
        return ConsolidationGateResult(
            action="propose",
            reasons=["repair_signal"],
            dominant_shift=dominant_shift or "REPAIR",
            grammar_event_ids=list(grammar_event_ids or []),
            window_novelty_max=novelty_max,
            window_significance_max=significance_max,
        )

    substantive_shift = any(
        str(_appraisal(t).get("shift_kind") or "NONE").upper() in _SHIFT_FORCE
        and float(_appraisal(t).get("novelty_score") or 0) >= min_novelty
        for t in turns
    )
    if substantive_shift:
        return ConsolidationGateResult(
            action="propose",
            reasons=["substantive_shift"],
            dominant_shift=dominant_shift,
            grammar_event_ids=list(grammar_event_ids or []),
            window_novelty_max=novelty_max,
            window_significance_max=significance_max,
        )

    if novelty_max >= min_novelty:
        return ConsolidationGateResult(
            action="propose",
            reasons=["novelty_above_floor"],
            dominant_shift=dominant_shift,
            grammar_event_ids=list(grammar_event_ids or []),
            window_novelty_max=novelty_max,
            window_significance_max=significance_max,
        )

    if significance_max >= min_significance:
        return ConsolidationGateResult(
            action="propose",
            reasons=["significance_above_floor"],
            dominant_shift=dominant_shift,
            grammar_event_ids=list(grammar_event_ids or []),
            window_novelty_max=novelty_max,
            window_significance_max=significance_max,
        )

    has_substantive_text = any(
        not is_low_info_social(str(t.get("prompt") or ""))
        or not is_low_info_social(str(t.get("response") or ""))
        for t in turns
    ) if turns else False
    if has_substantive_text:
        return ConsolidationGateResult(
            action="propose",
            reasons=["substantive_text"],
            dominant_shift=dominant_shift,
            grammar_event_ids=list(grammar_event_ids or []),
            window_novelty_max=novelty_max,
            window_significance_max=significance_max,
        )

    all_low_info = all(
        is_low_info_social(str(t.get("prompt") or ""))
        and is_low_info_social(str(t.get("response") or ""))
        for t in turns
    ) if turns else True
    if all_low_info:
        reasons.append("low_info_social")

    if novelty_max < min_novelty:
        reasons.append("novelty_below_floor")
    if significance_max < min_significance:
        reasons.append("significance_below_floor")

    return ConsolidationGateResult(
        action="skip",
        reasons=reasons or ["no_substantive_signal"],
        dominant_shift=None,
        grammar_event_ids=list(grammar_event_ids or []),
        window_novelty_max=novelty_max,
        window_significance_max=significance_max,
    )
