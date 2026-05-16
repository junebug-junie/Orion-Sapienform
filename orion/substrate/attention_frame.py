from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Sequence

from orion.schemas.attention_frame import AttentionFrameV1, AttentionSignalV1
from orion.substrate.attention.common import compact
from orion.substrate.attention.detectors import AttentionSignalDetector, default_attention_detectors
from orion.substrate.attention.policy import (
    base_suppressions,
    direct_work_turn,
    generic_reversal_present,
    select_actions,
)
from orion.substrate.attention.scoring import autonomy_pressure_from_signals, build_open_loops, merge_signals

_MAX_SOURCE_TEXT = 600


def attention_frame_enabled() -> bool:
    return str(os.getenv("ORION_CURIOSITY_FRAME_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name) or default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(os.getenv(name) or default)))
    except (TypeError, ValueError):
        return default


def _stale_thread_active(inputs: dict[str, Any]) -> bool:
    situation = inputs.get("situation") if isinstance(inputs.get("situation"), dict) else {}
    phase = situation.get("conversation_phase") if isinstance(situation.get("conversation_phase"), dict) else {}
    return str(phase.get("phase_change") or "") == "stale_thread"


def _detect_signals(
    *,
    detectors: Sequence[AttentionSignalDetector],
    ctx: dict[str, Any],
    inputs: dict[str, Any],
    belief_lineage: list[str],
) -> list[AttentionSignalV1]:
    signals: list[AttentionSignalV1] = []
    for detector in detectors:
        try:
            signals.extend(detector.detect(ctx, inputs, belief_lineage))
        except Exception:
            continue
    return signals


def build_attention_frame(
    *,
    ctx: dict[str, Any],
    inputs: dict[str, Any],
    belief_lineage: list[str] | None = None,
    detectors: Sequence[AttentionSignalDetector] | None = None,
) -> AttentionFrameV1:
    user_text = compact(ctx.get("user_message") or ctx.get("raw_user_text") or "", _MAX_SOURCE_TEXT)
    max_open = _env_int("ORION_CURIOSITY_MAX_OPEN_LOOPS", 5)
    max_asks = max(0, min(1, _env_int("ORION_CURIOSITY_MAX_SELECTED_ACTIONS", 1)))
    min_ask = _env_float("ORION_CURIOSITY_MIN_ASK_SCORE", 0.65)
    lineage = list(belief_lineage or [])
    detector_list = list(detectors) if detectors is not None else default_attention_detectors()
    signals = _detect_signals(detectors=detector_list, ctx=ctx, inputs=inputs, belief_lineage=lineage)
    merged_signals = merge_signals(signals, limit=max_open * 3)
    stale = _stale_thread_active(inputs)
    generic = generic_reversal_present(user_text)
    direct = direct_work_turn(user_text)
    open_loops = build_open_loops(
        signals=merged_signals,
        ctx=ctx,
        inputs=inputs,
        belief_lineage=lineage,
        direct_turn=direct,
        generic_reversal=generic,
        stale_thread_active=stale,
        max_open=max_open,
    )
    actions, selected, suppressions, deferred = select_actions(
        open_loops=open_loops,
        suppressions=base_suppressions(user_text=user_text, stale_thread_active=stale),
        min_ask=min_ask,
        max_asks=max_asks,
        generic_reversal=generic,
        stale_thread_active=stale,
    )
    _autonomy_value, autonomy_signals = autonomy_pressure_from_signals(merged_signals)
    return AttentionFrameV1(
        generated_at=datetime.now(timezone.utc),
        turn_id=str(ctx.get("turn_id") or ctx.get("message_id") or "") or None,
        session_id=str(ctx.get("session_id") or "") or None,
        correlation_id=str(ctx.get("correlation_id") or ctx.get("trace_id") or "") or None,
        open_loops=open_loops,
        live_unknowns=[loop.description for loop in open_loops if not loop.already_known],
        candidate_actions=actions,
        selected_action=selected,
        suppressions=suppressions,
        deferred_items=deferred[:max_open],
        debug={
            "enabled": True,
            "direct_turn": direct,
            "generic_reversal": generic,
            "max_open_loops": max_open,
            "max_selected_asks": max_asks,
            "min_ask_score": min_ask,
            "belief_lineage": lineage[:8],
            "detectors": [getattr(detector, "detector_id", type(detector).__name__) for detector in detector_list],
            "signal_count": len(signals),
            "merged_signal_count": len(merged_signals),
            "autonomy_signals": autonomy_signals,
        },
    )
