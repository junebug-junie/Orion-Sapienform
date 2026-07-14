"""Continuous global broadcast — rung 3 of the self-modeling loop.

The workspace competition in ``attention_frame.py`` (detectors → merge →
``select_actions`` → one winner) is chat-turn-scoped and gated. This module
runs the same competition over the substrate graph itself: nodes carrying
``dynamic_pressure`` (rung 1's pressure field) and ``prediction_error``, plus
the belief-derived nodes the rung-2 lanes materialize into the graph, compete
each tick and the winning coalition is re-broadcast as a projection other
organs can read.

No new selection policy: ``select_actions`` is reused with ``max_asks=0`` so
the broadcast selects a coalition without generating chat questions, and no
action is taken from the broadcast (that is rung 5's governed territory).
"""

from __future__ import annotations

import logging
import os
from collections import deque
from datetime import datetime, timezone
from typing import Any, Sequence

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    AttentionSignalV1,
)
from orion.substrate.attention.common import compact, stable_id
from orion.substrate.attention.policy import select_actions
from orion.substrate.attention.scoring import build_open_loops, merge_signals

_TRUTHY = {"1", "true", "yes", "on"}

BROADCAST_FLAG = "ORION_ATTENTION_BROADCAST_ENABLED"
BROADCAST_PROJECTION_ID = "substrate.attention.broadcast.v1"
DEFAULT_MIN_SALIENCE = 0.2
DEFAULT_MAX_SIGNALS = 24

# Hysteresis state: sliding window of recent coalition node IDs
_coalition_history: deque[frozenset[str]] = deque(maxlen=3)
_current_active_coalition: frozenset[str] | None = None
_dwell_ticks: int = 0
# Transition log: last 10 activation/decay events, mirrored into
# AttentionBroadcastProjectionV1.coalition_history (schema caps at 10).
_transition_history: deque[dict[str, Any]] = deque(maxlen=10)

# Recent selected-loop counts for habituation (inhibition-of-return). Capped.
_recent_selected_counts: dict[str, int] = {}
# First real-clock moment each loop_id was ever selected. Feeds
# salience._recency()'s half-life decay -- previously never wired up here,
# so history.first_seen_at was always an empty dict and _recency() always
# hit its "never seen before" branch and returned 1.0 unconditionally,
# forever, for every loop (found live 2026-07-14: a loop verdicted
# "resolved" on 07-08 and "dismissed" on 07-10 was still winning selection
# on 07-14 with byte-identical salience_features to the 07-10 snapshot,
# including recency=1.0). Same in-memory-only, capped-dict lifecycle as
# _recent_selected_counts -- resets on service restart, consistent with
# that field's existing, already-accepted behavior; not attempting to add
# cross-restart persistence here, out of scope for this fix.
_first_selected_at: dict[str, datetime] = {}
_MAX_TRACKED_THEMES = 64

# A theme selected at least this many times is "resonating" (stuck) — engage
# the resonance term of habituation so inhibition-of-return can release it.
_RESONANCE_MIN_COUNT = 3


def _record_selection(loop_id: str | None, *, now: datetime | None = None) -> None:
    if not loop_id:
        return
    _recent_selected_counts[loop_id] = _recent_selected_counts.get(loop_id, 0) + 1
    _first_selected_at.setdefault(loop_id, now or datetime.now(timezone.utc))
    if len(_recent_selected_counts) > _MAX_TRACKED_THEMES:
        drop = min(_recent_selected_counts, key=_recent_selected_counts.get)
        _recent_selected_counts.pop(drop, None)
        _first_selected_at.pop(drop, None)


def _current_history(resonance_theme_keys: set[str] | None = None) -> "SalienceHistory":
    from orion.substrate.attention.salience import SalienceHistory

    # A theme selected repeatedly IS resonating (stuck): engage the full
    # habituation penalty so inhibition-of-return can eventually release it.
    # Callers may still pass explicit keys (e.g. tests / other producers).
    if resonance_theme_keys is None:
        resonance_theme_keys = {
            k for k, v in _recent_selected_counts.items() if v >= _RESONANCE_MIN_COUNT
        }
    return SalienceHistory(
        dwell_ticks=_dwell_ticks,
        recent_theme_counts=dict(_recent_selected_counts),
        resonance_theme_keys=set(resonance_theme_keys),
        first_seen_at=dict(_first_selected_at),
    )


def attention_broadcast_enabled() -> bool:
    return str(os.getenv(BROADCAST_FLAG, "false")).strip().lower() in _TRUTHY


def _node_salience(metadata: dict[str, Any]) -> tuple[float, str]:
    """Salience for the workspace competition, and which signal drove it."""

    def _f(key: str) -> float:
        try:
            return max(0.0, min(1.0, float(metadata.get(key) or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    pressure = _f("dynamic_pressure")
    prediction_error = _f("prediction_error")
    if prediction_error >= pressure:
        return prediction_error, "prediction_error"
    return pressure, "pressure"


def substrate_pressure_signals(
    nodes: Sequence[Any],
    *,
    min_salience: float = DEFAULT_MIN_SALIENCE,
    limit: int = DEFAULT_MAX_SIGNALS,
) -> list[AttentionSignalV1]:
    """Map graph nodes into workspace signals; tolerant, never raises per-node."""
    signals: list[AttentionSignalV1] = []
    for node in nodes:
        try:
            metadata = dict(getattr(node, "metadata", None) or {})
            salience, kind = _node_salience(metadata)
            if salience < min_salience:
                continue
            node_id = str(getattr(node, "node_id", "") or "")
            label = compact(str(getattr(node, "label", "") or node_id), 120)
            if not label:
                continue
            confidence = 0.6
            node_signals = getattr(node, "signals", None)
            if node_signals is not None:
                try:
                    confidence = max(0.0, min(1.0, float(node_signals.confidence)))
                except (AttributeError, TypeError, ValueError):
                    pass
            signals.append(
                AttentionSignalV1(
                    signal_id=stable_id("substrate-signal", f"{node_id}|{kind}"),
                    source="substrate_broadcast",
                    target_text=label,
                    target_type_hint="anomaly" if kind == "prediction_error" else "concept",
                    signal_kind=f"substrate_{kind}",
                    salience=salience,
                    confidence=confidence,
                    evidence_refs=[node_id] if node_id else [],
                    provenance={"detector": "substrate_pressure", "signal_driver": kind},
                )
            )
        except Exception:
            continue
    signals.sort(key=lambda s: s.salience, reverse=True)
    return signals[: max(1, limit)]


def build_substrate_attention_frame(
    *,
    nodes: Sequence[Any],
    belief_lineage: list[str] | None = None,
    min_salience: float = DEFAULT_MIN_SALIENCE,
    max_signals: int = DEFAULT_MAX_SIGNALS,
    max_open: int = 5,
    now: datetime | None = None,
) -> AttentionFrameV1:
    """One workspace competition over the substrate graph; always one winner.

    Same pipeline as the chat-scoped ``build_attention_frame`` but with empty
    chat context and ``max_asks=0``: high-pressure loops may score as asks and
    are then demoted to ``watch``, so the selected coalition is the top loop
    without any question generation.
    """
    from orion.substrate.attention.salience import habituation_enabled

    lineage = list(belief_lineage or [])
    # Resolve once, reuse everywhere below -- previously build_open_loops()
    # (and therefore compute_salience()/_recency()) never received `now` at
    # all and always defaulted to real wall-clock execution time internally,
    # decoupled from this frame's own generated_at. Harmless in production
    # (both were "real now" anyway) but meant recency could never be tested
    # deterministically, and masked the real bug this patch fixes (see
    # _first_selected_at above) during development.
    resolved_now = now or datetime.now(timezone.utc)
    signals = substrate_pressure_signals(nodes, min_salience=min_salience, limit=max_signals)
    merged = merge_signals(signals, limit=max_open * 3)
    history = _current_history() if habituation_enabled() else None
    open_loops = build_open_loops(
        signals=merged,
        ctx={},
        inputs={},
        belief_lineage=lineage,
        direct_turn=False,
        generic_reversal=False,
        stale_thread_active=False,
        max_open=max_open,
        history=history,
        now=resolved_now,
    )
    actions, selected, suppressions, deferred = select_actions(
        open_loops=open_loops,
        suppressions=[],
        min_ask=0.65,
        max_asks=0,
        generic_reversal=False,
        stale_thread_active=False,
    )
    frame = AttentionFrameV1(
        generated_at=resolved_now,
        open_loops=open_loops,
        live_unknowns=[loop.description for loop in open_loops if not loop.already_known],
        candidate_actions=actions,
        selected_action=selected,
        suppressions=suppressions,
        deferred_items=deferred[:max_open],
        debug={
            "enabled": True,
            "mode": "substrate_broadcast",
            "signal_count": len(signals),
            "merged_signal_count": len(merged),
            "min_salience": min_salience,
            "belief_lineage": lineage[:8],
        },
    )
    return _apply_voluntary_attention(frame)


logger = logging.getLogger(__name__)


def _apply_voluntary_attention(
    frame: AttentionFrameV1, agency_readiness: float = 1.0
) -> AttentionFrameV1:
    """Layer top-down goal bias onto the bottom-up frame (spec Step 2).

    Default-off and never-raises: when the flag is off or there is no active goal,
    the frame is returned byte-identical to bottom-up selection. When an active
    goal flips the winner, fill each loop's top_down_bias/combined_salience,
    re-point ``selected_action`` to the combined winner, and record the
    ``voluntary_override`` trace.
    """
    try:
        from orion.substrate.attention.top_down import (
            TopDownBiasCombiner,
            TopDownConfig,
            top_down_enabled,
        )
        from orion.substrate.attention.goal_context import get_active_goal
        from orion.substrate.attention.salience import salience_v2_enabled

        if not top_down_enabled():
            return frame
        # The bottom-up basis here is loop.salience (the v2 combined-salience
        # output). With salience_v2 OFF, select_actions ranks by a legacy weighted
        # sum, so our bottom-up winner could disagree with the real selection —
        # only layer top-down when v2 is the active selection basis (spec target).
        if not salience_v2_enabled():
            return frame
        goal = get_active_goal()
        if goal is None or not frame.open_loops:
            return frame
        bottom_up = {loop.id: float(loop.salience) for loop in frame.open_loops}
        result = TopDownBiasCombiner(TopDownConfig.from_env()).apply(
            goal=goal, loops=frame.open_loops, bottom_up=bottom_up,
            agency_readiness=agency_readiness,
        )
        for loop in frame.open_loops:
            score = result.per_loop.get(loop.id)
            if score is not None:
                loop.top_down_bias = score.top_down_bias
                loop.combined_salience = score.combined_salience
        frame.effort_budget_used = result.effort_used
        if result.override is not None:
            # Only record the override when the winner actually has an action to
            # re-point to — otherwise the frame would claim an override it can't
            # enact (chosen_loop_id disagreeing with selected_action).
            winner_action = next(
                (a for a in frame.candidate_actions
                 if getattr(a, "open_loop_id", None) == result.winner_loop_id),
                None,
            )
            if winner_action is not None:
                frame.voluntary_override = result.override
                frame.selected_action = winner_action
        return frame
    except Exception:
        logger.warning("voluntary_attention_apply_failed", exc_info=True)
        return frame


def broadcast_projection_from_frame(frame: AttentionFrameV1) -> AttentionBroadcastProjectionV1:
    global _current_active_coalition, _dwell_ticks, _coalition_history

    selected = frame.selected_action
    selected_loop = None
    if selected is not None and selected.open_loop_id:
        selected_loop = next(
            (loop for loop in frame.open_loops if loop.id == selected.open_loop_id), None
        )

    attended_node_ids = list(selected_loop.source_refs[:16]) if selected_loop is not None else []
    coalition = frozenset(attended_node_ids)

    # Hysteresis: 2-tick activation, 3-tick decay
    _coalition_history.append(coalition)

    # Soft activation: coalition must appear in 2+ of last 3 ticks to become active
    coalition_count = sum(1 for c in _coalition_history if c == coalition)
    if coalition_count >= 2:
        if _current_active_coalition != coalition:
            _current_active_coalition = coalition
            _dwell_ticks = 0  # reset on transition
            _transition_history.append(
                {
                    "at": frame.generated_at.isoformat(),
                    "event": "activated",
                    "size": len(coalition),
                }
            )
        _dwell_ticks += 1
    else:
        # Decay: active coalition has left the recent window entirely
        if _current_active_coalition is not None and all(
            c != _current_active_coalition for c in _coalition_history
        ):
            _transition_history.append(
                {
                    "at": frame.generated_at.isoformat(),
                    "event": "decayed",
                    "size": len(_current_active_coalition),
                }
            )
            _current_active_coalition = None
            _dwell_ticks = 0

    # Compute stability score from recent salience consistency
    # (simplified: high if dwell_ticks > 3, medium if transitioning, low if flickering)
    if _dwell_ticks > 3:
        stability_score = 0.9
    elif _dwell_ticks > 0:
        stability_score = 0.6
    else:
        stability_score = 0.3

    _record_selection(
        selected.open_loop_id if selected is not None else None, now=frame.generated_at
    )

    return AttentionBroadcastProjectionV1(
        generated_at=frame.generated_at,
        frame=frame,
        selected_action_type=selected.action_type if selected is not None else "none",
        selected_open_loop_id=selected.open_loop_id if selected is not None else None,
        selected_description=selected_loop.description if selected_loop is not None else None,
        attended_node_ids=attended_node_ids,
        dwell_ticks=_dwell_ticks,
        coalition_stability_score=stability_score,
        coalition_history=list(_transition_history),
    )
