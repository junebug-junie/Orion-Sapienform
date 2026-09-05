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
    VOLUNTARY_OVERRIDE_ABSENT_REASON_KEY,
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    AttentionSignalV1,
    VoluntaryOverrideAbsentReasonV1,
)
from orion.substrate.attention.common import compact, stable_id
from orion.substrate.attention.policy import select_actions
from orion.substrate.attention.scoring import build_open_loops, merge_signals
from orion.substrate.attention.verdicts import load_terminal_verdict_loop_ids

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

# 2026-07-31: `_current_dwelling_loop_id`, `_recent_selected_counts`,
# `_first_selected_at`, `_record_selection()`, `_current_history()`,
# `_MAX_TRACKED_THEMES`, and `_RESONANCE_MIN_COUNT` were deleted here
# (kill means kill, CLAUDE.md Sec 0A). Their only real purpose was feeding
# `orion.substrate.attention.salience.SalienceHistory`'s now-killed
# recency/dwell/habituation/recurrence terms -- `_current_dwelling_loop_id`
# in particular had become write-only (set here, read nowhere) the moment
# `_current_history()` went away. `_dwell_ticks`/`_coalition_history`/
# `_current_active_coalition`/`_transition_history` below are a SEPARATE,
# real, kept mechanism (coalition hysteresis/stability -- feeds
# `AttentionBroadcastProjectionV1.dwell_ticks`/`coalition_stability_score`/
# `coalition_history`), not touched by this kill.


def attention_broadcast_enabled() -> bool:
    return str(os.getenv(BROADCAST_FLAG, "false")).strip().lower() in _TRUTHY


def _node_salience(metadata: dict[str, Any]) -> tuple[float, str]:
    """Salience for the workspace competition, and which signal drove it.

    Magnitude always comes from ``dynamic_pressure`` -- the dynamics engine's
    single time-decayed pressure signal (seeded from ``prediction_error`` via
    ``prediction_error_pressure()``, or from drive/contradiction pressure;
    see ``orion/substrate/dynamics.py``/``pressure.py``). The raw
    ``prediction_error`` metadata value must NOT be read as a magnitude here:
    it never decays on its own, and since ``dynamic_pressure`` is derived
    from it via ``raw * weight(<1) * decay(<=1)``, racing the two always
    picks the raw value -- silently discarding the dynamics engine's decay
    on every tick (found live 2026-07-14: a node's raw prediction_error sat
    at 1.0, undecayed, for 6+ days while dynamic_pressure correctly decayed
    around it and was never actually used). Still surface whether a
    prediction-error seed exists at all, for the anomaly-vs-concept typing
    downstream -- that's a stable category, not the buggy part.

    Typing itself has the same staleness problem the magnitude fix above
    solved: the raw ``prediction_error`` metadata field never clears once
    set, so a node whose *current* ``dynamic_pressure`` is now driven
    entirely by an unrelated source (a drive seed, contradiction
    propagation) still gets typed "anomaly" forever off an old, no-longer
    relevant prediction-error value. ``SubstrateDynamicsEngine.tick()``
    persists ``metadata["dynamic_pressure_reason"]`` -- the actual driver of
    *this tick's* pressure value (``"prediction_error_seed"``,
    ``"prediction_error_propagation:{predicate}"``, ``"drive_seed"``,
    ``"drive_propagation:{predicate}"``, ``"contradiction_unresolved"``,
    ``"contradiction_involved"``, or ``"none"``) -- so typing is derived from
    that instead. Nodes that predate a dynamics tick (no reason key present
    at all, e.g. freshly materialized or in tests that model synthetic
    pre-tick state) fall back to the old presence-check behavior so this
    change does not regress typing for those.
    """

    def _f(key: str) -> float:
        try:
            return max(0.0, min(1.0, float(metadata.get(key) or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    pressure = _f("dynamic_pressure")
    reason = metadata.get("dynamic_pressure_reason")
    if reason is not None:
        kind = "prediction_error" if str(reason).startswith("prediction_error") else "pressure"
    else:
        kind = "prediction_error" if _f("prediction_error") > 0.0 else "pressure"
    return pressure, kind


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
                    evidence_refs=(
                        ([node_id] if node_id else [])
                        + [str(t) for t in metadata.get("contributing_turn_ids") or []]
                    ),
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
    lineage = list(belief_lineage or [])
    # generated_at is stamped from this same resolved value below, keeping
    # the frame's own timestamp internally consistent.
    resolved_now = now or datetime.now(timezone.utc)
    signals = substrate_pressure_signals(nodes, min_salience=min_salience, limit=max_signals)
    merged = merge_signals(signals, limit=max_open * 3)
    open_loops = build_open_loops(
        signals=merged,
        ctx={},
        inputs={},
        belief_lineage=lineage,
        direct_turn=False,
        generic_reversal=False,
        stale_thread_active=False,
        max_open=max_open,
        # Substrate broadcast is rung-3's continuous re-broadcast, the exact
        # path a resolved/dismissed loop was found live still winning
        # indefinitely (2026-07-14 investigation). Excludes those loops
        # entirely rather than down-weighting them -- see
        # orion.substrate.attention.verdicts. Not wired into the chat-scoped
        # build_open_loops() caller (attention_frame.py): that path is
        # per-turn and ephemeral, and this fix is scoped to the workspace
        # competition that actually dominated indefinitely.
        #
        # `now=resolved_now` threads this tick's own timestamp into the TTL
        # check (verdicts.py's VERDICT_EXCLUSION_TTL_HOURS) instead of letting
        # it read the wall clock separately -- keeps "now" internally
        # consistent with this frame's own generated_at, and this was the
        # exact seam a resolved-forever exclusion was found live still
        # blocking a loop 2026-08-19 with no way to lapse.
        verdict_lookup=lambda ids: load_terminal_verdict_loop_ids(ids, now=resolved_now),
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


def _classify_override_absence(
    result: "TopDownResult",
    goal: "GoalContext",
    loops: "list[OpenLoopV1]",
) -> VoluntaryOverrideAbsentReasonV1:
    """Which flavour of "no override" this was. Caller guarantees a goal ran.

    Decided from the GOAL/LOOP JOIN and the effort actually applied -- never
    from "max bias is 0.0". Code review 2026-09-05 showed that inference is
    three facts wearing one face: bias is ``priority * relevance``, so it is
    also 0.0 when the goal's target IS competing but ``priority == 0.0``, and
    when the goal carries no target at all. Both were reproduced live, and both
    would have narrated as "the goal and the competition were about different
    things" -- false, and landing directly in the overlap number this patch
    exists to make trustworthy.

    Cases, each with a different owner:

    - ``goal_had_no_target``  -- the goal never said what it was about. A
      producer defect, not a fact about attention.
    - ``goal_matched_no_loop`` -- target set, but nothing competing carries it.
      A routing/overlap problem, upstream of this file.
    - ``goal_pushed_nothing`` -- the goal's target IS competing but no bias was
      actually applied to it (zero priority, or the effort budget let nothing
      through). Reads on ``applied_bias``, not raw bias, so a tick with
      ``effort_used == 0`` can never narrate as "pushed and lost".
    - ``goal_target_already_winning`` -- the bottom-up winner is one of the
      loops the goal wanted. Nothing to override; the goal already had it.
    - ``bias_did_not_flip_winner`` -- bias landed on a goal loop that then lost.
      The only genuine competitive defeat, and the only one that is evidence
      against Orion's agency.

    Relevance is read through ``relevance()`` itself rather than re-deriving
    the join here, so the two can never drift apart.

    Never raises. An unreadable result returns ``absence_unclassified`` -- it
    must not fall through to any of the narrow values above, all of which
    assert a specific cause.
    """
    try:
        from orion.substrate.attention.top_down import relevance

        target = getattr(goal, "target_id", None)
        if not target:
            return "goal_had_no_target"

        per_loop = result.per_loop or {}
        if not per_loop:
            return "absence_unclassified"

        matched = [lp.id for lp in (loops or []) if relevance(goal, lp) > 0.0]
        if not matched:
            return "goal_matched_no_loop"

        # applied_bias, not top_down_bias: the effort budget is what decides
        # whether anything was really pushed.
        applied = max(
            (per_loop[lid].applied_bias for lid in matched if lid in per_loop),
            default=0.0,
        )
        if applied <= 0.0:
            return "goal_pushed_nothing"

        bu_winner = result.bottom_up_winner_loop_id
        if bu_winner is None:
            return "absence_unclassified"
        # Not a judgment call: relevance() is binary, so `matched` is exactly
        # the set of loops carrying the goal's target. This is the precise
        # statement "the bottom-up winner is one of the loops the goal wanted".
        return (
            "goal_target_already_winning"
            if bu_winner in matched
            else "bias_did_not_flip_winner"
        )
    except Exception:
        return "absence_unclassified"


def _set_override_absent_reason(
    frame: AttentionFrameV1, reason: VoluntaryOverrideAbsentReasonV1 | None
) -> None:
    """Record (or clear) why no voluntary override fired, in `frame.debug`.

    Not a typed field on `AttentionFrameV1` on purpose -- that model sets
    `extra="forbid"` and crosses the bus nested inside `HubAssociationBundleV1`,
    so a new field there breaks any consumer not redeployed in the same window
    (one of them silently). See the note on
    `VOLUNTARY_OVERRIDE_ABSENT_REASON_KEY` in `orion/schemas/attention_frame.py`.
    The typed contract lives on `AttentionSelfModelV1`, which stores it.

    Clearing (`reason=None`) removes the key rather than storing a None, so a
    frame that fired an override carries no absence claim at all.
    """
    if reason is None:
        frame.debug.pop(VOLUNTARY_OVERRIDE_ABSENT_REASON_KEY, None)
    else:
        frame.debug[VOLUNTARY_OVERRIDE_ABSENT_REASON_KEY] = reason


def _apply_voluntary_attention(
    frame: AttentionFrameV1, agency_readiness: float = 1.0
) -> AttentionFrameV1:
    """Layer top-down goal bias onto the bottom-up frame (spec Step 2).

    Default-off and never-raises: when the flag is off or there is no active goal,
    selection is unchanged from bottom-up. When an active goal flips the winner,
    fill each loop's top_down_bias/combined_salience, re-point
    ``selected_action`` to the combined winner, and record the
    ``voluntary_override`` trace.

    **2026-09-04:** every exit now also records
    ``frame.voluntary_override_absent_reason``, so a frame without an override
    says *which* branch produced that. This function is therefore no longer
    byte-identical to its input on the no-op paths -- it sets exactly that one
    field and nothing else. The previous wording ("returned byte-identical")
    is retained here only to name what changed. The first three exits below are
    mutually indistinguishable downstream (all leave ``effort_budget_used`` at
    0.0 with no per-loop bias), which is why the reason is recorded here rather
    than inferred by a reader.
    """
    try:
        from orion.substrate.attention.top_down import (
            TopDownBiasCombiner,
            TopDownConfig,
            top_down_enabled,
        )
        from orion.substrate.attention.goal_context import get_active_goal

        if not top_down_enabled():
            _set_override_absent_reason(frame, "top_down_disabled")
            return frame
        # 2026-07-31: the `salience_v2_enabled()` gate that used to sit here
        # was removed. Its stated rationale ("with salience_v2 OFF,
        # select_actions ranks by a legacy weighted sum, so our bottom-up
        # winner could disagree with the real selection") no longer applies
        # -- score_loop() has exactly one formula now (loop.salience,
        # unconditionally; see scoring.py), so select_actions' real
        # selection basis and this function's bottom_up basis can never
        # disagree regardless of that flag's value. Keeping a check whose
        # own justification is gone would be exactly the kind of zombie
        # gate this program's kill-means-kill discipline exists to remove.
        goal = get_active_goal()
        # Split from a single `goal is None or not frame.open_loops` guard on
        # 2026-09-04: collapsed, the two cases were indistinguishable in the
        # record, and they mean very different things -- "Orion wanted nothing"
        # versus "there was nothing to want".
        if goal is None:
            _set_override_absent_reason(frame, "no_active_goal")
            return frame
        if not frame.open_loops:
            _set_override_absent_reason(frame, "no_open_loops")
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
        if result.failed:
            # Rule 8 swallowed an exception inside the combiner and handed back
            # a pure-bottom-up fallback. That fallback is byte-identical to a
            # real "goal was relevant to nothing" outcome -- populated per_loop,
            # every bias 0.0, a real winner -- so without this branch a crash
            # would be recorded as a confident causal claim about goal quality.
            # Checked BEFORE the override test because a failed result always
            # carries override=None and would otherwise fall through below.
            _set_override_absent_reason(frame, "combiner_error")
        elif result.override is None:
            # The combiner ran and top-down bias did not change the winner
            # (top_down.py Rule 6). Distinct from the guards above: a goal WAS
            # present. Which of three very different things happened is decided
            # here -- collapsing them into one string made the override rate
            # uninterpretable (see VoluntaryOverrideAbsentReasonV1).
            _set_override_absent_reason(
                frame, _classify_override_absence(result, goal, frame.open_loops)
            )
        else:
            # Only record the override when the winner actually has an action to
            # re-point to — otherwise the frame would claim an override it can't
            # enact (chosen_loop_id disagreeing with selected_action).
            winner_action = next(
                (a for a in frame.candidate_actions
                 if getattr(a, "open_loop_id", None) == result.winner_loop_id),
                None,
            )
            if winner_action is None:
                _set_override_absent_reason(frame, "winner_had_no_action")
            else:
                frame.voluntary_override = result.override
                frame.selected_action = winner_action
                # Load-bearing, not defensive: `frame.debug` is caller-supplied
                # and survives, so a frame arriving with a reason already in it
                # would otherwise keep it and make a successful override look
                # refused. Pinned by a test that passes exactly such a frame.
                _set_override_absent_reason(frame, None)
        return frame
    except Exception:
        logger.warning("voluntary_attention_apply_failed", exc_info=True)
        # A swallowed defect must not read as a normal quiet tick. Plain
        # setattr on a declared pydantic field (no validate_assignment on this
        # model), so this cannot itself raise and break the never-raises
        # contract this handler exists to keep.
        _set_override_absent_reason(frame, "combiner_error")
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
