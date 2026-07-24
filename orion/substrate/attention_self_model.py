"""AST/HOT attention self-model reducer — rung 4 (read-only, Phase 1).

Pure function, no I/O. Takes the already-parsed real inputs and returns one
`AttentionSelfModelV1` answering: what's currently salient, what was last
dispatched, *why* (top-down goal bias vs. pure bottom-up salience — the
"aboutness" claim AST/HOT instrumentation needs to make honestly), how
confident, and what's predicted to shift next.

Mirrors `scripts/analysis/measure_origination_gate.py`'s separation of a pure
replay/summary layer from an I/O layer, for the same testability reason that
script documents: this function is exercised directly with synthetic
fixtures, no DB or bus involved, by `orion/substrate/tests/
test_attention_self_model.py`.

Read-only. This module has no bus consumer/producer wiring — see
`docs/superpowers/specs/2026-07-18-objective-3-consciousness-scaffolded-roadmap-design.md`
Phase 1: the reducer must be measured against real historical data
(`scripts/analysis/measure_ast_hot_reducer.py`) before anything downstream
consumes it, matching the program charter's own "measure before minting"
process rule (sec 7).

**2026-07-23: `SelfStateV1` input removed, replaced with Active-Inference
substrate.** `predicted_shift`/`confidence`'s `self_state` fallback read a
producer that no longer exists (`orion/self_state/` was deleted, PR #1266;
`orion-athena-self-state-runtime` was confirmed stopped same-day per
`docs/superpowers/specs/2026-07-22-l6-self-model-ast-hot-active-inference-
design.md`'s Missing Question 1). Per that design doc's items 3/4: confidence
is now the inverse of aggregate prediction-error volatility across the five
real, live Predictive-Processing domains (execution/transport/biometrics/
chat/route -- `orion/substrate/prediction_error.py`), and predicted_shift now
names whichever domain's prediction-error is trending fastest, instead of a
hand-tuned `SelfStateV1` dimension. Both are supplied by the caller as
already-computed dicts (`prediction_error_by_domain`,
`prediction_error_trend_by_domain`) -- this function stays a pure formatter/
aggregator over them, matching its own "no I/O, no store coupling" design for
every other input here. `self_state_present`/`self_state` are gone from both
this signature and `AttentionSelfModelV1` -- not deprecated in place, removed
(this repo's own "kill means kill" convention: no fallback to the thing being
killed).
"""

from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.attention_self_model import AttentionSelfModelV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1

# 2x the live ORION_ATTENTION_BROADCAST_INTERVAL_SEC default (30s, confirmed
# live 2026-07-18 via `docker exec orion-athena-substrate-runtime env`) —
# same 2x-tick-interval convention `orion/substrate/felt_state_reader.py`
# already uses for its own staleness gates (curiosity_signals, reverie).
DEFAULT_BROADCAST_STALE_THRESHOLD_SEC = 60.0

# The real Active-Inference prediction_error domains used for the
# UNCONDITIONAL `prediction_error_confidence` field (see
# `_unconditional_prediction_error_confidence` below). Deliberately excludes
# `transport` -- confirmed live 2026-07-24
# (`docs/notes/2026-07-24-attention-reason-branch-starvation-finding.md`):
# `transport_prediction_error` reads exactly 0.0 for 100% of a real 8h
# window (`BUS_OBSERVER_STREAMS` only watches 2 real Redis Streams), so
# averaging it in with the other four real, varying domains systematically
# inflates confidence by a fixed, mechanical amount every tick -- harmless
# while this formula fired on 0.04% of ticks, not harmless once it fires on
# nearly all of them. Re-include `transport` here once a real, bounded,
# prediction-error-shaped transport signal exists -- tracked as the
# reversal condition against PR #1323's bus-synaptic-graph work, not before.
# This constant does NOT affect the pre-existing branch-gated
# `confidence`/`confidence_basis` fields below, which keep their original
# (unfiltered, all-domains) formula unchanged -- additive only.
ACTIVE_INFERENCE_DOMAINS = frozenset({"execution", "biometrics", "chat", "route"})


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _aggregate_prediction_error_confidence(
    prediction_error_by_domain: dict[str, float],
) -> tuple[float, str]:
    """mean(), not max(): item 1/2 above already surface the single loudest
    domain via dynamic_pressure-driven attention selection (see
    `orion/substrate/attention_broadcast.py`), so a max()-based confidence
    here would be a redundant, same-sensor restatement of what attention
    selection already answers -- CLAUDE.md's metric-quality-gate
    independence check. mean() answers a genuinely different question
    (overall systemic stability across every domain right now, not just the
    worst one) at the cost of being diluted by domains with little real
    signal. Confirmed against live data (2026-07-23, 6h window,
    `substrate_field_state`): biometrics carries almost all the real
    variance (mean=0.037, max=0.62); execution/chat/route are real but tiny
    (means ~1e-5); transport reads exactly 0.0 for the entire window (its
    documented single-queue narrow scope -- see
    `services/orion-substrate-runtime/README.md`'s "transport domain is one
    queue" note). This mean() is real and non-degenerate (it visibly moves
    with biometrics activity) but heavily damped by the other four domains'
    near-silence -- an honest, disclosed limitation, not a hidden one.
    """
    values = list(prediction_error_by_domain.values())
    mean_error = sum(values) / len(values)
    # Symmetric clamp: every real producer in prediction_error.py only ever emits
    # min(1.0, ...) (non-negative), but this is a general-purpose function over an
    # arbitrary caller-supplied dict -- an out-of-range (e.g. negative) input must
    # not silently produce confidence > 1.0. Found in code review (live-reproduced:
    # a single -0.5 domain value produced confidence=1.5, outside
    # AttentionSelfModelV1.confidence's own declared [0,1] range, which Pydantic v2
    # does not re-check on plain attribute assignment).
    mean_error = max(0.0, min(1.0, mean_error))
    confidence = 1.0 - mean_error
    domains = ", ".join(sorted(prediction_error_by_domain))
    basis = (
        f"1 - mean(prediction_error) across {len(values)} domains ({domains}) "
        "(broadcast lane stale/absent)"
    )
    return confidence, basis


def _unconditional_prediction_error_confidence(
    prediction_error_by_domain: dict[str, float] | None,
) -> tuple[float | None, str]:
    """Unconditional counterpart to `_aggregate_prediction_error_confidence`,
    populating the new `prediction_error_confidence`/`prediction_error_confidence_basis`
    fields regardless of which `attention_reason` branch wins -- mirrors
    `predicted_shift`'s existing unconditional positioning (computed once,
    before the elif branching, not gated on `field_salience_only` firing).

    Restricted to `ACTIVE_INFERENCE_DOMAINS` (excludes the confirmed-dead
    `transport` domain -- see that constant's docstring for the live-data
    finding behind this exclusion). Returns `(None, "")` when no domain in
    that set has data, matching every other "honestly absent" convention in
    this reducer (never invents a confidence value from nothing).

    Does NOT read or affect the pre-existing branch-gated
    `confidence`/`confidence_basis` fields -- those keep computing their
    original (unfiltered, all-domains) formula via
    `_aggregate_prediction_error_confidence` unchanged. This function exists
    so a real, non-diluted Active-Inference confidence signal is available
    on (almost) every tick instead of only the ~0.04% of ticks where the
    broadcast/GWT-dispatch lane happens to be absent or stale -- see
    `docs/notes/2026-07-24-attention-reason-branch-starvation-finding.md`.
    """
    if not prediction_error_by_domain:
        return None, ""
    filtered = {
        domain: value
        for domain, value in prediction_error_by_domain.items()
        if domain in ACTIVE_INFERENCE_DOMAINS
    }
    if not filtered:
        return None, ""
    confidence, _ = _aggregate_prediction_error_confidence(filtered)
    domains = ", ".join(sorted(filtered))
    basis = (
        f"1 - mean(prediction_error) across {len(filtered)} ACTIVE_INFERENCE_DOMAINS "
        f"({domains}) (unconditional -- computed regardless of attention_reason branch)"
    )
    return confidence, basis


def reduce_attention_self_model(
    broadcast: AttentionBroadcastProjectionV1 | None,
    field_frame: FieldAttentionFrameV1 | None,
    *,
    now: datetime | None = None,
    broadcast_stale_threshold_sec: float = DEFAULT_BROADCAST_STALE_THRESHOLD_SEC,
    harness_closure_signal: dict | None = None,
    prediction_error_by_domain: dict[str, float] | None = None,
    prediction_error_trend_by_domain: dict[str, float] | None = None,
) -> AttentionSelfModelV1:
    """Unify the GWT-dispatch lane and the general field lane into one
    inspectable AST/HOT self-model. Never raises: any input may be `None`
    (partial data honestly represented, not a crash).

    `now` is the reference tick this self-model is being generated *for* —
    normally the field-lane tick's own `generated_at` (the highest-frequency
    real signal, so it drives the replay/live cadence per the Phase 1 design).
    Falls back to `broadcast.generated_at`, then wall-clock
    `datetime.now(timezone.utc)` if neither input is present.

    `harness_closure_signal` is an optional, intentionally-minimal plain dict
    -- `{"prediction_error": float, "contributing_turn_ids": list[str]}` --
    built by the caller from the `node:substrate.harness_closure` FalkorDB
    node's now-durable metadata (`orion/substrate/falkor_codec.py`'s
    `contributing_turn_ids_json`, PR #1205 + this patch). Deliberately not a
    raw graph node or a `falkor_codec` import: this reducer stays decoupled
    from the store layer, matching every other input here (already-parsed
    Pydantic/plain data, not raw store objects). When present and non-trivial
    (`prediction_error > 0.0` and `contributing_turn_ids` non-empty), it only
    enriches the `field_salience_only` branch's `reason_narrative` with a
    clause naming sustained prediction-error surprise -- it never changes
    `attention_reason` itself, and every other branch (`top_down_override`,
    `bottom_up_salience`) ignores it entirely. Omitting this argument (the
    default) reproduces today's narrative byte-for-byte.

    `prediction_error_by_domain` is an optional, caller-supplied snapshot of
    the current raw `prediction_error` value for each of the five real
    Predictive-Processing domains (`{"execution": 0.0001, "transport": 0.0,
    "biometrics": 0.0459, "chat": ..., "route": ...}` -- keys are whatever
    the caller has data for; missing domains are simply absent, not
    defaulted to 0.0). Drives `confidence` in the `field_salience_only`
    branch (see `_aggregate_prediction_error_confidence`). Omitting this
    argument falls back to the pre-existing
    `field_attention_frame.dominant_targets[].confidence_score` mean.

    `prediction_error_trend_by_domain` is an optional, caller-supplied dict
    of each domain's recent prediction-error *trend* (already computed
    upstream -- see `scripts/analysis/measure_ast_hot_reducer.py::
    compute_prediction_error_trend()` for the real formula and the live-data
    validation behind its sign convention; this function does no
    time-series math of its own, matching its "no I/O, format what's given"
    design). Drives `predicted_shift`: whichever domain has the
    largest-magnitude trend is named as the honest candidate for "what
    surprises me next" -- mirroring the argmax-over-a-delta-dict shape the
    old `self_state.dimension_trajectory` branch used, just over real
    Active-Inference domains instead of a dead `SelfStateV1` dimension.
    """

    reference_ts = (
        now
        or (field_frame.generated_at if field_frame is not None else None)
        or (broadcast.generated_at if broadcast is not None else None)
        or datetime.now(timezone.utc)
    )
    if reference_ts.tzinfo is None:
        reference_ts = reference_ts.replace(tzinfo=timezone.utc)

    model = AttentionSelfModelV1(generated_at=reference_ts)

    # --- Field lane: what's currently salient -------------------------------
    if field_frame is not None:
        model.field_lane_present = True
        model.field_overall_salience = _round_or_none(field_frame.overall_salience)
        model.field_salient_target_ids = [t.target_id for t in field_frame.dominant_targets]

    # --- Predicted shift: whichever domain's prediction-error is trending --
    # fastest right now (Active Inference), replacing the dead
    # self_state.dimension_trajectory fallback. Computed unconditionally
    # (not gated on which attention_reason branch fires below), matching the
    # old self_state block's own positioning.
    if prediction_error_trend_by_domain:
        top_domain: str | None = None
        top_trend_val = 0.0
        for domain, trend in prediction_error_trend_by_domain.items():
            if abs(trend) > abs(top_trend_val):
                top_domain, top_trend_val = domain, trend
        if top_domain is not None and top_trend_val != 0.0:
            direction = "rising" if top_trend_val > 0 else "falling"
            model.predicted_shift = (
                f"{top_domain} prediction-error {direction} "
                f"(trend={top_trend_val:+.4f} over recent window)"
            )
            model.predicted_shift_basis = (
                "prediction_error_trend_by_domain (Active Inference, argmax by |trend|)"
            )

    # --- Prediction-error confidence (unconditional, ACTIVE_INFERENCE_DOMAINS
    # only): mirrors predicted_shift's positioning above -- computed regardless
    # of which attention_reason branch wins below. Additive only: does not
    # read or change the pre-existing branch-gated confidence/confidence_basis
    # fields set inside the elif branching further down.
    pe_confidence, pe_confidence_basis = _unconditional_prediction_error_confidence(
        prediction_error_by_domain
    )
    if pe_confidence is not None:
        model.prediction_error_confidence = _round_or_none(pe_confidence)
        model.prediction_error_confidence_basis = pe_confidence_basis

    # --- Broadcast lane: what was last dispatched + cadence-mismatch state -
    broadcast_age_sec: float | None = None
    broadcast_stale = True
    if broadcast is not None:
        model.broadcast_lane_present = True
        model.broadcast_selected_action_type = broadcast.selected_action_type
        model.broadcast_selected_open_loop_id = broadcast.selected_open_loop_id
        model.broadcast_selected_description = broadcast.selected_description
        model.broadcast_attended_node_ids = list(broadcast.attended_node_ids)

        b_ts = broadcast.generated_at
        if b_ts.tzinfo is None:
            b_ts = b_ts.replace(tzinfo=timezone.utc)
        broadcast_age_sec = (reference_ts - b_ts).total_seconds()
        # A negative age means the broadcast snapshot is *newer* than the
        # reference tick (the only real-world case this occurs today: the
        # broadcast lane is a singleton upsert row — see module docstring —
        # so a historical replay tick that predates the single available
        # snapshot has, honestly, no broadcast data at all for that moment;
        # treat it as absent rather than guessing what an earlier snapshot
        # might have looked like).
        if broadcast_age_sec < 0:
            model.broadcast_lane_present = False
            model.broadcast_selected_action_type = None
            model.broadcast_selected_open_loop_id = None
            model.broadcast_selected_description = None
            model.broadcast_attended_node_ids = []
            broadcast_age_sec = None
            broadcast_stale = True
        else:
            broadcast_stale = broadcast_age_sec > broadcast_stale_threshold_sec

    model.broadcast_lane_age_sec = _round_or_none(broadcast_age_sec, 3)
    model.broadcast_lane_stale = broadcast_stale

    # --- Why: branch explicitly on voluntary_override -----------------------
    override = None
    if (
        broadcast is not None
        and model.broadcast_lane_present
        and not broadcast_stale
        and broadcast.frame is not None
    ):
        override = broadcast.frame.voluntary_override

    if override is not None:
        model.attention_reason = "top_down_override"
        model.voluntary_override = override
        model.confidence = _round_or_none(broadcast.coalition_stability_score)
        model.confidence_basis = "broadcast.coalition_stability_score (fresh, top-down)"
        model.reason_narrative = (
            f"Top-down goal bias flipped the winner: loop '{override.chosen_loop_id}' "
            f"(bottom_up={override.chosen_bottom_up:.2f}) beat '{override.beat_loop_id}' "
            f"(bottom_up={override.beat_bottom_up:.2f}) via applied_bias="
            f"{override.applied_bias:.2f}, effort_spent={override.effort_spent:.2f}, "
            f"goal_artifact_id={override.goal_artifact_id!r}, "
            f"goal_drive_origin={override.goal_drive_origin!r}."
        )
    elif model.broadcast_lane_present and not broadcast_stale:
        model.attention_reason = "bottom_up_salience"
        model.confidence = _round_or_none(broadcast.coalition_stability_score)
        model.confidence_basis = "broadcast.coalition_stability_score (fresh, bottom-up)"
        model.reason_narrative = (
            f"Pure bottom-up dispatch: '{model.broadcast_selected_open_loop_id}' "
            f"selected by salience alone; no active goal override at this tick."
        )
    elif model.field_lane_present or prediction_error_by_domain or prediction_error_trend_by_domain:
        model.attention_reason = "field_salience_only"
        if prediction_error_by_domain:
            confidence, basis = _aggregate_prediction_error_confidence(prediction_error_by_domain)
            model.confidence = _round_or_none(confidence)
            model.confidence_basis = basis
        elif field_frame is not None and field_frame.dominant_targets:
            scores = [t.confidence_score for t in field_frame.dominant_targets]
            model.confidence = _round_or_none(sum(scores) / len(scores))
            model.confidence_basis = (
                "mean(field_attention_frame.dominant_targets[].confidence_score) "
                "(broadcast lane stale/absent)"
            )
        stale_note = (
            "no new GWT-dispatch-lane activity since last frame"
            if model.broadcast_lane_present
            else "no GWT-dispatch-lane data available"
        )
        n_targets = len(model.field_salient_target_ids)
        narrative = (
            f"{stale_note}; reading field-lane salience only "
            f"({n_targets} dominant target(s), "
            f"overall_salience={model.field_overall_salience})"
        )
        harness_error = 0.0
        harness_turn_ids: list = []
        if harness_closure_signal:
            try:
                harness_error = float(harness_closure_signal.get("prediction_error") or 0.0)
            except (TypeError, ValueError):
                harness_error = 0.0
            harness_turn_ids = list(harness_closure_signal.get("contributing_turn_ids") or [])
        if harness_error > 0.0 and harness_turn_ids:
            narrative += (
                f"; sustained prediction-error surprise across {len(harness_turn_ids)} "
                f"recent turn(s) (current magnitude={harness_error:.2f})"
            )
        model.reason_narrative = narrative + "."
    else:
        model.attention_reason = "no_data"
        model.reason_narrative = "No attention data available from either lane."

    return model
