from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine

from orion.substrate.pressure import PressureConfig

# Reuses the same horizon `orion/substrate/endogenous_curiosity.py`'s
# `_prediction_error_staleness_decay()` applies to this identical undecayed raw
# field, rather than inventing a fresh number.
STALENESS_HORIZON_SEC = PressureConfig().prediction_error_decay_horizon_seconds


def latest_bus_synaptic_prediction_error(engine: Engine) -> float | None:
    """Real, live surprise/epistemic-value signal, shared by every emitter that wants
    something better than a hardcoded or success/fail-proxy `surprise` value.

    Reads `orion/substrate/prediction_error.py::bus_synaptic_prediction_error()`'s
    already-written output off `substrate_field_state` (`node:substrate.bus_synaptic`'s
    `prediction_error` field) -- populated by `orion-field-digester`'s
    `state_deltas.py` (`target_kind == "prediction_signal"`, `mode="replace"`), not
    directly by `orion-substrate-runtime`, which only publishes the perturbation that
    triggers that write. Chosen over the other four domains (execution/biometrics/chat/
    route) because it is generic across the whole bus mesh rather than scoped to one
    reducer, and because it already went through a real live-data sanity check (PR #1391
    fixed a ~0.27 calm-floor bias found by recovering real numbers, not by eyeballing
    variance -- see that PR and
    docs/superpowers/specs/2026-07-26-transport-domain-retirement-bus-synaptic-successor-design.md).
    Edge-count cold-start filtering (`count < ~5` unreliable per
    `services/orion-bus-mirror/README.md`) already happens upstream, in
    `orion-substrate-runtime`'s `_bus_synaptic_tick`, before this value is ever written --
    not re-done here.

    **Staleness guard**: `prediction_error` was deliberately made an undecayed raw
    snapshot (removed from `NODE_DECAY_CHANNELS` 2026-07-26, per
    `services/orion-field-digester/app/digestion/decay.py`) -- it sits at whatever was
    last written until the next perturbation, and does not self-correct if the writing
    tick stalls or the producing service goes down. This function's callers have already
    been burned twice by trusting an undecayed/mis-decayed field at face value
    (`node:substrate.route` decaying unopposed for 48h; the bus_synaptic calm-floor
    bug). `orion/substrate/endogenous_curiosity.py::_prediction_error_staleness_decay()`
    guards the same raw field for its own consumer by age-decaying it; this function
    instead treats a row older than `STALENESS_HORIZON_SEC` as unavailable (`None`)
    rather than partially decaying it -- a single scalar has no good "partially stale"
    representation, and honest absence is the same choice #1379's design doc already
    made for this field.

    Extracted 2026-07-28 from `services/orion-execution-dispatch-runtime/app/store.py`
    (its first consumer) so a second emitter (`orion-spark-concept-induction`) reuses
    the exact same query and staleness logic instead of a second, divergence-prone copy
    -- the "reuse the live pipeline, don't parallel it" rule.

    Returns `None` (not `0.0`) when the node is absent, stale, or unparseable, so the
    caller can tell "no real signal available" apart from "genuinely calm" -- collapsing
    those would recreate the same degenerate-zero failure this function exists to fix.
    """
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT field_json -> 'node_vectors' -> 'node:substrate.bus_synaptic'
                       ->> 'prediction_error' AS value,
                       generated_at
                FROM substrate_field_state
                ORDER BY generated_at DESC
                LIMIT 1
                """
            )
        ).mappings().first()
    if not row or row["value"] is None:
        return None
    generated_at = row["generated_at"]
    if generated_at is not None:
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - generated_at).total_seconds()
        if age_seconds > STALENESS_HORIZON_SEC:
            return None
    try:
        return float(row["value"])
    except (TypeError, ValueError):
        return None
