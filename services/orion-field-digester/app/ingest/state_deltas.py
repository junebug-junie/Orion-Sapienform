from __future__ import annotations

import math
from dataclasses import dataclass

from orion.schemas.state_delta import StateDeltaV1

GPU_NODES = {"atlas", "circe"}


@dataclass(frozen=True)
class Perturbation:
    node_id: str
    channel: str
    intensity: float
    label: str
    mode: str = "add"  # "add" accumulates; "replace" sets the channel to exactly intensity


def _node_key(raw: str) -> str:
    nid = raw.strip().lower()
    return nid if nid.startswith("node:") else f"node:{nid}"


def delta_to_perturbations(delta: StateDeltaV1) -> list[Perturbation]:
    if delta.operation == "noop":
        return []
    after = delta.after or {}
    node_id = _node_key(str(after.get("node_id") or delta.target_id))
    out: list[Perturbation] = []

    if delta.target_kind == "active_node_pressure":
        score = float(after.get("pressure_score", 0.0))
        pressures = list(after.get("active_pressures") or [])
        if "strain" in pressures:
            channel = "gpu_pressure" if node_id.replace("node:", "") in GPU_NODES else "cpu_pressure"
            out.append(Perturbation(node_id=node_id, channel=channel, intensity=score, label=delta.delta_id))
        if "availability" in pressures:
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="availability",
                    intensity=max(0.0, 1.0 - min(1.0, score + 0.2)),
                    label=delta.delta_id,
                    # mode="replace" (2026-07-17 fix, live post-deploy finding):
                    # this intensity is a full current-availability estimate
                    # recomputed fresh from this tick's pressure_score, not an
                    # incremental delta -- same "current reading" shape as
                    # bus_health/delivery_confidence/execution_load elsewhere
                    # in this file. Without mode="replace",
                    # apply_perturbations() special-cases channel=="availability"
                    # to floor-only (min(current, intensity)): can decrease but
                    # never recover, even when a LATER event reports genuinely
                    # improved conditions (a higher intensity just gets min()'d
                    # away against the old low value). Confirmed live:
                    # node:atlas's availability was pinned at exactly 0.0 with
                    # staleness ~0 and gpu_pressure=0.72 (fresh, plausible
                    # telemetry) -- not a real ongoing outage, a one-way ratchet
                    # with the same shape as expected_offline_suppression's
                    # original bug (PR #1109), just via min()-floor instead of
                    # add-no-decay.
                    mode="replace",
                )
            )
        if delta.operation == "suppress":
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="expected_offline_suppression",
                    intensity=1.0,
                    label=delta.delta_id,
                )
            )

    if delta.target_kind == "node_biometrics":
        hints = dict(after.get("pressure_hints") or {})
        # mode="replace" (2026-07-17 fix, same reasoning as memory_pressure/
        # thermal_pressure/disk_pressure below): live corpus verification
        # (/mnt/telemetry/field_channels/corpus/field_channels.jsonl,
        # 133,968 rows / 2026-07-13..17) confirmed gpu_pressure/cpu_pressure
        # were hitting their post-decay ceiling of exactly
        # BIOMETRICS_FIELD_DECAY_RATE (0.92) in 16.60%/12.98% of all rows --
        # vs. 0.01%/0.00% for execution_load/execution_friction, which
        # already use mode="replace" -- and were bit-identical to each other
        # in 60.60% of rows despite deriving from independent strain/gpu
        # hints. Both signatures match repeated re-saturation from add-mode
        # duplicate deltas (see the memory/thermal/disk-pressure comment
        # below for the full per-trace fan-out mechanism), not real,
        # independent CPU/GPU utilization.
        if "gpu" in hints:
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="gpu_pressure",
                    intensity=float(hints["gpu"]),
                    label=delta.delta_id,
                    mode="replace",
                )
            )
        if "strain" in hints:
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="cpu_pressure",
                    intensity=float(hints["strain"]),
                    label=delta.delta_id,
                    mode="replace",
                )
            )
        # memory_pressure/thermal_pressure/disk_pressure (2026-07-16 fix):
        # previously only computed upstream and folded into the composite
        # "strain" hint above -- these three lattice channels sat pinned at
        # 0.0 for the whole live corpus. Additive: unlike gpu/strain, these
        # hint keys map 1:1 onto their own NODE_CHANNELS name (no remap).
        #
        # mode="replace" (same reasoning as gpu_pressure/cpu_pressure above):
        # every grammar event in a node_biometrics trace -- not just the atom
        # that first sets a given hint -- reaches this function with its own
        # uniquely-keyed StateDeltaV1 (node_reducer.py runs once per trace
        # event, including trace_started/edge_emitted/trace_ended, and each
        # carries the cumulative merged.pressure_hints forward). A single
        # ~22-event biometrics trace produces on the order of 14-16 separate
        # deltas that each still contain "memory_pressure"/"thermal_pressure"/
        # "disk_pressure" once those hints are first set. Under "add" mode
        # (raw += with no cross-delta dedup, see apply_perturbations) that
        # would re-add the same intensity that many times per telemetry
        # cycle, saturating the channel to the 1.0 clamp almost immediately
        # regardless of real load. execution_run/chat_turn below already hit
        # this exact class of bug for their own pressure_hints snapshots and
        # fixed it with mode="replace"; applying the same precedent here so
        # these three new channels don't inherit it.
        if "memory_pressure" in hints:
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="memory_pressure",
                    intensity=float(hints["memory_pressure"]),
                    label=delta.delta_id,
                    mode="replace",
                )
            )
        if "thermal_pressure" in hints:
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="thermal_pressure",
                    intensity=float(hints["thermal_pressure"]),
                    label=delta.delta_id,
                    mode="replace",
                )
            )
        if "disk_pressure" in hints:
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="disk_pressure",
                    intensity=float(hints["disk_pressure"]),
                    label=delta.delta_id,
                    mode="replace",
                )
            )
        status = str(after.get("availability_status") or "")
        if status == "stale":
            out.append(Perturbation(node_id=node_id, channel="staleness", intensity=0.5, label=delta.delta_id))
        if after.get("expected_online") is False:
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="expected_offline_suppression",
                    intensity=1.0,
                    label=delta.delta_id,
                )
            )
        # Mirror of the branch above. expected_offline_suppression is
        # intentionally sticky (mode="add", never in NODE_DECAY_CHANNELS --
        # "this node is known offline" should latch, not fade on a timer),
        # but until this branch existed there was no path anywhere that ever
        # set it back to 0.0: the only two writers were this block's
        # `is False` case and the active_node_pressure "suppress" op above
        # (target_kind == "active_node_pressure"), both one-directional.
        # mode="replace" bypasses the add-mode ceiling
        # entirely so a prior 1.0 is fully cleared, not just added to.
        #
        # Precedence (Juniper's explicit call, 2026-07-17 review follow-up):
        # if this clear and an active_node_pressure "suppress" op both land
        # on expected_offline_suppression for the same node in the same
        # tick, the clear wins -- biometrics liveness ground truth beats a
        # same-tick suppress signal. That's enforced order-independently in
        # app/digestion/perturbation.py::apply_perturbations() (not here),
        # since both perturbations only meet each other once they're
        # flattened into one list in app/worker.py::_tick().
        if after.get("expected_online") is True:
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="expected_offline_suppression",
                    intensity=0.0,
                    label=delta.delta_id,
                    mode="replace",
                )
            )

    if delta.target_kind == "execution_run":
        hints = dict(after.get("pressure_hints") or {})
        node_key = _node_key(str(after.get("node_id") or delta.target_id))
        for channel, key in (
            ("execution_load", "execution_load"),
            ("execution_friction", "execution_friction"),
            ("reasoning_load", "reasoning_load"),
            ("failure_pressure", "failure_pressure"),
        ):
            if key in hints:
                out.append(
                    Perturbation(
                        node_id=node_key,
                        channel=channel,
                        intensity=float(hints[key]),
                        label=delta.delta_id,
                        mode="replace",
                    )
                )
        try:
            egress_raw = float(hints["egress_confidence"]) if "egress_confidence" in hints else None
        except (TypeError, ValueError):
            egress_raw = None
        if egress_raw is not None and math.isfinite(egress_raw):
            out.append(
                Perturbation(
                    node_id=node_key,
                    channel="egress_confidence_deficit",
                    intensity=max(0.0, min(1.0, 1.0 - egress_raw)),
                    label=delta.delta_id,
                    mode="replace",
                )
            )

    if delta.target_kind == "chat_turn":
        hints = dict(after.get("pressure_hints") or {})
        for channel, key in (
            ("conversation_load", "conversation_load"),
            ("repair_pressure", "repair_pressure"),
        ):
            if key in hints:
                out.append(
                    Perturbation(
                        node_id=node_id,
                        channel=channel,
                        intensity=float(hints[key]),
                        label=delta.delta_id,
                        mode="replace",
                    )
                )

    if delta.target_kind == "transport_bus":
        hints = dict((delta.after or {}).get("pressure_hints") or {})
        node_key = _node_key(str((delta.after or {}).get("node_id") or delta.target_id.replace("bus:", "")))
        # bus_health / delivery_confidence are fresh-value-per-report readings
        # (orion/substrate/transport_loop/reducer.py::reduce_transport_trace_events
        # recomputes them from scratch via extract_transport_bus_state_from_events
        # every time it reduces new bus-trace events -- not an incremental delta),
        # the same "here's the current reading" shape already used for
        # execution_load/execution_friction/reasoning_load/failure_pressure/
        # egress_confidence_deficit/prediction_error elsewhere in this file, all
        # of which use mode="replace". Left as default mode="add" here, these two
        # channels ratchet to 1.0 and stay there (add-mode ceiling, never cleared)
        # since nothing else in this file wrote a downward value for them.
        # mode="replace" alone is the fix: worker.py::_transport_tick() only
        # reduces when fetch_transport_grammar_events() actually returns events
        # (`if not events: return None`), i.e. bus_health/delivery_confidence
        # deltas arrive only when there is genuine transport bus trace activity,
        # not unconditionally every field-digester tick. Deliberately NOT added
        # to NODE_DECAY_CHANNELS: both are "current health/confidence reading"
        # scores (0.0-1.0, default 0.5 -- see orion/schemas/transport_projection.py
        # and extract.py), not pressure accumulators. A health score fading
        # toward 0.0 between reports, just because the bus went quiet, would
        # misrepresent a quiet bus as a degrading one; mode="replace" already
        # guarantees the value reflects the most recent genuine report, and decay
        # would only distort that between reports rather than bound anything.
        if "bus_health" in hints:
            out.append(
                Perturbation(
                    node_id=node_key,
                    channel="bus_health",
                    intensity=float(hints["bus_health"]),
                    label=delta.delta_id,
                    mode="replace",
                )
            )
        if "delivery_confidence" in hints:
            out.append(
                Perturbation(
                    node_id=node_key,
                    channel="delivery_confidence",
                    intensity=float(hints["delivery_confidence"]),
                    label=delta.delta_id,
                    mode="replace",
                )
            )
        for channel, key in (
            ("transport_pressure", "transport_pressure"),
            ("catalog_drift_pressure", "catalog_drift_pressure"),
            ("observer_failure_pressure", "observer_failure_pressure"),
            ("reliability_pressure", "reliability_pressure"),
            ("contract_pressure", "contract_pressure"),
        ):
            if key in hints:
                out.append(
                    Perturbation(
                        node_id=node_key,
                        channel=channel,
                        intensity=float(hints[key]),
                        label=delta.delta_id,
                    )
                )
        if "stream_depth_pressure" in hints:
            out.append(
                Perturbation(
                    node_id=node_key,
                    channel="transport_pressure",
                    intensity=float(hints["stream_depth_pressure"]),
                    label=delta.delta_id,
                )
            )
        if "backpressure" in hints:
            out.append(
                Perturbation(
                    node_id=node_key,
                    channel="transport_pressure",
                    intensity=float(hints["backpressure"]),
                    label=delta.delta_id,
                )
            )

    if delta.target_kind == "prediction_signal":
        hints = dict((delta.after or {}).get("pressure_hints") or {})
        node_key = _node_key(str((delta.after or {}).get("node_id") or delta.target_id))
        if "prediction_error" in hints:
            out.append(
                Perturbation(
                    node_id=node_key,
                    channel="prediction_error",
                    intensity=max(0.0, min(1.0, float(hints["prediction_error"]))),
                    label=delta.delta_id,
                    mode="replace",
                )
            )
    return out
