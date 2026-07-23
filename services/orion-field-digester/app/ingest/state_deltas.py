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
        before_pressures = set((delta.before or {}).get("active_pressures") or [])
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
        elif "availability" in before_pressures:
            # Recovery transition (2026-07-22, code review finding on the
            # biometrics_loop "node_availability_recovered" fix): once
            # pressure_reducer.py can actually clear "availability" from
            # active_pressures again (previously a one-way ratchet -- see
            # the comment above), this delta's own after.active_pressures no
            # longer contains "availability" at all, so the branch above
            # never fires for it, or for anything after it. Without this,
            # the "availability" channel would freeze at whatever value it
            # last held at the moment of recovery instead of being restored
            # -- the exact same channel, a different (frozen, not
            # ratcheted) flavor of the same underlying bug. Explicitly
            # restore to fully-available on the True->False transition,
            # same "clear wins" pattern already used for
            # expected_offline_suppression below.
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="availability",
                    intensity=1.0,
                    label=delta.delta_id,
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
        # reasoning_load is attributed to whichever node actually served the
        # LLM call (after["llm_serving_node"], a first-class ExecutionRunStateV1
        # field threaded from llm-gateway's response meta.served_by -- see
        # orion/substrate/execution_loop/grammar_extract.py) rather than
        # node_key, the orchestrating service's own static identity. Read
        # from `after`, not `hints`/pressure_hints -- the latter must stay
        # float-only (orion/substrate/relational/adapters/execution_ctx.py
        # does max(hints.values()) over it). node_key is always athena today
        # (cortex-exec's NODE_NAME), which permanently zeroed
        # node:atlas.reasoning_load -- see this service's README.md,
        # reasoning_pressure glossary entry, for the live-confirmed root
        # cause. Falls back to node_key when a run made no LLM call.
        llm_serving_node = after.get("llm_serving_node")
        reasoning_node_key = (
            _node_key(str(llm_serving_node)) if llm_serving_node else node_key
        )
        for channel, key, target_node in (
            ("execution_load", "execution_load", node_key),
            ("execution_friction", "execution_friction", node_key),
            ("reasoning_load", "reasoning_load", reasoning_node_key),
            ("failure_pressure", "failure_pressure", node_key),
            # FCC-motor-process signals (step load, tool-failure streak, verbosity,
            # compliance) -- attributed to node_key (the orchestrating/governor node),
            # not reasoning_node_key, since these describe the motor process itself,
            # not which node served the LLM call. See
            # docs/superpowers/specs/2026-07-23-fcc-motor-field-digester-signals-design.md.
            ("harness_step_load", "harness_step_load", node_key),
            ("tool_failure_streak_pressure", "tool_failure_streak_pressure", node_key),
            ("avg_step_chars_pressure", "avg_step_chars_pressure", node_key),
            ("compliance_deficit", "compliance_deficit", node_key),
            # Distinct severity from compliance_deficit's worst rank: the governor
            # never responded at all (orion-hub-sourced exec_turn_timeout event, no
            # governor-side data ever existed for this run). node_key here resolves to
            # Hub's own node (parsed from the hub_turn_timeout-laned trace_id), not the
            # governor's -- this measures "Hub's view of governor unresponsiveness,"
            # not the governor's own node health, since they may differ.
            ("turn_incompletion", "turn_incompletion", node_key),
        ):
            if key in hints:
                out.append(
                    Perturbation(
                        node_id=target_node,
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
        # All seven pressure_hints channels below (bus_health,
        # delivery_confidence, and the five in the loop further down) are
        # fresh-value-per-report readings (orion/substrate/transport_loop/
        # reducer.py::reduce_transport_trace_events recomputes them from
        # scratch via extract_transport_bus_state_from_events every time it
        # reduces new bus-trace events -- not an incremental delta), the same
        # "here's the current reading" shape already used for
        # execution_load/execution_friction/reasoning_load/failure_pressure/
        # egress_confidence_deficit/prediction_error elsewhere in this file,
        # all of which use mode="replace".
        #
        # bus_health/delivery_confidence use mode="replace" and are
        # deliberately NOT in NODE_DECAY_CHANNELS: both are "current
        # health/confidence reading" scores (0.0-1.0, default 0.5 -- see
        # orion/schemas/transport_projection.py and extract.py), not pressure
        # accumulators, and have no decay story.
        #
        # transport_pressure/catalog_drift_pressure/observer_failure_pressure/
        # reliability_pressure/contract_pressure (the loop below) ARE in
        # NODE_DECAY_CHANNELS -- decaying toward baseline if the bus-observer
        # genuinely goes dark is correct for these. But until 2026-07-22 this
        # loop left mode at its default "add", which combined with
        # apply_perturbations() unconditionally stamping
        # node_vector_updated_at on every perturbation regardless of mode
        # (perturbation.py) to produce a stuck-value bug: a real "no drift"
        # report (intensity=0.0) is a no-op in add-mode, so it can never
        # correct a previously-injected nonzero value back down, while
        # simultaneously re-marking the channel "fresh" every ~10s -- which
        # makes apply_decay()'s hold-if-fresh logic skip decay too. Net
        # effect: whatever nonzero value one of these channels last picked up
        # from a genuine event got permanently frozen, immune to both
        # correction (add-mode 0.0 no-ops) and decay (perpetually "fresh").
        # Confirmed live 2026-07-22: catalog_drift_pressure stuck at
        # 0.13517857261119032 for 10+ minutes across a service restart while
        # transport_bus_reducer was continuously emitting the real current
        # value (0.0) every ~10s the whole time -- traced end to end via
        # docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md's
        # telemetry_anomaly investigation, which was firing on nearly every
        # tick because of this exact stale channel. mode="replace" is the
        # same fix already applied to bus_health/delivery_confidence above.
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
                        mode="replace",
                    )
                )
        # stream_depth_pressure/backpressure are NOT separately injected here:
        # extract.py's transport_pressure = max(stream_depth_pressure,
        # backpressure) already folds both into the "transport_pressure" hint
        # handled by the loop above. Re-injecting them as their own
        # mode="add" perturbations against the same channel (as this code did
        # until 2026-07-22) would fight the replace above -- whichever
        # Perturbation for "transport_pressure" apply_perturbations() sees
        # last in this list wins outright (replace) or silently inflates it
        # past the intended max() (add), depending on ordering. hints always
        # carries all of bus_health/delivery_confidence/stream_depth_pressure/
        # backpressure/catalog_drift_pressure/observer_failure_pressure/
        # transport_pressure/contract_pressure/reliability_pressure together
        # (single dict returned by compute_transport_pressures()), so
        # "transport_pressure" is always present whenever
        # "stream_depth_pressure"/"backpressure" would have been.

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
