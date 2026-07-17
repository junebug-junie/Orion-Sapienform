from __future__ import annotations

from orion.schemas.field_state import FieldStateV1

from app.ingest.state_deltas import Perturbation


def apply_perturbations(state: FieldStateV1, perturbations: list[Perturbation]) -> FieldStateV1:
    # Juniper's explicit call (2026-07-17, code-review follow-up on the
    # expected_offline_suppression ratchet fix): "clear wins". When a
    # node_biometrics "back online" clear (expected_online is True ->
    # mode="replace", intensity=0.0) and an active_node_pressure "suppress"
    # perturbation (mode="add", intensity=1.0) both target
    # expected_offline_suppression for the same node in the same tick, the
    # clear must always take effect -- biometrics liveness is treated as
    # more authoritative ground truth than a same-tick suppress signal.
    # Before this, both perturbations landed in the same flattened per-tick
    # list (app/worker.py::_tick() extends one list across every delta in
    # the batch) and whichever happened to be applied last won -- an
    # accident of delta fetch/receipt order, not a deliberate policy. This
    # pre-scan makes the outcome deterministic and order-independent within
    # a single call, rather than depending on callers to pre-sort the list.
    suppression_clear_nodes = {
        p.node_id
        for p in perturbations
        if p.channel == "expected_offline_suppression" and p.mode == "replace"
    }
    for p in perturbations:
        overridden_by_clear = (
            p.channel == "expected_offline_suppression"
            and p.mode != "replace"
            and p.node_id in suppression_clear_nodes
        )
        if not overridden_by_clear:
            node_vec = state.node_vectors.setdefault(p.node_id, {})
            if p.mode == "replace":
                node_vec[p.channel] = max(0.0, min(1.0, p.intensity))
            elif p.channel == "availability":
                node_vec[p.channel] = min(node_vec.get(p.channel, 1.0), p.intensity)
            else:
                node_vec[p.channel] = min(1.0, node_vec.get(p.channel, 0.0) + p.intensity)
        if p.label not in state.recent_perturbations:
            state.recent_perturbations.append(p.label)
    state.recent_perturbations = state.recent_perturbations[-20:]
    return state
