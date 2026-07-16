"""Causal Geometry v1, Rung 2B: co-activation stats -> field-topology weight-patch proposals.

Turns a `CausalGeometrySnapshotV1` (Phase A / Rung 2A observed-vs-designed divergence
report) into `MutationProposalV1` rows for the `field_topology_weight_patch` mutation
class (`orion/substrate/mutation_contracts.py`), scoped to capability<->capability
("cap->cap") field-lattice edges only (`FieldEdgeV1.edge_type == "capability_capability"`,
`orion/schemas/field_state.py`).

This module only *proposes*. Nothing here enqueues a live weight change: proposals
must pass through `FieldTopologyLearnedWeightsStore.adopt()`
(`orion/substrate/field_topology_learned_store.py`), an explicit HITL action. This
module must never import or call `orion.substrate.mutation_apply.PatchApplier` --
that is the auto-promote path and `field_topology_weight_patch` has
`auto_promote_default=False` for a reason (see the contract in `mutation_contracts.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

from orion.core.schemas.substrate_mutation import MutationPatchV1, MutationProposalV1
from orion.schemas.causal_geometry import CausalGeometryDivergenceEntryV1, CausalGeometrySnapshotV1
from orion.schemas.field_state import FieldEdgeV1
from orion.substrate.mutation_contracts import CONTRACTS

MUTATION_CLASS = "field_topology_weight_patch"
TARGET_SURFACE = "field_topology"
CAP_CAP_EDGE_TYPE = "capability_capability"

# Below this |observed - designed| delta, divergence is treated as noise rather than
# a real drift signal -- proposing here would just spam the HITL review queue with
# deltas nobody should act on. Named constant per CLAUDE.md's "no magic numbers" rule.
MIN_MEANINGFUL_DELTA = 0.02


def edge_ref_for(source_id: str, target_id: str) -> str:
    """Canonical edge identifier used as `MutationPatchV1.target_ref` for this class."""
    return f"{source_id}->{target_id}"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class FieldTopologyDivergenceCandidate:
    """One cap->cap edge whose observed strength diverges from its designed weight."""

    edge_ref: str
    source_id: str
    target_id: str
    observed_strength: float
    designed_weight: float
    raw_delta: float
    clamped_delta: float


def _cap_cap_pairs(field_edges: list[FieldEdgeV1]) -> set[tuple[str, str]]:
    return {(edge.source_id, edge.target_id) for edge in field_edges if edge.edge_type == CAP_CAP_EDGE_TYPE}


def find_cap_cap_divergence_candidates(
    snapshot: CausalGeometrySnapshotV1,
    *,
    field_edges: list[FieldEdgeV1],
    min_meaningful_delta: float = MIN_MEANINGFUL_DELTA,
) -> list[FieldTopologyDivergenceCandidate]:
    """Filter a snapshot's divergence entries down to cap->cap edges with a real delta.

    `CausalGeometryDivergenceEntryV1` (Rung 2A's schema) carries no edge_type of its
    own -- only `FieldEdgeV1` (the designed lattice topology) does. So membership in
    the cap->cap class has to be resolved by joining on (source_id, target_id) against
    the caller-supplied field-lattice edges (typically the current
    `FieldStateV1.edges`), not against the snapshot alone.
    """
    bounds = CONTRACTS[MUTATION_CLASS].bounds["edge_weight_delta"]
    lo, hi = bounds
    cap_cap_pairs = _cap_cap_pairs(field_edges)
    candidates: list[FieldTopologyDivergenceCandidate] = []
    for entry in snapshot.divergence:
        if (entry.source_id, entry.target_id) not in cap_cap_pairs:
            continue
        candidate = _candidate_from_entry(entry, lo=lo, hi=hi, min_meaningful_delta=min_meaningful_delta)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _candidate_from_entry(
    entry: CausalGeometryDivergenceEntryV1,
    *,
    lo: float,
    hi: float,
    min_meaningful_delta: float,
) -> FieldTopologyDivergenceCandidate | None:
    if entry.status != "both":
        # observed_only / designed_only / insufficient_data: nothing to propose against.
        return None
    if entry.observed_strength is None or entry.designed_weight is None:
        return None
    raw_delta = float(entry.observed_strength) - float(entry.designed_weight)
    if abs(raw_delta) < min_meaningful_delta:
        return None
    clamped_delta = _clamp(raw_delta, lo, hi)
    return FieldTopologyDivergenceCandidate(
        edge_ref=edge_ref_for(entry.source_id, entry.target_id),
        source_id=entry.source_id,
        target_id=entry.target_id,
        observed_strength=float(entry.observed_strength),
        designed_weight=float(entry.designed_weight),
        raw_delta=raw_delta,
        clamped_delta=clamped_delta,
    )


def _proposal_for_candidate(candidate: FieldTopologyDivergenceCandidate, *, snapshot: CausalGeometrySnapshotV1) -> MutationProposalV1:
    contract = CONTRACTS[MUTATION_CLASS]
    patch = MutationPatchV1(
        mutation_class=MUTATION_CLASS,
        target_surface=TARGET_SURFACE,
        target_ref=candidate.edge_ref,
        patch={"edge_weight_delta": candidate.clamped_delta},
        rollback_payload={"edge_weight_delta": 0.0},
    )
    evidence_ref = f"causal_geometry:snapshot:{snapshot.snapshot_id}"
    return MutationProposalV1(
        lane="operational",
        mutation_class=MUTATION_CLASS,
        risk_tier=contract.risk_tier,
        target_surface=TARGET_SURFACE,
        anchor_scope="orion",
        subject_ref=f"field_edge:{candidate.edge_ref}",
        rationale=(
            f"causal_geometry_divergence edge={candidate.edge_ref} "
            f"observed={candidate.observed_strength:.4f} designed={candidate.designed_weight:.4f} "
            f"raw_delta={candidate.raw_delta:.4f} clamped_delta={candidate.clamped_delta:.4f}"
        ),
        expected_effect=f"reduce_causal_geometry_divergence_for_edge:{candidate.edge_ref}",
        evidence_refs=[evidence_ref, f"field_edge:{candidate.edge_ref}"],
        source_signal_ids=[evidence_ref],
        source_pressure_id=f"causal-geometry-derived-{snapshot.snapshot_id}-{candidate.edge_ref}",
        patch=patch,
        notes=[
            f"edge:{candidate.edge_ref}",
            f"observed_strength:{candidate.observed_strength:.4f}",
            f"designed_weight:{candidate.designed_weight:.4f}",
            f"raw_delta:{candidate.raw_delta:.4f}",
            f"clamped_delta:{candidate.clamped_delta:.4f}",
            f"snapshot_id:{snapshot.snapshot_id}",
            "hitl_adoption_required:field_topology_learned_store",
        ],
    )


def propose_field_topology_patches(
    snapshot: CausalGeometrySnapshotV1,
    *,
    field_edges: list[FieldEdgeV1],
    min_meaningful_delta: float = MIN_MEANINGFUL_DELTA,
) -> list[MutationProposalV1]:
    """Build `field_topology_weight_patch` proposals for diverging cap->cap edges.

    Callers (Rung 3A / a periodic reducer) are responsible for de-duplicating repeat
    proposals across snapshots and for routing the result into
    `FieldTopologyLearnedWeightsStore.propose()`. Nothing in this function enqueues,
    applies, or auto-promotes anything.
    """
    candidates = find_cap_cap_divergence_candidates(
        snapshot,
        field_edges=field_edges,
        min_meaningful_delta=min_meaningful_delta,
    )
    return [_proposal_for_candidate(candidate, snapshot=snapshot) for candidate in candidates]
