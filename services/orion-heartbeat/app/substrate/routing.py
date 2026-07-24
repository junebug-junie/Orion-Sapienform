"""Site-routing rule for Heartbeat v0's tensor-network substrate.

Resolved this session (docs/superpowers/specs/2026-07-24-spark-field-holographic-
lattice-design.md, "Site-routing rule (v0) -- resolved"): reuses the shelved
2026-05-01 heartbeat charter's own `ChannelAssignment` shape (site_index,
operator_kind, operator_params) -- only how `site_index` is derived changes.
The charter hand-authored a 13-row SITE_ASSIGNMENT_TABLE.md; this derives the
site instead from `GrammarEventV1`'s own already-standardized fields, which is
exactly the fix for why the charter's original ingestion plan ("too messy to
create bespoke signals per organ") stalled.

WHERE (site_index): `GrammarProvenanceV1.source_service`, present on every
GrammarEventV1, real organ identity already, no inference needed.

HOW (operator_kind / operator_params): `GrammarAtomV1.atom_type` (a closed
Literal enum, unlike free-form `layer`/`dimensions`) selects the local
operator kind; `confidence`/`salience`/`uncertainty` (already on every atom)
parameterize its strength. No new computation, no bespoke per-organ mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

OperatorKind = Literal["amplitude", "phase", "rotation", "projection"]

# Five organs confirmed live this session by cross-referencing
# orion/bus/channels.yaml's orion:grammar:event producer list against
# services/orion-substrate-runtime/app/worker.py's five real reducer cursors
# (GRAMMAR_CURSOR_NAME/EXECUTION_GRAMMAR_CURSOR_NAME/TRANSPORT_GRAMMAR_CURSOR_NAME/
# CHAT_GRAMMAR_CURSOR_NAME/ROUTE_GRAMMAR_CURSOR_NAME). Vision (x3) and
# harness-governor also publish to the channel but have no reducer precedent
# anywhere -- deliberately excluded from v0, see the design doc's "Organ list
# -- resolved" section.
#
# Site ordering is boundary-block-then-bulk-block (sites 0-4 boundary, 5-9
# bulk) so `HeartbeatSubstrate.entropy_profile()` can read the boundary/bulk
# entanglement entropy as a single cheap MPS bipartite cut at index 5, rather
# than an expensive dense partial trace over an arbitrary subset (see
# reconstruction.py's module docstring for why that mattered).
#
# Site 4 (route) is adjacent to bulk site 5; site 0 (chat) is four hops away
# from the nearest bulk site. This asymmetry is a direct, disclosed
# consequence of the chosen ordering (design doc Missing Question 7, not
# resolved further this session) -- organs don't reach the bulk equally fast
# under HeartbeatSubstrate's nearest-neighbor entangling update.
ORGAN_SITE_MAP: dict[str, int] = {
    "orion-hub": 0,  # chat (build_chat_turn_grammar_events; only a user_utterance
    # atom is emitted today, no separate Orion-response-turn producer was found
    # this session -- one site, not the charter's original two-site chat split)
    "orion-biometrics": 1,
    "orion-cortex-exec": 2,  # execution
    "orion-bus": 3,  # transport
    "orion-cortex-orch": 4,  # route
}

N_SITES = 10
BOUNDARY_SITES: tuple[int, ...] = (0, 1, 2, 3, 4)
BULK_SITES: tuple[int, ...] = (5, 6, 7, 8, 9)
BOUNDARY_BULK_CUT = 5  # mps.entropy(BOUNDARY_BULK_CUT) is the single, cheap,
# native MPS bipartite-entanglement measurement of exactly this cut -- the
# central H1 v0 number.

BOND_DIM = 4  # unchanged from the charter -- kept conservative on purpose,
# see design doc: a larger bond dimension trivially makes reconstruction/
# entanglement easier, which would bias any redundancy finding toward a false
# positive.
PHYS_DIM = 4  # unchanged from the charter.

# atom_type -> operator_kind. Every AtomType from orion/schemas/grammar.py is
# covered explicitly (no silent catch-all) so a future grammar.py addition
# fails loud (KeyError) here rather than being routed by accident.
ATOM_TYPE_OPERATOR_KIND: dict[str, OperatorKind] = {
    "raw_span": "amplitude",
    "observation": "amplitude",
    "signal": "amplitude",
    "entity": "amplitude",
    "claim": "rotation",
    "affective_cue": "phase",
    "salience_marker": "phase",
    "uncertainty_marker": "rotation",
    "memory_claim": "rotation",
    "stance_influence": "phase",
    "reasoning_step": "rotation",
    "action_candidate": "projection",
    "spoken_output": "amplitude",
    "scene_state": "amplitude",
    "compaction": "projection",
    "projection": "projection",
}


@dataclass(frozen=True)
class SiteAssignment:
    """Mirrors the charter's own ChannelAssignment shape."""

    site_index: int
    operator_kind: OperatorKind
    confidence: float
    salience: float
    uncertainty: float


class UnroutableOrganError(ValueError):
    """Raised when a GrammarEventV1's source_service isn't in the v0 allowlist.

    Callers should treat this as "not one of the five v0 organs" and skip the
    event, not crash the tick loop -- see service.py's handler.
    """


class UnroutableAtomTypeError(ValueError):
    """Raised when an atom_type isn't in ATOM_TYPE_OPERATOR_KIND.

    Should be unreachable given the table covers every AtomType in
    orion/schemas/grammar.py as of this session; kept as a fail-loud guard
    against future grammar.py additions rather than a silent default.
    """


def route_atom(
    *,
    source_service: str,
    atom_type: str,
    confidence: Optional[float],
    salience: Optional[float],
    uncertainty: Optional[float],
) -> SiteAssignment:
    """WHERE + HOW for one GrammarAtomV1. Raises on anything outside v0's
    five-organ allowlist or an unrecognized atom_type -- callers decide
    whether to skip or fail loud (service.py skips; this module doesn't).
    """
    if source_service not in ORGAN_SITE_MAP:
        raise UnroutableOrganError(source_service)
    if atom_type not in ATOM_TYPE_OPERATOR_KIND:
        raise UnroutableAtomTypeError(atom_type)
    return SiteAssignment(
        site_index=ORGAN_SITE_MAP[source_service],
        operator_kind=ATOM_TYPE_OPERATOR_KIND[atom_type],
        confidence=float(confidence) if confidence is not None else 1.0,
        salience=float(salience) if salience is not None else 0.5,
        uncertainty=float(uncertainty) if uncertainty is not None else 0.5,
    )
