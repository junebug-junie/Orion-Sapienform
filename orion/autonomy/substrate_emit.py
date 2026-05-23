"""Thin substrate-emit helpers for the autonomy/pressure organ.

These do not modify any existing autonomy code — they exist so the pressure
loop can drop molecules onto the shared substrate from anywhere.
"""

from __future__ import annotations

from typing import Any

from orion.schema_kernel import ConceptRelationV1
from orion.substrate.molecules import SubstrateMoleculeV1


ORGAN_NAME = "autonomy"


def emit_pressure(
    *,
    label: str,
    magnitude: float,
    source_goal_id: str | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> SubstrateMoleculeV1:
    """A pressure molecule is a constraint+gradient pair.

    ``magnitude`` is folded directly into the salience gradient so the harness
    can see autonomy load aggregate over time.
    """

    relations = [
        ConceptRelationV1(
            source="primary",
            predicate="constrains",
            target="scope",
            weight=max(0.0, magnitude),
        )
    ]
    payload = {"label": label, "magnitude": magnitude}
    if extra_payload:
        payload.update(extra_payload)
    provenance = {"organ": ORGAN_NAME, "channel": "pressure"}
    if source_goal_id:
        provenance["source_id"] = source_goal_id
    molecule = SubstrateMoleculeV1(
        molecule_kind="pressure",
        atoms={"primary": "constraint", "scope": "context", "field": "gradient"},
        relations=relations,
        provenance=provenance,
        payload=payload,
    )
    molecule.gradients["salience"] = min(1.0, max(0.0, magnitude))
    return molecule


def emit_contradiction(
    *,
    summary: str,
    between: tuple[str, str],
    extra_payload: dict[str, Any] | None = None,
) -> SubstrateMoleculeV1:
    """Emit a contradiction molecule that points at two other molecule ids."""

    left, right = between
    relations = [
        ConceptRelationV1(
            source=left,
            predicate="contradicts",
            target=right,
            polarity=-1.0,
        )
    ]
    payload = {"summary": summary, "between": [left, right]}
    if extra_payload:
        payload.update(extra_payload)
    provenance = {"organ": ORGAN_NAME, "channel": "pressure"}
    molecule = SubstrateMoleculeV1(
        molecule_kind="contradiction",
        atoms={"primary": "constraint", "left": "evidence", "right": "evidence"},
        relations=relations,
        provenance=provenance,
        payload=payload,
    )
    molecule.gradients["contradiction"] = 0.5
    molecule.gradients["salience"] = 0.4
    return molecule
