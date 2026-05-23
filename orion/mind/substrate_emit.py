"""Thin substrate-emit helpers for the mind/chat organ.

This module deliberately does NOT touch any existing chat code. Callers (or a
future shim) can invoke ``emit_observation`` / ``emit_claim`` to put molecules
on the substrate without restructuring the mind layer.
"""

from __future__ import annotations

from typing import Any

from orion.schema_kernel import ConceptRelationV1
from orion.substrate.molecules import SubstrateMoleculeV1


ORGAN_NAME = "mind"


def emit_observation(
    *,
    surface_text: str,
    source_id: str | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> SubstrateMoleculeV1:
    """Build an ``observation`` molecule from a chat turn.

    The molecule binds two atoms: a primary signal (the utterance perturbation)
    and a context atom (the surrounding chat scope). No NLP — just structure.
    """

    relations = [
        ConceptRelationV1(
            source="primary",
            predicate="references",
            target="scope",
            weight=1.0,
        )
    ]
    payload = {"surface_text": surface_text}
    if extra_payload:
        payload.update(extra_payload)
    provenance = {"organ": ORGAN_NAME, "channel": "chat"}
    if source_id:
        provenance["source_id"] = source_id
    return SubstrateMoleculeV1(
        molecule_kind="observation",
        atoms={"primary": "signal", "scope": "context"},
        relations=relations,
        provenance=provenance,
        payload=payload,
    )


def emit_claim(
    *,
    claim_text: str,
    supports: list[str] | None = None,
    contradicts: list[str] | None = None,
    source_id: str | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> SubstrateMoleculeV1:
    """Build a ``claim`` molecule. Supports/contradicts are other molecule ids."""

    relations: list[ConceptRelationV1] = []
    for target in supports or ():
        relations.append(
            ConceptRelationV1(
                source="primary",
                predicate="supports",
                target=target,
                polarity=1.0,
            )
        )
    for target in contradicts or ():
        relations.append(
            ConceptRelationV1(
                source="primary",
                predicate="contradicts",
                target=target,
                polarity=-1.0,
            )
        )
    payload = {"claim_text": claim_text}
    if extra_payload:
        payload.update(extra_payload)
    provenance = {"organ": ORGAN_NAME, "channel": "chat"}
    if source_id:
        provenance["source_id"] = source_id
    return SubstrateMoleculeV1(
        molecule_kind="claim",
        atoms={"primary": "evidence", "scope": "context"},
        relations=relations,
        provenance=provenance,
        payload=payload,
    )
