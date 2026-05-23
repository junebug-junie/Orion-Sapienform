"""SubstrateMoleculeV1 — the shared substrate currency.

A molecule is a small composite of atoms + relations enriched with gradient
state and provenance. Every organ emits the *same* molecule shape; no
organ-specific schemas live here.

This module is intentionally decoupled from the existing
``orion.substrate.*`` graph layer — it is the MVP kernel, not the production
materializer. The two co-exist; future work can bridge them.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orion.schema_kernel import (
    ConceptRelationV1,
    DEFAULT_GRADIENT_KEYS,
    SchemaKernelRegistry,
    SchemaValidationError,
    empty_gradient_vector,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_molecule_id() -> str:
    return f"mol_{uuid.uuid4().hex[:16]}"


class SubstrateMoleculeV1(BaseModel):
    """A unit of shared substrate reality.

    ``molecule_kind`` examples (MVP): observation, claim, pressure, contradiction.
    ``atoms`` maps an in-molecule role -> atom key from the kernel.
    ``relations`` describe couplings inside or pointing out of the molecule.
    ``gradients`` carry the canonical evolving field state.
    ``provenance`` records emitting organ, source ids, etc.
    ``payload`` is a free-form bag for organ specifics that should *not* leak
        into the grammar.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    molecule_id: str = Field(default_factory=_new_molecule_id)
    molecule_kind: str

    atoms: dict[str, str] = Field(default_factory=dict)
    relations: list[ConceptRelationV1] = Field(default_factory=list)

    gradients: dict[str, float] = Field(default_factory=empty_gradient_vector)

    provenance: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=_utcnow)
    last_touched_at: datetime = Field(default_factory=_utcnow)

    # -- helpers ---------------------------------------------------------------

    def gradient(self, key: str) -> float:
        return self.gradients.get(key, 0.0)

    def touch(self, when: datetime | None = None) -> None:
        self.last_touched_at = when or _utcnow()

    def to_jsonable(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def validate_molecule(
    molecule: SubstrateMoleculeV1,
    registry: SchemaKernelRegistry,
) -> None:
    """Ensure a molecule fits the active kernel vocabulary."""

    if not registry.has_molecule_kind(molecule.molecule_kind):
        raise SchemaValidationError(
            f"molecule_kind '{molecule.molecule_kind}' is not registered"
        )
    for role, atom_key in molecule.atoms.items():
        if not registry.has_atom(atom_key):
            raise SchemaValidationError(
                f"role '{role}' references unregistered atom '{atom_key}'"
            )
    for relation in molecule.relations:
        if not registry.has_predicate(relation.predicate):
            raise SchemaValidationError(
                f"predicate '{relation.predicate}' is not registered"
            )
    for key in molecule.gradients:
        if key not in DEFAULT_GRADIENT_KEYS:
            raise SchemaValidationError(
                f"gradient key '{key}' is not part of the canonical set"
            )
