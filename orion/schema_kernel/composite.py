"""Composite — a small bundle of atoms + relations.

This is the *kernel-level* composite. The substrate layer wraps composites into
SubstrateMoleculeV1, which adds gradients, provenance, and persistence.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .relation import ConceptRelationV1


class CompositeV1(BaseModel):
    """A named bundle of atoms and the relations among them.

    `atoms` maps a *role name* inside the composite to the atom key it binds.
    e.g. {"primary": "signal", "scope": "context"}.
    """

    model_config = ConfigDict(frozen=True)

    composite_kind: str = Field(min_length=1)
    atoms: dict[str, str] = Field(default_factory=dict)
    relations: tuple[ConceptRelationV1, ...] = ()
