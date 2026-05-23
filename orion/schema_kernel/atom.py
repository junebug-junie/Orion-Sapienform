"""Atoms — reusable semantic invariants.

An atom is *not* a domain noun (memory, dream, emotion). It is a dimension of
interaction that recurs across cognition, biology, social, and metabolism.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field


ATOM_KINDS: Final[tuple[str, ...]] = (
    "signal",
    "constraint",
    "attention",
    "state",
    "change",
    "relation",
    "context",
    "agency",
    "evidence",
    "gradient",
    "persistence",
    "boundary",
)


class ConceptAtomV1(BaseModel):
    """A reusable semantic invariant.

    `key` is the unique handle (e.g. "signal.valence", "attention.allocation").
    `atom_kind` must come from `ATOM_KINDS` — the closed set of invariant
    families. Axes describe optional sub-dimensions the atom carries when it
    participates in a composite (kept open-ended on purpose).
    """

    model_config = ConfigDict(frozen=True)

    key: str = Field(min_length=1)
    atom_kind: str
    description: str | None = None
    axes: tuple[str, ...] = ()


DEFAULT_ATOMS: Final[tuple[ConceptAtomV1, ...]] = (
    ConceptAtomV1(
        key="signal",
        atom_kind="signal",
        description="A perturbation with valence/intensity/persistence.",
        axes=("valence", "intensity", "persistence"),
    ),
    ConceptAtomV1(
        key="constraint",
        atom_kind="constraint",
        description="A pressure or boundary that shapes admissible states.",
        axes=("pressure", "resistance"),
    ),
    ConceptAtomV1(
        key="attention",
        atom_kind="attention",
        description="Allocation/competition over signals.",
        axes=("allocation", "fixation", "competition"),
    ),
    ConceptAtomV1(
        key="state",
        atom_kind="state",
        description="A point in a configuration space.",
        axes=("value",),
    ),
    ConceptAtomV1(
        key="change",
        atom_kind="change",
        description="A transformation between states.",
        axes=("magnitude", "direction"),
    ),
    ConceptAtomV1(
        key="relation",
        atom_kind="relation",
        description="A coupling between two loci.",
        axes=("weight", "polarity"),
    ),
    ConceptAtomV1(
        key="context",
        atom_kind="context",
        description="The surround that gives a signal meaning.",
        axes=("scope",),
    ),
    ConceptAtomV1(
        key="agency",
        atom_kind="agency",
        description="The locus from which a transformation originates.",
        axes=("source", "intent"),
    ),
    ConceptAtomV1(
        key="evidence",
        atom_kind="evidence",
        description="A witness for or against a claim.",
        axes=("polarity", "weight", "source"),
    ),
    ConceptAtomV1(
        key="gradient",
        atom_kind="gradient",
        description="A directional pressure across a field.",
        axes=("magnitude", "direction"),
    ),
    ConceptAtomV1(
        key="persistence",
        atom_kind="persistence",
        description="How long a structure resists decay.",
        axes=("half_life",),
    ),
    ConceptAtomV1(
        key="boundary",
        atom_kind="boundary",
        description="A separation between inside/outside of a structure.",
        axes=("permeability",),
    ),
)
