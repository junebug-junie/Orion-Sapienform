"""Tiny validator — checks an atom/relation/composite against a registry."""

from __future__ import annotations

from .atom import ATOM_KINDS, ConceptAtomV1
from .composite import CompositeV1
from .relation import ConceptRelationV1
from .registry import SchemaKernelRegistry


class SchemaValidationError(ValueError):
    """Raised when a kernel object does not fit the registry vocabulary."""


def validate_atom(atom: ConceptAtomV1) -> None:
    if atom.atom_kind not in ATOM_KINDS:
        raise SchemaValidationError(
            f"atom_kind '{atom.atom_kind}' is not in ATOM_KINDS"
        )


def validate_relation(
    relation: ConceptRelationV1,
    registry: SchemaKernelRegistry,
) -> None:
    if not registry.has_predicate(relation.predicate):
        raise SchemaValidationError(
            f"predicate '{relation.predicate}' is not registered"
        )


def validate_composite(
    composite: CompositeV1,
    registry: SchemaKernelRegistry,
) -> None:
    for role, atom_key in composite.atoms.items():
        if not registry.has_atom(atom_key):
            raise SchemaValidationError(
                f"role '{role}' references unregistered atom '{atom_key}'"
            )
    for relation in composite.relations:
        validate_relation(relation, registry)
