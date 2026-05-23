"""Schema kernel: atoms, relations, registry, validator."""

from __future__ import annotations

import pytest

from orion.schema_kernel import (
    ATOM_KINDS,
    ConceptAtomV1,
    ConceptRelationV1,
    DEFAULT_ATOMS,
    DEFAULT_PREDICATES,
    SchemaKernelRegistry,
    SchemaValidationError,
    default_registry,
    validate_atom,
    validate_relation,
)


def test_default_registry_has_canonical_vocabulary():
    registry = default_registry()
    assert len(registry.atoms()) == len(DEFAULT_ATOMS)
    for atom in DEFAULT_ATOMS:
        assert registry.has_atom(atom.key)
    for predicate in DEFAULT_PREDICATES:
        assert registry.has_predicate(predicate)
    for kind in ("observation", "claim", "pressure", "contradiction"):
        assert registry.has_molecule_kind(kind)


def test_atom_kind_must_be_in_closed_set():
    rogue = ConceptAtomV1(key="dream", atom_kind="dream")
    with pytest.raises(SchemaValidationError):
        validate_atom(rogue)


def test_atom_default_set_passes_validation():
    for atom in DEFAULT_ATOMS:
        validate_atom(atom)
    # And every default atom_kind is in the closed set.
    for atom in DEFAULT_ATOMS:
        assert atom.atom_kind in ATOM_KINDS


def test_relation_polarity_bounded():
    with pytest.raises(ValueError):
        ConceptRelationV1(source="a", predicate="supports", target="b", polarity=2.0)
    with pytest.raises(ValueError):
        ConceptRelationV1(source="a", predicate="supports", target="b", polarity=-1.5)


def test_relation_weight_must_be_non_negative():
    with pytest.raises(ValueError):
        ConceptRelationV1(source="a", predicate="supports", target="b", weight=-0.1)


def test_validate_relation_against_registry():
    registry = default_registry()
    valid = ConceptRelationV1(source="a", predicate="supports", target="b")
    validate_relation(valid, registry)

    invalid = ConceptRelationV1(source="a", predicate="invented", target="b")
    with pytest.raises(SchemaValidationError):
        validate_relation(invalid, registry)


def test_registry_register_extends_vocabulary():
    registry = SchemaKernelRegistry()
    atom = ConceptAtomV1(key="signal", atom_kind="signal")
    registry.register_atom(atom)
    registry.register_predicate("references")
    registry.register_molecule_kind("observation")
    assert registry.has_atom("signal")
    assert registry.has_predicate("references")
    assert registry.has_molecule_kind("observation")
