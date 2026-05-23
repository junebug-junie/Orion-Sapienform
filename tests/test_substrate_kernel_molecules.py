"""Molecule schema + persistence."""

from __future__ import annotations

import pytest

from orion.schema_kernel import (
    ConceptRelationV1,
    DEFAULT_GRADIENT_KEYS,
    SchemaValidationError,
    default_registry,
)
from orion.substrate.molecule_store import MoleculeJsonlStore
from orion.substrate.molecules import SubstrateMoleculeV1, validate_molecule


def _good_molecule() -> SubstrateMoleculeV1:
    return SubstrateMoleculeV1(
        molecule_kind="observation",
        atoms={"primary": "signal", "scope": "context"},
        relations=[
            ConceptRelationV1(source="primary", predicate="references", target="scope")
        ],
        provenance={"organ": "mind"},
    )


def test_new_molecule_has_default_gradient_vector():
    molecule = _good_molecule()
    for key in DEFAULT_GRADIENT_KEYS:
        assert key in molecule.gradients
        assert molecule.gradients[key] == 0.0


def test_validate_molecule_accepts_canonical_shape():
    validate_molecule(_good_molecule(), default_registry())


def test_validate_molecule_rejects_unknown_kind():
    molecule = _good_molecule()
    molecule.molecule_kind = "made-up-kind"
    with pytest.raises(SchemaValidationError):
        validate_molecule(molecule, default_registry())


def test_validate_molecule_rejects_unknown_atom_role():
    molecule = _good_molecule()
    molecule.atoms = {"primary": "ghost_atom"}
    with pytest.raises(SchemaValidationError):
        validate_molecule(molecule, default_registry())


def test_molecule_jsonl_round_trip(tmp_path):
    path = tmp_path / "molecules.jsonl"
    store = MoleculeJsonlStore(path)
    first = _good_molecule()
    store.add(first)

    rehydrated = MoleculeJsonlStore(path)
    assert len(rehydrated) == 1
    fetched = rehydrated.get(first.molecule_id)
    assert fetched is not None
    assert fetched.molecule_kind == "observation"
    assert fetched.provenance["organ"] == "mind"


def test_molecule_store_filter_by_organ(tmp_path):
    store = MoleculeJsonlStore(tmp_path / "m.jsonl")
    mind_molecule = _good_molecule()
    auto_molecule = _good_molecule()
    auto_molecule.provenance = {"organ": "autonomy"}
    store.add(mind_molecule)
    store.add(auto_molecule)
    assert len(store.filter(organ="mind")) == 1
    assert len(store.filter(organ="autonomy")) == 1
    assert len(store.filter(molecule_kind="observation")) == 2
