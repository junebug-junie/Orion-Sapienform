"""Operators + traversal."""

from __future__ import annotations

from orion.substrate.molecules import SubstrateMoleculeV1
from orion.substrate.operators import (
    amplify_contradiction,
    decay_molecule,
    find_resonant_molecules,
    reinforce_molecule,
    stabilize_coherence,
)


def _mol(kind: str = "observation") -> SubstrateMoleculeV1:
    return SubstrateMoleculeV1(
        molecule_kind=kind,
        atoms={"primary": "signal", "scope": "context"},
        provenance={"organ": "mind"},
    )


def test_reinforce_raises_salience_and_coherence():
    molecule = _mol()
    delta = reinforce_molecule(molecule)
    assert molecule.gradients["salience"] > 0
    assert molecule.gradients["coherence"] > 0
    assert delta.cause == "reinforce"
    assert "salience" in delta.changed_keys()


def test_decay_lowers_salience_clamped_at_zero():
    molecule = _mol()
    decay_molecule(molecule)
    assert molecule.gradients["salience"] == 0.0
    assert molecule.gradients["coherence"] == 0.0


def test_amplify_contradiction_raises_both_fields():
    molecule = _mol()
    amplify_contradiction(molecule)
    assert molecule.gradients["contradiction"] > 0
    assert molecule.gradients["salience"] > 0


def test_stabilize_increases_coherence_and_drops_contradiction():
    molecule = _mol()
    amplify_contradiction(molecule)
    pre_contradiction = molecule.gradients["contradiction"]
    stabilize_coherence(molecule)
    assert molecule.gradients["coherence"] > 0
    assert molecule.gradients["contradiction"] < pre_contradiction


def test_observer_receives_delta():
    molecule = _mol()
    captured = []
    reinforce_molecule(molecule, observer=captured.append)
    assert len(captured) == 1
    assert captured[0].molecule_id == molecule.molecule_id


def test_find_resonant_molecules_returns_above_threshold_descending():
    low = _mol()
    mid = _mol()
    high = _mol()
    reinforce_molecule(mid)  # salience ~0.1
    for _ in range(3):
        reinforce_molecule(high)  # salience ~0.3

    hits = find_resonant_molecules(
        [low, mid, high], gradients=["salience"], threshold=0.05
    )
    assert hits[0].molecule_id == high.molecule_id
    assert hits[1].molecule_id == mid.molecule_id
    assert low not in hits
