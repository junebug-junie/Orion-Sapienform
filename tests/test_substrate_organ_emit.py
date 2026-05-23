"""Smoke tests for the mind + autonomy substrate_emit helpers.

The point is that BOTH organs produce the *same* molecule shape — no bespoke
schemas per organ.
"""

from __future__ import annotations

from orion.autonomy import substrate_emit as autonomy_emit
from orion.mind import substrate_emit as mind_emit
from orion.schema_kernel import default_registry
from orion.substrate.molecules import SubstrateMoleculeV1, validate_molecule


def test_both_organs_emit_substrate_molecules():
    observation = mind_emit.emit_observation(surface_text="hello world", source_id="msg-1")
    claim = mind_emit.emit_claim(claim_text="the sky is blue")
    pressure = autonomy_emit.emit_pressure(label="goal:learn", magnitude=0.6)
    contradiction = autonomy_emit.emit_contradiction(
        summary="claim X conflicts with observation Y",
        between=(claim.molecule_id, observation.molecule_id),
    )

    registry = default_registry()
    for molecule in (observation, claim, pressure, contradiction):
        assert isinstance(molecule, SubstrateMoleculeV1)
        validate_molecule(molecule, registry)


def test_mind_observation_carries_surface_text_in_payload():
    obs = mind_emit.emit_observation(surface_text="hello world")
    assert obs.payload["surface_text"] == "hello world"
    assert obs.provenance["organ"] == "mind"


def test_autonomy_pressure_magnitude_lands_in_salience_gradient():
    pressure = autonomy_emit.emit_pressure(label="goal:learn", magnitude=0.7)
    assert pressure.gradients["salience"] == 0.7
    assert pressure.provenance["organ"] == "autonomy"


def test_autonomy_contradiction_has_nonzero_contradiction_gradient():
    contradiction = autonomy_emit.emit_contradiction(
        summary="x vs y",
        between=("mol_a", "mol_b"),
    )
    assert contradiction.gradients["contradiction"] > 0
