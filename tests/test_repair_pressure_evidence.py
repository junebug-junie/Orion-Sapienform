"""Repair evidence detector tests — spec §14.1."""

from __future__ import annotations

from orion.mind.substrate_emit import emit_observation
from orion.substrate.appraisal.evidence import extract_repair_evidence
from orion.substrate.molecules import SubstrateMoleculeV1


def _obs(text: str) -> SubstrateMoleculeV1:
    return emit_observation(surface_text=text, source_id="msg-test")


def _max_score(evidence, kind):
    items = [e for e in evidence if e.evidence_kind == kind]
    return max((e.score for e in items), default=0.0)


def test_specificity_demand_nuts_and_bolts():
    ev = extract_repair_evidence([_obs("not hand wavy, give me nuts and bolts")])
    assert _max_score(ev, "specificity_demand") >= 0.75


def test_trust_rupture_garbage_directions():
    ev = extract_repair_evidence([_obs("you gave me garbage directions")])
    assert _max_score(ev, "trust_rupture") >= 0.75


def test_coherence_gap_swamp():
    ev = extract_repair_evidence([_obs("this is becoming a swamp")])
    assert _max_score(ev, "coherence_gap") >= 0.70


def test_repetition_failure_again_you_keep():
    ev = extract_repair_evidence([_obs("again, you keep doing this")])
    assert _max_score(ev, "repetition_failure") >= 0.65


def test_operational_block_design_spec():
    ev = extract_repair_evidence([_obs("build me a design spec for Claude")])
    assert _max_score(ev, "operational_block") >= 0.70


def test_explicit_repair_command_arsonist_pov():
    ev = extract_repair_evidence([_obs("arsonist POV only here")])
    assert _max_score(ev, "explicit_repair_command") >= 0.65


def test_evidence_emits_span_and_source_molecule_id():
    mol = _obs("you gave me garbage directions")
    ev = extract_repair_evidence([mol])
    matches = [e for e in ev if e.evidence_kind == "trust_rupture"]
    assert matches
    e = matches[0]
    assert e.source_molecule_id == mol.molecule_id
    assert e.span is not None and "garbage" in e.span.lower()
    assert 0.0 <= e.confidence <= 1.0


def test_evidence_returns_empty_for_neutral_text():
    ev = extract_repair_evidence([_obs("what is the weather like?")])
    assert ev == []


def test_evidence_skips_molecules_without_surface_text():
    mol = SubstrateMoleculeV1(molecule_kind="observation")
    assert extract_repair_evidence([mol]) == []
