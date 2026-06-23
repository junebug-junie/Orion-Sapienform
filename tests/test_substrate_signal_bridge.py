"""Substrate signal bridge: OrionSignalV1 → SubstrateMoleculeV1."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schema_kernel import default_registry
from orion.signals.models import OrganClass, OrionSignalV1
from orion.substrate.molecules import validate_molecule


def make_cognition_run_signal(
    *,
    signal_kind: str = "cognition_run",
    dimensions: dict[str, float] | None = None,
    source_event_id: str = "corr-test-1",
) -> OrionSignalV1:
    now = datetime.now(timezone.utc)
    return OrionSignalV1(
        signal_id="sig-test-1",
        organ_id="cortex_exec",
        organ_class=OrganClass.endogenous,
        signal_kind=signal_kind,
        dimensions=dimensions or {},
        causal_parents=[],
        source_event_id=source_event_id,
        otel_trace_id="00000000000000000000000000000001",
        otel_span_id="0000000000000001",
        observed_at=now,
        emitted_at=now,
        summary="test signal",
        notes=[],
    )


def test_default_registry_includes_organ_signal_molecule_kind():
    registry = default_registry()
    assert registry.has_molecule_kind("organ_signal"), (
        "default_registry() must register 'organ_signal' so the signal bridge "
        "can emit molecules that validate."
    )


def test_cognition_run_success_converts_to_organ_signal_molecule():
    from orion.substrate.signal_bridge import signal_to_molecule

    signal = make_cognition_run_signal(
        dimensions={
            "success": 1.0,
            "step_count": 0.15,
            "latency_level": 0.30,
            "recall_used": 1.0,
            "reasoning_present": 1.0,
            "final_text_present": 1.0,
        },
    )

    molecule = signal_to_molecule(signal)

    assert molecule.molecule_kind == "organ_signal"
    assert molecule.provenance["organ"] == "cortex_exec"
    assert molecule.provenance["signal_kind"] == "cognition_run"
    assert molecule.provenance["signal_id"] == "sig-test-1"
    assert molecule.provenance["source_event_id"] == "corr-test-1"
    assert molecule.atoms["primary"] == "signal"
    assert molecule.atoms["source_process"] == "agency"
    assert molecule.atoms["source_context"] == "context"
    assert molecule.atoms["field"] == "gradient"
    assert molecule.atoms["witness"] == "evidence"

    assert molecule.gradients["salience"] == pytest.approx(0.30)
    assert molecule.gradients["contradiction"] == pytest.approx(0.0)
    assert molecule.gradients["novelty"] == pytest.approx(0.0)
    assert molecule.gradients["coherence"] == pytest.approx(1.0)

    validate_molecule(molecule, default_registry())


def test_cognition_run_failure_maps_to_contradiction_gradient():
    from orion.substrate.signal_bridge import signal_to_molecule

    signal = make_cognition_run_signal(
        dimensions={
            "success": 0.0,
            "step_count": 0.30,
            "latency_level": 0.70,
        },
    )

    molecule = signal_to_molecule(signal)

    assert molecule.gradients["contradiction"] == pytest.approx(1.0)
    assert molecule.gradients["salience"] == pytest.approx(0.70)
    assert molecule.gradients["coherence"] == pytest.approx(0.0)
    assert molecule.gradients["novelty"] == pytest.approx(0.0)


def test_cognition_step_error_present_maps_to_contradiction():
    from orion.substrate.signal_bridge import signal_to_molecule

    signal = make_cognition_run_signal(
        signal_kind="cognition_step",
        dimensions={
            "success": 0.0,
            "latency_level": 0.20,
            "error_present": 1.0,
            "service_count": 0.20,
        },
    )

    molecule = signal_to_molecule(signal)

    assert molecule.provenance["signal_kind"] == "cognition_step"
    assert molecule.gradients["contradiction"] == pytest.approx(1.0)
    assert molecule.gradients["salience"] == pytest.approx(0.20)
    assert molecule.gradients["coherence"] == pytest.approx(0.0)


def test_signals_to_molecules_skips_unsupported_signals():
    from orion.substrate.signal_bridge import signals_to_molecules

    unsupported_other_organ = make_cognition_run_signal().model_copy(
        update={"organ_id": "biometrics", "signal_kind": "gpu_load"}
    )
    supported = make_cognition_run_signal(
        dimensions={"success": 1.0, "step_count": 0.10, "latency_level": 0.10},
    )

    result = signals_to_molecules([unsupported_other_organ, supported])

    assert len(result) == 1
    assert result[0].provenance["organ"] == "cortex_exec"
    assert result[0].provenance["signal_kind"] == "cognition_run"


def test_signal_to_molecule_raises_on_unsupported_signal():
    from orion.substrate.signal_bridge import signal_to_molecule

    rogue = make_cognition_run_signal().model_copy(
        update={"organ_id": "biometrics", "signal_kind": "gpu_load"}
    )

    with pytest.raises(ValueError, match="unsupported substrate signal bridge"):
        signal_to_molecule(rogue)


def test_bridged_molecule_validates_against_default_registry_for_both_kinds():
    from orion.substrate.signal_bridge import signal_to_molecule

    run = make_cognition_run_signal(
        dimensions={"success": 1.0, "step_count": 0.10, "latency_level": 0.10},
    )
    step = make_cognition_run_signal(
        signal_kind="cognition_step",
        dimensions={
            "success": 1.0,
            "latency_level": 0.10,
            "error_present": 0.0,
            "service_count": 0.20,
        },
    )

    for molecule in (signal_to_molecule(run), signal_to_molecule(step)):
        validate_molecule(molecule, default_registry())

        registry = default_registry()
        for role, atom_key in molecule.atoms.items():
            assert registry.has_atom(atom_key), (
                f"role {role!r} maps to unregistered atom {atom_key!r}"
            )
        for relation in molecule.relations:
            assert registry.has_predicate(relation.predicate)


def test_turn_change_signal_converts_to_organ_signal_molecule():
    from orion.memory.turn_change_signal import build_turn_change_signal
    from orion.signals.signal_ids import make_signal_id
    from orion.substrate.signal_bridge import signal_to_molecule

    signal = build_turn_change_signal(
        correlation_id="corr-abc",
        shift_kind="TOPIC",
        novelty_score=0.82,
        confidence=0.91,
    )
    molecule = signal_to_molecule(signal)
    assert molecule.molecule_kind == "organ_signal"
    assert molecule.gradients["novelty"] > 0.5
    assert molecule.gradients["salience"] > 0.5
    assert molecule.provenance["source_event_id"] == "corr-abc"
    assert signal.causal_parents == [make_signal_id("hub", "corr-abc")]
