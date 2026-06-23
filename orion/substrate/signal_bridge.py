"""Substrate signal bridge — project OrionSignalV1 into SubstrateMoleculeV1.

This module is a pure projection. It does not subscribe to a bus, does not
persist anything, and does not introduce a new signal schema. It exists so the
substrate can consume the existing organ signal stream through one well-defined
seam.

Supported inputs (MVP):
    (organ_id="cortex_exec", signal_kind="cognition_run")
    (organ_id="cortex_exec", signal_kind="cognition_step")
    (organ_id="memory_consolidation", signal_kind="turn_change")

These are the signals produced by orion.signals.adapters.cognition_trace.
"""

from __future__ import annotations

from typing import Iterable

from orion.schema_kernel import (
    ConceptRelationV1,
    clamp_gradient,
    default_registry,
)
from orion.signals.models import OrionSignalV1
from orion.substrate.molecules import SubstrateMoleculeV1, validate_molecule


SUPPORTED_SIGNAL_KINDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("cortex_exec", "cognition_run"),
        ("cortex_exec", "cognition_step"),
        ("memory_consolidation", "turn_change"),
    }
)


def supports_signal(signal: OrionSignalV1) -> bool:
    return (signal.organ_id, signal.signal_kind) in SUPPORTED_SIGNAL_KINDS


def _dim(signal: OrionSignalV1, key: str, default: float = 0.0) -> float:
    value = (signal.dimensions or {}).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def signal_intensity(signal: OrionSignalV1) -> float:
    return clamp_gradient(
        max(
            _dim(signal, "salience"),
            _dim(signal, "level"),
            _dim(signal, "latency_level"),
            _dim(signal, "step_count"),
            _dim(signal, "service_count"),
        )
    )


def signal_confidence(signal: OrionSignalV1) -> float:
    return clamp_gradient(_dim(signal, "confidence", 1.0))


def dimensions_to_gradients(signal: OrionSignalV1) -> dict[str, float]:
    dims = signal.dimensions or {}
    success_present = "success" in dims
    success = _dim(signal, "success", 0.0)

    salience = signal_intensity(signal)

    contradiction = max(
        _dim(signal, "contradiction"),
        _dim(signal, "error_present"),
        1.0 - success if success_present else 0.0,
    )

    novelty = max(
        _dim(signal, "novelty"),
        _dim(signal, "surprise"),
    )

    coherence = max(
        _dim(signal, "coherence"),
        success if success_present else 0.0,
    )

    confidence = signal_confidence(signal)

    return {
        "salience": clamp_gradient(salience),
        "contradiction": clamp_gradient(contradiction),
        "novelty": clamp_gradient(novelty),
        "coherence": clamp_gradient(coherence * confidence),
    }


def signal_to_molecule(signal: OrionSignalV1) -> SubstrateMoleculeV1:
    if not supports_signal(signal):
        raise ValueError(
            f"unsupported substrate signal bridge: "
            f"{signal.organ_id}.{signal.signal_kind}"
        )

    intensity = signal_intensity(signal)
    confidence = signal_confidence(signal)

    molecule = SubstrateMoleculeV1(
        molecule_kind="organ_signal",
        atoms={
            "primary": "signal",
            "source_process": "agency",
            "source_context": "context",
            "field": "gradient",
            "witness": "evidence",
        },
        relations=[
            ConceptRelationV1(
                source="primary",
                predicate="references",
                target="source_process",
                weight=1.0,
            ),
            ConceptRelationV1(
                source="primary",
                predicate="elicits",
                target="field",
                weight=intensity,
            ),
            ConceptRelationV1(
                source="witness",
                predicate="supports",
                target="primary",
                weight=confidence,
                polarity=1.0,
            ),
        ],
        gradients=dimensions_to_gradients(signal),
        provenance={
            "organ": signal.organ_id,
            "signal_id": signal.signal_id,
            "signal_kind": signal.signal_kind,
            "source_event_id": signal.source_event_id,
            "otel_trace_id": signal.otel_trace_id,
            "otel_span_id": signal.otel_span_id,
        },
        payload={
            "dimensions": dict(signal.dimensions or {}),
            "summary": signal.summary,
            "notes": list(signal.notes or []),
            "causal_parents": list(signal.causal_parents or []),
            "observed_at": signal.observed_at.isoformat(),
            "emitted_at": signal.emitted_at.isoformat(),
        },
    )

    validate_molecule(molecule, default_registry())
    return molecule


def signals_to_molecules(
    signals: Iterable[OrionSignalV1],
) -> list[SubstrateMoleculeV1]:
    return [signal_to_molecule(s) for s in signals if supports_signal(s)]
