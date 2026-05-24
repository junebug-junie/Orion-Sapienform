from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1


def test_atom_roundtrip() -> None:
    atom = GrammarAtomV1(
        atom_id="atom:vision:person:01JTEST",
        trace_id="trace:vision:01JTEST",
        atom_type="observation",
        semantic_role="person_presence",
        layer="sensor_semantic",
        dimensions=["visual", "spatial", "epistemic"],
        summary="Possible person detected near doorway",
        confidence=0.72,
    )
    raw = atom.model_dump(mode="json")
    assert GrammarAtomV1.model_validate(raw).atom_id == atom.atom_id


def test_invalid_atom_type_rejected() -> None:
    with pytest.raises(ValidationError):
        GrammarAtomV1(
            atom_id="atom:x",
            trace_id="trace:x",
            atom_type="not_a_real_type",  # type: ignore[arg-type]
            semantic_role="x",
            layer="sensor_raw",
            summary="x",
        )


def test_grammar_event_requires_provenance() -> None:
    now = datetime.now(timezone.utc)
    ev = GrammarEventV1(
        event_id="evt:01JTEST",
        event_kind="atom_emitted",
        trace_id="trace:vision:01JTEST",
        emitted_at=now,
        atom=GrammarAtomV1(
            atom_id="atom:vision:person:01JTEST",
            trace_id="trace:vision:01JTEST",
            atom_type="observation",
            semantic_role="person_presence",
            layer="sensor_semantic",
            summary="Possible person",
        ),
        provenance=GrammarProvenanceV1(
            source_service="orion-vision-retina",
            source_component="detector",
        ),
    )
    assert ev.event_kind == "atom_emitted"
