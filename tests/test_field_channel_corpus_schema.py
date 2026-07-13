from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.telemetry.field_channel_corpus import FieldChannelCorpusRowV1
from orion.self_state.inner_state_registry import REGISTRY, CompositionStatus, get


def _row_kwargs() -> dict:
    return dict(
        generated_at=datetime(2026, 7, 13, tzinfo=timezone.utc),
        tick_id="tick_test123",
        channels={"cpu_pressure": 0.3, "gpu_pressure": 0.7, "recent_perturbation_count": 0.0},
    )


def test_field_channel_corpus_row_round_trips() -> None:
    row = FieldChannelCorpusRowV1(**_row_kwargs())
    restored = FieldChannelCorpusRowV1.model_validate_json(row.model_dump_json())
    assert restored == row


def test_field_channel_corpus_row_forbids_unexpected_kwarg() -> None:
    kwargs = _row_kwargs()
    kwargs["not_a_real_field"] = "nope"
    with pytest.raises(ValidationError):
        FieldChannelCorpusRowV1(**kwargs)


def test_field_channel_corpus_row_channels_width_is_variable() -> None:
    # Row width is NOT fixed -- the channel set can vary tick to tick.
    # Confirm the schema accepts both a wide and a narrow channels dict
    # without complaint (no fixed-key validation on `channels`).
    wide = FieldChannelCorpusRowV1(
        generated_at=datetime(2026, 7, 13, tzinfo=timezone.utc),
        tick_id="tick_wide",
        channels={f"channel_{i}": float(i) for i in range(20)},
    )
    narrow = FieldChannelCorpusRowV1(
        generated_at=datetime(2026, 7, 13, tzinfo=timezone.utc),
        tick_id="tick_narrow",
        channels={},
    )
    assert len(wide.channels) == 20
    assert narrow.channels == {}


def test_field_channel_corpus_registered_in_inner_state_registry() -> None:
    # NOT in orion/schemas/registry.py -- that's the general-purpose
    # dynamic-dispatch registry; this is file-only training-corpus data,
    # same as its mood_arc_corpus.v1 sibling (which is also absent from a
    # dedicated registry-resolution test of that kind -- this mirrors
    # tests/test_mood_arc_encoder_schema.py's *pattern* of asserting a
    # resolvable registry entry, applied to orion/self_state/
    # inner_state_registry.py instead, which is the actual registry this
    # schema belongs to).
    entry = get("field_channel_corpus.v1")
    assert entry.schema is FieldChannelCorpusRowV1
    assert entry.producer_service == "orion-field-digester"
    assert entry.composition_status is CompositionStatus.REHEARSAL
    assert entry.cognition_consumers == ()
    assert entry in REGISTRY
