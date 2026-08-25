import pytest
from pydantic import ValidationError

from orion.schemas.attention_frame import OpenLoopV1, SalienceFeaturesV1
from orion.schemas.registry import resolve


def test_salience_features_defaults_are_bounded():
    f = SalienceFeaturesV1()
    assert f.evidence_strength == 0.0
    assert f.evidence_breadth == 0.0
    dumped = f.model_dump(mode="json")
    # Trimmed 2026-07-31: recurrence/recency/novelty_vs_known/dwell/
    # habituation killed with nothing put back -- see
    # orion.substrate.attention.salience's module docstring.
    assert set(dumped) == {"schema_version", "evidence_strength", "evidence_breadth"}


def test_open_loop_carries_salience_fields():
    loop = OpenLoopV1(id="open-loop-x", description="thing")
    assert loop.salience == 0.0
    assert loop.salience_features == {}


def test_salience_features_registered():
    assert resolve("SalienceFeaturesV1") is SalienceFeaturesV1


# --------------------------------------------------------------------------
# emotional_charge removal, 2026-08-25 (code review finding): OpenLoopV1
# nests inside AttentionBroadcastProjectionV1, which is persisted as JSONB
# and replayed via strict model_validate() -- extra="forbid" alone would
# make removing a field a breaking change for every already-stored row
# that still carries it (168h append-only history +
# scripts/analysis/measure_ast_hot_reducer.py's replay tool).
# --------------------------------------------------------------------------


def test_a_stored_row_still_carrying_the_removed_field_still_parses():
    """The exact shape a pre-2026-08-25 substrate_attention_broadcast_log
    row has on disk right now -- this must keep parsing, not raise."""
    stored_payload = {
        "id": "open-loop-legacy",
        "description": "thing",
        "novelty": 0.2,
        "emotional_charge": 0.65,  # the removed field, still on old rows
    }
    loop = OpenLoopV1.model_validate(stored_payload)
    assert loop.id == "open-loop-legacy"
    assert not hasattr(loop, "emotional_charge")


def test_a_genuinely_unknown_field_still_raises():
    """extra=\"forbid\" still does its real job for anything that isn't a
    known-removed legacy key -- the backward-compat carve-out is narrow,
    not a blanket loosening to extra=\"ignore\"."""
    with pytest.raises(ValidationError):
        OpenLoopV1.model_validate({"id": "x", "description": "y", "totally_made_up_field": 1})


def test_constructing_fresh_with_the_removed_field_is_silently_stripped_not_rejected():
    """A pydantic mode="before" validator can't distinguish "this dict came
    from a historical JSONB row" from "this is a live kwarg call a producer
    forgot to update" -- both go through the same strip. Accepted tradeoff:
    the sole live producer (build_open_loops()) was already updated in this
    same patch, so there is no real call site left that could silently rely
    on this; documented here so the behavior is asserted, not assumed."""
    loop = OpenLoopV1(id="x", description="y", emotional_charge=0.5)
    assert not hasattr(loop, "emotional_charge")
