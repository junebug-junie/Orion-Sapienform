"""R1 of the phase-5 roadmap: a dimension's value must name its producer.

docs/superpowers/specs/2026-08-13-phase5-liveness-scope.md

Every fixture below is hand-computed from CHANNEL_DIMENSION_MAP and the two
max() merges, then asserted -- not read back off the implementation. The
expected values are written out longhand in each test's comment so a future
reader can re-derive them without running the code.
"""
from datetime import datetime, timezone

from orion.field.pressure import (
    CHANNEL_DIMENSION_MAP,
    DimensionContributor,
    field_pressures,
    field_pressures_with_provenance,
    map_channels_to_dimensions,
    map_channels_to_dimensions_with_provenance,
)
from orion.schemas.field_state import FieldStateV1

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)


def _state(node_vectors: dict[str, dict[str, float]]) -> FieldStateV1:
    return FieldStateV1(generated_at=NOW, tick_id="tick_r1", node_vectors=node_vectors)


def test_two_channels_into_one_dimension_names_the_winner() -> None:
    """social_pressure is the only dimension fed by two channels today
    (repair_pressure and conversation_load both map to it), so it is the one
    case where the dimension-level max() actually chooses.

    Hand-computed:
      channel merge (max across nodes, both are PRESSURE_CHANNELS):
        repair_pressure    = max(0.30, 0.50) = 0.50, won by node:b
        conversation_load  = max(0.70, 0.20) = 0.70, won by node:a
      dimension merge:
        social_pressure    = max(0.50, 0.70) = 0.70, won by conversation_load
                             -> whose source is node:a
    """
    state = _state(
        {
            "node:a": {"repair_pressure": 0.30, "conversation_load": 0.70},
            "node:b": {"repair_pressure": 0.50, "conversation_load": 0.20},
        }
    )
    dims, detail = field_pressures_with_provenance(state)

    assert dims["social_pressure"] == 0.70
    social = detail["social_pressure"]
    assert social.value == 0.70
    assert social.winning_channel == "conversation_load"
    assert social.winning_source_id == "node:a"

    # The loser survives -- R3 cannot tell a real max() from a walkover
    # without it.
    assert set(social.contributors) == {
        DimensionContributor(channel="repair_pressure", value=0.50, source_id="node:b"),
        DimensionContributor(
            channel="conversation_load", value=0.70, source_id="node:a"
        ),
    }


def test_dimension_provenance_survives_a_walkover() -> None:
    """A single-contributor dimension still names its source.

    Hand-computed: `pressure` -> resource_pressure, 0.42, only contributor,
    won by node:only.
    """
    state = _state({"node:only": {"pressure": 0.42}})
    dims, detail = field_pressures_with_provenance(state)

    assert dims == {"resource_pressure": 0.42}
    resource = detail["resource_pressure"]
    assert resource.winning_channel == "pressure"
    assert resource.winning_source_id == "node:only"
    assert len(resource.contributors) == 1


def test_values_are_identical_to_the_provenance_free_path() -> None:
    """field_pressures() delegates, so the two can never disagree.

    Hand-computed for this fixture:
      staleness 0.25 -> continuity_pressure 0.25
      pressure  0.60 -> resource_pressure   0.60
      execution_pressure 0.90 -> execution_pressure 0.90
      repair_pressure 0.10, conversation_load 0.80 -> social_pressure 0.80
    """
    state = _state(
        {
            "node:a": {
                "staleness": 0.25,
                "pressure": 0.60,
                "execution_pressure": 0.90,
                "repair_pressure": 0.10,
                "conversation_load": 0.80,
            }
        }
    )
    expected = {
        "continuity_pressure": 0.25,
        "resource_pressure": 0.60,
        "execution_pressure": 0.90,
        "social_pressure": 0.80,
    }
    dims, detail = field_pressures_with_provenance(state)

    assert dims == expected
    assert field_pressures(state) == expected
    assert set(detail) == set(expected)
    for dim_id, value in expected.items():
        assert detail[dim_id].value == value


def test_unmapped_channels_produce_no_dimension() -> None:
    """A channel with no CHANNEL_DIMENSION_MAP entry is absent, not 0.0 --
    the "no empty-shell cognition" rule. cpu_pressure is a real, live channel
    that deliberately routes to no dimension."""
    assert "cpu_pressure" not in CHANNEL_DIMENSION_MAP
    dims, detail = field_pressures_with_provenance(
        _state({"node:a": {"cpu_pressure": 0.99}})
    )
    assert dims == {}
    assert detail == {}


def test_ties_resolve_last_wins_matching_the_channel_merge() -> None:
    """Both layers use >= on the running max, so at equal values the
    last-iterated contender wins. Pinned deliberately: if one layer flips to
    strict > the two would silently disagree about which source is
    responsible while still agreeing on the number.

    Hand-computed: repair_pressure and conversation_load both 0.50; insertion
    order puts conversation_load second, so it wins.
    """
    pressures = {"repair_pressure": 0.50, "conversation_load": 0.50}
    provenance = {"repair_pressure": "node:first", "conversation_load": "node:second"}
    dims, detail = map_channels_to_dimensions_with_provenance(pressures, provenance)

    assert dims == {"social_pressure": 0.50}
    assert detail["social_pressure"].winning_channel == "conversation_load"
    assert detail["social_pressure"].winning_source_id == "node:second"


def test_missing_channel_provenance_reads_as_none_not_a_fabricated_source() -> None:
    """Called without a channel-provenance dict (the back-compat path),
    source_id is None. Explicitly NOT a placeholder string -- an unknown
    producer must not be indistinguishable from a named one."""
    dims, detail = map_channels_to_dimensions_with_provenance({"pressure": 0.33})
    assert dims == {"resource_pressure": 0.33}
    assert detail["resource_pressure"].winning_source_id is None
    assert detail["resource_pressure"].contributors[0].source_id is None


def test_empty_field_produces_no_dimensions() -> None:
    dims, detail = field_pressures_with_provenance(_state({}))
    assert dims == {}
    assert detail == {}
    assert map_channels_to_dimensions({}) == {}


def test_values_are_clamped_before_the_dimension_merge() -> None:
    """clamp01 applies to the contributor record too, so a stored value and
    its provenance entry can never report different numbers."""
    dims, detail = map_channels_to_dimensions_with_provenance(
        {"pressure": 1.8}, {"pressure": "node:hot"}
    )
    assert dims == {"resource_pressure": 1.0}
    assert detail["resource_pressure"].value == 1.0
    assert detail["resource_pressure"].contributors[0].value == 1.0
