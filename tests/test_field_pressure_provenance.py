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
    collect_field_channel_pressures,
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


def test_winner_is_the_max_not_the_last_channel_iterated() -> None:
    """The mutant this suite was originally blind to (found by code review):
    attributing a dimension to the last channel the loop touched rather than
    to the one that actually won. Every other fixture here happens to make
    those the same channel, so none of them catch it.

    The two contributors MUST come from different nodes. A first attempt at
    this fixture put both on node:a, and the mutant survived it: looking up
    the wrong channel still returned the right source, because there was only
    one source. The source has to differ for the assertion to bite.

    Hand-computed:
      channel merge: repair_pressure   = 0.80, won by node:a
                     conversation_load = 0.20, won by node:b
      insertion order puts conversation_load LAST, repair_pressure wins on
      value:
      social_pressure = max(0.80, 0.20) = 0.80, won by repair_pressure
                        -> whose source is node:a, NOT node:b
    """
    state = _state(
        {"node:a": {"repair_pressure": 0.80}, "node:b": {"conversation_load": 0.20}}
    )
    dims, detail = field_pressures_with_provenance(state)

    assert dims["social_pressure"] == 0.80
    social = detail["social_pressure"]
    assert social.winning_channel == "repair_pressure"
    # The load-bearing assertion: node:b is the last channel's source.
    assert social.winning_source_id == "node:a"
    # The winner is the FIRST contributor here, not the last.
    assert social.contributors[0].channel == "repair_pressure"
    assert social.contributors[-1].channel == "conversation_load"
    assert social.winner_is_unique is True


def test_channel_layer_tie_break_is_last_source_wins() -> None:
    """Pins the CHANNEL-layer tie-break through the real
    collect_field_channel_pressures() merge, not just the dimension layer.

    The dataclass docstring claims both layers share a `>=` last-wins
    convention. Code review showed nothing in the repo actually pinned the
    channel layer -- flipping its `>=` to `>` left 55 related tests green.

    Hand-computed: `pressure` is in PRESSURE_CHANNELS, both nodes read 0.50,
    so the merge stores 0.50 and provenance goes to whichever node iterated
    last -- node:second.
    """
    state = _state({"node:first": {"pressure": 0.50}, "node:second": {"pressure": 0.50}})
    channels, channel_provenance = collect_field_channel_pressures(state)

    assert channels["pressure"] == 0.50
    assert channel_provenance["pressure"] == "node:second"

    _dims, detail = field_pressures_with_provenance(state)
    assert detail["resource_pressure"].winning_source_id == "node:second"


def test_tied_winner_is_flagged_as_not_unique() -> None:
    """A tie names a winner chosen by dict order, which is not a fact about
    the world. R3 tallies win rates, so the degenerate case has to be
    detectable rather than merely documented.

    Hand-computed: both social channels read 0.0 -- a calm tick. Both are in
    PRESSURE_CHANNELS so 0.0 is stored, not filtered. social_pressure = 0.0
    with two tied contributors.
    """
    state = _state({"node:a": {"repair_pressure": 0.0, "conversation_load": 0.0}})
    dims, detail = field_pressures_with_provenance(state)

    assert dims["social_pressure"] == 0.0
    social = detail["social_pressure"]
    assert len(social.contributors) == 2
    assert social.winner_is_unique is False

    # ...and the genuine single-winner case is still reported as unique.
    contested = _state({"node:a": {"repair_pressure": 0.4, "conversation_load": 0.9}})
    _d, contested_detail = field_pressures_with_provenance(contested)
    assert contested_detail["social_pressure"].winner_is_unique is True


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
      reasoning_pressure 0.35 -> reasoning_pressure 0.35
      reliability_pressure 0.15 -> reliability_pressure 0.15
      egress_confidence_deficit 0.55 -> introspection_pressure 0.55
      repair_pressure 0.10, conversation_load 0.80 -> social_pressure 0.80

    Covers all 7 dimensions CHANNEL_DIMENSION_MAP can produce, so a routing
    entry silently disappearing shows up here.
    """
    state = _state(
        {
            "node:a": {
                "staleness": 0.25,
                "pressure": 0.60,
                "execution_pressure": 0.90,
                "reasoning_pressure": 0.35,
                "reliability_pressure": 0.15,
                "egress_confidence_deficit": 0.55,
                "repair_pressure": 0.10,
                "conversation_load": 0.80,
            }
        }
    )
    expected = {
        "continuity_pressure": 0.25,
        "resource_pressure": 0.60,
        "execution_pressure": 0.90,
        "reasoning_pressure": 0.35,
        "reliability_pressure": 0.15,
        "introspection_pressure": 0.55,
        "social_pressure": 0.80,
    }
    assert set(expected) == set(CHANNEL_DIMENSION_MAP.values())
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
