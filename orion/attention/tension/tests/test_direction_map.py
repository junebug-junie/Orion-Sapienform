"""Direction map loading. The real shipped YAML is exercised, not only fixtures --
a gate whose map silently resolved to nothing would still report a clean run."""
from __future__ import annotations

import pytest

from orion.attention.tension.direction_map import (
    DirectionMapError,
    load_direction_map,
)
from orion.field.pressure import HIGHER_IS_BETTER_CHANNELS


def _write(tmp_path, body: str):
    path = tmp_path / "map.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_higher_is_better_polarity_is_derived_not_hand_listed():
    """Regression 2026-08-14. The first version hand-listed availability/
    delivery_confidence/stream_backlog_health in the YAML -- a strict SUBSET of
    `orion.field.pressure.HIGHER_IS_BETTER_CHANNELS`, silently missing
    `confidence` and `available_capacity`, which would have had this module
    disagreeing with `orion/attention/field_attention/selectors.py` about two
    channels' polarity."""
    dm = load_direction_map()
    for channel in HIGHER_IS_BETTER_CHANNELS:
        assert dm.worse_for(channel) == "down", channel
    # And the ones the hand-written subset missed, named explicitly so a future
    # edit that drops the derivation fails here loudly.
    assert dm.worse_for("confidence") == "down"
    assert dm.worse_for("available_capacity") == "down"


def test_yaml_may_not_contradict_the_polarity_constant(tmp_path):
    body = "suffix_rules:\n  '*_pressure': up\nchannels:\n  availability: up\n"
    with pytest.raises(DirectionMapError, match="HIGHER_IS_BETTER_CHANNELS"):
        load_direction_map(_write(tmp_path, body))


def test_ships_a_real_map_that_covers_the_live_channels():
    """Against the real config, not a synthetic fixture: these are the channel
    names actually present in `substrate_field_state.node_vectors` on
    2026-08-14. A map that stopped covering them would be inert in production
    while every unit test still passed."""
    dm = load_direction_map()
    assert dm.worse_for("cpu_pressure") == "up"
    assert dm.worse_for("reasoning_load") == "up"
    assert dm.worse_for("egress_confidence_deficit") == "up"
    assert dm.worse_for("execution_friction") == "up"
    assert dm.worse_for("prediction_error") == "up"
    assert dm.worse_for("field_coherence_warning") == "up"
    assert dm.worse_for("turn_incompletion") == "up"
    assert dm.worse_for("staleness") == "up"
    assert dm.worse_for("availability") == "down"
    assert dm.worse_for("delivery_confidence") == "down"
    assert dm.worse_for("stream_backlog_health") == "down"


def test_deliberately_unmapped_channels_do_not_vote():
    dm = load_direction_map()
    assert dm.worse_for("context_gathering_ratio") is None
    assert dm.worse_for("expected_offline_suppression") is None


def test_unknown_channel_is_inert_rather_than_defaulted():
    assert load_direction_map().worse_for("some_future_channel") is None


def test_exact_entry_beats_suffix_rule(tmp_path):
    path = _write(
        tmp_path,
        "suffix_rules:\n  '*_health': up\nchannels:\n  stream_backlog_health: down\n",
    )
    dm = load_direction_map(path)
    assert dm.worse_for("stream_backlog_health") == "down"
    assert dm.worse_for("other_health") == "up"


def test_longest_suffix_wins(tmp_path):
    path = _write(
        tmp_path,
        "suffix_rules:\n  '*_pressure': up\n  '*_confidence_pressure': down\n",
    )
    dm = load_direction_map(path)
    assert dm.worse_for("egress_confidence_pressure") == "down"
    assert dm.worse_for("cpu_pressure") == "up"


def test_map_contributing_no_rules_of_its_own_refuses_to_load(tmp_path):
    """The guard is on the YAML's OWN contribution. `exact` is seeded from
    HIGHER_IS_BETTER_CHANNELS, so a merged-map emptiness check would be
    permanently satisfied and therefore inert -- exactly the class of dead gate
    this repo has been bitten by before."""
    with pytest.raises(DirectionMapError, match="contributes no rules of its own"):
        load_direction_map(_write(tmp_path, "version: 1\n"))


def test_missing_file_raises(tmp_path):
    with pytest.raises(DirectionMapError, match="not found"):
        load_direction_map(tmp_path / "absent.yaml")


def test_invalid_direction_raises(tmp_path):
    with pytest.raises(DirectionMapError, match="not in"):
        load_direction_map(_write(tmp_path, "channels:\n  cpu_pressure: sideways\n"))


def test_suffix_rule_without_star_raises(tmp_path):
    with pytest.raises(DirectionMapError, match="must start with"):
        load_direction_map(_write(tmp_path, "suffix_rules:\n  _pressure: up\n"))


def test_channel_both_mapped_and_unmapped_raises(tmp_path):
    body = "channels:\n  cpu_pressure: up\nunmapped:\n  - cpu_pressure\n"
    with pytest.raises(DirectionMapError, match="both mapped and unmapped"):
        load_direction_map(_write(tmp_path, body))


# ---------------------------------------------------------------------------
# Regression: malformed sections must fail loudly, not silently (review)
# ---------------------------------------------------------------------------


def test_unmapped_as_scalar_raises_instead_of_becoming_a_set_of_characters(tmp_path):
    """Caught in review 2026-08-14. `unmapped: context_gathering_ratio` (a scalar
    instead of a list) silently became `frozenset({'_','n','a','i','r',...})`, so
    the channel it named stayed mapped and kept voting -- a quietly-wrong map,
    which is worse than a missing one because the run still reports clean."""
    body = "channels:\n  cpu_pressure: up\nunmapped: context_gathering_ratio\n"
    with pytest.raises(DirectionMapError, match="'unmapped' must be a list"):
        load_direction_map(_write(tmp_path, body))


def test_channels_as_list_raises_direction_map_error_not_attribute_error(tmp_path):
    body = "channels:\n  - cpu_pressure\n"
    with pytest.raises(DirectionMapError, match="'channels' must be a dict"):
        load_direction_map(_write(tmp_path, body))


def test_suffix_rules_as_list_raises(tmp_path):
    body = "suffix_rules:\n  - '*_pressure'\n"
    with pytest.raises(DirectionMapError, match="'suffix_rules' must be a dict"):
        load_direction_map(_write(tmp_path, body))


def test_non_mapping_top_level_raises(tmp_path):
    with pytest.raises(DirectionMapError, match="top level must be a mapping"):
        load_direction_map(_write(tmp_path, "- just\n- a\n- list\n"))
