"""Direction map loading. The real shipped YAML is exercised, not only fixtures --
a gate whose map silently resolved to nothing would still report a clean run."""
from __future__ import annotations

import pytest

from orion.attention.tension.direction_map import (
    DirectionMapError,
    load_direction_map,
)


def _write(tmp_path, body: str):
    path = tmp_path / "map.yaml"
    path.write_text(body, encoding="utf-8")
    return path


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


def test_empty_map_refuses_to_load(tmp_path):
    with pytest.raises(DirectionMapError, match="no rules"):
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
