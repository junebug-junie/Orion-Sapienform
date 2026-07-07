from __future__ import annotations

import random

from orion.embodiment.resolver import resolve_destination
from orion.schemas.embodiment import EmbodimentIntentV1

PLAYERS = [
    {"id": "orion", "position": {"x": 0.0, "y": 0.0}},
    {"id": "juniper", "name": "Juniper", "position": {"x": 10.0, "y": 0.0}},
    {"id": "near", "name": "Near", "position": {"x": 1.0, "y": 1.0}},
]


def _intent(kind, ref=None):
    return EmbodimentIntentV1(kind=kind, source="deliberate", reason="r", correlation_id="c", ref=ref)


def test_idle_is_noop():
    r = resolve_destination(_intent("idle"), orion_player_id="orion", players=PLAYERS)
    assert r.status == "resolved_noop"
    assert r.destination is None


def test_approach_named_player():
    r = resolve_destination(_intent("approach_player", ref="Juniper"), orion_player_id="orion", players=PLAYERS)
    assert r.status == "actuated"
    assert r.destination == {"x": 10.0, "y": 0.0}


def test_approach_nearest_when_ref_omitted():
    r = resolve_destination(_intent("approach_player"), orion_player_id="orion", players=PLAYERS)
    assert r.status == "actuated"
    assert r.destination == {"x": 1.0, "y": 1.0}  # "near" is closest non-Orion


def test_approach_denied_when_alone():
    r = resolve_destination(_intent("approach_player"), orion_player_id="orion",
                            players=[{"id": "orion", "position": {"x": 0.0, "y": 0.0}}])
    assert r.status == "denied"


def test_go_to_location_uses_config():
    r = resolve_destination(_intent("go_to_location", ref="fountain"), orion_player_id="orion",
                            players=PLAYERS, locations={"fountain": {"x": 5.0, "y": 5.0}})
    assert r.status == "actuated"
    assert r.destination == {"x": 5.0, "y": 5.0}


def test_go_to_unknown_location_denied():
    r = resolve_destination(_intent("go_to_location", ref="void"), orion_player_id="orion",
                            players=PLAYERS, locations={})
    assert r.status == "denied"


def test_wander_offsets_within_radius():
    r = resolve_destination(_intent("wander"), orion_player_id="orion", players=PLAYERS,
                            wander_radius=3.0, rng=random.Random(1))
    assert r.status == "actuated"
    assert abs(r.destination["x"]) <= 3.0 and abs(r.destination["y"]) <= 3.0


def test_wander_constrained_to_walkable_tiles():
    # All 8 neighbors walkable, origin (0,0) excluded -> any accepted tile is a neighbor.
    walkable = {(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)} - {(0, 0)}
    r = resolve_destination(_intent("wander"), orion_player_id="orion", players=PLAYERS,
                            wander_radius=1.0, rng=random.Random(1), walkable=walkable)
    assert r.status == "actuated"
    tile = (int(r.destination["x"]), int(r.destination["y"]))
    assert tile in walkable and tile != (0, 0)


def test_wander_noop_when_no_walkable_tile():
    r = resolve_destination(_intent("wander"), orion_player_id="orion", players=PLAYERS,
                            wander_radius=3.0, rng=random.Random(1), walkable=set())
    assert r.status == "resolved_noop"
    assert "walkable" in r.reason


def test_start_conversation_resolves_target():
    r = resolve_destination(_intent("start_conversation", ref="Juniper"), orion_player_id="orion", players=PLAYERS)
    assert r.status == "actuated"
    assert r.ref_player_id == "juniper"
    assert r.destination == {"x": 10.0, "y": 0.0}
