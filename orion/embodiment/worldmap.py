"""Walkability derived from an AI Town ``worldMap``.

AI Town stores object/collision layers in ``objectTiles``, indexed
``[layer][x][y]`` where ``-1`` means "no object" (walkable floor) and any other
value is an object occupying that tile (blocked). A tile is walkable only when it
is inside the map bounds and no object layer occupies it.

Used to keep involuntary wander destinations on reachable floor — an unconstrained
random offset frequently lands in a wall, and AI Town silently drops a ``moveTo``
to an unreachable tile, so Orion never moves.
"""
from __future__ import annotations

from typing import Any


def walkable_tiles(world_map: dict[str, Any]) -> set[tuple[int, int]]:
    """Return the set of walkable integer tiles ``(x, y)`` for a worldMap.

    Fail-open: returns an empty set if the map shape is missing/malformed, which
    callers treat as "no map knowledge" and fall back to unconstrained behavior.
    """
    try:
        width = int(world_map["width"])
        height = int(world_map["height"])
    except (KeyError, TypeError, ValueError):
        return set()
    layers = world_map.get("objectTiles") or []
    walkable: set[tuple[int, int]] = set()
    for x in range(width):
        for y in range(height):
            blocked = False
            for layer in layers:
                try:
                    if layer[x][y] != -1:
                        blocked = True
                        break
                except (IndexError, TypeError):
                    continue
            if not blocked:
                walkable.add((x, y))
    return walkable
