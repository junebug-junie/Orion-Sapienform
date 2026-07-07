from __future__ import annotations

from orion.embodiment.worldmap import walkable_tiles


def _map(width, height, blocked):
    """objectTiles indexed [layer][x][y]; -1 = empty. `blocked` is a set of (x,y)."""
    layer = [[(-1 if (x, y) not in blocked else 5) for y in range(height)] for x in range(width)]
    return {"width": width, "height": height, "objectTiles": [layer]}


def test_walkable_excludes_blocked_tiles():
    wm = _map(3, 2, blocked={(1, 0), (2, 1)})
    tiles = walkable_tiles(wm)
    assert (1, 0) not in tiles and (2, 1) not in tiles
    assert (0, 0) in tiles and (0, 1) in tiles and (2, 0) in tiles
    assert len(tiles) == 3 * 2 - 2


def test_walkable_multiple_layers_any_block():
    l0 = [[-1 for _ in range(2)] for _ in range(2)]
    l1 = [[-1 for _ in range(2)] for _ in range(2)]
    l1[0][1] = 9  # block (0,1) via second layer
    wm = {"width": 2, "height": 2, "objectTiles": [l0, l1]}
    tiles = walkable_tiles(wm)
    assert (0, 1) not in tiles
    assert len(tiles) == 3


def test_walkable_empty_on_malformed():
    assert walkable_tiles({}) == set()
    assert walkable_tiles({"width": "x", "height": 2, "objectTiles": []}) == set()
