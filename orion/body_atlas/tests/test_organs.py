from __future__ import annotations

import pytest

from orion.body_atlas.organs import louvain, modularity


def _undirected(edges):
    adj = {}
    for a, b, w in edges:
        adj.setdefault(a, {})[b] = adj.get(a, {}).get(b, 0.0) + w
        adj.setdefault(b, {})[a] = adj.get(b, {}).get(a, 0.0) + w
    return adj


def _two_triangles():
    return _undirected([("a", "b", 1), ("b", "c", 1), ("a", "c", 1),
                        ("x", "y", 1), ("y", "z", 1), ("x", "z", 1), ("c", "x", 1)])


def test_two_bridged_triangles_split_into_two_organs_with_known_q():
    adj = _two_triangles()
    com = louvain(adj)
    assert com["a"] == com["b"] == com["c"]
    assert com["x"] == com["y"] == com["z"]
    assert com["a"] != com["x"]
    # m = 7; each side in = 3, tot = 7  ->  Q = 2 * (3/7 - (7/14)^2)
    assert modularity(adj, com) == pytest.approx(2 * (3 / 7 - 0.25))


def test_one_community_has_zero_modularity():
    adj = _two_triangles()
    assert modularity(adj, {n: 0 for n in adj}) == pytest.approx(0.0)


def test_louvain_is_deterministic_and_keeps_isolated_nodes_single():
    adj = _two_triangles()
    adj["lonely"] = {}
    first, second = louvain(adj), louvain(dict(reversed(list(adj.items()))))
    assert first == second
    assert sum(1 for n in adj if first[n] == first["lonely"]) == 1


def test_empty_weight_graph_is_all_singletons():
    com = louvain({"a": {}, "b": {}})
    assert com["a"] != com["b"]
