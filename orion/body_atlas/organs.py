"""Organs = communities of the circulation graph (Louvain over Newman modularity), no taxonomy.

Theory anchor: a group of parts is an organ when they exchange more with each other than a random
graph with the same degrees would predict. That is exactly Newman's modularity
``Q = sum_c [ in_c / m - (tot_c / 2m)^2 ]``. Louvain (Blondel et al. 2008) greedily maximizes Q.
Q is reported with every atlas, so "these organs are mush" is a number, not an impression.

Deterministic: nodes are visited in sorted order and ties keep the current community, so the
same graph always yields the same organs. Pure Python; the graph is ~10^2 nodes.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Mapping

# adj[i][j] == adj[j][i] for i != j; adj[i][i] is a self-loop counted once.
Graph = dict[Hashable, dict[Hashable, float]]
_EPS = 1e-12


def _degree(adj: Graph, i) -> float:
    return sum(w for j, w in adj[i].items() if j != i) + 2.0 * adj[i].get(i, 0.0)


def _total_weight(adj: Graph) -> float:
    off = sum(w for i, row in adj.items() for j, w in row.items() if j != i) / 2.0
    return off + sum(row.get(i, 0.0) for i, row in adj.items())


def modularity(adj: Graph, partition: Mapping[Hashable, int]) -> float:
    m = _total_weight(adj)
    if m <= 0:
        return 0.0
    inside: dict[int, float] = defaultdict(float)
    tot: dict[int, float] = defaultdict(float)
    for i, row in adj.items():
        tot[partition[i]] += _degree(adj, i)
        for j, w in row.items():
            if partition[i] != partition[j]:
                continue
            # off-diagonal pairs are seen twice, loops once
            inside[partition[i]] += w if i == j else w / 2.0
    return sum(inside[c] / m - (tot[c] / (2.0 * m)) ** 2 for c in tot)


def _one_level(adj: Graph) -> dict[Hashable, int]:
    nodes = sorted(adj, key=str)
    m2 = 2.0 * _total_weight(adj)
    com = {n: k for k, n in enumerate(nodes)}
    degree = {n: _degree(adj, n) for n in nodes}
    tot = {com[n]: degree[n] for n in nodes}
    moved = True
    while moved:
        moved = False
        for i in nodes:
            ci, ki = com[i], degree[i]
            neigh: dict[int, float] = defaultdict(float)
            for j, w in adj[i].items():
                if j != i:
                    neigh[com[j]] += w
            tot[ci] -= ki
            best, best_gain = ci, neigh.get(ci, 0.0) - tot[ci] * ki / m2
            for c in sorted(neigh):
                gain = neigh[c] - tot[c] * ki / m2
                if gain > best_gain + _EPS:
                    best, best_gain = c, gain
            tot[best] += ki
            if best != ci:
                com[i] = best
                moved = True
    # relabel 0..k-1 by first member in sorted order
    labels: dict[int, int] = {}
    for n in nodes:
        labels.setdefault(com[n], len(labels))
    return {n: labels[com[n]] for n in nodes}


def _aggregate(adj: Graph, com: Mapping[Hashable, int]) -> Graph:
    out: Graph = {c: {} for c in set(com.values())}
    for i, row in adj.items():
        for j, w in row.items():
            if str(j) < str(i):
                continue  # each unordered pair once
            ci, cj = com[i], com[j]
            if ci == cj:
                out[ci][ci] = out[ci].get(ci, 0.0) + w
            else:
                out[ci][cj] = out[ci].get(cj, 0.0) + w
                out[cj][ci] = out[cj].get(ci, 0.0) + w
    return out


def louvain(adj: Graph) -> dict[Hashable, int]:
    """Node -> community index. Nodes with no edges stay singletons."""
    if _total_weight(adj) <= 0:
        return {n: k for k, n in enumerate(sorted(adj, key=str))}
    membership = {n: n for n in adj}  # original node -> node of the current level
    level = adj
    while True:
        com = _one_level(level)
        if len(set(com.values())) == len(level):
            break
        membership = {n: com[c] for n, c in membership.items()}
        level = _aggregate(level, com)
    labels: dict[Hashable, int] = {}
    for n in sorted(adj, key=str):
        labels.setdefault(membership[n], len(labels))
    return {n: labels[membership[n]] for n in adj}
