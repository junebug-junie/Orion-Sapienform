from __future__ import annotations

import logging
from typing import List, Set, Tuple

import networkx as nx

logger = logging.getLogger("orion.graph-compression.clustering.leiden")

Triple = Tuple[str, str, str]


def build_graph_from_triples(triples: List[Triple]) -> nx.Graph:
    G = nx.Graph()
    for s, _p, o in triples:
        # Only add subject↔object edges (predicate is edge label, not node)
        if s and o and s != o:
            G.add_edge(s, o)
    return G


def leiden_cluster(
    G: nx.Graph,
    *,
    resolution: float = 1.0,
    n_iterations: int = 10,
) -> List[Set[str]]:
    if G.number_of_nodes() == 0:
        return []
    if G.number_of_nodes() == 1:
        return [set(G.nodes)]
    try:
        import igraph as ig
        import leidenalg

        # Map networkx → igraph
        nodes = list(G.nodes)
        node_index = {n: i for i, n in enumerate(nodes)}
        edges = [(node_index[u], node_index[v]) for u, v in G.edges()]
        ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)

        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            n_iterations=n_iterations,
        )
        return [
            {nodes[i] for i in community}
            for community in partition
            if len(community) > 0
        ]
    except Exception as exc:
        logger.warning("leiden_cluster_failed reason=%s — falling back to connected components", exc)
        return [set(c) for c in nx.connected_components(G)]
