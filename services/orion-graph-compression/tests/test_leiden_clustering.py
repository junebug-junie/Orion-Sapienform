import pytest
import networkx as nx


def test_leiden_cluster_small_graph():
    """Three connected triples form at least one community."""
    from app.clustering.leiden import leiden_cluster

    G = nx.Graph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "A")
    G.add_edge("D", "E")
    G.add_edge("E", "F")
    G.add_edge("F", "D")

    communities = leiden_cluster(G, resolution=1.0, n_iterations=2)
    assert len(communities) >= 1
    all_nodes = set().union(*communities)
    assert all_nodes == set(G.nodes)


def test_leiden_cluster_empty_graph_returns_empty():
    from app.clustering.leiden import leiden_cluster
    G = nx.Graph()
    communities = leiden_cluster(G, resolution=1.0, n_iterations=2)
    assert communities == []


def test_leiden_cluster_single_node():
    from app.clustering.leiden import leiden_cluster
    G = nx.Graph()
    G.add_node("solo")
    communities = leiden_cluster(G, resolution=1.0, n_iterations=2)
    assert communities == [{"solo"}]


def test_build_graph_from_triples():
    from app.clustering.leiden import build_graph_from_triples
    triples = [
        ("http://A", "http://rel", "http://B"),
        ("http://B", "http://rel", "http://C"),
    ]
    G = build_graph_from_triples(triples)
    assert G.has_node("http://A")
    assert G.has_edge("http://A", "http://B")
    assert len(G.nodes) == 3
