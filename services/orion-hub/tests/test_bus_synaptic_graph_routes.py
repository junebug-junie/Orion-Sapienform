from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import bus_synaptic_graph_routes  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(bus_synaptic_graph_routes.router)
    return TestClient(app)


class FakeGraphClient:
    """Cypher-string-dispatching fake, mirroring the FakeGraphClient pattern
    used in orion-bus-mirror's own test suite -- scripted per-query-shape
    responses rather than a full graph engine.
    """

    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def graph_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append((cypher, params))
        for marker, rows in self._responses.items():
            if marker in cypher:
                return rows
        return []


@pytest.mark.parametrize(
    "path",
    [
        "/api/bus-synaptic-graph/summary",
        "/api/bus-synaptic-graph/hot-organs",
        "/api/bus-synaptic-graph/hot-edges",
        "/api/bus-synaptic-graph/anomalies",
    ],
)
def test_falkordb_not_configured_returns_503(client: TestClient, path: str) -> None:
    # All four endpoints share the same _client() helper -- confirm the 503
    # guard actually applies to each route, not just /summary.
    with patch.object(bus_synaptic_graph_routes.settings, "FALKORDB_URI", ""):
        r = client.get(path)
    assert r.status_code == 503
    assert r.json()["detail"] == "falkordb_uri_not_configured"


def test_hot_organs_limit_rejects_out_of_range_values(client: TestClient) -> None:
    fake = FakeGraphClient({"channel_out_degree": []})
    with patch.object(bus_synaptic_graph_routes, "_client", return_value=fake):
        too_high = client.get("/api/bus-synaptic-graph/hot-organs?limit=1000")
        too_low = client.get("/api/bus-synaptic-graph/hot-organs?limit=0")
    assert too_high.status_code == 422
    assert too_low.status_code == 422


def test_hot_edges_limit_rejects_out_of_range_values(client: TestClient) -> None:
    fake = FakeGraphClient({"latency_ewma_sec": []})
    with patch.object(bus_synaptic_graph_routes, "_client", return_value=fake):
        r = client.get("/api/bus-synaptic-graph/hot-edges?limit=1000")
    assert r.status_code == 422


def test_summary_returns_all_counts(client: TestClient) -> None:
    fake = FakeGraphClient(
        {
            "(o:Organ)": [{"c": 51}],
            "(c:Channel)": [{"c": 8929}],
            "(v:Verb)": [{"c": 9}],
            "PUBLISHES]->()": [{"c": 8547}],
            "CAUSALLY_FOLLOWED_BY]->()": [{"c": 126}],
            "EXECUTES_VERB]->()": [{"c": 11}],
        }
    )
    with patch.object(bus_synaptic_graph_routes, "_client", return_value=fake):
        r = client.get("/api/bus-synaptic-graph/summary")

    assert r.status_code == 200
    body = r.json()
    assert body["organ_count"] == 51
    assert body["channel_count"] == 8929
    assert body["verb_count"] == 9
    assert body["publishes_edge_count"] == 8547
    assert body["causally_followed_by_edge_count"] == 126
    assert body["executes_verb_edge_count"] == 11


def test_summary_handles_empty_graph_without_error(client: TestClient) -> None:
    fake = FakeGraphClient({})  # every query returns []
    with patch.object(bus_synaptic_graph_routes, "_client", return_value=fake):
        r = client.get("/api/bus-synaptic-graph/summary")

    assert r.status_code == 200
    assert r.json()["organ_count"] == 0


def test_hot_organs_returns_ranked_list(client: TestClient) -> None:
    fake = FakeGraphClient(
        {
            "channel_out_degree": [
                {"organ_id": "vision-host", "channel_out_degree": 6238},
                {"organ_id": "llm-gateway", "channel_out_degree": 654},
            ]
        }
    )
    with patch.object(bus_synaptic_graph_routes, "_client", return_value=fake):
        r = client.get("/api/bus-synaptic-graph/hot-organs")

    assert r.status_code == 200
    organs = r.json()["organs"]
    assert organs[0]["organ_id"] == "vision-host"
    assert organs[0]["channel_out_degree"] == 6238
    # limit param actually threaded through to the query
    _, params = fake.calls[-1]
    assert params["limit"] == 10


def test_hot_organs_limit_param_is_configurable(client: TestClient) -> None:
    fake = FakeGraphClient({"channel_out_degree": []})
    with patch.object(bus_synaptic_graph_routes, "_client", return_value=fake):
        client.get("/api/bus-synaptic-graph/hot-organs?limit=3")

    _, params = fake.calls[-1]
    assert params["limit"] == 3


def test_hot_edges_returns_ranked_list(client: TestClient) -> None:
    # "latency_ewma_sec" is unique to hot_edges's query -- the anomalies
    # endpoint's CAUSALLY_FOLLOWED_BY query returns latency_zscore instead,
    # and both queries otherwise share the same MATCH/RETURN prefix.
    fake = FakeGraphClient(
        {
            "latency_ewma_sec": [
                {
                    "source_organ": "orion-feedback-runtime",
                    "target_organ": "spark-concept-induction",
                    "count": 15731,
                    "latency_ewma_sec": 1.36,
                }
            ]
        }
    )
    with patch.object(bus_synaptic_graph_routes, "_client", return_value=fake):
        r = client.get("/api/bus-synaptic-graph/hot-edges")

    assert r.status_code == 200
    edges = r.json()["edges"]
    assert edges[0]["source_organ"] == "orion-feedback-runtime"
    assert edges[0]["count"] == 15731


def test_anomalies_returns_both_publish_and_causal_lists(client: TestClient) -> None:
    fake = FakeGraphClient(
        {
            "gap_zscore": [{"organ_id": "cortex-exec", "channel": "orion:x", "zscore": 12.4, "count": 40}],
            "latency_zscore": [{"source_organ": "a", "target_organ": "b", "zscore": -8.1, "count": 20}],
        }
    )
    with patch.object(bus_synaptic_graph_routes, "_client", return_value=fake):
        r = client.get("/api/bus-synaptic-graph/anomalies")

    assert r.status_code == 200
    body = r.json()
    assert len(body["publish_gap_anomalies"]) == 1
    assert len(body["causal_latency_anomalies"]) == 1


def test_anomalies_thresholds_are_configurable(client: TestClient) -> None:
    fake = FakeGraphClient({"gap_zscore": [], "latency_zscore": []})
    with patch.object(bus_synaptic_graph_routes, "_client", return_value=fake):
        client.get("/api/bus-synaptic-graph/anomalies?zscore_threshold=5&min_count=10")

    query_params = [p for _, p in fake.calls]
    assert all(p["threshold"] == 5.0 and p["min_count"] == 10 for p in query_params)
