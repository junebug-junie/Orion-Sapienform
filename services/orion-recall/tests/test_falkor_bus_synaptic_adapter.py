"""fetch_bus_synaptic_anomaly_fragments: fixed anomaly Cypher against
orion_bus_synapse (the same queries already live-verified in
services/orion-hub/scripts/bus_synaptic_graph_routes.py::anomalies()).

Same return-fragment contract as the other Falkor adapters
(id/source/source_ref/uri/text/ts/tags/score/meta) -- fusion.py has no
per-backend adapter layer, so shape drift here would silently break scoring.
"""

from __future__ import annotations

import asyncio
import time

from app.storage import falkor_bus_synaptic_adapter as bsa


class _FakeFalkorClient:
    def __init__(self, publish_rows=None, causal_rows=None):
        self._publish_rows = publish_rows or []
        self._causal_rows = causal_rows or []
        self.calls = []

    def graph_query(self, cypher, params=None):
        self.calls.append((cypher, params))
        if "PUBLISHES" in cypher:
            return self._publish_rows
        if "CAUSALLY_FOLLOWED_BY" in cypher:
            return self._causal_rows
        return []


def _run(coro):
    return asyncio.run(coro)


def _recent_epoch(offset_sec: float = 60.0) -> float:
    return time.time() - offset_sec


def test_no_client_configured_returns_empty_list(monkeypatch) -> None:
    monkeypatch.setattr(bsa, "get_bus_synaptic_falkor_client", lambda: None)
    out = _run(bsa.fetch_bus_synaptic_anomaly_fragments())
    assert out == []


def test_no_anomalies_returns_empty_list_not_a_placeholder_fragment(monkeypatch) -> None:
    fake_client = _FakeFalkorClient(publish_rows=[], causal_rows=[])
    monkeypatch.setattr(bsa, "get_bus_synaptic_falkor_client", lambda: fake_client)

    out = _run(bsa.fetch_bus_synaptic_anomaly_fragments())

    assert out == []


def test_publish_anomaly_produces_a_real_fragment(monkeypatch) -> None:
    fake_client = _FakeFalkorClient(
        publish_rows=[
            {
                "organ_id": "cortex-exec",
                "channel": "orion:cognition:trace",
                "zscore": 5.2,
                "count": 40,
                "last_seen_epoch": _recent_epoch(120),
            }
        ]
    )
    monkeypatch.setattr(bsa, "get_bus_synaptic_falkor_client", lambda: fake_client)

    out = _run(bsa.fetch_bus_synaptic_anomaly_fragments())

    assert len(out) == 1
    frag = out[0]
    assert frag["source"] == "bus_synaptic_anomaly"
    assert frag["source_ref"] == "falkordb"
    assert frag["uri"] == frag["id"]
    assert "cortex-exec" in frag["text"]
    assert "orion:cognition:trace" in frag["text"]
    assert "5.2" in frag["text"]
    assert "inter-arrival gap" in frag["text"]
    assert "not live traffic" in frag["text"]
    assert "not current publish rate" in frag["text"]
    assert frag["tags"] == ["bus_synaptic", "anomaly", "publish_gap", "stale_telemetry_snapshot"]
    assert frag["meta"]["zscore"] == 5.2
    assert frag["meta"]["count"] == 40
    assert frag["meta"]["signal_kind"] == "publish_gap_zscore"
    assert isinstance(frag["ts"], float)


def test_causal_anomaly_produces_a_real_fragment(monkeypatch) -> None:
    fake_client = _FakeFalkorClient(
        causal_rows=[
            {
                "source_organ": "orion-feedback-runtime",
                "target_organ": "spark-concept-induction",
                "zscore": -4.1,
                "count": 200,
                "last_seen_epoch": _recent_epoch(300),
            }
        ]
    )
    monkeypatch.setattr(bsa, "get_bus_synaptic_falkor_client", lambda: fake_client)

    out = _run(bsa.fetch_bus_synaptic_anomaly_fragments())

    assert len(out) == 1
    frag = out[0]
    assert frag["tags"] == ["bus_synaptic", "anomaly", "causal_latency", "stale_telemetry_snapshot"]
    assert "orion-feedback-runtime" in frag["text"]
    assert "spark-concept-induction" in frag["text"]
    assert "not live traffic" in frag["text"]


def test_both_anomaly_kinds_combine_and_respect_max_items(monkeypatch) -> None:
    fake_client = _FakeFalkorClient(
        publish_rows=[
            {
                "organ_id": f"organ-{i}",
                "channel": "c",
                "zscore": 5.0,
                "count": 10,
                "last_seen_epoch": _recent_epoch(30),
            }
            for i in range(3)
        ],
        causal_rows=[
            {
                "source_organ": "a",
                "target_organ": "b",
                "zscore": 4.0,
                "count": 10,
                "last_seen_epoch": _recent_epoch(30),
            }
            for _ in range(3)
        ],
    )
    monkeypatch.setattr(bsa, "get_bus_synaptic_falkor_client", lambda: fake_client)

    out = _run(bsa.fetch_bus_synaptic_anomaly_fragments(max_items=4))

    assert len(out) == 4


def test_rows_missing_zscore_are_skipped_not_crashed(monkeypatch) -> None:
    fake_client = _FakeFalkorClient(
        publish_rows=[{"organ_id": "x", "channel": "c", "zscore": None, "count": 1, "last_seen_epoch": _recent_epoch()}]
    )
    monkeypatch.setattr(bsa, "get_bus_synaptic_falkor_client", lambda: fake_client)

    out = _run(bsa.fetch_bus_synaptic_anomaly_fragments())

    assert out == []


def test_graph_query_exception_fails_open_to_empty_list(monkeypatch) -> None:
    class _RaisingClient:
        def graph_query(self, cypher, params=None):
            raise RuntimeError("falkor unreachable")

    monkeypatch.setattr(bsa, "get_bus_synaptic_falkor_client", lambda: _RaisingClient())

    out = _run(bsa.fetch_bus_synaptic_anomaly_fragments())

    assert out == []


def test_thresholds_and_recency_are_threaded_through_to_both_queries(monkeypatch) -> None:
    fake_client = _FakeFalkorClient()
    monkeypatch.setattr(bsa, "get_bus_synaptic_falkor_client", lambda: fake_client)
    before = time.time()

    _run(
        bsa.fetch_bus_synaptic_anomaly_fragments(
            zscore_threshold=5.0,
            min_count=10,
            max_edge_age_sec=7200.0,
        )
    )

    assert len(fake_client.calls) == 2
    for _, params in fake_client.calls:
        assert params["threshold"] == 5.0
        assert params["min_count"] == 10
        assert params["min_last_seen_epoch"] >= before - 7200.0 - 1.0
        assert params["min_last_seen_epoch"] <= before - 7200.0 + 1.0
        assert "last_seen_epoch" in fake_client.calls[0][0]


def test_get_bus_synaptic_falkor_client_returns_none_without_falkordb_uri(monkeypatch) -> None:
    bsa._CLIENT = None
    monkeypatch.delenv("FALKORDB_URI", raising=False)
    assert bsa.get_bus_synaptic_falkor_client() is None


def test_get_bus_synaptic_falkor_client_is_a_lazy_singleton(monkeypatch) -> None:
    bsa._CLIENT = None
    monkeypatch.setenv("FALKORDB_URI", "redis://localhost:6379")
    first = bsa.get_bus_synaptic_falkor_client()
    second = bsa.get_bus_synaptic_falkor_client()
    assert first is second
    bsa._CLIENT = None


def _fusion_profile() -> dict:
    return {
        "profile": "reflect.v1",
        "max_per_source": 10,
        "max_total_items": 20,
        "render_budget_tokens": 256,
        "time_decay_half_life_hours": 72,
        "relevance": {"backend_weights": {"bus_synaptic_anomaly": 1.0}},
    }


def test_multiple_real_anomalies_all_survive_fuse_candidates(monkeypatch) -> None:
    from app.fusion import fuse_candidates

    fake_client = _FakeFalkorClient(
        publish_rows=[
            {
                "organ_id": "cortex-orch",
                "channel": "orion:cortex:result*",
                "zscore": 7.1,
                "count": 10,
                "last_seen_epoch": _recent_epoch(45),
            }
        ],
        causal_rows=[
            {
                "source_organ": "actions",
                "target_organ": "spark-concept-induction",
                "zscore": 11.5,
                "count": 8,
                "last_seen_epoch": _recent_epoch(45),
            },
            {
                "source_organ": "orion-thought",
                "target_organ": "landing-pad",
                "zscore": 8.6,
                "count": 232,
                "last_seen_epoch": _recent_epoch(45),
            },
        ],
    )
    monkeypatch.setattr(bsa, "get_bus_synaptic_falkor_client", lambda: fake_client)

    fragments = _run(bsa.fetch_bus_synaptic_anomaly_fragments())
    assert len(fragments) == 3

    bundle, _ = fuse_candidates(candidates=fragments, profile=_fusion_profile(), diagnostic=True)

    assert len(bundle.items) == 3
    surviving_ids = {item.id for item in bundle.items}
    assert surviving_ids == {f["id"] for f in fragments}
