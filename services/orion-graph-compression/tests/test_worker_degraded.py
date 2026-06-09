import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch


def _make_worker(enable=True):
    with patch("app.settings.get_settings") as mock_settings:
        s = MagicMock()
        s.enable_compression_runtime = enable
        s.compression_batch_size = 5
        s.compression_poll_interval_sec = 0.01
        s.compression_max_tokens_per_summary = 200
        s.compression_llm_budget_per_tick = 5000
        s.compression_max_age_sec = 86400
        s.compression_policy_path = "config/compression_policy.v1.yaml"
        s.rdf_store_query_url = "http://fuseki/query"
        s.rdf_store_update_url = "http://fuseki/update"
        s.rdf_store_user = "admin"
        s.rdf_store_pass = "orion"
        s.rdf_store_timeout_sec = 5.0
        s.llm_gateway_bus_channel = "orion:exec:request:LLMGatewayService"
        s.channel_graph_compression_events = "orion:graph:compression:events"
        s.service_name = "orion-graph-compression"
        s.service_version = "0.1.0"
        mock_settings.return_value = s

        from app.worker import CompressionWorker
        store = MagicMock()
        bus = MagicMock()
        worker = CompressionWorker(store=store, bus=bus)
        return worker, store, bus


def test_worker_disabled_skips_tick():
    """When ENABLE_COMPRESSION_RUNTIME=false the tick does nothing."""
    worker, store, bus = _make_worker(enable=False)
    worker._tick()
    store.drain_stale_queue.assert_not_called()


def test_worker_empty_federators_no_crash():
    """All federators returning [] must not raise — just skip region."""
    worker, store, bus = _make_worker(enable=True)

    store.drain_stale_queue.return_value = [
        {"id": 1, "region_id": None, "scope": "episodic", "reason": "test", "priority": 0}
    ]

    with patch("app.worker.EpisodicFederator") as mock_ep, \
         patch("app.worker.SubstrateFederator") as mock_sub, \
         patch("app.worker.SelfStudyFederator") as mock_ss:
        for mock_cls in [mock_ep, mock_sub, mock_ss]:
            instance = MagicMock()
            instance.fetch.return_value = []
            mock_cls.return_value = instance

        # Should complete without raising
        worker._tick()
        store.delete_stale_queue_items.assert_called_once_with([1])


def test_worker_budget_gate_halts_mid_batch():
    """If LLM token budget is 0, no summarization occurs but regions are still processed structurally."""
    worker, store, bus = _make_worker(enable=True)
    worker._settings.compression_llm_budget_per_tick = 0

    store.drain_stale_queue.return_value = [
        {"id": 1, "region_id": None, "scope": "episodic", "reason": "test", "priority": 0}
    ]

    # Federator returns some triples
    fake_triples = [("http://A", "http://rel", "http://B"), ("http://B", "http://rel", "http://C")]

    with patch("app.worker.EpisodicFederator") as mock_ep, \
         patch("app.worker.SubstrateFederator") as mock_sub, \
         patch("app.worker.SelfStudyFederator") as mock_ss:
        mock_ep.return_value.fetch.return_value = fake_triples
        mock_sub.return_value.fetch.return_value = []
        mock_ss.return_value.fetch.return_value = []

        worker._tick()
        # Should have processed without calling LLM (structural fallback used)
        store.upsert_artifact.assert_called()


def test_substrate_scope_labels_clusters_hotspot_not_contradiction():
    """Substrate clusters must default to 'hotspot' so we don't spuriously flood
    substrate mutation pressure (which only fires for 'contradiction')."""
    worker, store, bus = _make_worker(enable=True)
    store.drain_stale_queue.return_value = [
        {"id": 1, "region_id": None, "scope": "substrate", "reason": "test", "priority": 0}
    ]
    triangle = [
        ("http://A", "http://rel", "http://B"),
        ("http://B", "http://rel", "http://C"),
        ("http://C", "http://rel", "http://A"),
    ]
    with patch("app.worker.EpisodicFederator") as mock_ep, \
         patch("app.worker.SubstrateFederator") as mock_sub, \
         patch("app.worker.SelfStudyFederator") as mock_ss, \
         patch("app.writer.CompressionWriter") as mock_writer:
        mock_ep.return_value.fetch.return_value = []
        mock_sub.return_value.fetch.return_value = triangle
        mock_ss.return_value.fetch.return_value = []
        mock_writer.return_value.write.return_value = True

        worker._tick()

        store.upsert_artifact.assert_called()
        kinds = {c.kwargs["kind"] for c in store.upsert_artifact.call_args_list}
        assert kinds == {"hotspot"}
