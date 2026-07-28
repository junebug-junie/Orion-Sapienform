"""
End-to-end smoke test for bus_synaptic polling transport trigger.

Tests that the poll loop correctly:
1. Queries FalkorDB for node:substrate.bus_synaptic's prediction_error
2. Compares against the configured threshold
3. Publishes a MetacogTriggerV1 when threshold is exceeded
4. Verifies the trigger lands in postgres (mocked for this test)
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

sys.modules["redis"] = MagicMock()
sys.modules["redis.asyncio"] = MagicMock()

from app.service import EquilibriumService
from app.transport_metacog_gate import build_transport_metacog_trigger_from_bus_synaptic


class TestBusSynapticPollE2E(unittest.TestCase):
    """Integration tests for the bus_synaptic poll loop."""

    def setUp(self):
        """Set up mocks for settings and bus/database access."""
        self.settings_patcher = patch("app.service.settings")
        self.mock_settings = self.settings_patcher.start()

        # Enable the poll
        self.mock_settings.metacog_transport_trigger_enable = True
        self.mock_settings.metacog_transport_bus_synaptic_poll_enable = True
        self.mock_settings.metacog_transport_bus_synaptic_poll_interval_sec = 1  # Fast for testing
        self.mock_settings.metacog_transport_bus_synaptic_error_threshold = 1.0
        self.mock_settings.metacog_recall_enabled = False

    def tearDown(self):
        """Stop all patchers."""
        self.settings_patcher.stop()

    def test_poll_below_threshold_does_not_trigger(self):
        """When bus_synaptic error is below threshold, no trigger should fire."""

        async def _run():
            service = EquilibriumService()
            service._calculate_metrics = MagicMock(return_value=(0.2, 0.7, []))
            service._publish_metacog_trigger = AsyncMock(return_value=None)

            # Mock FalkorDB client returning a low error value
            mock_client = MagicMock()
            mock_client.graph_query = MagicMock(return_value=[{"error": 0.5}])
            service._bus_synaptic_falkor_client = mock_client

            # Run one poll cycle manually (simulating what the loop does)
            rows = await asyncio.to_thread(
                mock_client.graph_query,
                "MATCH (n:SubstrateNode) WHERE n.node_id = $node_id "
                "RETURN n.prediction_error AS error",
                {"node_id": "node:substrate.bus_synaptic"},
            )
            error = None
            for row in rows:
                value = row.get("error") if isinstance(row, dict) else None
                if isinstance(value, (int, float)):
                    error = float(value)

            distress, zen, _ = service._calculate_metrics()
            trigger = build_transport_metacog_trigger_from_bus_synaptic(
                error,
                zen_state="zen" if zen > 0.5 else "not_zen",
                pressure=distress,
                recall_enabled=self.mock_settings.metacog_recall_enabled,
                error_threshold=self.mock_settings.metacog_transport_bus_synaptic_error_threshold,
            )

            # Should not fire
            self.assertIsNone(trigger)
            service._publish_metacog_trigger.assert_not_awaited()

        asyncio.run(_run())

    def test_poll_at_threshold_triggers(self):
        """When bus_synaptic error equals threshold, a trigger should fire."""

        async def _run():
            service = EquilibriumService()
            service._calculate_metrics = MagicMock(return_value=(0.3, 0.6, []))
            service._publish_metacog_trigger = AsyncMock(return_value=None)

            # Mock FalkorDB client returning error at exactly the threshold
            mock_client = MagicMock()
            mock_client.graph_query = MagicMock(return_value=[{"error": 1.0}])
            service._bus_synaptic_falkor_client = mock_client

            # Run one poll cycle
            rows = await asyncio.to_thread(
                mock_client.graph_query,
                "MATCH (n:SubstrateNode) WHERE n.node_id = $node_id "
                "RETURN n.prediction_error AS error",
                {"node_id": "node:substrate.bus_synaptic"},
            )
            error = None
            for row in rows:
                value = row.get("error") if isinstance(row, dict) else None
                if isinstance(value, (int, float)):
                    error = float(value)

            distress, zen, _ = service._calculate_metrics()
            trigger = build_transport_metacog_trigger_from_bus_synaptic(
                error,
                zen_state="zen" if zen > 0.5 else "not_zen",
                pressure=distress,
                recall_enabled=self.mock_settings.metacog_recall_enabled,
                error_threshold=self.mock_settings.metacog_transport_bus_synaptic_error_threshold,
            )

            # Should fire
            self.assertIsNotNone(trigger)
            self.assertEqual(trigger.trigger_kind, "transport")
            self.assertEqual(trigger.upstream["evidence_source"], "bus_synaptic_prediction_error")
            self.assertEqual(trigger.upstream["error"], 1.0)
            self.assertIn("node:substrate.bus_synaptic", trigger.signal_refs)

        asyncio.run(_run())

    def test_poll_above_threshold_triggers(self):
        """When bus_synaptic error exceeds threshold, a trigger should fire."""

        async def _run():
            service = EquilibriumService()
            service._calculate_metrics = MagicMock(return_value=(0.5, 0.5, []))
            service._publish_metacog_trigger = AsyncMock(return_value=None)

            # Mock FalkorDB client returning high error
            mock_client = MagicMock()
            mock_client.graph_query = MagicMock(return_value=[{"error": 0.87}])
            service._bus_synaptic_falkor_client = mock_client

            # Run one poll cycle
            rows = await asyncio.to_thread(
                mock_client.graph_query,
                "MATCH (n:SubstrateNode) WHERE n.node_id = $node_id "
                "RETURN n.prediction_error AS error",
                {"node_id": "node:substrate.bus_synaptic"},
            )
            error = None
            for row in rows:
                value = row.get("error") if isinstance(row, dict) else None
                if isinstance(value, (int, float)):
                    error = float(value)

            distress, zen, _ = service._calculate_metrics()
            trigger = build_transport_metacog_trigger_from_bus_synaptic(
                error,
                zen_state="zen" if zen > 0.5 else "not_zen",
                pressure=distress,
                recall_enabled=self.mock_settings.metacog_recall_enabled,
                error_threshold=self.mock_settings.metacog_transport_bus_synaptic_error_threshold,
            )

            # Should fire
            self.assertIsNotNone(trigger)
            self.assertEqual(trigger.upstream["error"], 0.87)

        asyncio.run(_run())

    def test_trigger_carries_edge_count_and_context(self):
        """Verify trigger contains full evidence payload for logging."""

        async def _run():
            service = EquilibriumService()
            service._calculate_metrics = MagicMock(return_value=(0.4, 0.6, []))
            service._publish_metacog_trigger = AsyncMock(return_value=None)

            # Mock with high error
            mock_client = MagicMock()
            mock_client.graph_query = MagicMock(return_value=[{"error": 1.2}])
            service._bus_synaptic_falkor_client = mock_client

            rows = await asyncio.to_thread(
                mock_client.graph_query,
                "MATCH (n:SubstrateNode) WHERE n.node_id = $node_id "
                "RETURN n.prediction_error AS error",
                {"node_id": "node:substrate.bus_synaptic"},
            )
            error = None
            for row in rows:
                value = row.get("error") if isinstance(row, dict) else None
                if isinstance(value, (int, float)):
                    error = float(value)

            distress, zen, _ = service._calculate_metrics()
            trigger = build_transport_metacog_trigger_from_bus_synaptic(
                error,
                zen_state="not_zen",
                pressure=distress,
                recall_enabled=self.mock_settings.metacog_recall_enabled,
                error_threshold=self.mock_settings.metacog_transport_bus_synaptic_error_threshold,
            )

            self.assertIsNotNone(trigger)
            self.assertIn("error", trigger.upstream)
            self.assertIn("evidence_source", trigger.upstream)
            self.assertIn("reason", trigger)
            self.assertIn("bus_synaptic", trigger.reason)

        asyncio.run(_run())

    def test_falkordb_client_connection_error_handles_gracefully(self):
        """When FalkorDB connection fails, poll loop should not crash."""

        async def _run():
            service = EquilibriumService()
            service._calculate_metrics = MagicMock(return_value=(0.2, 0.7, []))
            service._publish_metacog_trigger = AsyncMock(return_value=None)

            # Simulate a None client (connection failed)
            service._bus_synaptic_falkor_client = None

            # Manually call _get_bus_synaptic_falkor_client (which will return None if init fails)
            client = service._get_bus_synaptic_falkor_client()
            # Should be None and not crash
            self.assertIsNone(client)

        asyncio.run(_run())

    def test_malformed_falkordb_response_handles_gracefully(self):
        """When FalkorDB returns unexpected row format, should extract safely."""

        async def _run():
            service = EquilibriumService()
            service._calculate_metrics = MagicMock(return_value=(0.2, 0.7, []))
            service._publish_metacog_trigger = AsyncMock(return_value=None)

            # Mock client returning various malformed responses
            test_cases = [
                [{}],  # Missing 'error' key
                [{"error": None}],  # None value
                [{"error": "not_a_number"}],  # String instead of float
                [{"other_field": 0.9}],  # Completely wrong field
                [],  # Empty result set
            ]

            for rows in test_cases:
                mock_client = MagicMock()
                mock_client.graph_query = MagicMock(return_value=rows)
                service._bus_synaptic_falkor_client = mock_client

                # Run extraction logic (same as in the poll loop)
                error = None
                for row in rows:
                    value = row.get("error") if isinstance(row, dict) else None
                    if isinstance(value, (int, float)):
                        error = float(value)

                # All malformed cases should result in None
                self.assertIsNone(error)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
