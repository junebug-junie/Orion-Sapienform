import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Mock redis before importing service
sys.modules["redis"] = MagicMock()
sys.modules["redis.asyncio"] = MagicMock()

from app.service import EquilibriumService

class TestBaselineHygiene(unittest.TestCase):
    def setUp(self):
        self.settings_patcher = patch("app.service.settings")
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.service_name = "test"
        self.mock_settings.service_version = "1.0"
        self.mock_settings.node_name = "test-node"
        self.mock_settings.instance_id = "test-id"
        self.mock_settings.orion_bus_url = "redis://localhost"
        self.mock_settings.orion_bus_enabled = False
        self.mock_settings.heartbeat_interval_sec = 1.0
        self.mock_settings.health_channel = "health"
        self.mock_settings.metacog_enable = True
        self.mock_settings.metacog_baseline_interval_sec = 0.01
        self.mock_settings.expected_services.return_value = []

    def tearDown(self):
        self.settings_patcher.stop()

    async def run_hygiene_test(self):
        service = EquilibriumService()
        service.bus = MagicMock()
        service.bus.publish = AsyncMock()
        # Mock _publish_metacog_trigger to avoid side effects and just count calls
        service._publish_metacog_trigger = AsyncMock()
        service._calculate_metrics = MagicMock()

        # 1. First run: should emit (init state -1)
        service._calculate_metrics.return_value = (0.5, 0.5, [])

        # Run loop in background
        task = asyncio.create_task(service._metacog_baseline_loop())

        # Wait for a few iterations (interval 0.01s)
        await asyncio.sleep(0.05)

        # Should have emitted exactly ONCE (because subsequent loops see same score 0.5)
        # Verify skip logic
        self.assertEqual(service._publish_metacog_trigger.call_count, 1, "Should emit once initially")
        self.assertTrue(service._baseline_skip_count > 0, "Should have skipped subsequent identical triggers")

        # 2. Change scores significantly
        service._calculate_metrics.return_value = (0.6, 0.4, [])
        await asyncio.sleep(0.05)

        # Should have emitted one more time (total 2)
        self.assertEqual(service._publish_metacog_trigger.call_count, 2, "Should emit again when score changes")

        # 3. Force update via skip limit (max 10)
        # We need to wait longer. 10 * 0.01 = 0.1s.
        # Let's reset mocks to track clearly
        service._publish_metacog_trigger.reset_mock()
        service._calculate_metrics.return_value = (0.6, 0.4, []) # Same score
        service._baseline_skip_count = 0 # Reset count manually to match post-emit state

        # Wait for 15 intervals (0.15s)
        await asyncio.sleep(0.2)

        # Should emit at least once due to forced update (skip < 10 check)
        # Actually logic is: if ... and skip < 10: continue. Else: emit.
        # So after 10 skips, it emits.
        self.assertGreaterEqual(service._publish_metacog_trigger.call_count, 1, "Should force emit after max skips")

        service._stop.set()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    def test_hygiene(self):
        asyncio.run(self.run_hygiene_test())

if __name__ == "__main__":
    unittest.main()
