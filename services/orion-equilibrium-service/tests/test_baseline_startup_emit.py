import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

sys.modules["redis"] = MagicMock()
sys.modules["redis.asyncio"] = MagicMock()

from app.service import EquilibriumService


class TestBaselineStartupEmit(unittest.TestCase):
    def setUp(self):
        self.settings_patcher = patch("app.service.settings")
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.metacog_recall_enabled = False

    def tearDown(self):
        self.settings_patcher.stop()

    def test_maybe_emit_runs_immediately_on_cold_start(self):
        async def _run():
            service = EquilibriumService()
            service._calculate_metrics = MagicMock(return_value=(0.5, 0.5, []))
            service._publish_metacog_trigger = AsyncMock(return_value=None)

            emitted = await service._maybe_emit_baseline_metacog_trigger()
            self.assertTrue(emitted)
            service._publish_metacog_trigger.assert_awaited_once()

            service._publish_metacog_trigger.reset_mock()
            emitted_again = await service._maybe_emit_baseline_metacog_trigger()
            self.assertFalse(emitted_again)
            service._publish_metacog_trigger.assert_not_awaited()

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
