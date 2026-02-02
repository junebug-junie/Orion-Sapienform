import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.executor import call_step_services
from orion.schemas.cortex.schemas import ExecutionStep
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2
from uuid import uuid4

class TestFirebreak(unittest.TestCase):
    def test_firebreak_skips_baseline_fallback(self):
        # Mock bus
        mock_bus = MagicMock(spec=OrionBusAsync)
        mock_bus.publish = AsyncMock()

        source = ServiceRef(name="test", node="test", version="1.0")
        corr_id = str(uuid4())

        # Setup context
        ctx = {
            "trigger": {"trigger_kind": "baseline"},
            "final_entry": {
                "id": "123",
                "state_snapshot": {
                    "telemetry": {
                        "metacog_draft_mode": "fallback"
                    }
                }
            }
        }

        step = ExecutionStep(
            step_name="publish",
            verb_name="metacog_log",
            services=["MetacogPublishService"],
            order=1
        )

        # Run
        result = asyncio.run(call_step_services(
            bus=mock_bus,
            source=source,
            step=step,
            ctx=ctx,
            correlation_id=corr_id
        ))

        # Verify
        self.assertEqual(result.status, "success")
        self.assertTrue(result.result["MetacogPublishService"]["skipped"])
        self.assertEqual(result.result["MetacogPublishService"]["reason"], "firebreak_baseline_fallback")
        mock_bus.publish.assert_not_called()

    def test_firebreak_allows_baseline_llm(self):
        # Mock bus
        mock_bus = MagicMock(spec=OrionBusAsync)
        mock_bus.publish = AsyncMock()

        source = ServiceRef(name="test", node="test", version="1.0")
        corr_id = str(uuid4())

        # Setup context with minimal valid entry
        # Using a valid CollapseMirrorEntryV2 dict structure
        valid_entry = CollapseMirrorEntryV2(
            event_id="evt-1",
            id="evt-1",
            trigger="baseline",
            observer="orion",
            observer_state=["zen"],
            type="flow",
            emergent_entity="Test",
            summary="Test summary",
            mantra="Test mantra",
            field_resonance="Test resonance",
            resonance_signature="Test sig",
            source_service="metacog"
        ).model_dump(mode="json")

        valid_entry["state_snapshot"] = {
            "telemetry": {
                "metacog_draft_mode": "llm"
            }
        }

        ctx = {
            "trigger": {"trigger_kind": "baseline"},
            "final_entry": valid_entry
        }

        step = ExecutionStep(
            step_name="publish",
            verb_name="metacog_log",
            services=["MetacogPublishService"],
            order=1
        )

        # Run
        result = asyncio.run(call_step_services(
            bus=mock_bus,
            source=source,
            step=step,
            ctx=ctx,
            correlation_id=corr_id
        ))

        # Verify
        self.assertEqual(result.status, "success")
        if "skipped" in result.result["MetacogPublishService"]:
             self.fail(f"Should not skip: {result.result['MetacogPublishService']}")

        mock_bus.publish.assert_called()

if __name__ == "__main__":
    unittest.main()
