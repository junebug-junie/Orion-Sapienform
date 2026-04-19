import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.executor import call_step_services  # noqa: E402
from app.settings import settings  # noqa: E402
from orion.core.bus.async_service import OrionBusAsync  # noqa: E402
from orion.core.bus.bus_schemas import ChatResponsePayload, ServiceRef  # noqa: E402
from orion.schemas.cortex.schemas import ExecutionStep  # noqa: E402


class TestMetacogRouteProfile(unittest.TestCase):
    def test_metacog_route_injects_atlas_profile_when_missing(self) -> None:
        mock_bus = MagicMock(spec=OrionBusAsync)
        source = ServiceRef(name="test", node="test", version="1.0")
        corr_id = str(uuid4())
        step = ExecutionStep(
            step_name="generate",
            verb_name="internal_metacog_probe",
            services=["LLMGatewayService"],
            order=1,
            prompt_template="You are metacog.",
        )
        ctx = {
            "mode": "metacog",
            "messages": [{"role": "user", "content": "hello"}],
        }
        old_profile = settings.atlas_metacog_profile_name
        settings.atlas_metacog_profile_name = "llama3-8b-instruct-q4km-atlas-metacog"
        try:
            with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(return_value=ChatResponsePayload(content="ok"))) as llm_chat:
                result = asyncio.run(
                    call_step_services(
                        bus=mock_bus,
                        source=source,
                        step=step,
                        ctx=ctx,
                        correlation_id=corr_id,
                    )
                )
            self.assertEqual(result.status, "success")
            req = llm_chat.await_args.kwargs["req"]
            self.assertEqual(req.route, "metacog")
            self.assertEqual(req.profile, "llama3-8b-instruct-q4km-atlas-metacog")
        finally:
            settings.atlas_metacog_profile_name = old_profile

    def test_metacog_route_keeps_explicit_profile_name(self) -> None:
        mock_bus = MagicMock(spec=OrionBusAsync)
        source = ServiceRef(name="test", node="test", version="1.0")
        corr_id = str(uuid4())
        step = ExecutionStep(
            step_name="generate",
            verb_name="internal_metacog_probe",
            services=["LLMGatewayService"],
            order=1,
            prompt_template="You are metacog.",
        )
        ctx = {
            "mode": "metacog",
            "profile_name": "explicit-profile",
            "messages": [{"role": "user", "content": "hello"}],
        }
        with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(return_value=ChatResponsePayload(content="ok"))) as llm_chat:
            result = asyncio.run(
                call_step_services(
                    bus=mock_bus,
                    source=source,
                    step=step,
                    ctx=ctx,
                    correlation_id=corr_id,
                )
            )
        self.assertEqual(result.status, "success")
        req = llm_chat.await_args.kwargs["req"]
        self.assertEqual(req.route, "metacog")
        self.assertEqual(req.profile, "explicit-profile")


if __name__ == "__main__":
    unittest.main()
