import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.executor import call_step_services
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ChatResponsePayload, ServiceRef
from orion.core.contracts.recall import MemoryBundleStatsV1, MemoryBundleV1, MemoryItemV1, RecallReplyV1
from orion.schemas.cortex.schemas import ExecutionStep


class TestStoryWeaveSmoke(unittest.TestCase):
    def test_story_weave_recall_to_prompt_to_llm(self):
        mock_bus = MagicMock(spec=OrionBusAsync)
        source = ServiceRef(name="test", node="test", version="1.0")
        corr_id = str(uuid4())

        ctx = {
            "verb": "story_weave",
            "messages": [{"role": "user", "content": "We spoke about resilience after failure."}],
            "raw_user_text": "We spoke about resilience after failure.",
            "session_id": "s-test",
            "recall": {},
            "plan_recall_profile": "story_weave",
        }

        recall_step = ExecutionStep(
            step_name="gather_memory_fragments",
            verb_name="story_weave",
            services=["RecallService"],
            order=0,
            recall_profile="story_weave",
            requires_memory=True,
        )

        memory_item = MemoryItemV1(
            id="m1",
            source="vector",
            source_ref="episodes",
            uri="orion://memory/m1",
            score=0.91,
            snippet="I learned that setbacks can be reframed as data for the next attempt.",
            tags=["resilience", "learning"],
        )
        recall_reply = RecallReplyV1(
            bundle=MemoryBundleV1(
                rendered="- [vector:episodes] I learned that setbacks can be reframed as data for the next attempt.",
                items=[memory_item],
                stats=MemoryBundleStatsV1(profile="story_weave"),
            )
        )

        with patch("app.executor.RecallClient.query", new=AsyncMock(return_value=recall_reply)) as recall_query:
            recall_result = asyncio.run(
                call_step_services(
                    bus=mock_bus,
                    source=source,
                    step=recall_step,
                    ctx=ctx,
                    correlation_id=corr_id,
                )
            )

        self.assertEqual(recall_result.status, "success")
        sent_req = recall_query.await_args.kwargs["req"]
        self.assertEqual(sent_req.profile, "story_weave")

        prompt_template = Path("orion/cognition/prompts/story_weave_prompt.j2").read_text(encoding="utf-8")
        llm_step = ExecutionStep(
            step_name="llm_generate_story",
            verb_name="story_weave",
            services=["LLMGatewayService"],
            order=1,
            prompt_template=prompt_template,
        )

        llm_reply = ChatResponsePayload(content="Narrative: We turned failure into wisdom.")
        with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(return_value=llm_reply)) as llm_chat:
            llm_result = asyncio.run(
                call_step_services(
                    bus=mock_bus,
                    source=source,
                    step=llm_step,
                    ctx=ctx,
                    correlation_id=corr_id,
                )
            )

        self.assertEqual(llm_result.status, "success")
        chat_req = llm_chat.await_args.kwargs["req"]
        system_prompt = chat_req.messages[0].content
        self.assertIn("setbacks can be reframed", system_prompt)
        self.assertTrue((llm_result.result.get("LLMGatewayService") or {}).get("content"))


if __name__ == "__main__":
    unittest.main()
