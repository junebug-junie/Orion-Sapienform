import asyncio
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef

from app.main import handle_chat


@pytest.mark.asyncio
async def test_handle_chat_meta_includes_llm_uncertainty():
    fake_result = {
        "text": "hi",
        "spark_meta": {},
        "raw": {"usage": {}},
        "llm_uncertainty": {"schema_version": "v1", "available": True, "mean_logprob": -0.5},
    }
    req = BaseEnvelope(
        kind="llm.chat.request",
        source=ServiceRef(name="test", node="n", version="0"),
        correlation_id=str(uuid.uuid4()),
        payload=ChatRequestPayload(
            messages=[LLMMessage(role="user", content="ping")],
            route="quick",
        ).model_dump(mode="json"),
    )
    with patch("app.main.run_llm_chat", return_value=fake_result):
        out = await handle_chat(req)
    assert out.payload.meta is not None
    assert out.payload.meta.get("llm_uncertainty", {}).get("available") is True
