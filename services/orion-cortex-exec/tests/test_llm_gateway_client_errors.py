from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)
REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.clients import LLMGatewayClient  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ServiceRef  # noqa: E402


class _Codec:
    @staticmethod
    def decode(data):
        return SimpleNamespace(ok=True, error=None, envelope=BaseEnvelope.model_validate(data))


class _FakeBus:
    def __init__(self, *, kind: str, payload: dict) -> None:
        self.codec = _Codec()
        self.kind = kind
        self.payload = payload

    async def rpc_request(self, channel, env, reply_channel=None, timeout_sec=None):
        return {
            "data": BaseEnvelope(
                kind=self.kind,
                source=ServiceRef(name="orion-llm-gateway", version="0.1.0"),
                correlation_id=str(env.correlation_id),
                payload=self.payload,
            ).model_dump(mode="json")
        }


def _req() -> ChatRequestPayload:
    return ChatRequestPayload(profile="default", messages=[], raw_user_text="hi")


@pytest.mark.asyncio
async def test_system_error_reply_raises_instead_of_a_hollow_success() -> None:
    """Regression test for 2026-08-29: ChatResponsePayload (an alias for
    bus_schemas.ChatResultPayload) has every field Optional with
    extra="ignore", so model_validate() on a system.error payload used to
    succeed silently into a content=None "successful" reply instead of
    raising -- the caller had no way to tell generation had failed.
    orion-llm-gateway/app/main.py:216 publishes exactly this shape for an
    unsupported/failed chat request."""
    bus = _FakeBus(kind="system.error", payload={"error": "unsupported_kind:legacy.message"})
    client = LLMGatewayClient(bus)

    with pytest.raises(RuntimeError, match="unsupported_kind"):
        await client.chat(
            source=ServiceRef(name="cortex-exec", version="0.1.0"),
            req=_req(),
            correlation_id="00000000-0000-4000-8000-000000000002",
            reply_to="orion:llm:gateway:result:corr-2",
        )


@pytest.mark.asyncio
async def test_successful_reply_still_returns_content() -> None:
    """The kind check must be narrow: an ordinary successful reply is
    unaffected."""
    bus = _FakeBus(kind="llm.chat.result", payload={"content": "hello there"})
    client = LLMGatewayClient(bus)

    resp = await client.chat(
        source=ServiceRef(name="cortex-exec", version="0.1.0"),
        req=_req(),
        correlation_id="00000000-0000-4000-8000-000000000003",
        reply_to="orion:llm:gateway:result:corr-3",
    )

    assert resp.content == "hello there"
