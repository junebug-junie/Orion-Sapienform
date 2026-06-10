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

from app.clients import ContextExecClient  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.context_exec import ContextExecRequestV1, ContextExecRunV1  # noqa: E402


class _Codec:
    @staticmethod
    def decode(data):
        return SimpleNamespace(ok=True, error=None, envelope=BaseEnvelope.model_validate(data))


class _FakeBus:
    def __init__(self, payload: dict) -> None:
        self.codec = _Codec()
        self.payload = payload
        self.last_env: BaseEnvelope | None = None

    async def rpc_request(self, channel, env, reply_channel=None, timeout_sec=None):
        self.last_env = env
        return {
            "data": BaseEnvelope(
                kind="context.exec.result.v1",
                source=ServiceRef(name="context-exec", version="0.1.0"),
                correlation_id=str(env.correlation_id),
                payload=self.payload,
            ).model_dump(mode="json")
        }


@pytest.mark.asyncio
async def test_context_exec_client_emits_native_kind():
    payload = ContextExecRunV1(
        run_id="ctxrun_test",
        status="ok",
        mode="belief_provenance",
        text="test",
        final_text="done",
    ).model_dump(mode="json")
    bus = _FakeBus(payload)
    client = ContextExecClient(bus)
    req = ContextExecRequestV1(text="probe", mode="belief_provenance")
    run = await client.run(
        source=ServiceRef(name="cortex-exec", version="0.2.0"),
        req=req,
        correlation_id="00000000-0000-4000-8000-000000000001",
        reply_to="orion:exec:result:ContextExecService:corr-1",
    )
    assert bus.last_env is not None
    assert bus.last_env.kind == "context.exec.request.v1"
    assert run.final_text == "done"
