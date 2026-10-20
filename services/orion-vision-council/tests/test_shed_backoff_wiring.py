"""While the GPU pool sheds the council route, a window skips its LLM call instead of asking the
gateway every ~6 s; the first window after the heat clears calls again; non-shed failures are
unchanged (orion/gpu_pool/shed_backoff.py)."""
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.gpu_pool.shed_backoff import BASE_SEC, ShedBackoff

from app.main import CouncilService

SHED = {"content": "", "raw": {"error": "gpu_pool_unavailable",
                               "details": {"reason": "shed:cabinet_hot", "route": "metacog_background"}}}
OK = {"content": "{\"ok\": true}", "raw": {}}
OTHER = {"content": "", "raw": {"error": "gpu_pool_unavailable", "details": {"reason": "no_serviceable_role"}}}


class FakeRpcBus:
    def __init__(self) -> None:
        self.codec = OrionCodec()
        self.calls = 0
        self.reply = SHED

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec):
        self.calls += 1
        out = BaseEnvelope(kind="llm.chat.result", source=ServiceRef(name="gw"), correlation_id=uuid.uuid4(),
                           payload=self.reply)
        return {"data": self.codec.encode(out)}


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def _svc():
    svc = CouncilService()
    svc._rpc_bus = FakeRpcBus()
    clock = Clock()
    svc._shed_backoff = ShedBackoff(name="vision-council", clock=clock, rng=lambda: 0.5, log=lambda m: None)
    return svc, clock


def _src():
    return BaseEnvelope(kind="vision.window", source=ServiceRef(name="t"), correlation_id=uuid.uuid4(), payload={})


@pytest.mark.asyncio
async def test_shed_refusal_skips_calls_until_the_window_then_resumes():
    svc, clock = _svc()
    assert await svc._call_llm_raw("p", _src()) is None
    assert svc._rpc_bus.calls == 1
    for _ in range(4):                       # windows every 6 s inside the backoff
        clock.t += 6.0
        assert await svc._call_llm_raw("p", _src()) is None
    assert svc._rpc_bus.calls == 1
    svc._rpc_bus.reply = OK                  # heat clears
    clock.t += BASE_SEC
    assert await svc._call_llm_raw("p", _src()) == "{\"ok\": true}"
    assert await svc._call_llm_raw("p", _src()) == "{\"ok\": true}"
    assert svc._rpc_bus.calls == 3


@pytest.mark.asyncio
async def test_non_shed_failure_is_not_backed_off():
    svc, clock = _svc()
    svc._rpc_bus.reply = OTHER
    for _ in range(3):
        assert await svc._call_llm_raw("p", _src()) is None
    assert svc._rpc_bus.calls == 3
