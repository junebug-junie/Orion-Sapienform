"""memory.episode_distill's gateway call reports the provider finish_reason.

The distill graph needs it to tell an answer cut at max_tokens (``length``) from a model that wrote
broken JSON. Live 2026-10-31 the two were indistinguishable: 12 calls stopped at 4096 tokens and
every attempt was logged as ``distill_json_invalid``.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.runner import DurableRunner
from app.settings import Settings
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.memory_episode import EpisodeDistillBriefV1

HOLD = GpuLeaseRefV1(lease_id="hold-d", generation=1, role="agent", holder="durable-runs:memdistill")
BRIEF = EpisodeDistillBriefV1(episode_id="ep-1", turn_ids=["c-1"], started_at="2026-10-31T12:00:00Z",
                              ended_at="2026-10-31T13:00:00Z", close_reason="v2:phase_next_day",
                              llm_route="memory_distill", timeout_sec=30.0, max_tokens=8192)


def _runner(raw: dict) -> DurableRunner:
    codec = OrionCodec()
    reply = BaseEnvelope(kind="llm.chat.result", source=ServiceRef(name="orion-llm-gateway"),
                         payload={"content": '{"memories": [', "raw": raw})
    bus = SimpleNamespace(codec=codec, rpc_request=AsyncMock(return_value={"data": codec.encode(reply)}))
    settings = Settings(_env_file=None, DURABLE_RUNS_GRAPH_HOST="", POSTGRES_URI="postgresql://unused",
                        ORION_BUS_ENABLED=False)
    return DurableRunner(settings, bus=bus, checkpointer=None)


@pytest.mark.parametrize("raw, expected", [
    ({"choices": [{"finish_reason": "length"}], "usage": {"completion_tokens": 8192}}, "length"),
    ({"choices": [{"finish_reason": "stop"}]}, "stop"),
    ({"usage": {}}, None),
])
def test_finish_reason_is_returned(raw, expected):
    answer = asyncio.run(_runner(raw)._call_memory_distill_llm(
        "prompt", brief=BRIEF, run_id="memdistill-ep-1", correlation_id="ep-1", gpu_lease=HOLD))
    assert answer["finish_reason"] == expected


def test_default_budget_fits_the_longest_observed_answer():
    """4,037 tokens was a real Flash-Next answer (thinking off); 4096 left no headroom."""
    assert Settings(_env_file=None, POSTGRES_URI="postgresql://unused").memory_episode_distill_max_tokens >= 8192
