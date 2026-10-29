"""Per-call token ledger: reading counts from real llama.cpp replies, the sink, the publisher.

Fixtures under fixtures/call_usage/ are real replies captured 2026-10-29 from circe's chat lane
(Qwen3.6-35B llama.cpp b10398) through the gateway passthroughs, max_tokens=8, prompt "Say hi".
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path

from orion.llm.call_usage import UsageSink, publish_batch, run_usage_publisher, token_counts
from orion.schemas.llm_call_usage import LLM_CALL_USAGE_CHANNEL, LLM_CALL_USAGE_KIND, LlmCallUsageV1

FIX = Path(__file__).parent / "fixtures" / "call_usage"


def _read(name: str) -> bytes:
    return (FIX / name).read_bytes()


def test_openai_nonstream_reads_usage_and_decode_speed():
    import json

    c = token_counts(json.loads(_read("openai_nonstream.json")))
    assert (c.prompt_tokens, c.cached_prompt_tokens, c.completion_tokens) == (14, 0, 8)
    assert c.token_source == "usage"
    assert 70 < c.decode_tps < 71


def test_openai_stream_without_usage_chunk_falls_back_to_llama_timings():
    """Live: an OpenAI stream has no usage chunk, only timings in the last chunk (prompt_n=4 new +
    cache_n=10 reused = 14 prompt tokens, the same 14 the non-stream call reported)."""
    raw = _read("openai_stream.sse")
    c = token_counts(raw[:4096], raw[-4096:])
    assert (c.prompt_tokens, c.cached_prompt_tokens, c.completion_tokens) == (14, 10, 8)
    assert c.token_source == "timings"
    assert c.decode_tps and c.decode_tps > 0


def test_anthropic_stream_needs_head_and_tail():
    """Prompt counts ride message_start (head); the completion count rides message_delta (tail).
    Anthropic input_tokens EXCLUDES cache reads, so prompt = 4 + 10."""
    raw = _read("anthropic_stream.sse")
    head, tail = raw[:400], raw[-400:]
    both = token_counts(head, tail)
    assert (both.prompt_tokens, both.cached_prompt_tokens, both.completion_tokens) == (14, 10, 8)
    assert both.token_source == "usage"
    # Tail only: no prompt count. Head only: completion unknown, NOT message_start's placeholder 0.
    assert token_counts(tail).prompt_tokens is None
    head_only = token_counts(head)
    assert head_only.prompt_tokens == 14 and head_only.completion_tokens is None


def test_anthropic_nonstream():
    import json

    c = token_counts(json.loads(_read("anthropic_nonstream.json")))
    assert (c.prompt_tokens, c.cached_prompt_tokens, c.completion_tokens, c.token_source) == (14, 10, 8, "usage")


def test_nothing_reported_is_none_not_zero():
    c = token_counts({"choices": []}, b"garbage", b"data: {not json")
    assert (c.prompt_tokens, c.completion_tokens, c.token_source) == (None, None, "none")
    assert token_counts().token_source == "none"


def test_negative_and_bool_counts_are_rejected():
    c = token_counts({"usage": {"prompt_tokens": -3, "completion_tokens": True}})
    assert c.prompt_tokens is None and c.completion_tokens is None


def _event(i: int = 0, corr: str | None = "3f2a1c7e-0000-4000-8000-000000000001") -> LlmCallUsageV1:
    return LlmCallUsageV1(call_id=f"c{i}", recorded_at=datetime.now(timezone.utc), recorder="gateway",
                          path="bus_rpc", correlation_id=corr, outcome="served", prompt_tokens=1,
                          completion_tokens=2, token_source="usage")


def test_sink_is_bounded_drops_oldest_and_is_thread_safe():
    sink = UsageSink(maxlen=100)
    threads = [threading.Thread(target=lambda k=k: [sink.offer(_event(k * 1000 + j)) for j in range(50)])
               for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(sink) == 100 and sink.dropped == 100
    assert len(sink.drain(limit=60)) == 60 and len(sink) == 40


class _Bus:
    def __init__(self, fail: bool = False):
        self.sent: list = []
        self.fail = fail

    async def publish(self, channel, envelope):
        if self.fail:
            raise RuntimeError("bus down")
        self.sent.append((channel, envelope))


def test_publish_batch_envelope_carries_kind_and_uuid_correlation():
    from orion.core.bus.bus_schemas import ServiceRef

    bus = _Bus()
    src = ServiceRef(name="t", version="0", node="n")
    asyncio.run(publish_batch(bus, [_event(1), _event(2, corr="not-a-uuid")], source=src,
                              channel=LLM_CALL_USAGE_CHANNEL))
    (ch1, env1), (_, env2) = bus.sent
    assert ch1 == LLM_CALL_USAGE_CHANNEL and env1.kind == LLM_CALL_USAGE_KIND
    assert str(env1.correlation_id) == "3f2a1c7e-0000-4000-8000-000000000001"
    assert env2.payload["correlation_id"] == "not-a-uuid"  # kept in the payload, envelope gets its own
    assert LlmCallUsageV1.model_validate(env1.payload).completion_tokens == 2


def test_publisher_drains_then_exits_on_stop_and_survives_bus_failure():
    from orion.core.bus.bus_schemas import ServiceRef

    src = ServiceRef(name="t", version="0", node="n")

    async def run(bus):
        sink = UsageSink()
        for i in range(3):
            sink.offer(_event(i))
        stop = asyncio.Event()
        stop.set()
        await asyncio.wait_for(run_usage_publisher(bus, sink, source=src, stop=stop, interval_sec=0.2), 5)
        return sink

    ok = _Bus()
    assert len(asyncio.run(run(ok)).drain()) == 0 and len(ok.sent) == 3
    bad = _Bus(fail=True)
    asyncio.run(run(bad))  # logs and returns; never raises


def test_schema_is_registered_and_channel_catalogued():
    import yaml

    from orion.schemas.registry import resolve

    assert resolve("LlmCallUsageV1") is LlmCallUsageV1
    root = Path(__file__).resolve().parents[3]
    catalog = yaml.safe_load((root / "orion/bus/channels.yaml").read_text())
    ch = next(c for c in catalog["channels"] if c["name"] == LLM_CALL_USAGE_CHANNEL)
    assert ch["schema_id"] == "LlmCallUsageV1" and ch["message_kind"] == LLM_CALL_USAGE_KIND
    assert set(ch["producer_services"]) == {"orion-llm-gateway", "orion-harness-governor"}
    assert ch["consumer_services"] == ["orion-sql-writer"]
