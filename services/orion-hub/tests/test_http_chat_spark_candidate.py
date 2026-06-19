from __future__ import annotations

import asyncio
import os

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")

from scripts import api_routes
from scripts.spark_candidate import build_spark_introspect_candidate_envelope


def test_build_spark_introspect_candidate_envelope_carries_turn_effect() -> None:
    corr_id = "00000000-0000-0000-0000-00000000c001"
    env = build_spark_introspect_candidate_envelope(
        trace_id=corr_id,
        prompt="hello",
        response="hi",
        spark_meta={
            "turn_effect": {"turn": {"novelty": 0.42}},
            "turn_effect_status": "present",
        },
        source="hub_http",
        correlation_id=corr_id,
    )
    payload = env.payload if isinstance(env.payload, dict) else {}
    sm = payload.get("spark_meta") or {}
    assert sm["turn_effect"]["turn"]["novelty"] == 0.42
    assert env.kind == "spark.candidate"


def test_api_chat_publishes_spark_candidate(monkeypatch) -> None:
    import scripts.main as hub_main

    class _Bus:
        enabled = True

        def __init__(self) -> None:
            self.published: list[tuple[str, object]] = []

        async def publish(self, channel, env):
            self.published.append((channel, env))

    bus = _Bus()

    async def _fake_ensure_session(_session_id, _bus):
        return "sid-normal"

    async def _fake_handle_chat_request(*_args, **_kwargs):
        return {
            "session_id": "sid-normal",
            "mode": "brain",
            "use_recall": True,
            "text": "normal response",
            "workflow": None,
            "workflow_metadata_only": False,
            "correlation_id": "00000000-0000-0000-0000-00000000a001",
            "routing_debug": {},
            "memory_digest": None,
            "metacog_traces": [],
            "raw": {
                "metadata": {
                    "turn_effect": {"turn": {"novelty": 0.17}},
                    "turn_effect_status": "present",
                }
            },
        }

    calls = {"history": 0, "turn": 0}

    async def _fake_publish_chat_history(*_args, **_kwargs):
        calls["history"] += 1

    async def _fake_publish_chat_turn(*_args, **_kwargs):
        calls["turn"] += 1

    monkeypatch.setattr(hub_main, "bus", bus)
    monkeypatch.setattr(hub_main, "cortex_client", object())
    monkeypatch.setattr(api_routes, "ensure_session", _fake_ensure_session)
    monkeypatch.setattr(api_routes, "handle_chat_request", _fake_handle_chat_request)
    monkeypatch.setattr(api_routes, "publish_chat_history", _fake_publish_chat_history)
    monkeypatch.setattr(api_routes, "publish_chat_turn", _fake_publish_chat_turn)

    payload = {"messages": [{"role": "user", "content": "hello"}]}
    result = asyncio.run(api_routes.api_chat(payload, None, None))
    assert result["text"] == "normal response"
    assert calls == {"history": 1, "turn": 1}

    spark_publishes = [
        (channel, env)
        for channel, env in bus.published
        if isinstance(env, object) and getattr(env, "kind", None) == "spark.candidate"
    ]
    assert len(spark_publishes) == 1
    _channel, env = spark_publishes[0]
    payload = env.payload if isinstance(env.payload, dict) else {}
    sm = payload.get("spark_meta") or {}
    assert sm.get("turn_effect", {}).get("turn", {}).get("novelty") == 0.17
