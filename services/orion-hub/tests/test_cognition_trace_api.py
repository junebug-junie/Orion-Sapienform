"""Tests for Hub CognitionTraceCache and cognition trace API (Runtime Trace Nexus A4)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]

for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

_cache_path = HUB_ROOT / "scripts" / "cognition_trace_cache.py"
_cache_spec = importlib.util.spec_from_file_location("hub_cognition_trace_cache", _cache_path)
assert _cache_spec and _cache_spec.loader
_cache_mod = importlib.util.module_from_spec(_cache_spec)
_cache_spec.loader.exec_module(_cache_mod)
CognitionTraceCache = _cache_mod.CognitionTraceCache

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


def test_cognition_trace_cache_api_redacted_default() -> None:
    import asyncio

    from orion.schemas.cortex.types import StepExecutionResult
    from orion.schemas.telemetry.cognition_trace import CognitionTracePayload

    cache = CognitionTraceCache(
        enabled=True,
        subscribe_channel="orion:cognition:trace",
        max_entries=10,
        ttl_sec=300.0,
        api_debug=False,
    )
    trace = CognitionTracePayload(
        correlation_id="corr-1",
        mode="brain",
        verb="chat_general",
        final_text="SECRET",
        steps=[
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="llm_chat_general",
                order=0,
                result={"LLMGatewayService": {}},
                latency_ms=1,
            )
        ],
        recall_used=True,
        metadata={
            "repair_pressure_contract": {"mode": "concrete_bias", "rules": ["be more specific"]},
            "substrate_effect_attached": True,
            "repair_pressure_mode": "concrete_bias",
            "repair_pressure_level_label": "MEDIUM",
        },
    )

    async def _run():
        await cache.put("corr-1", trace, otel_trace_id=None)
        return await cache.get_redacted("corr-1")

    body = asyncio.run(_run())
    assert body is not None
    assert body["final_text_present"] is True
    assert "SECRET" not in str(body)
    assert body["metadata"]["repair_pressure_contract"]["mode"] == "concrete_bias"
    assert body["metadata"]["substrate_effect_attached"] is True
    assert body["steps"][0]["error_present"] is False
