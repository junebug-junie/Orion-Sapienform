"""Tests for Hub signal inspect cache (Phase 2b); avoids repo ``scripts/`` vs Hub ``scripts`` import clash."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
_CACHE_PATH = HUB_ROOT / "scripts" / "signals_inspect_cache.py"

for candidate in (str(REPO_ROOT),):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


def _load_signals_inspect_cache():
    spec = importlib.util.spec_from_file_location("hub_signals_inspect_cache", _CACHE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hub_signals_inspect_cache"] = mod
    spec.loader.exec_module(mod)
    return mod.SignalsInspectCache


@pytest.mark.asyncio
async def test_signals_inspect_cache_trace_roundtrip() -> None:
    from datetime import datetime, timezone

    from orion.signals.models import OrganClass, OrionSignalV1

    SignalsInspectCache = _load_signals_inspect_cache()
    now = datetime.now(timezone.utc)
    tid = "a" * 32
    sig = OrionSignalV1(
        signal_id="s1",
        organ_id="biometrics",
        organ_class=OrganClass.exogenous,
        signal_kind="gpu_load",
        dimensions={"level": 0.5},
        observed_at=now,
        emitted_at=now,
        otel_trace_id=tid,
        otel_span_id="b" * 16,
    )
    cache = SignalsInspectCache(
        enabled=True,
        subscribe_pattern="orion:signals:*",
        window_sec=60.0,
        trace_enabled=True,
        trace_max_traces=10,
        trace_ttl_sec=600.0,
        trace_max_signals_per_trace=8,
    )
    async with cache._lock:
        cache._latest_by_organ[sig.organ_id] = sig
        cache._chains[tid] = [sig]
        cache._trace_touch_mono[tid] = __import__("time").monotonic()
    out = await cache.get_trace(tid)
    assert out is not None
    assert out["trace_id"] == tid
    assert len(out["chain"]) == 1
    assert out["chain"][0]["organ_id"] == "biometrics"
