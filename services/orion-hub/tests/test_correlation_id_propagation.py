"""Correlation ID propagation tests (Runtime Trace Nexus §5.8 gate).

Manual staging verification checklist (one live ``chat_general`` turn):

1. Hub chat response ``correlation_id`` / ``trace_linkage.correlation_id``
2. ``CognitionTracePayload`` envelope on bus (``orion:cognition:trace`` log)
3. ``orion:signals:cortex_exec`` run signal ``source_event_id``
4. ``GET /api/cognition/trace/{id}``
5. ``GET /api/signals/correlation/{id}``

All five must resolve the same canonical correlation id.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]

for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
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


def test_chat_response_metadata_includes_canonical_correlation_id() -> None:
    """Hub must expose the same corr id used for cortex + trace APIs (spec §5.8)."""
    from scripts.api_routes import _chat_turn_trace_linkage

    out = _chat_turn_trace_linkage(
        hub_corr_id="hub-corr-abc",
        cortex_corr_id="hub-corr-abc",
        root_correlation_id=None,
    )
    assert out["correlation_id"] == "hub-corr-abc"
    assert out["root_correlation_id"] is None

    out2 = _chat_turn_trace_linkage(
        hub_corr_id="hub-corr-abc",
        cortex_corr_id="cortex-different",
        root_correlation_id="hub-corr-abc",
    )
    assert out2["correlation_id"] == "hub-corr-abc"
    assert out2["root_correlation_id"] == "hub-corr-abc"
