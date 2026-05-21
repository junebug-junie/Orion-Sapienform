"""Correlation API fallback from cognition trace cache."""
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


def test_correlation_chain_from_cognition_trace_builds_step_chain() -> None:
    from scripts.correlation_chain_fallback import correlation_chain_from_cognition_trace

    out = correlation_chain_from_cognition_trace(
        {
            "correlation_id": "d6ac504a-c05f-42a4-a741-7e9db3758a7e",
            "verb": "chat_general",
            "steps": [
                {
                    "step_name": "recall",
                    "order": -1,
                    "status": "success",
                    "services": ["RecallService"],
                },
                {
                    "step_name": "collect_metacog_context",
                    "order": 0,
                    "status": "success",
                    "services": ["MetacogContextService"],
                },
                {
                    "step_name": "llm_chat_general",
                    "order": 2,
                    "status": "success",
                    "services": ["LLMGatewayService"],
                },
            ],
        }
    )
    assert out["correlation_id"] == "d6ac504a-c05f-42a4-a741-7e9db3758a7e"
    assert out["source"] == "cognition_trace_fallback"
    assert len(out["chain"]) == 4
    assert out["chain"][0]["signal_kind"] == "cognition_run"
    assert out["chain"][1]["organ_id"] == "recall"
    assert out["chain"][-1]["organ_id"] == "llm_gateway"
