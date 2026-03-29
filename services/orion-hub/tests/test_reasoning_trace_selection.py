from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HUB_ROOT) not in sys.path:
    sys.path.insert(0, str(HUB_ROOT))

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

from scripts import chat_history, websocket_handler  # type: ignore


def test_reasoning_content_only_synthesizes_reasoning_trace() -> None:
    trace, source = chat_history.select_reasoning_trace_for_history(
        correlation_id="corr-1",
        reasoning_trace=None,
        metacog_traces=None,
        reasoning_content="model-side reasoning",
        session_id="sid-1",
        message_id="mid-1",
        model="m-1",
    )
    assert source == "reasoning_content"
    assert isinstance(trace, dict)
    assert trace.get("content") == "model-side reasoning"
    assert trace.get("trace_role") == "reasoning"


def test_metacog_only_selects_first_reasoning_trace() -> None:
    trace, source = chat_history.select_reasoning_trace_for_history(
        correlation_id="corr-2",
        reasoning_trace=None,
        metacog_traces=[
            {"trace_role": "stance", "content": "stance text"},
            {"trace_role": "reasoning", "trace_stage": "post_answer", "content": "reasoning from metacog"},
        ],
        reasoning_content=None,
        session_id="sid-2",
    )
    assert source == "metacog_traces[1]"
    assert isinstance(trace, dict)
    assert trace.get("content") == "reasoning from metacog"


def test_no_reasoning_inputs_returns_none() -> None:
    trace, source = chat_history.select_reasoning_trace_for_history(
        correlation_id="corr-3",
        reasoning_trace=None,
        metacog_traces=[],
        reasoning_content=None,
        session_id="sid-3",
    )
    assert trace is None
    assert source == "none"


def test_route_debug_logs_effective_verb_for_null_emitted_verb(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_info(msg: str, payload: str) -> None:
        captured["msg"] = msg
        captured["payload"] = payload

    monkeypatch.setattr(websocket_handler.logger, "info", _fake_info)
    websocket_handler._log_hub_route_decision(
        corr_id="corr-verb",
        session_id="sid-verb",
        route_debug={"mode": "brain", "verb": None, "options": {}},
        user_prompt="hello",
    )

    summary = json.loads(captured["payload"])
    assert summary["emitted_verb"] is None
    assert summary["effective_verb"] == "chat_general"
