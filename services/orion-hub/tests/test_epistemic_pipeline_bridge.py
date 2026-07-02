"""Tests for hub epistemic pipeline wiring (answer contract + user voice split)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _hub_imports():
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    sys.path = [p for p in sys.path if "orion-context-exec" not in p.replace("\\", "/")]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))
    from scripts.context_exec_agent_bridge import (
        build_context_exec_chat_response,
        build_context_exec_request,
    )
    from orion.schemas.context_exec import ContextExecRunV1
    from orion.schemas.cortex.contracts import CortexChatRequest

    return build_context_exec_request, build_context_exec_chat_response, ContextExecRunV1, CortexChatRequest


for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


def test_v2_request_threads_answer_contract() -> None:
    build_context_exec_request, _, _, CortexChatRequest = _hub_imports()
    prompt = "why do you give shallow responses?"
    meta = {"answer_contract_draft": {"request_kind": "conceptual", "asks_for_explanation": True}}
    with patch("scripts.context_exec_agent_bridge.agent_repl_enabled", return_value=False):
        with patch("scripts.context_exec_agent_bridge.investigation_v2_enabled", return_value=True):
            req = CortexChatRequest(prompt=prompt, mode="agent", trace_id="corr-ac", metadata=meta)
            body = build_context_exec_request(req=req, prompt=prompt, llm_profile="agent")
    assert body.answer_contract is not None
    assert body.answer_contract.request_kind == "conceptual"
    assert body.answer_contract.requires_repo_grounding is False


def test_chat_response_uses_final_text_not_operator_report() -> None:
    _, build_context_exec_chat_response, ContextExecRunV1, _ = _hub_imports()
    run = ContextExecRunV1(
        run_id="run-1",
        status="ok",
        mode="investigation_v2",
        text="why shallow?",
        final_text="I hear you — that felt shallow. What was missing?",
        artifact={
            "answer_status": "partial_grounding",
            "summary": "Operator: recall hit(s).",
            "sections": {},
        },
        artifact_type="InvestigationReportV2",
    )
    out = build_context_exec_chat_response(run=run, correlation_id="corr-1")
    assert out["llm_response"] == "I hear you — that felt shallow. What was missing?"
    assert "Partially grounded investigation" not in out["llm_response"]
    op_report = out["raw"]["metadata"]["context_exec"].get("operator_report_text") or ""
    assert "partial_grounding" in op_report or "Operator" in op_report or op_report
