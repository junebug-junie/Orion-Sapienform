"""Tests for investigation_v2 Hub request plumbing (PR1)."""

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
    from scripts.context_exec_agent_bridge import build_context_exec_request
    from orion.schemas.context_exec import context_exec_permissions_for_llm_profile
    from orion.schemas.cortex.contracts import CortexChatRequest

    return build_context_exec_request, context_exec_permissions_for_llm_profile, CortexChatRequest


for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


CORTEX_CHANGE_PROMPT = "what would happen if we changed the cortex-exec runtime?"


def test_agent_profile_permissions_read_broad_write_none() -> None:
    _, context_exec_permissions_for_llm_profile, _ = _hub_imports()
    perms = context_exec_permissions_for_llm_profile("agent")
    assert perms.read_memory is True
    assert perms.read_graph is True
    assert perms.read_recall is True
    assert perms.read_repo is True
    assert perms.read_runtime_logs is True
    assert perms.read_redis_traces is True
    assert perms.write_memory is False
    assert perms.write_graph is False
    assert perms.write_repo is False
    assert perms.mutate_runtime is False
    assert perms.network_enabled is False
    assert perms.shell_enabled is False


def test_quick_profile_permissions_remain_narrow() -> None:
    _, context_exec_permissions_for_llm_profile, _ = _hub_imports()
    perms = context_exec_permissions_for_llm_profile("quick")
    assert perms.read_repo is False
    assert perms.write_repo is False
    assert perms.mutate_runtime is False
    assert perms.write_memory is False


def test_v2_enabled_agent_lane_uses_investigation_v2_without_repo_keywords() -> None:
    build_context_exec_request, _, CortexChatRequest = _hub_imports()
    with patch("scripts.context_exec_agent_bridge.investigation_v2_enabled", return_value=True):
        req = CortexChatRequest(prompt=CORTEX_CHANGE_PROMPT, mode="agent", trace_id="corr-v2")
        body = build_context_exec_request(req=req, prompt=req.prompt, llm_profile="agent")
    assert body.mode == "investigation_v2"
    assert body.permissions.read_repo is True
    assert body.text == CORTEX_CHANGE_PROMPT


def test_v2_bypasses_infer_context_exec_mode() -> None:
    build_context_exec_request, _, CortexChatRequest = _hub_imports()
    with patch("scripts.context_exec_agent_bridge.investigation_v2_enabled", return_value=True):
        with patch(
            "scripts.context_exec_agent_bridge._infer_context_exec_mode",
            return_value="repo_impact_analysis",
        ) as infer_mock:
            req = CortexChatRequest(prompt=CORTEX_CHANGE_PROMPT, mode="agent", trace_id="corr-bypass")
            body = build_context_exec_request(req=req, prompt=req.prompt, llm_profile="agent")
    infer_mock.assert_not_called()
    assert body.mode == "investigation_v2"
    assert body.permissions.read_repo is True


def test_no_magic_phrase_repo_permission() -> None:
    build_context_exec_request, _, CortexChatRequest = _hub_imports()
    prompt = CORTEX_CHANGE_PROMPT
    assert "repo" not in prompt.lower()
    assert "impact" not in prompt.lower()
    with patch("scripts.context_exec_agent_bridge.investigation_v2_enabled", return_value=True):
        req = CortexChatRequest(prompt=prompt, mode="agent", trace_id="corr-no-magic")
        body = build_context_exec_request(req=req, prompt=prompt, llm_profile="agent")
    assert body.mode == "investigation_v2"
    assert body.permissions.read_repo is True


def test_v2_disabled_preserves_keyword_mode_inference() -> None:
    build_context_exec_request, _, CortexChatRequest = _hub_imports()
    with patch("scripts.context_exec_agent_bridge.investigation_v2_enabled", return_value=False):
        req = CortexChatRequest(
            prompt="what breaks if we replace agent-chain-service with context-exec?",
            mode="agent",
            trace_id="corr-legacy",
        )
        body = build_context_exec_request(req=req, prompt=req.prompt, llm_profile="agent")
    assert body.mode == "repo_impact_analysis"
    assert body.permissions.read_repo is True


def test_v2_enabled_quick_profile_grants_read_broad_agent_permissions() -> None:
    build_context_exec_request, _, CortexChatRequest = _hub_imports()
    with patch("scripts.context_exec_agent_bridge.investigation_v2_enabled", return_value=True):
        req = CortexChatRequest(prompt=CORTEX_CHANGE_PROMPT, mode="agent", trace_id="corr-quick")
        body = build_context_exec_request(req=req, prompt=req.prompt, llm_profile="quick")
    assert body.mode == "investigation_v2"
    assert body.llm_profile == "quick"
    assert body.permissions.read_repo is True
    assert body.permissions.read_recall is True
    assert body.permissions.read_runtime_logs is True
    assert body.permissions.write_repo is False
    assert body.permissions.mutate_runtime is False


def test_hub_format_investigation_v2_report_sections() -> None:
    _hub_imports()
    from scripts.context_exec_agent_bridge import format_investigation_v2_report

    artifact = {
        "answer_status": "partial_grounding",
        "summary": "Repo grounded summary.",
        "sections": {
            "repo": {
                "title": "Repository impact",
                "status": "hit",
                "summary": "2 affected path(s).",
            },
            "recall": {
                "title": "Recall",
                "status": "unavailable",
                "summary": "Recall dependency failure: timeout",
            },
        },
        "unavailable_sources": ["recall"],
        "limitations": ["recall unavailable: timeout"],
    }
    text = format_investigation_v2_report(artifact)
    assert "Partially grounded investigation" in text
    assert "Summary: Repo grounded summary." in text
    assert "Repository impact [hit]" in text
    assert "Unavailable sources: recall" in text
    assert "Limitations:" in text
