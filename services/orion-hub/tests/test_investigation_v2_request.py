"""Tests for investigation_v2 Hub request plumbing (PR1)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(HUB_ROOT), str(REPO_ROOT)):
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

from orion.schemas.context_exec import context_exec_permissions_for_llm_profile
from orion.schemas.cortex.contracts import CortexChatRequest
from scripts.context_exec_agent_bridge import build_context_exec_request


CORTEX_CHANGE_PROMPT = "what would happen if we changed the cortex-exec runtime?"


def test_agent_profile_permissions_read_broad_write_none() -> None:
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
    perms = context_exec_permissions_for_llm_profile("quick")
    assert perms.read_repo is False
    assert perms.write_repo is False
    assert perms.mutate_runtime is False
    assert perms.write_memory is False


@patch("scripts.context_exec_agent_bridge.investigation_v2_enabled", return_value=True)
def test_v2_enabled_agent_lane_uses_investigation_v2_without_repo_keywords(_mock_v2: object) -> None:
    req = CortexChatRequest(prompt=CORTEX_CHANGE_PROMPT, mode="agent", trace_id="corr-v2")
    body = build_context_exec_request(req=req, prompt=req.prompt, llm_profile="agent")
    assert body.mode == "investigation_v2"
    assert body.permissions.read_repo is True
    assert body.text == CORTEX_CHANGE_PROMPT


@patch("scripts.context_exec_agent_bridge.investigation_v2_enabled", return_value=False)
def test_v2_disabled_preserves_keyword_mode_inference(_mock_v2: object) -> None:
    req = CortexChatRequest(prompt=CORTEX_CHANGE_PROMPT, mode="agent", trace_id="corr-legacy")
    body = build_context_exec_request(req=req, prompt=req.prompt, llm_profile="agent")
    assert body.mode == "general_investigation"
    assert body.permissions.read_repo is False


@patch("scripts.context_exec_agent_bridge.investigation_v2_enabled", return_value=True)
def test_v2_enabled_quick_profile_does_not_grant_mutation_permissions(_mock_v2: object) -> None:
    req = CortexChatRequest(prompt=CORTEX_CHANGE_PROMPT, mode="agent", trace_id="corr-quick")
    body = build_context_exec_request(req=req, prompt=req.prompt, llm_profile="quick")
    assert body.mode == "investigation_v2"
    assert body.permissions.read_repo is False
    assert body.permissions.write_repo is False
    assert body.permissions.mutate_runtime is False
