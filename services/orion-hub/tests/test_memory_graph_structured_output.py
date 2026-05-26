"""Hub memory_graph_suggest structured-output request wiring."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()


@pytest.fixture()
def hub_settings(monkeypatch):
    _ensure_hub_scripts_import_path()
    import scripts.settings as ss

    monkeypatch.setattr(
        ss.settings, "MEMORY_GRAPH_SUGGEST_STRUCTURED_OUTPUT_METHOD", "json_object_schema", raising=False
    )
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_MAX_TOKENS", 1600, raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_PRIMARY_ROUTE", "quick", raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_ESCALATION_ROUTE", "brain", raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_ENABLE_ESCALATION", False, raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_INCLUDE_GROUNDING", False, raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_BRAIN_TIMEOUT_SEC", 60.0, raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_QUICK_TIMEOUT_SEC", 60.0, raising=False)
    monkeypatch.setattr(ss.settings, "HUB_AUTO_DEFAULT_ENABLED", False, raising=False)
    monkeypatch.setattr(ss.settings, "HUB_CONTEXT_TURNS", 4, raising=False)
    monkeypatch.setattr(ss.settings, "NODE_NAME", "athena", raising=False)
    return ss.settings


def _valid_draft_json() -> str:
    p = REPO_ROOT / "tests/fixtures/memory_graph/joey_cats_draft.json"
    return p.read_text(encoding="utf-8").strip()


def _client_result(text: str) -> Any:
    from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult

    cr = CortexClientResult(
        ok=True,
        mode="brain",
        verb="memory_graph_suggest",
        status="ok",
        final_text=text,
    )
    return CortexChatResult(cortex_result=cr, final_text=text)


def _msg_payload(content: str) -> Dict[str, Any]:
    return {
        "mode": "brain",
        "verbs": ["memory_graph_suggest"],
        "messages": [{"role": "user", "content": content}],
        "use_recall": False,
        "no_write": True,
        "diagnostic": True,
    }


def test_suggest_request_includes_structured_options(hub_settings, monkeypatch) -> None:
    from scripts.memory_graph_suggest import suggest_with_escalation

    captured: Dict[str, Any] = {}

    cortex_client = MagicMock()

    async def chat(req, correlation_id=None, rpc_timeout_sec=None):
        captured["options"] = dict(req.options or {})
        return _client_result(_valid_draft_json())

    cortex_client.chat = chat

    out = asyncio.run(
        suggest_with_escalation(
            cortex_client=cortex_client,
            payload=_msg_payload("role=user\nJoey angered Juniper last week about cats"),
            session_id="sess",
            user_id=None,
            settings=hub_settings,
        )
    )
    opts = captured.get("options") or {}
    assert opts.get("structured_output_schema_name") == "SuggestDraftV1"
    assert isinstance(opts.get("structured_output_schema"), dict)
    assert opts.get("structured_output_thinking_policy") == "disabled_for_artifact"
    assert opts.get("chat_template_kwargs") == {"enable_thinking": False}
    assert int(opts.get("max_tokens") or 0) >= 4096
    assert out.get("ok") is True
    diag = out.get("structured_output_diagnostics") or {}
    assert diag.get("structured_output_method_requested") == "json_object_schema"
