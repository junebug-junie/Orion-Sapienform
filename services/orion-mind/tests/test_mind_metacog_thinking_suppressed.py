"""Regression test for the 2026-09-08 incident: metacog's model defaults to
inline <think> reasoning unless a caller explicitly disables it via
chat_template_kwargs.enable_thinking. semantic_synthesis and stance_handoff
moved onto metacog on 2026-09-07 without this, so unsuppressed reasoning ate
the whole max_tokens budget before any JSON appeared -- json_parse_failed on
every metacog-routed call. Confirmed live against the gateway's own THINK_HOP
trace logs and the served chat template
(`{% if enable_thinking is defined and enable_thinking is false %}`).

The fix lives in MindLLMClient.request_json itself (llm_client.py), not at
each call site: `thinking=False` (the default) now actually sets
chat_template_kwargs.enable_thinking=False, replacing the old
`options["thinking"] = True` line that nothing in orion-llm-gateway ever read
(confirmed: no `.get("thinking")` anywhere in that service). That fixes every
current and future caller that leaves `thinking` at its default, not just the
two call sites that broke this time.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(SERVICE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"


def _mind_prep() -> None:
    spec = importlib.util.spec_from_file_location("_mind_guard_lazy", _guard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_orion_mind_app()


def test_request_json_disables_thinking_by_default(monkeypatch) -> None:
    """thinking=False (the default, used by semantic_synthesis and
    stance_handoff) must set chat_template_kwargs.enable_thinking=False --
    that's the only option the gateway actually reads to suppress reasoning."""
    _mind_prep()
    from app.llm_client import MindLLMClient

    seen: dict[str, Any] = {}

    def fake_bus_chat(self, *, system_prompt, user_prompt, route, options, context, timeout_sec):
        seen["options"] = options
        return '{"ok": true}', {}, "metacog", {}

    monkeypatch.setattr(MindLLMClient, "_bus_chat", fake_bus_chat)
    monkeypatch.setattr("app.llm_client.settings.MIND_LLM_USE_BUS", True)
    monkeypatch.setattr("app.llm_client.settings.ORION_BUS_ENABLED", True)

    client = MindLLMClient()
    client.request_json(
        system_prompt="sys",
        user_prompt="user",
        route="metacog",
        max_tokens=2048,
    )
    assert seen["options"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_request_json_leaves_thinking_alone_when_true(monkeypatch) -> None:
    """thinking=True (appraisal's setting) must NOT force enable_thinking --
    appraisal budgets enough tokens to survive the model's own default and
    this preserves its exact current live behavior."""
    _mind_prep()
    from app.llm_client import MindLLMClient

    seen: dict[str, Any] = {}

    def fake_bus_chat(self, *, system_prompt, user_prompt, route, options, context, timeout_sec):
        seen["options"] = options
        return '{"ok": true}', {}, "metacog", {}

    monkeypatch.setattr(MindLLMClient, "_bus_chat", fake_bus_chat)
    monkeypatch.setattr("app.llm_client.settings.MIND_LLM_USE_BUS", True)
    monkeypatch.setattr("app.llm_client.settings.ORION_BUS_ENABLED", True)

    client = MindLLMClient()
    client.request_json(
        system_prompt="sys",
        user_prompt="user",
        route="metacog",
        max_tokens=3072,
        thinking=True,
    )
    assert "chat_template_kwargs" not in seen["options"]


def test_request_json_respects_caller_supplied_chat_template_kwargs(monkeypatch) -> None:
    """A caller-supplied chat_template_kwargs (via extra_options) is extended,
    not clobbered, when thinking defaults to False."""
    _mind_prep()
    from app.llm_client import MindLLMClient

    seen: dict[str, Any] = {}

    def fake_bus_chat(self, *, system_prompt, user_prompt, route, options, context, timeout_sec):
        seen["options"] = options
        return '{"ok": true}', {}, "metacog", {}

    monkeypatch.setattr(MindLLMClient, "_bus_chat", fake_bus_chat)
    monkeypatch.setattr("app.llm_client.settings.MIND_LLM_USE_BUS", True)
    monkeypatch.setattr("app.llm_client.settings.ORION_BUS_ENABLED", True)

    client = MindLLMClient()
    client.request_json(
        system_prompt="sys",
        user_prompt="user",
        route="metacog",
        max_tokens=2048,
        extra_options={"chat_template_kwargs": {"some_other_flag": True}},
    )
    assert seen["options"]["chat_template_kwargs"] == {
        "some_other_flag": True,
        "enable_thinking": False,
    }
