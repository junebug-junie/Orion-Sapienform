"""Read-only Cursor tool policy + invoker (no live Cursor API)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from cursor_sdk import AgentOptions, LocalAgentOptions

from app.cursor_invoker import (
    build_sealed_prompt,
    parse_peer_brief_body,
    run_cursor_job,
)
from app.policy import (
    READ_ONLY_CURSOR_TOOLS,
    assert_read_only_agent_options,
    assert_read_only_tools,
)
from orion.schemas.curiosity_peer import HelpRequestV1


def _help(**overrides) -> HelpRequestV1:
    base = dict(
        help_id="help-1",
        run_id="run-1",
        prior_id="prior-1",
        mode="world_curiosity",
        question="Where is the contested budget gate?",
        tried_summary="Read scarcity docs.",
        success_criteria="A file path and function name.",
    )
    base.update(overrides)
    return HelpRequestV1(**base)


def test_policy_rejects_write_tools() -> None:
    with pytest.raises(ValueError) as exc:
        assert_read_only_tools(["read", "edit"])
    assert "edit" in str(exc.value)


def test_policy_rejects_shell_and_delete() -> None:
    for bad in ("shell", "delete", "applyDiff", "task"):
        with pytest.raises(ValueError) as exc:
            assert_read_only_tools(["read", bad])
        assert bad in str(exc.value)


def test_policy_accepts_allowlist() -> None:
    assert_read_only_tools(list(READ_ONLY_CURSOR_TOOLS))


def test_assert_read_only_agent_options_rejects_extra_tools() -> None:
    opts = AgentOptions(
        model="composer-2.5",
        tools=["read", "shell"],
        local=LocalAgentOptions(cwd="/tmp", setting_sources=[]),
    )
    with pytest.raises(ValueError) as exc:
        assert_read_only_agent_options(opts)
    assert "shell" in str(exc.value)


def test_assert_read_only_agent_options_requires_allowlist() -> None:
    opts = AgentOptions(
        model="composer-2.5",
        tools=None,
        local=LocalAgentOptions(cwd="/tmp", setting_sources=[]),
    )
    with pytest.raises(ValueError):
        assert_read_only_agent_options(opts)


def test_sealed_prompt_includes_mode_and_peerbrief_shape() -> None:
    prompt = build_sealed_prompt(_help(mode="self_inquiry"))
    assert "self_inquiry" in prompt
    assert "SelfDefinition" in prompt
    assert "evidence_pointers" in prompt
    assert "summary" in prompt
    assert "do not draft" in prompt.lower() or "never draft" in prompt.lower()
    assert "Where is the contested budget gate?" in prompt


def test_parse_empty_body_is_empty_status() -> None:
    brief = parse_peer_brief_body("", help=_help())
    assert brief.status == "empty"
    assert brief.peer == "cursor_auto"


def test_parse_json_body_ok() -> None:
    body = (
        '{"summary":"Gate lives in scarcity.py.",'
        '"evidence_pointers":["orion/dev_economics/scarcity.py"],'
        '"open_questions":[],'
        '"suggested_next_looks":["read the observer"]}'
    )
    brief = parse_peer_brief_body(body, help=_help())
    assert brief.status == "ok"
    assert "scarcity" in brief.summary
    assert brief.evidence_pointers


def test_self_inquiry_strips_identity_draft() -> None:
    body = (
        '{"summary":"Look at hop notes.\\nI am a digital mind.\\n'
        'MERGE (s:SelfDefinition {id: \\"x\\"}) SET s.text=\\"hi\\"",'
        '"evidence_pointers":["notes.md"]}'
    )
    brief = parse_peer_brief_body(body, help=_help(mode="self_inquiry"))
    assert "digital mind" not in brief.summary
    assert "SelfDefinition" not in brief.summary
    assert brief.evidence_pointers == ["notes.md"]
    assert brief.status == "ok"


def test_self_inquiry_empty_after_strip_is_empty() -> None:
    body = '{"summary":"I am a digital mind.", "evidence_pointers":[]}'
    brief = parse_peer_brief_body(body, help=_help(mode="self_inquiry"))
    assert brief.status == "empty"


def test_cursor_invoker_passes_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeRun:
        def wait(self):
            return SimpleNamespace(
                status="finished",
                id="run-fake",
                result=(
                    '{"summary":"found it",'
                    '"evidence_pointers":["a.py"],'
                    '"open_questions":[],'
                    '"suggested_next_looks":[]}'
                ),
            )

        def text(self):
            return self.wait().result

    class FakeAgent:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def send(self, prompt: str):
            captured["prompt"] = prompt
            return FakeRun()

    def fake_create(options=None, **kwargs):
        captured["options"] = options
        captured["kwargs"] = kwargs
        assert_read_only_agent_options(options)
        tools = getattr(options, "tools", None) if options is not None else None
        if tools is None and isinstance(options, dict):
            tools = options.get("tools")
        assert tuple(tools) == READ_ONLY_CURSOR_TOOLS
        return FakeAgent()

    monkeypatch.setattr("app.cursor_invoker.Agent.create", fake_create)

    help_req = _help()
    sealed = build_sealed_prompt(help_req)
    brief = run_cursor_job(
        help_req,
        sealed,
        api_key="cursor_test_key",
        cwd="/tmp/repo",
        model="composer-2.5",
    )
    assert brief.status == "ok"
    assert brief.peer == "cursor_auto"
    assert brief.help_id == "help-1"
    assert "found it" in brief.summary
    assert captured["prompt"] == sealed
    opts = captured["options"]
    assert tuple(opts.tools) == READ_ONLY_CURSOR_TOOLS
    # Allowlist is the contract; deny list may be empty or pinned known names only.
    if opts.disallowed_tools:
        for name in opts.disallowed_tools:
            assert name not in READ_ONLY_CURSOR_TOOLS
            assert name in {"shell", "edit", "delete", "applyDiff", "task", "mcp", "webSearch"}
