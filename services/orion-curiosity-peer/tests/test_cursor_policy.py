"""Read-only Cursor Agent CLI policy + invoker (no live agent)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.cursor_errors import TokenUnavailable
from app.cursor_invoker import (
    build_sealed_prompt,
    parse_peer_brief_body,
    run_cursor_job,
)
from app.policy import (
    assert_read_only_cli_argv,
    build_cursor_agent_argv,
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


def test_build_argv_is_ask_print_mode() -> None:
    argv = build_cursor_agent_argv(
        agent_bin="/opt/cursor-agent/versions/x/cursor-agent",
        prompt="sealed",
        workspace="/repo",
        model="composer-2.5",
    )
    assert_read_only_cli_argv(argv)
    assert argv[0].endswith("cursor-agent")
    assert "-p" in argv
    assert argv[argv.index("--mode") + 1] == "ask"
    assert "--workspace" in argv
    assert argv[argv.index("--workspace") + 1] == "/repo"
    assert "--trust" in argv
    assert "--output-format" in argv
    assert argv[argv.index("--output-format") + 1] == "text"
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "composer-2.5"
    assert argv[-1] == "sealed"
    for bad in ("--force", "--yolo", "--approve-mcps", "--plan"):
        assert bad not in argv


def test_policy_rejects_force_yolo_approve_mcps() -> None:
    base = build_cursor_agent_argv(
        agent_bin="cursor-agent",
        prompt="x",
        workspace="/repo",
    )
    for bad in ("--force", "-f", "--yolo", "--approve-mcps"):
        with pytest.raises(ValueError) as exc:
            assert_read_only_cli_argv([*base[:-1], bad, base[-1]])
        assert bad in str(exc.value) or "forbidden" in str(exc.value).lower()


def test_policy_rejects_plan_mode() -> None:
    argv = [
        "cursor-agent",
        "-p",
        "--mode",
        "plan",
        "--workspace",
        "/repo",
        "--trust",
        "prompt",
    ]
    with pytest.raises(ValueError) as exc:
        assert_read_only_cli_argv(argv)
    assert "ask" in str(exc.value).lower()


def test_policy_rejects_missing_print_or_trust_or_workspace() -> None:
    good = build_cursor_agent_argv(
        agent_bin="cursor-agent", prompt="x", workspace="/repo"
    )
    no_print = [t for t in good if t not in ("-p", "--print")]
    with pytest.raises(ValueError):
        assert_read_only_cli_argv(no_print)

    no_trust = [t for t in good if t != "--trust"]
    with pytest.raises(ValueError):
        assert_read_only_cli_argv(no_trust)

    # Drop --workspace and its value.
    idx = good.index("--workspace")
    no_ws = good[:idx] + good[idx + 2 :]
    with pytest.raises(ValueError):
        assert_read_only_cli_argv(no_ws)


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


def test_self_inquiry_strips_identity_from_list_fields() -> None:
    body = (
        '{"summary":"Check hop notes only.",'
        '"evidence_pointers":['
        '"notes.md",'
        '"MERGE (s:SelfDefinition {id: \\"x\\"}) SET s.text=\\"hi\\""'
        "],"
        '"open_questions":["I am a digital mind.","What did the hop say?"],'
        '"suggested_next_looks":["here is a SelfDefinition draft","read ledger"]}'
    )
    brief = parse_peer_brief_body(body, help=_help(mode="self_inquiry"))
    assert brief.status == "ok"
    assert "SelfDefinition" not in brief.summary
    assert brief.evidence_pointers == ["notes.md"]
    assert brief.open_questions == ["What did the hop say?"]
    assert brief.suggested_next_looks == ["read ledger"]
    assert "SelfDefinition" not in " ".join(brief.evidence_pointers)
    assert "digital mind" not in " ".join(brief.open_questions)
    assert "SelfDefinition" not in " ".join(brief.suggested_next_looks)


def test_self_inquiry_clears_lists_when_only_identity_markers() -> None:
    body = (
        '{"summary":"hop notes path is fine",'
        '"evidence_pointers":["I am a digital mind."],'
        '"open_questions":["MERGE (s:SelfDefinition {id: \\"x\\"})"],'
        '"suggested_next_looks":["here is the SelfDefinition"]}'
    )
    brief = parse_peer_brief_body(body, help=_help(mode="self_inquiry"))
    assert brief.evidence_pointers == []
    assert brief.open_questions == []
    assert brief.suggested_next_looks == []
    assert brief.status == "ok"
    assert "hop notes" in brief.summary


def test_self_inquiry_empty_after_strip_is_empty() -> None:
    body = '{"summary":"I am a digital mind.", "evidence_pointers":[]}'
    brief = parse_peer_brief_body(body, help=_help(mode="self_inquiry"))
    assert brief.status == "empty"


def test_cursor_invoker_spawns_read_only_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        assert_read_only_cli_argv(argv)
        body = (
            '{"summary":"found it",'
            '"evidence_pointers":["a.py"],'
            '"open_questions":[],'
            '"suggested_next_looks":[]}'
        )
        return SimpleNamespace(returncode=0, stdout=body, stderr="")

    help_req = _help()
    sealed = build_sealed_prompt(help_req)
    brief = run_cursor_job(
        help_req,
        sealed,
        agent_bin="/opt/cursor-agent/versions/x/cursor-agent",
        cwd="/tmp/repo",
        model="composer-2.5",
        run=fake_run,
    )
    assert brief.status == "ok"
    assert brief.peer == "cursor_auto"
    assert brief.help_id == "help-1"
    assert "found it" in brief.summary
    argv = captured["argv"]
    assert argv[-1] == sealed
    assert argv[argv.index("--mode") + 1] == "ask"
    assert "--force" not in argv
    assert "--yolo" not in argv
    assert captured["kwargs"].get("capture_output") is True


def test_cursor_invoker_maps_login_failure_to_token_unavailable() -> None:
    def fake_run(argv, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Error: not logged in. Please run `agent login`.",
        )

    with pytest.raises(TokenUnavailable):
        run_cursor_job(
            _help(),
            "sealed",
            agent_bin="cursor-agent",
            cwd="/repo",
            model="composer-2.5",
            run=fake_run,
        )


def test_cursor_invoker_raises_on_other_nonzero() -> None:
    def fake_run(argv, **kwargs):
        return SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="internal boom",
        )

    with pytest.raises(RuntimeError) as exc:
        run_cursor_job(
            _help(),
            "sealed",
            agent_bin="cursor-agent",
            cwd="/repo",
            model="composer-2.5",
            run=fake_run,
        )
    assert "exited 2" in str(exc.value)


def test_cursor_invoker_missing_binary_is_token_unavailable() -> None:
    def fake_run(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    with pytest.raises(TokenUnavailable):
        run_cursor_job(
            _help(),
            "sealed",
            agent_bin="/missing/cursor-agent",
            cwd="/repo",
            model="composer-2.5",
            run=fake_run,
        )
