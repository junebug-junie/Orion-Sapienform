"""Context guard tracks LIVE context (rebased on CLI compaction); cut-short turns
finalize from their own findings.

Live bug 2026-09-22..10-01: `fcc_draft_length_ceiling_exceeded` killed long
autonomous turns (8/39, 15/79, 8/78 per burst day) whose drafts were 74-546
chars. The motor's running total never reset when the claude CLI compacted, so
turns that had already compacted were killed anyway, and the runner then
shipped the last text fragment ("Let me check what `outcome_from_followup`
produces:") as the answer.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock

import pytest

from orion.fcc.context_budget import is_compact_boundary_event, post_compaction_context_chars
from orion.harness import fcc_motor as motor
from orion.harness.cut_short import (
    CUT_SHORT_MARKER,
    TurnFindings,
    build_cut_short_draft,
    ensure_cut_short_marked,
)
from orion.harness.runner import HarnessRunner
from orion.harness.tests.fixtures import make_thought
from orion.harness.tests.test_fcc_motor_mcp import _FakeProc, _fake_fcc_env
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1

LIVE_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fcc"
    / "tests"
    / "fixtures"
    / "fcc_repeat_failure_a153451fe423.jsonl"
)


def _text(text: str) -> str:
    return json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}})


def _tool_round(tool_id: str, result: str) -> List[str]:
    return [
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": tool_id, "name": "Read", "input": {"file_path": f"/x/{tool_id}.py"}}]}}),
        json.dumps({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": tool_id, "content": result}]}}),
    ]


def _compact(post_tokens: int | None = None) -> str:
    meta: dict[str, Any] = {"trigger": "auto", "pre_tokens": 900}
    if post_tokens is not None:
        meta["post_tokens"] = post_tokens
    return json.dumps({"type": "system", "subtype": "compact_boundary", "session_id": "s", "compact_metadata": meta})


async def _run(monkeypatch: pytest.MonkeyPatch, lines: List[str], *, ceiling_tokens: int) -> tuple[list, _FakeProc]:
    proc = _FakeProc(lines)

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", _fake_fcc_env)
    monkeypatch.setattr(motor, "_maybe_render_mcp_config", lambda **k: None)
    monkeypatch.setenv("HARNESS_FCC_MAX_CONTEXT_TOKENS", str(ceiling_tokens))
    monkeypatch.setenv("ORION_FCC_CHARS_PER_TOKEN", "1")
    events = [ev async for ev in motor.run_fcc_turn(
        prompt="inspect", fcc_model_label="MODEL_HAIKU", correlation_id="corr-compact",
        workspace="/tmp", fcc_server_url="http://127.0.0.1:8082", auth_token="tok",
        claude_bin="claude", timeout_sec=30.0,
    )]
    return events, proc


def test_live_cli_fixture_carries_the_compact_boundary_shape() -> None:
    """Anchor: the signal is what the real CLI emitted on a live turn, not a guess."""
    events = [json.loads(line) for line in LIVE_FIXTURE.read_text().splitlines() if line.strip()]
    assert sum(is_compact_boundary_event(ev) for ev in events) == 1
    assert not is_compact_boundary_event({"type": "system", "subtype": "thinking_tokens"})
    assert is_compact_boundary_event({"type": "system", "raw": {"type": "system", "subtype": "compact_boundary"}})


def test_post_compaction_size_prefers_cli_post_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_FCC_CHARS_PER_TOKEN", "4")
    assert post_compaction_context_chars(json.loads(_compact(post_tokens=1000)), prompt_chars=7) == 4000
    assert post_compaction_context_chars(json.loads(_compact()), prompt_chars=7) == 7
    assert post_compaction_context_chars(json.loads(_compact(post_tokens=0)), prompt_chars=7) == 7


@pytest.mark.asyncio
async def test_turn_that_compacted_mid_turn_is_not_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lifetime total 7 + 300 + 300 = 607 >= 400, live context never above ~307.

    Before the fix this was killed as fcc_draft_length_ceiling_exceeded right
    after the compaction it had already done.
    """
    lines = (
        _tool_round("a", "r" * 290)
        + [_compact()]
        + _tool_round("b", "s" * 290)
        + [_text("Found it."), json.dumps({"type": "result", "result": "Found it.", "session_id": "s"})]
    )
    events, proc = await _run(monkeypatch, lines, ceiling_tokens=400)
    assert proc.killed is False
    assert events[-1]["type"] == "final"
    assert events[-1]["llm_response"] == "Found it."
    assert events[-1]["metadata"]["fcc_compactions"] == 1
    assert all(ev.get("type") != "error" for ev in events)


@pytest.mark.asyncio
async def test_genuine_runaway_without_compaction_still_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = _tool_round("a", "r" * 290) + _tool_round("b", "s" * 290) + [_text("never reached")]
    events, proc = await _run(monkeypatch, lines, ceiling_tokens=400)
    assert proc.killed is True
    assert events[-1]["type"] == "error"
    assert events[-1]["error_code"] == motor.FCC_CONTEXT_CEILING_ERROR_CODE == "fcc_context_ceiling_exceeded"
    assert events[-1]["metadata"]["fcc_compactions"] == 0


@pytest.mark.asyncio
async def test_runaway_after_compaction_counts_from_cli_post_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rebase is to the CLI's own post-compaction size, not to zero: a compaction
    that barely freed anything leaves the turn close to the ceiling."""
    lines = _tool_round("a", "r" * 100) + [_compact(post_tokens=350)] + _tool_round("b", "s" * 60)
    events, proc = await _run(monkeypatch, lines, ceiling_tokens=400)
    assert proc.killed is True
    assert events[-1]["error_code"] == "fcc_context_ceiling_exceeded"
    assert events[-1]["metadata"]["fcc_compactions"] == 1


# ---- cut-short finalize -------------------------------------------------------


def _step(raw: dict[str, Any]) -> dict[str, Any]:
    return {"type": "step", "step": {"type": str(raw.get("type")), "raw": raw}}


def _request(corr: str) -> HarnessRunRequestV1:
    return HarnessRunRequestV1(
        correlation_id=corr,
        thought_event=make_thought(),
        user_message="investigate",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )


LEAD_IN = "Let me check what `outcome_from_followup` produces:"


@pytest.mark.asyncio
async def test_cut_short_turn_finalizes_from_its_findings_not_the_last_line() -> None:
    async def _runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        for line in _tool_round("t1", "candidate table: 11090 rows, focal_edge_refs never populated"):
            yield _step(json.loads(line))
        yield _step(json.loads(_text("ontology_sparse_region is edge-blind; strength saturates at 1.0.")))
        yield _step(json.loads(_text(LEAD_IN)))
        yield {
            "type": "error",
            "error": "context ceiling",
            "error_code": "fcc_context_ceiling_exceeded",
            "llm_response": LEAD_IN,
        }

    result = await HarnessRunner(AsyncMock(), fcc_runner=_runner).run(_request("c-cut"))

    assert result.draft_text != LEAD_IN
    assert result.draft_text.startswith(CUT_SHORT_MARKER)
    assert "not a finished answer" in result.draft_text
    assert "focal_edge_refs never populated" in result.draft_text  # the tool result
    assert "/x/t1.py" in result.draft_text  # paired with the call that produced it
    assert "edge-blind" in result.draft_text  # Orion's interim note
    assert result.compliance_verdict == "partial"
    assert result.grounding_status == "fcc_context_ceiling_exceeded"
    assert result.cut_short_reason == "fcc_context_ceiling_exceeded"
    assert result.draft_molecule is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["fcc_timeout", "fcc_stream_stalled", "fcc_draft_length_ceiling_exceeded"])
async def test_every_budget_cut_uses_the_findings_path(code: str) -> None:
    async def _runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        for line in _tool_round("t1", "real evidence"):
            yield _step(json.loads(line))
        yield {"type": "error", "error": "x", "error_code": code, "llm_response": LEAD_IN}

    result = await HarnessRunner(AsyncMock(), fcc_runner=_runner).run(_request(f"c-{code}"))
    assert result.draft_text.startswith(CUT_SHORT_MARKER)
    assert "real evidence" in result.draft_text
    assert result.cut_short_reason == code


@pytest.mark.asyncio
async def test_cut_short_with_no_findings_fails_instead_of_shipping_a_shell() -> None:
    async def _runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "error", "error": "x", "error_code": "fcc_context_ceiling_exceeded", "llm_response": LEAD_IN}

    result = await HarnessRunner(AsyncMock(), fcc_runner=_runner).run(_request("c-empty"))
    assert result.draft_text == ""
    assert result.compliance_verdict == "failed"
    assert result.grounding_status == "fcc_context_ceiling_exceeded"


@pytest.mark.asyncio
async def test_non_budget_errors_keep_the_old_partial_path() -> None:
    async def _runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield _step(json.loads(_text("half an answer")))
        yield {"type": "error", "error": "x", "error_code": "fcc_nonzero_exit", "llm_response": "half an answer"}

    result = await HarnessRunner(AsyncMock(), fcc_runner=_runner).run(_request("c-nonzero"))
    assert result.draft_text == "half an answer"
    assert result.cut_short_reason is None


def test_findings_ignore_string_user_messages_and_cap_length() -> None:
    findings = TurnFindings()
    findings.observe({"type": "user", "raw": {"type": "user", "message": {
        "content": "This session is being continued from a previous conversation that ran out of context."}}})
    assert not findings.has_findings()
    for i in range(200):
        for line in _tool_round(f"t{i}", f"result-{i} " + "z" * 400):
            findings.observe({"type": "x", "raw": json.loads(line)})
    draft = build_cut_short_draft(error_code="fcc_timeout", step_count=400, findings=findings, char_budget=3000)
    assert len(draft) < 4000
    assert "result-199" in draft  # tail kept
    assert "earlier entries omitted" in draft


def test_ensure_cut_short_marked_reattaches_marker_after_repair() -> None:
    assert ensure_cut_short_marked("Here is my full answer.").startswith(CUT_SHORT_MARKER)
    already = f"{CUT_SHORT_MARKER}\nfindings"
    assert ensure_cut_short_marked(already) == already
    assert ensure_cut_short_marked(None) is None


# ---- review follow-ups ---------------------------------------------------------


@pytest.mark.asyncio
async def test_motor_flags_a_hang_after_the_cli_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI reported the turn complete, then the process hung (e.g. MCP teardown)."""
    proc = _FakeProc([_text("The full answer."), json.dumps({"type": "result", "result": "The full answer."})],
                     block_after_stdout=True)

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", _fake_fcc_env)
    monkeypatch.setattr(motor, "_maybe_render_mcp_config", lambda **k: None)
    events = [ev async for ev in motor.run_fcc_turn(
        prompt="inspect", fcc_model_label="MODEL_HAIKU", correlation_id="corr-hang",
        workspace="/tmp", fcc_server_url="http://127.0.0.1:8082", auth_token="tok",
        claude_bin="claude", timeout_sec=0.3,
    )]
    assert events[-1]["type"] == "error"
    assert events[-1]["error_code"] in ("fcc_timeout", "fcc_stream_stalled")
    assert events[-1]["metadata"]["fcc_result_seen"] is True
    assert events[-1]["llm_response"] == "The full answer."


@pytest.mark.asyncio
async def test_hang_after_result_keeps_the_answer_unmarked() -> None:
    async def _runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        for line in _tool_round("t1", "evidence"):
            yield _step(json.loads(line))
        yield _step(json.loads(_text("The full answer.")))
        yield {"type": "error", "error": "x", "error_code": "fcc_timeout",
               "llm_response": "The full answer.", "metadata": {"fcc_result_seen": True}}

    result = await HarnessRunner(AsyncMock(), fcc_runner=_runner).run(_request("c-hang"))
    assert result.draft_text == "The full answer."
    assert result.cut_short_reason is None


@pytest.mark.asyncio
async def test_reading_only_turn_keeps_the_clean_turn_error_path() -> None:
    """World-pulse reading turns need JSON and retry on `turn_error:<code>`; a findings
    draft would become a finalize JSON-parse failure that loses the real code."""
    async def _runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        for line in _tool_round("t1", "fetched page"):
            yield _step(json.loads(line))
        yield {"type": "error", "error": "x", "error_code": "fcc_timeout"}

    request = _request("c-reading")
    request = request.model_copy(update={"reading_only": True})
    result = await HarnessRunner(AsyncMock(), fcc_runner=_runner).run(request)
    assert result.draft_text == ""
    assert result.compliance_verdict == "failed"
    assert result.grounding_status == "fcc_timeout"
    assert result.cut_short_reason is None


def test_tool_output_is_scrubbed_of_credentials() -> None:
    findings = TurnFindings()
    secret_dump = (
        "POSTGRES_PASSWORD=hunter2hunter2\nAuthorization: Bearer abc.def.ghi\n"
        "url=redis://user:s3cr3tpw@host:6379/0 key ghp_ABCDEFGHIJKLMNOP1234"
    )
    for line in _tool_round("t1", secret_dump):
        findings.observe({"type": "x", "raw": json.loads(line)})
    draft = build_cut_short_draft(error_code="fcc_timeout", step_count=2, findings=findings)
    for secret in ("hunter2hunter2", "abc.def.ghi", "s3cr3tpw", "ABCDEFGHIJKLMNOP1234"):
        assert secret not in draft
    assert "[redacted]" in draft


def test_long_last_text_is_kept_whole_once_with_its_paragraphs() -> None:
    findings = TurnFindings()
    long_text = "First paragraph of the write-up.\n\n" + ("evidence line. " * 80)
    findings.observe({"type": "x", "raw": json.loads(_text(long_text))})
    draft = build_cut_short_draft(error_code="fcc_timeout", step_count=1, findings=findings, last_text=long_text)
    assert draft.count("First paragraph of the write-up.") == 1
    assert "First paragraph of the write-up.\n\nevidence line." in draft


def test_short_note_keeps_its_line_breaks() -> None:
    findings = TurnFindings()
    findings.observe({"type": "x", "raw": json.loads(_text("line one\nline two"))})
    assert "line one\n  line two" in findings.entries[0]


def test_marker_check_tolerates_re_rendering() -> None:
    rerendered = "[Cut short — not a finished answer.]\nbody"
    assert ensure_cut_short_marked(rerendered) == rerendered
    assert ensure_cut_short_marked("[cut short] body") == "[cut short] body"
