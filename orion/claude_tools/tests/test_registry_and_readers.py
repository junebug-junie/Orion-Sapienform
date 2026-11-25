"""The tool table is a read-only allowlist; readers run argument lists, never a shell.

Acceptance checks 6 (allowlist) and 7 (redaction) of
docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md.
"""
from __future__ import annotations

import asyncio
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from orion.claude_tools import readers
from orion.claude_tools.registry import (
    BODY_ECONOMY_SECTIONS,
    INTROSPECT_TOOLS,
    TOOLS,
    ArcArguments,
    BodyEconomyArguments,
    ClaudeDreamsArguments,
    ClaudeOrionDayArguments,
    ConsultsArguments,
    PrimeNowArguments,
)
from orion.claude_tools.render import redact_tool_output
from orion.schemas.claude_tools import CLAUDE_TOOL_CATEGORIES

# Every dig subcommand that writes, every introspect write tool, and Hub's debug surface.
DIG_WRITES = {"name", "capacity", "portrait write", "field-note write", "field-note rewrite", "reminisce write",
              "reminisce rewrite", "notes append", "open", "close", "claude-grants set"}
INTROSPECT_WRITES = {"record_correction", "queue_curiosity_focus"}


def test_tool_table_names_and_categories():
    assert {n: t.category for n, t in TOOLS.items()} == {
        "prime_now": "prime", "body_economy": "body", "consults": "consults", "arc": "self",
        "dreams": "memory", "curiosity": "memory", "reading_results": "memory", "orion_day": "self",
    }
    assert all(t.category in CLAUDE_TOOL_CATEGORIES for t in TOOLS.values())
    # Every tool takes Claude's reason, and the MCP schema is small enough to list.
    for t in TOOLS.values():
        assert "reason" in t.arguments.model_fields and len(str(t.arguments.model_json_schema())) < 4000


def test_no_write_tool_is_reachable():
    names = set(TOOLS)
    assert not names & INTROSPECT_WRITES
    for t in TOOLS.values():
        text = (t.name + " " + t.description).lower()
        assert "/api/debug" not in text
    assert not {own for own, _ in INTROSPECT_TOOLS.values()} & INTROSPECT_WRITES
    # Every dig call is a read-only subcommand.
    for argv in (
        readers.body_economy_argv(BodyEconomyArguments(reason="checking a claim", section="limits")),
        readers.consults_argv(ConsultsArguments(reason="juniper asked", consult_id="23c16e4cb94f")),
        readers.consults_argv(ConsultsArguments(reason="juniper asked")),
        readers.arc_argv(ArcArguments(reason="juniper asked", id="0123456789abcdef")),
        readers.arc_argv(ArcArguments(reason="juniper asked", id="act_" + "a" * 24)),
    ):
        sub = argv[argv.index("claude-tools") + 1:]
        assert sub[0] in ("body-economy", "consult", "arc", "act")
        assert not any(" ".join(sub).startswith(w) for w in DIG_WRITES)


def test_sections_match_the_physiology_report():
    from orion.physiology.report_text import SECTIONS

    assert tuple(BODY_ECONOMY_SECTIONS) == tuple(SECTIONS)


@pytest.mark.parametrize("hostile", [
    {"section": "limits; rm -rf /"}, {"section": "--help"}, {"host": "$(id)"}, {"host": "../../etc"},
    {"window": "24h --dig ../../x"}, {"extra": 1},
])
def test_hostile_arguments_never_validate(hostile):
    with pytest.raises(ValueError):
        BodyEconomyArguments.model_validate({"reason": "a fine reason", **hostile})


def test_reason_is_required_and_bounded():
    with pytest.raises(ValueError):
        PrimeNowArguments.model_validate({})
    with pytest.raises(ValueError):
        PrimeNowArguments.model_validate({"reason": "x" * 301})


def test_body_economy_argv_is_a_list_without_a_shell():
    argv = readers.body_economy_argv(BodyEconomyArguments(reason="why not", section="hosts", host="circe",
                                                          window="7d"), python="/py")
    assert argv == ["/py", "-P", "-m", "orion.dig", "--dig", "claude-tools", "body-economy", "hosts",
                    "--window", "7d", "--host", "circe"]
    assert readers.body_economy_argv(BodyEconomyArguments(reason="why not"), python="/py")[-3:] == [
        "body-economy", "--window", "24h"]


def test_body_economy_reader_drops_the_report_path_and_deletes_its_file(tmp_path):
    report = tmp_path / "dig" / "claude-tools" / "body-economy-24h-1.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}")
    seen = {}

    def runner(argv, **kw):
        seen.update(kw, argv=argv)
        return subprocess.CompletedProcess(argv, 0, stdout=f"Limits now:\n- money: slack\nfull report: {report}\n",
                                           stderr="")

    text = readers.run_body_economy(BodyEconomyArguments(reason="a reason"), runner=runner,
                                    env={"ORION_DIG_HOME": str(tmp_path)})
    assert text == "Limits now:\n- money: slack"
    assert not report.exists()
    assert seen["timeout"] == readers.BODY_ECONOMY_TIMEOUT_SEC and "shell" not in seen


def test_body_economy_reader_never_deletes_outside_its_own_dig(tmp_path):
    other = tmp_path / "self" / "other_ledger.jsonl"
    other.parent.mkdir(parents=True)
    other.write_text("keep")

    def runner(argv, **kw):
        return subprocess.CompletedProcess(argv, 0, stdout=f"x\nfull report: {other}\n", stderr="")

    readers.run_body_economy(BodyEconomyArguments(reason="a reason"), runner=runner, env={"ORION_DIG_HOME": str(tmp_path)})
    assert other.read_text() == "keep"


@pytest.mark.parametrize("proc,why", [
    (subprocess.CompletedProcess([], 1, stdout="", stderr="dig: body-economy: report unavailable (503)"),
     "dig: body-economy: report unavailable (503)"),
    (subprocess.CompletedProcess([], 0, stdout="  \n", stderr=""), "printed nothing"),
])
def test_body_economy_reader_failures_are_unavailable(proc, why):
    import re

    with pytest.raises(readers.ReaderUnavailable, match=re.escape(why.split(":")[-1].strip()[:20])):
        readers.run_body_economy(BodyEconomyArguments(reason="a reason"), runner=lambda *a, **k: proc, env={})


def test_body_economy_timeout_is_unavailable():
    def runner(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    with pytest.raises(readers.ReaderUnavailable, match="did not answer"):
        readers.run_body_economy(BodyEconomyArguments(reason="a reason"), runner=runner, env={})


class _R:
    def __init__(self, state, view):
        self.state, self.view = state, view

    async def get(self, key):
        return self.state if key == "orion:prime:state" else self.view


def test_prime_now_absent_state_is_unavailable_not_data():
    with pytest.raises(readers.ReaderUnavailable, match="no saved state"):
        asyncio.run(readers.read_prime_now(_R(None, None), PrimeNowArguments(reason="a reason")))


def test_prime_now_returns_the_rendered_block():
    from orion.prime.state import PrimeConcernsViewV1, PrimeStateV1

    now = datetime.now(timezone.utc)
    state = PrimeStateV1().model_dump_json()
    view = PrimeConcernsViewV1(as_of=now - timedelta(seconds=20)).model_dump_json()
    text = asyncio.run(readers.read_prime_now(_R(state, view), PrimeNowArguments(reason="a reason")))
    assert text.startswith("PRIME (my slow loop between turns)") and len(text) <= 600


def test_redaction_catches_a_dsn_and_token_shapes():
    raw = ("db at postgresql://user:pass@host:5432/conjourney; GH ghp_abcdefghijklmnopqrstuvwxyz0123456789; "
           "Authorization: Bearer abcdef0123456789abcdef; ORION_GITHUB_TOKEN=abcd1234efgh5678; "
           "key sk-ant-api03-abcdefghijklmnopqrstuv")
    out = redact_tool_output(raw, secret_values=())
    for leaked in ("user:pass", "ghp_abc", "abcdef0123456789abcdef", "abcd1234efgh5678", "sk-ant-api03"):
        assert leaked not in out, leaked
    # Ordinary body-economy words survive.
    assert redact_tool_output("tokens: 1,234 per hour; 412 W", secret_values=()) == "tokens: 1,234 per hour; 412 W"


def test_redaction_catches_credential_env_values():
    out = redact_tool_output("value is s3cr3t-value-123", secret_values=("s3cr3t-value-123",))
    assert "s3cr3t" not in out


@pytest.mark.parametrize("model,hostile", [
    (ConsultsArguments, {"consult_id": "23c16e4cb94f; id"}),
    (ConsultsArguments, {"consult_id": "--help"}),
    (ConsultsArguments, {"consult_id": "23C16E4CB94F"}),
    (ArcArguments, {"id": "../../etc/passwd"}),
    (ArcArguments, {"id": "act_123"}),
    (ArcArguments, {}),
    (ClaudeDreamsArguments, {"limit": 500}),
    (ClaudeDreamsArguments, {"extra": 1}),
])
def test_hostile_new_arguments_never_validate(model, hostile):
    with pytest.raises(ValueError):
        model.model_validate({"reason": "a fine reason", **hostile})


def test_consult_and_arc_argv():
    assert readers.consults_argv(ConsultsArguments(reason="why not", consult_id="23c16e4cb94f"), python="/py") == [
        "/py", "-P", "-m", "orion.dig", "--dig", "claude-tools", "consult", "23c16e4cb94f"]
    assert readers.consults_argv(ConsultsArguments(reason="why not"), python="/py")[-1] == "consult"
    assert readers.arc_argv(ArcArguments(reason="why not", id="act_" + "b" * 24), python="/py")[-2:] == [
        "act", "act_" + "b" * 24]
    assert readers.arc_argv(ArcArguments(reason="why not", id="0123456789abcdef"), python="/py")[-2] == "arc"


def test_dig_reader_drops_any_full_result_line_and_its_file(tmp_path):
    report = tmp_path / "dig" / "claude-tools" / "consult-23c16e4cb94f-1.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}")

    def runner(argv, **kw):
        return subprocess.CompletedProcess(
            argv, 0, stdout=f"consult 23c16e4cb94f: brief_sent\nfull consult (every turn, untrimmed): {report}\n",
            stderr="")

    text = readers.run_dig(["x"], "consult", runner=runner, env={"ORION_DIG_HOME": str(tmp_path)})
    assert text == "consult 23c16e4cb94f: brief_sent" and not report.exists()


class _Introspect:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.calls = result, exc, []

    async def invoke(self, name, arguments):
        self.calls.append((name, arguments))
        if self.exc:
            raise self.exc
        return self.result


def _item(i, sensitivity=None, text="a dream about the cabinet"):
    return {"id": f"d{i}", "occurred_at": "2026-11-24T08:00:00+00:00", "kind": "dream_narrative",
            "epistemic_status": "unsettled", "text": text, "sensitivity": sensitivity, "extra": {"themes": ["heat"]}}


def test_introspect_reader_strips_reason_and_drops_intimate_items():
    tools = _Introspect({"ok": True, "operation": "dreams", "as_of": "2026-11-25T00:00:00+00:00",
                         "total_available": 3, "items": [_item(1), _item(2, "intimate", "SECRET"), _item(3, "private")]})
    read = readers.introspect_reader(tools, "dreams")
    text = asyncio.run(read(ClaudeDreamsArguments(reason="juniper asked", query="cabinet")))
    assert tools.calls == [("dreams", {"query": "cabinet", "limit": 5})]
    assert "SECRET" not in text and "d2" not in text and "1 intimate item(s) withheld" in text
    assert "d1" in text and "d3" in text and '"themes": ["heat"]' in text


def test_introspect_reader_empty_says_nothing_matched():
    tools = _Introspect({"ok": True, "operation": "orion_day", "as_of": "2026-11-25T00:00:00+00:00", "items": []})
    text = asyncio.run(readers.introspect_reader(tools, "orion_day")(ClaudeOrionDayArguments(reason="juniper asked")))
    assert "nothing matched" in text and tools.calls == [("orion_day", {"part": "list", "limit": 5})]


def test_introspect_reader_error_is_unavailable_not_empty():
    from orion.introspect.tools import IntrospectUnknownError

    tools = _Introspect(exc=IntrospectUnknownError("dreams: answer unknown (no reply: TimeoutError)"))
    with pytest.raises(readers.ReaderUnavailable, match="answer unknown"):
        asyncio.run(readers.introspect_reader(tools, "dreams")(ClaudeDreamsArguments(reason="juniper asked")))


def test_introspect_item_text_is_cut():
    tools = _Introspect({"ok": True, "operation": "dreams", "as_of": "2026-11-25T00:00:00+00:00",
                         "items": [_item(1, text="x" * 5000)]})
    text = asyncio.run(readers.introspect_reader(tools, "dreams")(ClaudeDreamsArguments(reason="juniper asked")))
    assert "x" * readers.ITEM_TEXT_CHARS not in text and "…" in text


def test_only_the_last_line_counts_as_dig_full_result_line(tmp_path):
    other = tmp_path / "dig" / "claude-tools" / "someone-elses-call-1.json"
    other.parent.mkdir(parents=True)
    other.write_text("{}")
    quoted = f"full report: {other}"

    def runner(argv, **kw):
        return subprocess.CompletedProcess(argv, 0, stdout=f"turn 2 said:\n{quoted}\nend of turn\n\n", stderr="")

    text = readers.run_dig(["x"], "consult", runner=runner, env={"ORION_DIG_HOME": str(tmp_path)})
    assert quoted in text and other.exists()
