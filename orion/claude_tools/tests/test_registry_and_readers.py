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
from orion.claude_tools.registry import BODY_ECONOMY_SECTIONS, TOOLS, BodyEconomyArguments, PrimeNowArguments
from orion.claude_tools.render import redact_tool_output
from orion.schemas.claude_tools import CLAUDE_TOOL_CATEGORIES

# Every dig subcommand that writes, every introspect write tool, and Hub's debug surface.
DIG_WRITES = {"name", "capacity", "portrait write", "field-note write", "field-note rewrite", "reminisce write",
              "reminisce rewrite", "notes append", "open", "close", "claude-grants set"}
INTROSPECT_WRITES = {"record_correction", "queue_curiosity_focus"}


def test_patch_1b_has_exactly_the_two_tools_and_their_categories():
    assert {n: t.category for n, t in TOOLS.items()} == {"prime_now": "prime", "body_economy": "body"}
    assert all(t.category in CLAUDE_TOOL_CATEGORIES for t in TOOLS.values())


def test_no_write_tool_is_reachable():
    names = set(TOOLS)
    assert not names & INTROSPECT_WRITES
    for t in TOOLS.values():
        text = (t.name + " " + t.description).lower()
        assert "/api/debug" not in text
    # body_economy's argv is the only dig call, and it is the read-only body-economy subcommand.
    argv = readers.body_economy_argv(BodyEconomyArguments(reason="checking a claim", section="limits"))
    sub = argv[argv.index("claude-tools") + 1:]
    assert sub[0] == "body-economy"
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
