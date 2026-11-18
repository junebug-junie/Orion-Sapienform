"""`dig claude-grants`: Orion's standing settings for Claude's tools, changed one category at a time with a
reason (spec docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md section 3)."""
from __future__ import annotations

import json

from orion.dig import __main__ as dig


def run(capsys, *argv):
    code = dig.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_show_lists_every_category_at_ask_by_default(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    code, out, _ = run(capsys, "claude-grants")
    assert code == 0 and "  body: ask (default)" in out and "  people: ask (default)" in out
    assert "claude-grants set <category> allow|ask|deny because <reason>" in out
    assert len(out) <= dig.MAX_CHARS


def test_set_appends_one_line_with_the_reason_and_dig(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    code, out, _ = run(capsys, "--dig", "chat-1", "claude-grants", "set", "body", "allow", "because", "my", "costs",
                       "are", "fine", "to", "share")
    assert code == 0 and 'body ask -> allow ("my costs are fine to share")' in out
    [line] = (tmp_path / "self" / "claude_grants.jsonl").read_text().splitlines()
    rec = json.loads(line)
    assert (rec["category"], rec["setting"], rec["reason"], rec["dig"], rec["set_by"]) == (
        "body", "allow", "my costs are fine to share", "chat-1", "orion")
    code, out, _ = run(capsys, "claude-grants")
    assert '  body: allow (set ' in out and '"my costs are fine to share"' in out


def test_set_refuses_all_unknown_short_reason_and_bad_setting(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    for argv in (("all", "allow"), ("bodyy", "allow"), ("body", "always")):
        code, _, err = run(capsys, "claude-grants", "set", *argv, "because", "a perfectly fine reason")
        assert code == 1 and err.startswith("dig: claude-grants:")
    code, _, err = run(capsys, "claude-grants", "set", "body", "deny", "because", "no")
    assert code == 1 and "too short" in err
    assert not (tmp_path / "self" / "claude_grants.jsonl").exists()


def test_set_without_because_is_malformed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    code, _, err = run(capsys, "claude-grants", "set", "body", "allow", "fine", "by", "me")
    assert code == 2 and "unknown or malformed" in err
