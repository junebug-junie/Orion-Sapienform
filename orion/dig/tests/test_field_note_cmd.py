"""`dig field-note` / `field-notes` / `history` marks (spec 2026-11-04-pr-field-notes-design.md, Patch 1)."""
from __future__ import annotations

import hashlib
import json

from pathlib import Path

import pytest

from orion.dig import __main__ as dig
from orion.dig.tests.test_body_whole import _whole_atlas
from orion.dig.tests.test_dig import run, world  # noqa: F401  (fixture)


def _pr(number=7, paths=("services/orion-cortex-exec/app/x.py", "docs/superpowers/specs/s.md"), adds=120,
        merged="2025-07-03T18:00:00Z", **kw):
    pr = {"number": number, "title": f"feat: pr {number}", "body": "## Summary\n- moved the plan step\n" + "x" * 9000,
          "state": "MERGED", "mergedAt": merged, "mergeCommit": {"oid": "f" * 40}, "labels": [],
          "files": [{"path": p, "additions": 5, "deletions": 1} for p in paths], "additions": adds, "deletions": 1,
          "changedFiles": len(paths), "headRefName": "feat/x", "commits": [{"authors": [{"email": "a@b"}]}]}
    pr.update(kw)
    return pr


@pytest.fixture
def gh(world, monkeypatch):  # noqa: F811
    prs = {7: _pr(), 8: _pr(8, paths=("docs/a.md",), adds=3), 9: _pr(9, state="OPEN", mergedAt=None)}
    monkeypatch.setattr(dig, "_gh_pr", lambda n: prs[n])
    monkeypatch.setattr(dig, "_gh_file_at", lambda sha, p: "\n".join(f"spec line {i}" for i in range(80)))
    monkeypatch.setattr(dig, "_live_atlas", _whole_atlas)
    deny = world / "repo" / "config" / "privacy" / "personal_terms.sha256"
    deny.parent.mkdir(parents=True)
    deny.write_text(hashlib.sha256(b"zorblax").hexdigest() + "\n")
    return prs


def test_read_prints_the_body_map_docs_and_a_write_line_within_the_caps(gh, capsys):
    code, out, err = run(capsys, "field-note", "7")
    assert code == 0, err
    assert len(out.splitlines()) <= dig.MAX_LINES and len(out) <= dig.MAX_CHARS + 1
    assert "services/orion-cortex-exec/ (1 files) -> part:orion-cortex-exec in organ organ:" in out
    assert "not body: docs 1" in out
    assert "cut: docs/superpowers/specs/s.md: first 60 of 80 lines" in out
    assert 'python -P -m orion.dig field-note 7 write "[pr_text] ... [reflection] ..." because "#7' in out
    assert "x" * 300 not in out  # the body is cut in the printout, whole in the file
    full = json.loads(Path(out.strip().splitlines()[-1].split(": ", 1)[1]).read_text())
    assert len(full["pr"]["body"]) > 9000 and full["read"][0]["truncated"]
    assert not dig.field_notes_path().exists()  # read-only


def test_a_huge_pr_packet_stays_inside_the_caps_and_says_so(gh, capsys):
    gh[7] = _pr(paths=tuple(f"services/s{i}/app/f.py" for i in range(100)), adds=31000, changedFiles=130)
    code, out, _ = run(capsys, "field-note", "7")
    assert code == 0
    assert len(out.splitlines()) <= dig.MAX_LINES and len(out) <= dig.MAX_CHARS + 1
    assert "huge: say in the note what you could not read" in out
    assert "file list: first 100 of 130 files" in out or "the rest is in the file" in out


def test_write_saves_one_line_and_prints_the_receipt_then_refuses_a_second(gh, capsys):
    note = "[pr_text] The builders moved the plan step into the executor, because it was split. [reflection] I may plan faster."
    code, out, err = run(capsys, "field-note", "7", "write", note, "because", "#7")
    assert code == 0, err
    assert out.startswith("field-note #7 saved fn:7:")
    lines = dig.field_notes_path().read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["note"].startswith("The builders moved")  # last `because` splits
    code, _, err = run(capsys, "field-note", "7", "write", note, "because", "#7")
    assert code == 1 and "already has a note" in err
    code, out, err = run(capsys, "field-note", "7", "rewrite", "[reflection] On a second look the plan step move is small.",
                         "because", "#7")
    assert code == 0, err
    assert "supersedes fn:7:" in out
    assert len(dig.field_notes_path().read_text().splitlines()) == 2
    code, out, _ = run(capsys, "field-note", "7")
    assert "your note (fn:7:" in out and "On a second look" in out and "rewrite" in out


@pytest.mark.parametrize("note,because,words", [
    ("no label here at all, just some words about it", "#7", "carries a label"),
    ("[pr_text] The builders moved the plan step into the executor.", "#99", "refs must be #7"),
    ("[pr_text] Zorblax moved the plan step into the executor today.", "#7", "personal term"),
])
def test_a_bad_write_is_refused_and_nothing_is_saved(gh, capsys, note, because, words):
    code, _, err = run(capsys, "field-note", "7", "write", note, "because", because)
    assert code == 1 and words in err
    assert not dig.field_notes_path().exists()


def test_tiny_docs_and_unmerged_prs_get_no_note(gh, capsys):
    code, out, _ = run(capsys, "field-note", "8")
    assert code == 0 and "code record, not a note" in out and "write:" not in out
    code, _, err = run(capsys, "field-note", "8", "write", "[pr_text] A docs fix to the dig README, nothing more.", "because", "#8")
    assert code == 1 and "code record, not a note" in err
    code, _, err = run(capsys, "field-note", "9")
    assert code == 1 and "not merged" in err


def test_write_waits_when_the_live_body_is_unread(gh, capsys, monkeypatch):
    def boom():
        raise dig.DigError("live read failed")
    monkeypatch.setattr(dig, "_live_atlas", boom)
    code, out, _ = run(capsys, "field-note", "7")
    assert code == 0 and "live body UNREAD" in out
    code, _, err = run(capsys, "field-note", "7", "write", "[pr_text] The builders moved the plan step into the executor.",
                       "because", "#7")
    assert code == 1 and "Not saved" in err and not dig.field_notes_path().exists()


def test_field_notes_and_history_marks_show_coverage(gh, capsys, monkeypatch):
    monkeypatch.setattr(dig, "_gh_merged", lambda s, e: [
        {"number": 7, "title": "feat: pr 7", "mergedAt": "2025-07-03T18:00:00Z", "body": ""},
        {"number": 10, "title": "feat: pr 10", "mergedAt": "2025-07-03T19:00:00Z", "body": ""}])
    run(capsys, "field-note", "7", "write", "[pr_text] The builders moved the plan step into the executor.", "because", "#7")
    code, out, err = run(capsys, "field-notes", "2025-07-03", "2025-07-03")
    assert code == 0, err
    assert "1 field notes, 0 code records; coverage 1/2 merged PRs (50%)" in out
    assert "#7 2025-07-03 The builders moved" in out and "no note yet: #10" in out
    code, out, _ = run(capsys, "history", "2025-07-01", "2025-07-03")
    assert "#7 2025-07-03 feat: pr 7 (noted)" in out and "#10 2025-07-03 feat: pr 10 (no note)" in out


def test_help_lists_the_commands_and_bad_numbers_are_refused(world, capsys):  # noqa: F811
    code, out, _ = run(capsys, "help")
    assert "field-note <pr> [write|rewrite" in out and "field-notes <days>" in out
    code, _, err = run(capsys, "field-note", "abc")
    assert code == 1 and "not a PR number" in err
