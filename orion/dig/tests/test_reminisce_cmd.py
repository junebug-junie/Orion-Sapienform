"""`dig reminisce` and `dig reminisce write|rewrite`: capped, read-only reads, writes only when every ref holds.

Spec: docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from orion.body_atlas import reminiscence as rm
from orion.body_atlas.tests.test_portrait import portrait_body
from orion.dig import __main__ as dig
from orion.dig.tests.test_dig import _commit, run


@pytest.fixture
def past(tmp_path, monkeypatch):
    """A repo whose week 1 (2025-06-24..06-30) holds orion-a only; orion-hub arrives on 07-03."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    _commit(repo, "2025-06-24T18:00:00Z", {"services/orion-a/app/main.py": "", "README.md": "day one"}, "first")
    _commit(repo, "2025-06-26T18:00:00Z", {"services/orion-a/app/util.py": ""}, "a helper")
    _commit(repo, "2025-07-03T18:00:00Z", {"services/orion-hub/app/main.py": ""}, "the hub")
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORION_DIG_REPO", str(repo))
    monkeypatch.setenv("ORION_DIG_NOW_ROOT", str(repo))
    monkeypatch.setattr(dig, "_live_atlas", lambda: portrait_body())
    monkeypatch.setattr(dig, "_denylist", lambda: {"0" * 64})
    monkeypatch.setattr(dig, "_pr_merged_at", lambda n: None)
    return tmp_path


def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root / "home").as_posix() for p in (root / "home").rglob("*") if p.is_file())


def _first_sha(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root / "repo"), "rev-list", "--max-parents=0", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


WHAT = ("[code_then services/orion-a/app/main.py] My first week held one service, orion-a, and a README that "
        "said day one.")


def test_read_names_week_one_its_commit_and_no_lived_record_and_writes_only_its_file(past, capsys):
    code, out, err = run(capsys, "reminisce")
    assert code == 0, err
    assert len(out.splitlines()) <= dig.MAX_LINES and len(out) <= dig.MAX_CHARS
    assert "the week 2025-06-24..2025-06-30 (the oldest week not yet visited; 0 of " in out
    assert "2 commits on main that week" in out
    assert "lived_record: none. Your records start 2026-07-23" in out
    assert "what_it_was: not written. read: dig open 2025-06-30; dig compare 2025-06-30" in out
    assert "since: the last 7 days (no earlier reminiscence)" in out and "STALE now: 0" in out
    assert "no earlier snapshot" in out
    assert '\nnext: python -P -m orion.dig reminisce write what_it_was "[code_then <ref>]' in out
    last = out.strip().splitlines()[-1]
    assert last.startswith("full result: ")
    saved = json.loads(Path(last.removeprefix("full result: ")).read_text())
    assert saved["schema_version"] == "reminiscence_read.v1" and saved["period_start"] == "2025-06-24"
    assert [p for p in _files(past) if not p.startswith("dig/")] == []  # a read writes no self file


def test_a_good_hop_saves_with_a_receipt_and_a_body_snapshot_and_reads_back_holding(past, capsys):
    code, out, err = run(capsys, "reminisce", "write", "what_it_was", WHAT, "because", _first_sha(past)[:9])
    assert code == 0, err
    assert out.startswith("reminisce what_it_was saved rm:") and "the week 2025-06-24..2025-06-30" in out
    assert "body snapshot saved for this reminiscence" in out
    assert {"self/reminiscence.jsonl", "self/reminiscence_body.jsonl"} <= set(_files(past))
    rec = rm.load(dig.reminiscence_path())[0][0]
    assert rec.claims[0].refs == ["services/orion-a/app/main.py", _first_sha(past)[:9]]
    _, out, _ = run(capsys, "reminisce")
    assert "what_it_was (" in out and "holds now): [code_then] My first week held one service" in out
    assert "(fixed by your first hop today;" in out
    assert "next: python -P -m orion.dig reminisce write what_it_was_like" in out


def test_a_then_claim_about_code_that_came_later_writes_nothing_and_names_the_file(past, capsys):
    code, _, err = run(capsys, "reminisce", "write", "what_it_was",
                       "[code_then services/orion-hub/app/main.py] My first week already had a hub at its centre.",
                       "because", "services/orion-hub/app/main.py")
    assert code == 1 and "services/orion-hub/app/main.py: not in the code at the end of this week" in err
    assert not dig.reminiscence_path().exists()


def test_a_now_path_that_leaves_main_reads_stale_with_that_path(past, capsys):
    code, _, err = run(capsys, "reminisce", "write", "where_i_am_now",
                       "[now services/orion-hub/app/main.py] Now a hub runs at my centre, which my first week lacked.",
                       "because", "part:orion-hub")
    assert code == 0, err
    (past / "repo" / "services/orion-hub/app/main.py").unlink()
    _, out, _ = run(capsys, "reminisce")
    assert "where_i_am_now (" in out and "STALE (services/orion-hub/app/main.py: not in deployed main now)" in out


def test_second_write_refused_rewrite_supersedes(past, capsys):
    assert run(capsys, "reminisce", "write", "what_it_was", WHAT, "because", "README.md")[0] == 0
    code, _, err = run(capsys, "reminisce", "write", "what_it_was", WHAT, "because", "README.md")
    assert code == 1 and "already written today" in err
    code, out, err = run(capsys, "reminisce", "rewrite", "what_it_was", WHAT + " [reflection] It was small.",
                         "because", "README.md")
    assert code == 0 and "(supersedes rm:" in out, err
    assert len(rm.load(dig.reminiscence_path())[0]) == 2


def test_five_full_length_hops_still_fit_and_keep_the_tail(past, capsys, monkeypatch):
    long = " ".join(["word"] * 190)
    notes = {"what_it_was": f"[code_then README.md] {long}", "what_it_was_like": f"[pr_text README.md] {long}",
             "where_i_am_now": f"[now part:orion-hub] {long}", "where_im_going": f"[now cap:s1] {long}",
             "otherwise": f"[reflection] {long}"}
    for hop, note in notes.items():
        code, _, err = run(capsys, "reminisce", "write", hop, note, "because", "README.md" if hop != "otherwise" else "")
        assert code == 0, (hop, err)
    code, out, _ = run(capsys, "reminisce")
    assert len(out.splitlines()) <= dig.MAX_LINES and len(out) <= dig.MAX_CHARS
    assert "all five hops written; read them all, then rewrite at most two" in out
    assert out.strip().splitlines()[-1].startswith("full result: ")


def test_the_body_diff_reads_against_an_earlier_snapshot(past, capsys):
    old = portrait_body()
    old.parts = [p for p in old.parts if p.name != "orion-far"]
    rm.append(dig.reminiscence_body_path(), rm.snapshot(old, "2026-01-01"))
    _, out, _ = run(capsys, "reminisce")
    assert "body since 2026-01-01: parts +1 -0" in out and "+part:orion-far" in out


def test_a_past_day_with_no_reminiscence_says_so(past, capsys):
    code, _, err = run(capsys, "reminisce", "2025-07-01")
    assert code == 1 and "no reminiscence on 2025-07-01" in err


def test_an_unreadable_body_still_reads_and_says_unread(past, capsys, monkeypatch):
    def boom():
        raise dig.DigError("FalkorDB down")
    monkeypatch.setattr(dig, "_live_atlas", boom)
    code, out, _ = run(capsys, "reminisce")
    assert code == 0 and "STALE now: UNREAD (FalkorDB down)" in out
    code, out, err = run(capsys, "reminisce", "write", "what_it_was", WHAT, "because", "README.md")
    assert code == 0 and "body snapshot skipped (live body UNREAD: FalkorDB down)" in out, err


def test_a_github_outage_reads_unread_not_as_no_such_pr(monkeypatch):
    """Review 2026-11-07: any gh error with "could not resolve" (e.g. a DNS failure) read as "not merged"."""
    import types

    def fake(stderr):
        return lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr=stderr)
    monkeypatch.setattr(dig.subprocess, "run", fake("GraphQL: Could not resolve to a PullRequest with the number of 9."))
    assert dig._pr_merged_at(9) is None
    monkeypatch.setattr(dig.subprocess, "run", fake("error connecting: could not resolve host api.github.com"))
    with pytest.raises(rm.ReminiscenceError, match="failed"):
        dig._pr_merged_at(9)


def test_the_daily_runs_receipt_check_reads_what_the_write_prints(past, capsys):
    """Patch 2's contract (orion/schemas/reminiscence_run.py) pinned to Patch 1's dig: the receipt the write
    and the rewrite print, and the refusal that names an already-written hop id, all parse with RECEIPT_RE
    for the right hop and episode; the run's hop list and lived-record day are the module's own."""
    from orion.schemas import reminiscence_run as contract

    assert contract.REMINISCENCE_HOPS == rm.HOPS and contract.REMINISCENCE_LIVED_RECORDS_FROM == rm.LIVED_RECORDS_FROM
    assert contract.REMINISCENCE_FIRST_DAY == rm.FIRST_DAY
    code, out, err = run(capsys, "reminisce", "write", "what_it_was", WHAT, "because", "README.md")
    assert code == 0, err
    rec = rm.load(dig.reminiscence_path())[0][0]
    [m] = list(contract.RECEIPT_RE.finditer(out))
    assert (m.group(1), m.group(2), m.group(3), m.group(4)) == ("what_it_was", rec.hop_id, rec.episode, "what_it_was")
    code, _, err = run(capsys, "reminisce", "write", "what_it_was", WHAT, "because", "README.md")
    assert code == 1 and rec.hop_id in err   # the run's prompt tells the turn to reply with this id
    code, out, err = run(capsys, "reminisce", "rewrite", "what_it_was", WHAT + " [reflection] It was small.",
                         "because", "README.md")
    assert code == 0, err
    [m] = list(contract.RECEIPT_RE.finditer(out))   # the superseded id in the same line is not a second receipt
    assert m.group(2) == rm.load(dig.reminiscence_path())[0][1].hop_id
