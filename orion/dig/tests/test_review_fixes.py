"""Regressions for the 2026-10-27 review of the dig kit (one test per finding)."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from orion import self_inventory
from orion.dig import __main__ as dig
from orion.dig.tests.test_dig import _commit, run, world  # noqa: F401  (fixture)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _files(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file() and p.name != dig.MARKER)


def test_concurrent_opens_of_one_day_yield_one_complete_export(world):  # noqa: F811
    repo = world / "repo"
    _commit(repo, "2025-07-05T18:00:00Z", {f"bulk/f{i}.txt": "x" * 2000 for i in range(300)}, "bulk")
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    procs = [subprocess.Popen([sys.executable, "-P", "-m", "orion.dig", "--dig", "t", "open", "2025-07-05"],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for _ in range(4)]
    results = [p.communicate(timeout=120) + (p.returncode,) for p in procs]
    assert all(code == 0 for _, _, code in results), results
    assert sum("opened 2025-07-05" in out for out, _, _ in results) == 1
    snap = world / "home" / "past" / "2025-07-05"
    expected = int(subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD"],
                                           text=True).count("\n"))
    assert _files(snap) == expected
    assert not list((world / "home" / "past").glob(f"{dig.PARTIAL_PREFIX}*"))


def test_a_killed_export_leaves_nothing_that_survives_the_next_open(world, capsys):  # noqa: F811
    leftover = world / "home" / "past" / f"{dig.PARTIAL_PREFIX}abc123"
    (leftover / "services").mkdir(parents=True)
    assert run(capsys, "open", "2025-07-01")[0] == 0
    assert not leftover.exists()


def test_history_splits_a_window_that_hits_the_search_cap(monkeypatch):
    # 3 days x 600 PRs: one query over the window would return only GH_SEARCH_CAP of 1800
    per_day = {d: [{"number": d.day * 1000 + i, "title": "t", "mergedAt": f"{d}T12:00:00Z", "body": ""}
                   for i in range(600)] for d in (date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3))}

    def fake(start, end):
        hits = [p for d, prs in per_day.items() if start <= d <= end for p in prs]
        return hits[: dig.GH_SEARCH_CAP]

    monkeypatch.setattr(dig, "_gh_merged", fake)
    prs, capped_days = dig._prs(date(2026, 1, 1), date(2026, 1, 3))
    assert len({p["number"] for p in prs}) == 1800 and capped_days == []


def test_history_names_a_day_it_could_not_finish(monkeypatch):
    monkeypatch.setattr(dig, "_gh_merged", lambda s, e: [{"number": i} for i in range(dig.GH_SEARCH_CAP)])
    _, capped_days = dig._prs(date(2026, 1, 1), date(2026, 1, 2))
    assert capped_days == ["2026-01-01", "2026-01-02"]


def test_live_body_failure_is_one_short_dig_error(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setenv("FALKORDB_URI", "redis://127.0.0.1:1")
    code, out, err = run(capsys, "body")
    assert code == 1 and err.startswith("dig: live read of") and "Traceback" not in err
    assert len(err.splitlines()) <= dig.MAX_LINES


def test_live_body_refuses_an_empty_graph(world, capsys, monkeypatch):  # noqa: F811
    from orion.body_atlas import build

    monkeypatch.setenv("FALKORDB_URI", "redis://example:6379")
    monkeypatch.setattr("orion.body_atlas.__main__._RoGraphReader.__init__", lambda self, uri, graph: None)
    monkeypatch.setattr("orion.body_atlas.__main__._RoGraphReader.close", lambda self: None)
    monkeypatch.setattr(build, "fetch_observed", lambda reader: build.ObservedBus())
    code, _, err = run(capsys, "body")
    assert code == 1 and "no PUBLISHES edges" in err


@pytest.mark.parametrize("bad", ["../../outside", "/tmp/escape", "a/b", ".hidden", ""])
def test_dig_name_cannot_leave_the_dig_home(world, capsys, bad):  # noqa: F811
    assert dig.main(["--dig", bad, "notes", "append", "hi"]) == 2
    assert "plain name" in capsys.readouterr().err


def test_notes_show_the_newest_lines(world, capsys):  # noqa: F811
    for i in range(60):
        run(capsys, "notes", "append", f"note {i}")
    _, out, _ = run(capsys, "notes")
    assert "note 59" in out and "note 0:" not in out and "showing the newest" in out
    assert len(out.splitlines()) <= dig.MAX_LINES


def test_strict_inventory_fails_loudly_on_a_wrong_root(tmp_path):
    assert self_inventory.services(tmp_path) == []
    with pytest.raises(FileNotFoundError):
        self_inventory.services(tmp_path, strict=True)
    with pytest.raises(FileNotFoundError):
        self_inventory.schemas(tmp_path, strict=True)


def test_history_by_day_count_ends_today(world, capsys, monkeypatch):  # noqa: F811
    # 2026-10-27 live turn: "a few cool PRs came out" went to the GitHub tool (2 compactions)
    seen = []
    monkeypatch.setattr(dig, "_gh_merged", lambda s, e: seen.append((s, e)) or [])
    assert run(capsys, "history", "2")[0] == 0
    today = dig.datetime.now(dig.ZONE).date()
    assert seen[0] == (today - dig.timedelta(days=1), today)
    assert dig._last_days("99999999999") == (dig.FIRST_DAY, today)  # clamped, no OverflowError
    for bad in ("0", "-1", "two", "²", "٣"):
        code, _, err = run(capsys, "history", bad)
        assert code == 1 and "day count" in err


def test_brief_points_recent_prs_at_the_capped_command():
    from orion.dig.brief import DIG_BRIEF_LINE

    assert "orion.dig history 2" in DIG_BRIEF_LINE and "not the GitHub tool" in DIG_BRIEF_LINE


def test_name_writes_the_ledger_and_live_body_shows_it(world, capsys, monkeypatch):  # noqa: F811
    from orion.body_atlas.build import build_atlas
    from orion.body_atlas.tests.test_build import CATALOG, NOW

    atlas = build_atlas(CATALOG, None, now=NOW)
    monkeypatch.setattr(dig, "_live_atlas", lambda: atlas)
    organ = next(o for o in atlas.organs if "part:orion-vision-host" in o.members)
    code, out, _ = run(capsys, "name", organ.organ_id, "Eyes", "because", "takes", "frames", "from", "orion-vision-edge")
    assert code == 0 and '"Eyes"' in out and dig.names_path().exists()
    code, out, _ = run(capsys, "body")
    assert code == 0 and f'"Eyes" {organ.organ_id}' in out
    code, _, err = run(capsys, "name", organ.organ_id, "Eyes", "because", "no citation in this reason at all")
    assert code == 1 and "cite" in err


def test_a_past_body_never_wears_todays_names(world, capsys, monkeypatch):  # noqa: F811
    from orion.body_atlas import names

    monkeypatch.setattr(names, "apply_names", lambda *a: pytest.fail("names applied to a past body"))
    run(capsys, "open", "2025-07-01")
    assert run(capsys, "body", "2025-07-01")[0] == 0


def test_history_says_how_fresh_the_clone_is(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_gh_merged", lambda s, e: [])
    assert "clone never fetched" in run(capsys, "history", "2025-07-01", "2025-07-03")[1]
    (world / "repo" / ".git" / "FETCH_HEAD").write_text("")
    assert "clone fetched 20" in run(capsys, "history", "2025-07-01", "2025-07-03")[1]


def test_name_takes_quoted_or_unquoted_words_split_at_because(world, capsys, monkeypatch):  # noqa: F811
    # review MEDIUM-2: `name X Big Eyes because ...` once stored the name "Big"
    from orion.body_atlas import names
    from orion.body_atlas.build import build_atlas
    from orion.body_atlas.tests.test_build import CATALOG, NOW

    atlas = build_atlas(CATALOG, None, now=NOW)
    monkeypatch.setattr(dig, "_live_atlas", lambda: atlas)
    oid = next(o for o in atlas.organs if "part:orion-vision-host" in o.members).organ_id
    assert run(capsys, "name", oid, "Big", "Eyes", "because", "-fed- by", "orion-vision-edge", "--dig", "x")[0] == 0
    assert run(capsys, "name", oid, "Sharp Eyes", "because", "fed by orion-vision-edge frames")[0] == 0
    assert [r.name for r in names.load(dig.names_path())[0]] == ["Big Eyes", "Sharp Eyes"]
    assert "--dig x" in names.load(dig.names_path())[0][0].because  # a reason word, not an option
    code, _, err = run(capsys, "name", oid, "Eyes", "fed by orion-vision-edge frames")
    assert code == 2 and "malformed" in err  # no `because`: refused, nothing written
    assert len(names.load(dig.names_path())[0]) == 2
