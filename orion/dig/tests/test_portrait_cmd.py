"""`dig portrait` and `dig portrait write`: capped, the full portrait in a file, a write only when it holds."""
from __future__ import annotations

import json
from pathlib import Path

from orion.body_atlas.tests.test_portrait import portrait_body
from orion.dig import __main__ as dig
from orion.dig.tests.test_dig import run, world  # noqa: F401  (fixture)

WATTS = {"athena": {"host": "athena", "chassis_w_now": 362.0, "chassis_w_rest": 340.0,
                    "usd_per_month_at_this_rate": 2.02}}


def _setup(monkeypatch, *, hub_on: str = "athena", watts=WATTS):
    monkeypatch.setattr(dig, "_live_atlas", lambda: portrait_body(hub_on=hub_on))
    monkeypatch.setattr(dig, "_host_watts", lambda: watts)


def _files(world):  # noqa: F811
    return sorted(p.relative_to(world / "home").as_posix() for p in (world / "home").rglob("*") if p.is_file())


def test_portrait_is_read_only_capped_and_ends_with_ready_lines_and_the_file(world, capsys, monkeypatch):  # noqa: F811
    _setup(monkeypatch)
    code, out, err = run(capsys, "portrait")
    assert code == 0, err
    assert len(out.splitlines()) <= dig.MAX_LINES and len(out) <= dig.MAX_CHARS
    assert "I am: not written yet" in out and "athena live orchestration · 362 W now, rest 340" in out
    assert "nothing of mine runs on (facts only): hecate" in out
    assert "  python -P -m orion.dig portrait write athena " in out
    last = out.strip().splitlines()[-1]
    assert last.startswith("full result: ")
    saved = json.loads(Path(last.removeprefix("full result: ")).read_text())
    assert saved["schema_version"] == "portrait.v1" and {m["name"] for m in saved["machines"]} == {"athena", "circe", "hecate"}
    assert _files(world) == ["dig/t/portrait-1.json"]  # no ledger written by a read


def test_turn_2544158f_machine_claim_is_saved_says_it_holds_and_reads_back(world, capsys, monkeypatch):  # noqa: F811
    """Turn 2544158f: "Athena claims orchestration and vision ... she carries organs substrate, cortex" was
    prose only ("ledger entry #4", nothing saved). The same claim is now one lasting, checked line."""
    _setup(monkeypatch)
    code, out, err = run(capsys, "portrait", "write", "athena",
                         "athena carries cortex and substrate: orion-hub and orion-gpu-pool run here",
                         "because", "orion-hub and orion-gpu-pool run on athena")
    assert code == 0, err
    assert out.startswith('wrote host:athena "athena carries cortex and substrate')
    assert "it holds now; code re-checks it on every `portrait` read" in out
    assert "self/portrait.jsonl" in _files(world)
    _, out, _ = run(capsys, "portrait")
    assert "    lived_record " in out and "holds now: \"athena carries cortex and substrate" in out
    assert "portrait write athena" not in out  # written: no ready line for it any more


def test_a_write_that_would_not_hold_writes_nothing_and_names_the_id(world, capsys, monkeypatch):  # noqa: F811
    _setup(monkeypatch)
    code, _, err = run(capsys, "portrait", "write", "athena", "athena runs orion-hub and answers orion-elsewhere",
                       "because", "orion-elsewhere asks orion-hub")
    assert code == 1 and "not written" in err and "orion-elsewhere: not on athena, and no flow joins it" in err
    code, _, err = run(capsys, "portrait", "write", "hecate", "hecate is my deep thinking node, asleep",
                       "because", "one V100 seated, one in the mail")
    assert code == 1 and "hecate: nothing live of mine runs there now (dormant)" in err
    assert not any(f.startswith("self/") for f in _files(world))


def test_a_moved_part_makes_the_line_stale_on_the_next_read(world, capsys, monkeypatch):  # noqa: F811
    _setup(monkeypatch)
    run(capsys, "portrait", "write", "athena", "athena runs orion-hub, where Juniper talks to me",
        "because", "orion-hub runs on athena today")
    _setup(monkeypatch, hub_on="circe")
    _, out, _ = run(capsys, "portrait")
    assert "STALE (anchor: cite at least one part that runs on athena" in out
    assert "Rewrite 1 STALE line(s)" in out


def test_the_portrait_says_when_watts_are_unread_and_still_prints(world, capsys, monkeypatch):  # noqa: F811
    _setup(monkeypatch, watts="the body report at http://x is unreachable (refused)")
    code, out, _ = run(capsys, "portrait")
    assert code == 0 and "watts UNREAD: the body report at http://x is unreachable" in out


def test_two_bare_becauses_are_refused_and_a_quoted_one_is_kept(world, capsys, monkeypatch):  # noqa: F811
    _setup(monkeypatch)
    code, _, err = run(capsys, "portrait", "write", "circe", "circe", "matters", "because", "llm-worker-a", "speaks",
                       "because", "llm-worker-a runs on circe")
    assert code == 1 and "two bare `because` words; quote both" in err
    assert not any(f.startswith("self/") for f in _files(world))
    code, out, err = run(capsys, "portrait", "write", "circe", "circe matters because llm-worker-a speaks for me",
                         "because", "llm-worker-a runs on circe")
    assert code == 0, err
    assert out.startswith('wrote host:circe "circe matters because llm-worker-a speaks for me"')


def test_help_lists_the_portrait(world, capsys):  # noqa: F811
    code, out, _ = run(capsys, "help")
    assert code == 0 and "portrait [write <machine|whole> <sentence> because <reason>]" in out
    assert "`portrait`, never from memory" in out


def test_portrait_with_an_unknown_word_is_malformed(world, capsys, monkeypatch):  # noqa: F811
    _setup(monkeypatch)
    code, _, err = run(capsys, "portrait", "athena")
    assert code == 2 and "unknown or malformed" in err
