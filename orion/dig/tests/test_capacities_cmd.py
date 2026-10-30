"""`dig capacities`: read-only, capped, the full reading in a file, no ledger written."""
from __future__ import annotations

import json
from pathlib import Path

from orion.body_atlas.tests.test_capacities import _body
from orion.dig import __main__ as dig
from orion.dig.tests.test_dig import run, world  # noqa: F401  (fixture)


def test_capacities_prints_every_job_and_writes_the_full_reading_only(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    code, out, _ = run(capsys, "capacities")
    assert code == 0 and len(out.splitlines()) <= dig.MAX_LINES and len(out) <= dig.MAX_CHARS
    assert "S2 coordination" in out and "pool (6 requesters / 3 workers, replies 2)" in out
    assert "servers gateway" in out
    path = Path(out.strip().splitlines()[-1].removeprefix("full result: "))
    reading = json.loads(path.read_text())
    assert reading["schema_version"] == "capacity_reading.v1" and reading["live_parts"] == 13
    written = sorted(p.relative_to(world / "home").as_posix() for p in (world / "home").rglob("*") if p.is_file())
    assert written == ["dig/t/capacities-1.json"]  # no ledger, no self/ file


def test_capacities_gaps_lists_code_written_gap_lines(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    code, out, _ = run(capsys, "capacities", "gaps")
    assert code == 0 and "I cannot remove them" in out and "- S3: I cannot see my own control" in out


def test_capacities_with_an_unknown_word_is_malformed(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    code, _, err = run(capsys, "capacities", "claim")
    assert code == 2 and "unknown or malformed" in err


def _files(world):  # noqa: F811
    return sorted(p.relative_to(world / "home").as_posix() for p in (world / "home").rglob("*") if p.is_file())


def test_capacities_ends_with_a_ready_claim_line_and_help_has_the_rule(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    code, out, _ = run(capsys, "capacities")
    assert code == 0 and out.strip().splitlines()[-2].startswith("to claim a job: python -P -m orion.dig capacity claim")
    code, out, _ = run(capsys, "capacity", "help")
    assert code == 0 and "citing by full id a part or channel on one of its flows" in out
    assert "capacity write cap:s2" in out and "a claim never clears" in out
    code, out, _ = run(capsys, "help")
    assert code == 0 and len(out) <= 2500 and "capacity claim <part> <job> because" in out
    assert "cite one of its flows by full id or a repo file" in out


def test_claim_appends_one_ledger_line_and_never_clears_a_gap(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    _, before, _ = run(capsys, "capacities", "gaps")
    code, out, err = run(capsys, "capacity", "claim", "gateway", "S3", "because", "gateway answers r0 on llm:reply")
    assert code == 0, err
    assert out.startswith("claimed gateway S3 (cites: llm:reply, r0); Orion's claim until Juniper confirms")
    assert "self/capacity_claims.jsonl" in _files(world) and "self/capacities.jsonl" not in _files(world)
    _, after, _ = run(capsys, "capacities", "gaps")
    assert before.splitlines()[:-1] == after.splitlines()[:-1]  # only the file path differs
    _, out, _ = run(capsys, "capacities")
    assert "unconfirmed: gateway S3 (my claim, not a role until Juniper confirms)" in out


def test_a_refused_claim_writes_nothing_and_says_why(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    code, _, err = run(capsys, "capacity", "claim", "pool", "S2", "because", "pool reads lease:request from r0")
    assert code == 1 and "already holds S2 by R-matchmaker" in err
    code, _, err = run(capsys, "capacity", "claim", "gateway", "extra", "S3", "because", "gateway answers r0 on llm:reply")
    assert code == 1 and "one part and one job" in err
    assert not any(f.startswith("self/") for f in _files(world))


def test_write_a_capacity_then_it_prints_in_my_words(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    code, out, err = run(capsys, "capacity", "write", "cap:s2", "pool", "hands", "out", "my", "workers", "to",
                         "whoever", "asks", "because", "R-matchmaker", "found", "pool", "between", "two", "crowds")
    assert code == 0, err
    assert out.startswith('wrote cap:s2 "pool hands out my workers to whoever asks" (cites: pool)')
    _, out, _ = run(capsys, "capacities")
    assert 'cap:s2         "pool hands out my workers to whoever asks"' in out
    code, _, err = run(capsys, "capacity", "write", "cap:s4", "I look ahead at the world every day", "because",
                       "nothing holds it yet, I just feel it")
    assert code == 1 and "gap line stands" in err


def test_juniper_confirmation_file_in_the_deployed_checkout_gives_the_role(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    conf = world / "repo" / "config" / "body_atlas" / "vsm_confirmed.yaml"
    conf.parent.mkdir(parents=True)
    conf.write_text("confirmed:\n  - {part: gateway, role: S3, date: 2026-10-31}\n")
    run(capsys, "capacity", "claim", "gateway", "S3", "because", "gateway answers r0 on llm:reply")
    _, out, _ = run(capsys, "capacities")
    assert "S3 control            CONFIRMED     confirmed by Juniper: gateway" in out
    assert "my claims: 1 (1 confirmed by Juniper)" in out
    conf.write_text("confirmed: [{part: gateway, role: nope, date: x}]\n")
    code, _, err = run(capsys, "capacities")
    assert code == 1 and "vsm_confirmed.yaml does not read" in err
