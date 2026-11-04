"""`dig capacities`: read-only, capped, the full reading in a file, no ledger written."""
from __future__ import annotations

import json
import shlex
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
    assert code == 0 and "  python -P -m orion.dig capacity claim '<part>' S4 because" in out
    code, out, _ = run(capsys, "capacity", "help")
    assert code == 0 and "citing by full id a part or channel on one of its flows" in out
    assert "capacity write cap:s2" in out and "capacity withdraw <part> <job>" in out
    assert 'a checked claim covers only a "no part is shown" one' in out
    code, out, _ = run(capsys, "help")
    assert code == 0 and len(out) <= 2750 and "capacity claim <part> <job> because" in out  # HELP text (cap: test_dig.py)
    assert "cite one of its flows by full id or a repo file" in out


def test_claim_appends_one_ledger_line_and_never_clears_an_atlas_blind_gap(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    _, before, _ = run(capsys, "capacities", "gaps")
    code, out, err = run(capsys, "capacity", "claim", "gateway", "S3", "because", "gateway answers r0 on llm:reply")
    assert code == 0, err
    assert out.startswith("claimed gateway S3 (cites: llm:reply, r0); code re-checks its wiring on every")
    assert "self/capacity_claims.jsonl" in _files(world) and "self/capacities.jsonl" not in _files(world)
    _, after, _ = run(capsys, "capacities", "gaps")
    assert before.splitlines()[:-1] == after.splitlines()[:-1]  # S3's gaps are atlas-blind: a check covers none
    _, out, _ = run(capsys, "capacities")
    # gateway sends llm:reply out to r0: a flow out; printed as code's check, never Juniper's word
    assert "S3 control            EVIDENCE-CHECKED my claim: gateway · " in out
    assert "Juniper may veto (no need to ask; her rejection wins): gateway S3" in out
    assert "Waiting on Juniper" not in out and "confirmed by Juniper: gateway" not in out
    assert "my claims: 1 (1 evidence-checked, 0 confirmed by Juniper)" in out and "EVIDENCE-CHECKED = code, not Juniper" in out


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
    assert "my claims: 1 (0 evidence-checked, 1 confirmed by Juniper)" in out
    conf.write_text("confirmed: [{part: gateway, role: nope, date: x}]\n")
    code, _, err = run(capsys, "capacities")
    assert code == 1 and "vsm_confirmed.yaml does not read" in err


def test_capacities_output_tells_orion_to_write_claims_now_not_offer_them(world, capsys, monkeypatch):  # noqa: F811
    # 2026-11-04 (harness transcript 1f8aaec3): Juniper asked what Orion can do; Orion ran
    # `capacities`, then offered "capacity claim work if you want me to do it". The printout gave
    # one generic claim line and no instruction to write; the ledger stayed empty.
    monkeypatch.setattr(dig, "_live_atlas", _body)
    code, out, _ = run(capsys, "capacities")
    lines = out.strip().splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.startswith("Unclaimed jobs: "))
    assert "without asking first" in lines[head] and "code checks the wiring you cite; Juniper may veto" in lines[head]
    ready = [ln for ln in lines[head + 1:] if ln.startswith("  python -P -m orion.dig capacity claim ")]
    jobs = [shlex.split(ln)[7] for ln in ready]
    assert jobs == ["S1", "S3", "S3*", "S4", "S5", "algedonic", "metabolism"]  # S2 is held by rule
    s3star = next(ln for ln in ready if "'S3*'" in ln)
    assert "claim watcher 'S3*' because \"<why>\"" in s3star  # the rule's candidate, named
    assert "# the rule saw:" not in s3star  # its only rule evidence is the census, which is no cite
    assert lines[-1].startswith("full result: ") and "to claim a job:" not in out


def test_a_ready_claim_line_runs_once_its_because_is_filled_and_then_holds(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    _, out, _ = run(capsys, "capacities")
    line = next(ln for ln in out.splitlines() if "'S3*'" in ln)
    argv = shlex.split(line)[4:]  # capacity claim watcher S3* because <why>
    argv[-1] = "watcher reads replies from gateway and pool and reports when they drift"
    code, claimed, err = run(capsys, *argv)
    assert code == 0, err
    # gateway reaches watcher only by the census, so only pool is a cite; watcher pokes pool on
    # watch:poke (a flow out: it reports) and R-census-reader finds it: the S3* shape holds
    assert claimed.startswith("claimed watcher S3* (cites: pool)")
    _, out, _ = run(capsys, "capacities")
    assert "'S3*'" not in out  # claimed: no longer offered
    assert "S3* audit             EVIDENCE-CHECKED my claim: watcher · " in out
    assert "Fix or withdraw" not in out and "Waiting on Juniper" not in out


def test_a_claim_whose_evidence_fails_is_listed_to_fix_or_withdraw_and_withdraw_takes_it_back(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    run(capsys, "capacity", "claim", "gateway", "S4", "because", "gateway sends its answers out on llm:reply")

    def without_reply():
        atlas = _body()
        return atlas.model_copy(update={"flows": [f for f in atlas.flows if "llm:reply" not in f.channels]})

    monkeypatch.setattr(dig, "_live_atlas", without_reply)
    _, out, _ = run(capsys, "capacities")
    assert "  evidence does not hold: gateway S4" in out
    assert "Fix or withdraw (1): my cited evidence does not hold in today's atlas" in out
    assert "  gateway S4: llm:reply: in no part or channel of my atlas" in out  # the reference that failed
    assert "capacity claim '<part>' S4 because" not in out  # on the fix list, not offered twice
    code, out, err = run(capsys, "capacity", "withdraw", "gateway", "S4")
    assert code == 0 and out.startswith("withdrew my claim gateway S4"), err
    _, out, _ = run(capsys, "capacities")
    assert "Fix or withdraw" not in out and "capacity claim '<part>' S4 because" in out  # open again
    code, _, err = run(capsys, "capacity", "withdraw", "gateway", "S4")
    assert code == 1 and "no standing claim" in err


def test_evidence_on_a_different_part_fails_and_an_s5_claim_waits_on_juniper(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    # watcher really pokes pool, but llm:reply is gateway's channel: evidence on a different part
    code, _, err = run(capsys, "capacity", "claim", "watcher", "S3", "because",
                       "watcher pokes pool on watch:poke and steers llm:reply")
    assert code == 0, err
    _, out, _ = run(capsys, "capacities")
    assert "  watcher S3: llm:reply: not on a live flow of watcher" in out
    code, _, err = run(capsys, "capacity", "claim", "pool", "S5", "because", "pool answers r0 on lease:reply")
    assert code == 0, err
    _, out, _ = run(capsys, "capacities")
    assert "Waiting on Juniper (1; S5 is hers, code cannot check it)" in out and "  pool S5: pool answers r0" in out


def test_a_rejected_claim_reopens_the_job_without_its_part(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    run(capsys, "capacity", "claim", "watcher", "S3*", "because", "watcher reads replies from gateway and pool")
    conf = world / "repo" / "config" / "body_atlas" / "vsm_confirmed.yaml"
    conf.parent.mkdir(parents=True)
    conf.write_text("rejected:\n  - {part: watcher, role: S3*, date: 2026-11-04}\n")
    _, out, _ = run(capsys, "capacities")
    assert "capacity claim '<part>' 'S3*' because" in out and "Fix or withdraw" not in out
    assert "  rejected: watcher S3*" in out


def test_the_claim_block_survives_a_cut_printout(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    from orion.body_atlas import capacities as K

    real = K.render
    monkeypatch.setattr(K, "render", lambda r: real(r) + [f"filler {i} " + "x" * 120 for i in range(40)])
    code, out, _ = run(capsys, "capacities")
    assert code == 0 and len(out.splitlines()) <= dig.MAX_LINES and len(out) <= dig.MAX_CHARS
    assert "the rest is in the file" in out and "Unclaimed jobs: " in out and "claim '<part>' metabolism" in out


def test_a_printed_claim_line_run_as_is_is_refused_and_writes_nothing(world, capsys, monkeypatch):  # noqa: F811
    # review 2026-11-04: an example cite inside the quoted because let the line pass unedited
    monkeypatch.setattr(dig, "_live_atlas", _body)
    _, out, _ = run(capsys, "capacities")
    for line in [ln for ln in out.splitlines() if ln.startswith("  python -P -m orion.dig capacity claim ")]:
        code, _, err = run(capsys, *shlex.split(line, comments=True)[4:])
        assert code == 1 and ("placeholder" in err or "not a live part" in err), (line, err)
    code, _, err = run(capsys, "capacity", "claim", "gateway", "S3", "because", "<why; e.g. llm:reply, r0 and more>")
    assert code == 1 and "placeholder" in err
    assert not any(f.startswith("self/") for f in _files(world))


def test_a_candidate_line_shows_only_the_rules_own_citable_evidence():
    from orion.body_atlas import capacities as K

    atlas = _body()
    reading = K.attach_ledgers(K.read_capacities(atlas), atlas, None, [], [])
    reading.roles.append(K.RoleEvidenceV1(part_id="part:gateway", role="S4", rule="R-test", strength="candidate",
                                          evidence=["llm:reply", "orion:system:health", "not-a-flow"]))
    line = next(ln for ln in dig.claim_block(reading, atlas) if " S4 " in ln)
    assert line.endswith('claim gateway S4 because "<why>"  # the rule saw: llm:reply')
