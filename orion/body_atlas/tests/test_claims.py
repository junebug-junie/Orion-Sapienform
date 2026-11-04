"""Patch 2: claims need evidence on the part's own flows and never clear gaps; Juniper's
confirmations give a role and cover "no part is shown" gaps; capacities cite only holders.
2026-11-04: a claim holds its role (EVIDENCE-CHECKED) while its evidence checks on each read."""
from __future__ import annotations

from pathlib import Path

import pytest

from orion.body_atlas import capacities as K
from orion.body_atlas import claims as C
from orion.body_atlas.model import FlowV1, PartV1
from orion.body_atlas.tests.test_capacities import _body

_REPO = Path(__file__).resolve().parents[3]


def _ok(*held):
    """A recheck whose evidence check passes, citing ``held``."""
    return lambda c: K.EvidenceCheckV1(part_id=c.part_id, role=c.role, at=c.at, holds=True, held=list(held))


def _gone(c):
    """A recheck whose evidence check fails."""
    return K.EvidenceCheckV1(part_id=c.part_id, role=c.role, at=c.at, holds=False, failed=["x: gone"])


def _claim(reading, atlas, part, role, because, kind="claim", root=_REPO):
    return C.check_claim(reading, atlas, part, role, because, kind=kind, root=root, dig="t")


def _vsm(*confirmed, rejected=()) -> K.VsmConfirmedV1:
    return K.VsmConfirmedV1(confirmed=[K.ConfirmationV1(part=p, role=r, date="2026-10-30") for p, r in confirmed],
                            rejected=[K.ConfirmationV1(part=p, role=r, date="2026-10-30") for p, r in rejected])


def test_a_claim_must_cite_a_flow_of_that_part_or_a_repo_file():
    atlas = _body()
    r = K.read_capacities(atlas)
    rec = _claim(r, atlas, "gateway", "s3", "gateway answers r0 on llm:reply, so it shapes what they do")
    assert rec.role == "S3" and rec.cites == ["llm:reply", "r0"] and rec.part_id == "part:gateway"
    with pytest.raises(C.ClaimError, match="cite"):
        _claim(r, atlas, "gateway", "S3", "gateway is like w0 and w1, which it never talks to")
    with pytest.raises(C.ClaimError, match="cite"):  # the census is everyone's; not evidence of a job
        _claim(r, atlas, "watcher", "S3", "watcher reads orion:system:health from the whole body")
    rec = _claim(r, atlas, "gateway", "S3", "see orion/body_atlas/capacities.py and orion/nope/missing.py.")
    assert rec.cites == ["orion/body_atlas/capacities.py"]  # the missing file is not a cite
    with pytest.raises(C.ClaimError, match="cite"):
        _claim(r, atlas, "gateway", "S3", "see ../../etc/passwd and orion/../README.md please")


def test_a_claim_cannot_take_a_rule_result_but_a_dispute_can_argue_with_one():
    atlas = _body()
    r = K.read_capacities(atlas)
    with pytest.raises(C.ClaimError, match="already holds S2 by R-matchmaker"):
        _claim(r, atlas, "pool", "S2", "pool reads lease:request from r0 and more")
    rec = _claim(r, atlas, "pool", "S2", "pool only answers r0 and r1 on lease:reply", kind="dispute")
    assert rec.kind == "dispute"
    with pytest.raises(C.ClaimError, match="nothing to dispute"):
        _claim(r, atlas, "gateway", "S2", "gateway answers r0 on llm:reply", kind="dispute")
    view = K.attach_ledgers(r, atlas, None, [rec], [])
    assert "I dispute pool" in next(j for j in view.jobs if j.job == "S2").text
    assert K.hits(view, "R-matchmaker") == {"part:pool"}  # a dispute changes nothing


def test_a_claim_needs_a_live_part_a_known_job_and_a_reason_of_the_right_size():
    atlas = _body(worker_status="transient")
    r = K.read_capacities(atlas)
    with pytest.raises(C.ClaimError, match="not a live part"):
        _claim(r, atlas, "w0", "S1", "w0 announces itself on worker:announce")
    with pytest.raises(C.ClaimError, match="no job"):
        _claim(r, atlas, "gateway", "S9", "gateway answers r0 on llm:reply")
    with pytest.raises(C.ClaimError, match="because is"):
        _claim(r, atlas, "gateway", "S3", "r0 llm:reply")


def _read(atlas, claims, vsm=None, root=_REPO, written=()):
    r = K.read_capacities(atlas, vsm=vsm)
    return K.attach_ledgers(r, atlas, vsm, list(claims), list(written), recheck=C.recheck(r, atlas, root))


def _job(view, job):
    return next(j for j in view.jobs if j.job == job)


S3_BECAUSE = "gateway sends its answers out on llm:reply to the callers"


def test_a_claim_whose_evidence_checks_holds_its_role_and_is_citable_but_clears_no_gap():
    atlas = _body()
    r = K.read_capacities(atlas)
    claim = _claim(r, atlas, "gateway", "S3", S3_BECAUSE)
    view = _read(atlas, [claim])
    assert [c.status for c in view.claims] == ["evidence-checked"]
    assert [(c.part_id, c.role, c.strength, c.evidence) for c in view.claimed] == [
        ("part:gateway", "S3", "claimed", ["llm:reply (out)"])]
    s3 = _job(view, "S3")
    assert s3.status == "EVIDENCE-CHECKED" and s3.text.startswith("my claim: gateway · ")
    assert K.holders(view, "cap:s3") == ["part:gateway"]
    assert [g.gap_id for g in view.gaps] == [g.gap_id for g in r.gaps]  # S3's gaps are atlas-blind: none cleared
    rec = C.check_capacity(view, atlas, [], "cap:s3", "gateway steers my callers by what it answers",
                           "gateway on llm:reply", dig="t")
    assert rec.cites == ["gateway"]
    text = "\n".join(K.render(view))
    assert "status: " in text and "EVIDENCE-CHECKED = code, not Juniper" in text
    assert "  claimed:" not in text  # a claim that holds prints on its job line, not twice
    assert "my claims: 1 (1 evidence-checked, 0 confirmed by Juniper)" in text


def test_a_claim_stops_holding_when_its_cited_flow_is_gone():
    atlas = _body()
    claim = _claim(K.read_capacities(atlas), atlas, "gateway", "S3", S3_BECAUSE)
    later = atlas.model_copy(update={"flows": [f for f in atlas.flows if "llm:reply" not in f.channels]})
    view = _read(later, [claim])
    assert [c.status for c in view.claims] == ["evidence does not hold"] and view.claimed == []
    assert _job(view, "S3").status == "NO RULE" and K.holders(view, "cap:s3") == []
    assert [c.part_id for c in K.failed_claims(view)] == ["part:gateway"]
    assert K.failed_claims(view)[0].failed[0] == "llm:reply: in no part or channel of my atlas"
    assert "S3" not in [j for j, _ in K.open_jobs(view)]  # it is on the fix-or-withdraw list instead


def test_a_claim_citing_a_repo_file_stops_holding_when_the_file_is_gone(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "control.md").write_text("x")
    atlas = _body()
    claim = _claim(K.read_capacities(atlas), atlas, "gateway", "S3", S3_BECAUSE + "; see notes/control.md",
                   root=tmp_path)
    assert [c.status for c in _read(atlas, [claim], root=tmp_path).claims] == ["evidence-checked"]
    (tmp_path / "notes" / "control.md").unlink()
    view = _read(atlas, [claim], root=tmp_path)
    assert [(c.status, c.failed) for c in view.claims] == [
        ("evidence does not hold", ["notes/control.md: no such file in the repo"])]
    # a file alone is not wiring: code does not check it into a role
    alone = _claim(K.read_capacities(atlas), atlas, "gateway", "S3", "see orion/body_atlas/claims.py for how it steers")
    assert [c.status for c in _read(atlas, [alone]).claims] == ["evidence does not hold"]


def test_a_claim_on_a_part_that_died_holds_nothing():
    atlas = _body()
    claim = _claim(K.read_capacities(atlas), atlas, "gateway", "S3", S3_BECAUSE)
    dead = atlas.model_copy(update={"parts": [p.model_copy(update={"status": "transient"}) if p.name == "gateway"
                                              else p for p in atlas.parts]})
    view = _read(dead, [claim])
    assert [c.status for c in view.claims] == ["part not live now"] and view.claimed == []
    assert _job(view, "S3").status == "NO RULE"


def test_juniper_rejecting_a_claim_takes_its_role_away():
    atlas = _body()
    claim = _claim(K.read_capacities(atlas), atlas, "gateway", "S3", S3_BECAUSE)
    view = _read(atlas, [claim], vsm=_vsm(rejected=[("gateway", "S3")]))
    assert [c.status for c in view.claims] == ["rejected"] and view.claimed == []
    assert _job(view, "S3").status == "NO RULE" and K.holders(view, "cap:s3") == []


def test_juniper_confirming_outranks_a_claim_on_the_same_job():
    atlas = _body()
    r = K.read_capacities(atlas)
    claims = [_claim(r, atlas, "gateway", "S3", S3_BECAUSE),
              _claim(r, atlas, "watcher", "S3", "watcher pokes pool on watch:poke when it looks off")]
    view = _read(atlas, claims, vsm=_vsm(("gateway", "S3")))
    assert {(c.part_id, c.status) for c in view.claims} == {("part:gateway", "confirmed"), ("part:watcher", "evidence-checked")}
    s3 = _job(view, "S3")
    assert s3.status == "CONFIRMED" and s3.text.startswith("confirmed by Juniper: gateway")
    assert s3.text.endswith(" · my claim: watcher")
    assert [c.part_id for c in view.claimed] == ["part:watcher"]  # the confirmed one is not counted twice


def test_a_rule_saying_not_s2_or_a_later_dispute_keeps_a_claim_from_holding():
    atlas = _body(workers_overlap=True)  # no S2 by rule; gateway is a shared server (near miss: not S2)
    r = K.read_capacities(atlas)
    claim = _claim(r, atlas, "gateway", "S2", "gateway answers r0 on llm:reply")
    view = _read(atlas, [claim])
    assert [c.status for c in view.claims] == ["rule says no"] and view.claimed == []
    assert any(g.gap_id == "gap:s2-none" for g in view.gaps) and K.holders(view, "cap:s2") == []
    with pytest.raises(C.ClaimError, match="gap line stands"):
        C.check_capacity(view, atlas, [], "cap:s2", "gateway keeps my workers from fighting", "gateway on llm:reply", dig="t")
    # watcher is the rule's S3* candidate: Orion may claim it, then dispute it; the later word counts
    atlas = _body()
    r = K.read_capacities(atlas)
    claim = _claim(r, atlas, "watcher", "S3*", "watcher pokes pool on watch:poke when it looks off")
    dispute = _claim(r, atlas, "watcher", "S3*", "watcher only listens; it pokes pool on watch:poke", kind="dispute")
    claim = claim.model_copy(update={"at": "2026-11-04T00:00:00+00:00"})
    late = dispute.model_copy(update={"at": "2026-11-04T01:00:00+00:00"})
    view = _read(atlas, [claim, late])
    assert {(c.kind, c.status) for c in view.claims} == {("claim", "disputed"), ("dispute", "rule holds")}
    assert view.claimed == [] and _job(view, "S3*").status == "CANDIDATE"
    early = dispute.model_copy(update={"at": "2026-11-03T00:00:00+00:00"})
    view = _read(atlas, [early, claim])
    assert ("claim", "evidence-checked") in {(c.kind, c.status) for c in view.claims}
    assert _job(view, "S3*").status == "EVIDENCE-CHECKED"


def test_with_claims_hold_off_a_claim_waits_on_juniper(monkeypatch):
    monkeypatch.setattr(K, "CLAIMS_HOLD", False)
    atlas = _body()
    claim = _claim(K.read_capacities(atlas), atlas, "gateway", "S3", S3_BECAUSE)
    view = _read(atlas, [claim])
    assert [c.status for c in view.claims] == ["unconfirmed"] and view.claimed == []
    assert "  unconfirmed: gateway S3" in "\n".join(K.render(view)) and K.pending_claims(view)


def test_a_claim_needs_recheck_to_be_read():
    atlas = _body()
    claim = _claim(K.read_capacities(atlas), atlas, "gateway", "S3", S3_BECAUSE)
    with pytest.raises(ValueError, match="recheck"):
        K.attach_ledgers(K.read_capacities(atlas), atlas, None, [claim], [])


def test_juniper_confirming_gives_the_role_covers_its_gap_and_rejecting_shows():
    atlas = _body(workers_overlap=True)
    vsm = _vsm(("gateway", "S2"), rejected=[("watcher", "S3")])
    r = K.read_capacities(atlas, vsm=vsm)
    assert not any(g.gap_id == "gap:s2-none" for g in r.gaps)
    assert any(g.gap_id == "gap:s3-unseen" for g in r.gaps)  # an atlas-blind gap stays
    s2 = next(j for j in r.jobs if j.job == "S2")
    assert s2.status == "CONFIRMED" and "confirmed by Juniper: gateway" in s2.text
    base = K.read_capacities(atlas)
    claims = [_claim(base, atlas, "gateway", "S2", "gateway answers r0 on llm:reply"),
              _claim(base, atlas, "watcher", "S3", "watcher pokes pool on watch:poke")]
    view = K.attach_ledgers(r, atlas, vsm, claims, [], recheck=C.recheck(r, atlas, _REPO))
    assert {(c.part_id, c.status) for c in view.claims} == {("part:gateway", "confirmed"), ("part:watcher", "rejected")}
    assert "part:gateway" in K.holders(view, "cap:s2")
    text = "\n".join(K.render(view))
    assert "rejected: watcher S3" in text and "my claims: 2 (0 evidence-checked, 1 confirmed by Juniper)" in text


def test_a_confirmation_for_a_part_that_is_not_live_is_a_note_not_a_role():
    atlas = _body()
    r = K.read_capacities(atlas, vsm=_vsm(("ghost", "S4"), ("gateway", "S4")))
    assert [c.part_id for c in r.confirmed] == ["part:gateway"]
    assert any("ghost" in n for n in r.notes)
    r = K.read_capacities(atlas, vsm=_vsm(("gateway", "S4"), rejected=[("gateway", "S4")]))
    assert r.confirmed == [] and any("read as rejected" in n for n in r.notes)


def test_a_capacity_cites_only_holders_and_names_no_other_part():
    atlas = _body()
    r = K.attach_ledgers(K.read_capacities(atlas), atlas, None, [], [])
    rec = C.check_capacity(r, atlas, [], "cap:s2", "pool hands out my workers to whoever asks for one",
                           "R-matchmaker found pool between two crowds", dig="t")
    assert rec.cites == ["pool"] and rec.replaces is None
    with pytest.raises(C.ClaimError, match="names gateway"):
        C.check_capacity(r, atlas, [], "cap:s2", "pool and gateway hand out my workers", "pool is the matchmaker", dig="t")
    with pytest.raises(C.ClaimError, match="no capacity"):
        C.check_capacity(r, atlas, [], "cap:s9", "pool hands out my workers", "pool is the matchmaker", dig="t")
    with pytest.raises(C.ClaimError, match="no part holds S3/S3"):  # the S3* candidate is not a holder
        C.check_capacity(r, atlas, [], "cap:s3", "watcher listens to my whole body", "watcher reads health", dig="t")
    again = C.check_capacity(r, atlas, [rec], "cap:s2", "pool keeps my workers from fighting over GPUs",
                             "pool reads lease:request", dig="t")
    assert again.replaces == rec.at
    view = K.attach_ledgers(K.read_capacities(atlas), atlas, None, [], [rec, again])
    assert next(c for c in view.capacities if c.cap_id == "cap:s2").sentence == again.sentence


def test_a_capacity_whose_holder_lost_the_role_reads_stale():
    atlas = _body()
    rec = C.check_capacity(K.read_capacities(atlas), atlas, [], "cap:s2", "pool hands out my workers to whoever asks",
                           "pool is the matchmaker", dig="t")
    later = _body(workers_overlap=True)
    view = K.attach_ledgers(K.read_capacities(later), later, None, [], [rec])
    cap = next(c for c in view.capacities if c.cap_id == "cap:s2")
    assert cap.status == "stale" and "pool no longer holds S2" in cap.text


def test_cap_s5_is_shared_with_juniper_and_may_cite_her_signoff_files():
    atlas = _body()
    vsm = C.load_confirmed(_REPO)
    r = K.read_capacities(atlas, vsm=vsm)
    assert "shared with Juniper" in next(j for j in r.jobs if j.job == "S5").text
    assert K.holders(r, "cap:s5") == ["config/physiology/families.yaml", "config/physiology/contract.yaml"]
    with pytest.raises(C.ClaimError, match="name Juniper"):
        C.check_capacity(r, atlas, [], "cap:s5", "My policy lives in config/physiology/families.yaml",
                         "tiers are signed off", dig="t")
    rec =C.check_capacity(r, atlas, [], "cap:s5", "Who I am is shared with Juniper: she signs off my tiers",
                           "config/physiology/families.yaml says tiers need her sign-off", dig="t")
    assert rec.cites == ["config/physiology/families.yaml"]


def test_the_checked_in_confirmations_file_reads_and_confirms_nothing_on_her_behalf():
    vsm = C.load_confirmed(_REPO)
    assert vsm.confirmed == [] and vsm.rejected == []
    assert vsm.s5_shared_with is not None and vsm.s5_shared_with.who == "Juniper"
    for f in vsm.s5_shared_with.signoff_files:
        assert (_REPO / f).is_file()


def test_a_malformed_confirmations_file_is_an_error_not_empty(tmp_path):
    (tmp_path / C.CONFIRMED_RELPATH).parent.mkdir(parents=True)
    (tmp_path / C.CONFIRMED_RELPATH).write_text("confirmed:\n  - {part: x, role: S9, date: 2026-10-30}\n")
    with pytest.raises(C.ClaimError, match="does not read"):
        C.load_confirmed(tmp_path)
    assert C.load_confirmed(tmp_path / "none") == K.VsmConfirmedV1()


def test_a_bad_ledger_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "c.jsonl"
    atlas = _body()
    rec = _claim(K.read_capacities(atlas), atlas, "gateway", "S3", "gateway answers r0 on llm:reply")
    C.append(path, rec)
    with path.open("a") as fh:
        fh.write("{not json\n")
    C.append(path, rec)
    got, skipped = C.load_ledger(path, K.CapacityClaimV1)
    assert len(got) == 2 and skipped == 1


def test_the_printout_still_fits_with_six_long_capacities_and_claims():
    atlas = _body()
    long = "x" * C.SENTENCE_MAX_CHARS
    written = [K.CapacityV1(cap_id=c, sentence=long, because="b" * 20, cites=["pool"], at="t", dig="t")
               for c in K.CAP_JOBS]
    claim = _claim(K.read_capacities(atlas), atlas, "gateway", "S3", "gateway answers r0 on llm:reply")
    claims = [claim.model_copy(update={"part_id": f"part:{p}", "role": r, "kind": k, "because": "y" * 400})
              for p, r, k in (("gateway", "S3", "claim"), ("r0", "S4", "claim"), ("r1", "S5", "claim"),
                              ("ghost", "S4", "claim"), ("pool", "S2", "dispute"), ("watcher", "S3*", "dispute"),
                              ("w0", "S1", "dispute"))]
    view = K.attach_ledgers(K.read_capacities(atlas, vsm=_vsm(rejected=[("r1", "S5")])), atlas,
                            _vsm(rejected=[("r1", "S5")]), claims, written, recheck=_ok("llm:reply"))
    view.notes += ["n" * 150] * 3
    text = "\n".join(K.render(view))
    assert len([c for c in view.claims]) == 7 and "I dispute pool" in text and "part not live now" in text
    assert len(text.splitlines()) <= 38 and len(text) <= 3700, (len(text.splitlines()), len(text))


def test_a_sentence_cannot_slip_a_part_in_as_part_colon_name():
    atlas = _body()
    r = K.attach_ledgers(K.read_capacities(atlas), atlas, None, [], [])
    with pytest.raises(C.ClaimError, match="names gateway"):
        C.check_capacity(r, atlas, [], "cap:s2", "part:gateway and pool hand out my workers", "pool is the matchmaker",
                         dig="t")
    rec = _claim(r, atlas, "gateway", "S3", "part:r0 hears back from it on llm:reply")
    assert "r0" in rec.cites


def test_a_folder_a_dot_dir_or_a_dead_part_is_not_evidence():
    atlas = _body(worker_status="transient")
    r = K.read_capacities(atlas)
    with pytest.raises(C.ClaimError, match="cite"):
        _claim(r, atlas, "gateway", "S3", "see the folder orion/body_atlas/tests for how it works")
    with pytest.raises(C.ClaimError, match="cite"):  # w0 is on a flow into pool, but not live
        _claim(r, atlas, "pool", "S3", "pool hears w0 on its announce line a lot")


def test_juniper_confirming_the_candidate_makes_the_job_confirmed():
    atlas = _body()
    r = K.read_capacities(atlas, vsm=_vsm(("watcher", "S3*")))
    assert next(j for j in r.jobs if j.job == "S3*").status == "CONFIRMED"
    r = K.read_capacities(atlas, vsm=_vsm(("gateway", "S3*")))
    assert next(j for j in r.jobs if j.job == "S3*").status == "CANDIDATE"  # a different part: still a candidate
    assert next(j for j in r.jobs if j.job == "S2").status == "BY RULE"


def test_a_held_claim_on_the_rules_candidate_is_listed_once_in_its_organ():
    atlas = _body()
    claim = _claim(K.read_capacities(atlas), atlas, "watcher", "S3*", "watcher pokes pool on watch:poke when it looks off")
    view = _read(atlas, [claim])
    mix = next(m for m in view.organ_mixes if "part:watcher" in m.roles.get("S3*", []))
    assert mix.roles["S3*"] == ["part:watcher"]


def test_s5_claims_never_hold_and_a_checked_claim_covers_only_the_gaps_a_confirmation_would():
    atlas = _body()
    atlas.parts.append(PartV1(part_id="part:pg:self", name="pg:self", kind="store", status="live", declared=True))
    atlas.flows.append(FlowV1(source="part:gateway", target="part:pg:self", relation="writes", weight=1.0,
                              evidence=["declared_code"]))
    claims = [_claim(K.read_capacities(atlas), atlas, "pg:self", "S5", "gateway writes who I am into it, every turn")]
    claims.append(_claim(K.read_capacities(atlas), atlas, "watcher", "S5", "watcher pokes pool on watch:poke to keep me me"))
    view = _read(atlas, claims)
    # 2026-11-04 (evidence check): S5 is Juniper's (decision 3); no S5 claim holds or covers gap:s5-unread
    assert {c.status for c in view.claims} == {"unconfirmed"} and view.claimed == []
    assert any(g.gap_id == "gap:s5-unread" for g in view.gaps)
    # a claim that passes covers its job's "no part is shown" gap, as a confirmation would
    audit = _claim(K.read_capacities(atlas), atlas, "watcher", "S3*", "watcher pokes pool on watch:poke when it looks off")
    view = _read(atlas, [audit])
    assert [c.status for c in view.claims] == ["evidence-checked"]
    assert not any(g.gap_id == "gap:s3star-unconfirmed" for g in view.gaps)
    assert any(g.gap_id == "gap:s3-unseen" for g in view.gaps)  # an atlas-blind gap stays


def test_a_claim_a_rule_gives_since_prints_as_rule_gives_it_and_evidence_gone_skips_held_jobs():
    atlas = _body()
    claim = _claim(K.read_capacities(atlas), atlas, "w0", "S1", "w0 announces itself to pool on worker:announce")
    r = K.read_capacities(atlas)  # later, a rule gives w0 S1 (added by hand, as a rule result would be)
    r.roles.append(K.RoleEvidenceV1(part_id="part:w0", role="S1", rule="R-test", strength="rule"))
    view = K.attach_ledgers(r, atlas, None, [claim], [], recheck=C.recheck(r, atlas, _REPO))
    assert [c.status for c in view.claims] == ["rule gives it"] and view.claimed == []
    assert "rule gives it: w0 S1" in "\n".join(K.render(view))
    gone = K.attach_ledgers(r, atlas, None, [claim], [], recheck=_gone)
    assert [c.status for c in gone.claims] == ["evidence does not hold"]
    assert K.failed_claims(gone) == []  # S1 is held by rule now: nothing to re-claim
