"""Code re-checks Orion's capacity claims against the live atlas on every read (rung 4 decision
note, 2026-11-04): a claim counts as a role, EVIDENCE-CHECKED and never Juniper's, only while
every reference it cites is on a live flow of the claimed part and the flows fit the job."""
from __future__ import annotations

from pathlib import Path

from orion.body_atlas import capacities as K
from orion.body_atlas import claims as C
from orion.body_atlas.model import FlowV1
from orion.body_atlas.tests.test_capacities import HEALTH, _atlas

_REPO = Path(__file__).resolve().parents[3]


def _flow(s: str, t: str, rel: str) -> FlowV1:
    return FlowV1(source=f"part:{s}", target=f"part:{t}", weight=1.0, evidence=["declared_code"], relation=rel)


def _wired(*, cmd_live: bool = True, forecast: str = "dormant"):
    """orion-ctl takes orion:ctl:cmd from two parts and acts on orion-lanes; orion-ledger writes two
    tables (one dormant); orion-strain writes a key four parts in two organs read."""
    svc = ["orion-ctl", "orion-lanes", "orion-ask-a", "orion-ask-b", "orion-ledger", "orion-strain",
           "orion-r1", "orion-r2", "orion-r3", "orion-r4", "orion-other"]
    parts = {n: ("service", "live") for n in svc}
    parts.update({"pg:costs": ("store", "live"), "pg:forecast": ("store", forecast),
                  "redis:orion:strain:latest": ("store", "live")})
    sig = [("orion-ask-a", "orion-ctl", "orion:ctl:cmd"), ("orion-ask-b", "orion-ctl", "orion:ctl:cmd"),
           ("orion-ctl", "orion-lanes", "orion:lanes:set")]
    if not cmd_live:
        sig = [x for x in sig if x[2] != "orion:ctl:cmd"]
    sig += [("orion-other", "orion-r1", "orion:other:chat")]
    sig += [(n, "orion-r1", HEALTH) for n in svc if n != "orion-r1"]
    other = [_flow("orion-ledger", "pg:costs", "writes"), _flow("orion-ledger", "pg:forecast", "writes"),
             _flow("orion-strain", "redis:orion:strain:latest", "writes")]
    other += [_flow(r, "redis:orion:strain:latest", "reads") for r in ("orion-r1", "orion-r2", "orion-r3", "orion-r4")]
    organs = {"a": svc[:8] + ["pg:costs", "pg:forecast"], "b": svc[8:] + ["redis:orion:strain:latest"]}
    return _atlas(parts, sig, organs=organs, other=other)


def _c(part: str, role: str, because: str, at: str = "2026-11-04T05:00:00+00:00") -> K.CapacityClaimV1:
    return K.CapacityClaimV1(part_id=f"part:{part}", role=role, kind="claim", because=because, cites=[], at=at, dig="t")


def _check(atlas, claim):
    return C.check_claims(atlas, [claim], _REPO)[0]


S3_OK = "orion-ctl takes orion:ctl:cmd from orion-ask-a and sets orion-lanes on orion:lanes:set"


def test_valid_wiring_passes_and_names_its_direction():
    chk = _check(_wired(), _c("orion-ctl", "S3", S3_OK))
    assert chk.holds and chk.failed == []
    assert "orion:lanes:set (out)" in chk.held and "orion-ask-a (in)" in chk.held


def test_a_cited_channel_absent_from_the_atlas_fails_and_says_which():
    chk = _check(_wired(), _c("orion-ctl", "S3", S3_OK + " and orion:ctl:ghost"))
    assert not chk.holds and chk.failed == ["orion:ctl:ghost: in no part or channel of my atlas"]


def test_wiring_that_disappears_drops_the_claim_back_on_the_next_read():
    claim = _c("orion-ctl", "S3", S3_OK)
    assert _check(_wired(), claim).holds
    gone = _check(_wired(cmd_live=False), claim)
    assert not gone.holds and "orion:ctl:cmd: in no part or channel of my atlas" in gone.failed
    r = K.read_capacities(_wired(cmd_live=False), checks=[gone])
    assert r.checked == [] and K.holders(r, "cap:s3") == []


def test_evidence_on_a_different_part_fails():
    # orion:other:chat is real wiring, but between orion-other and orion-r1, not orion-ctl
    chk = _check(_wired(), _c("orion-ctl", "S3", S3_OK + "; see orion:other:chat and orion-r2"))
    assert not chk.holds
    assert chk.failed == ["orion-r2: not on a live flow of orion-ctl", "orion:other:chat: not on a live flow of orion-ctl"]


def test_a_dormant_store_and_the_census_are_not_evidence():
    chk = _check(_wired(), _c("orion-ledger", "metabolism", "orion-ledger writes pg:costs and pg:forecast, "
                              "and sends orion:system:health"))
    assert not chk.holds
    assert "pg:forecast: dormant, not live" in chk.failed and "orion:system:health: the census, which every part sends" in chk.failed
    assert _check(_wired(forecast="live"), _c("orion-ledger", "metabolism", "orion-ledger writes pg:costs and pg:forecast")).holds


def test_s5_never_passes_even_with_good_wiring():
    chk = _check(_wired(), _c("orion-ctl", "S5", S3_OK))
    assert not chk.holds and chk.failed == ["shape: S5 is shared with Juniper; code cannot check it into a role"]
    forced = chk.model_copy(update={"holds": True, "failed": []})  # even a forged pass is not a role
    r = K.read_capacities(_wired(), checks=[forced])
    assert r.checked == [] and next(j for j in r.jobs if j.job == "S5").status == "NO RULE"
    view = K.attach_ledgers(r, _wired(), None, [_c("orion-ctl", "S5", S3_OK)], [], [forced])
    assert [c.status for c in view.claims] == ["unconfirmed"] and K.pending_claims(view)


def test_shape_misses_fail_each_job_by_its_own_rule():
    inbound_only = "orion-ctl takes orion:ctl:cmd from orion-ask-a and orion-ask-b"
    assert _check(_wired(), _c("orion-ctl", "S3", inbound_only)).failed == [
        "shape: S3 needs a cited flow out to another part"]
    assert _check(_wired(), _c("orion-ctl", "S2", inbound_only)).failed == ["shape: S2 needs a cited flow in and one out"]
    assert _check(_wired(), _c("orion-ctl", "S2", S3_OK)).holds
    # R-census-reader finds orion-r1 (reads health from all, answers none); orion-ctl it does not
    assert "shape: S3* needs R-census-reader" in _check(_wired(), _c("orion-ctl", "S3*", S3_OK)).failed[0]
    # algedonic: a key R-broadcast-key finds, written by the part
    assert _check(_wired(), _c("orion-strain", "algedonic", "orion-strain writes redis:orion:strain:latest")).holds
    assert "shape: algedonic needs" in _check(_wired(), _c("orion-ctl", "algedonic", S3_OK)).failed[0]
    assert "shape: algedonic needs" in _check(_wired(), _c("orion-r1", "algedonic", "orion-r1 reads redis:orion:strain:latest")).failed[0]


def test_files_must_exist_and_cannot_carry_a_claim_alone():
    ok = _check(_wired(), _c("orion-ctl", "S3", S3_OK + " (orion/body_atlas/claims.py)"))
    assert ok.holds and "orion/body_atlas/claims.py (file)" in ok.held
    bad = _check(_wired(), _c("orion-ctl", "S3", S3_OK + " (orion/nope/missing.py), and/or more"))
    assert bad.failed == ["orion/nope/missing.py: no such file in the repo"]  # "and/or" is prose, not a path
    alone = _check(_wired(), _c("orion-ctl", "S3", "orion-ctl lives in orion/body_atlas/claims.py"))
    assert alone.failed == ["shape: it cites no wiring of the part (files alone are not checked as a role)"]


def test_a_checked_claim_holds_the_job_prints_as_code_not_juniper_and_covers_a_none_gap():
    atlas = _wired()
    claims = [_c("orion-strain", "algedonic", "orion-strain writes redis:orion:strain:latest")]
    checks = C.check_claims(atlas, claims, _REPO)
    base = K.read_capacities(atlas)
    assert any(g.gap_id == "gap:algedonic-none" for g in base.gaps)
    r = K.attach_ledgers(K.read_capacities(atlas, checks=checks), atlas, None, claims, [], checks)
    line = next(j for j in r.jobs if j.job == "algedonic")
    assert line.status == "EVIDENCE-CHECKED" and "my claim, checked by code (not Juniper): orion-strain" in line.text
    assert "Juniper" not in line.text.replace("(not Juniper)", "")
    assert not any(g.gap_id == "gap:algedonic-none" for g in r.gaps)
    assert K.holders(r, "cap:s5") == ["part:orion-strain"]
    assert [c.status for c in r.claims] == ["evidence-checked"] and K.vetoable_claims(r)
    assert K.open_jobs(r) and "algedonic" not in [j for j, _ in K.open_jobs(r)]


def test_juniper_reject_overrides_a_passing_check_and_her_confirm_outranks_it():
    atlas = _wired()
    claims = [_c("orion-ctl", "S3", S3_OK)]
    checks = C.check_claims(atlas, claims, _REPO)
    assert checks[0].holds
    no = K.VsmConfirmedV1(rejected=[K.ConfirmationV1(part="orion-ctl", role="S3", date="2026-11-04")])
    r = K.attach_ledgers(K.read_capacities(atlas, vsm=no, checks=checks), atlas, no, claims, [], checks)
    assert r.checked == [] and [c.status for c in r.claims] == ["rejected"] and K.holders(r, "cap:s3") == []
    assert next(j for j in r.jobs if j.job == "S3").status == "NO RULE"
    yes = K.VsmConfirmedV1(confirmed=[K.ConfirmationV1(part="orion-ctl", role="S3", date="2026-11-04")])
    r = K.attach_ledgers(K.read_capacities(atlas, vsm=yes, checks=checks), atlas, yes, claims, [], checks)
    assert r.checked == [] and [c.status for c in r.claims] == ["confirmed"]
    assert next(j for j in r.jobs if j.job == "S3").status == "CONFIRMED"


def test_a_withdrawn_claim_is_not_checked_and_a_later_claim_stands_again():
    atlas = _wired()
    claim = _c("orion-ctl", "S3", S3_OK, at="2026-11-04T05:00:00+00:00")
    back = C.withdraw_claim(atlas, [claim], "orion-ctl", "s3", dig="t")
    assert C.check_claims(atlas, [claim, back], _REPO) == []
    again = _c("orion-ctl", "S3", S3_OK, at="2026-11-05T05:00:00+00:00")
    assert len(C.check_claims(atlas, [claim, back, again], _REPO)) == 1


# --- review findings, 2026-11-04 ------------------------------------------------------------

def test_a_write_time_file_cite_without_an_extension_still_reads_as_a_file():
    claim = _c("orion-ctl", "S3", S3_OK + " (services/orion-hub/Dockerfile)").model_copy(
        update={"cites": ["orion-ask-a", "services/orion-hub/Dockerfile"]})
    chk = _check(_wired(), claim)
    assert chk.holds, chk.failed
    assert "services/orion-hub/Dockerfile (file)" in chk.held


def test_a_channel_to_a_dormant_part_is_not_live_wiring():
    atlas = _wired()
    next(p for p in atlas.parts if p.name == "orion-lanes").status = "dormant"
    chk = _check(atlas, _c("orion-ctl", "S3", S3_OK))
    assert not chk.holds and "orion:lanes:set: not on a live flow of orion-ctl" in chk.failed


def test_s3_that_only_writes_a_table_does_not_act_on_a_part():
    chk = _check(_wired(), _c("orion-ledger", "S3", "orion-ledger writes pg:costs every hour"))
    assert chk.failed == ["shape: S3 needs a cited flow out to another part"]


def test_s3star_on_the_census_reader_needs_a_flow_out():
    # orion-r1 is R-census-reader's hit; it only hears orion:other:chat
    chk = _check(_wired(), _c("orion-r1", "S3*", "orion-r1 hears orion-other on orion:other:chat"))
    assert chk.failed == ["shape: S3* needs a cited flow out to another part (it reports back)"]


def test_a_passing_claim_on_a_job_a_rule_now_holds_is_not_a_vetoable_role():
    atlas = _wired()
    claim = _c("orion-ctl", "S3", S3_OK)
    checks = C.check_claims(atlas, [claim], _REPO)
    reading = K.read_capacities(atlas, checks=checks)
    reading.roles.append(K.RoleEvidenceV1(part_id="part:orion-ctl", role="S3", rule="R-test", strength="rule"))
    reading.checked = []  # what read_capacities does when a rule holds the job (``taken``)
    view = K.attach_ledgers(reading, atlas, None, [claim], [], checks)
    assert [c.status for c in view.claims] == ["rule holds"] and K.vetoable_claims(view) == []


def test_a_withdraw_written_before_its_claim_in_the_file_is_read_by_time():
    atlas = _wired()
    claim = _c("orion-ctl", "S3", S3_OK, at="2026-11-04T05:00:00+00:00")
    back = C.withdraw_claim(atlas, [claim], "orion-ctl", "S3", dig="t").model_copy(
        update={"at": "2026-11-04T06:00:00+00:00"})
    assert C.check_claims(atlas, [back, claim], _REPO) == []  # out of file order, still withdrawn


def test_a_url_in_prose_is_not_an_unknown_id():
    atlas = _wired()
    atlas.parts.append(atlas.parts[0].model_copy(update={"part_id": "part:http:anthropic", "name": "http:anthropic",
                                                           "kind": "endpoint"}))
    chk = _check(atlas, _c("orion-ctl", "S3", S3_OK + ", see http://example.org/x"))
    assert chk.holds, chk.failed
