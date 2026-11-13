"""The market's consent gate (attention patch 3d).

Spec: docs/superpowers/specs/2026-11-07-attention-patch3-market-decides-design.md section 3 and acceptance
check 4. The gate is pure (orion/physiology/consent.py); Choice asks it first for a want with a consent
request; the binding rule never admits a consent deferral.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from orion.physiology.binding import bind_verdict, consent_block
from orion.physiology.choice import ReservationBook, decide
from orion.physiology.consent import consent_verdict, load_consent, parse_consent
from orion.physiology.tests.consent_fixture import LIVE_WANT_KEYS, live_delta, live_wants
from orion.physiology.tests.test_choice import FAMS, NOW, SELF, body, ctx_for, history, lane, want
from orion.schemas.physiology_want import ConsentRequestV1, WantDecisionV1, WantV1

REPO = Path(__file__).resolve().parents[3]
RULES, RULES_ERR = load_consent(REPO / "config/physiology/consent.yaml")
STANDING = {"dream-compaction-downscale": {"granted_by": "juniper", "granted_on": "2026-11-11",
                                           "pr": "test", "note": "fixture only"}}


def rules_with(standing=None, extra_rules=None):
    import yaml

    doc = yaml.safe_load((REPO / "config/physiology/consent.yaml").read_text())
    if standing is not None:
        doc["standing"] = standing
    if extra_rules:
        doc["rules"].update(extra_rules)
    return parse_consent(doc)


# Juniper merged `standing.dream-compaction-downscale` (2026-11-13); the needs_consent path is still the
# state of any host whose consent.yaml lacks it, so it is tested against the shipped rules minus standing.
NO_STANDING = rules_with({})


def cwant(verb="compact", *, rev="snapshot_restorable", attests=("snapshot_dir_writable", "downscale_only"),
          family="dream.compaction", i=0, **req_kw):
    wid = f"cw{i}"
    req = ConsentRequestV1(want_id=req_kw.pop("want_id", wid), family=req_kw.pop("req_family", family),
                           verb=req_kw.pop("req_verb", verb), object_ref="compaction-delta:x",
                           payload_digest="sha256:abc", reversibility=rev, attests=list(attests))
    return WantV1(want_id=wid, family=family, producer="orion-dream.compaction_applier.consent", verb=verb, tier=2,
                  consent_request=req)


def verdict_of(w, rules=RULES, now=NOW):
    return consent_verdict(w, w.consent_request, rules, now)


# --- the shipped config --------------------------------------------------------------------------------

def test_the_shipped_consent_config_loads_and_names_the_dream_rules():
    assert RULES_ERR is None, RULES_ERR
    assert RULES.get("dream.compaction", "compact").rule_id == "dream.compaction.downscale_only"
    assert RULES.get("dream.compaction", "compact").requires == ("snapshot_dir_writable", "downscale_only")
    assert RULES.get("dream.compaction", "prune").reversibility == "irreversible"
    # Juniper's standing consent for downscale only (2026-11-13), and the rule that names it.
    granted = RULES.standing["dream-compaction-downscale"]
    assert granted.granted_by == "juniper" and granted.expires is None
    assert RULES.get("dream.compaction", "compact").consent == "standing:dream-compaction-downscale"
    assert set(RULES.standing) == {"dream-compaction-downscale"}


def test_the_shipped_config_allows_downscale_only_and_still_refuses_prune():
    """Regression: the standing consent opens downscale and nothing else. Prune stays refused."""
    assert verdict_of(cwant()).verdict == "allow"
    v = verdict_of(cwant("prune", rev="irreversible"))
    assert v.verdict == "refuse" and v.rule_id == "dream.compaction.prune" and v.reason.startswith("refused_by_rule")
    assert RULES.get("dream.compaction", "prune").consent == "refused"


def test_every_consent_rule_names_a_family_the_market_knows():
    for (family, _verb), rule in RULES.rules.items():
        cfg = FAMS.get(family)
        assert cfg is not None and cfg.bids, f"{rule.rule_id}: {family} must be a bidding family in families.yaml"


def test_dream_compaction_is_tier2_with_no_lanes_and_no_receptors():
    cfg = FAMS.get("dream.compaction")
    assert cfg.tier == 2 and cfg.lanes == () and cfg.receptors == ()


# --- acceptance check 4: the fixture ---------------------------------------------------------------------

def test_irreversible_is_refused_even_with_every_attest():
    v = verdict_of(cwant("prune", rev="irreversible"), rules=rules_with(STANDING))
    assert v.verdict == "refuse" and v.rule_id == "dream.compaction.prune" and v.reason.startswith("refused_by_rule")


def test_an_unknown_non_reversible_act_is_refused_and_an_unknown_reversible_one_allowed():
    for rev in ("snapshot_restorable", "irreversible"):
        v = verdict_of(cwant("compact", rev=rev, family="thought.painting"))
        assert v.verdict == "refuse" and v.rule_id is None and v.reason.startswith("no_rule_not_reversible")
    v = verdict_of(cwant("make", rev="reversible", family="thought.painting"))
    assert v.verdict == "allow" and v.reason.startswith("no_rule_reversible")


@pytest.mark.parametrize("attests,missing", [
    (("snapshot_dir_writable",), "downscale_only"),
    (("downscale_only",), "snapshot_dir_writable"),
    ((), "snapshot_dir_writable,downscale_only"),
])
def test_a_missing_attest_is_refused(attests, missing):
    v = verdict_of(cwant(attests=attests), rules=rules_with(STANDING))
    assert v.verdict == "refuse" and v.reason == f"missing_attest:{missing}"
    assert v.rule_id == "dream.compaction.downscale_only"


def test_downscale_only_with_both_attests_and_a_standing_consent_is_allowed():
    v = verdict_of(cwant(), rules=rules_with(STANDING))
    assert v.verdict == "allow" and v.rule_id == "dream.compaction.downscale_only"
    assert v.consent_ref == "standing:dream-compaction-downscale"


def test_without_the_standing_consent_it_needs_consent():
    v = verdict_of(cwant(), rules=NO_STANDING)
    assert v.verdict == "needs_consent" and v.reason.startswith("standing_consent_missing")
    assert v.consent_ref == "standing:dream-compaction-downscale"


def test_an_expired_standing_consent_needs_consent_again():
    rules = rules_with({"dream-compaction-downscale": {**STANDING["dream-compaction-downscale"], "expires": "2026-10-28"}})
    assert verdict_of(cwant(), rules=rules, now=NOW).verdict == "allow"                     # 10-27
    assert verdict_of(cwant(), rules=rules, now=NOW + timedelta(days=1)).verdict == "needs_consent"


def test_no_rules_refuses_everything_that_asks():
    assert verdict_of(cwant(), rules=None).reason.startswith("rules_unavailable")
    assert verdict_of(cwant("make", rev="reversible", family="thought.painting"), rules=None).verdict == "refuse"


def test_a_request_for_another_want_or_verb_is_refused():
    assert verdict_of(cwant(want_id="someone-else"), rules=rules_with(STANDING)).reason.startswith("request_mismatch")
    assert verdict_of(cwant(req_verb="prune"), rules=rules_with(STANDING)).reason.startswith("request_mismatch")


def test_a_request_claiming_a_softer_class_than_its_rule_is_refused():
    v = verdict_of(cwant("prune", rev="snapshot_restorable"), rules=rules_with(STANDING))
    assert v.verdict == "refuse" and v.reason.startswith("reversibility_mismatch")


def test_the_verdict_is_deterministic():
    w = cwant()
    assert {verdict_of(w).model_dump_json() for _ in range(5)} == {verdict_of(w).model_dump_json()}


# --- config parsing fails closed -------------------------------------------------------------------------

@pytest.mark.parametrize("rule,msg", [
    ({"family": "f", "verb": "v", "reversibility": "irreversible", "consent": "standing:x"}, "irreversible"),
    ({"family": "f", "verb": "v", "reversibility": "snapshot_restorable", "consent": "none"}, "only a reversible"),
    ({"family": "f", "verb": "v", "reversibility": "maybe", "consent": "refused"}, "reversibility"),
    ({"family": "f", "verb": "v", "reversibility": "reversible", "consent": "standing:"}, "consent must be"),
    ({"family": "dream.compaction", "verb": "compact", "reversibility": "reversible", "consent": "none"}, "already has"),
])
def test_a_rule_that_could_fail_open_is_a_config_error(rule, msg):
    with pytest.raises(ValueError, match=msg):
        rules_with(extra_rules={"bad.rule": rule})


def test_a_standing_consent_must_say_who_agreed_and_when():
    with pytest.raises(ValueError, match="granted_by"):
        rules_with({"dream-compaction-downscale": {"granted_on": "2026-11-11"}})


def test_a_malformed_or_missing_file_is_a_reason_not_a_crash(tmp_path):
    p = tmp_path / "consent.yaml"
    p.write_text("version: 2\n")
    assert load_consent(p)[0] is None and load_consent(p)[1].startswith("malformed")
    assert load_consent(tmp_path / "nope.yaml") == (None, "missing:nope.yaml")


# --- Choice: the gate runs first; no request is unchanged ------------------------------------------------

def test_with_no_consent_request_every_decision_is_byte_identical():
    """Acceptance check 4: live want shapes (fixture), decided with and without the consent rules."""
    st = body(lane("agent"), lane("agent-gpu2", slots=2), lane("metacog"), lane("diffusion"), lane("chat"),
              lane("fast"))
    acts = [a for w in live_wants() for a in history(w.family, slot={"agent": 600.0})]
    for w in live_wants():
        assert set(json.loads(w.model_dump_json())) - {"event_ref", "salience"} == LIVE_WANT_KEYS
        a = decide(w, st, ctx_for(acts, st), ReservationBook(), NOW)
        b = decide(w, st, ctx_for(acts, st), ReservationBook(), NOW, consent_rules=RULES)
        assert a.model_dump_json() == b.model_dump_json(), w.family
        assert "consent" not in a.model_dump_json() and a.consent is None


def test_a_consent_deferral_is_decided_before_prices_and_reserves_nothing():
    st = body(lane("agent"))
    book = ReservationBook()
    d = decide(cwant(), st, ctx_for(SELF, st), book, NOW, consent_rules=NO_STANDING)
    assert d.outcome == "deferred" and d.reason == "consent" and d.binding_constraint == "consent"
    assert d.consent.verdict == "needs_consent" and d.consent.rule_id == "dream.compaction.downscale_only"
    assert d.net is None and d.reserved == {} and book.snapshot(NOW) == []
    assert d.tier == 2 and "never admitted" in d.why


def test_the_gate_runs_even_with_no_body_state_and_an_unknown_family():
    """A stale body or unknown family is "no answer", which binding producers admit. The gate must win."""
    d = decide(cwant(), None, ctx_for(SELF, body()), ReservationBook(), NOW, consent_rules=NO_STANDING)
    assert d.reason == "consent"
    d = decide(cwant("compact", family="not.a.family"), body(), ctx_for(SELF, body()), ReservationBook(), NOW,
               consent_rules=NO_STANDING)
    assert d.reason == "consent" and d.consent.reason.startswith("no_rule_not_reversible")


def test_an_allowed_want_is_priced_as_any_want_and_carries_its_verdict():
    st = body(lane("agent"))
    d = decide(cwant(), st, ctx_for(SELF, st), ReservationBook(), NOW, consent_rules=rules_with(STANDING))
    assert d.reason != "consent" and d.consent is not None and d.consent.verdict == "allow"
    back = WantDecisionV1.model_validate_json(d.model_dump_json())
    assert back.consent == d.consent and back.want.consent_request == d.want.consent_request


def test_an_old_consumer_with_forbid_still_parses_a_decision_without_consent():
    """want_ledger's writer and other WantDecisionV1 readers rebuild first; until then a decision with no
    consent request must not carry the new keys."""
    old_fields = set(WantDecisionV1.model_fields) - {"consent"}

    class OldDecision(BaseModel):
        model_config = ConfigDict(extra="forbid")
        __annotations__ = {k: object for k in old_fields}

    st = body(lane("agent"))
    d = decide(want(), st, ctx_for(SELF, st), ReservationBook(), NOW, consent_rules=RULES)
    OldDecision.model_validate(json.loads(d.model_dump_json()))


# --- binding: a consent deferral is never admitted ---------------------------------------------------------

def test_a_consent_deferral_is_never_admitted_not_at_the_window_end_not_with_no_end():
    d = decide(cwant(), body(lane("agent")), ctx_for(SELF, body(lane("agent"))), ReservationBook(), NOW,
               consent_rules=NO_STANDING)
    for end in (NOW + timedelta(hours=1), NOW, NOW - timedelta(hours=1), None):
        v = bind_verdict(d, NOW, end)
        assert v.admit is False and v.why == "deferred:consent:needs_consent", end
        assert v.retry_at == NOW + timedelta(hours=1)
    assert v.detail["consent_rule"] == "dream.compaction.downscale_only"


def test_a_price_deferral_without_consent_still_goes_at_the_window_end():
    """Unchanged rule for every other deferral (regression guard)."""
    st = body(lane("agent", p=50.0, state="binding"))
    d = decide(want(), st, ctx_for(SELF, st), ReservationBook(), NOW, consent_rules=RULES)
    assert d.outcome == "deferred" and d.reason == "price"
    assert bind_verdict(d, NOW, NOW).why == "window_end"


def test_a_request_answered_without_a_verdict_is_blocked():
    """A market that dropped the gate (or a hand-built decision) fails closed."""
    st = body(lane("agent"))
    d = decide(cwant(), st, ctx_for(SELF, st), ReservationBook(), NOW, consent_rules=rules_with(STANDING))
    stripped = d.model_copy(update={"consent": None})
    assert consent_block(stripped) == "no_verdict"
    assert bind_verdict(stripped, NOW, NOW).admit is False
    assert consent_block(d) is None


def test_the_live_delta_shape_maps_to_the_compact_rule():
    """Live deltas are all consolidate cards (no downscale, no prune): the compact rule, downscale only."""
    delta = live_delta()
    assert not delta.downscale and not delta.prune and len(delta.consolidate) == 3


# --- review findings (PR #3185) ------------------------------------------------------------------------

def _uneven_use_history():
    """Live-shaped evidence: two Tier 2 families with linked, window-closed downstream uses at very
    different rates (enough acts to leave the prior), so the tier's weight normalization really moves."""
    from orion.physiology.weights import UseRecord

    uses = []
    for fam, rate in (("durable.curiosity_investigate", 0.9), ("thought.painting", 0.2)):
        for i in range(40):
            uses.append(UseRecord(act_id=f"{fam}{i}", family=fam, started_at=NOW - timedelta(days=1, hours=i),
                                  link_status="linked", used=(i % 10) < rate * 10, window_closed=True))
    return uses


def _without(fams, family):
    from orion.physiology.families import FamiliesConfig

    return FamiliesConfig(families={f: c for f, c in fams.families.items() if f != family})


def test_adding_the_consent_only_family_moves_no_other_familys_weight():
    """Review finding 1: a new Tier 2 family with no acts of its own sat at its prior in the tier's weight
    normalization and lowered every other Tier 2 weight. ``weighted: false`` keeps it out of the pool."""
    import dataclasses

    from orion.physiology.families import FamiliesConfig
    from orion.physiology.weights import weights_now

    uses = _uneven_use_history()
    with_it = weights_now(FAMS, [], uses, NOW)
    without = weights_now(_without(FAMS, "dream.compaction"), [], uses, NOW)
    assert "dream.compaction" not in with_it
    assert {f: w.weight for f, w in with_it.items()} == {f: w.weight for f, w in without.items()}
    # The test is not vacuous: in the pool, the family does move the others.
    pooled = FamiliesConfig(families={**FAMS.families, "dream.compaction": dataclasses.replace(
        FAMS.get("dream.compaction"), weighted=True)})
    moved = weights_now(pooled, [], uses, NOW)
    assert moved["durable.curiosity_investigate"].weight != without["durable.curiosity_investigate"].weight


def test_live_want_shapes_decide_identically_with_and_without_the_new_family():
    """Acceptance check 4 against the families.yaml change itself, with real (non-empty) weights."""
    from orion.physiology.choice import build_context
    from orion.physiology.weights import weights_now

    st = body(lane("agent"), lane("agent-gpu2", slots=2), lane("metacog"), lane("diffusion"), lane("chat"),
              lane("fast"))
    acts = [a for w in live_wants() for a in history(w.family, slot={"agent": 600.0})]
    uses = _uneven_use_history()
    old = _without(FAMS, "dream.compaction")
    for w in live_wants():
        a = decide(w, st, build_context(old, acts, uses, st, NOW, weights=weights_now(old, acts, uses, NOW)),
                   ReservationBook(), NOW)
        b = decide(w, st, build_context(FAMS, acts, uses, st, NOW, weights=weights_now(FAMS, acts, uses, NOW)),
                   ReservationBook(), NOW, consent_rules=RULES)
        assert a.model_dump_json() == b.model_dump_json(), w.family


def test_a_ruled_act_without_a_request_is_refused():
    """Review finding 3: the gate's guarantee does not rest on the producer remembering to ask."""
    st = body(lane("agent"))
    for verb in ("compact", "prune"):
        w = WantV1(want_id=f"bare-{verb}", family="dream.compaction", producer="someone", verb=verb, tier=2)
        d = decide(w, st, ctx_for(SELF, st), ReservationBook(), NOW, consent_rules=rules_with(STANDING))
        assert d.reason == "consent" and d.consent.verdict == "refuse"
        assert d.consent.reason.startswith("request_missing")
        assert bind_verdict(d, NOW, NOW).admit is False


def test_a_standing_consent_no_rule_reads_is_a_config_error():
    """Review finding 7: a typo in Juniper's consent PR must fail its CI, not read as 'missing' forever."""
    with pytest.raises(ValueError, match="not used by any rule"):
        rules_with({"dream-compaction-downscal": {"granted_by": "juniper", "granted_on": "2026-11-11"}})


def test_the_enum_tuples_match_the_schema_literals():
    from typing import get_args

    from orion.schemas.physiology_want import (CONSENT_VERDICTS, DECISION_REASONS, REVERSIBILITY, ConsentRequestV1,
                                               ConsentVerdictV1)

    assert get_args(ConsentRequestV1.model_fields["reversibility"].annotation) == REVERSIBILITY
    assert get_args(ConsentVerdictV1.model_fields["verdict"].annotation) == CONSENT_VERDICTS
    assert set(get_args(WantDecisionV1.model_fields["reason"].annotation)) == set(DECISION_REASONS)
