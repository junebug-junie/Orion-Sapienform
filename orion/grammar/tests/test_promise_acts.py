"""Intention ledger D2: which promise acts one delivered turn opens and closes (pure).

Pins amendments 4 (an unaccepted offer opens nothing), 5 (a keep needs its delivering quote),
6 (the act points at the promising turn's record and holds no quote), 10 (the close says which kind of
evidence closed it) and 11 (items without force/done_when are skipped and counted; one faculty bucket).
"""
from __future__ import annotations

from orion.grammar.acts import stable_act_id
from orion.grammar.promise_acts import (
    PROMISE_ACTOR,
    SHADOW_ACT_SIGNALS,
    VOID_NOTE,
    is_shadow_signal,
    kept_evidence_kind,
    parse_promise_ref,
    plan_promise_events,
    promise_act_id,
    promise_close_kind,
    promise_ref,
)
from orion.schemas.harness_finalize import FinalizeReflectionV1, PromiseKeptV1, PromiseV1
from orion.self_monitor.adapters import act_faculty

DONE = "a later reply gives Juniper the digested deep read"


def _p(quote: str, *, offer: bool = False, force: str | None = "commits", done_when: str | None = DONE) -> PromiseV1:
    return PromiseV1(quote=quote, offer=offer, doer="orion", force=force, act="bring it back", done_when=done_when)


def _reflection(**kw) -> FinalizeReflectionV1:
    base = dict(correlation_id="c2", thought_event_id="t", substrate_appraisal_id="s", draft_hash="h",
                imperative="i", tone="t", strain_refs=[], alignment_verdict="aligned", alignment_notes=[],
                strain_unresolved=False)
    base.update(kw)
    return FinalizeReflectionV1(**base)


def test_firm_promise_opens_one_act_pointing_at_the_turn_record_and_offer_waits() -> None:
    plan = plan_promise_events(correlation_id="c2", session_id="s", reflection=_reflection(promises_made=[
        _p("I'll hold the deep read, then pull it back."), _p("If you want, I'll re-read it later.", offer=True)]))
    assert plan.counts()["opened"] == 1 and plan.offers_not_opened == 1
    (event,) = plan.events
    act = event.act
    assert event.event_kind == "act_opened" and act.actor == PROMISE_ACTOR and act.verb == "say"
    assert act.test_before == "harness_turn_trace:c2#promises_made[0]"
    assert act.expect.signal_id == "promise_kept" and act.expect.horizon_sec == 72 * 3600.0
    assert act.parent_act_id == stable_act_id("hub.chat_reply", "say", "c2")
    # No copy of the quote on the act: only the pointer.
    assert "deep read" not in act.model_dump_json()
    # The same ref Prime's hand-back links promises by, and a deterministic act id.
    assert act.act_id == promise_act_id(promise_ref("c2", 0))
    assert parse_promise_ref(act.test_before) == ("c2", 0)


def test_unaccepted_offer_never_opens_and_accepted_offer_opens_on_the_earlier_turn() -> None:
    prior = [(promise_ref("c1", 1), "If you want, I'll re-read it the next time I'm cut.")]
    none = plan_promise_events(correlation_id="c2", session_id="s", reflection=_reflection(), prior_offers=prior)
    assert none.events == []
    yes = plan_promise_events(correlation_id="c2", session_id="s", prior_offers=prior, reflection=_reflection(
        offers_accepted=["If you want, I’ll re-read it the next time I'm cut."]))  # curly quote: same offer
    (event,) = yes.events
    assert event.act.test_before == promise_ref("c1", 1)
    assert event.act.parent_act_id == stable_act_id("hub.chat_reply", "say", "c1")
    assert yes.opened_offers == [promise_act_id(promise_ref("c1", 1))]
    stray = plan_promise_events(correlation_id="c2", session_id="s", prior_offers=prior,
                                reflection=_reflection(offers_accepted=["something never offered"]))
    assert stray.events == [] and stray.accepted_unmatched == 1


def test_items_without_force_or_done_when_are_skipped_and_counted() -> None:
    plan = plan_promise_events(correlation_id="c2", session_id="s", reflection=_reflection(promises_made=[
        _p("an old row from before force existed", force=None, done_when=None),
        _p("a line with no completion", done_when=None)]))
    assert plan.events == [] and plan.skipped_no_force == 2


def test_keep_closes_with_the_reply_as_evidence_and_a_keep_without_quote_does_not() -> None:
    a, b = promise_act_id(promise_ref("c1", 0)), promise_act_id(promise_ref("c1", 1))
    plan = plan_promise_events(correlation_id="c3", session_id="s", reflection=_reflection(
        promises_kept=[a, b], promises_kept_evidence=[PromiseKeptV1(id=a, quote="Here is the read.")]))
    (event,) = plan.events
    assert event.event_kind == "act_closed" and event.act.act_id == a
    assert event.act.outcome.value == 1.0 and event.act.outcome.source == "producer"
    assert kept_evidence_kind(event.act.outcome.note) == "reply"
    assert promise_close_kind("producer", event.act.outcome.note) == "kept"
    assert plan.kept_without_evidence == 1


def test_close_kinds_and_shadow_set() -> None:
    assert promise_close_kind(None, None, closed=False) == "open"
    assert promise_close_kind("expired", VOID_NOTE) == "void"
    assert promise_close_kind("expired", "no keep within horizon") == "expired"
    assert promise_close_kind("producer", "late:30s kept_in:c9 evidence=reply") == "expired"  # a late keep is a miss
    assert is_shadow_signal("promise_kept") and not is_shadow_signal("juniper_replied")
    assert SHADOW_ACT_SIGNALS == frozenset({"promise_kept"})


def test_every_promise_is_one_faculty_bucket() -> None:
    assert act_faculty(PROMISE_ACTOR, "say", "juniper") == "act.hub.promise.say"
