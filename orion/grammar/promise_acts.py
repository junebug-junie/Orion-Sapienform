"""Intention ledger D2: one act per promise Orion makes Juniper, on the existing act ledger.

Design: docs/superpowers/specs/2026-11-11-intention-ledger-design.md (D2, and the 2026-11-16
amendments). Pure: builds grammar events and reads closes; orion-hub publishes them
(orion/hub/promise_ledger.py), orion-sql-writer persists and resolves them
(orion/grammar/act_ledger.py).

One promise = one act:

* actor ``hub.promise``, verb ``say``, object ``juniper``. Its own actor, not ``hub.chat_reply``:
  the reply's per-actor readers (workspace binding, body-economy family, downstream use) must
  not count a promise as a second reply. One faculty bucket (``act.hub.promise.say``) and one
  posterior key for every promise (amendment 11).
* ``test_before`` = ``harness_turn_trace:<corr>#promises_made[<i>]``: the quote stays in the
  promising turn's record. The act holds no copy. Prime's hand-back links promises by the
  same string (orion/prime/handback.py ``promise_refs``).
* act id = ``stable_act_id(actor, verb, <ref>)``: the same promise always has the same act.
* signal ``promise_kept``, horizon 72 h (Juniper's fixed horizon, 2026-11-12).
* Opened when Juniper sees the reply that makes a firm promise; an offer opens only when her
  next message accepts it (Juniper's answer 1; amendment 4). Closed:
  - kept: a later judged reply quotes the words that deliver it (producer, 1,
    ``kept_in:<corr> evidence=reply``);
  - expired: the horizon passed with Juniper writing in it and no keep (resolver, 0);
  - void: the horizon passed and Juniper wrote no chat turn in it, so the judge never had a
    reply to see the keep in (censored; ``void:`` note; never graded; amendment 7).
  Settle classes (#3595, amendments 2026-11-18): the resolver reads the promise item the act's
  ``test_before`` points at. An unkept act whose item
  - waits on a ``condition`` that no judged reply inside the horizon listed as met
    (``FinalizeReflectionV1.promise_conditions_met``) closes ``void:not_due``;
  - ``settles_in == "absence"`` closes ``void:absence_unchecked`` (no breach check exists yet);
  - ``settles_in == "run"`` closes ``void:not_chat`` (a chat reply's silence is not a break);
  - otherwise (``chat``, or a row written before the fields) follows the two rules above.
  Every one keeps the ``void:`` prefix, so every reader that skips void closes (the act ledger's
  posterior and outcome writes, the self-monitor, the report's keep rate) still treats it as
  censored. A chat reply that delivers a run or absence promise still keeps it (evidence=reply).
  A later carry run that delivers a void act is a NOTE in the report, never a reopen.

Offers (Juniper's answer 1, amendment 4; acceptance window #3595). The judge is shown the firm offers
in Orion's immediately previous reply in this session, and only when that reply answered a Juniper turn
and was stored at most ``OFFER_ACCEPT_TTL_SEC`` before this turn's view read (one clock with the
promise horizon). A later acceptance of an older offer matches nothing. Two counters can fire on a miss:
the gate's ``dropped_unknown_offer`` (orion/harness/promise_gate.py rule 2) when the judge quotes an
offer it was not shown (the offer was older than the TTL, or another row landed in between), and this
plan's ``accepted_unmatched`` when an accepted quote survived the gate but matches no offer the view
read (the view and the gate disagree; should stay 0).

SHADOW (2026-11-16, until #3531): every act on a signal in ``SHADOW_ACT_SIGNALS`` is persisted and
closed, but nothing learns from it: sql-writer writes no outcome row and no posterior update,
the self-monitor builds no settled bet from it, and the act-chain forecaster leaves it out.
Going live is one code change here (empty the set), reviewed in its own PR.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Sequence

from orion.grammar.acts import act_closed_event, act_opened_event, build_act, stable_act_id
from orion.schemas.grammar import ActExpectV1, ActV1, GrammarEventV1
from orion.schemas.harness_finalize import (
    PROMISE_DONE_WHEN_MIN_CHARS,
    PROMISE_FORCE_COMMITS,
    PROMISE_SETTLE_CLASSES,
    PROMISE_SETTLES_CHAT,
    FinalizeReflectionV1,
    PromiseV1,
)
from orion.schemas.promise_horizon import PROMISE_HORIZON_SEC  # the one 72 h promise clock (#3595)

PROMISE_ACTOR = "hub.promise"
PROMISE_VERB = "say"
PROMISE_SIGNAL = "promise_kept"
PROMISE_OBJECT_REF = "juniper"
PROMISE_BASIS = (
    "promise_kept by 72h: Juniper's fixed horizon for every promise (intention ledger, answer 2, "
    "2026-11-12); kept = a later judged reply quotes the words that deliver it; no Juniper chat "
    "turn inside the horizon = void (no reply existed to keep it in), not 0"
)
REPLY_ACTOR = "hub.chat_reply"
# How long an offer stays acceptable (#3595): one clock with the promise horizon. Applied on top of the
# "immediately previous reply in this session" rule (orion/grammar/promise_view.py).
OFFER_ACCEPT_TTL_SEC = PROMISE_HORIZON_SEC

# Signals whose acts are on the ledger in shadow: persisted and closed, never graded, never a
# settled bet, never a learner's input. #3531 decides when promise_kept leaves this set.
SHADOW_ACT_SIGNALS: frozenset[str] = frozenset({PROMISE_SIGNAL})
# For SQL readers: ``AND <column> IS DISTINCT FROM ALL(...)`` is awkward in text(); this is the
# literal list, built from the set above so the two cannot drift.
SHADOW_ACT_SIGNALS_SQL = ", ".join(f"'{s}'" for s in sorted(SHADOW_ACT_SIGNALS))

KEPT_NOTE_PREFIX = "kept_in:"
VOID_NOTE_PREFIX = "void:"
VOID_NOTE = "void: no juniper chat turn started before horizon (censored, not graded)"
EXPIRED_NOTE = "no keep within horizon (juniper wrote, no reply delivered it)"
# The resolver's settle-class voids (#3595). Each starts with VOID_NOTE_PREFIX: censored, never graded.
VOID_NOT_DUE_NOTE = "void:not_due condition never met within horizon (censored, not graded)"
VOID_NOT_CHAT_NOTE = "void:not_chat delivered by a run or outside chat, no chat keep seen (censored, not graded)"
VOID_ABSENCE_NOTE = "void:absence_unchecked abstention, no breach check exists yet (censored, not graded)"
# void kind -> note; "no_juniper_turn" is the v1 void (VOID_NOTE).
VOID_KINDS: tuple[str, ...] = ("no_juniper_turn", "not_due", "not_chat", "absence_unchecked")
EVIDENCE_REPLY = "reply"
# A promise repeated word for word opens no second act only when it is at least this long: a short
# generic line ("I'll check back tomorrow.") said again on a new topic is a new promise.
REPEAT_MIN_CHARS = 40

_REF = re.compile(r"^harness_turn_trace:(?P<corr>[^#\s]{1,200})#promises_made\[(?P<i>\d{1,3})\]$")


def promise_ref(correlation_id: str, index: int) -> str:
    return f"harness_turn_trace:{correlation_id}#promises_made[{int(index)}]"


def parse_promise_ref(ref: str) -> Optional[tuple[str, int]]:
    m = _REF.match(ref or "")
    return (m.group("corr"), int(m.group("i"))) if m else None


def promise_act_id(ref: str) -> str:
    return stable_act_id(PROMISE_ACTOR, PROMISE_VERB, ref)


def is_firm_ledger_promise(promise: PromiseV1) -> bool:
    """A stored item the ledger can open: it commits and names a completion. Rows written before
    the force/done_when fields (#3532) lack them; the ledger skips those and counts them
    (amendment 11) rather than guessing."""
    return promise.force == PROMISE_FORCE_COMMITS and len((promise.done_when or "").strip()) >= PROMISE_DONE_WHEN_MIN_CHARS


def build_promise_act(ref: str, *, parent_act_id: Optional[str]) -> ActV1:
    return build_act(
        actor=PROMISE_ACTOR, verb=PROMISE_VERB, key=ref, signal_id=PROMISE_SIGNAL,
        horizon_sec=PROMISE_HORIZON_SEC, basis=PROMISE_BASIS, object_ref=PROMISE_OBJECT_REF,
        test_before=ref, parent_act_id=parent_act_id,
    )


def _act_for_id(act_id: str) -> ActV1:
    """The act a keep closes, rebuilt from its id (the writer closes by act_id; the open row holds
    the rest)."""
    return ActV1(act_id=act_id, level="deliberate", actor=PROMISE_ACTOR, verb=PROMISE_VERB,
                 object_ref=PROMISE_OBJECT_REF,
                 expect=ActExpectV1(signal_id=PROMISE_SIGNAL, direction="increase",
                                    horizon_sec=PROMISE_HORIZON_SEC, basis=PROMISE_BASIS))


def _norm(text: str) -> str:
    from orion.harness.promise_gate import normalize_for_quote_match

    return normalize_for_quote_match(text)


class PromiseLedgerPlan:
    """What one delivered turn does to the ledger, with the counts the Hub logs."""

    def __init__(self) -> None:
        self.events: list[GrammarEventV1] = []
        self.opened: list[str] = []          # act ids opened for this reply's own firm promises
        self.opened_offers: list[str] = []   # act ids opened for earlier offers this message accepted
        self.kept: list[str] = []            # act ids closed as kept
        self.offers_not_opened = 0           # this reply's own offers: wait for acceptance
        self.skipped_no_force = 0            # items without force/done_when (pre-#3532 judge)
        self.accepted_unmatched = 0          # an accepted quote not among the prior offers
        self.kept_without_evidence = 0       # a kept id with no delivering quote (gate should prevent)
        self.items = 0                       # promises_made items the gated reflection carried
        self.repeats_open: list[str] = []    # open act ids a firm promise here repeated word for word
        self.conditions_met = 0              # open conditional promises whose condition this turn saw arise

    def counts(self) -> dict[str, int]:
        return {"items": self.items, "opened": len(self.opened), "opened_offers": len(self.opened_offers),
                "kept": len(self.kept), "offers_not_opened": self.offers_not_opened,
                "skipped_no_force": self.skipped_no_force, "repeats_open": len(self.repeats_open),
                "accepted_unmatched": self.accepted_unmatched,
                "kept_without_evidence": self.kept_without_evidence, "conditions_met": self.conditions_met}

    def outcome(self) -> str:
        """One word for the Hub's per-turn line, so a quiet turn and a ledger that never ran look
        different (Orion's PR #3561 question 1): ``found_nothing`` (the gated reflection carried no
        promise, offer, acceptance or keep), ``nothing_opened`` (it carried items, and every one was
        skipped: an offer waiting, a pre-#3532 item, a repeat, an unmatched acceptance), or ``acted``
        (at least one event: an open, an accepted offer opened, or a keep closed; a keep-only turn is
        ``acted`` too)."""
        if self.events:
            return "acted"
        c = self.counts()
        if c["items"] or c["accepted_unmatched"] or c["kept_without_evidence"] or c["conditions_met"]:
            return "nothing_opened"
        return "found_nothing"


def plan_promise_events(
    *,
    correlation_id: str,
    session_id: Optional[str],
    reflection: Optional[FinalizeReflectionV1],
    prior_offers: Sequence[tuple[str, str]] = (),
    open_promises: Sequence[tuple[str, str]] = (),
    at: Optional[datetime] = None,
) -> PromiseLedgerPlan:
    """The act events for one delivered Juniper turn. ``reflection`` is the gated reflection on the
    run artifact (checked against the text Juniper saw). ``prior_offers``: (ref, quote) of the
    offers in Orion's previous reply in this session, the ones the judge was shown. ``open_promises``:
    (act id, quote) of the open promises the judge was shown; a firm promise with the same words
    (after quote normalization, and at least ``REPEAT_MIN_CHARS`` long, so a short generic line like
    "I'll check back tomorrow." on a new topic still counts as a new promise) is the open promise
    said again, not a second promise, so it opens no act (``repeats_open`` holds the matched act
    ids). A reworded repeat is not caught here."""
    plan = PromiseLedgerPlan()
    if reflection is None:
        return plan
    corr = str(correlation_id)
    # Read by the resolver from the stored artifact, not published: counted for the Hub's line only.
    plan.conditions_met = len(reflection.promise_conditions_met)
    reply_act = stable_act_id(REPLY_ACTOR, "say", corr)
    already_open = {_norm(q): act_id for act_id, q in open_promises if q and len(_norm(q)) >= REPEAT_MIN_CHARS}
    plan.items = len(reflection.promises_made)
    for i, promise in enumerate(reflection.promises_made):
        if not is_firm_ledger_promise(promise):
            plan.skipped_no_force += 1
            continue
        if promise.offer:
            plan.offers_not_opened += 1
            continue
        repeated = already_open.get(_norm(promise.quote))
        if repeated is not None:
            plan.repeats_open.append(repeated)
            continue
        act = build_promise_act(promise_ref(corr, i), parent_act_id=reply_act)
        plan.events.append(act_opened_event(act, source_service="orion-hub", source_component="promise_ledger.open",
                                            correlation_id=corr, session_id=session_id, at=at))
        plan.opened.append(act.act_id)
    offers = {_norm(quote): ref for ref, quote in prior_offers}
    for quote in reflection.offers_accepted:
        ref = offers.get(_norm(quote))
        parsed = parse_promise_ref(ref or "")
        if parsed is None:
            plan.accepted_unmatched += 1
            continue
        act = build_promise_act(ref, parent_act_id=stable_act_id(REPLY_ACTOR, "say", parsed[0]))
        if act.act_id in plan.opened_offers:
            continue
        plan.events.append(act_opened_event(act, source_service="orion-hub",
                                            source_component="promise_ledger.offer_accepted",
                                            correlation_id=corr, session_id=session_id, at=at))
        plan.opened_offers.append(act.act_id)
    evidence = {ev.id for ev in reflection.promises_kept_evidence}
    for act_id in reflection.promises_kept:
        if act_id not in evidence:
            plan.kept_without_evidence += 1
            continue
        if act_id in plan.kept:
            continue
        plan.events.append(act_closed_event(
            _act_for_id(act_id), produced=True, source_service="orion-hub", source_component="promise_ledger.kept",
            note=f"{KEPT_NOTE_PREFIX}{corr} evidence={EVIDENCE_REPLY}", correlation_id=corr, session_id=session_id,
            at=at))
        plan.kept.append(act_id)
    return plan


def promise_close_kind(outcome_source: Optional[str], outcome_note: Optional[str],
                       closed: bool = True) -> str:
    """open | kept | expired | void, from a grammar_acts row (the report and the self-monitor read
    closes this one way)."""
    if not closed:
        return "open"
    note = (outcome_note or "").strip()
    if note.startswith(VOID_NOTE_PREFIX):
        return "void"
    if (outcome_source or "") == "producer" and note.startswith(KEPT_NOTE_PREFIX):
        return "kept"
    return "expired"


def promise_void_kind(note: Optional[str]) -> Optional[str]:
    """Which void a close note is (``VOID_KINDS``), None when it is not a void."""
    text = (note or "").strip()
    if not text.startswith(VOID_NOTE_PREFIX):
        return None
    head = text[len(VOID_NOTE_PREFIX):].split(" ", 1)[0]
    return head if head in VOID_KINDS[1:] else "no_juniper_turn"


def settle_class(item: Optional[dict]) -> str:
    """A stored promise item's settle class; ``chat`` when unknown (rows before #3595, an unreadable item)."""
    value = (item or {}).get("settles_in")
    return value if value in PROMISE_SETTLE_CLASSES else PROMISE_SETTLES_CHAT


def promise_condition(item: Optional[dict]) -> Optional[str]:
    """A stored promise item's condition (stripped), None when unconditional or unknown."""
    value = (item or {}).get("condition")
    return (value.strip() or None) if isinstance(value, str) else None


def horizon_void_note(item: Optional[dict], *, condition_met: bool) -> Optional[str]:
    """The settle-class void for an UNKEPT promise act at its horizon (#3595), or None when the v1 rule
    applies (expired 0 if Juniper wrote, void if she did not). ``item`` is the stored promise the act's
    ref points at (None: unreadable, read as chat and unconditional)."""
    if promise_condition(item) is not None and not condition_met:
        return VOID_NOT_DUE_NOTE
    cls = settle_class(item)
    if cls == "absence":
        return VOID_ABSENCE_NOTE
    if cls == "run":
        return VOID_NOT_CHAT_NOTE
    return None


def is_shadow_signal(signal_id: Optional[str]) -> bool:
    return (signal_id or "") in SHADOW_ACT_SIGNALS


def is_void_note(note: Optional[str]) -> bool:
    return (note or "").strip().startswith(VOID_NOTE_PREFIX)


def kept_evidence_kind(note: Optional[str]) -> Optional[str]:
    m = re.search(r"\bevidence=(\w+)", note or "")
    return m.group(1) if m else None


__all__ = [
    "EVIDENCE_REPLY", "EXPIRED_NOTE", "KEPT_NOTE_PREFIX", "OFFER_ACCEPT_TTL_SEC", "PROMISE_ACTOR", "PROMISE_BASIS",
    "PROMISE_HORIZON_SEC", "PROMISE_OBJECT_REF", "PROMISE_SIGNAL", "PROMISE_VERB", "PromiseLedgerPlan",
    "REPEAT_MIN_CHARS", "SHADOW_ACT_SIGNALS", "SHADOW_ACT_SIGNALS_SQL", "VOID_ABSENCE_NOTE", "VOID_KINDS",
    "VOID_NOTE", "VOID_NOTE_PREFIX", "VOID_NOT_CHAT_NOTE", "VOID_NOT_DUE_NOTE", "build_promise_act",
    "horizon_void_note", "is_firm_ledger_promise", "is_shadow_signal", "is_void_note", "kept_evidence_kind",
    "parse_promise_ref", "plan_promise_events", "promise_act_id", "promise_close_kind", "promise_condition",
    "promise_ref", "promise_void_kind", "settle_class",
]
