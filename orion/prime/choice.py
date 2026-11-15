"""Prime's choice point (personality spec rev 4, move 1): one ``PersonalityChoiceV1`` per thought.

Spec: docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md (rev 4 "Move 1",
rev 3 "Prime wiring"). Each Prime thought already chooses at most one held concern to act on, or
none. The thought's answer now says so explicitly (``PrimeAnswerV1.pick``: a concern_id from YOUR
CONCERNS, or "none"), in the same model call. No new model call.

The options are the concerns the frame showed (captured when the frame was built, so a concern the
same answer adds or drops does not change what Orion chose among). Rules, all deterministic:

* pick = a shown concern and at least 2 concerns shown: chosen = that concern, alternatives = the
  others.
* pick = "none" (or JSON null) and at least 1 concern shown: ``declined_all``, alternatives = all.
* a pick that contradicts the move's ``concern_id`` (both naming shown concerns, or "none" with a move about
  a shown concern): no row (``pick_conflicts_move``).
* no ``pick`` key: the move's ``concern_id`` counts as the pick when it names a shown concern (a
  move about a concern is a choice of it). Otherwise no row: the answer did not say.
* anything else (one concern and it was picked, an unknown id, "new", nothing held): no row, with
  the reason, so the log shows why a thought made no choice.

``market_rank``: Prime holds concerns in the order it added them, which is not a ranking. The one
real order is the body economy's: in event mode (PRIME_CONCERNS_FROM_EVENT_WANTS) each concern names
a funded event want, and the frame lists those "highest value first" (durable-runs
GET /physiology/event-wants). A concern's market rank is its event's position in that list, counted
among the shown concerns only (1 = the option the market valued most). A concern whose event is not
in the list (a turn promise, a day-letter thread, arc mode) has no rank (None). When fewer than 2
shown concerns are in the list, the market ordered nothing among them: every rank is None.

``key`` (personality rev 4 Move 2, source (c)): a concern whose event is in the funded event wants list is
keyed ``<want family>::<concern_id>`` (``orion.physiology.weights.endorsement_option_key``), so a Prime
choice of a reading concern counts for ``hub.world_pulse_read``, not for ``prime`` as a lump. A concern
with no event in the list keeps its plain id (``option_family`` then reads the lane's family, ``prime``).

``care_promoted``: the chosen concern's event want was in the list only through the care term (the list's
own ``care_promoted`` flag, computed by durable-runs over every funded event want with and without care).
Family and flag are recorded on the concern when it opens (``PrimeConcernV1.event_family`` /
``event_care_promoted``): the guard is about the moment care put the option in front of Orion, and a concern
outlives its want's place on the list. A concern from before that has neither; the live list is read.

``situation``: what Prime's BODY section showed: the binding constraints and the strain dial (Prime's
frame shows no arousal reading; strain is the body-state number it does see).

Pure. The runtime (services/orion-mind/app/prime_loop.py) publishes the result, fail-open.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Collection, Mapping, Optional, Sequence

from orion.physiology.weights import endorsement_option_key
from orion.prime.day_letter import LETTER_EVENT_PREFIX
from orion.prime.handback import TURN_EVENT_PREFIX, WHO_WORDS
from orion.schemas.personality import CHOICE_OPTION_TEXT_MAX, ChoiceOptionV1, PersonalityChoiceV1

LANE = "prime"
PICK_NONE = "none"


@dataclass(frozen=True)
class ShownConcern:
    """One concern as the frame showed it."""

    concern_id: str
    text: str
    event_ref: Optional[str] = None
    # As recorded when the concern opened (``PrimeConcernV1.event_family`` / ``event_care_promoted``); None:
    # not recorded (a concern from before 2026-11-13), the live list is read instead.
    family: Optional[str] = None
    care_promoted: Optional[bool] = None


def shown_concerns(concerns: Sequence[Any]) -> list[ShownConcern]:
    """Snapshot of the concerns in the frame (``PrimeConcernV1``-like objects)."""
    return [ShownConcern(c.concern_id, c.text, getattr(c, "event_ref", None), getattr(c, "event_family", None),
                         getattr(c, "event_care_promoted", None)) for c in concerns]


def market_ranks(shown: Sequence[ShownConcern], ranked_event_ids: Sequence[str]) -> dict[str, Optional[int]]:
    """concern_id -> its rank among the shown concerns by the market's event order (1 = highest value);
    None when its event is not in the list. Fewer than 2 placed: the market ordered nothing among these
    options, so every rank is None (a lone placed concern is not "the market's top" over unranked ones)."""
    pos = {e: i for i, e in enumerate(ranked_event_ids)}
    placed = sorted((pos[c.event_ref], c.concern_id) for c in shown if c.event_ref in pos)
    ranks: dict[str, Optional[int]] = {c.concern_id: None for c in shown}
    if len(placed) < 2:
        return ranks
    for r, (_p, cid) in enumerate(placed, start=1):
        ranks[cid] = r
    return ranks


def listed_wants(events: Optional[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """event_ref -> its want in a ``FundedEventWantsV1`` dump (Prime's ``inputs.events``). The list is
    highest net first, so when two wants share an event the first (top) one wins, for every reader."""
    out: dict[str, Mapping[str, Any]] = {}
    for w in (events or {}).get("wants") or []:
        if isinstance(w, Mapping) and w.get("event_ref"):
            out.setdefault(str(w["event_ref"]), w)
    return out


def event_families(events: Optional[Mapping[str, Any]]) -> dict[str, str]:
    """event_ref -> want family, from the listed wants."""
    return {ref: str(w["family"]) for ref, w in listed_wants(events).items() if w.get("family")}


def care_promoted_events(events: Optional[Mapping[str, Any]]) -> set[str]:
    """The event_refs whose (top) listed want reached the list only through care (``care_promoted``)."""
    return {ref for ref, w in listed_wants(events).items() if w.get("care_promoted") is True}


def option_key(concern: ShownConcern, families: Mapping[str, str]) -> str:
    """The concern's option key: ``<family>::<concern_id>`` when its event names a want family (recorded at
    open, else the live list); else the plain concern id (lane family ``prime``)."""
    family = concern.family or (families.get(concern.event_ref) if concern.event_ref else None)
    key = endorsement_option_key(family, concern.concern_id) if family else concern.concern_id
    return key[:200]


def concern_care_promoted(concern: ShownConcern, live: Collection[str]) -> bool:
    """Recorded at open when known; else the live list's flag for its event."""
    if concern.care_promoted is not None:
        return concern.care_promoted
    return bool(concern.event_ref) and concern.event_ref in set(live or ())


def _text(text: str) -> str:
    text = " ".join((text or "").split()) or "(no words)"
    return text if len(text) <= CHOICE_OPTION_TEXT_MAX else text[: CHOICE_OPTION_TEXT_MAX - 1] + "…"


# --- the option's words in the choice record (live 2026-11-14) ---------------------------------------------
# A concern's ``text`` leads with where it came from ("a thread from my letter for <date>: ...", "a promise I made
# in a curiosity run (1 promise, <trace ref>)"), so every tendency label built from the first words of the chosen
# and declined options read "a thread from my letter over a promise I made". The record's text leads with the
# SUBJECT instead, then a short "about" clause, with markdown and trace refs taken out. Prime's own prompt keeps
# ``text`` as it is (its refs are what the citation gate checks); the refs stay on the option's key and the
# concern. Deterministic, no model.
_MARKUP = re.compile(r"\*\*|__|`")
# "[curiosity:7119b210c705]", "[juniper_chat:7a286bdd-acc]", "harness_turn_trace:<id>#promises_made[0]".
_REF = re.compile(r"\s*\[[a-z_]+:[0-9A-Za-z_-]{6,}[^\]]*\]|\s*\b[a-z_]+_trace:\S+")
_LETTER = re.compile(r"^a thread from my letter for (?P<day>\d{4}-\d{2}-\d{2}):\s*(?P<thread>.*)$", re.S)
# Old ("a thread I opened", before 11-12) and current ("a promise I made") promise concern text.
_PROMISE = re.compile(r"^a (?:promise I made|thread I opened) in an? (?P<words>[^(]+?) \((?P<count>[^)]*)\)"
                      r"(?::\s*(?P<rest>.*))?$", re.S)
_JUNIPER_LINE = re.compile(r'^Juniper: "(?P<line>.*?)"(?=\s*(?:/ me:|$))', re.S)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHO_BY_WORDS = {words: who for who, words in WHO_WORDS.items()}


def _plain(text: str) -> str:
    """Markdown emphasis, backticks and trace refs out; whitespace collapsed."""
    return " ".join(_REF.sub("", _MARKUP.sub("", text or "")).split())


def _short_day(day: str) -> str:
    return day[5:] if len(day) == 10 else day


def _letter_text(day: str, thread: str) -> str:
    """'<headline> — <rest> (a thread from my letter for MM-DD)'. The headline is the thread's bold lead when
    it has one (the letter writes "**The walkway render break is unresolved.** 59 failures..."), else its
    first sentence."""
    bold = re.match(r"^\s*\*\*(?P<head>.+?)\*\*\s*(?P<rest>.*)$", thread, re.S)
    if bold:
        head, rest = _plain(bold["head"]), _plain(bold["rest"])
    else:
        parts = _SENTENCE_END.split(_plain(thread), maxsplit=1)
        head, rest = parts[0], (parts[1] if len(parts) > 1 else "")
    head = head.rstrip(".").strip() or "(no words)"
    tail = f"a thread from my letter for {_short_day(day)}"
    # Cut the middle, never the tail: the record keeps where the thread came from.
    room = CHOICE_OPTION_TEXT_MAX - len(head) - len(tail) - 6
    if len(rest) > max(room, 0):
        rest = rest[: max(room - 1, 0)].rstrip() + "…" if room > 1 else ""
    return f"{head} — {rest} ({tail})" if rest else f"{head} ({tail})"


def _promise_text(words: str, rest: str) -> str:
    """A promise option, built from the concern's public text only. The promise sentence itself
    (``PrimeConcernV1.promise_text``) never goes in: it is shown only in Prime's own thinking frame, and the
    record text becomes tendency labels that reach every turn's Prime block (choice-as-EFE acceptance S-1.8).
    A chat promise leads with Juniper's own line, the reply she is owed; any other names where it was made."""
    who = _WHO_BY_WORDS.get(words.strip())
    where = words.strip()
    line = _JUNIPER_LINE.match(rest or "")
    if who == "chat_juniper" or line:
        asked = _plain(line["line"]) if line else ""
        return f'"{asked}" — a reply I owe Juniper from chat' if asked else "A reply I owe Juniper from chat"
    return f"{where[:1].upper()}{where[1:]} promise — the words stay in my own thinking"


def option_text(concern: ShownConcern) -> str:
    """The option's words in the choice record: subject first, then what it is about (see above)."""
    raw = " ".join((concern.text or "").split())
    m = _LETTER.match(raw)
    if m and (not concern.event_ref or concern.event_ref.startswith(LETTER_EVENT_PREFIX)):
        return _text(_letter_text(m["day"], m["thread"]))
    p = _PROMISE.match(raw)
    if p and (not concern.event_ref or concern.event_ref.startswith(TURN_EVENT_PREFIX)):
        return _text(_promise_text(p["words"], p["rest"] or ""))
    return _text(_plain(raw))


def resolve_pick(answer: Any, shown: Sequence[ShownConcern]) -> tuple[Optional[str], Optional[str]]:
    """(pick, why-not). pick is a shown concern_id or PICK_NONE; None with the reason when the answer
    made no usable pick."""
    ids = {c.concern_id for c in shown}
    move_cid = getattr(getattr(answer, "move", None), "concern_id", None)
    move_cid = move_cid if move_cid in ids else None
    if "pick" in getattr(answer, "model_fields_set", set()):
        raw = answer.pick
        pick = PICK_NONE if raw is None else str(raw).strip()
        if pick.lower() in (PICK_NONE, "null", "no_move", "no move"):
            pick = PICK_NONE
        elif pick not in ids:
            return None, "pick_not_shown"
        # The prompt says a move about a concern picks it: a pick that contradicts the act Orion took is
        # not evidence either way (it would be a false decline or a false departure).
        if move_cid is not None and pick != move_cid:
            return None, "pick_conflicts_move"
        return pick, None
    if move_cid is not None:
        return move_cid, None
    return None, "no_pick"


def prime_choice(
    answer: Any,
    shown: Sequence[ShownConcern],
    *,
    thought_id: str,
    occurred_at: datetime,
    ranked_event_ids: Sequence[str] = (),
    situation: Optional[dict[str, Any]] = None,
    act_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    families: Optional[Mapping[str, str]] = None,
    care_promoted: Collection[str] = (),
) -> tuple[Optional[PersonalityChoiceV1], str]:
    """(the choice, reason). The choice is None when this thought made no recordable choice; the
    reason then says why (``no_concerns``, ``one_concern_picked``, ``pick_not_shown``, ``pick_conflicts_move``,
    ``no_pick``). ``families``: event_ref -> want family (``event_families``). ``care_promoted``: event_refs
    whose want reached the list only through care (``care_promoted_events``)."""
    if not shown:
        return None, "no_concerns"
    pick, why = resolve_pick(answer, shown)
    if pick is None:
        return None, why or "no_pick"
    ranks = market_ranks(shown, ranked_event_ids)
    fams = families or {}
    options = {c.concern_id: ChoiceOptionV1(key=option_key(c, fams), text=option_text(c),
                                            market_rank=ranks[c.concern_id]) for c in shown}
    common = dict(choice_id=f"{LANE}:{thought_id}", lane=LANE, occurred_at=occurred_at,
                  situation=dict(situation or {}), act_id=act_id, correlation_id=correlation_id)
    if pick == PICK_NONE:
        return PersonalityChoiceV1(declined_all=True, alternatives=list(options.values()), **common), "declined_all"
    if len(shown) < 2:
        return None, "one_concern_picked"
    alternatives = [o for k, o in options.items() if k != pick]
    picked = next(c for c in shown if c.concern_id == pick)
    promoted = concern_care_promoted(picked, care_promoted)
    return PersonalityChoiceV1(chosen=options[pick], alternatives=alternatives, care_promoted=promoted,
                               **common), "chosen"


def body_situation(body: Optional[dict[str, Any]], binding: Optional[Sequence[str]], *,
                   event_mode: bool) -> dict[str, Any]:
    """The flat situation dict: what the BODY section showed. Unread body -> body_read False."""
    out: dict[str, Any] = {"body_read": body is not None, "event_mode": bool(event_mode)}
    if binding is not None:
        out["binding"] = ",".join(binding)[:200]
        out["binding_count"] = len(binding)
    strain = (body or {}).get("strain")
    if isinstance(strain, dict):
        value = strain.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out["strain"] = round(float(value), 4)
        if strain.get("constraint"):
            out["strain_constraint"] = str(strain["constraint"])[:200]
    return out


__all__ = ["LANE", "PICK_NONE", "ShownConcern", "body_situation", "care_promoted_events", "concern_care_promoted",
           "event_families", "listed_wants", "market_ranks", "option_key", "option_text", "prime_choice", "resolve_pick",
           "shown_concerns"]
