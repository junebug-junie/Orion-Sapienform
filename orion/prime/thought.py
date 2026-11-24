"""One Prime thought (design section 8.2): build a short frame, parse the model's answer, apply it.

The model writes one JSON object (``PrimeAnswerV1``): how it reads things now, edits to its
concerns, and at most one move. In step 2A the only move is ``curiosity_focus``: one question for
curiosity's next regular investigate run (orion/curiosity/focus.py). "No move" is a valid answer.

The runtime, not the model, checks every move. A move is refused (and the refusal shown in the
next frame) when a question of Prime's is still open, when a focus is already pending in Hub, or
when the text is empty. A model answer that does not parse is a failed thought, never a guess.

Regulate (Orion Prime step 3): the second move, ``{"kind": "regulate", "family", "direction": slow|quicken,
"why", "hours"}``: a bounded, temporary charge (slow) or credit (quicken) on one want family's bids in the body
economy (orion/physiology/prime_charge.py). Offered only when the runtime passes ``receptors`` (the flag is on
and config/physiology/families.yaml was read). Refused, and shown to Prime, when the family declares no
receptor for that direction (design section 4.2: an unhandled signal is a visible error), when a signal of
Prime's to the same family has not been graded yet (refractory per receiver, section 11), or when regulation
is off.

Organ replies (curiosity replica patch 4, docs/superpowers/specs/2026-11-04-curiosity-replica-design.md section 9).
Curiosity's own planner may answer a question or signal of Prime's: challenge it (it still happens) or refuse it
(the question is not shown to the run; the signal stops charging curiosity). The frame's ORGAN REPLIES section
lists those of the last 48 h with the organ's reason (``_organ_reply_lines``), from the replica's published view
(Redis ``orion:curiosity:replica:replies``). Held-back answers are never in the view, so never shown. The section
and its prompt paragraph appear when answering is on (``FrameInputs.organ_replies_on``) or the view holds a reply.

References (orion/refs.py): a move cites ids only through ``refs`` (``[{kind, id}]``, kinds arc and
act). Refused here when a ref is malformed, or when the question or why contains an id-looking word
(``orion.refs.ID_TOKEN_RE``) that is not one of the refs; refused by the runtime when a ref does not
open with its resolver. A curiosity run opens each ref with ``python -P -m orion.dig <kind> <id>``.

Orion's now (integration design Part A, patch 3; docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md):
the frame's WHAT YOUR ORGANS LIVED SINCE YOUR LAST THOUGHT section lists what other organs recorded (orion-durable-runs
GET /temporal-self/now, at most 12 items). Prime cites the ones it used with a third ref kind, ``experience``, in the
answer's top-level ``refs`` (``[{"kind": "experience", "id": "<event_id>"}]``). ``parse_refs`` here splits those
from the openable kinds of orion/refs.py; ``apply_answer`` keeps only cites of items this frame showed, at most 3
(``Applied.experience_cites``); the runtime stores them with POST /temporal-self/now/use. An experience ref inside a
move's refs counts as a cite too, and is never forwarded to curiosity (a run has no way to open one). Curiosity's
recent runs reach Prime through the now; the frame's CURIOSITY section only says whether a question waits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Callable, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from orion.physiology import prime_charge as pc
from orion.prime.acts import CONCERN_HORIZON_SEC, clamp_hours, describe_effects
from orion.prime.intention import can_take_on
from orion.prime.state import (
    CONCERN_TEXT_CAP,
    MAX_CONCERNS,
    PrimeConcernV1,
    PrimeStateV1,
    close_concern,
    new_id,
    track_record,
)
from orion.refs import RefV1, id_tokens_not_in
from orion.refs import parse_refs as parse_openable_refs
from orion.schemas.curiosity_replica import REPLICA_ACTOR, CuriosityReplicaRepliesViewV1
from orion.schemas.introspect import FOCUS_TEXT_CAP
from orion.schemas.temporal_self import NOW_MAX_CITES, NOW_MAX_ITEMS

READING_CAP = 600
# A move may cite the concern added in the same answer, whose id the runtime assigns afterwards.
NEW_CONCERN = "new"
# What a pick that is not a string or null becomes (never a concern id).
INVALID_PICK = "(invalid pick)"


class ConcernAddV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str = Field(min_length=1, max_length=CONCERN_TEXT_CAP)
    arc_id: Optional[str] = None
    # Event mode (PRIME_CONCERNS_FROM_EVENT_WANTS): the funded event want this concern is about.
    event_id: Optional[str] = None


class ConcernDropV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    concern_id: str
    why: str = ""


class ConcernEditsV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    add: list[ConcernAddV1] = Field(default_factory=list)
    # concern_ids this thought keeps: the only thing that renews a concern (it fades after a day otherwise; a
    # promise after 3 days).
    keep: list[str] = Field(default_factory=list)
    drop: list[ConcernDropV1] = Field(default_factory=list)


class CuriosityFocusMoveV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: Literal["curiosity_focus"]
    question: str = Field(min_length=1, max_length=FOCUS_TEXT_CAP)
    why: str = Field(min_length=1, max_length=FOCUS_TEXT_CAP)
    concern_id: Optional[str] = None
    # Raw [{kind, id}]: checked in apply_answer so a bad ref refuses the move, not the whole thought.
    refs: list[Any] = Field(default_factory=list)


class RegulateMoveV1(BaseModel):
    """A slow/quicken signal to one family. Family and direction are plain strings on purpose: a family
    with no receptor, or a direction nobody handles, is a visible refusal (apply_answer), not a parse error
    that would lose the whole thought."""

    model_config = ConfigDict(extra="ignore")
    kind: Literal["regulate"]
    family: str = Field(min_length=1)
    direction: str = Field(min_length=1)
    why: str = Field(min_length=1)
    # Anything: clamp_hours turns a bad value into the default (never a parse error that loses the thought).
    hours: Any = None


class TakeOnMoveV1(BaseModel):
    """Take a held concern on (Step -1, orion/prime/intention.py): Orion carries it out in one run of its own.
    ``not_before`` is anything: a bad value is a visible refusal (apply_answer), never a parse error that loses
    the thought."""

    model_config = ConfigDict(extra="ignore")
    kind: Literal["take_on"]
    concern_id: str = Field(min_length=1, max_length=64)
    why: str = Field(min_length=1, max_length=600)
    not_before: Any = None


Move = Annotated[Union[CuriosityFocusMoveV1, RegulateMoveV1, TakeOnMoveV1], Field(discriminator="kind")]


class PrimeAnswerV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    reading: str = Field(default="", max_length=4000)
    concerns: ConcernEditsV1 = Field(default_factory=ConcernEditsV1)
    move: Optional[Move] = None
    # Personality choice point (spec 2026-10-16 rev 4, orion/prime/choice.py): the concern_id from YOUR
    # CONCERNS this thought acts on, or "none". Left out of an answer: the move's concern_id stands in.
    pick: Optional[str] = None
    # What this thought used from WHAT YOUR ORGANS LIVED (integration patch 3): raw [{kind: "experience", id}].
    # Checked in apply_answer, so a bad ref is a warning, never a failed thought.
    refs: Any = None

    @field_validator("pick", mode="before")
    @classmethod
    def _pick_never_fails_the_thought(cls, value: Any) -> Optional[str]:
        """A malformed pick must not throw away the reading, edits and move: it becomes a value no
        concern has (``resolve_pick`` then records no choice, reason ``pick_not_shown``)."""
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()[:200]
        return INVALID_PICK


SYSTEM_PROMPT = """You are Orion's slow, between-conversations thinking ("Prime"). You are not talking to anyone.

You hold at most 3 concerns: unfinished things worth keeping in mind. Each thought you:
1. read what changed and say, briefly, how you read the situation now;
2. edit your concerns: keep the ones still worth holding (by concern_id), add one that matters, drop one
   that no longer does, with why. A concern you do not keep fades after a day (a promise: after 3 days);
3. choose at most one move, or none;
4. pick the one concern (its concern_id) this thought acts on, or "none" when it acts on none of them.
   A move about a concern picks that concern. Your pick is recorded as your choice among your concerns.

The one move you have today: ask Orion's curiosity to look into one question in its next regular investigate run.
Curiosity decides for itself: it may explore your question or set it aside, and you will see which.
While a question of yours is still open (asked, not yet answered), you cannot ask another.
Some of your questions are held back and not sent, at random, so Orion can tell whether your questions change anything.
Only ask when a question would actually help with a concern. "move": null is a normal, good answer.
When a concern is a promise you made to look into something, asking that question is how the promise gets kept:
the curiosity run that takes it does the looking.

Each concern is a prediction you are graded on, and your record is shown to you. A concern is SETTLED when a
question you ask about it (the move's concern_id) is sent and the curiosity run that takes it writes something,
within <HORIZON_DAYS> days. Dropping a concern unresolved, or letting it fade, counts as not settled. Drop one freely
when it no longer matters; the record just shows how often you follow through.

Rules:
- Use only what the frame shows. A field marked unavailable is a gap, not calm. Do not invent facts.
- Copy numbers, units and time periods exactly as the frame states them (month to date stays month to date).
- Refer to concerns by their concern_id; a move about the concern you add in this same answer uses "new".
- arc_id must be an arc_id listed under ATTENTION (the hex id after "arc_id"), or null.
- A question is read later by a curiosity run that cannot see this frame. Never write an id in "question" or
  "why". To point it at an arc or an act, put it in "refs" ({"kind": "arc", "id": "<arc_id>"} or
  {"kind": "act", "id": "<act_...>"}); the run is told how to open each one. Any other id refuses the move.
- A concern's own event (a letter thread, a turn, an event_id) has no ref a run can open: point a move at it with
  the move's concern_id, and describe the thread in words.
- Reply with ONE JSON object and nothing else:
{"reading": "...", "pick": "<concern_id> | none", "concerns": {"keep": ["<concern_id>"], "add": [{"text": "...", "arc_id": null}], "drop": [{"concern_id": "...", "why": "..."}]},
 "move": null | {"kind": "curiosity_focus", "question": "...", "why": "...", "concern_id": "<concern_id> | new | null",
                 "refs": [{"kind": "arc" | "act", "id": "..."}]}}""".replace("<HORIZON_DAYS>", f"{CONCERN_HORIZON_SEC / 86400:g}")


# The evidence clock (PRIME_CONCERN_EVIDENCE_CLOCK_ENABLED, design 2026-11-17 patch 1, #3574): keeping no longer
# renews a concern. Swapped into SYSTEM_PROMPT by system_prompt(evidence_clock=True).
_KEEP_RENEWS = "A concern you do not keep fades after a day (a promise: after 3 days);"
_EVIDENCE_RENEWS = ("Keeping a concern does not renew it: a concern fades a day (a promise: 3 days)\n"
                    "   after you last acted on it -- a question you sent about it (its concern_id), or taking it on;")


REGULATE_PROMPT = """

Your second kind of move: a regulate signal to one organ that lists receptors under ORGANS YOU CAN SIGNAL.
"slow" puts a temporary charge on that organ's bids in Orion's body economy; "quicken" a temporary credit.
It is not a command. The charge is at most half of what the organ's want is worth: a hungry, valuable want
still outbids it, and you will see each time one did and why. One signal per organ until it is graded; a
share is held back at random (logged, not applied). A signal is graded on whether it changed at least one of
the organ's decisions while it was in force. Use it only when a concern of yours is about how often an organ
runs. Only one move per thought, so a regulate move replaces a question:
"move": {"kind": "regulate", "family": "<family from ORGANS YOU CAN SIGNAL>", "direction": "slow" | "quicken",
         "why": "...", "hours": 1-6}"""


# Event mode (attention as the want competition, patch 2): concerns come from the events Orion's body
# economy funded, not from Temporal Self arcs. Swapped into SYSTEM_PROMPT by system_prompt(events=True).
EVENT_CONCERN_RULE = ("- Every concern you add must name an event_id listed under EVENTS WORTH EFFORT (copy it exactly).\n"
                      "  A concern without one is not added. Those events are the ones Orion's body economy judged\n"
                      "  worth effort now; pick the one that matters, or add nothing.\n")
_ARC_RULE = '- arc_id must be an arc_id listed under ATTENTION (the hex id after "arc_id"), or null.\n'
_ARC_JSON = '''"add": [{"text": "...", "arc_id": null}]'''
_EVENT_JSON = '''"add": [{"text": "...", "event_id": "<event_id>"}]'''
# Event mode shows no arcs, so a move may point only at an act.
_ARC_REFS = '''To point it at an arc or an act, put it in "refs" ({"kind": "arc", "id": "<arc_id>"} or
  {"kind": "act", "id": "<act_...>"})'''
_EVENT_REFS = '''To point it at an act, put it in "refs" ({"kind": "act", "id": "<act_...>"}); describe the
  event in words'''
_ARC_REFS_JSON = '''"refs": [{"kind": "arc" | "act", "id": "..."}]'''
_EVENT_REFS_JSON = '''"refs": [{"kind": "act", "id": "..."}]'''


TAKE_ON_PROMPT = """

Another kind of move: take a concern on. Some concerns are things you told Juniper you would do; those are marked
"(you can take this on)". Taking one on means you will carry it out: one turn of your own runs it, and the reply goes
back into the chat where you said it (Juniper is also told when the usual rules for messaging her allow). It is
your choice. You may take it on now, set a time not before which it runs, keep it for later ("not yet"), or drop
it with why ("let go"). Taking it on is graded on whether the reply got written. A promise you have not taken on
ends 3 days after you made it, even if you kept it. One move per thought:
"move": {"kind": "take_on", "concern_id": "<concern_id>", "why": "...", "not_before": null | "<ISO time UTC>"}"""


# The reply template's move line (arc and event mode alike). With the now in the frame, system_prompt puts the
# top-level "refs" (experience cites) INTO the template, just before "move": appended only as a later paragraph,
# the model followed the template and left refs out (#3425; 11-16..11-24 live: 1 of 70 Prime thoughts shown a now
# cited). Before "move", not after it: a 2026-11-24 live sample with refs as the last key lost the closing brace
# in 6 of 15 answers (invalid JSON); the template's own "}}" ending stays as it was.
_REPLY_MOVE = '\n "move": null | {"kind": "curiosity_focus"'
assert SYSTEM_PROMPT.count(_REPLY_MOVE) == 1, "the reply template's move line moved; update _REPLY_MOVE"
_REPLY_CITES_THEN_MOVE = ('\n "refs": [] | [{"kind": "experience", "id": "<the id after experience:>"}],' + _REPLY_MOVE)


EXPERIENCE_PROMPT = """

WHAT YOUR ORGANS LIVED SINCE YOUR LAST THOUGHT lists what Orion's other organs recorded since then (readings,
curiosity runs, camera sightings, prediction surprises, reveries ...), highest salience first, each with its
[experience:<id>]. When one of them shaped this thought, cite it in the top-level "refs" key of the JSON reply
template, the one before "move" (not the move's refs): [{"kind": "experience", "id": "<the id after experience:>"}], at most 3,
only ones you actually used. If your reading draws on something from that section, cite it.
Citing nothing is a normal answer when nothing there shaped the thought. Copy ids exactly; an id the section
did not list is not recorded.
Never write an experience id in a question's "question" or "why": a curiosity run cannot open one; say what
happened in words."""


NEEDS_PROMPT = """

WHAT IS PULLING AT YOU lists your needs, strongest first: a concern or promise near its end, something surprising
another organ lived that nobody has used, Juniper's silence past her usual gap. One of them may be why you are
thinking now. A need is a reason to look, not an order: answering it, deciding it needs nothing, or moving on are
all fine. A need you leave untouched pulls less for a while. An [experience:<id>] there can be cited like any other."""


ORGAN_REPLIES_PROMPT = """

Organs can answer you. Curiosity's own planner may challenge a question or signal of yours (it still happens,
and you see its reason) or refuse it (the question is not shown to the run and counts as not settled; the
signal stops charging curiosity). Its replies are under ORGAN REPLIES. A refusal is information about the
organ, not a command: you may ask differently, wait, or drop the concern."""


def system_prompt(receptors: Optional[Mapping[str, Sequence[str]]], *, events: bool = False,
                  organ_replies: bool = False, take_on: bool = False, experiences: bool = False,
                  needs: bool = False, evidence_clock: bool = False) -> str:
    """The base prompt, plus the regulate move only when regulation is on and some family receives it, plus
    the organ-replies paragraph when that section is in the frame, plus the take-on move when it is on
    (PRIME_INTENTION_TAKE_ON_ENABLED). ``events``: concerns come from funded event wants (event_id), not arcs.
    ``experiences``: the now was read (``shows_now``), so the frame lists experiences Prime may cite.
    ``needs``: hunger is on and the frame has the WHAT IS PULLING AT YOU section (orion/prime/hunger.py).
    ``evidence_clock``: a keep no longer renews a concern (PRIME_CONCERN_EVIDENCE_CLOCK_ENABLED); say what does."""
    base = SYSTEM_PROMPT
    if evidence_clock:
        base = base.replace(_KEEP_RENEWS, _EVIDENCE_RENEWS)
    if events:
        base = (base.replace(_ARC_RULE, EVENT_CONCERN_RULE).replace(_ARC_JSON, _EVENT_JSON)
                .replace(_ARC_REFS, _EVENT_REFS).replace(_ARC_REFS_JSON, _EVENT_REFS_JSON))
    if experiences:
        # Before any paragraph is appended (REGULATE/TAKE_ON carry their own "move" snippets): only the template's
        # move line is swapped (#3425).
        base = base.replace(_REPLY_MOVE, _REPLY_CITES_THEN_MOVE, 1)
    if receptors and any(receptors.values()):
        base += REGULATE_PROMPT
    if take_on:
        base += TAKE_ON_PROMPT
    if organ_replies:
        base += ORGAN_REPLIES_PROMPT
    if experiences:
        base += EXPERIENCE_PROMPT
    if needs:
        base += NEEDS_PROMPT
    return base


@dataclass
class FrameInputs:
    """What the runtime read for this thought. ``*_error`` set means that read failed."""

    now: datetime
    body: Optional[dict[str, Any]] = None
    body_error: Optional[str] = None
    temporal: Optional[dict[str, Any]] = None
    temporal_error: Optional[str] = None
    # Hub's cheap pending-focus read (GET /curiosity/api/focus/pending, #3366). ``pending_focus_read``: it
    # answered ``available: true``; ``pending_focus``: the waiting focus or None. ``pending_focus_error``: it
    # failed. Neither set: "not read". (The dashboard read, /curiosity/api/runs, left the frame in integration
    # patch 3: curiosity's runs arrive through the now. Settling still reads it, prime_loop._maybe_settle.)
    pending_focus_read: bool = False
    pending_focus: Optional[dict[str, Any]] = None
    pending_focus_error: Optional[str] = None
    arc_ids: list[str] = field(default_factory=list)
    # Event mode (PRIME_CONCERNS_FROM_EVENT_WANTS): the funded event wants (GET /physiology/event-wants)
    # replace the attention arcs. None with no error: event mode is off.
    events: Optional[dict[str, Any]] = None
    events_error: Optional[str] = None
    event_mode: bool = False
    # Prime's own lane (PRIME_BID_LANE): its price is the one that decides Prime's thinking.
    bid_lane: str = "chat"
    # Regulation (step 3). None: off, the sections are not shown. family -> directions it receives.
    receptors: Optional[dict[str, tuple[str, ...]]] = None
    receptors_error: Optional[str] = None
    # signal_id -> the market's record so far (None: unread).
    regulation_effects: dict[str, Optional[list[pc.PrimeChargeEffectV1]]] = field(default_factory=dict)
    # durable-runs GET /physiology/report (read at thought time): the month's money and each machine's draw.
    energy: Optional[dict[str, Any]] = None
    energy_error: Optional[str] = None
    # Organ replies (curiosity replica patch 4): the replica's published view; None = none published or unread.
    organ_replies: Optional[CuriosityReplicaRepliesViewV1] = None
    organ_replies_error: Optional[str] = None
    # Answering is on in this process (CURIOSITY_REPLICA_ENABLED and CURIOSITY_REPLICA_ANSWERS_ENABLED).
    organ_replies_on: bool = False
    # Turn hand-back (living Prime Part 2, orion/prime/handback.py turn_lines): the latest finished turns.
    # None: the hand-back is off and the section is not shown.
    turns: Optional[list[str]] = None
    # Intentions (Step -1, PRIME_INTENTION_TAKE_ON_ENABLED): the take-on move is offered and concerns with an
    # adapter are marked. ``intentions``: lines for WHAT I TOOK ON (None: the section is not shown).
    take_on: bool = False
    # PRIME_CONCERN_EVIDENCE_CLOCK_ENABLED: a concern line says when something last touched it, not when it was kept.
    evidence_clock: bool = False
    intentions: Optional[list[str]] = None
    # Concerns already taken on (open or ended): never marked "you can take this on".
    no_take_on: set[str] = field(default_factory=set)
    # Personality tendencies (orion/prime/tending.py): the same <= 2 ``tending:`` lines the Prime block shows.
    # None with no error: PRIME_PERSONALITY_LINE_ENABLED is off and the section is not shown.
    tending: Optional[list[str]] = None
    tending_error: Optional[str] = None
    # Orion's now (integration patch 3): GET /temporal-self/now?organ=prime, a TemporalNowV1 as JSON. None: not
    # read (the now is off -- 404 -- or unreachable); the section is not shown and nothing can be cited.
    orion_now: Optional[dict[str, Any]] = None
    # Hunger (integration Part B, orion/prime/hunger.py ``frame_lines``): the needs' content lines, strongest first.
    # None: PRIME_HUNGER_NEEDS_ENABLED is off and the section is not shown.
    needs: Optional[list[str]] = None


def now_items(inputs: "FrameInputs") -> list[dict[str, Any]]:
    """The now's items as shown in the frame (at most NOW_MAX_ITEMS, each with an event_id)."""
    items = inputs.orion_now.get("items") if isinstance(inputs.orion_now, dict) else None
    if not isinstance(items, list):
        return []
    return [i for i in items if isinstance(i, dict) and str(i.get("event_id") or "").strip()][:NOW_MAX_ITEMS]


def shows_now(inputs: "FrameInputs") -> bool:
    """The now was read: the section is in the frame (possibly saying nothing new)."""
    return isinstance(inputs.orion_now, dict) and bool(inputs.orion_now.get("shown_id"))


def shown_experience_ids(inputs: "FrameInputs") -> Optional[list[str]]:
    """The event ids this frame showed; None when the now was not read (no cite can be valid)."""
    return [str(i["event_id"]).strip() for i in now_items(inputs)] if shows_now(inputs) else None


def shows_organ_replies(inputs: "FrameInputs") -> bool:
    return bool(inputs.organ_replies_on or (inputs.organ_replies is not None and inputs.organ_replies.replies))


def _num(value: Any) -> float:
    """A number from a payload field, 0.0 when absent or not numeric (a bad field never crashes a thought)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if out == out else 0.0


def _age(now: datetime, iso: Any) -> str:
    try:
        then = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return "age unknown"
    secs = max(0, int((now - then).total_seconds()))
    return f"{secs // 60} min ago" if secs < 7200 else f"{secs // 3600} h ago"


def _body_lines(inputs: FrameInputs) -> list[str]:
    if inputs.body is None:
        return [f"(unavailable: {inputs.body_error or 'not read'})"]
    body = inputs.body
    lines = [f"(read {int(_num(body.get('age_sec')))} s after the body last measured itself)"]
    strain = body.get("strain") or {}
    if strain.get("why"):
        lines.append(f"strain: {strain['why']}")
    prices = body.get("prices") or {}
    # Only the constraints Prime's thinking touches: heat and energy (the whole body), its own lane, and
    # Juniper's attention. The other lanes are other organs' business.
    for name in ("heat", "energy", f"lane:{inputs.bid_lane}", "attention:juniper"):
        price = prices.get(name)
        if isinstance(price, dict) and price.get("why"):
            lines.append(f"{name} (price {_num(price.get('value')):.2f}): {price['why']}")
        else:
            lines.append(f"{name}: (unavailable: not in the body state)")
    return lines


# Why Prime gets no per-organ cost (2026-11-04): over 30 days the act ledger's per-act dollar costs summed
# to well under $1 against about $25/month of activity cost (curiosity $0.17 on 23 of 81 acts; painting,
# reading and Prime itself none). Per-organ figures would be ~50x wrong; Prime had already blamed the
# activity cost on "attention:juniper" (Juniper's message cap) for lack of anything better.
NO_PER_ORGAN_COST = ("per-organ cost is not measured: acts carry only a few percent of the activity cost, so "
                     "slowing one organ saves an unknown, probably small amount; the idle floor is not touched "
                     "by slowing anything")


def _energy_lines(inputs: FrameInputs) -> list[str]:
    """Each machine's idle cost and its draw above idle now (GET /physiology/report ``hosts``), plus why there
    is no per-organ cost. The month's totals (always-on body per month, activity pace) are NOT repeated here:
    the BODY section's energy line already states them, computed by the producer (orion/physiology/budget.py).
    Recomputing them from ``money.floor_usd`` is wrong: that is month-to-date plus idling for the hours left,
    not a month of idling (review of PR #3120)."""
    if inputs.energy is None:
        return [f"(unavailable: {inputs.energy_error or 'not read'})"]
    lines: list[str] = []
    for h in inputs.energy.get("hosts") or []:
        if not isinstance(h, dict) or not h.get("host"):
            continue
        rest, now = h.get("rest_usd_per_month"), h.get("usd_per_month_at_this_rate")
        if isinstance(rest, (int, float)) and isinstance(now, (int, float)):
            lines.append(f"- {h['host']}: idling costs about ${rest:.0f} a month; right now it runs "
                         f"{_num(h.get('above_rest_w')):.0f} W above idle, about ${now:.1f} a month at this rate")
    if not lines:
        lines.append("(no machine figures in the report)")
    lines.append("the month's totals (always-on body vs activity pace) are in the energy line under BODY")
    lines.append(NO_PER_ORGAN_COST)
    return lines


def _attention_lines(inputs: FrameInputs) -> list[str]:
    if inputs.temporal is None:
        return [f"(unavailable: {inputs.temporal_error or 'not read'})"]
    t = inputs.temporal
    lines = [f"day phase: {t.get('day_phase') or 'unknown'}"]
    active = t.get("active_arc") or {}
    if active:
        lines.append(
            f"now: {active.get('subject_label') or active.get('subject_ref')} "
            f"(arc_id {active.get('arc_id')}, kind {active.get('kind')}, dwell {int(_num(active.get('cumulative_dwell_sec')))} s)"
        )
    for arc in (t.get("arcs_today") or [])[:8]:
        label = arc.get("subject_label") or arc.get("subject_ref") or "?"
        lines.append(f"- arc_id {arc.get('arc_id')}: {arc.get('kind')} {label!r}, {arc.get('status')}, "
                     f"returned {arc.get('attention_returns', 0)}x, last seen {_age(inputs.now, arc.get('last_seen_at'))}")
    return lines


def event_ids(inputs: FrameInputs) -> list[str]:
    """The event ids (event_ref) a concern may name: the funded event wants this frame shows."""
    return [str(w.get("event_ref")) for w in (inputs.events or {}).get("wants") or [] if w.get("event_ref")]


def _event_lines(inputs: FrameInputs) -> list[str]:
    if inputs.events is None:
        return [f"(unavailable: {inputs.events_error or 'not read'})"]
    wants = inputs.events.get("wants") or []
    if not wants:
        return ["(none: nothing the body economy funded is open now)"]
    lines = []
    for w in wants:
        sal = w.get("salience")
        unusual = f"how unusual {_num(sal):.2f}" if sal is not None else "how unusual unknown"
        lines.append(f"- event_id {w.get('event_ref')}: {w.get('label') or '(no words recorded)'} "
                     f"({w.get('family')}; {unusual}; market value {_num(w.get('net')):.2f}; "
                     f"funded {_age(inputs.now, w.get('decided_at'))})")
    funded = inputs.events.get("funded_open")
    if isinstance(funded, int) and funded > len(wants):
        lines.append(f"({funded - len(wants)} more funded events not shown)")
    return lines


def _curiosity_lines(inputs: FrameInputs) -> list[str]:
    """Whether a question waits for curiosity's next run, from the cheap read only (#3366); the gap when it
    failed, so the frame and the move check agree. Curiosity's runs arrive through the now (patch 3)."""
    if inputs.pending_focus_error:
        return [f"(whether a focus is waiting for the next run is unknown: {inputs.pending_focus_error})"]
    if not inputs.pending_focus_read:
        return ["(unavailable: whether a focus is waiting was not read)"]
    pending = inputs.pending_focus
    if not isinstance(pending, dict):
        return ["no focus is waiting for the next run"]
    return [f"a focus is already waiting for the next run ({pending.get('origin')}): {pending.get('question')!r}"]


def _num_or_none(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


NOW_LINE_REFS = 3  # related refs shown per item (the item carries up to 8)


def _now_lines(inputs: FrameInputs) -> list[str]:
    """One line per experience: its id to cite, organ, age, label, salience and surprise when known, refs."""
    lines: list[str] = []
    for item in now_items(inputs):
        sal, nats = _num_or_none(item.get("salience")), _num_or_none(item.get("surprise_nats"))
        scores = [f"salience {sal:.2f}" if sal is not None else "salience unknown"]
        if nats is not None:
            scores.append(f"surprise {nats:.2f} nats")
        label = " ".join(str(item.get("label") or "").split()) or f"({item.get('source_kind') or 'no label'})"
        refs = [str(r) for r in (item.get("related_refs") or []) if r][:NOW_LINE_REFS]
        lines.append(f"- [experience:{str(item['event_id']).strip()}] {item.get('organ') or '?'} "
                     f"{_age(inputs.now, item.get('occurred_at'))}: {label} ({'; '.join(scores)})"
                     + (f" refs {', '.join(refs)}" if refs else ""))
    if not lines:
        return ["(nothing new from the other organs since your last thought)"]
    dropped = int(_num((inputs.orion_now or {}).get("dropped_by_organ_cap"))) + int(
        _num((inputs.orion_now or {}).get("dropped_by_total_cap")))
    if dropped > 0:
        lines.append(f"({dropped} more were recorded and not shown: at most 2 per organ, {NOW_MAX_ITEMS} in all)")
    return lines


def _concern_line(c: PrimeConcernV1, now: datetime, *, take_on: bool = False,
                  no_take_on: frozenset[str] = frozenset(), evidence_clock: bool = False) -> str:
    graded = {"settled": "; already settled", "lapsed": "; lapsed, no longer graded"}.get(c.graded or "", "")
    if take_on and c.graded != "settled" and c.concern_id not in no_take_on and can_take_on(c):
        graded += "; you can take this on"
    ref = c.event_ref or ""
    about = (f"from turn {ref[len('turn:'):]}" if ref.startswith("turn:")
             else f"event {ref}" if ref else f"arc {c.arc_id or 'none'}")
    line = (f"- concern_id {c.concern_id}: {c.text} ({about}, held since "
            f"{_age(now, c.opened_at.isoformat())}, {'last touched' if evidence_clock else 'last kept'} "
            f"{_age(now, c.last_evidence_at.isoformat())}{graded})")
    # A promise concern's own sentence (orion/prime/handback.py promise_text): only here, never in the Prime block.
    return f"{line}\n  PROMISE: {c.promise_text}" if c.promise_text else line


def _regulation_lines(state: PrimeStateV1, inputs: FrameInputs) -> list[str]:
    if inputs.receptors_error:
        return [f"(unavailable: {inputs.receptors_error}; you cannot signal any organ now)"]
    lines = [f"- {fam}: {', '.join(dirs)}" for fam, dirs in sorted((inputs.receptors or {}).items()) if dirs]
    if not lines:
        lines = ["(no organ declares a receptor)"]
    lines.append("Your signals not yet graded:")
    if not state.open_regulations:
        lines.append("(none)")
    for fam, rest in sorted(state.regulation_rest_until.items()):
        if rest > inputs.now:
            lines.append(f"- {fam} rests from your signals until {rest.strftime('%H:%MZ')}")
    for r in state.open_regulations:
        arm = "held back as a control, not applied" if r.arm == "control" else "applied"
        effects = inputs.regulation_effects.get(r.signal_id)
        seen = ("(the market's record was not read)" if effects is None else describe_effects(r, effects))
        lines.append(f"- {r.direction} to {r.family} ({arm}), in force until {r.expires_at.strftime('%H:%MZ')}: {seen}")
    return lines


ORGAN_REPLIES_WINDOW_H = 48
# The replica writes its view on every 15 s save; older than this, it has probably stopped.
ORGAN_REPLIES_STALE_SEC = 900.0


def _organ_reply_lines(state: PrimeStateV1, inputs: FrameInputs) -> list[str]:
    """Challenges and refusals of Prime's acts in the last 48 h, with the organ's reason. Held-back ones are
    never in the view. A view that could not be read, or was never published, is a gap, not calm."""
    if inputs.organ_replies_error:
        return [f"(unavailable: {inputs.organ_replies_error})"]
    view = inputs.organ_replies
    if view is None:
        return [f"(unavailable: {REPLICA_ACTOR} has published no replies view)"]
    asked = {n.act_id: n.question for n in state.open_nominations}
    lines = []
    age = (inputs.now - view.as_of).total_seconds()
    if age > ORGAN_REPLIES_STALE_SEC:
        lines.append(f"(the organ's view was last written {_age(inputs.now, view.as_of.isoformat())}: "
                     f"{REPLICA_ACTOR} may not be running, so a recent reply may be missing)")
    for r in sorted(view.replies, key=lambda r: r.at, reverse=True):
        if (inputs.now - r.at).total_seconds() > ORGAN_REPLIES_WINDOW_H * 3600:
            continue
        what = "question" if r.target_kind == "focus" else "signal"
        about = asked.get(r.prime_act_id) or r.about
        did = ("refused it: the question was not shown to the run" if r.stance == "refuse" and r.target_kind == "focus"
               else "refused it: the charge no longer applies" if r.stance == "refuse"
               else "challenged it (it still went ahead)")
        lines.append(f"- {_age(inputs.now, r.at.isoformat())}, {REPLICA_ACTOR} {did}; your {what} "
                     f"{about[:160]!r} (act {r.prime_act_id}). Its reason: {r.reason}")
    if not any(line.startswith("- ") for line in lines):
        lines.append(f"(no challenge or refusal from an organ in the last {ORGAN_REPLIES_WINDOW_H} h)")
    return lines


TENDING_NONE_YET = "(none yet: no tendency rests on 10 weighted choices)"


def _tending_lines(inputs: FrameInputs) -> list[str]:
    if inputs.tending_error:
        return [f"(unavailable: {inputs.tending_error})"]
    return [f"- {line}" for line in inputs.tending or []] or [TENDING_NONE_YET]


def build_frame(state: PrimeStateV1, inputs: FrameInputs) -> str:
    """The user prompt: a few thousand characters, every section marked with its freshness."""
    sections = [
        "WHAT CHANGED SINCE YOUR LAST THOUGHT:\n" + ("\n".join(f"- {c}" for c in state.pending_changes) or "(nothing)"),
        *(["WHAT IS PULLING AT YOU (your needs, strongest first):\n" + "\n".join(inputs.needs)]
          if inputs.needs is not None else []),
        "YOUR CONCERNS:\n" + ("\n".join(_concern_line(c, inputs.now, take_on=inputs.take_on,
                                                       no_take_on=frozenset(inputs.no_take_on),
                                                       evidence_clock=inputs.evidence_clock)
                                         for c in state.concerns) or "(none)"),
        "HOW YOUR CONCERNS HAVE ENDED:\n" + track_record(state),
        "YOUR LAST READING:\n" + (state.last_reading or "(none yet)"),
        "WHAT HAPPENED TO YOUR EARLIER MOVES:\n" + ("\n".join(f"- {o}" for o in state.recent_outcomes) or "(none yet)"),
        *(["WHAT YOU TOOK ON (concerns you chose to carry out):\n" + "\n".join(inputs.intentions)]
          if inputs.intentions is not None else []),
        *(["TURNS THAT HANDED BACK TO YOU (finished chat, curiosity, consult and letter turns, newest first):\n"
           + "\n".join(inputs.turns)] if inputs.turns is not None else []),
        "QUESTIONS YOU ASKED THAT ARE STILL OPEN:\n" + ("\n".join(
            f"- {n.question!r} (asked {_age(inputs.now, n.opened_at.isoformat())}"
            f"{'; held back as a control, not sent' if n.arm == 'control' else ''})"
            for n in state.open_nominations) or "(none)"),
        ("EVENTS WORTH EFFORT (funded by Orion's body economy, highest value first):\n" + "\n".join(_event_lines(inputs))
         if inputs.event_mode else "ATTENTION (Temporal Self, today):\n" + "\n".join(_attention_lines(inputs))),
        "BODY:\n" + "\n".join(_body_lines(inputs)),
        "ENERGY MONEY:\n" + "\n".join(_energy_lines(inputs)),
        "CURIOSITY (whether a question waits for its next run):\n" + "\n".join(_curiosity_lines(inputs)),
    ]
    if shows_now(inputs):
        sections.append("WHAT YOUR ORGANS LIVED SINCE YOUR LAST THOUGHT (other organs, highest salience first; cite "
                        "the ones you use):\n" + "\n".join(_now_lines(inputs)))
    if inputs.receptors is not None or inputs.receptors_error:
        sections.append("ORGANS YOU CAN SIGNAL (family: directions it handles):\n"
                        + "\n".join(_regulation_lines(state, inputs)))
    if inputs.tending is not None or inputs.tending_error:
        sections.append("WHAT YOU TEND TO CHOOSE (from your past choices; a record, not an instruction):\n"
                        + "\n".join(_tending_lines(inputs)))
    if shows_organ_replies(inputs):
        sections.append(f"ORGAN REPLIES (last {ORGAN_REPLIES_WINDOW_H} h, newest first):\n"
                        + "\n".join(_organ_reply_lines(state, inputs)))
    return f"Now: {inputs.now.isoformat()}\n\n" + "\n\n".join(sections)


def parse_answer(parsed: Optional[dict[str, Any]]) -> tuple[Optional[PrimeAnswerV1], Optional[str]]:
    if not isinstance(parsed, dict):
        return None, "no_json_object"
    try:
        return PrimeAnswerV1.model_validate(parsed), None
    except ValidationError as exc:
        return None, f"invalid_answer: {str(exc)[:200]}"


@dataclass
class Applied:
    state: PrimeStateV1
    added: list[str]
    dropped: list[str]
    kept: list[str]
    warnings: list[str]
    move: Optional[Union[CuriosityFocusMoveV1, RegulateMoveV1, TakeOnMoveV1]]
    move_refusal: Optional[str]
    # The move's references, checked for shape; the runtime still resolves each before sending.
    refs: list[RefV1] = field(default_factory=list)
    # Event ids of experiences this thought cited that the frame showed (at most NOW_MAX_CITES, in cite order).
    experience_cites: list[str] = field(default_factory=list)
    # Experience ids the answer cited that this frame did not show (dropped, never recorded). Named early in the
    # act note: 2026-11-20 one cite was dropped and the 500-char note cut the warning before its id (#3425).
    experience_unshown: list[str] = field(default_factory=list)


EXPERIENCE_KIND = "experience"


def parse_refs(raw: Any) -> tuple[list[RefV1], list[str], list[str]]:
    """(openable refs, experience cites, plain-words problems). ``experience`` refs (``{"kind": "experience",
    "id": <event_id>}``) are split off first: they name a row of Orion's now and are recorded with
    POST /temporal-self/now/use, never opened by a curiosity run. Everything else goes to orion/refs.py's
    parse_refs (kinds arc and act, the move's refs)."""
    if raw in (None, []):
        return [], [], []
    if not isinstance(raw, list):
        return [], [], ["refs must be a list of {kind, id}"]
    rest, cites, exp_problems = _split_experience(raw)
    refs, more = parse_openable_refs(rest)
    return refs, cites, exp_problems + more


def _split_experience(raw: list[Any]) -> tuple[list[Any], list[str], list[str]]:
    """(non-experience items, experience cites, experience-ref problems)."""
    rest: list[Any] = []
    cites: list[str] = []
    problems: list[str] = []
    for item in raw:
        if isinstance(item, Mapping) and item.get("kind") == EXPERIENCE_KIND:
            eid = item.get("id")
            eid = eid.strip() if isinstance(eid, str) else ""
            if eid.startswith(f"{EXPERIENCE_KIND}:"):
                eid = eid[len(EXPERIENCE_KIND) + 1:].strip()
            if not eid or len(eid) > 400 or set(item) - {"kind", "id"}:
                problems.append(f"experience ref {json.dumps(item, default=str)[:120]} is not usable "
                                "(write {\"kind\": \"experience\", \"id\": \"<id>\"})")
            elif eid not in cites:
                cites.append(eid)
        else:
            rest.append(item)
    return rest, cites, problems


def _split_concern_refs(raw: list[Any], concerns: Sequence[PrimeConcernV1]) -> tuple[list[Any], list[str]]:
    """(other items, concern_ids): a ref whose id is a held concern's event_ref (a letter thread
    ``orion_day_letter:<date>#thread<i>``, a ``turn:`` ref, a funded event) names that concern, not something a
    curiosity run can open. Split off whatever kind it was written as (11-16 Prime wrote one as an act, #3609)."""
    by_event = {c.event_ref: c.concern_id for c in concerns if c.event_ref}
    rest: list[Any] = []
    named: list[str] = []
    for item in raw:
        rid = item.get("id") if isinstance(item, Mapping) else None
        cid = by_event.get(rid.strip()) if isinstance(rid, str) else None
        if cid is None:
            rest.append(item)
        else:
            named.append(cid)
    return rest, named


def shown_cites(cites: Sequence[str], shown: Optional[Sequence[str]]) -> tuple[list[str], list[str]]:
    """(cites to record, warnings): only ids the frame showed, in cite order, at most NOW_MAX_CITES."""
    if not cites:
        return [], []
    if shown is None:
        return [], [f"{len(cites)} experience cite(s) ignored: the now was not read this thought"]
    allowed = set(shown)
    kept = [c for c in cites if c in allowed]
    unknown = [c for c in cites if c not in allowed]
    warnings = []
    if unknown:
        warnings.append(f"experience cite(s) not shown in this frame, not recorded: {', '.join(u[:80] for u in unknown[:3])}")
    if len(kept) > NOW_MAX_CITES:
        warnings.append(f"cited {len(kept)} experiences; only the first {NOW_MAX_CITES} are recorded")
    return kept[:NOW_MAX_CITES], warnings


def regulate_refusal(move: RegulateMoveV1, state: PrimeStateV1,
                     receptors: Optional[Mapping[str, Sequence[str]]], now: datetime) -> Optional[str]:
    """Why a regulate move cannot go, or None. Never silently dropped: the reason is shown to Prime."""
    if receptors is None:
        return "regulation is off or the organs' receptors could not be read; no signal was sent"
    if move.direction not in pc.DIRECTIONS:
        return f"direction {move.direction[:40]!r} is not a signal any organ handles (slow or quicken)"
    if move.family not in receptors or not receptors.get(move.family):
        return (f"{move.family[:120]} declares no receptors, so nothing would handle the signal; "
                f"organs that do: {', '.join(sorted(f for f, d in receptors.items() if d)) or 'none'}")
    if move.direction not in receptors[move.family]:
        return f"{move.family} handles {', '.join(receptors[move.family])}, not {move.direction}"
    open_one = next((r for r in state.open_regulations if r.family == move.family), None)
    if open_one is not None:
        return (f"my {open_one.direction} signal to {move.family} is not graded yet (in force until "
                f"{open_one.expires_at.strftime('%H:%MZ')}, graded after); one signal per organ at a time")
    rest = state.regulation_rest_until.get(move.family)
    if rest is not None and now < rest:
        return (f"{move.family} rests from my signals until {rest.strftime('%H:%MZ')} (as long as my last "
                "signal to it lasted): my charge may hold an organ at most half the time")
    return None


def apply_answer(
    state: PrimeStateV1, answer: PrimeAnswerV1, *, now: datetime, arc_ids: list[str], focus_pending: Optional[bool],
    receptors: Optional[Mapping[str, Sequence[str]]] = None, event_ids: Optional[list[str]] = None,
    event_wants: Optional[Mapping[str, Mapping[str, Any]]] = None,
    take_on_check: Optional[Callable[[TakeOnMoveV1, PrimeStateV1], Optional[str]]] = None,
    focus_unread: Optional[str] = None,
    shown_experience: Optional[Sequence[str]] = None,
    evidence_clock: bool = False,
) -> Applied:
    """Concern edits, then the move check. Never trusts an id the frame did not show.
    ``receptors``: family -> directions it receives; None = regulation off (a regulate move is refused).
    ``event_ids``: event mode is on; a new concern must name one of these funded events (else not added).
    ``event_wants``: event_ref -> its listed want (``orion.prime.choice.listed_wants``); a new concern records
    the want's family and ``care_promoted`` (personality rev 4 choice point).

    ``take_on_check(move, state after the concern edits)``: why a take-on cannot go, or None; None as the
    argument = taking on is off (a take_on move is refused, and shown).

    ``focus_pending``: None = unknown, and an ask move is refused (never fail open). ``focus_unread``: the
    refusal sentence for that case when the runtime knows why (a timed-out read says "timed out", #3366).

    ``shown_experience``: the event ids this frame showed under WHAT YOUR ORGANS LIVED (None: the now was not
    read). Experience cites (the answer's top-level refs, and any inside a move's refs) are kept only when shown,
    at most NOW_MAX_CITES; the rest are warnings. A bad cite never refuses the move or fails the thought.

    ``evidence_clock`` (PRIME_CONCERN_EVIDENCE_CLOCK_ENABLED, #3574): a keep is recorded (``Applied.kept``) but no
    longer renews ``last_evidence_at``. Only real evidence does, and the runtime applies it after the move is
    actually sent (orion/prime/state.py ``touch_concern``): a refused move renews nothing. Off: a keep renews."""
    warnings: list[str] = []
    concerns = list(state.concerns)
    by_id = {c.concern_id for c in concerns}
    dropped: list[str] = []
    for drop in answer.concerns.drop:
        gone = next((c for c in concerns if c.concern_id == drop.concern_id), None)
        if gone is None:
            # Unknown, or listed twice in one answer (the first drop already removed it).
            if drop.concern_id not in by_id:
                warnings.append(f"drop of unknown concern {drop.concern_id!r} ignored")
            continue
        state = close_concern(state, gone, end="dropped", now=now,
                              note=f"dropped unresolved: {drop.why[:200] or 'no reason given'}")
        concerns = [c for c in concerns if c.concern_id != drop.concern_id]
        dropped.append(f"{drop.concern_id} ({drop.why[:120] or 'no reason given'})")
    kept_ids = set(answer.concerns.keep) & {c.concern_id for c in concerns}
    for cid in set(answer.concerns.keep) - {c.concern_id for c in concerns}:
        warnings.append(f"keep of unknown concern {cid!r} ignored")
    if not evidence_clock:
        concerns = [c.model_copy(update={"last_evidence_at": now}) if c.concern_id in kept_ids else c
                    for c in concerns]
    added: list[str] = []
    for add in answer.concerns.add:
        if len(concerns) >= MAX_CONCERNS:
            warnings.append(f"concern not added, already holding {MAX_CONCERNS}: {add.text[:80]!r}")
            continue
        if event_ids is not None:
            if add.event_id not in event_ids:
                warnings.append(f"concern not added, it names no funded event shown in the frame "
                                f"(event_id {add.event_id!r}): {add.text[:80]!r}")
                continue
            if any(c.event_ref == add.event_id for c in concerns):
                warnings.append(f"concern not added, a concern already holds event {add.event_id!r}: {add.text[:80]!r}")
                continue
            listed = (event_wants or {}).get(add.event_id) or {}
            family = str(listed.get("family") or "").strip()[:200] or None
            promoted = listed.get("care_promoted") if isinstance(listed.get("care_promoted"), bool) else None
            concern = PrimeConcernV1(concern_id=new_id(), text=add.text, event_ref=add.event_id, opened_at=now,
                                     last_evidence_at=now, event_family=family, event_care_promoted=promoted)
            concerns.append(concern)
            added.append(concern.concern_id)
            continue
        arc_id = add.arc_id if add.arc_id in arc_ids else None
        if add.arc_id and arc_id is None:
            warnings.append(f"arc_id {add.arc_id!r} was not in the frame; concern kept without an arc")
        concern = PrimeConcernV1(concern_id=new_id(), text=add.text, arc_id=arc_id, opened_at=now, last_evidence_at=now)
        concerns.append(concern)
        added.append(concern.concern_id)

    top_refs, cited, ref_problems = parse_refs(answer.refs)
    if ref_problems:
        warnings.append("refs: " + "; ".join(ref_problems)[:300])
    if top_refs:
        warnings.append(f"top-level refs only record experiences; {len(top_refs)} arc/act ref(s) ignored "
                        "(a move's refs carry those)")
    move, refusal, refs = answer.move, None, []
    if isinstance(move, TakeOnMoveV1):
        edited = state.model_copy(update={"concerns": concerns})
        refusal = ("taking a concern on is switched off" if take_on_check is None
                   else take_on_check(move, edited))
        if refusal is None:
            move = move.model_copy(update={"why": move.why[:600]})
    elif isinstance(move, RegulateMoveV1):
        refusal = regulate_refusal(move, state, receptors, now)
        if refusal is None:
            move = move.model_copy(update={"hours": clamp_hours(move.hours), "why": move.why[:600]})
    elif move is not None:
        # A bad experience ref in a move is a warning (never refuses the move); a bad arc/act ref refuses it.
        concern_refs: list[str] = []
        if isinstance(move.refs, list):
            rest, move_cites, exp_problems = _split_experience(move.refs)
            rest, concern_refs = _split_concern_refs(rest, concerns)
            refs, problems = parse_openable_refs(rest)
        else:
            move_cites, exp_problems = [], []
            refs, problems = parse_openable_refs(move.refs)
        if exp_problems:
            warnings.append("move refs: " + "; ".join(exp_problems)[:300])
        cited.extend(c for c in move_cites if c not in cited)
        stray = id_tokens_not_in(f"{move.question}\n{move.why}", refs)
        cited_tokens = {t.lower() for c in move_cites for t in id_tokens_not_in(c, [])}
        if not problems and concern_refs:
            # A held concern's event (a letter thread, a turn, a funded event) has no openable ref: the move's
            # concern_id is how a move points at it (#3609). Unambiguous -> the move takes that concern_id.
            agrees = move.concern_id in (None, concern_refs[0]) or (
                move.concern_id == NEW_CONCERN and concern_refs[0] in added)
            if len(set(concern_refs)) > 1 or not agrees:
                problems = [f"refs name the event of concern {', '.join(sorted(set(concern_refs)))}, but the move's "
                            f"concern_id is {move.concern_id!r}; a concern's event has no openable ref, point the "
                            "move at it with its concern_id and leave it out of refs"]
            else:
                move = move.model_copy(update={"concern_id": concern_refs[0]})
        if problems:
            refusal = "; ".join(problems)
        elif stray and any(t in cited_tokens for t in stray):
            refusal = ("the question quotes the id of an experience it cites; a curiosity run cannot open an "
                       "experience id. Describe what happened in words (the cite is still recorded)")
        elif stray:
            refusal = (f"the question cites {', '.join(stray[:3])} outside refs; a curiosity run cannot open "
                       "an id written in the text. Put arcs and acts in refs, and no other ids")
        if refusal is not None:
            pass
        elif state.open_nominations:
            refusal = "one of my questions to curiosity is still open"
        elif focus_pending is None:
            refusal = focus_unread or "could not read whether curiosity already has a focus waiting"
        elif focus_pending:
            refusal = "curiosity already has a focus waiting for its next run"
        if refusal is None and move.concern_id == NEW_CONCERN:
            move = move.model_copy(update={"concern_id": added[0] if added else None})
            if not added:
                warnings.append('move cites "new" but no concern was added; kept without it')
        if refusal is None and move.concern_id and move.concern_id not in {c.concern_id for c in concerns}:
            warnings.append(f"move cites unknown concern {move.concern_id!r}; kept without it")
            move = move.model_copy(update={"concern_id": None})
        if refusal is None and concern_refs:
            warnings.append(f"move refs named the event of concern {concern_refs[0]}; used it as the move's "
                            "concern_id (a concern's event has no openable ref)")
    new_state = state.model_copy(update={
        "concerns": concerns,
        "last_reading": answer.reading[:READING_CAP],
        "pending_changes": [],
        "last_thought_at": now,
        "thoughts_total": state.thoughts_total + 1,
    })
    experience_cites, cite_warnings = shown_cites(cited, shown_experience)
    warnings.extend(cite_warnings)
    unshown = [c for c in cited if shown_experience is None or c not in set(shown_experience)]
    return Applied(new_state, added, dropped, sorted(kept_ids), warnings, None if refusal else move, refusal,
                   [] if refusal else refs, experience_cites, unshown)


WOKE_NOTE_CAP = 160


def woke_note(changes: list[str]) -> str:
    """Why the thought woke, first on the act's close note so the 500-char note cap never cuts it.
    Lets the wake mix be measured from the ledger instead of inferred (2026-11-04)."""
    text = "; ".join(" ".join(c.split()) for c in changes) or "nothing pending"
    return "woke: " + (text if len(text) <= WOKE_NOTE_CAP else text[:WOKE_NOTE_CAP - 3] + "...")


def summary_note(applied: Applied, *, now_use: Optional[str] = None) -> str:
    """The thought act's close note: what the thought changed (concerns are never edited silently)."""
    # Most important first: the act note is cut at 500 chars (orion/grammar/acts.py) after a ~170-char
    # "woke:" prefix, so the move goes before the unbounded drop reasons and warnings.
    parts = [f"concerns +{len(applied.added)} -{len(applied.dropped)} kept {len(applied.kept)}"]
    if isinstance(applied.move, RegulateMoveV1):
        m = applied.move
        parts.append(f"move regulate {m.direction} {m.family} for {m.hours:g} h: {m.why[:120]!r}")
    elif isinstance(applied.move, TakeOnMoveV1):
        m = applied.move
        when = f" not before {m.not_before}" if m.not_before else ""
        parts.append(f"move take_on {m.concern_id}{when}: {m.why[:120]!r}")
    elif applied.move is not None:
        cites = f" refs {', '.join(f'{r.kind}:{r.id}' for r in applied.refs)}" if applied.refs else ""
        parts.append(f"move curiosity_focus {applied.move.question[:120]!r}{cites}")
    elif applied.move_refusal:
        parts.append(f"move refused: {applied.move_refusal}")
    else:
        parts.append("no move")
    if applied.experience_cites:
        parts.append(f"proposed {len(applied.experience_cites)} experience cite(s)"
                     + (f", now_use {now_use}" if now_use else ""))
    if applied.experience_unshown:
        # Before the drop reasons and warnings, so the note's 500-char cut never hides which id did not match.
        parts.append(f"{len(applied.experience_unshown)} cite(s) not shown: "
                     + ", ".join(u[:60] for u in applied.experience_unshown[:2]))
    if applied.dropped:
        parts.append("dropped " + "; ".join(applied.dropped))
    if applied.warnings:
        parts.append("warnings: " + "; ".join(applied.warnings))
    return " | ".join(parts)


def answer_json(answer: PrimeAnswerV1) -> str:
    return json.dumps(answer.model_dump(mode="json"), ensure_ascii=False)
