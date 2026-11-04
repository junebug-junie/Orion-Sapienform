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

References (orion/refs.py): a move cites ids only through ``refs`` (``[{kind, id}]``, kinds arc and
act). Refused here when a ref is malformed, or when the question or why contains an id-looking word
(``orion.refs.ID_TOKEN_RE``) that is not one of the refs; refused by the runtime when a ref does not
open with its resolver. A curiosity run opens each ref with ``python -P -m orion.dig <kind> <id>``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orion.physiology import prime_charge as pc
from orion.prime.acts import CONCERN_HORIZON_SEC, clamp_hours, describe_effects
from orion.prime.state import (
    CONCERN_TEXT_CAP,
    MAX_CONCERNS,
    PrimeConcernV1,
    PrimeStateV1,
    close_concern,
    new_id,
    track_record,
)
from orion.refs import RefV1, id_tokens_not_in, parse_refs
from orion.schemas.introspect import FOCUS_TEXT_CAP

READING_CAP = 600
# A move may cite the concern added in the same answer, whose id the runtime assigns afterwards.
NEW_CONCERN = "new"


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
    # concern_ids this thought keeps: the only thing that renews a concern (it fades after a day otherwise).
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


Move = Annotated[Union[CuriosityFocusMoveV1, RegulateMoveV1], Field(discriminator="kind")]


class PrimeAnswerV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    reading: str = Field(default="", max_length=4000)
    concerns: ConcernEditsV1 = Field(default_factory=ConcernEditsV1)
    move: Optional[Move] = None


SYSTEM_PROMPT = """You are Orion's slow, between-conversations thinking ("Prime"). You are not talking to anyone.

You hold at most 3 concerns: unfinished things worth keeping in mind. Each thought you:
1. read what changed and say, briefly, how you read the situation now;
2. edit your concerns: keep the ones still worth holding (by concern_id), add one that matters, drop one
   that no longer does, with why. A concern you do not keep fades after a day;
3. choose at most one move, or none.

The one move you have today: ask Orion's curiosity to look into one question in its next regular investigate run.
Curiosity decides for itself: it may explore your question or set it aside, and you will see which.
While a question of yours is still open (asked, not yet answered), you cannot ask another.
Some of your questions are held back and not sent, at random, so Orion can tell whether your questions change anything.
Only ask when a question would actually help with a concern. "move": null is a normal, good answer.

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
- Reply with ONE JSON object and nothing else:
{"reading": "...", "concerns": {"keep": ["<concern_id>"], "add": [{"text": "...", "arc_id": null}], "drop": [{"concern_id": "...", "why": "..."}]},
 "move": null | {"kind": "curiosity_focus", "question": "...", "why": "...", "concern_id": "<concern_id> | new | null",
                 "refs": [{"kind": "arc" | "act", "id": "..."}]}}""".replace("<HORIZON_DAYS>", f"{CONCERN_HORIZON_SEC / 86400:g}")


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


def system_prompt(receptors: Optional[Mapping[str, Sequence[str]]], *, events: bool = False) -> str:
    """The base prompt, plus the regulate move only when regulation is on and some family receives it.
    ``events``: concerns come from funded event wants (event_id), not arcs."""
    base = SYSTEM_PROMPT
    if events:
        base = (base.replace(_ARC_RULE, EVENT_CONCERN_RULE).replace(_ARC_JSON, _EVENT_JSON)
                .replace(_ARC_REFS, _EVENT_REFS).replace(_ARC_REFS_JSON, _EVENT_REFS_JSON))
    if receptors and any(receptors.values()):
        return base + REGULATE_PROMPT
    return base


@dataclass
class FrameInputs:
    """What the runtime read for this thought. ``*_error`` set means that read failed."""

    now: datetime
    body: Optional[dict[str, Any]] = None
    body_error: Optional[str] = None
    temporal: Optional[dict[str, Any]] = None
    temporal_error: Optional[str] = None
    curiosity: Optional[dict[str, Any]] = None
    curiosity_error: Optional[str] = None
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
    if inputs.curiosity is None:
        return [f"(unavailable: {inputs.curiosity_error or 'not read'})"]
    c = inputs.curiosity
    lines: list[str] = []
    line = ((c.get("schedule") or {}).get("lines") or {}).get("investigate") or {}
    if line:
        lines.append(f"investigate runs today {line.get('runs_today')} of {line.get('daily_cap')}, "
                     f"next eligible {line.get('next_eligible_at')}")
    pending = ((c.get("schedule") or {}).get("pending_focus") or {}).get("focus")
    if pending:
        lines.append(f"a focus is already waiting for the next run ({pending.get('origin')}): {pending.get('question')!r}")
    for run in (c.get("runs") or [])[:4]:
        # What the run found, never its kickoff prompt (the "about" text is the brief Orion was handed).
        finding = " ".join(str(run.get("finding_text") or "").split())
        focus = (run.get("focus") or {}).get("question")
        took = f" [took a focus: {focus[:120]!r}]" if focus else ""
        lines.append(f"- run {run.get('run_id')} {run.get('status')}/{run.get('outcome_kind')} "
                     f"{_age(inputs.now, run.get('started_at_iso'))}{took}: {finding[:240] or '(no finding text)'}")
    return lines or ["(no recent investigate runs)"]


def _concern_line(c: PrimeConcernV1, now: datetime) -> str:
    graded = {"settled": "; already settled", "lapsed": "; lapsed, no longer graded"}.get(c.graded or "", "")
    about = f"event {c.event_ref}" if c.event_ref else f"arc {c.arc_id or 'none'}"
    return (f"- concern_id {c.concern_id}: {c.text} ({about}, held since "
            f"{_age(now, c.opened_at.isoformat())}, last kept {_age(now, c.last_evidence_at.isoformat())}{graded})")


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


def build_frame(state: PrimeStateV1, inputs: FrameInputs) -> str:
    """The user prompt: a few thousand characters, every section marked with its freshness."""
    sections = [
        "WHAT CHANGED SINCE YOUR LAST THOUGHT:\n" + "\n".join(f"- {c}" for c in state.pending_changes),
        "YOUR CONCERNS:\n" + ("\n".join(_concern_line(c, inputs.now) for c in state.concerns) or "(none)"),
        "HOW YOUR CONCERNS HAVE ENDED:\n" + track_record(state),
        "YOUR LAST READING:\n" + (state.last_reading or "(none yet)"),
        "WHAT HAPPENED TO YOUR EARLIER MOVES:\n" + ("\n".join(f"- {o}" for o in state.recent_outcomes) or "(none yet)"),
        "QUESTIONS YOU ASKED THAT ARE STILL OPEN:\n" + ("\n".join(
            f"- {n.question!r} (asked {_age(inputs.now, n.opened_at.isoformat())}"
            f"{'; held back as a control, not sent' if n.arm == 'control' else ''})"
            for n in state.open_nominations) or "(none)"),
        ("EVENTS WORTH EFFORT (funded by Orion's body economy, highest value first):\n" + "\n".join(_event_lines(inputs))
         if inputs.event_mode else "ATTENTION (Temporal Self, today):\n" + "\n".join(_attention_lines(inputs))),
        "BODY:\n" + "\n".join(_body_lines(inputs)),
        "ENERGY MONEY:\n" + "\n".join(_energy_lines(inputs)),
        "CURIOSITY:\n" + "\n".join(_curiosity_lines(inputs)),
    ]
    if inputs.receptors is not None or inputs.receptors_error:
        sections.append("ORGANS YOU CAN SIGNAL (family: directions it handles):\n"
                        + "\n".join(_regulation_lines(state, inputs)))
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
    move: Optional[Union[CuriosityFocusMoveV1, RegulateMoveV1]]
    move_refusal: Optional[str]
    # The move's references, checked for shape; the runtime still resolves each before sending.
    refs: list[RefV1] = field(default_factory=list)


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
) -> Applied:
    """Concern edits, then the move check. Never trusts an id the frame did not show.
    ``receptors``: family -> directions it receives; None = regulation off (a regulate move is refused).
    ``event_ids``: event mode is on; a new concern must name one of these funded events (else not added)."""
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
    concerns = [c.model_copy(update={"last_evidence_at": now}) if c.concern_id in kept_ids else c for c in concerns]
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
            concern = PrimeConcernV1(concern_id=new_id(), text=add.text, event_ref=add.event_id, opened_at=now,
                                     last_evidence_at=now)
            concerns.append(concern)
            added.append(concern.concern_id)
            continue
        arc_id = add.arc_id if add.arc_id in arc_ids else None
        if add.arc_id and arc_id is None:
            warnings.append(f"arc_id {add.arc_id!r} was not in the frame; concern kept without an arc")
        concern = PrimeConcernV1(concern_id=new_id(), text=add.text, arc_id=arc_id, opened_at=now, last_evidence_at=now)
        concerns.append(concern)
        added.append(concern.concern_id)

    move, refusal, refs = answer.move, None, []
    if isinstance(move, RegulateMoveV1):
        refusal = regulate_refusal(move, state, receptors, now)
        if refusal is None:
            move = move.model_copy(update={"hours": clamp_hours(move.hours), "why": move.why[:600]})
    elif move is not None:
        refs, problems = parse_refs(move.refs)
        stray = id_tokens_not_in(f"{move.question}\n{move.why}", refs)
        if problems:
            refusal = "; ".join(problems)
        elif stray:
            refusal = (f"the question cites {', '.join(stray[:3])} outside refs; a curiosity run cannot open "
                       "an id written in the text. Put arcs and acts in refs, and no other ids")
        if refusal is not None:
            pass
        elif state.open_nominations:
            refusal = "one of my questions to curiosity is still open"
        elif focus_pending is None:
            refusal = "could not read whether curiosity already has a focus waiting"
        elif focus_pending:
            refusal = "curiosity already has a focus waiting for its next run"
        if refusal is None and move.concern_id == NEW_CONCERN:
            move = move.model_copy(update={"concern_id": added[0] if added else None})
            if not added:
                warnings.append('move cites "new" but no concern was added; kept without it')
        if refusal is None and move.concern_id and move.concern_id not in {c.concern_id for c in concerns}:
            warnings.append(f"move cites unknown concern {move.concern_id!r}; kept without it")
            move = move.model_copy(update={"concern_id": None})
    new_state = state.model_copy(update={
        "concerns": concerns,
        "last_reading": answer.reading[:READING_CAP],
        "pending_changes": [],
        "last_thought_at": now,
        "thoughts_total": state.thoughts_total + 1,
    })
    return Applied(new_state, added, dropped, sorted(kept_ids), warnings, None if refusal else move, refusal,
                   [] if refusal else refs)


WOKE_NOTE_CAP = 160


def woke_note(changes: list[str]) -> str:
    """Why the thought woke, first on the act's close note so the 500-char note cap never cuts it.
    Lets the wake mix be measured from the ledger instead of inferred (2026-11-04)."""
    text = "; ".join(" ".join(c.split()) for c in changes) or "nothing pending"
    return "woke: " + (text if len(text) <= WOKE_NOTE_CAP else text[:WOKE_NOTE_CAP - 3] + "...")


def summary_note(applied: Applied) -> str:
    """The thought act's close note: what the thought changed (concerns are never edited silently)."""
    # Most important first: the act note is cut at 500 chars (orion/grammar/acts.py) after a ~170-char
    # "woke:" prefix, so the move goes before the unbounded drop reasons and warnings.
    parts = [f"concerns +{len(applied.added)} -{len(applied.dropped)} kept {len(applied.kept)}"]
    if isinstance(applied.move, RegulateMoveV1):
        m = applied.move
        parts.append(f"move regulate {m.direction} {m.family} for {m.hours:g} h: {m.why[:120]!r}")
    elif applied.move is not None:
        cites = f" refs {', '.join(f'{r.kind}:{r.id}' for r in applied.refs)}" if applied.refs else ""
        parts.append(f"move curiosity_focus {applied.move.question[:120]!r}{cites}")
    elif applied.move_refusal:
        parts.append(f"move refused: {applied.move_refusal}")
    else:
        parts.append("no move")
    if applied.dropped:
        parts.append("dropped " + "; ".join(applied.dropped))
    if applied.warnings:
        parts.append("warnings: " + "; ".join(applied.warnings))
    return " | ".join(parts)


def answer_json(answer: PrimeAnswerV1) -> str:
    return json.dumps(answer.model_dump(mode="json"), ensure_ascii=False)
