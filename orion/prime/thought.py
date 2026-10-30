"""One Prime thought (design section 8.2): build a short frame, parse the model's answer, apply it.

The model writes one JSON object (``PrimeAnswerV1``): how it reads things now, edits to its
concerns, and at most one move. In step 2A the only move is ``curiosity_focus``: one question for
curiosity's next regular investigate run (orion/curiosity/focus.py). "No move" is a valid answer.

The runtime, not the model, checks every move. A move is refused (and the refusal shown in the
next frame) when a question of Prime's is still open, when a focus is already pending in Hub, or
when the text is empty. A model answer that does not parse is a failed thought, never a guess.

References (orion/refs.py): a move cites ids only through ``refs`` (``[{kind, id}]``, kinds arc and
act). Refused here when a ref is malformed, or when the question or why contains an id-looking word
(``orion.refs.ID_TOKEN_RE``) that is not one of the refs; refused by the runtime when a ref does not
open with its resolver. A curiosity run opens each ref with ``python -P -m orion.dig <kind> <id>``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orion.prime.acts import CONCERN_HORIZON_SEC
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


class PrimeAnswerV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    reading: str = Field(default="", max_length=4000)
    concerns: ConcernEditsV1 = Field(default_factory=ConcernEditsV1)
    move: Optional[CuriosityFocusMoveV1] = None


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
    # Prime's own lane (PRIME_BID_LANE): its price is the one that decides Prime's thinking.
    bid_lane: str = "chat"


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
    return (f"- concern_id {c.concern_id}: {c.text} (arc {c.arc_id or 'none'}, held since "
            f"{_age(now, c.opened_at.isoformat())}, last kept {_age(now, c.last_evidence_at.isoformat())}{graded})")


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
        "ATTENTION (Temporal Self, today):\n" + "\n".join(_attention_lines(inputs)),
        "BODY:\n" + "\n".join(_body_lines(inputs)),
        "CURIOSITY:\n" + "\n".join(_curiosity_lines(inputs)),
    ]
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
    move: Optional[CuriosityFocusMoveV1]
    move_refusal: Optional[str]
    # The move's references, checked for shape; the runtime still resolves each before sending.
    refs: list[RefV1] = field(default_factory=list)


def apply_answer(
    state: PrimeStateV1, answer: PrimeAnswerV1, *, now: datetime, arc_ids: list[str], focus_pending: Optional[bool],
) -> Applied:
    """Concern edits, then the move check. Never trusts an id the frame did not show."""
    warnings: list[str] = []
    concerns = list(state.concerns)
    by_id = {c.concern_id for c in concerns}
    dropped: list[str] = []
    for drop in answer.concerns.drop:
        if drop.concern_id not in by_id:
            warnings.append(f"drop of unknown concern {drop.concern_id!r} ignored")
            continue
        gone = next(c for c in concerns if c.concern_id == drop.concern_id)
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
        arc_id = add.arc_id if add.arc_id in arc_ids else None
        if add.arc_id and arc_id is None:
            warnings.append(f"arc_id {add.arc_id!r} was not in the frame; concern kept without an arc")
        concern = PrimeConcernV1(concern_id=new_id(), text=add.text, arc_id=arc_id, opened_at=now, last_evidence_at=now)
        concerns.append(concern)
        added.append(concern.concern_id)

    move, refusal, refs = answer.move, None, []
    if move is not None:
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


def summary_note(applied: Applied) -> str:
    """The thought act's close note: what the thought changed (concerns are never edited silently)."""
    parts = [f"concerns +{len(applied.added)} -{len(applied.dropped)} kept {len(applied.kept)}"]
    if applied.dropped:
        parts.append("dropped " + "; ".join(applied.dropped))
    if applied.move is not None:
        cites = f" refs {', '.join(f'{r.kind}:{r.id}' for r in applied.refs)}" if applied.refs else ""
        parts.append(f"move curiosity_focus {applied.move.question[:120]!r}{cites}")
    elif applied.move_refusal:
        parts.append(f"move refused: {applied.move_refusal}")
    else:
        parts.append("no move")
    if applied.warnings:
        parts.append("warnings: " + "; ".join(applied.warnings))
    return " | ".join(parts)


def answer_json(answer: PrimeAnswerV1) -> str:
    return json.dumps(answer.model_dump(mode="json"), ensure_ascii=False)
