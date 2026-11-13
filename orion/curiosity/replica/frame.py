"""What the curiosity replica reads before it plans (its frame), its system prompt, and its answer's parse.

Design sections 2-3. Every section carries its age. A read that failed is shown as
``(unavailable: <why>)``, never as an empty section: a gap must not read as calm (the same rule as Prime's
frame). Sections:

* NOW, and how long the replica has to answer;
* CANDIDATES (from Hub's request, so the replica sees exactly what can be offered), in a neutral order;
* STALE PRIORS (count and ids: the run is shown them anyway, they are not pickable);
* RECENT RUNS (Hub's ``/curiosity/api/runs``, the read Prime uses, plus each run's spend-log score);
* PRIME (Prime's published concerns view, Redis ``orion:prime:concerns``, and any regulate signal in force
  on curiosity, Redis ``orion:prime:regulate:durable.curiosity_investigate``; never Prime's private state);
* PENDING FOCUS (from the request), and WHAT JUNIPER HAS BEEN TALKING ABOUT (the newest chat-history compactor
  digest: its card summary and journal title, with its age; no raw chat text);
* TRACK RECORD (its own last plans, ``orion/curiosity/replica/track_record.py``);
* YOUR ANSWERS TO PRIME (patch 4): its refusal rate over its last 20 answers, and its last answers
  (``orion/curiosity/replica/answers.py``).

A second frame, the REPLY frame (``build_reply_frame``), is for a thought woken by a new Prime act with no plan
request (design section 9, "between runs"): what Prime did, Prime's concerns, recent runs, the digest and its
own answers; the answer is one stance (``ReplyAnswerV1``).

A regulate signal is shown without its holdback arm: the replica must not be able to tell Prime's treated
signals from its controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from pydantic import ValidationError

from orion.curiosity.replica.answers import AnswerRecordV1, ReplyAnswerV1, TargetV1, answer_lines
from orion.curiosity.replica.track_record import record_lines, tested_nats
from orion.prime.state import PrimeConcernsViewV1
from orion.schemas.curiosity_replica import (
    PICKS_MAX,
    PICKS_MIN,
    CuriosityOfferPlanAnswerV1,
    CuriosityOfferPlanRequestV1,
)

RECENT_RUNS_SHOWN = 6
FINDING_CHARS = 240
DIGEST_SUMMARY_CHARS = 1600
STALE_IDS_SHOWN = 20
# Prime saves its state (and this view) every 15 s tick. Older than this: Prime is probably not running.
CONCERNS_STALE_SEC = 900.0

# What a stance does (design section 9, patch 4). Shared by the plan prompt and the reply prompt.
STANCE_RULES = """- What a stance does. It answers Prime, Orion's slow between-conversations thinking, and Prime is told.
  "support": you agree; nothing changes. "challenge": you disagree, but it still happens; Prime is shown your
  reason. "refuse": a question Prime queued is NOT shown to the run; a signal from Prime stops charging
  curiosity's bids from the next bid on. Every challenge and refusal must carry a one-sentence reason.
- You can only refuse what PRIME did. A focus Juniper asked for, or one Orion chose in a conversation, can be
  challenged but never refused.
- You answer each act of Prime's once, at most <PER_DAY> times a day, with at least <REFRACTORY_MIN> minutes between answers. Some
  challenges and refusals are held back at random (recorded, not applied), so Orion can tell whether they help.
  Your refusal rate is watched: refuse only when Prime's push would cost this run something real."""

SYSTEM_PROMPT = f"""You are curiosity's planner (the curiosity replica): a small helper inside Orion's curiosity organ.
You are not Orion. Nothing you write is shown to Orion's curiosity run; only the reasons you give for
challenging or refusing an act of Prime's are shown to Prime.

Before one of Orion's curiosity runs, you choose which of Orion's open beliefs (priors) are shown FIRST.
Orion then decides for itself what, if anything, to test. You only set the order of the top of the list;
the rest of the list, and everything else in the run, stays the same.

Your plan is scored on one number: how much Orion's beliefs actually changed in the run (belief change in
nats, summed over the priors Orion tested). A pick Orion does not test scores nothing. Orion tests the first
prior it is shown about a third of the time, and about half of all runs test nothing at all.

Answer with ONE JSON object and nothing else:
{{"reading": "one or two sentences: how you read curiosity's situation now",
 "picks": [{{"prior_id": "<an id from CANDIDATES, copied exactly>", "why": "one sentence"}}],
 "focus_stance": null,
 "signal_stance": null,
 "abstain": false}}

Rules:
- {PICKS_MIN} to {PICKS_MAX} picks, in the order you want them shown, each id from CANDIDATES exactly once.
  Copy each prior_id character for character. A plan with any bad pick is thrown away whole.
- "why": one sentence, at most 200 characters.
- "abstain": true (with "picks": []) means nothing here is worth reordering; the rule order is used.
- "focus_stance": your view of the PENDING FOCUS, if there is one: {{"stance": "support" | "challenge" |
  "refuse", "reason": "one sentence"}}. "signal_stance": the same for PRIME'S SIGNAL ON CURIOSITY, with its
  "signal_id". Both are optional; null is a normal answer.
{STANCE_RULES}
- A section marked "(unavailable: ...)" could not be read. Do not treat it as empty.
"""


def _caps(text: str, per_day: int, refractory_min: int) -> str:
    return text.replace("<PER_DAY>", str(int(per_day))).replace("<REFRACTORY_MIN>", str(int(refractory_min)))


def system_prompt(*, per_day: int = 3, refractory_min: int = 30) -> str:
    """The plan prompt; the answer caps are the live settings (CURIOSITY_REPLICA_ANSWERS_PER_DAY,
    CURIOSITY_REPLICA_ANSWER_REFRACTORY_SEC)."""
    return _caps(SYSTEM_PROMPT, per_day, refractory_min)


@dataclass
class ReplicaFrameInputs:
    """What the runtime read for this plan. ``*_error`` set means that read failed."""

    now: datetime
    request: CuriosityOfferPlanRequestV1
    runs: Optional[list[dict[str, Any]]] = None
    runs_error: Optional[str] = None
    # run_id -> its curiosity_run_outcomes row (turn_ok, realized_nats, per_prior); None: unread.
    outcomes: Optional[dict[str, Mapping[str, Any]]] = None
    outcomes_error: Optional[str] = None
    concerns: Optional[PrimeConcernsViewV1] = None
    concerns_error: Optional[str] = None
    # The regulate signal in force on curiosity: {"signal_id", "direction", "why", "expires_at"}; None = none
    # (when signal_error is None) or unread.
    signal: Optional[dict[str, Any]] = None
    signal_error: Optional[str] = None
    # {"summary", "title", "window_end", "created_at", "index"}; None = none found (digest_error None) or unread.
    digest: Optional[dict[str, Any]] = None
    digest_error: Optional[str] = None
    record_rows: Optional[list[Mapping[str, Any]]] = None
    record_error: Optional[str] = None
    fallback_counts: Optional[dict[str, int]] = None
    plans_total: Optional[int] = None
    notes: list[str] = field(default_factory=list)
    # Its own answers to Prime (orion/curiosity/replica/answers.py), from its own state.
    answers: list[AnswerRecordV1] = field(default_factory=list)
    # Whether answering Prime is on (CURIOSITY_REPLICA_ANSWERS_ENABLED); off, the section says stances are recorded only.
    answers_on: bool = True


def _aware(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _ago(now: datetime, then: Any) -> str:
    t = _aware(then)
    if t is None:
        return "at an unknown time"
    sec = (now - t).total_seconds()
    if sec < 0:
        return "just now"
    if sec < 90:
        return f"{int(sec)} s ago"
    if sec < 5400:
        return f"{int(sec // 60)} min ago"
    return f"{sec / 3600:.0f} h ago"


def _one_line(text: Any, limit: int) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _candidate_line(c: Any) -> str:
    conf = f"{c.confidence:.2f}" if c.confidence is not None else "no usable confidence"
    if c.times_tested <= 0:
        tested = "never tested"
    else:
        tested = f"tested {c.times_tested}x"
        tested += f", last {c.last_tested_age_h:.0f} h ago" if c.last_tested_age_h is not None else ", last test time unknown"
        tested += f", last move {c.last_move:+.2f}" if c.last_move is not None else ", last move unknown"
    return f"- {c.prior_id} | confidence {conf} | {tested} | {c.claim}"


def _runs_lines(inp: ReplicaFrameInputs) -> list[str]:
    if inp.runs is None:
        return [f"(unavailable: {inp.runs_error or 'not read'})"]
    runs = sorted(inp.runs, key=lambda r: r.get("started_at") or 0, reverse=True)[:RECENT_RUNS_SHOWN]
    if not runs:
        return ["(no investigate runs in the last days)"]
    lines = []
    for r in runs:
        rid = r.get("run_id")
        part = f"- {r.get('started_at_iso') or '?'} run {rid}: {r.get('outcome_kind') or r.get('status') or '?'}"
        if inp.outcomes is None:
            part += f"; score unavailable ({inp.outcomes_error or 'not read'})"
        else:
            o = inp.outcomes.get(str(rid))
            if o is None:
                part += "; not scored"
            elif o.get("realized_nats") is None:
                part += "; score unknown"
            else:
                tested = tested_nats(o)
                part += f"; belief change {float(o['realized_nats']):.3f} nats"
                part += f"; tested {', '.join(tested) if tested else 'nothing'}"
        focus = (r.get("focus") or {}).get("question") if isinstance(r.get("focus"), dict) else None
        if focus:
            part += f"; focus taken: {_one_line(focus, 160)!r}"
        finding = r.get("finding_text")
        if finding:
            part += f"; found: {_one_line(finding, FINDING_CHARS)!r}"
        lines.append(part)
    return lines


def _prime_lines(inp: ReplicaFrameInputs) -> list[str]:
    now = inp.now
    lines: list[str] = []
    if inp.concerns is None:
        lines.append(f"PRIME'S CONCERNS: (unavailable: {inp.concerns_error or 'Prime has published none'})")
    else:
        age = (now - inp.concerns.as_of).total_seconds()
        stale = " -- STALE, Prime may not be running" if age > CONCERNS_STALE_SEC else ""
        lines.append(f"PRIME'S CONCERNS (Prime's published view, {_ago(now, inp.concerns.as_of)}{stale}):")
        if not inp.concerns.concerns:
            lines.append("(Prime holds no open concern)")
        for c in inp.concerns.concerns:
            held = c.age_h + max(0.0, age) / 3600.0
            arc = f", about arc {c.arc_id}" if c.arc_id else ""
            lines.append(f"- {_one_line(c.text, 300)} (held {held:.0f} h{arc})")
    if inp.signal_error:
        lines.append(f"PRIME'S SIGNAL ON CURIOSITY: (unavailable: {inp.signal_error})")
    elif inp.signal is None:
        lines.append("PRIME'S SIGNAL ON CURIOSITY: none in force")
    else:
        s = inp.signal
        until = _aware(s.get("expires_at"))
        line = f"PRIME'S SIGNAL ON CURIOSITY: {s.get('direction')} (signal_id {s.get('signal_id')}"
        line += f", until {until:%H:%M} UTC)" if until else ")"
        if s.get("why"):
            line += f": {_one_line(s.get('why'), 300)!r}"
        lines.append(line)
    return lines


def _focus_lines(inp: ReplicaFrameInputs) -> list[str]:
    f = inp.request.pending_focus
    if f is None:
        return ["PENDING FOCUS: none waiting"]
    if f.origin == "juniper":
        who = "Juniper asked for this"
    elif f.from_prime:
        who = "Prime queued this"
    else:
        who = "Orion chose this in a conversation with Juniper"
    return [f"PENDING FOCUS ({who}; the run is shown it first): {_one_line(f.question, 400)!r}"]


def _digest_lines(inp: ReplicaFrameInputs) -> list[str]:
    if inp.digest_error:
        return [f"WHAT JUNIPER HAS BEEN TALKING ABOUT: (unavailable: {inp.digest_error})"]
    d = inp.digest
    if not d:
        return ["WHAT JUNIPER HAS BEEN TALKING ABOUT: (unavailable: no chat digest has been written)"]
    covers = _aware(d.get("window_end"))
    head = (f"WHAT JUNIPER HAS BEEN TALKING ABOUT (the daily chat digest {d.get('index') or ''}, covering up to "
            f"{covers:%Y-%m-%d %H:%M} UTC, written {_ago(inp.now, d.get('created_at'))}):" if covers else
            f"WHAT JUNIPER HAS BEEN TALKING ABOUT (the daily chat digest, written {_ago(inp.now, d.get('created_at'))}):")
    out = [head]
    if d.get("title"):
        out.append(f"title: {_one_line(d['title'], 200)}")
    out.append(_one_line(d.get("summary"), DIGEST_SUMMARY_CHARS) or "(the digest has no summary)")
    return out


def build_frame(inp: ReplicaFrameInputs) -> str:
    req = inp.request
    left = (req.deadline_at - inp.now).total_seconds()
    lines = [f"NOW: {inp.now:%Y-%m-%d %H:%M} UTC. Plan for curiosity run {req.run_id}; Hub waits {max(0, int(left))} s more.",
             ""]
    lines.append(f"CANDIDATES ({len(req.candidates)} fresh live priors Hub can show; not in any order that means "
                 f"something; pick {PICKS_MIN} to {PICKS_MAX} by prior_id):")
    lines += [_candidate_line(c) for c in req.candidates] or ["(none)"]
    lines.append("")
    shown = ", ".join(req.stale_ids[:STALE_IDS_SHOWN]) + (" ..." if len(req.stale_ids) > STALE_IDS_SHOWN else "")
    lines.append(f"STALE PRIORS (tested 3+ times; shown to the run anyway in their own bucket; not pickable): "
                 f"{len(req.stale_ids)}" + (f": {shown}" if req.stale_ids else ""))
    lines.append("")
    lines.append("RECENT RUNS (newest first):")
    lines += _runs_lines(inp)
    lines.append("")
    lines += _prime_lines(inp)
    lines.append("")
    lines += _focus_lines(inp)
    lines.append("")
    lines += _digest_lines(inp)
    lines.append("")
    lines.append("YOUR TRACK RECORD (your last plans, newest first):")
    if inp.record_rows is None:
        lines.append(f"(unavailable: {inp.record_error or 'not read'})")
    else:
        lines += record_lines(inp.record_rows, fallback_counts=inp.fallback_counts, plans_total=inp.plans_total)
    lines.append("")
    lines += _answers_section(inp)
    if inp.notes:
        lines.append("")
        lines.append("NOTES:")
        lines += [f"- {n}" for n in inp.notes[-3:]]
    return "\n".join(lines)


def _answers_section(inp: "ReplicaFrameInputs | ReplyFrameInputs") -> list[str]:
    if not inp.answers_on:
        return ["YOUR ANSWERS TO PRIME: answering Prime is switched off; a stance is recorded only, nothing acts on it."]
    return ["YOUR ANSWERS TO PRIME (newest first):", *answer_lines(inp.answers, inp.now)]


@dataclass
class ReplyFrameInputs:
    """A reply thought: no plan request, one Prime act to answer."""

    now: datetime
    target: TargetV1
    runs: Optional[list[dict[str, Any]]] = None
    runs_error: Optional[str] = None
    outcomes: Optional[dict[str, Mapping[str, Any]]] = None
    outcomes_error: Optional[str] = None
    concerns: Optional[PrimeConcernsViewV1] = None
    concerns_error: Optional[str] = None
    signal: Optional[dict[str, Any]] = None
    signal_error: Optional[str] = None
    digest: Optional[dict[str, Any]] = None
    digest_error: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    answers: list[AnswerRecordV1] = field(default_factory=list)
    answers_on: bool = True


REPLY_SYSTEM_PROMPT = f"""You are curiosity's planner (the curiosity replica): a small helper inside Orion's curiosity organ.
You are not Orion. Nothing you write is shown to Orion's curiosity run; only the reasons you give for
challenging or refusing an act of Prime's are shown to Prime.

Prime, Orion's slow between-conversations thinking, just acted on curiosity: it queued a question for curiosity's
next run, or sent a signal that slows down (or quickens) how often curiosity runs. You answer it once.

Answer with ONE JSON object and nothing else:
{{"reading": "one sentence: how you read curiosity's situation now",
 "stance": "support" | "challenge" | "refuse",
 "reason": "one sentence, at most 200 characters"}}

Rules:
{STANCE_RULES}
- Judge by what curiosity has been finding and failing (RECENT RUNS), what Prime is concerned about, and what
  Juniper has been talking about. "support" is a normal, good answer.
- A section marked "(unavailable: ...)" could not be read. Do not treat it as empty.
"""


def reply_system_prompt(*, per_day: int = 3, refractory_min: int = 30) -> str:
    return _caps(REPLY_SYSTEM_PROMPT, per_day, refractory_min)


def _target_lines(inp: ReplyFrameInputs) -> list[str]:
    t = inp.target
    if t.kind == "focus":
        return ["WHAT PRIME DID: queued a question for curiosity's next regular run (the run is shown it first "
                f"unless you refuse it): {_one_line(t.about, 500)!r}"]
    until = f", until {t.expires_at:%H:%M} UTC" if t.expires_at else ""
    return [f"WHAT PRIME DID: sent a {t.direction or '?'} signal to curiosity (signal_id {t.target_id}{until}): "
            f"{_one_line(t.about, 500)!r}",
            "A slow signal is a temporary charge on curiosity's bids in Orion's body economy: a hungry run still "
            "outbids it. Refusing it lifts the charge from the next bid on."]


def build_reply_frame(inp: ReplyFrameInputs) -> str:
    lines = [f"NOW: {inp.now:%Y-%m-%d %H:%M} UTC. Answer one act of Prime's.", ""]
    lines += _target_lines(inp)
    lines.append("")
    lines.append("RECENT RUNS (newest first):")
    lines += _runs_lines(inp)  # type: ignore[arg-type]  -- the same fields as a plan's inputs
    lines.append("")
    lines += _prime_lines(inp)  # type: ignore[arg-type]
    lines.append("")
    lines += _digest_lines(inp)  # type: ignore[arg-type]
    lines.append("")
    lines += _answers_section(inp)
    if inp.notes:
        lines.append("")
        lines.append("NOTES:")
        lines += [f"- {n}" for n in inp.notes[-3:]]
    return "\n".join(lines)


def parse_reply(parsed: Optional[dict[str, Any]]) -> tuple[Optional[ReplyAnswerV1], Optional[str]]:
    if not isinstance(parsed, dict):
        return None, "no_json_object"
    try:
        return ReplyAnswerV1.model_validate(parsed), None
    except ValidationError as exc:
        return None, f"invalid_answer: {str(exc)[:200]}"


def parse_answer(parsed: Optional[dict[str, Any]]) -> tuple[Optional[CuriosityOfferPlanAnswerV1], Optional[str]]:
    if not isinstance(parsed, dict):
        return None, "no_json_object"
    try:
        return CuriosityOfferPlanAnswerV1.model_validate(parsed), None
    except ValidationError as exc:
        return None, f"invalid_answer: {str(exc)[:200]}"


def frame_tokens(system: str, frame: str) -> int:
    """A rough size (4 characters per token), recorded on the plan row."""
    return (len(system) + len(frame)) // 4


__all__ = [
    "CONCERNS_STALE_SEC",
    "REPLY_SYSTEM_PROMPT",
    "ReplicaFrameInputs",
    "ReplyFrameInputs",
    "STANCE_RULES",
    "SYSTEM_PROMPT",
    "build_frame",
    "build_reply_frame",
    "parse_reply",
    "reply_system_prompt",
    "frame_tokens",
    "parse_answer",
    "system_prompt",
]
