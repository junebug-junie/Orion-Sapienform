"""The curiosity replica's answers to Prime: support, challenge or refuse (design section 9, patch 4).

Design: docs/superpowers/specs/2026-11-04-curiosity-replica-design.md section 9 (Juniper chose challenge AND
refuse, 2026-11-04). Pure: no I/O. The runtime (services/orion-mind/app/replica_loop.py) reads, delivers and
publishes.

What the replica may answer (a ``TargetV1``): one of Prime's acts on curiosity --

* **a focus question Prime queued** (Hub's pending slot, ``chat_correlation_id`` ``prime:<thought_id>``): Prime's
  act is its question act, ``stable_act_id("prime", "ask", focus_id)``. Never a focus Juniper asked for, never one
  Orion chose in a conversation: those are not Prime's acts, so they are never a target (a stance on them stays
  on the plan row only). Refusal applies to Prime's acts only.
* **a regulate signal on durable.curiosity_investigate** (Redis ``orion:prime:regulate:<family>``): Prime's act is
  the signal's ``act_id``. The replica cannot tell Prime's treated signals from its controls, by design.

Every answer is an act: actor ``curiosity.replica``, verb ``ask``, ``parent_act_id`` = Prime's act, keyed on
Prime's act (so a second answer to the same act is the same act id), ``test_before = holdback:<arm>``. It
closes at once; its note starts with the stance (``refuse: <reason>``), which is how the readout reads it.

The rules (``answer_blocker``): one answer per Prime act; at most ``per_day`` answers in any 24 h; at least
``refractory_sec`` between two answers; never an answer to an act whose parent is itself an answer (no
Prime-replica argument loop).

Holdback (``draw_answer_arm``): a deterministic draw per (Prime act, stance), like Prime's ``draw_arm``.
``control``: a refusal is recorded but not applied (the focus is still shown, the charge still applies); a
challenge is recorded but never reaches Prime. A support is never held back (it changes nothing).

What reaches Prime (``replies_view``): the delivered challenges and refusals of the last 48 h, with reasons --
never a held-back one, never a support.

``artifact_produced`` of an answer act: 1 when the answer did what its arm says -- delivered (treated) or
recorded held back (control); 0 when a treated answer could not be delivered (Hub would not refuse the focus,
the signal was gone or replaced, a write failed). Same rule both arms, so the holdback comparison is fair.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Iterable, Literal, Optional, Sequence

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from orion.grammar.acts import build_act, stable_act_id
from orion.schemas.curiosity_replica import (
    REPLICA_ACTOR,
    REPLIES_MAX,
    REPLIES_WINDOW_SEC,
    STANCE_REASON_MAX,
    AnswerTargetKind,
    CuriosityReplicaRepliesViewV1,
    CuriosityReplicaReplyV1,
    Stance,
)
from orion.schemas.grammar import ActV1

PRIME_ACTOR = "prime"
ANSWER_COMPONENT = "curiosity_replica.answer"
REPLY_REF_PREFIX = "curiosity_replica_reply:"
# An answer closes in the same coroutine that opens it; the horizon only covers a crash in between.
ANSWER_HORIZON_SEC = 600.0
ANSWER_BASIS = (
    "artifact_produced = the answer did what its arm says: delivered to what it answers (treated: a refusal "
    "applied, a challenge shown to Prime) or recorded held back (control); 0 when a treated answer could not be "
    "delivered. Closes at once; the note starts with the stance and carries the reason"
)
REPLY_BASIS = (
    "artifact_produced = the model returned a stance answer that parsed (a thought woken by a new Prime act, "
    "no plan request); horizon = the seat wait + the LLM timeout + 120s"
)
REPLY_MARGIN_SEC = 120.0
# The refusal-rate guard (design section 9): the share of refusals over the replica's last 20 answers.
REFUSAL_RATE_WINDOW = 20
REFUSAL_RATE_ALARM = 0.5
REFUSAL_ALARM_DAYS = 14
ANSWERS_KEPT = 60
WOKEN_KEPT = 40
# The answer's reason as an act note (orion/grammar/acts.py cuts notes at 500 characters).
NOTE_REASON_CHARS = 300

Arm = Literal["treated", "control"]
AnswerSource = Literal["plan", "reply"]


class TargetV1(BaseModel):
    """One of Prime's acts the replica may answer."""

    model_config = ConfigDict(extra="forbid")

    kind: AnswerTargetKind
    target_id: str = Field(min_length=1)
    prime_act_id: str = Field(min_length=1)
    # The Prime act's own parent, when known (Prime's thought act for a question); used by the loop check.
    parent_act_id: Optional[str] = None
    actor: str = PRIME_ACTOR
    # What Prime asked or signalled, for the frame and the replies view.
    about: str = Field(default="", max_length=600)
    # Signals: the direction and when it ends (shown in the frame).
    direction: Optional[str] = None
    expires_at: Optional[AwareDatetime] = None


class AnswerRecordV1(BaseModel):
    """One answer the replica made, in its own state (never published as-is: see ``replies_view``)."""

    model_config = ConfigDict(extra="forbid")

    answer_act_id: str
    prime_act_id: str
    kind: AnswerTargetKind
    target_id: str
    stance: Stance
    reason: str = Field(max_length=STANCE_REASON_MAX)
    arm: Arm
    # treated: did it reach its target; control: False (held back by design).
    delivered: bool = False
    delivery_note: str = ""
    about: str = ""
    source: AnswerSource = "reply"
    at: AwareDatetime


class ReplyAnswerV1(BaseModel):
    """The model's JSON on a reply thought (no plan request): one stance on one Prime act."""

    model_config = ConfigDict(extra="ignore")

    reading: str = Field(default="", max_length=600)
    stance: Stance
    reason: str = Field(min_length=1, max_length=STANCE_REASON_MAX)


def question_act_id(focus_id: str) -> str:
    """Prime's question act for ``focus_id`` (orion/prime/acts.py nomination_act: key = focus_id)."""
    return stable_act_id(PRIME_ACTOR, "ask", str(focus_id))


def prime_thought_act_id(thought_id: str) -> str:
    """Prime's thought act (orion/prime/acts.py thought_act: key = thought_id)."""
    return stable_act_id(PRIME_ACTOR, "infer", str(thought_id))


def focus_target(*, focus_id: str, question: str, chat_correlation_id: Optional[str]) -> Optional[TargetV1]:
    """A pending focus as a target, or None when it is not Prime's (Juniper's, or Orion's from a conversation)."""
    corr = str(chat_correlation_id or "")
    if not corr.startswith("prime:") or len(corr) <= len("prime:"):
        return None
    return TargetV1(kind="focus", target_id=str(focus_id), prime_act_id=question_act_id(str(focus_id)),
                    parent_act_id=prime_thought_act_id(corr[len("prime:"):]), about=" ".join(str(question).split())[:600])


def signal_target(sig: object) -> TargetV1:
    """A regulate signal (``prime_charge.PrimeRegulationV1``) as a target. Its arm is not carried."""
    why = " ".join(str(getattr(sig, "why", "") or "").split())
    return TargetV1(kind="signal", target_id=str(getattr(sig, "signal_id")), prime_act_id=str(getattr(sig, "act_id")),
                    about=f"{getattr(sig, 'direction')} curiosity: {why}"[:600], direction=str(getattr(sig, "direction")),
                    expires_at=getattr(sig, "expires_at", None))


def draw_answer_arm(prime_act_id: str, stance: str, *, refusal_fraction: float, challenge_fraction: float) -> Arm:
    """Deterministic per (Prime act, stance), so a retry draws the same arm. Support: always treated."""
    if stance == "support":
        return "treated"
    fraction = refusal_fraction if stance == "refuse" else challenge_fraction
    u = int(hashlib.sha256(f"curiosity-replica-holdback|{stance}|{prime_act_id}".encode("utf-8")).hexdigest()[:8],
            16) / 0x100000000
    return "control" if u < max(0.0, min(1.0, float(fraction))) else "treated"


def answer_act(target: TargetV1, *, arm: Arm) -> ActV1:
    """Keyed on Prime's act: one answer per Prime act, whatever the stance."""
    return build_act(
        actor=REPLICA_ACTOR, verb="ask", key=f"answer:{target.prime_act_id}", signal_id="artifact_produced",
        horizon_sec=ANSWER_HORIZON_SEC, basis=ANSWER_BASIS,
        object_ref=(f"curiosity_focus:{target.target_id}" if target.kind == "focus"
                    else f"prime_signal:{target.target_id}"),
        test_before=f"holdback:{arm}", parent_act_id=target.prime_act_id,
    )


def reply_act(thought_id: str, *, horizon_sec: float) -> ActV1:
    """The thought behind a reply (verb infer), like a plan's thought act."""
    return build_act(
        actor=REPLICA_ACTOR, verb="infer", key=thought_id, signal_id="artifact_produced",
        horizon_sec=float(horizon_sec) + REPLY_MARGIN_SEC, basis=REPLY_BASIS,
        object_ref=f"{REPLY_REF_PREFIX}{thought_id}",
    )


def answer_note(record: AnswerRecordV1) -> str:
    """The answer act's close note: the stance first (the readout reads it), the reason, what happened."""
    if record.arm == "control":
        what = "held back (control): recorded, not applied" if record.stance == "refuse" else "held back (control): not shown to Prime"
    elif record.stance == "support":
        what = "recorded"
    else:
        what = "delivered" if record.delivered else f"not delivered: {record.delivery_note or 'unknown'}"
    return f"{record.stance}: {record.reason[:NOTE_REASON_CHARS]} | {record.kind} {record.target_id} | {what}"


def answer_produced(record: AnswerRecordV1) -> bool:
    return True if record.arm == "control" or record.stance == "support" else bool(record.delivered)


def answer_blocker(answers: Sequence[AnswerRecordV1], target: TargetV1, now: datetime, *, per_day: int,
                   refractory_sec: float, own_answer_ids: Iterable[str] = ()) -> Optional[str]:
    """Why the replica may NOT answer ``target`` now, or None."""
    own = set(own_answer_ids) | {a.answer_act_id for a in answers}
    if target.actor == REPLICA_ACTOR or target.prime_act_id in own:
        return "it is one of my own answers"
    if target.parent_act_id and target.parent_act_id in own:
        return "its parent is one of my answers (no answer to an answer)"
    if any(a.prime_act_id == target.prime_act_id for a in answers):
        return "I already answered it"
    recent = [a for a in answers if (now - a.at) < timedelta(hours=24)]
    if len(recent) >= max(0, int(per_day)):
        return f"I already answered Prime {len(recent)} times in the last 24 h (at most {int(per_day)})"
    if answers:
        last = max(a.at for a in answers)
        if (now - last).total_seconds() < float(refractory_sec):
            return (f"my last answer was {int((now - last).total_seconds() // 60)} min ago "
                    f"(at least {int(float(refractory_sec) // 60)} min between answers)")
    return None


def keep_answers(answers: Sequence[AnswerRecordV1], new: AnswerRecordV1) -> list[AnswerRecordV1]:
    return [*[a for a in answers if a.prime_act_id != new.prime_act_id], new][-ANSWERS_KEPT:]


def refusal_rate(answers: Sequence[AnswerRecordV1], *, window: int = REFUSAL_RATE_WINDOW) -> tuple[int, int]:
    """(refusals, answers) over the last ``window`` answers, held-back ones included (they are answers)."""
    last = sorted(answers, key=lambda a: a.at)[-window:]
    return sum(1 for a in last if a.stance == "refuse"), len(last)


def refusal_rate_line(answers: Sequence[AnswerRecordV1]) -> str:
    refusals, total = refusal_rate(answers)
    if total == 0:
        return "your refusal rate: no answers to Prime yet"
    share = refusals / total
    flag = (f" -- above {REFUSAL_RATE_ALARM:.0%}; Juniper is shown this if it stays there for "
            f"{REFUSAL_ALARM_DAYS} days" if share > REFUSAL_RATE_ALARM else "")
    return f"your refusal rate: {refusals} of your last {total} answers to Prime were refusals ({share:.0%}){flag}"


def answer_lines(answers: Sequence[AnswerRecordV1], now: datetime, *, shown: int = 5) -> list[str]:
    """The replica's own last answers, for its frames. Held-back ones are marked (the replica may know its own
    draw: it changes nothing it could chase; Prime never sees them)."""
    lines = [refusal_rate_line(answers)]
    for a in sorted(answers, key=lambda a: a.at, reverse=True)[:shown]:
        hours = max(0.0, (now - a.at).total_seconds() / 3600.0)
        if a.arm == "control":
            what = "held back at random (not applied, Prime not told)"
        elif a.stance == "support":
            what = "recorded"
        else:
            what = "delivered" if a.delivered else f"not delivered ({a.delivery_note or 'unknown'})"
        lines.append(f"- {hours:.0f} h ago, {a.stance} Prime's {'question' if a.kind == 'focus' else 'signal'} "
                     f"{a.about[:120]!r}: {a.reason[:160]} [{what}]")
    return lines


def replies_view(answers: Sequence[AnswerRecordV1], now: datetime) -> CuriosityReplicaRepliesViewV1:
    """What Prime hears: delivered challenges and refusals of the last 48 h. Never a held-back one."""
    out: list[CuriosityReplicaReplyV1] = []
    for a in sorted(answers, key=lambda a: a.at, reverse=True):
        if a.arm != "treated" or not a.delivered or a.stance not in ("challenge", "refuse"):
            continue
        if (now - a.at).total_seconds() > REPLIES_WINDOW_SEC:
            continue
        out.append(CuriosityReplicaReplyV1(answer_act_id=a.answer_act_id, prime_act_id=a.prime_act_id,
                                           target_kind=a.kind, target_id=a.target_id, stance=a.stance,
                                           reason=a.reason, about=a.about[:200], at=a.at))
    return CuriosityReplicaRepliesViewV1(as_of=now, replies=out[:REPLIES_MAX])


def stance_from_note(note: Optional[str]) -> Optional[str]:
    """The stance an answer act's close note starts with (``answer_note``), or None."""
    head = str(note or "").split(":", 1)[0].strip()
    return head if head in ("support", "challenge", "refuse") else None


def refusal_alarm(daily_rates: Sequence[tuple[str, Optional[float]]], *, days: int = REFUSAL_ALARM_DAYS,
                  threshold: float = REFUSAL_RATE_ALARM) -> bool:
    """True when the last ``days`` days (oldest first: (day, rate over the last 20 answers as of that day))
    ALL read above ``threshold``. A day with no answers yet (None) breaks the run: unknown is never an alarm."""
    tail = list(daily_rates)[-days:]
    return len(tail) == days and all(r is not None and r > threshold for _, r in tail)


__all__ = [
    "ANSWER_BASIS",
    "ANSWER_COMPONENT",
    "ANSWER_HORIZON_SEC",
    "AnswerRecordV1",
    "REFUSAL_ALARM_DAYS",
    "REFUSAL_RATE_ALARM",
    "REFUSAL_RATE_WINDOW",
    "REPLY_REF_PREFIX",
    "ReplyAnswerV1",
    "TargetV1",
    "WOKEN_KEPT",
    "answer_act",
    "answer_blocker",
    "answer_lines",
    "answer_note",
    "answer_produced",
    "draw_answer_arm",
    "focus_target",
    "keep_answers",
    "prime_thought_act_id",
    "question_act_id",
    "refusal_alarm",
    "refusal_rate",
    "refusal_rate_line",
    "replies_view",
    "reply_act",
    "signal_target",
    "stance_from_note",
]
