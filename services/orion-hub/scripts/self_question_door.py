"""The self-question door: Orion's own self-questions are judged before any sitting answers them.

Design: docs/superpowers/specs/2026-11-25-orion-self-reflection-drift-design.md ("The door", "The check
after the sitting"). Every question Orion writes for its own self-inquiry is stored ``door = 'held'``
(``orion.curiosity.self_question_pool``: ``UPSERT_MINT_SQL`` for a sitting's ``:SelfQuestionMint``,
``INSERT_HELD_SQL`` for the self-monitor's "Why?" question), and ``pick_question`` never picks a held row.
Each tick this loop:

1. **Judges** up to ``BATCH`` held rows, one call at a time, with the finding-register judge
   (``scripts.curiosity_finding_register.judge_finding``: prompt v2, temperature 0, thinking off, on the
   curiosity lane's own route; a label is kept only when the served model is in ``PASSED_JUDGE_MODELS``).
   The question text is the item; ``register.QUESTION_LINE`` says what it is.

   * ``philosophical`` -> ``admitted`` (enters the self-inquiry pool as before);
   * ``operational`` -> ``routed`` (never asked as self-inquiry);
   * ``no_finding`` or a model's judge error (not one JSON object, unknown label) -> stays ``held``,
     ``door_attempts + 1``; at ``max_attempts`` -> ``unjudged``, never admitted or routed by default;
   * a gateway refusal or dead transport (``call:`` errors), or a reply from a model that never passed the
     eval (``model_not_evaluated``), is not a judgement: stays ``held``, the attempt is not counted, a later
     tick tries again (least recently tried first).

2. **Routes** at most one routed or demoted question: a self focus request (``orion.curiosity.focus.queue_focus``,
   ``origin='self'``, ``keep_pending=True``, correlation ``self_question_door:<question_id>``) when no focus is
   pending and no door offer was made in the last ``OFFER_SPACING_SEC``; otherwise it stays recorded (the door
   column) and a later tick tries again. A pending door focus never blocks Prime: Prime's own focus replaces
   it. A door focus replaced or expired before any run took it is offered again (``#2``, ``#3``), at most
   ``MAX_OFFERS`` offers; one a run took is never offered again. Re-judged new text keeps the question's
   offer count (a question whose earlier words a run already dug is not offered again).

The check after the sitting (``demote_after_finding``, called by the finding register): a self-inquiry
finding labelled ``operational`` demotes the admitted Orion question that sitting answered
(``door = 'demoted'``, ``door_demoted_by_run``), and the loop then routes it like any routed question.

Juniper's questions never pass the door: every write here is ``WHERE minted_by = 'orion'``.
Log lines: ``self_question_door admitted|routed|held|unjudged|demoted|offered question_id=... model=...``.
Kill switch ``CURIOSITY_SELF_QUESTION_DOOR_ENABLED=false``: the loop does not start, nothing is demoted,
and the pick ignores the door (every Orion question eligible again, as before the door).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Optional

from orion.curiosity.focus import (
    DOOR_CORRELATION_PREFIX,
    ORIGIN_SELF,
    FocusRefused,
    queue_focus,
)
from orion.curiosity.register import JUDGE_PROMPT_SHA256, QUESTION_LINE, REASON_CHARS
from orion.curiosity.self_question_pool import (
    DOOR_ADMITTED,
    DOOR_DEMOTED,
    DOOR_HELD,
    DOOR_ROUTED,
    DOOR_UNJUDGED,
)
from .curiosity_finding_register import JudgeResult

logger = logging.getLogger("orion-hub.self_question_door")

BATCH = 8  # held rows judged per tick, one call at a time (the backfill's 88 rows clear in ~11 ticks)
# At most one door offer per this many seconds, so routed questions never fill the one focus slot run after
# run. Prime's own between-conversations focus also replaces a pending door focus (focus.queue_focus).
OFFER_SPACING_SEC = 6 * 3600.0
# A routed question whose offer was replaced or expired before any run took it is offered again, this many
# offers at most. One a run took (or that is pending) is never offered again.
MAX_OFFERS = 3

# Least recently tried first (door_judged_at is stamped on every try), so a row that keeps getting a
# gateway refusal cannot hold the head of the batch.
HELD_SQL = (
    "SELECT question_id, text, door_attempts FROM curiosity_self_questions "
    "WHERE minted_by = 'orion' AND door = 'held' "
    "ORDER BY door_judged_at NULLS FIRST, created_at, question_id LIMIT $1"
)
# Guarded on door = 'held' AND the text that was judged: a re-mint with new words between the read and the
# write keeps its new words held for their own judgement.
SET_DOOR_SQL = (
    "UPDATE curiosity_self_questions SET door = $2, door_reason = $3, door_judged_at = $4, "
    "door_judge_model = $5, door_prompt_sha256 = $6, door_attempts = $7 "
    "WHERE question_id = $1 AND minted_by = 'orion' AND door = 'held' AND text = $8"
)
# One door question's offers: correlation "self_question_door:<question_id>" or "...#<n>" for a re-offer.
_OFFERS_OF_Q = ("FROM curiosity_focus_request f WHERE split_part(f.chat_correlation_id, '#', 1) = $1 || q.question_id")
# Routed or demoted, no offer pending or seen by a run, and under MAX_OFFERS offers so far.
UNOFFERED_SQL = (
    f"SELECT q.question_id, q.text, q.door, (SELECT count(*) {_OFFERS_OF_Q}) AS offers "
    "FROM curiosity_self_questions q "
    "WHERE q.minted_by = 'orion' AND q.door IN ('routed', 'demoted') AND q.status <> 'parked' "
    f"AND NOT EXISTS (SELECT 1 {_OFFERS_OF_Q} AND (f.status = 'pending' OR f.offered_run_id IS NOT NULL)) "
    f"AND (SELECT count(*) {_OFFERS_OF_Q}) < $2 "
    "ORDER BY q.door_judged_at NULLS LAST, q.question_id LIMIT 1"
)
PENDING_FOCUS_SQL = (
    "SELECT 1 FROM curiosity_focus_request WHERE status = 'pending' AND expires_at > now() LIMIT 1"
)
RECENT_OFFER_SQL = (
    "SELECT 1 FROM curiosity_focus_request WHERE left(chat_correlation_id, length($1)) = $1 "
    "AND created_at > now() - make_interval(secs => $2::double precision) LIMIT 1"
)
DEMOTE_SQL = (
    "UPDATE curiosity_self_questions SET door = 'demoted', door_demoted_by_run = $2, door_reason = $3 "
    "WHERE question_id = $1 AND minted_by = 'orion' AND door = 'admitted' RETURNING question_id"
)

# Hub's words, shown to Orion as Hub's (focus.format_focus_section), never as Orion's own.
ROUTE_WHY = {
    DOOR_ROUTED: ("The judge read this question as being about how a part of Orion works, so it was sent to the "
                  "investigate line to be dug into instead of sat with in self-inquiry."),
    DOOR_DEMOTED: ("Orion sat with this question once in self-inquiry, and the judge read that sitting's answer as "
                   "being about how a part of Orion works, so the question was sent to the investigate line."),
}

Judge = Callable[[str, str], Awaitable[JudgeResult]]


@dataclass(frozen=True)
class DoorVerdict:
    door: str
    reason: Optional[str]
    attempts: int


def door_verdict(result: JudgeResult, *, attempts: int, max_attempts: int) -> DoorVerdict:
    """What the door does with one judge reply. Pure.

    A label is only ever the judge's: ``JudgeResult.register`` is None on every error, including a model
    not in PASSED_JUDGE_MODELS, so no error can admit or route."""
    if result.register == "philosophical":
        return DoorVerdict(DOOR_ADMITTED, result.reason, attempts)
    if result.register == "operational":
        return DoorVerdict(DOOR_ROUTED, result.reason, attempts)
    if result.transient or str(result.error or "").startswith("model_not_evaluated"):
        # Not a judgement: a gateway refusal, or a reply from a model that never passed the eval (a short
        # fallback must not turn every held question unjudged). Held, not counted, tried again.
        return DoorVerdict(DOOR_HELD, result.error, attempts)
    tried = attempts + 1
    reason = result.error or f"no_finding: {result.reason or ''}".strip()
    if tried >= max(1, int(max_attempts)):
        return DoorVerdict(DOOR_UNJUDGED, reason, tried)
    return DoorVerdict(DOOR_HELD, reason, tried)


def demotion_for(row: Mapping[str, Any]) -> Optional[str]:
    """The question id a finding-register row demotes, or None. Pure.

    Only a self-inquiry finding the judge labelled ``operational`` (a passed model's label: the register
    column is NULL on every judge error) on a sitting that answered a self-question."""
    if row.get("line") != "self_inquiry" or row.get("register") != "operational":
        return None
    ref = str(row.get("question_ref") or "")
    if not ref.startswith("self_question:"):
        return None
    qid = ref[len("self_question:"):].strip()
    return qid or None


def _provided(provider: Callable[[], Any]) -> Any:
    try:
        return provider()
    except Exception:  # noqa: BLE001 -- not ready yet
        return None


async def demote_after_finding(pool: Any, row: Mapping[str, Any], *, enabled: bool = True) -> Optional[str]:
    """The check after the sitting. Returns the demoted question id, or None. Never raises."""
    if not enabled or pool is None:
        return None
    qid = demotion_for(row)
    if qid is None:
        return None
    run_id = str(row.get("run_id") or "") or None
    reason = (f"finding operational: {row.get('register_reason') or ''}".strip())[:REASON_CHARS]
    try:
        async with pool.acquire() as conn:
            got = await conn.fetchval(DEMOTE_SQL, qid, run_id, reason)
    except Exception as exc:  # noqa: BLE001
        logger.warning("self_question_door_demote_failed question_id=%s run=%s err=%s", qid, run_id, exc)
        return None
    if got:
        logger.info("self_question_door demoted question_id=%s run=%s model=%s", qid, run_id, row.get("judge_model"))
        return str(got)
    return None


class SelfQuestionDoor:
    def __init__(
        self,
        *,
        enabled: bool,
        pool_provider: Callable[[], Any],
        judge: Judge,
        tick_interval_sec: float = 300.0,
        max_attempts: int = 3,
        batch: int = BATCH,
        offer_spacing_sec: float = OFFER_SPACING_SEC,
        max_offers: int = MAX_OFFERS,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self._pool_provider = pool_provider
        self._judge = judge
        self.tick_interval_sec = max(30.0, float(tick_interval_sec))
        self.max_attempts = max(1, int(max_attempts))
        self.batch = max(1, int(batch))
        self.offer_spacing_sec = max(0.0, float(offer_spacing_sec))
        self.max_offers = max(1, int(max_offers))
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._task: Optional[asyncio.Task] = None
        self.last_tick: dict[str, Any] = {}

    async def start(self) -> None:
        if not self.enabled:
            logger.info("self_question_door disabled")
            return
        self._task = asyncio.create_task(self._run())
        logger.info("self_question_door started tick=%ss max_attempts=%s", self.tick_interval_sec, self.max_attempts)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 -- one bad tick must not stop the loop
                logger.exception("self_question_door_tick_failed")
            await asyncio.sleep(self.tick_interval_sec)

    async def tick(self) -> dict[str, Any]:
        now = self._clock()
        pool = _provided(self._pool_provider)
        if pool is None:
            self.last_tick = {"at": now.isoformat(), "skipped": "no_pool"}
            return self.last_tick
        async with pool.acquire() as conn:
            held = [dict(r) for r in await conn.fetch(HELD_SQL, self.batch)]
        counts: dict[str, int] = {}
        for row in held:
            door = await self._judge_one(pool, row)
            counts[door] = counts.get(door, 0) + 1
        offered = await self._offer_one(pool)
        self.last_tick = {"at": now.isoformat(), "judged": len(held), "doors": counts, "offered": offered}
        return self.last_tick

    async def _judge_one(self, pool: Any, row: Mapping[str, Any]) -> str:
        qid, text = str(row["question_id"]), str(row.get("text") or "")
        attempts = int(row.get("door_attempts") or 0)
        try:
            result = await self._judge(QUESTION_LINE, text)
        except Exception as exc:  # noqa: BLE001 -- judge_finding never raises; a fake might
            result = JudgeResult(None, None, f"call: {type(exc).__name__}: {exc}"[:200], None)
        verdict = door_verdict(result, attempts=attempts, max_attempts=self.max_attempts)
        async with pool.acquire() as conn:
            status = await conn.execute(
                SET_DOOR_SQL, qid, verdict.door, (verdict.reason or None) and verdict.reason[:REASON_CHARS],
                self._clock(), result.model, JUDGE_PROMPT_SHA256, verdict.attempts, str(row.get("text") or ""))
        if not str(status or "").endswith(" 1"):
            logger.info("self_question_door skipped_changed question_id=%s (re-minted or edited meanwhile)", qid)
            return "skipped_changed"
        logger.info("self_question_door %s question_id=%s model=%s attempts=%s reason=%r",
                    verdict.door, qid, result.model, verdict.attempts, (verdict.reason or "")[:120])
        return verdict.door

    async def _offer_one(self, pool: Any) -> Optional[str]:
        """Offer one never-offered routed/demoted question to the investigate line, when the slot is free."""
        async with pool.acquire() as conn:
            if await conn.fetchval(PENDING_FOCUS_SQL):
                return None
            if await conn.fetchval(RECENT_OFFER_SQL, DOOR_CORRELATION_PREFIX, self.offer_spacing_sec):
                return None
            row = await conn.fetchrow(UNOFFERED_SQL, DOOR_CORRELATION_PREFIX, self.max_offers)
        if row is None:
            return None
        qid, door, n = str(row["question_id"]), str(row["door"]), int(row["offers"] or 0)
        corr = f"{DOOR_CORRELATION_PREFIX}{qid}" + (f"#{n + 1}" if n else "")
        try:
            result = await queue_focus(
                pool, question=" ".join(str(row["text"] or "").split()), why=ROUTE_WHY.get(door, ROUTE_WHY[DOOR_ROUTED]),
                origin=ORIGIN_SELF, chat_correlation_id=corr, keep_pending=True)
        except FocusRefused as exc:  # a focus arrived between the check and the queue: next tick
            logger.info("self_question_door offer_waiting question_id=%s why=%s", qid, str(exc)[:120])
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("self_question_door_offer_failed question_id=%s err=%s", qid, exc)
            return None
        logger.info("self_question_door offered question_id=%s door=%s focus_id=%s", qid, door, result.focus.focus_id)
        return qid


__all__ = ["BATCH", "DEMOTE_SQL", "HELD_SQL", "SET_DOOR_SQL", "UNOFFERED_SQL", "DoorVerdict", "SelfQuestionDoor",
           "demote_after_finding", "demotion_for", "door_verdict"]
