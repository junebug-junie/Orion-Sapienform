"""Consult loop: Orion takes a plumbing question to Claude before it reaches Juniper.

Design: docs/superpowers/specs/2026-11-11-consult-claude-before-juniper-design.md.
Pure rules (brief gate, budget gate, prompts, reply format): ``orion/consult/core.py``.

One consult at a time, all in Hub:

1. **Sources** offer a question (``offer``): a curiosity run the door policy kept from Juniper
   (``curiosity_investigation``), a faculty the self-monitor holds broken, or a belief Orion has
   tested many times without settling (``ask_claude_trigger.decide``). One source, one consult,
   ever (``source_ref`` is unique).
2. **Orion's turn** is an admitted durable run (``consult.orion_turn``, run id
   ``consult-<id>-t<n>``): the GPU pool queues it, durable-runs retries it, and Hub runs it as a real
   harness turn on the consult's own session (none of Juniper's chats), where Orion can run digs.
   Hub reads the result back with ``GET /runs/{run_id}`` on its tick.
3. **Claude's turn** goes to the room companion on the existing room channels (room id
   ``consult:<id>``), where Claude reads the deployed code read-only. Before each Claude turn the
   budget gate reads the Claude usage meter (``orion:substrate:claude_limit``) and fails closed.
4. **The ending** comes from Claude's reply. ``answered``: nothing goes to Juniper; Orion is told at
   their next curiosity run. ``agreed_proposal`` / ``disagree``: the gated brief goes to Hub chat
   (tagging Juniper, with Yes / No / No-because buttons) and to her email. A brief that fails the
   gate is never sent; Claude gets one chance to send it complete.

Every turn is shown live in Hub chat and stored in ``orion_consult_turn``. Turns are NOT written to
``chat_history_log``: Orion's recall keeps the brief only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from orion.consult.core import (
    DEFAULT_MAX_TURNS,
    budget_refusal,
    claude_turn_prompt,
    consult_cost_line,
    consult_id_from_room,
    consult_room_id,
    gate_ending,
    orion_session_id,
    orion_turn_prompt,
    parse_claude_turn,
    render_brief,
    source_line,
)
from orion.consult.core import self_monitor_sources, stuck_prior_source
from orion.schemas.claude_limit import ClaudeLimitObservationV1
from orion.schemas.room_claude import RoomClaudeUtteranceV1

logger = logging.getLogger("orion-hub.consult")

MIGRATION = "services/orion-sql-db/manual_migration_orion_consult_v1.sql"
CONSULT_TURN_KIND = "consult_turn"
CONSULT_BRIEF_KIND = "consult_brief"
CONSULT_BRIEF_TAG = "consult_brief"
EMAIL_EVENT_KIND = "orion.consult.brief"

# status_reason while a consult is open: who it is waiting for.
AWAIT_ORION = "awaiting_orion"
# Orion's turn is a durable run (consult.orion_turn); pending_correlation_id holds its run_id.
AWAIT_ORION_RUN = "awaiting_orion_run"
AWAIT_CLAUDE = "awaiting_claude"
BUDGET = "budget:"
MAX_CLAUDE_RESENDS = 1

# Columns set_status may write besides status/status_reason. Code-only names: never request data.
_SETTABLE = frozenset({
    "ending", "answer", "brief_json", "brief_text", "brief_sent_at", "email_status",
    "pending_prompt", "pending_correlation_id", "gate_retry_part", "claude_resends",
})

_COLUMNS = (
    "consult_id, source_kind, source_ref, source_detail, question, status, status_reason, ending, "
    "turns, claude_cost_usd, answer, brief_json, brief_text, brief_sent_at, email_status, decision, "
    "decision_reason, decided_at, orion_told_run_id, pending_prompt, pending_correlation_id, gate_retry_part, "
    "claude_resends, created_at, updated_at"
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def consult_uuid(consult_id: str, purpose: str) -> str:
    """A stable UUID for one consult's brief / email: bus envelopes require UUID correlation ids."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"orion.consult/{consult_id}/{purpose}"))


def _row(rec: Any) -> Dict[str, Any]:
    out = dict(rec)
    if isinstance(out.get("brief_json"), str):
        try:
            out["brief_json"] = json.loads(out["brief_json"])
        except ValueError:
            pass
    return out


# --- store (asyncpg) ---------------------------------------------------------------------------


async def offer(pool, *, source_kind: str, source_ref: str, question: str, detail: str = "") -> Optional[str]:
    """A pending consult for this source, or None when this source already had one."""
    cid = uuid.uuid4().hex[:12]
    rec = await pool.fetchrow(
        "INSERT INTO orion_consult (consult_id, source_kind, source_ref, source_detail, question) "
        "VALUES ($1, $2, $3, $4, $5) ON CONFLICT (source_ref) DO NOTHING RETURNING consult_id",
        cid, source_kind, source_ref, detail or None, question,
    )
    return rec["consult_id"] if rec else None


async def get(pool, consult_id: str) -> Optional[Dict[str, Any]]:
    rec = await pool.fetchrow(f"SELECT {_COLUMNS} FROM orion_consult WHERE consult_id = $1", consult_id)
    return _row(rec) if rec else None


async def open_consult(pool) -> Optional[Dict[str, Any]]:
    rec = await pool.fetchrow(
        f"SELECT {_COLUMNS} FROM orion_consult WHERE status = 'open' ORDER BY created_at LIMIT 1"
    )
    return _row(rec) if rec else None


async def next_pending(pool, *, max_age: timedelta) -> Optional[Dict[str, Any]]:
    rec = await pool.fetchrow(
        f"SELECT {_COLUMNS} FROM orion_consult WHERE status = 'pending' AND created_at >= $1 "
        "ORDER BY created_at LIMIT 1",
        _now() - max_age,
    )
    return _row(rec) if rec else None


async def expire_old_pending(pool, *, max_age: timedelta) -> int:
    rows = await pool.fetch(
        "UPDATE orion_consult SET status = 'expired', status_reason = 'never started', updated_at = now() "
        "WHERE status = 'pending' AND created_at < $1 RETURNING consult_id",
        _now() - max_age,
    )
    return len(rows)


async def set_status(pool, consult_id: str, status: str, reason: Optional[str] = None, **fields: Any) -> None:
    sets = ["status = $2", "status_reason = $3", "updated_at = now()"]
    args: List[Any] = [consult_id, status, reason]
    for key, value in fields.items():
        assert key in _SETTABLE, f"set_status: {key} is not a settable column"
        args.append(json.dumps(value) if key == "brief_json" and value is not None else value)
        sets.append(f"{key} = ${len(args)}")
    await pool.execute(f"UPDATE orion_consult SET {', '.join(sets)} WHERE consult_id = $1", *args)


async def turns(pool, consult_id: str) -> List[Dict[str, Any]]:
    rows = await pool.fetch(
        "SELECT turn_no, speaker, text, ending, model, cost_usd, error, created_at FROM orion_consult_turn "
        "WHERE consult_id = $1 ORDER BY turn_no",
        consult_id,
    )
    return [dict(r) for r in rows]


async def add_turn(pool, consult_id: str, *, turn_no: int, speaker: str, text: str, ending: Optional[str] = None,
                   model: Optional[str] = None, cost_usd: Optional[float] = None, error: Optional[str] = None) -> None:
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO orion_consult_turn (consult_id, turn_no, speaker, text, ending, model, cost_usd, error) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                consult_id, turn_no, speaker, text, ending, model, cost_usd, error,
            )
            await conn.execute(
                "UPDATE orion_consult SET turns = turns + 1, claude_cost_usd = claude_cost_usd + $2, "
                "updated_at = now() WHERE consult_id = $1",
                consult_id, float(cost_usd or 0.0),
            )


async def briefs_sent_since(pool, since: datetime) -> int:
    rec = await pool.fetchrow("SELECT count(*) AS n FROM orion_consult WHERE brief_sent_at >= $1", since)
    return int(rec["n"]) if rec else 0


async def decide(pool, consult_id: str, *, decision: str, reason: Optional[str]) -> Optional[Dict[str, Any]]:
    """Record Juniper's answer once. None when the consult has no undecided brief."""
    rec = await pool.fetchrow(
        "UPDATE orion_consult SET decision = $2, decision_reason = $3, decided_at = now(), updated_at = now() "
        f"WHERE consult_id = $1 AND status = 'brief_sent' AND decision IS NULL RETURNING {_COLUMNS}",
        consult_id, decision, reason,
    )
    return _row(rec) if rec else None


async def take_outcomes(pool, *, run_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Claim consult outcomes Orion has not been shown: answered ones and decided briefs. Once."""
    rows = await pool.fetch(
        "UPDATE orion_consult SET orion_told_run_id = $1, orion_told_at = now() WHERE consult_id IN ("
        " SELECT consult_id FROM orion_consult WHERE orion_told_run_id IS NULL AND"
        " (status = 'answered' OR (status = 'brief_sent' AND decision IS NOT NULL))"
        " ORDER BY updated_at LIMIT $2 FOR UPDATE SKIP LOCKED"
        f") RETURNING {_COLUMNS}",
        run_id, limit,
    )
    return [_row(r) for r in rows]


async def recent(pool, *, undecided_only: bool, limit: int = 20) -> List[Dict[str, Any]]:
    where = "WHERE status = 'brief_sent' AND decision IS NULL" if undecided_only else ""
    rows = await pool.fetch(
        f"SELECT {_COLUMNS} FROM orion_consult {where} ORDER BY created_at DESC LIMIT $1", limit
    )
    return [_row(r) for r in rows]


# --- the loop ----------------------------------------------------------------------------------


class ConsultLoop:
    def __init__(
        self,
        *,
        enabled: bool,
        pool_provider: Callable[[], Any],
        durable_client: Any,
        relay_provider: Callable[[], Any],
        notify_client: Any = None,
        connections_provider: Callable[[], Dict[str, Any]] = lambda: {},
        self_monitor_reader: Optional[Callable[[], Awaitable[Any]]] = None,
        priors_reader: Optional[Callable[[], Awaitable[Any]]] = None,
        claude_limit_channel: str = "orion:substrate:claude_limit",
        tick_sec: float = 120.0,
        max_turns: int = DEFAULT_MAX_TURNS,
        weekly_brief_cap: int = 3,
        claude_timeout_sec: float = 1200.0,
        orion_timeout_sec: float = 900.0,
        source_max_age_hours: float = 72.0,
        source_service: str = "orion-hub",
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.enabled = enabled
        self._pool_provider = pool_provider
        self._durable = durable_client
        self._relay_provider = relay_provider
        self._notify = notify_client
        self._connections_provider = connections_provider
        self._self_monitor_reader = self_monitor_reader
        self._priors_reader = priors_reader
        self.claude_limit_channel = claude_limit_channel
        self.tick_sec = tick_sec
        self.max_turns = max_turns
        self.weekly_brief_cap = weekly_brief_cap
        self.claude_timeout_sec = claude_timeout_sec
        self.orion_timeout_sec = orion_timeout_sec
        self.source_max_age = timedelta(hours=source_max_age_hours)
        self.source_service = source_service
        self._clock = clock
        self.meter: Dict[str, ClaudeLimitObservationV1] = {}
        self._lock = asyncio.Lock()
        self._tasks: List[asyncio.Task] = []
        self._bus = None
        self.last_tick: Dict[str, Any] = {}

    # -- lifecycle ------------------------------------------------------------------------------

    async def start(self, bus: Any) -> None:
        if not self.enabled:
            logger.info("consult_loop_disabled")
            return
        self._bus = bus
        self._tasks.append(asyncio.create_task(self._meter_loop(), name="consult_meter"))
        self._tasks.append(asyncio.create_task(self._tick_loop(), name="consult_tick"))
        logger.info("consult_loop_started tick=%ss max_turns=%s weekly_cap=%s",
                    self.tick_sec, self.max_turns, self.weekly_brief_cap)

    async def stop(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def _meter_loop(self) -> None:
        while True:
            try:
                async with self._bus.subscribe(self.claude_limit_channel) as pubsub:
                    async for msg in self._bus.iter_messages(pubsub):
                        self.note_meter(msg)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- resubscribe, never stop
                logger.warning("consult_meter_subscribe_failed err=%s", exc)
            await asyncio.sleep(5.0)

    def note_meter(self, msg: Any) -> None:
        try:
            data = msg.get("data") if isinstance(msg, dict) else msg
            raw = json.loads(data) if isinstance(data, (str, bytes)) else data
            payload = raw.get("payload") if isinstance(raw, dict) and isinstance(raw.get("payload"), dict) else raw
            obs = ClaudeLimitObservationV1.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 -- one bad message is not a meter
            logger.debug("consult_meter_bad_message err=%s", exc)
            return
        self.meter[obs.kind] = obs

    def budget(self) -> Optional[str]:
        return budget_refusal(self.meter.get("session_limit"), self.meter.get("weekly_limit"), now=self._clock())

    async def _tick_loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 -- one bad tick must not stop the loop
                logger.exception("consult_tick_failed")
            await asyncio.sleep(self.tick_sec)

    # -- sources --------------------------------------------------------------------------------

    async def offer(self, *, source_kind: str, source_ref: str, question: str, detail: str = "") -> Optional[str]:
        """Called by sources. Never raises: a source must not fail because consults are down."""
        if not self.enabled or not (question or "").strip():
            return None
        pool = self._pool_provider()
        if pool is None:
            return None
        try:
            cid = await offer(pool, source_kind=source_kind, source_ref=source_ref,
                              question=question.strip(), detail=detail)
        except Exception as exc:  # noqa: BLE001
            logger.warning("consult_offer_failed ref=%s err=%s", source_ref, exc)
            return None
        if cid:
            logger.info("consult_offered id=%s kind=%s ref=%s", cid, source_kind, source_ref)
        return cid

    async def _gather_sources(self) -> None:
        if self._self_monitor_reader is not None:
            try:
                view = await self._self_monitor_reader()
                for ref, question in self_monitor_sources(view):
                    await self.offer(source_kind="self_monitor", source_ref=ref, question=question)
            except Exception as exc:  # noqa: BLE001
                logger.warning("consult_self_monitor_source_failed err=%s", exc)
        if self._priors_reader is not None:
            try:
                decision = await self._priors_reader(self.meter.get("weekly_limit"))
                found = stuck_prior_source(decision)
                if found:
                    await self.offer(source_kind="stuck_prior", source_ref=found[0], question=found[1])
            except Exception as exc:  # noqa: BLE001
                logger.warning("consult_stuck_prior_source_failed err=%s", exc)

    # -- the tick -------------------------------------------------------------------------------

    async def tick(self) -> Dict[str, Any]:
        """Advance the open consult, or start the next one. One thing per tick."""
        now = self._clock()
        pool = self._pool_provider()
        if pool is None:
            self.last_tick = {"at": now.isoformat(), "skipped": "no_pool"}
            return self.last_tick
        if self._lock.locked():
            self.last_tick = {"at": now.isoformat(), "skipped": "busy"}
            return self.last_tick
        async with self._lock:
            row = await open_consult(pool)
            if row is not None:
                self.last_tick = {"at": now.isoformat(), **await self._advance(pool, row, now)}
                return self.last_tick
            await self._gather_sources()
            await expire_old_pending(pool, max_age=self.source_max_age)
            sent = await briefs_sent_since(pool, now - timedelta(days=7))
            if sent >= self.weekly_brief_cap:
                self.last_tick = {"at": now.isoformat(), "skipped": "weekly_brief_cap", "briefs_7d": sent}
                return self.last_tick
            refusal = self.budget()
            if refusal:
                self.last_tick = {"at": now.isoformat(), "skipped": f"budget:{refusal}"}
                return self.last_tick
            row = await next_pending(pool, max_age=self.source_max_age)
            if row is None:
                self.last_tick = {"at": now.isoformat(), "skipped": "nothing_pending"}
                return self.last_tick
            await set_status(pool, row["consult_id"], "open", AWAIT_ORION)
            logger.info("consult_opened id=%s kind=%s ref=%s", row["consult_id"], row["source_kind"], row["source_ref"])
            row["status"], row["status_reason"] = "open", AWAIT_ORION
            self.last_tick = {"at": now.isoformat(), "opened": row["consult_id"], **await self._advance(pool, row, now)}
            return self.last_tick

    async def _advance(self, pool, row: Dict[str, Any], now: datetime) -> Dict[str, Any]:
        cid, reason = row["consult_id"], row.get("status_reason") or ""
        if reason == AWAIT_ORION:
            return await self._orion_turn(pool, row)
        if reason == AWAIT_ORION_RUN:
            return await self._poll_orion_run(pool, row)
        if reason.startswith(BUDGET):
            return await self._send_claude(pool, row)
        if reason == AWAIT_CLAUDE:
            updated = row.get("updated_at")
            if isinstance(updated, datetime) and (now - updated).total_seconds() > self.claude_timeout_sec:
                if int(row.get("claude_resends") or 0) < MAX_CLAUDE_RESENDS:
                    # The reply rides pub/sub: a Hub restart while Claude worked loses it. The
                    # companion resumes the session, so one resend keeps Claude's context.
                    await set_status(pool, cid, "open", AWAIT_CLAUDE,
                                     claude_resends=int(row.get("claude_resends") or 0) + 1)
                    row = await get(pool, cid) or row
                    logger.warning("consult_claude_reply_lost id=%s -- resending once", cid)
                    return await self._send_claude(pool, row, resend=True)
                await set_status(pool, cid, "failed", "claude_no_reply_after_resend")
                self._push_turn(cid, 0, "Hub", "Claude did not reply in time, twice; this consult stopped.")
                return {"failed": cid, "why": "claude_no_reply_after_resend"}
            return {"waiting": cid, "for": "claude"}
        await set_status(pool, cid, "failed", f"unknown_state:{reason}")
        return {"failed": cid, "why": reason}

    # -- Orion's turn ---------------------------------------------------------------------------

    def _source(self, row: Dict[str, Any]) -> str:
        return source_line(row["source_kind"], row["source_ref"], row.get("source_detail") or "")

    async def _orion_turn(self, pool, row: Dict[str, Any]) -> Dict[str, Any]:
        """Submit Orion's turn as a durable run (``consult.orion_turn``). The GPU pool admits it, so it
        queues rather than times out, and durable-runs owns the retries. Hub only reads the result
        (``_poll_orion_run``). Never a retry queue here: a refused submit is simply tried next tick."""
        from orion.schemas.consult_turn_run import (
            CONSULT_ORION_TURN_WORKFLOW, ConsultOrionTurnBriefV1, consult_turn_run_id,
        )
        from orion.schemas.durable_run import DurableRunRequestV1
        from orion.schemas.resource_admission import ResourceRequirementV1

        cid = row["consult_id"]
        history = await turns(pool, cid)
        turn_no = len(history) + 1
        prompt = orion_turn_prompt(
            source=self._source(row), question=row["question"],
            transcript=[(t["speaker"], t["text"]) for t in history], turn_no=turn_no, max_turns=self.max_turns,
        )
        run_id = consult_turn_run_id(cid, turn_no)
        try:
            request = DurableRunRequestV1(
                # Deterministic: a resubmit of this turn is byte-identical, so durable-runs takes it as
                # the same run (it refuses a different request under the same run_id with 409).
                run_id=run_id, workflow=CONSULT_ORION_TURN_WORKFLOW,
                correlation_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"orion:consult:{run_id}")),
                brief=ConsultOrionTurnBriefV1(
                    consult="consult.orion_turn", consult_id=cid, turn_no=turn_no, prompt=prompt,
                    # Orion's consult session (none of Juniper's chats); recall searches the question.
                    session_id=orion_session_id(cid), retrieval_query=row["question"][:1000],
                    timeout_sec=self.orion_timeout_sec,
                ),
                admission=ResourceRequirementV1(preferred_lane="agent", resource="llm.route.agent", priority="background"),
            )
        except Exception as exc:  # noqa: BLE001 -- a brief that cannot be built would jam every tick
            await set_status(pool, cid, "failed", f"orion_brief_invalid:{type(exc).__name__}")
            logger.warning("consult_orion_brief_invalid id=%s err=%s", cid, str(exc)[:300])
            return {"failed": cid, "why": "orion_brief_invalid"}
        try:
            code, body = await self._durable.submit(request.model_dump(mode="json"))
        except Exception as exc:  # noqa: BLE001 -- durable-runs down: the next tick submits again
            logger.warning("consult_orion_run_submit_failed id=%s err=%s", cid, exc)
            return {"waiting": cid, "for": "durable_runs"}
        if code == 409:
            # The run already exists (a crash between submit and our status write, or a changed
            # setting): adopt it rather than fail the consult.
            try:
                existing = await self._durable.get_run(run_id)
            except Exception:  # noqa: BLE001
                existing = None
            if existing is not None:
                logger.info("consult_orion_run_adopted id=%s run=%s", cid, run_id)
                code = 202
        if code >= 400:
            logger.warning("consult_orion_run_refused id=%s run=%s http=%s body=%s", cid, run_id, code, body)
            await set_status(pool, cid, "failed", f"orion_run_refused:http_{code}")
            return {"failed": cid, "why": f"orion_run_refused:http_{code}"}
        await set_status(pool, cid, "open", AWAIT_ORION_RUN, pending_correlation_id=run_id)
        logger.info("consult_orion_run_submitted id=%s run=%s turn=%s", cid, run_id, turn_no)
        # One read right away: a resubmit of a turn that already finished (a Hub restart after the
        # run completed) is picked up now, not a tick later. A fresh run just reads as queued.
        return await self._poll_orion_run(pool, await get(pool, cid) or {**row, "pending_correlation_id": run_id})

    async def _poll_orion_run(self, pool, row: Dict[str, Any]) -> Dict[str, Any]:
        """Read Orion's durable run. Completed: the turn's full text becomes Orion's turn and Claude
        is invited. Failed or cancelled (durable-runs spent its retries): the consult fails."""
        cid, run_id = row["consult_id"], row.get("pending_correlation_id") or ""
        try:
            run = await self._durable.get_run(run_id)
        except Exception as exc:  # noqa: BLE001 -- unreachable: ask again next tick
            return {"waiting": cid, "for": f"durable_runs:{type(exc).__name__}"}
        if run is None:
            await set_status(pool, cid, "open", AWAIT_ORION, pending_correlation_id=None)
            return {"waiting": cid, "for": "orion_run_missing_resubmit"}
        status = str(run.get("status") or "")
        if status in ("failed", "cancelled"):
            error = str(run.get("error") or status)[:200]
            await set_status(pool, cid, "failed", f"orion_run_{status}:{error}", pending_correlation_id=None)
            logger.warning("consult_orion_run_ended id=%s run=%s status=%s err=%s", cid, run_id, status, error)
            return {"failed": cid, "why": f"orion_run_{status}"}
        if status != "completed":
            return {"waiting": cid, "for": f"orion_run:{status or 'unknown'}"}
        step = (run.get("harness_steps") or {}).get("turn") or {}
        text = str(step.get("text") or "").strip()
        if not text:
            await set_status(pool, cid, "failed", "orion_run_completed_empty", pending_correlation_id=None)
            return {"failed": cid, "why": "orion_run_completed_empty"}
        history = await turns(pool, cid)
        # The turn number is the run's (consult-<id>-t<n>), not len(history)+1: a crash after storing
        # the turn but before Claude was invited must not store it twice under the next number.
        try:
            turn_no = int(run_id.rsplit("-t", 1)[1])
        except (IndexError, ValueError):
            turn_no = len(history) + 1
        if not any(t["turn_no"] == turn_no for t in history):
            await add_turn(pool, cid, turn_no=turn_no, speaker="Orion", text=text,
                           model=(step.get("debug") or {}).get("fcc_model_label"))
            self._push_turn(cid, turn_no, "Orion", text)
        row = await get(pool, cid) or row
        return await self._send_claude(pool, row, prompt=self._claude_prompt(row, await turns(pool, cid)))

    # -- Claude's turn --------------------------------------------------------------------------

    def _claude_prompt(self, row: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
        orion_text = history[-1]["text"] if history and history[-1]["speaker"] == "Orion" else ""
        return claude_turn_prompt(
            orion_text=orion_text, turn_no=len(history) + 1, max_turns=self.max_turns,
            first=not any(t["speaker"] == "Claude" for t in history),
            source=self._source(row), question=row["question"],
        )

    async def _send_claude(self, pool, row: Dict[str, Any], *, prompt: Optional[str] = None,
                           resend: bool = False) -> Dict[str, Any]:
        """Send one Claude turn. The prompt is stored first, so a budget wait or a lost reply
        re-sends exactly it (``prompt=None`` re-sends the stored one). Each send gets a fresh
        correlation id; a reply that does not carry the current one is ignored."""
        cid = row["consult_id"]
        prompt = prompt if prompt is not None else (row.get("pending_prompt") or "")
        if not prompt:
            await set_status(pool, cid, "failed", "no_pending_claude_prompt")
            return {"failed": cid, "why": "no_pending_claude_prompt"}
        refusal = self.budget()
        if refusal:
            await set_status(pool, cid, "open", f"{BUDGET}{refusal}", pending_prompt=prompt,
                             pending_correlation_id=None)
            return {"waiting": cid, "for": f"budget:{refusal}"}
        relay = self._relay_provider()
        if relay is None:
            await set_status(pool, cid, "open", f"{BUDGET}no_relay", pending_prompt=prompt, pending_correlation_id=None)
            return {"waiting": cid, "for": "relay"}
        correlation_id = str(uuid.uuid4())
        # Written BEFORE the invite: a crash after it is a lost reply (resent once), never an
        # unknown request in flight.
        await set_status(pool, cid, "open", AWAIT_CLAUDE, pending_prompt=prompt, pending_correlation_id=correlation_id)
        await relay.invite(
            prompt=prompt, invited_by="Orion", session_id=None, room_id=consult_room_id(cid),
            correlation_id=correlation_id, trigger="auto",
        )
        return {"sent": cid, "to": "claude", "resend": resend}

    async def on_claude_utterance(self, utt: RoomClaudeUtteranceV1) -> None:
        """Called by the room relay for a consult room. Never raises into the relay."""
        try:
            await self._on_claude_utterance(utt)
        except Exception:  # noqa: BLE001
            logger.exception("consult_claude_utterance_failed room=%s", utt.room_id)

    async def _on_claude_utterance(self, utt: RoomClaudeUtteranceV1) -> None:
        cid = consult_id_from_room(utt.room_id)
        pool = self._pool_provider()
        if cid is None or pool is None:
            return
        async with self._lock:
            row = await get(pool, cid)
            if row is None or row["status"] != "open":
                logger.info("consult_utterance_ignored id=%s status=%s", cid, row and row["status"])
                return
            reason = row.get("status_reason") or ""
            if reason != AWAIT_CLAUDE or utt.correlation_id != row.get("pending_correlation_id"):
                # A late reply to an earlier send, or a reply while nothing is in flight.
                logger.info("consult_utterance_unexpected id=%s state=%s corr=%s want=%s",
                            cid, reason, utt.correlation_id, row.get("pending_correlation_id"))
                return
            await set_status(pool, cid, "open", AWAIT_CLAUDE, pending_correlation_id=None)
            gate_retry = row.get("gate_retry_part")
            history = await turns(pool, cid)
            turn_no = len(history) + 1
            if not utt.ok:
                await add_turn(pool, cid, turn_no=turn_no, speaker="Claude", text="", model=utt.model,
                               cost_usd=utt.cost_usd, error=(utt.error or "claude turn failed")[:2000])
                await set_status(pool, cid, "failed", "claude_error")
                self._push_turn(cid, turn_no, "Claude", f"[Claude could not reply: {utt.error}]")
                return
            parsed = parse_claude_turn(utt.text)
            await add_turn(pool, cid, turn_no=turn_no, speaker="Claude", text=parsed.room_text or utt.text,
                           ending=parsed.ending, model=utt.model, cost_usd=utt.cost_usd, error=parsed.error)
            self._push_turn(cid, turn_no, "Claude", parsed.room_text or utt.text)
            claude_turns = sum(1 for t in history if t["speaker"] == "Claude") + 1
            if parsed.ending == "continue":
                if turn_no >= self.max_turns or gate_retry:
                    await set_status(pool, cid, "brief_refused", parsed.error or "no_ending_on_last_turn")
                    return
                await set_status(pool, cid, "open", AWAIT_ORION)
                self._spawn(self._continue(cid))
                return
            result = gate_ending(parsed.ending, parsed.payload)
            if not result.ok:
                if gate_retry:
                    await set_status(pool, cid, "brief_refused", f"missing:{result.refused_part}", ending=parsed.ending)
                    logger.warning("consult_brief_refused id=%s part=%s", cid, result.refused_part)
                    return
                await set_status(pool, cid, "open", AWAIT_CLAUDE, ending=parsed.ending,
                                 gate_retry_part=result.refused_part)
                await self._send_claude(pool, await get(pool, cid) or row, prompt=(
                    f"Your ending block was refused: the part `{result.refused_part}` is missing or invalid. "
                    "Send your ending block again, complete, with the same ending. Nothing else is needed."
                ))
                return
            if parsed.ending == "answered":
                await set_status(pool, cid, "answered", None, ending="answered", answer=result.answer.answer)
                logger.info("consult_answered id=%s", cid)
                return
            row = await get(pool, cid) or row
            await self._deliver_brief(pool, row, result.brief, parsed.ending, claude_turns)

    async def _continue(self, cid: str) -> None:
        """Orion's next turn right after Claude's, instead of waiting for the tick. Never raises:
        on an error the tick picks the turn up."""
        pool = self._pool_provider()
        if pool is None:
            return
        try:
            async with self._lock:
                row = await get(pool, cid)
                if row and row["status"] == "open" and row.get("status_reason") == AWAIT_ORION:
                    await self._orion_turn(pool, row)
        except Exception:  # noqa: BLE001
            logger.exception("consult_continue_failed id=%s", cid)

    def _spawn(self, coro: Awaitable[Any]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)

    # -- the brief ------------------------------------------------------------------------------

    async def _deliver_brief(self, pool, row: Dict[str, Any], brief: Any, ending: str, claude_turns: int) -> None:
        cid = row["consult_id"]
        text = render_brief(
            brief, ending=ending, source=self._source(row),
            consult_cost=consult_cost_line(float(row.get("claude_cost_usd") or 0.0), claude_turns),
            transcript_hint="The whole conversation is in Hub chat (Show the conversation, under this brief).",
        )
        await set_status(pool, cid, "brief_sent", None, ending=ending, brief_json=brief.model_dump(mode="json"),
                         brief_text=text, brief_sent_at=self._clock())
        self._push({"kind": CONSULT_BRIEF_KIND, "consult_id": cid, "llm_response": text, "mode": "orion"})
        await self._history(cid, text)
        email_status = await self._email(cid, row["question"], text)
        await set_status(pool, cid, "brief_sent", None, email_status=email_status)
        logger.info("consult_brief_sent id=%s ending=%s email=%s", cid, ending, email_status)

    async def _history(self, cid: str, text: str) -> None:
        """The brief, and only the brief, goes to chat history (unsolicited: out of turn recall)."""
        bus = self._bus
        if bus is None:
            return
        try:
            from scripts.chat_history import build_chat_history_envelope, publish_chat_history

            env = build_chat_history_envelope(
                content=text, role="assistant", session_id=orion_session_id(cid),
                correlation_id=consult_uuid(cid, "brief"), speaker="Orion", tags=[CONSULT_BRIEF_TAG],
                message_id=consult_uuid(cid, "brief-message"),
                client_meta={"unsolicited": True, "source": CONSULT_BRIEF_TAG, "consult_id": cid},
            )
            await publish_chat_history(bus, [env])
        except Exception as exc:  # noqa: BLE001
            logger.warning("consult_history_failed id=%s err=%s", cid, exc)

    async def _email(self, cid: str, question: str, text: str) -> str:
        if self._notify is None:
            return "no_notify_client"
        try:
            from orion.schemas.notify import NotificationRequest

            first = " ".join((question or "").split())[:80]
            request = NotificationRequest(
                source_service=self.source_service, event_kind=EMAIL_EVENT_KIND, severity="info",
                title=f"Orion + Claude: {first}", body_md=text + "\n\nAnswer in Hub chat (Yes / No / No because).",
                tags=["consult"], channels_requested=["email"], dedupe_key=f"consult:{cid}",
                correlation_id=consult_uuid(cid, "email"),
            )
            accepted = await asyncio.to_thread(self._notify.send, request)
            return str(getattr(accepted, "email_status", None) or ("accepted" if getattr(accepted, "ok", False) else "refused"))
        except Exception as exc:  # noqa: BLE001 -- the chat brief already landed
            logger.warning("consult_email_failed id=%s err=%s", cid, exc)
            return f"failed:{type(exc).__name__}"

    # -- Hub chat -------------------------------------------------------------------------------

    def _push_turn(self, cid: str, turn_no: int, speaker: str, text: str) -> None:
        self._push({"kind": CONSULT_TURN_KIND, "consult_id": cid, "turn_no": turn_no, "speaker": speaker,
                    "llm_response": text, "mode": "consult"})

    def _push(self, payload: Dict[str, Any]) -> None:
        """Every live chat window sees a consult (it is Orion's, not one session's)."""
        for connection_id, entry in list((self._connections_provider() or {}).items()):
            queue = entry.get("queue") if isinstance(entry, dict) else None
            if queue is None:
                continue
            try:
                queue.put_nowait(dict(payload))
            except Exception as exc:  # noqa: BLE001
                logger.warning("consult_push_failed connection=%s err=%s", connection_id, exc)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "last_tick": self.last_tick,
            "budget_refusal": self.budget(),
            "meter": {k: v.model_dump(mode="json") for k, v in self.meter.items()},
        }


# --- source readers, built by main -------------------------------------------------------------


def make_self_monitor_reader(redis_provider: Callable[[], Any]) -> Callable[[], Awaitable[Any]]:
    """The self-monitor's published state as a view. Unknown when Redis or the state is missing."""
    from orion.self_monitor.view import SELF_MONITOR_REDIS_KEY, self_monitor_view

    async def _read() -> Any:
        redis = redis_provider()
        raw = await redis.get(SELF_MONITOR_REDIS_KEY) if redis is not None else None
        return self_monitor_view(raw, now=_now())

    return _read


def make_priors_reader(reader_provider: Callable[[], Any]) -> Callable[[Any], Awaitable[Any]]:
    """The stuck-prior trigger's decision over Orion's live beliefs (read-only graph query).

    Uses ``ask_claude_trigger.decide`` unchanged: it picks the most-tested unsettled belief, and
    refuses when the weekly meter reading is not a fresh ``clear``."""
    from orion.autonomy.ask_claude_trigger import decide
    from orion.curiosity.worldview import LIVE_PRIORS_CYPHER, build_prior, collapse_duplicate_priors

    async def _read(weekly: Any) -> Any:
        reader = reader_provider()
        if reader is None:
            return None
        rows = await asyncio.to_thread(reader.query, LIVE_PRIORS_CYPHER)
        priors, _ = collapse_duplicate_priors([p for p in (build_prior(r) for r in rows) if p is not None])
        return decide(priors=priors, limit=weekly)

    return _read
