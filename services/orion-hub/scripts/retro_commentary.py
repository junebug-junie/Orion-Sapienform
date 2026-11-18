"""Orion speaks while a person plays (game company patch 3): the line, and the chat box's frames.

Design: docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md, revision 4: sections 4 (the
line), 5 (frames for turns from the game page), 6 (Prime gets in: seams 1-3) and 8 (settled looks, the read-back of
the now). The rules are pure (orion/retro/commentary.py). This module is their I/O, in hub-app: the process that
holds the chat sockets and the outreach door. The look itself runs in the Retro gateway (scripts/retro_watch.py);
the two talk through the gateway's loopback HTTP routes and one Redis key.

A line, in order:

1. **Gate** (every TICK_SEC while a sitting is open, ``orion.retro.commentary.gate``): the commentary flag is on,
   the watch mode in force is ``on`` (watch and talk; ``quiet`` means looks only), no turn of Juniper's runs, no
   line runs, no look is in flight, the last line or chat-box exchange ended ``HUB_RETRO_LINE_MIN_GAP_SEC`` ago,
   and a note was stored since then.
2. **Bid.** One want in family ``hub.retro_commentary`` (tier 1). Only ``funded`` runs a line; anything else is
   released and logged. A stale line never runs: the want is never held.
3. **Row first.** A ``retro_line`` row with outcome ``running`` (no row, no line: fail closed), and the Redis key
   ``orion:retro:line_turn`` so the gateway bids for no look meanwhile.
4. **Frames.** The gateway's newest frames of the ring, posted to the percept store. No ring -> outcome
   ``no_frames``, no turn: no prompt ever names frames it does not carry.
5. **The now.** ``GET /temporal-self/now?organ=play`` on orion-durable-runs; up to 3 items go in the game block.
6. **Seat.** A GPU pool hold of class ``chat`` at priority ``system``, carrying the want id. The turn rides the
   hold (payload ``gpu_lease``): its pre-turn check, stance, motor and finalize all attach to it on the hold's role
   at the held route's priority (config/gpu_pool.yaml ``agent``: system). So a line never runs at ``interactive``,
   below Juniper's own turn, and the 34 C reflex refuses it like any system work.
7. **Turn.** A real unified turn through the outreach door (``EndogenousOutreach.compose``: Prime block, stance,
   motor, finalize), with the game block and the frames attached as percepts.
8. **Judge.** The reply's CHECKS block settles each shown note (``checkable``, ``verdict``, ``settled_at``) -- also
   when Orion passes. Only ``checked_holds`` looks become Temporal Self events (source kind ``retro_look``).
9. **Say or pass.** PASS: outcome ``passed``, one decision row (``orion_passed``), no act. Otherwise delivery
   claims the row (running -> delivered) under outreach's send lock; a yielded row refuses, also when the draft
   finished after the yield. Delivered: pushed to every chat socket (the game page shows it in its line strip),
   one chat history row of Orion's own (no prompt), the act ``hub.retro_commentary say``, no notification.
10. **Cites.** ``[experience:<id>]`` cites in the line go to ``POST /temporal-self/now/use``.

Juniper goes first: any message a person sends on a chat socket while a line turn runs marks the line ``yielded``
(the row first), then cancels the turn: the Hub task, the motor (``HarnessGovernorClient.cancel``) and the hold.
orion-thought (the stance call) has no cancel listener (#3658): a line cancelled mid-stance lets that one call
finish on the GPU, and nothing it returns can be said.

Every failure is logged and swallowed: no game and no chat turn ever waits for a line.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from orion.retro import commentary as C

logger = logging.getLogger("orion-hub.retro.commentary")

TICK_SEC = 15.0
OP_TIMEOUT_SEC = 4.0           # one database statement or loopback call
NOW_TIMEOUT_SEC = 5.0
FRAMES_TIMEOUT_SEC = 20.0      # the gateway decodes and posts up to 4 frames
HOT_HOLD_SEC = 300.0           # after a heat refusal, no line is bid for this long
LINE_LOCK_KEY = "orion:retro:line_turn"   # scripts/retro_watch.py LINE_LOCK_KEY
HOLDER = "orion-hub.retro_commentary"
NOW_ORGAN = "play"
CITING_PREFIX = "retro_line"

NEWEST_NOTE_SQL = ("SELECT max(looked_at) FROM retro_watch_look WHERE sitting_id = $1 AND outcome = 'stored'")
NEWEST_NOTE_TEXT_SQL = ("SELECT note FROM retro_watch_look WHERE sitting_id = $1 AND outcome = 'stored' "
                        "ORDER BY looked_at DESC LIMIT 1")
NOTES_SQL = """
SELECT look_id, extract(epoch FROM looked_at) AS looked_at, note, similar_to_previous
FROM retro_watch_look
WHERE sitting_id = $1 AND outcome = 'stored' AND looked_at > $2
ORDER BY looked_at DESC LIMIT 60
"""
LAST_ATTEMPT_SQL = ("SELECT max(coalesce(finished_at, decided_at)) FROM retro_line WHERE sitting_id = $1")
LAST_DELIVERED_SQL = ("SELECT max(decided_at) FROM retro_line WHERE sitting_id = $1 AND outcome = 'delivered'")
PREVIOUS_LINES_SQL = """
SELECT text FROM retro_line WHERE sitting_id = $1 AND outcome = 'delivered' AND text IS NOT NULL
ORDER BY decided_at DESC LIMIT 5
"""
INSERT_SQL = """
INSERT INTO retro_line (line_id, sitting_id, game_id, want_id, correlation_id, decided_at, outcome, player_stated,
                        hold_lease_id, lease_priorities)
VALUES ($1, $2, $3, $4, $5, $6, 'running', $7, NULL, $8::text[])
"""
# Juniper goes first: the row is marked before the turn is killed. Only a running line can yield.
YIELD_SQL = ("UPDATE retro_line SET outcome = 'yielded', finished_at = now(), failure = $2 "
             "WHERE line_id = $1 AND outcome = 'running' RETURNING line_id")
# Delivery's claim: only a line still running can be said. A yielded row returns nothing -> no push, no history.
CLAIM_SQL = "UPDATE retro_line SET outcome = 'delivered' WHERE line_id = $1 AND outcome = 'running' RETURNING line_id"
# The rest of the row. The outcome moves only from running (a yield or a claim already decided it otherwise).
FINISH_SQL = """
UPDATE retro_line SET
    outcome = CASE WHEN outcome = 'running' THEN $2 ELSE outcome END,
    finished_at = coalesce(finished_at, $3), failure = coalesce($4, failure), text = coalesce($5, text),
    look_ids = $6::text[], frame_hashes = $7::text[], frames_expire_at = $8, now_cites = $9::text[],
    looks_shown = $10, looks_checkable = $11, looks_contradicted = $12, stance_sec = $13, motor_sec = $14,
    finalize_sec = $15, total_sec = $16, hold_lease_id = coalesce($17, hold_lease_id),
    lease_priorities = $18::text[], message_id = coalesce($19, message_id)
WHERE line_id = $1
RETURNING outcome
"""
SETTLE_SQL = """
UPDATE retro_watch_look SET checkable = $2, verdict = $3, settled_at = $4, settled_by_line = $5
WHERE look_id = $1 AND settled_at IS NULL
"""
# A chat message within 2 minutes of a delivered line is a response to it (spec section 4, "Did the player like it").
REPLIED_SQL = """
UPDATE retro_line SET replied_at = now(), replied_by = $2
WHERE sitting_id = $1 AND outcome = 'delivered' AND replied_at IS NULL
  AND finished_at >= now() - make_interval(secs => $3)
"""
# The thumb (scripts/retro_routes.py, the gateway): only a delivered line has one; the newest thumb wins.
THUMB_SQL = """
UPDATE retro_line SET thumb = $2, thumb_by = $3, thumb_at = now()
WHERE line_id = $1 AND outcome = 'delivered'
RETURNING thumb_by
"""


def _utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(float(epoch), timezone.utc)


def _epoch(value: Any) -> Optional[float]:
    return value.timestamp() if isinstance(value, datetime) else None


def _setting(config: Any, name: str, default: float) -> float:
    value = getattr(config, name, None)
    return float(default if value is None else value)


def held_route_priority() -> str:
    """The priority every call of a held turn asks the pool for: the route a held turn names
    (orion.llm.resource_lease.GPU_LEASE_ROUTE) in config/gpu_pool.yaml. The line refuses to start if this is ever
    ``interactive`` (spec section 4: lines do not ship at interactive)."""
    from orion.gpu_pool.config import load_pool_config
    from orion.llm.resource_lease import GPU_LEASE_ROUTE

    try:
        route = load_pool_config().routes.get(GPU_LEASE_ROUTE)
    except Exception:  # noqa: BLE001 -- no table in this image: the CI test pins it
        logger.warning("retro_commentary_route_table_unread", exc_info=True)
        return "unknown"
    return str(getattr(route, "priority", "") or "unknown")


@dataclass
class StageClock:
    """Seconds by stage from the turn's own cockpit hops (stance = thought_rpc started -> done; the harness leg =
    harness_dispatch started -> the turn's end; motor = the governor's own FCC seconds; finalize = harness leg minus
    motor, governor overhead included)."""
    clock: Callable[[], float] = time.monotonic
    marks: dict = field(default_factory=dict)

    def note(self, frames: list) -> None:
        now = self.clock()
        for f in frames or []:
            hop = f.get("hop") if isinstance(f, dict) else None
            if not isinstance(hop, dict):
                continue
            stage, status = hop.get("stage"), hop.get("status")
            if stage in ("thought_rpc", "harness_dispatch") and status == "started":
                self.marks.setdefault(f"{stage}:start", now)
            elif stage == "thought_rpc" and status in ("ok", "failed", "skipped"):
                self.marks.setdefault("thought_rpc:end", now)

    def seconds(self, end: float, motor_sec: Optional[float]) -> dict:
        m = self.marks
        stance = (m["thought_rpc:end"] - m["thought_rpc:start"]) if {"thought_rpc:start", "thought_rpc:end"} <= set(m) else None
        harness = (end - m["harness_dispatch:start"]) if "harness_dispatch:start" in m else None
        motor = float(motor_sec) if isinstance(motor_sec, (int, float)) else None
        finalize = (harness - motor) if harness is not None and motor is not None else None
        r = lambda x: None if x is None else round(max(0.0, x), 2)  # noqa: E731
        return {"stance_sec": r(stance), "motor_sec": r(motor), "finalize_sec": r(finalize)}


@dataclass
class Running:
    line_id: str
    sitting_id: str
    correlation_id: str
    task: Optional[asyncio.Task] = None
    yielded: bool = False


class CommentaryService:
    """One per hub-app process. Every provider is asked again on each use (Hub opens them in the background)."""

    def __init__(self, config: Any, *, bus_provider: Callable[[], Any], pool_provider: Callable[[], Any],
                 outreach_provider: Callable[[], Any], clock: Callable[[], float] = time.time,
                 bid: Optional[Callable[..., Awaitable[Any]]] = None,
                 release: Optional[Callable[[str], Awaitable[Any]]] = None,
                 seat: Optional[Callable[..., Any]] = None,
                 gateway_get: Optional[Callable[..., Awaitable[Any]]] = None,
                 read_now: Optional[Callable[[str], Awaitable[Any]]] = None,
                 record_use: Optional[Callable[..., Awaitable[Any]]] = None,
                 publish_act: Optional[Callable[..., Awaitable[bool]]] = None,
                 cancel_motor: Optional[Callable[[str], Awaitable[Any]]] = None):
        self.config = config
        self.enabled = bool(getattr(config, "HUB_RETRO_COMMENTARY_ENABLED", True))
        self.min_gap = _setting(config, "HUB_RETRO_LINE_MIN_GAP_SEC", C.LINE_MIN_GAP_SEC)
        self.turn_timeout = _setting(config, "HUB_RETRO_LINE_TIMEOUT_SEC", 240.0)
        self.seat_wait = _setting(config, "HUB_RETRO_LINE_SEAT_WAIT_SEC", 30.0)
        self.bus_provider, self.pool_provider, self.outreach_provider = bus_provider, pool_provider, outreach_provider
        self.clock = clock
        self._bid = bid or self._default_bid
        self._release = release or self._default_release
        self._seat = seat or self._default_seat
        self._gateway_get = gateway_get or self._default_gateway_get
        self._read_now = read_now or self._default_read_now
        self._record_use = record_use or self._default_record_use
        self._publish_act = publish_act or self._default_publish_act
        self._cancel_motor = cancel_motor or self._default_cancel_motor
        self.running: Optional[Running] = None
        self._sitting_id: Optional[str] = None
        self._last_exchange_at: Optional[float] = None
        self._hot_until = 0.0
        self.last_gate: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._jobs: set[asyncio.Task] = set()
        self._logged: dict[str, float] = {}

    # --- lifecycle ----------------------------------------------------------------------------------
    async def start(self) -> None:
        if not self.enabled:
            logger.info("retro_commentary disabled")
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="hub-retro-commentary")
            logger.info("retro_commentary started tick=%.0fs min_gap=%.0fs", TICK_SEC, self.min_gap)

    async def stop(self) -> None:
        if self.running is not None:
            await self.yield_line("shutdown")
        for t in [self._task, *self._jobs]:
            if t is not None and not t.done():
                t.cancel()
        await asyncio.gather(*[t for t in [self._task, *self._jobs] if t is not None], return_exceptions=True)

    def _spawn(self, coro: Awaitable[Any]) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._jobs.add(task)
        task.add_done_callback(self._jobs.discard)
        return task

    def _warn(self, event: str, *args: Any) -> None:
        now = time.monotonic()
        if now - self._logged.get(event, -60.0) >= 60.0:
            self._logged[event] = now
            logger.warning(event + (" " + " ".join(str(a) for a in args) if args else ""), exc_info=True)

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(TICK_SEC)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 -- the next tick tries again
                self._warn("retro_commentary_tick_failed")

    # --- the database -------------------------------------------------------------------------------
    async def _db(self, method: str, sql: str, *values: Any, quiet: Optional[str] = None) -> Any:
        try:
            pool = self.pool_provider()
            if pool is None:
                raise LookupError("no_pool")

            async def run() -> Any:
                async with pool.acquire() as conn:
                    return await getattr(conn, method)(sql, *values)
            return await asyncio.wait_for(run(), OP_TIMEOUT_SEC)
        except Exception:  # noqa: BLE001
            if quiet is None:
                raise
            self._warn(quiet)
            return None

    # --- reads --------------------------------------------------------------------------------------
    async def _sitting(self) -> Optional[dict]:
        from scripts.retro_sitting import read_sitting

        bus = self.bus_provider()
        try:
            redis = bus.redis
        except Exception:  # noqa: BLE001 -- no bus yet
            return None
        data = await read_sitting(redis)
        return data if data and data.get("ended_at") is None and data.get("sitting_id") else None

    async def _watch(self) -> dict:
        """The gateway's watch status (mode in force, look in flight), or {} when it cannot be read."""
        try:
            status, body = await self._gateway_get("/api/retro/sitting", auth=False, timeout=OP_TIMEOUT_SEC)
        except Exception:  # noqa: BLE001
            self._warn("retro_commentary_watch_unread")
            return {}
        watch = body.get("watch") if status == 200 and isinstance(body, dict) else None
        return watch if isinstance(watch, dict) else {}

    def _juniper_turn_running(self) -> bool:
        outreach = self.outreach_provider()
        try:
            return bool(outreach is not None and outreach._turn_in_flight())
        except Exception:  # noqa: BLE001
            return False

    async def frames(self) -> dict:
        """The gateway's newest frames for a turn (``status`` ok|no_sitting|no_frames|look_in_flight|failed:...)."""
        try:
            status, body = await self._gateway_get("/api/retro/sitting/frames?n=4", auth=True,
                                                   timeout=FRAMES_TIMEOUT_SEC)
        except Exception as exc:  # noqa: BLE001
            return {"status": f"failed:gateway_{type(exc).__name__}"}
        if status != 200 or not isinstance(body, dict):
            return {"status": f"failed:gateway_http_{status}"}
        return body

    @staticmethod
    def attachments(frames: list) -> list[dict]:
        """Turn attachments (AttachmentRefV1 shape + ``frames_expire_at``) from the gateway's frames."""
        keys = ("kind", "sha256", "mime", "bytes", "width", "height", "source_url", "frames_expire_at")
        return [{k: f[k] for k in keys if f.get(k) is not None} for f in frames if isinstance(f, dict)][:C.MAX_FRAMES]

    # --- the tick -----------------------------------------------------------------------------------
    async def tick(self) -> str:
        """One gate check; runs a line when every gate passes. Returns the gate's answer."""
        if not self.enabled:
            return C.GATE_DISABLED
        sitting = await self._sitting()
        if sitting is None:
            self._sitting_id = None
            self.last_gate = C.GATE_NO_SITTING
            return C.GATE_NO_SITTING
        sid = str(sitting["sitting_id"])
        if sid != self._sitting_id:
            self._sitting_id, self._last_exchange_at = sid, None
        watch = await self._watch()
        newest = _epoch(await self._db("fetchval", NEWEST_NOTE_SQL, sid, quiet="retro_commentary_notes_unread"))
        last_attempt = _epoch(await self._db("fetchval", LAST_ATTEMPT_SQL, sid, quiet="retro_commentary_lines_unread"))
        spoke = max([t for t in (last_attempt, self._last_exchange_at, float(sitting.get("started_at") or 0)) if t])
        now = self.clock()
        result = C.gate(enabled=self.enabled, sitting_open=True, mode=watch.get("mode"),
                        juniper_turn_running=self._juniper_turn_running(),
                        line_running=self.running is not None or bool(watch.get("looking")),
                        now=now, last_spoke_at=spoke, newest_note_at=newest, hot=now < self._hot_until,
                        min_gap_sec=self.min_gap)
        if result == C.GATE_LINE and watch.get("looking") and self.running is None:
            result = C.GATE_LOOK
        if result != self.last_gate:
            logger.info("retro_commentary_gate sitting=%s gate=%s mode=%s", sid, result, watch.get("mode"))
        self.last_gate = result
        if result == C.GATE_READY:
            await self.run_line(sitting)
        return result

    # --- the line -----------------------------------------------------------------------------------
    def _want(self, want_id: str, sitting_id: str) -> Any:
        from orion.schemas.physiology_want import WantV1, WantWindowV1

        now = _utc(self.clock())
        return WantV1(want_id=want_id, family=C.FAMILY, producer=C.BINDING_PRODUCER, verb=C.VERB, tier=C.TIER,
                      object_ref=C.OBJECT_PREFIX + sitting_id, correlation_id=want_id, opened_at=now,
                      window=WantWindowV1(latest=now + timedelta(seconds=C.DEFERRABLE_WINDOW_S)), lanes=["chat"])

    async def run_line(self, sitting: dict) -> Optional[str]:
        """One line attempt. Returns the line's outcome, or None when no row was written (bid not funded,
        no database)."""
        sid = str(sitting["sitting_id"])
        priority = held_route_priority()
        if priority == "interactive" or C.POOL_PRIORITY == "interactive":
            logger.error("retro_commentary_refused reason=held_route_interactive route_priority=%s", priority)
            return None
        line_id = str(uuid.uuid4())
        corr = line_id
        # The object text is the newest note (spec section 9), so the market's care term can weigh it.
        object_text = str(sitting.get("game_title") or sitting.get("game_id") or "a game")
        newest_note = await self._db("fetchval", NEWEST_NOTE_TEXT_SQL, sid, quiet="retro_commentary_notes_unread")
        if newest_note:
            object_text = f"{object_text}: {newest_note}"
        decision = await self._bid(self._want(line_id, sid), object_text)
        outcome = getattr(decision, "outcome", None)
        if outcome != "funded":
            reason = f"{outcome}:{getattr(decision, 'reason', None) or '-'}" if decision is not None else "no_answer"
            logger.info("retro_commentary_bid_refused sitting=%s want_id=%s reason=%s", sid, line_id, reason)
            if decision is not None:
                await self._release(line_id)
            return None
        priorities = [C.POOL_PRIORITY, priority]
        inserted = await self._db("execute", INSERT_SQL, line_id, sid, sitting.get("game_id"), line_id, corr,
                                  _utc(self.clock()), sitting.get("player_stated"), priorities,
                                  quiet="retro_commentary_row_failed")
        if inserted is None:
            await self._release(line_id)       # no row, no line: every delivered line has its row
            return None
        self.running = Running(line_id=line_id, sitting_id=sid, correlation_id=corr)
        await self._set_lock(line_id)
        started = time.monotonic()
        result: dict = {"outcome": "failed", "failure": None}
        try:
            self.running.task = asyncio.ensure_future(self._line_body(sitting, line_id, corr, result, started))
            await self.running.task
        except asyncio.CancelledError:
            if not (self.running and self.running.yielded):
                raise
            result.update(outcome="yielded", failure=result.get("failure") or "yielded")
        except Exception as exc:  # noqa: BLE001 -- the row is still finished
            result["failure"] = f"error:{type(exc).__name__}"
            self._warn("retro_commentary_line_error", line_id)
        finally:
            yielded = bool(self.running and self.running.yielded)
            self.running = None
            await self._clear_lock(line_id)
            await self._release(line_id)
            final = await self._finish_row(line_id, result, started)
            self._last_exchange_at = max(self._last_exchange_at or 0.0, self.clock())
            logger.info("retro_commentary_line line_id=%s sitting=%s outcome=%s failure=%s yielded=%s total_sec=%.1f",
                        line_id, sid, final or result.get("outcome"), result.get("failure"), yielded,
                        time.monotonic() - started)
        return final or result.get("outcome")

    async def _line_body(self, sitting: dict, line_id: str, corr: str, result: dict, started: float) -> None:
        sid = str(sitting["sitting_id"])
        frames = await self.frames()
        if frames.get("status") != "ok" or not frames.get("frames"):
            status = str(frames.get("status") or "no_frames")
            result.update(outcome="no_frames" if status in ("no_frames", "no_sitting") else "failed", failure=status)
            return
        refs = self.attachments(frames["frames"])
        frame_times = [float(f["kept_at"]) for f in frames["frames"] if isinstance(f.get("kept_at"), (int, float))]
        result.update(frame_hashes=[r["sha256"] for r in refs], frames_expire_at=frames.get("frames_expire_at"))
        since = _epoch(await self._db("fetchval", LAST_DELIVERED_SQL, sid, quiet="retro_commentary_lines_unread"))
        since = since or float(sitting.get("started_at") or 0.0)
        rows = await self._db("fetch", NOTES_SQL, sid, _utc(since), quiet="retro_commentary_notes_unread") or []
        notes = [C.Note(look_id=r["look_id"], looked_at=float(r["looked_at"]), note=str(r["note"] or ""),
                        similar_to_previous=r["similar_to_previous"]) for r in reversed(list(rows))]
        groups = C.group_notes(notes, frame_times)
        previous = [r["text"] for r in (await self._db("fetch", PREVIOUS_LINES_SQL, sid,
                                                       quiet="retro_commentary_lines_unread") or [])][::-1]
        now_obj = await self._read_now(line_id)
        now_items = list(getattr(now_obj, "items", None) or [])[:C.MAX_NOW_ITEMS]
        from orion.temporal_self.now import EXPERIENCE_PREFIX

        now_lines = [f"[{EXPERIENCE_PREFIX}{i.event_id}] {i.organ}: {i.label}" for i in now_items]
        prompt = C.build_prompt(
            game_title=sitting.get("game_title") or sitting.get("game_id"),
            minutes_played=max(0.0, float(sitting.get("last_input_at") or 0) - float(sitting.get("started_at") or 0)) / 60.0,
            saves=int(sitting.get("saves") or 0), loads=int(sitting.get("loads") or 0),
            player_stated=sitting.get("player_stated"), groups=groups, n_frames=len(refs),
            previous_lines=previous, now_lines=now_lines)
        result["look_ids"] = [n.look_id for g in groups for n in g.notes]
        outreach = self.outreach_provider()
        bus = self.bus_provider()
        if outreach is None or bus is None:
            result.update(outcome="failed", failure="no_outreach" if outreach is None else "no_bus")
            return
        clock = StageClock()

        async def sink(cockpit_frames: list) -> None:
            clock.note(cockpit_frames)
            from orion.hub.turn_orchestrator import publish_cockpit_frames

            await publish_cockpit_frames(bus, cockpit_frames)

        try:
            seat_cm = self._seat(bus, line_id)
            lease = await seat_cm.__aenter__()
        except Exception as exc:  # noqa: BLE001 -- LeaseUnavailable, a pool timeout
            reason = f"pool:{getattr(exc, 'reason', None) or type(exc).__name__}"
            if "cabinet_hot" in reason:
                self._hot_until = self.clock() + HOT_HOLD_SEC
                reason = "heat:" + reason
            result.update(outcome="failed", failure=reason)
            return
        error: Optional[BaseException] = None
        try:
            from orion.schemas.gpu_pool import GpuLeaseRefV1

            g = lease.grant
            ref = GpuLeaseRefV1(lease_id=g.lease_id, generation=g.generation, role=g.role, holder=HOLDER)
            result["hold_lease_id"] = g.lease_id
            text, debug = await outreach.compose(
                prompt, corr, lane="chat", attachments=refs, gpu_lease=ref, source=C.SOURCE, cockpit_sink=sink,
                session_id=outreach.juniper_session_id(), timeout_sec=self.turn_timeout)
            result["stages"] = clock.seconds(time.monotonic(), (debug or {}).get("harness_fcc_elapsed_sec"))
        except BaseException as exc:
            error = exc
            raise
        finally:
            with contextlib.suppress(Exception):
                await seat_cm.__aexit__(type(error) if error else None, error, error.__traceback__ if error else None)
        if self.running is not None and self.running.yielded:
            result.update(outcome="yielded", failure="yielded")
            return
        if not str(text or "").strip():
            result.update(outcome="failed", failure=str((debug or {}).get("error") or "empty_generation"))
            return
        reply = C.parse_reply(text)
        settlements = C.settle(groups, reply.judgments, frame_times)
        await self._settle(settlements, line_id)
        rate = C.error_rate(settlements)
        result.update(looks_shown=rate["shown"], looks_checkable=rate["checkable"],
                      looks_contradicted=rate["contradicted"])
        from scripts.endogenous_outreach import is_pass_response

        meta = {"retro_sitting_id": sid, "line_id": line_id}
        if not reply.line or is_pass_response(reply.line):
            result.update(outcome="passed", failure=None)
            outreach.record_blocked("orion_passed", correlation_id=corr, source=C.SOURCE, extra=meta)
            return
        from orion.curiosity.now_section import extract_cites

        shown_ids = [i.event_id for i in now_items]
        cites = extract_cites(reply.line, shown=shown_ids)
        result["now_cites"] = cites

        async def claim() -> bool:
            if self.running is None or self.running.yielded:
                return False
            return bool(await self._db("fetchval", CLAIM_SQL, line_id, quiet="retro_commentary_claim_failed"))

        sent = await outreach.deliver_line(text=reply.line, correlation_id=corr, source_tag=C.SOURCE, claim=claim,
                                           session_id=outreach.juniper_session_id(),
                                           model=(debug or {}).get("fcc_model_label"),
                                           push_extra=meta, meta=meta)
        if sent.get("reason") == "sent":
            result.update(outcome="delivered", failure=None, text=reply.line, message_id=sent.get("message_id"))
            await self._publish_act(bus, line_id=line_id, sitting_id=sid, text=reply.line)
            if cites and getattr(now_obj, "shown_id", None):
                await self._record_use(now_obj.shown_id, f"{CITING_PREFIX}:{line_id}", cites)
        elif sent.get("reason") == "yielded":
            result.update(outcome="yielded", failure="yielded_before_delivery")
        else:
            result.update(outcome="failed", failure=str(sent.get("reason") or "not_sent"))

    async def _settle(self, settlements: list, line_id: str) -> None:
        if not settlements:
            return
        at = _utc(self.clock())
        await self._db("executemany", SETTLE_SQL,
                       [(s.look_id, s.checkable, s.verdict, at, line_id) for s in settlements],
                       quiet="retro_commentary_settle_failed")

    async def _finish_row(self, line_id: str, result: dict, started: float) -> Optional[str]:
        stages = result.get("stages") or {}
        expires = result.get("frames_expire_at")
        if isinstance(expires, str):
            try:
                expires = datetime.fromisoformat(expires.replace("Z", "+00:00"))
            except ValueError:
                expires = None
        return await self._db(
            "fetchval", FINISH_SQL, line_id, result.get("outcome") or "failed", _utc(self.clock()),
            result.get("failure"), result.get("text"), list(result.get("look_ids") or []),
            list(result.get("frame_hashes") or []), expires, list(result.get("now_cites") or []),
            int(result.get("looks_shown") or 0), int(result.get("looks_checkable") or 0),
            int(result.get("looks_contradicted") or 0), stages.get("stance_sec"), stages.get("motor_sec"),
            stages.get("finalize_sec"), round(time.monotonic() - started, 2), result.get("hold_lease_id"),
            [C.POOL_PRIORITY, held_route_priority()], result.get("message_id"),
            quiet="retro_commentary_finish_failed")

    # --- Juniper goes first ---------------------------------------------------------------------------
    async def yield_line(self, why: str = "juniper_spoke") -> bool:
        """Mark the running line yielded (its row FIRST), then cancel it: the Hub task, the motor, the hold.
        True when a line was running."""
        run = self.running
        if run is None:
            return False
        run.yielded = True
        await self._db("fetchval", YIELD_SQL, run.line_id, why, quiet="retro_commentary_yield_failed")
        if run.task is not None and not run.task.done():
            run.task.cancel()
        with contextlib.suppress(Exception):
            await self._cancel_motor(run.correlation_id)
        logger.info("retro_commentary_yielded line_id=%s why=%s", run.line_id, why)
        return True

    def note_human_message(self, *, surface: Optional[str] = None, player: Optional[str] = None) -> None:
        """A person sent a chat message (any socket). Sync, never raises: the yield runs in the background, so
        the person's own turn never waits for it. Also: a reply within 2 min marks the newest line answered, and
        the exchange restarts the line gap."""
        if not self.enabled:
            return
        try:
            self._last_exchange_at = self.clock()
            if self.running is not None:
                self._spawn(self.yield_line("juniper_spoke"))
            sid = self._sitting_id
            if sid:
                who = "juniper" if (player == "juniper" or surface != "retro") else "guest"
                self._spawn(self._db("execute", REPLIED_SQL, sid, who, C.REPLY_WINDOW_SEC,
                                     quiet="retro_commentary_reply_failed"))
        except Exception:  # noqa: BLE001
            self._warn("retro_commentary_note_message_failed")

    # --- the lock the gateway reads -------------------------------------------------------------------
    async def _set_lock(self, line_id: str) -> None:
        try:
            redis = self.bus_provider().redis
            await asyncio.wait_for(redis.set(LINE_LOCK_KEY, line_id, ex=int(self.turn_timeout + 120)), OP_TIMEOUT_SEC)
        except Exception:  # noqa: BLE001 -- the line's own hold still keeps a look's hold behind it
            self._warn("retro_commentary_lock_failed")

    async def _clear_lock(self, line_id: str) -> None:
        try:
            redis = self.bus_provider().redis
            await asyncio.wait_for(redis.delete(LINE_LOCK_KEY), OP_TIMEOUT_SEC)
        except Exception:  # noqa: BLE001 -- the key expires on its own
            self._warn("retro_commentary_unlock_failed")

    # --- default I/O (tests replace each) ---------------------------------------------------------------
    @staticmethod
    async def _default_bid(want: Any, object_text: str) -> Any:
        from orion.physiology.client import abid

        return await abid(want, timeout_sec=5.0, object_text=object_text)

    @staticmethod
    async def _default_release(want_id: str) -> Any:
        from orion.physiology.client import arelease

        with contextlib.suppress(Exception):
            return await asyncio.wait_for(arelease(want_id), 3.0)
        return None

    def _default_seat(self, bus: Any, want_id: str) -> Any:
        from orion.gpu_pool.client import gpu_lease

        return gpu_lease(bus, work_class=C.POOL_CLASS, holder=HOLDER, priority=C.POOL_PRIORITY, kind="hold",
                         deadline_sec=self.seat_wait, turn_correlation_id=want_id,
                         request_id=f"retro_commentary:{want_id}")

    async def _default_gateway_get(self, path: str, *, auth: bool, timeout: float) -> tuple[int, Any]:
        import httpx

        port = int(getattr(self.config, "HUB_RETRO_GATEWAY_PORT", 8092) or 8092)
        headers = {}
        if auth:
            token = Path(str(getattr(self.config, "HUB_RETRO_AUTH_DIR", "/run/orion-retro-auth")), "hub.token").read_text().strip()
            headers["Authorization"] = "Bearer " + token
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"http://127.0.0.1:{port}{path}", headers=headers)
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, None

    def _durable_runs_url(self) -> str:
        return str(getattr(self.config, "HUB_CURIOSITY_DURABLE_RUNS_URL", "") or "").strip().rstrip("/")

    async def _default_read_now(self, line_id: str) -> Any:
        """``GET /temporal-self/now?organ=play`` (Orion's now, read back by the line), or None."""
        root = self._durable_runs_url()
        if not root:
            return None
        import httpx
        from orion.curiosity.now_section import parse_now

        try:
            async with httpx.AsyncClient(timeout=NOW_TIMEOUT_SEC) as client:
                resp = await client.get(f"{root}/temporal-self/now",
                                        params={"organ": NOW_ORGAN, "run_ref": f"{CITING_PREFIX}:{line_id}"})
            if resp.status_code != 200:
                logger.info("retro_commentary_now_unavailable status=%s", resp.status_code)
                return None
            now = parse_now(resp.json())
        except Exception as exc:  # noqa: BLE001
            logger.info("retro_commentary_now_unavailable err=%s", type(exc).__name__)
            return None
        logger.info("retro_commentary_now_read line_id=%s shown_id=%s items=%d", line_id,
                    getattr(now, "shown_id", None), len(getattr(now, "items", None) or []))
        return now

    async def _default_record_use(self, shown_id: str, citing_ref: str, cites: list) -> Any:
        """``POST /temporal-self/now/use`` with the line's cites (durable-runs keeps only ids that were shown)."""
        root = self._durable_runs_url()
        if not (root and shown_id and cites):
            return None
        import httpx

        try:
            async with httpx.AsyncClient(timeout=NOW_TIMEOUT_SEC) as client:
                resp = await client.post(f"{root}/temporal-self/now/use",
                                         json={"shown_id": shown_id, "citing_ref": citing_ref, "cites": list(cites)})
            logger.info("retro_commentary_now_cites citing=%s cites=%d status=%s", citing_ref, len(cites),
                        resp.status_code)
            return resp.status_code
        except Exception as exc:  # noqa: BLE001
            logger.info("retro_commentary_now_cites_failed err=%s", type(exc).__name__)
            return None

    @staticmethod
    async def _default_publish_act(bus: Any, *, line_id: str, sitting_id: str, text: str) -> bool:
        from scripts.act_emit import publish_line_said

        return await publish_line_said(bus, line_id=line_id, sitting_id=sitting_id, text=text)

    async def _default_cancel_motor(self, correlation_id: str) -> None:
        from scripts.harness_governor_client import HarnessGovernorClient

        outreach = self.outreach_provider()
        bus = getattr(outreach, "_harness_rpc_bus", None) or self.bus_provider()
        if bus is not None:
            await HarnessGovernorClient(bus).cancel(correlation_id=correlation_id, reason="retro_line_yielded")

    # --- frames for a chat-box turn from the game page ---------------------------------------------------
    async def chat_frames(self) -> list[dict]:
        """Percept attachments for a turn sent from the game page (surface ``retro``), or [] (no ring, a look in
        flight, the gateway unreachable: the turn runs without frames and the prompt names none)."""
        frames = await self.frames()
        if frames.get("status") != "ok":
            logger.info("retro_commentary_chat_frames status=%s", frames.get("status"))
            return []
        return self.attachments(frames.get("frames") or [])


GUEST_PREFIX = ("(Typed in the game page's chat box by the person playing, who has not said they are Juniper. "
                "Use no name. Say only what is fine for a child to read.)\n\n")


async def prepare_retro_turn(data: dict, transcript: str, *, bus: Any, commentary: Optional[CommentaryService]
                             ) -> tuple[dict, Optional[str], str, str]:
    """A chat-box message from the game page (surface ``retro``), before its turn runs (spec section 5).

    Who typed is what the page was told, read from the open sitting at THIS message (Redis, the gateway's own copy),
    so a flip mid-sitting changes only later messages. Juniper stated: her normal turn. Anyone else (a guest, or
    nobody stated): utterance origin None, chat history source ``retro_guest`` (the three "Juniper spoke" readers
    never match it), and the message says who typed it. Both get the game's newest frames as percepts, when there
    is a ring. Returns (payload, utterance_origin, user message, player)."""
    from scripts.retro_sitting import read_sitting

    data = dict(data or {})
    ctx = dict(data.get("surface_context") or {})
    sitting = None
    try:
        sitting = await read_sitting(bus.redis)
    except Exception:  # noqa: BLE001 -- no bus: unknown reads as a guest
        sitting = None
    if sitting is not None and sitting.get("ended_at") is not None:
        sitting = None
    player = "juniper" if (sitting or {}).get("player_stated") == "juniper" else "guest"
    ctx.update(surface="retro", player=player)
    if sitting is not None:
        ctx["retro_sitting_id"] = str(sitting.get("sitting_id"))
    data["surface_context"] = ctx
    # Server side only: a client can never pick its own history source or turn source.
    data.pop("source", None)
    data.pop("chat_history_source", None)
    origin: Optional[str] = "juniper"
    if player != "juniper":
        data["chat_history_source"] = C.GUEST_SOURCE
        origin = None
        transcript = GUEST_PREFIX + str(transcript or "")
    frames = await commentary.chat_frames() if (commentary is not None and sitting is not None) else []
    if frames:
        own = [a for a in (data.get("attachments") or []) if isinstance(a, dict)]
        room = max(0, C.MAX_FRAMES - len(frames))
        data["attachments"] = own[:room] + frames
    logger.info("retro_chat_turn player=%s sitting=%s frames=%d", player, ctx.get("retro_sitting_id"), len(frames))
    return data, origin, transcript, player


__all__ = ["CommentaryService", "GUEST_PREFIX", "prepare_retro_turn", "LINE_LOCK_KEY", "StageClock", "THUMB_SQL", "held_route_priority"]
