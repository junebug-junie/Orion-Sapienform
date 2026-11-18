"""Hub's record of who plays on the Retro service: the sitting (game company design, patch 1).

Design: docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md, section 1.
The rules of a sitting are pure (orion/retro/sitting.py). This module is the I/O around them, in
the process that relays the browser's input (the Retro gateway; Hub's own process when it serves
/api/retro itself):

- ``note_input`` is called by the relay for every input message. It never raises and never waits.
- the open sitting is kept in Redis (``orion:retro:sitting``, JSON, refreshed every tick) so Hub's
  chat turn and the office reach-out can read it from the other process;
- ``retro_sitting`` (Postgres) gets the row once a minute while the sitting is open, and at its close;
- a closed sitting of ``HUB_RETRO_SITTING_MIN_SEC`` or more becomes one ``retro_play_sessions`` row
  of kind ``human_play``, which the Temporal Self source ``retro_play`` turns into a play arc;
- ``recover`` repairs what a crash or a database fault left behind: a row that never closed, or a
  closed sitting with no play row. It runs when the gateway's pool opens and when a sitting opens.

It credits Orion with nothing: orion_frames is 0, no memory card is written and no act is opened
(a person playing is not an act of Orion's). Who plays is only what the page was told.

Orion's own looks at the screen (game company patch 2) are kept by scripts/retro_watch.py, not here; the
only thing this record takes from them is a count: ``looks_for`` tells the close how many notes Orion
stored in the sitting, and the play row's summary says "Orion watched: N notes." when N > 0.

No model is called here. Every failure is logged and swallowed: a game never waits for a record.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from orion.retro.sitting import (
    END_SESSION_STOPPED, GAP_SEC_DEFAULT, MIN_SEC_DEFAULT, PLAYERS, SITTING_KEY, Sitting, SittingTracker,
    parse_input_message, who_words, worth_recording,
)
from orion.schemas.retro_play import RetroPlaySessionSummaryV1
from orion.temporal_self.stance_cue import span_words
from scripts.retro_memory_export import TABLE as PLAY_TABLE, insert_sql, row_values

logger = logging.getLogger("orion-hub.retro.sitting")

TICK_SEC = 15.0            # how often the open sitting is checked for its end and its Redis copy refreshed
PERSIST_SEC = 60.0         # how often the open sitting's row is written
KEY_TTL_SEC = 180          # a crashed process's key expires on its own (12 missed ticks)
OP_TIMEOUT_SEC = 2.0       # one Redis call; a database call (connection and statement together) gets twice this
LOG_EVERY_SEC = 60.0       # a fault that repeats per input message is logged once in this time
RECOVER_DAYS = 7           # how far back ``recover`` looks for a closed sitting with no play row
END_RECOVERED = "recovered"
ROW_PREFIX = "sitting:"    # retro_play_sessions.session_id for a human_play row

UPSERT_SQL = """
INSERT INTO retro_sitting (sitting_id, retro_session_id, game_id, game_title, started_at, last_input_at,
                           ended_at, end_reason, mask_changes, saves, loads, pauses, player_stated, backfilled, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, now())
ON CONFLICT (sitting_id) DO UPDATE SET
    game_id = COALESCE(EXCLUDED.game_id, retro_sitting.game_id),
    game_title = COALESCE(EXCLUDED.game_title, retro_sitting.game_title),
    last_input_at = EXCLUDED.last_input_at, ended_at = EXCLUDED.ended_at, end_reason = EXCLUDED.end_reason,
    mask_changes = EXCLUDED.mask_changes, saves = EXCLUDED.saves, loads = EXCLUDED.loads, pauses = EXCLUDED.pauses,
    player_stated = EXCLUDED.player_stated, updated_at = now()
"""

# The play row of a sitting: the exporter's own column list (scripts/retro_memory_export.py) plus the
# one column only a sitting fills. One list, so the two writers of retro_play_sessions cannot drift.
PLAY_EXTRA = ("player_stated", "looks")
PLAY_INSERT_SQL = insert_sql(extra=PLAY_EXTRA)

# A row that never closed belongs to a process that died: this process holds the only live tracker,
# and $1 is the sitting it has open now (or '').
RECOVER_OPEN_SQL = """
UPDATE retro_sitting SET ended_at = last_input_at, end_reason = '""" + END_RECOVERED + """', updated_at = now()
WHERE ended_at IS NULL AND NOT backfilled AND sitting_id <> $1
RETURNING sitting_id
"""
# Closed, long enough, and no play row: the close could not write it (database down, shutdown cut short).
RECOVER_MISSING_SQL = f"""
SELECT s.sitting_id, s.retro_session_id, s.game_id, s.game_title, s.started_at, s.last_input_at, s.ended_at,
       s.end_reason, s.mask_changes, s.saves, s.loads, s.pauses, s.player_stated, s.notes_stored
FROM retro_sitting s
WHERE NOT s.backfilled AND s.ended_at IS NOT NULL AND s.ended_at >= now() - make_interval(days => $2)
  AND s.ended_at - s.started_at >= make_interval(secs => $1)
  AND NOT EXISTS (SELECT 1 FROM {PLAY_TABLE} p WHERE p.session_id = '{ROW_PREFIX}' || s.sitting_id)
ORDER BY s.ended_at
"""


def _ts(epoch: Optional[float]) -> Optional[datetime]:
    return None if epoch is None else datetime.fromtimestamp(float(epoch), timezone.utc)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def sitting_summary(s: Sitting, looks: Optional[int] = None) -> str:
    """One deterministic sentence for the play arc. Counts only; no model writes it. ``looks``: notes
    Orion stored while watching (scripts/retro_watch.py); said only when there is at least one."""
    title = s.game_title or s.game_id or "a game"
    text = f"{who_words(s.player_stated)} played {title} for {span_words(s.duration_sec)}."
    extras = [_plural(n, w) for n, w in ((s.saves, "save"), (s.loads, "load")) if n]
    if extras:
        text += f" {' and '.join(extras).capitalize()}."
    if looks:
        text += f" Orion watched: {_plural(int(looks), 'note')}."
    return text


def play_row(s: Sitting, looks: Optional[int] = None) -> RetroPlaySessionSummaryV1:
    """The ``retro_play_sessions`` row of one closed sitting. Frames are not counted (basis
    ``human_input``): human_frames and orion_frames are 0, and no share is claimed."""
    assert s.ended_at is not None
    return RetroPlaySessionSummaryV1(
        session_id=ROW_PREFIX + s.sitting_id, retro_session_id=s.retro_session_id, lineage_id=None,
        game_id=s.game_id or "unknown", game_title=s.game_title or s.game_id or "a game", kind="human_play",
        started_at=_ts(s.started_at), ended_at=_ts(s.ended_at), duration_sec=round(s.duration_sec, 1),
        end_reason=s.end_reason, control=[], control_basis="human_input", human_may_be_in_control=True,
        human_frames=0, orion_frames=0, human_share=None, played_together=False,
        predictions_met=0, predictions_refuted=0, human_interventions=0,
        summary=sitting_summary(s, looks), player_stated=s.player_stated if s.player_stated in PLAYERS else None,
        looks=int(looks) if looks is not None else None,
    )


def _play_values(row: RetroPlaySessionSummaryV1) -> list[Any]:
    return row_values(row, extra=PLAY_EXTRA)


def _setting(config: Any, name: str, default: float) -> float:
    """A number from settings. 0 is a value (``or default`` would read it as unset)."""
    value = getattr(config, name, None)
    return float(default if value is None else value)


def _epoch(ts: Any) -> Optional[float]:
    return ts.timestamp() if isinstance(ts, datetime) else None


def sitting_values(s: Sitting) -> list[Any]:
    return [s.sitting_id, s.retro_session_id, s.game_id, s.game_title, _ts(s.started_at), _ts(s.last_input_at),
            _ts(s.ended_at), s.end_reason, s.mask_changes, s.saves, s.loads, s.pauses, s.player_stated, s.backfilled]


def sitting_line(data: Any, now: Optional[float] = None) -> Optional[str]:
    """One plain sentence about an open sitting, for a prompt (the chat turn's situation block, the
    office reach-out). None when ``data`` is not an open sitting. Takes the Redis JSON as parsed."""
    if not isinstance(data, dict) or data.get("ended_at") is not None:
        return None
    try:
        started, last = float(data["started_at"]), float(data["last_input_at"])
    except (KeyError, TypeError, ValueError):
        return None
    now = time.time() if now is None else now
    title = str(data.get("game_title") or data.get("game_id") or "a game")
    who = who_words(data.get("player_stated"))
    known = "" if data.get("player_stated") else " You do not know who holds the controller."
    idle = max(0.0, now - last)
    tail = "" if idle < 60 else f" The last button press was {span_words(idle)} ago."
    return (f"{who} is playing {title} on your Retro service, for {span_words(max(0.0, last - started))} so far."
            f"{tail}{known} This line comes from the controller's input only: it holds nothing you saw on the screen.")


async def read_sitting(redis: Any) -> Optional[dict]:
    """The open sitting from Redis, parsed, or None (also on any read fault)."""
    try:
        raw = await asyncio.wait_for(redis.get(SITTING_KEY), OP_TIMEOUT_SEC)
    except Exception:  # noqa: BLE001 -- a reader of the sitting never fails its caller
        return None
    if raw is None:
        return None
    try:
        data = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


async def open_sitting_line(bus: Any, now: Optional[float] = None) -> Optional[str]:
    """``sitting_line`` of the open sitting, read through a bus client's Redis. None when nobody plays,
    and on any fault (no bus, not connected, unreadable): a prompt never fails for this line."""
    try:
        redis = bus.redis
    except Exception:  # noqa: BLE001 -- no bus, or not connected
        return None
    return sitting_line(await read_sitting(redis), now)


GameResolver = Callable[[str], Awaitable[tuple[Optional[str], Optional[str]]]]


class SittingService:
    """One per process. ``redis_provider`` / ``pool_provider`` return the client or None; both are
    asked again on every use, because the gateway opens them in the background."""

    def __init__(self, config: Any, *, redis_provider: Callable[[], Any] = lambda: None,
                 pool_provider: Callable[[], Any] = lambda: None, resolve_game: Optional[GameResolver] = None,
                 clock: Callable[[], float] = time.time, tracker: Optional[SittingTracker] = None):
        self.enabled = bool(getattr(config, "HUB_RETRO_SITTING_ENABLED", True))
        self.min_sec = _setting(config, "HUB_RETRO_SITTING_MIN_SEC", MIN_SEC_DEFAULT)
        self.tracker = tracker or SittingTracker(gap_sec=_setting(config, "HUB_RETRO_SITTING_GAP_SEC", GAP_SEC_DEFAULT))
        self.redis_provider, self.pool_provider = redis_provider, pool_provider
        self.resolve_game, self.clock = resolve_game, clock
        # Set by scripts/retro_watch.py: sitting id -> notes Orion stored in it (None: the watch did not run).
        self.looks_for: Callable[[str], Optional[int]] = lambda _sid: None
        self._loop_task: Optional[asyncio.Task] = None
        self._jobs: set[asyncio.Task] = set()
        self._persisted_at = 0.0
        self._logged: dict[str, float] = {}

    def _warn(self, event: str) -> None:
        """Log a caught fault with its traceback, at most once per LOG_EVERY_SEC per event: a fault
        that repeats with every input message (up to 120 a second) must not flood the log."""
        now = time.monotonic()
        if now - self._logged.get(event, -LOG_EVERY_SEC) >= LOG_EVERY_SEC:
            self._logged[event] = now
            logger.warning(event, exc_info=True)

    # --- called from the relay and the proxy: sync, never raise, never wait ---------------------
    def note_input(self, session_id: str, raw: str) -> None:
        if not self.enabled:
            return
        try:
            parsed = parse_input_message(raw)
            if parsed is None:
                return
            port, mask, axes = parsed
            before = self.tracker.current.sitting_id if self.tracker.current else None
            closed = self.tracker.on_input(session_id, port, mask, self.clock(), axes)
            cur = self.tracker.current
            if closed or (cur is not None and cur.sitting_id != before):
                self._spawn(self._after_change(closed, cur if cur is not None and cur.sitting_id != before else None))
            elif cur is not None:
                self._ensure_loop()        # an open sitting always has its loop, also after a loop fault
        except Exception:  # noqa: BLE001 -- the relay never fails for a record
            self._warn("retro_sitting_input_failed")

    def note_disconnect(self, session_id: str) -> None:
        """An input socket of this Retro session closed: forget its button state (not the sitting)."""
        try:
            self.tracker.forget_masks(session_id)
        except Exception:  # noqa: BLE001
            self._warn("retro_sitting_disconnect_failed")

    def note_action(self, session_id: str, kind: str) -> None:
        """A proxied POST the player made: save, load, pause count; stop ends the sitting."""
        if not self.enabled:
            return
        try:
            if kind == "stop":
                closed = self.tracker.end_session(session_id, END_SESSION_STOPPED)
                if closed:
                    self._spawn(self._after_change(closed, None))
            else:
                self.tracker.on_action(session_id, kind)
        except Exception:  # noqa: BLE001
            self._warn("retro_sitting_action_failed")

    # --- the page -------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        cur = self.tracker.current
        return {"enabled": self.enabled, "sitting": cur.as_dict() if cur is not None else None}

    async def state_player(self, player: Any, sitting_id: Optional[str] = None) -> bool:
        """What the page was told. ``sitting_id`` is the sitting the page asked about: an answer to a
        question about an earlier sitting is refused, never put on the one open now."""
        cur = self.tracker.current
        if not self.enabled or cur is None or (sitting_id is not None and sitting_id != cur.sitting_id):
            return False
        if not self.tracker.state_player(player if isinstance(player, str) else None):
            return False
        logger.info("retro_sitting_player_stated sitting=%s player=%s", cur.sitting_id, player)
        # ``cur`` is held across the awaits: the sitting may close meanwhile, and its row still gets the answer.
        await self._publish()
        await self._persist(cur)
        return True

    # --- background -----------------------------------------------------------------------------
    def _ensure_loop(self) -> None:
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.ensure_future(self._loop())

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._jobs.add(task)
        task.add_done_callback(self._jobs.discard)
        self._ensure_loop()

    async def _resolve(self, s: Sitting) -> None:
        """The game's id and title, asked of Retro. Tried again at each row write until it answers."""
        if self.resolve_game is None or s.game_title:
            return
        try:
            game_id, title = await asyncio.wait_for(self.resolve_game(s.retro_session_id), 5.0)
            s.game_id, s.game_title = game_id or s.game_id, title or s.game_title
        except Exception:  # noqa: BLE001 -- a sitting without a title is still a sitting
            self._warn("retro_sitting_game_unresolved")

    async def _after_change(self, closed: list[Sitting], opened: Optional[Sitting]) -> None:
        await self._publish()      # readers first: a prompt must not say "is playing" while the database is slow
        for s in closed:
            await self._finish(s)
        if opened is not None:
            logger.info("retro_sitting_opened sitting=%s retro_session=%s", opened.sitting_id, opened.retro_session_id)
            await self._resolve(opened)
            await self._persist(opened)
            await self._publish()
            await self.recover()   # a sitting an earlier process could not record

    async def _loop(self) -> None:
        """Runs while a sitting is open; ends itself when none is. Any input starts it again."""
        try:
            while self.tracker.current is not None:
                await asyncio.sleep(TICK_SEC)
                now = self.clock()
                closed = self.tracker.tick(now)
                await self._publish()
                for s in closed:
                    # Its own task: a shutdown that cancels this loop must not cut a close in two.
                    self._spawn(self._finish(s))
                cur = self.tracker.current
                if cur is not None and now - self._persisted_at >= PERSIST_SEC:
                    await self._resolve(cur)
                    await self._persist(cur)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- the next input starts the loop again (note_input)
            self._warn("retro_sitting_loop_failed")

    async def _publish(self) -> None:
        """The open sitting into Redis, or the key removed when none is open."""
        redis = self.redis_provider()
        if redis is None:
            return
        cur = self.tracker.current
        try:
            if cur is None:
                await asyncio.wait_for(redis.delete(SITTING_KEY), OP_TIMEOUT_SEC)
            else:
                await asyncio.wait_for(redis.set(SITTING_KEY, json.dumps(cur.as_dict(), separators=(",", ":")),
                                                 ex=KEY_TTL_SEC), OP_TIMEOUT_SEC)
        except Exception:  # noqa: BLE001
            self._warn("retro_sitting_publish_failed")

    async def _db(self, method: str, sql: str, *values: Any) -> Any:
        """One statement on a pooled connection. The wait for a connection and the statement share
        one bound: a database that is restarting must not hold the loop or a shutdown."""
        pool = self.pool_provider()
        if pool is None:
            raise LookupError("no_pool")

        async def run() -> Any:
            async with pool.acquire() as conn:
                return await getattr(conn, method)(sql, *values)
        return await asyncio.wait_for(run(), OP_TIMEOUT_SEC * 2)

    async def _persist(self, s: Sitting) -> bool:
        try:
            await self._db("execute", UPSERT_SQL, *sitting_values(s))
            self._persisted_at = self.clock()
            return True
        except LookupError:
            return False
        except Exception:  # noqa: BLE001
            self._warn("retro_sitting_persist_failed")
            return False

    def _looks(self, s: Sitting) -> Optional[int]:
        try:
            n = self.looks_for(s.sitting_id)
            return int(n) if n is not None else None
        except Exception:  # noqa: BLE001 -- a count never stops a play row
            self._warn("retro_sitting_looks_unread")
            return None

    async def _play(self, s: Sitting, looks: Optional[int] = None) -> str:
        """Write the play row of a closed sitting. Safe to repeat: the row's key is the sitting.
        ``looks``: the stored notes (the live watch's count, or the row's ``notes_stored`` in recovery)."""
        if looks is None:
            looks = self._looks(s)
        try:
            return ("play_row" if await self._db("fetchval", PLAY_INSERT_SQL, *_play_values(play_row(s, looks)))
                    else "duplicate")
        except LookupError:
            return "no_pool"
        except Exception:  # noqa: BLE001
            self._warn("retro_sitting_play_row_failed")
            return "failed"

    async def _finish(self, s: Sitting) -> None:
        """A closed sitting: its row, and a play row when it was long enough to matter. What fails
        here is repaired by ``recover``."""
        await self._resolve(s)
        stored = await self._persist(s)
        recorded = await self._play(s) if worth_recording(s, self.min_sec) else "too_short"
        logger.info("retro_sitting_closed sitting=%s reason=%s duration_sec=%.0f mask_changes=%d player=%s "
                    "sitting_row=%s recorded=%s", s.sitting_id, s.end_reason, s.duration_sec, s.mask_changes,
                    s.player_stated, stored, recorded)

    async def recover(self) -> int:
        """Repair what an earlier close could not write. Returns how many play rows it wrote.

        1. A ``retro_sitting`` row that never closed (the process was killed) ends at its last input.
        2. A closed sitting of ``min_sec`` or more with no play row gets one (the database was down at
           its close, or a shutdown cut the close short).
        Call it only in the process that relays the input: the row of the sitting open in THIS
        tracker is left alone, and any other open row is taken to be dead. Never raises."""
        if not self.enabled:
            return 0
        cur = self.tracker.current
        written = 0
        try:
            closed = await self._db("fetch", RECOVER_OPEN_SQL, cur.sitting_id if cur is not None else "")
            rows = await self._db("fetch", RECOVER_MISSING_SQL, float(self.min_sec), RECOVER_DAYS)
            for r in rows:
                s = Sitting(
                    sitting_id=r["sitting_id"], retro_session_id=r["retro_session_id"], game_id=r["game_id"],
                    game_title=r["game_title"], started_at=_epoch(r["started_at"]), last_input_at=_epoch(r["last_input_at"]),
                    ended_at=_epoch(r["ended_at"]), end_reason=r["end_reason"], mask_changes=r["mask_changes"],
                    saves=r["saves"], loads=r["loads"], pauses=r["pauses"], player_stated=r["player_stated"])
                written += (await self._play(s, r.get("notes_stored") or None)) == "play_row"
            if closed or rows:
                logger.info("retro_sitting_recovered closed_rows=%d missing_play_rows=%d written=%d",
                            len(closed), len(rows), written)
        except LookupError:
            pass                    # no pool yet: the gateway calls again when its pool opens
        except Exception:  # noqa: BLE001
            self._warn("retro_sitting_recover_failed")
        return written

    async def close(self) -> None:
        """Process shutdown: the open sitting ends at its last input and is recorded."""
        if self._loop_task is not None:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
        closed = self.tracker.close_all()
        await self._publish()
        for s in closed:
            await self._finish(s)
        if self._jobs:
            await asyncio.gather(*list(self._jobs), return_exceptions=True)
