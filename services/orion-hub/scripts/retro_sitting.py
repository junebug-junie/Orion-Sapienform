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
  of kind ``human_play``, which the Temporal Self source ``retro_play`` turns into a play arc.

It credits Orion with nothing: orion_frames is 0, no memory card is written and no act is opened
(a person playing is not an act of Orion's). Who plays is only what the page was told.

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
    parse_input_message, worth_recording,
)
from orion.schemas.retro_play import RetroPlaySessionSummaryV1
from orion.temporal_self.stance_cue import span_words

logger = logging.getLogger("orion-hub.retro.sitting")

TICK_SEC = 15.0            # how often the open sitting is checked for its end and its Redis copy refreshed
PERSIST_SEC = 60.0         # how often the open sitting's row is written
KEY_TTL_SEC = 180          # a crashed process's key expires on its own (12 missed ticks)
OP_TIMEOUT_SEC = 2.0
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

_PLAY_COLUMNS = ("session_id", "retro_session_id", "lineage_id", "game_id", "game_title", "kind", "dispatch_id",
                 "started_at", "ended_at", "duration_sec", "end_reason", "control", "control_basis",
                 "human_may_be_in_control", "human_frames", "orion_frames", "human_share", "played_together",
                 "rooms_new", "caves_new", "items_gained", "deaths", "death_rooms", "decisions", "predictions_met",
                 "predictions_refuted", "predictions_unknown", "human_interventions", "summary", "player_stated")
_PLAY_JSON = {"control", "items_gained", "death_rooms"}
PLAY_INSERT_SQL = (
    f"INSERT INTO retro_play_sessions ({', '.join(_PLAY_COLUMNS)}) VALUES ("
    + ", ".join(f"${i + 1}::jsonb" if c in _PLAY_JSON else f"${i + 1}" for i, c in enumerate(_PLAY_COLUMNS))
    + ") ON CONFLICT (session_id) DO NOTHING RETURNING session_id"
)


def _ts(epoch: Optional[float]) -> Optional[datetime]:
    return None if epoch is None else datetime.fromtimestamp(float(epoch), timezone.utc)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def who_words(player_stated: Optional[str]) -> str:
    """The only names this organ ever writes: Juniper when the page was told so, else "Someone"."""
    return "Juniper" if player_stated == "juniper" else "Someone"


def sitting_summary(s: Sitting) -> str:
    """One deterministic sentence for the play arc. Counts only; no model writes it."""
    title = s.game_title or s.game_id or "a game"
    text = f"{who_words(s.player_stated)} played {title} for {span_words(s.duration_sec)}."
    extras = [_plural(n, w) for n, w in ((s.saves, "save"), (s.loads, "load")) if n]
    if extras:
        text += f" {' and '.join(extras).capitalize()}."
    return text


def play_row(s: Sitting) -> RetroPlaySessionSummaryV1:
    """The ``retro_play_sessions`` row of one closed sitting. Frames are not counted (basis
    ``human_input``): human_frames and orion_frames are 0, and no share is claimed."""
    assert s.ended_at is not None
    return RetroPlaySessionSummaryV1(
        session_id=ROW_PREFIX + s.sitting_id, retro_session_id=s.retro_session_id, lineage_id=None,
        game_id=s.game_id or "unknown", game_title=s.game_title or s.game_id or "unknown", kind="human_play",
        started_at=_ts(s.started_at), ended_at=_ts(s.ended_at), duration_sec=round(s.duration_sec, 1),
        end_reason=s.end_reason, control=[], control_basis="human_input", human_may_be_in_control=True,
        human_frames=0, orion_frames=0, human_share=None, played_together=False,
        predictions_met=0, predictions_refuted=0, human_interventions=0,
        summary=sitting_summary(s), player_stated=s.player_stated if s.player_stated in PLAYERS else None,
    )


def _play_values(row: RetroPlaySessionSummaryV1) -> list[Any]:
    data = row.model_dump(mode="python")
    return [json.dumps(data[c]) if c in _PLAY_JSON else data[c] for c in _PLAY_COLUMNS]


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
            f"{tail}{known} You are not watching the screen: you know this from the controller's input only.")


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
        self.min_sec = float(getattr(config, "HUB_RETRO_SITTING_MIN_SEC", MIN_SEC_DEFAULT) or MIN_SEC_DEFAULT)
        gap = float(getattr(config, "HUB_RETRO_SITTING_GAP_SEC", GAP_SEC_DEFAULT) or GAP_SEC_DEFAULT)
        self.tracker = tracker or SittingTracker(gap_sec=gap)
        self.redis_provider, self.pool_provider = redis_provider, pool_provider
        self.resolve_game, self.clock = resolve_game, clock
        self._loop_task: Optional[asyncio.Task] = None
        self._jobs: set[asyncio.Task] = set()
        self._persisted_at = 0.0

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
        except Exception:  # noqa: BLE001 -- the relay never fails for a record
            logger.warning("retro_sitting_input_failed", exc_info=True)

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
            logger.warning("retro_sitting_action_failed", exc_info=True)

    # --- the page -------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        cur = self.tracker.current
        return {"enabled": self.enabled, "sitting": cur.as_dict() if cur is not None else None}

    async def state_player(self, player: Any) -> bool:
        if not self.enabled or not self.tracker.state_player(player if isinstance(player, str) else None):
            return False
        logger.info("retro_sitting_player_stated sitting=%s player=%s", self.tracker.current.sitting_id, player)
        await self._publish()
        await self._persist(self.tracker.current)
        return True

    # --- background -----------------------------------------------------------------------------
    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._jobs.add(task)
        task.add_done_callback(self._jobs.discard)
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.ensure_future(self._loop())

    async def _after_change(self, closed: list[Sitting], opened: Optional[Sitting]) -> None:
        for s in closed:
            await self._finish(s)
        if opened is not None:
            logger.info("retro_sitting_opened sitting=%s retro_session=%s", opened.sitting_id, opened.retro_session_id)
            if self.resolve_game is not None:
                try:
                    game_id, title = await asyncio.wait_for(self.resolve_game(opened.retro_session_id), 5.0)
                    opened.game_id, opened.game_title = game_id or opened.game_id, title or opened.game_title
                except Exception:  # noqa: BLE001 -- a sitting without a title is still a sitting
                    logger.warning("retro_sitting_game_unresolved sitting=%s", opened.sitting_id, exc_info=True)
            await self._persist(opened)
        await self._publish()

    async def _loop(self) -> None:
        """Runs while a sitting is open; ends itself when none is, and restarts on the next input."""
        try:
            while self.tracker.current is not None:
                await asyncio.sleep(TICK_SEC)
                now = self.clock()
                for s in self.tracker.tick(now):
                    await self._finish(s)
                await self._publish()
                cur = self.tracker.current
                if cur is not None and now - self._persisted_at >= PERSIST_SEC:
                    await self._persist(cur)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- the next input starts the loop again
            logger.warning("retro_sitting_loop_failed", exc_info=True)

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
            logger.warning("retro_sitting_publish_failed", exc_info=True)

    async def _persist(self, s: Sitting) -> bool:
        pool = self.pool_provider()
        if pool is None:
            return False
        try:
            async with pool.acquire() as conn:
                await asyncio.wait_for(conn.execute(UPSERT_SQL, *sitting_values(s)), OP_TIMEOUT_SEC * 2)
            self._persisted_at = self.clock()
            return True
        except Exception:  # noqa: BLE001
            logger.warning("retro_sitting_persist_failed sitting=%s", s.sitting_id, exc_info=True)
            return False

    async def _finish(self, s: Sitting) -> None:
        """A closed sitting: its row, and a play row when it was long enough to matter."""
        stored = await self._persist(s)
        recorded = "too_short"
        if worth_recording(s, self.min_sec):
            recorded = "no_pool"
            pool = self.pool_provider()
            if pool is not None:
                try:
                    async with pool.acquire() as conn:
                        got = await asyncio.wait_for(conn.fetchval(PLAY_INSERT_SQL, *_play_values(play_row(s))),
                                                     OP_TIMEOUT_SEC * 2)
                    recorded = "play_row" if got else "duplicate"
                except Exception:  # noqa: BLE001
                    recorded = "failed"
                    logger.warning("retro_sitting_play_row_failed sitting=%s", s.sitting_id, exc_info=True)
        logger.info("retro_sitting_closed sitting=%s reason=%s duration_sec=%.0f mask_changes=%d player=%s "
                    "sitting_row=%s recorded=%s", s.sitting_id, s.end_reason, s.duration_sec, s.mask_changes,
                    s.player_stated, stored, recorded)

    async def close(self) -> None:
        """Process shutdown: the open sitting ends at its last input and is recorded."""
        if self._loop_task is not None:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
        for s in self.tracker.close_all():
            await self._finish(s)
        await self._publish()
        if self._jobs:
            await asyncio.gather(*list(self._jobs), return_exceptions=True)
