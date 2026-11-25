"""Orion looks at the game screen while a person plays (game company, patch 2).

Design: docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md, revision 4: sections 2
(the screen is a camera), 3 (the look), 6 part 8 (Orion's own switch), 8 (the temporal record) and 9 (the
body economy). The rules are pure (orion/retro/watch.py, orion/retro/frames.py); this module is their I/O,
in the process that relays the game (Hub's Retro gateway).

The whole path, in order:

1. **Tap.** The relay hands every video packet it already sends to the browser to ``note_media``
   (scripts/retro_transport.py, ``on_media``). One packet in 2 s is kept, compressed, in a ring of 30 (the
   last minute). No second connection to the engine is opened. Sync, never raises, never waits.
2. **Check.** Every 2 s while a sitting is open (scripts/retro_sitting.py), the newest kept packet is decoded
   to a 32 x 28 brightness grid. Its change over the last 30 s (against the check about 30 s earlier: the
   look's own window, the statistic that separated still from play in step 0) goes into the game's
   histogram, labelled by whether a button changed or stayed held in those 30 s; its change since the last
   look goes through the gates
   (orion/retro/watch.py ``gate``). Every gate outcome is a counter on the sitting row.
3. **Floor.** Once a minute the histograms are added to ``retro_watch_change_hist`` and each game's floor is
   computed again (``calibrate``) into ``retro_watch_floor``. A game without a floor (any status but ``set``)
   looks on a timer instead: one look every ``HUB_RETRO_WATCH_NO_FLOOR_LOOK_SEC`` (60) while a button changed
   in the last 60 s, counted ``timer_looks`` when it bids; it bids and seats exactly as a floor look does. On a timer look
   an exact repeat of the previous note is stored, not ``copied``: the prompt is stateless, so on a still
   screen (SimEarth) the same words are a true look, and dropping them would leave no fresh note.
4. **Mode.** Before every bid, Orion's own choices (``dig watch``: act rows of actor ``orion.watch_choice``)
   are copied into ``retro_watch_preference`` and read with Juniper's page switch. The lower wins. Off: no bid.
5. **Bid.** One want in family ``hub.retro_watch`` (tier 1). Only ``funded`` runs the look; anything else is
   logged with the market's reason and shown on the page. No fallback.
6. **Seat.** The look's act opens. A GPU pool hold of class ``chat`` (the sighted lane) at priority ``system``,
   carrying the want id. The frames are taken when the hold is GRANTED, not when the bid was made: up to 4
   packets spread over the last 30 s of the ring at that moment.
7. **Frames.** Each is decoded, enlarged 2x with hard edges, PNG, posted to orion-percept-store (gone an hour
   after its last post; ``frames_expire_at`` moves with every post of the same bytes).
8. **Call.** One llm.chat.request on route ``retro_watch`` under the hold, thinking off, <= 160 tokens. The
   prompt is stateless: the game title, the minutes played, the frames. The gateway refuses images on a blind
   lane, so a look can never return words about frames the model did not receive.
9. **Note.** stored / empty / unclear / copied (or failed), one ``retro_watch_look`` row, the act closed 1 only
   for stored. The hold and the want are released.

No look writes a Temporal Self event here: a line turn (patch 3, scripts/retro_commentary.py in hub-app) judges the
notes it was shown, and only a note it checked and found to hold becomes one (source kind retro_look,
orion-durable-runs). The sitting's play row carries the count ("Orion watched: N notes"). Nothing here speaks, and
nothing here touches a person record.

Patch 3 adds two things here: ``frames_for_turn`` (hub-app's line and chat-box turns ask for the newest frames over
``GET /api/retro/sitting/frames``), and the line lock: no look is bid for while hub-app holds ``LINE_LOCK_KEY``
(one play call at a time on the chat lane).

Every failure is logged and swallowed: the game never waits for a look.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from orion.retro import watch as W

logger = logging.getLogger("orion-hub.retro.watch")

HOLDER = "orion-hub.retro_watch"
SOURCE_NAME = "orion-hub-retro-watch"
FLUSH_SEC = 60.0
OP_TIMEOUT_SEC = 4.0           # one database statement, connection wait included
CALL_TIMEOUT_SEC = 60.0        # the model call under the hold
LOG_EVERY_SEC = 60.0
STALE_RING_SEC = 60.0          # no packet for this long and no sitting: the check loop ends
PREF_SYNC_FIRST_DAYS = 400     # the first copy of Orion's choices after a start reaches back this far
PREF_SYNC_DAYS = 2
KEEP_FINISHED = 8              # counts of recently closed sittings kept for their play row
# Game company patch 4: the watch's state for Prime's frame line (orion/prime/sitting.py reads it from the bus Redis,
# next to orion:retro:sitting). Refreshed every WATCH_PUBLISH_SEC while a sitting is open; expires on its own.
from orion.retro.sitting import WATCH_KEY  # noqa: E402 -- one name for the key, shared with orion/prime/sitting.py
WATCH_PUBLISH_SEC = 15.0
WATCH_KEY_TTL_SEC = 90
# Held in Redis by hub-app while a line turn runs (scripts/retro_commentary.py): no look is bid for meanwhile.
LINE_LOCK_KEY = "orion:retro:line_turn"

HIST_UPSERT_SQL = """
INSERT INTO retro_watch_change_hist (game_id, input_in_window, bin, checks, updated_at)
VALUES ($1, $2, $3, $4, now())
ON CONFLICT (game_id, input_in_window, bin) DO UPDATE SET
    checks = retro_watch_change_hist.checks + EXCLUDED.checks, updated_at = now()
"""
HIST_READ_SQL = "SELECT input_in_window, bin, checks FROM retro_watch_change_hist WHERE game_id = $1"
FLOOR_UPSERT_SQL = """
INSERT INTO retro_watch_floor (game_id, floor, status, checks_input, checks_no_input, seconds, noinput_above,
                               input_above, histogram, computed_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, now())
ON CONFLICT (game_id) DO UPDATE SET
    floor = EXCLUDED.floor, status = EXCLUDED.status, checks_input = EXCLUDED.checks_input,
    checks_no_input = EXCLUDED.checks_no_input, seconds = EXCLUDED.seconds, noinput_above = EXCLUDED.noinput_above,
    input_above = EXCLUDED.input_above, histogram = EXCLUDED.histogram, computed_at = now()
"""
COUNTERS_SQL = ("UPDATE retro_sitting SET " + ", ".join(f"{c} = ${i + 2}" for i, c in enumerate(W.COUNTERS))
                + ", updated_at = now() WHERE sitting_id = $1")
LOOK_INSERT_SQL = """
INSERT INTO retro_watch_look (look_id, sitting_id, game_id, looked_at, frame_hashes, frames_expire_at, note, outcome,
                              failure, screen_change, want_id, lease_id, lane_wait_ms, seconds, prompt_tokens,
                              similar_to_previous)
VALUES ($1, $2, $3, $4, $5::text[], $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
ON CONFLICT (look_id) DO NOTHING
"""
# A frame lives one hour after its LAST post (the percept store refreshes a re-posted blob's age): every row
# that names a hash we just posted moves its expiry with it.
EXPIRY_SQL = """
UPDATE retro_watch_look SET frames_expire_at = $2
WHERE looked_at > now() - interval '2 hours' AND frame_hashes && $1::text[]
  AND (frames_expire_at IS NULL OR frames_expire_at < $2)
"""
# Orion's own choices: act rows of `dig watch` (orion/dig/watch.py) not yet copied.
CHOICE_ACTS_SQL = f"""
SELECT a.act_id, a.object_ref, a.test_before, a.outcome_note, a.session_id, coalesce(a.opened_at, a.created_at) AS set_at
FROM grammar_acts a
WHERE a.verb = 'regulate' AND a.actor = '{W.CHOICE_ACTOR}'
  AND a.opened_at >= now() - make_interval(days => $1)
  AND NOT EXISTS (SELECT 1 FROM retro_watch_preference p WHERE p.pref_id = 'act:' || a.act_id AND p.why IS NOT NULL)
ORDER BY a.opened_at
"""
PREF_INSERT_SQL = """
INSERT INTO retro_watch_preference (pref_id, scope, scope_id, set_by, mode, why, act_id, correlation_id, set_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
ON CONFLICT (pref_id) DO UPDATE SET why = COALESCE(retro_watch_preference.why, EXCLUDED.why)
"""
PREF_READ_SQL = """
SELECT scope, scope_id, set_by, mode, why, set_at FROM retro_watch_preference
WHERE set_by = 'orion' AND ((scope = 'sitting' AND scope_id = $1) OR (scope = 'game' AND scope_id = $2))
UNION ALL
(SELECT scope, scope_id, set_by, mode, why, set_at FROM retro_watch_preference
 WHERE set_by = 'juniper' AND scope = 'all' ORDER BY set_at DESC LIMIT 1)
"""


class LookFailed(Exception):
    """A funded look that could not produce a note. ``reason`` is stored on the row."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _setting(config: Any, name: str, default: float) -> float:
    value = getattr(config, name, None)
    return float(default if value is None else value)


def _utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(float(epoch), timezone.utc)


@dataclass
class SittingWatch:
    """What the watch knows about one open sitting."""
    sitting_id: str
    retro_session_id: str
    started_at: float
    counters: dict = field(default_factory=lambda: {c: 0 for c in W.COUNTERS})
    ref_grid: Any = None                  # the last frame of the last look (or the sitting's first check)
    prev_check_at: Optional[float] = None
    # The checks of the last ~32 s: (kept_at, grid, a button changed or was held since the check before).
    recent: deque = field(default_factory=lambda: deque(maxlen=20))
    last_packet_at: Optional[float] = None
    max_change: float = 0.0               # largest change vs ref_grid since the last look
    last_change: Optional[float] = None
    last_look_at: Optional[float] = None  # when a look was last started (the 60 s ceiling)
    last_over_floor_at: Optional[float] = None
    last_stored_at: Optional[float] = None
    last_note: Optional[str] = None       # the previous note of any outcome (for "copied")
    last_stored_note: Optional[str] = None
    last_refusal: Optional[tuple[str, float]] = None
    looking: bool = False
    first_check_at: Optional[float] = None  # the timer of a no-floor game starts here (the ring holds frames)


# --- default I/O (tests replace each) ---------------------------------------------------------------

def post_percept(data: bytes, *, base_url: str, token: str, timeout_sec: float) -> str:
    """POST one PNG to orion-percept-store; its sha256 (checked against ours). urllib: no new dependency."""
    local = hashlib.sha256(data).hexdigest()
    req = urllib.request.Request(str(base_url).rstrip("/"), data=data, method="POST")
    req.add_header("Content-Type", "application/octet-stream")
    if token:
        req.add_header("X-Orion-Percept-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:   # noqa: S310 -- internal service URL
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 -- URLError, OSError, http.client.HTTPException, bad JSON
        raise LookFailed(f"percept_upload_failed:{type(exc).__name__}") from exc
    if str(body.get("sha256") or "") != local:
        raise LookFailed("percept_hash_mismatch")
    return local


class WatchService:
    """One per process, beside the sitting record. ``sitting``: scripts.retro_sitting.SittingService."""

    def __init__(self, config: Any, *, sitting: Any, pool_provider: Callable[[], Any] = lambda: None,
                 clock: Callable[[], float] = time.time,
                 bid: Optional[Callable[..., Awaitable[Any]]] = None,
                 release: Optional[Callable[[str], Awaitable[Any]]] = None,
                 bus_factory: Optional[Callable[[], Any]] = None,
                 seat: Optional[Callable[..., Any]] = None,
                 call_model: Optional[Callable[..., Awaitable[dict]]] = None,
                 post_frame: Optional[Callable[[bytes], str]] = None,
                 publish_act: Optional[Callable[..., Awaitable[bool]]] = None):
        self.config = config
        self.sitting = sitting
        self.enabled = bool(getattr(config, "HUB_RETRO_WATCH_ENABLED", True)) and bool(getattr(sitting, "enabled", False))
        self.look_gap = _setting(config, "HUB_RETRO_LOOK_SEC", W.LOOK_GAP_SEC)
        self.no_floor_look = _setting(config, "HUB_RETRO_WATCH_NO_FLOOR_LOOK_SEC", W.NO_FLOOR_LOOK_SEC)
        self.seat_wait = _setting(config, "HUB_RETRO_WATCH_SEAT_WAIT_SEC", 30.0)
        self.check_sec = W.CHECK_SEC
        self.pool_provider, self.clock = pool_provider, clock
        self._bid, self._release = bid or self._default_bid, release or self._default_release
        self._bus_factory = bus_factory or self._default_bus
        self._seat = seat or self._default_seat
        self._call_model = call_model or self._default_call
        self._post_frame = post_frame or self._default_post
        self._publish_act = publish_act or self._default_publish_act
        self.ring: deque = deque(maxlen=W.RING_MAX)     # (kept_at, retro_session_id, packet)
        self._last_kept = float("-inf")
        self.state: Optional[SittingWatch] = None
        self.finished: dict[str, dict] = {}             # sitting id -> counters (recently closed)
        self.calibrations: dict[str, W.Calibration] = {}
        self.hist_pending: dict[tuple[str, bool, int], int] = {}
        self.orion_rows: list[dict] = []
        self.juniper: Optional[dict] = None
        self.mode = W.Mode("on", None)
        self._pref_synced_once = False
        self._loop_task: Optional[asyncio.Task] = None
        self._look_task: Optional[asyncio.Task] = None
        self._mode_known = False      # no look is bid for until one read of both sides succeeded (fail closed)
        self._jobs: set[asyncio.Task] = set()
        self._flushing = False
        self._flushed_at = float("-inf")
        self._logged: dict[str, float] = {}
        if self.enabled:
            sitting.looks_for = self.looks_for

    # --- logging ----------------------------------------------------------------------------------
    def _warn(self, event: str, *args: Any) -> None:
        now = time.monotonic()
        if now - self._logged.get(event, -LOG_EVERY_SEC) >= LOG_EVERY_SEC:
            self._logged[event] = now
            logger.warning(event + (" " + " ".join(str(a) for a in args) if args else ""), exc_info=True)

    # --- the tap: sync, never raises, never waits ---------------------------------------------------
    def note_media(self, session_id: str, data: Any) -> None:
        if not self.enabled or not isinstance(data, (bytes, bytearray)):
            return
        try:
            now = self.clock()
            if self.ring and self.ring[-1][1] != str(session_id):
                self.ring.clear()                    # another game session: none of these are its frames
                self._last_kept = float("-inf")
                if time.monotonic() - self._logged.get("ring_reset", -LOG_EVERY_SEC) >= LOG_EVERY_SEC:
                    self._logged["ring_reset"] = time.monotonic()      # two live sessions would log per packet
                    logger.info("retro_watch_ring_reset retro_session=%s", session_id)
            if now - self._last_kept < self.check_sec * 0.95:
                return
            self.ring.append((now, str(session_id), bytes(data)))
            self._last_kept = now
            self._ensure_loop()
        except Exception:  # noqa: BLE001 -- the relay never fails for a look
            self._warn("retro_watch_tap_failed")

    def looks_for(self, sitting_id: str) -> Optional[int]:
        """Notes stored in a sitting (the sitting record's play row). None: this watch never saw it."""
        st = self.state
        if st is not None and st.sitting_id == sitting_id:
            return int(st.counters.get("notes_stored", 0))
        counts = self.finished.get(sitting_id)
        return int(counts.get("notes_stored", 0)) if counts is not None else None

    # --- the loop ---------------------------------------------------------------------------------
    def _ensure_loop(self) -> None:
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.ensure_future(self._loop())

    def _spawn(self, coro: Awaitable[Any]) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._jobs.add(task)
        task.add_done_callback(self._jobs.discard)
        return task

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.check_sec)
                now = self.clock()
                self.check(now)
                if now - getattr(self, "_watch_published_at", 0.0) >= WATCH_PUBLISH_SEC:
                    self._watch_published_at = now
                    self._spawn(self.publish_watch())
                if now - self._flushed_at >= FLUSH_SEC and not self._flushing:
                    self._flushed_at = now
                    self._spawn(self.flush())
                newest = self.ring[-1][0] if self.ring else None
                if self.sitting.tracker.current is None and (newest is None or now - newest > STALE_RING_SEC):
                    self._spawn(self.flush())
                    return
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- the next packet starts the loop again
            self._warn("retro_watch_loop_failed")

    def watch_snapshot(self) -> Optional[dict]:
        """What Prime's frame line needs about the watch of the open sitting, or None when none is open."""
        cur = self.sitting.tracker.current
        st = self.state
        if cur is None or st is None or st.sitting_id != cur.sitting_id:
            return None
        return {"sitting_id": cur.sitting_id, "mode": self.mode.mode, "set_by": self.mode.set_by,
                "why": self.mode.why, "counts": dict(st.counters), "last_note": st.last_stored_note,
                "last_note_at": st.last_stored_at, "status": self.status().get("status", {}).get("code"),
                "at": self.clock()}

    async def publish_watch(self) -> None:
        """``watch_snapshot`` into Redis (WATCH_KEY, with a TTL). Never raises: Prime's line then says unknown."""
        snap = self.watch_snapshot()
        if snap is None:
            return
        try:
            redis = self.sitting.redis_provider()
            if redis is None:
                return
            await asyncio.wait_for(redis.set(WATCH_KEY, json.dumps(snap, separators=(",", ":"), default=str),
                                             ex=WATCH_KEY_TTL_SEC), OP_TIMEOUT_SEC)
        except Exception:  # noqa: BLE001
            self._warn("retro_watch_publish_failed")

    def _state_for(self, cur: Any) -> SittingWatch:
        st = self.state
        if st is not None and st.sitting_id == cur.sitting_id:
            return st
        if st is not None:
            self._retire(st)
        self.state = SittingWatch(sitting_id=cur.sitting_id, retro_session_id=cur.retro_session_id,
                                  started_at=float(cur.started_at))
        self._spawn(self.refresh(cur))
        return self.state

    def _retire(self, st: SittingWatch) -> None:
        self.finished[st.sitting_id] = dict(st.counters)
        while len(self.finished) > KEEP_FINISHED:
            self.finished.pop(next(iter(self.finished)))
        self._spawn(self._write_counters(st.sitting_id, dict(st.counters)))

    def check(self, now: float) -> Optional[str]:
        """One ring check. Returns the gate it stopped at (None: nothing to check). Never raises."""
        try:
            return self._check(now)
        except Exception:  # noqa: BLE001
            self._warn("retro_watch_check_failed")
            return None

    def _check(self, now: float) -> Optional[str]:
        cur = self.sitting.tracker.current
        if cur is None:
            if self.state is not None:
                self._retire(self.state)
                self.state = None
            return None
        st = self._state_for(cur)
        if not self.ring:
            return None
        kept_at, sid, packet = self.ring[-1]
        if sid != cur.retro_session_id or (st.last_packet_at is not None and kept_at <= st.last_packet_at):
            return None                      # no new frame of this game since the last check
        if now - kept_at > 2 * self.check_sec + 1:
            return None                      # the browser stopped receiving pictures
        st.last_packet_at = kept_at
        from orion.retro.frames import FrameError, decode_packet, luma_grid, screen_change
        try:
            grid = luma_grid(decode_packet(packet).rgb)
        except FrameError:
            self._warn("retro_watch_undecodable_packet")
            return None
        st.counters["ring_checks"] += 1
        if st.first_check_at is None:
            st.first_check_at = kept_at
        held = bool(getattr(self.sitting.tracker, "held", lambda _s: False)(sid))
        since = st.prev_check_at if st.prev_check_at is not None else kept_at - self.check_sec
        input_in_window = held or float(cur.last_input_at) >= since
        game = cur.game_id
        st.recent.append((kept_at, grid, input_in_window))
        while st.recent and kept_at - st.recent[0][0] > W.CALIBRATION_WINDOW_SEC + self.check_sec:
            st.recent.popleft()
        base = st.recent[0]
        if game and kept_at - base[0] >= W.CALIBRATION_WINDOW_SEC - self.check_sec:
            # The look window's statistic (step 0's "window_mean"): this frame against the one ~30 s before,
            # labelled by any button change or hold in between.
            window_input = any(flag for _t, _g, flag in list(st.recent)[1:])
            key = (str(game), bool(window_input), W.bin_of(screen_change(grid, base[1])))
            self.hist_pending[key] = self.hist_pending.get(key, 0) + 1
        st.prev_check_at = kept_at
        if st.ref_grid is None:
            st.ref_grid = grid
        st.max_change = max(st.max_change, screen_change(grid, st.ref_grid))
        st.last_change = st.max_change
        cal = self.calibrations.get(str(game)) if game else None
        floor = cal.floor if cal is not None else None
        result = W.gate(st.max_change, floor, now=now, last_input_at=float(cur.last_input_at),
                        last_look_at=st.last_look_at, look_gap_sec=self.look_gap,
                        timer_sec=self._timer_sec(cal), timer_from=st.first_check_at)
        W.count(st.counters, result, input_in_window=input_in_window)
        if result not in (W.GATE_NO_FLOOR, W.GATE_STATIC) and result not in W.TIMER_GATES:
            st.last_over_floor_at = now
        if result in W.LOOK_GATES and not st.looking and (self._look_task is None or self._look_task.done()):
            timer = result == W.GATE_TIMER
            st.looking, st.last_look_at = True, now
            self._look_task = self._spawn(self._look(st, cur, st.max_change, timer=timer))   # one look at a time
        return result

    def _timer_sec(self, cal: Optional[W.Calibration]) -> Optional[float]:
        """The no-floor interval, only once the game's calibration was read and it has no floor."""
        if cal is None or cal.floor is not None or self.no_floor_look <= 0:
            return None
        return self.no_floor_look

    # --- the look ---------------------------------------------------------------------------------
    async def _look(self, st: SittingWatch, cur: Any, change: float, *, timer: bool = False) -> None:
        try:
            await self._look_inner(st, cur, change, timer=timer)
        except Exception:  # noqa: BLE001
            self._warn("retro_watch_look_failed", st.sitting_id)
        finally:
            st.looking = False

    async def line_running(self) -> bool:
        """True while hub-app runs a line turn (scripts/retro_commentary.py holds LINE_LOCK_KEY in Redis): one
        play call at a time on the chat lane (spec section 4, "Juniper goes first"). Unreadable Redis reads as
        False: the line's own GPU hold (class chat) still queues a look's hold behind it on the same role."""
        redis = None
        try:
            redis = self.sitting.redis_provider()
            if redis is None:
                return False
            return bool(await asyncio.wait_for(redis.get(LINE_LOCK_KEY), OP_TIMEOUT_SEC))
        except Exception:  # noqa: BLE001
            self._warn("retro_watch_line_lock_unread")
            return False

    async def _look_inner(self, st: SittingWatch, cur: Any, change: float, *, timer: bool = False) -> None:
        if await self.line_running():
            st.last_refusal = ("line_running", self.clock())
            logger.info("retro_watch_no_bid sitting=%s reason=line_running", st.sitting_id)
            return
        mode = await self.refresh_mode(cur)
        if not self._mode_known:
            st.last_refusal = ("mode_unread", self.clock())
            logger.info("retro_watch_no_bid sitting=%s reason=mode_unread", st.sitting_id)
            return
        if mode.mode == "off":
            logger.info("retro_watch_no_bid sitting=%s mode=off set_by=%s", st.sitting_id, mode.set_by)
            return
        if timer:
            st.counters["timer_looks"] += 1       # a timer look that bids (not one the line or the switch stopped)
        want_id = str(uuid.uuid4())
        decision = await self._bid(self._want(want_id, st, cur, timer=timer), self._object_text(cur, st))
        outcome = getattr(decision, "outcome", None)
        if outcome != "funded":
            reason = (f"{outcome}:{getattr(decision, 'reason', None) or '-'}" if decision is not None else "no_answer")
            st.last_refusal = (reason, self.clock())
            logger.info("retro_watch_bid_refused sitting=%s want_id=%s reason=%s why=%r", st.sitting_id, want_id,
                        reason, str(getattr(decision, "why", "") or "")[:200])
            if decision is not None:
                await self._release(want_id)
            return
        st.counters["bids_funded"] += 1
        bus = None
        try:
            try:
                bus = await self._bus_factory()
            except Exception:  # noqa: BLE001 -- a funded look that could not start is a failed look
                st.counters["failed_looks"] += 1
                st.last_refusal = ("no_bus", self.clock())
                self._warn("retro_watch_bus_failed")
                return
            await self._publish_act(bus, opened=True, want_id=want_id, sitting_id=st.sitting_id)
            produced, note = await self._seated(bus, st, cur, want_id, change, timer=timer)
            await self._publish_act(bus, opened=False, want_id=want_id, sitting_id=st.sitting_id,
                                    produced=produced, note=note)
        finally:
            await self._release(want_id)
            if bus is not None:
                with contextlib.suppress(Exception):
                    await bus.close()

    async def _seated(self, bus: Any, st: SittingWatch, cur: Any, want_id: str, change: float, *,
                      timer: bool = False) -> tuple[bool, str]:
        """The hold, the frames, the call, the row. (act produced, act note)."""
        asked = self.clock()
        try:
            seat_cm = self._seat(bus, want_id)
            lease = await seat_cm.__aenter__()
        except Exception as exc:  # noqa: BLE001 -- LeaseUnavailable, a pool timeout, no bus
            reason = f"pool:{getattr(exc, 'reason', None) or type(exc).__name__}"
            if "cabinet_hot" in reason:          # the pool's heat reflex (orion/gpu_pool/shed.py), not the price
                reason = W.HEAT_PREFIX + reason
            st.last_refusal = (reason, self.clock())
            st.counters["failed_looks"] += 1      # funded but never seated: a failed look, not an invisible one
            logger.info("retro_watch_no_seat sitting=%s want_id=%s reason=%s", st.sitting_id, want_id, reason)
            return False, f"look no_seat: {reason}"
        error: Optional[BaseException] = None
        try:
            granted = self.clock()
            return await self._at_grant(bus, st, cur, want_id, change, lease,
                                        lane_wait_ms=int((granted - asked) * 1000), timer=timer)
        except BaseException as exc:
            error = exc
            raise
        finally:
            await seat_cm.__aexit__(type(error) if error else None, error, error.__traceback__ if error else None)

    async def _at_grant(self, bus: Any, st: SittingWatch, cur: Any, want_id: str, change: float, lease: Any, *,
                        lane_wait_ms: int, timer: bool = False) -> tuple[bool, str]:
        from orion.retro.frames import FrameError, decode_packet, luma_grid, png

        now = self.clock()
        entries = [e for e in list(self.ring) if e[1] == cur.retro_session_id]
        idx = W.pick_frames([e[0] for e in entries], now)
        picked = [entries[i] for i in idx]
        hashes: list[str] = []
        frames_expire_at: Optional[datetime] = None
        note, outcome, failure, seconds, prompt_tokens = None, "failed", None, None, None
        lease_id = getattr(lease, "lease_id", None)
        try:
            if len(picked) < W.MIN_FRAMES:
                raise LookFailed(f"no_frames:{len(picked)}")
            def render() -> list[tuple[bytes, int, int, Any]]:
                # Off the relay's event loop: four inflates and four PNG encodes.
                out = []
                for _kept_at, _sid, packet in picked:
                    try:
                        frame = decode_packet(packet)
                    except FrameError as exc:
                        raise LookFailed("undecodable_frame") from exc
                    out.append((png(frame.rgb), frame.width * 2, frame.height * 2, luma_grid(frame.rgb)))
                return out
            refs, last_grid = [], None
            for data, width, height, grid in await asyncio.to_thread(render):
                sha = await asyncio.to_thread(self._post_frame, data)
                hashes.append(sha)
                refs.append(self._ref(sha, data, width, height))
                last_grid = grid
            frames_expire_at = _utc(self.clock() + W.FRAME_TTL_SEC)
            text = W.prompt(cur.game_title or cur.game_id, (now - float(cur.started_at)) / 60.0, len(refs))
            st.counters["model_calls"] += 1
            started = time.monotonic()
            reply = await self._call_model(bus=bus, want_id=want_id, prompt=text, refs=refs, lease=lease)
            seconds = round(time.monotonic() - started, 2)
            usage = reply.get("usage") if isinstance(reply.get("usage"), dict) else {}
            prompt_tokens = usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None
            raw = reply.get("raw") if isinstance(reply.get("raw"), dict) else {}
            if raw.get("error"):
                raise LookFailed(f"gateway:{str(raw.get('error'))[:120]}")
            # The gateway refuses images on a blind lane (or one it could not fetch them for) with an "[Error: ...]"
            # TEXT and raw.vision.status refused/fetch_failed, not raw.error. Only "attached" means the model saw them.
            vision = raw.get("vision") if isinstance(raw.get("vision"), dict) else {}
            if vision.get("status") != "attached":
                raise LookFailed(f"vision:{vision.get('status') or 'unconfirmed'}")
            content = str(reply.get("content") or reply.get("text") or "")
            if content.lstrip().startswith("[Error:"):
                raise LookFailed("gateway_error_text")
            note = W.clean_note(content)
            outcome = W.classify(note, st.last_note)
            if timer and outcome == "copied":
                outcome = "stored"        # a still screen, a stateless prompt: the same words are a true look
            if last_grid is not None:
                st.ref_grid, st.max_change = last_grid, 0.0
        except LookFailed as exc:
            failure = exc.reason
        except Exception as exc:  # noqa: BLE001 -- the row and the act's close are still written
            failure = f"error:{type(exc).__name__}"
            self._warn("retro_watch_look_error")
        similar = W.similar(note, st.last_stored_note) if outcome == "stored" else None
        if outcome == "stored":
            st.counters["notes_stored"] += 1
            st.last_stored_at, st.last_stored_note = self.clock(), note
        else:
            st.counters["failed_looks"] += 1
        if note is not None:
            st.last_note = note
        await self._db("execute", LOOK_INSERT_SQL, want_id, st.sitting_id, cur.game_id, _utc(now), hashes,
                       frames_expire_at, note or None, outcome, failure, round(float(change), 6), want_id, lease_id,
                       lane_wait_ms, seconds, prompt_tokens, similar, quiet="retro_watch_look_row_failed")
        if hashes and frames_expire_at is not None:
            await self._db("execute", EXPIRY_SQL, hashes, frames_expire_at, quiet="retro_watch_expiry_failed")
        logger.info("retro_watch_look sitting=%s want_id=%s lease_id=%s outcome=%s failure=%s frames=%d "
                    "change=%.4f lane_wait_ms=%s seconds=%s prompt_tokens=%s trigger=%s", st.sitting_id, want_id,
                    lease_id, outcome, failure, len(hashes), change, lane_wait_ms, seconds, prompt_tokens,
                    "timer" if timer else "floor")
        # Never "failed:..." at the start: self-monitor reads such a note as a run that never happened.
        return outcome == "stored", f"look {outcome}: {note or failure or '-'}"

    def looking(self) -> bool:
        return self._look_task is not None and not self._look_task.done()

    async def frames_for_turn(self, n: int = W.FRAMES_PER_LOOK) -> dict:
        """The newest frames of the open sitting for a turn (a line, or a chat-box message from the game page):
        up to ``n`` ring packets spread over the last 30 s, each posted to the percept store (gone one hour after
        its last post; every look row naming a re-posted hash moves its ``frames_expire_at``). ``status``: ok |
        no_sitting | no_frames (the browser closed or reconnected: no ring) | look_in_flight | failed:<reason>.
        Frames are taken now, never from a stale ring (the same freshness rule as a ring check)."""
        cur = self.sitting.tracker.current
        if not self.enabled or cur is None:
            return {"status": "no_sitting"}
        if self.looking():
            return {"status": "look_in_flight", "sitting_id": cur.sitting_id}
        from orion.retro.frames import FrameError, decode_packet, png

        now = self.clock()
        entries = [e for e in list(self.ring) if e[1] == cur.retro_session_id]
        if not entries or now - entries[-1][0] > 2 * self.check_sec + 1:
            return {"status": "no_frames", "sitting_id": cur.sitting_id}
        picked = [entries[i] for i in W.pick_frames([e[0] for e in entries], now, max(1, min(int(n), W.FRAMES_PER_LOOK)))]

        def render() -> list[tuple[float, bytes, int, int]]:
            out = []
            for kept_at, _sid, packet in picked:
                try:
                    frame = decode_packet(packet)
                except FrameError:
                    continue
                out.append((kept_at, png(frame.rgb), frame.width * 2, frame.height * 2))
            return out
        frames, hashes = [], []
        try:
            for kept_at, data, width, height in await asyncio.to_thread(render):
                sha = await asyncio.to_thread(self._post_frame, data)
                ref = self._ref(sha, data, width, height).model_dump(mode="json")
                frames.append({**ref, "kept_at": kept_at})
                hashes.append(sha)
        except LookFailed as exc:
            return {"status": f"failed:{exc.reason}", "sitting_id": cur.sitting_id}
        if not frames:
            return {"status": "no_frames", "sitting_id": cur.sitting_id}
        expires = _utc(self.clock() + W.FRAME_TTL_SEC)
        await self._db("execute", EXPIRY_SQL, hashes, expires, quiet="retro_watch_expiry_failed")
        for f in frames:
            f["frames_expire_at"] = expires.isoformat()
        logger.info("retro_watch_frames_for_turn sitting=%s frames=%d", cur.sitting_id, len(frames))
        return {"status": "ok", "sitting_id": cur.sitting_id, "frames": frames, "frames_expire_at": expires.isoformat()}

    def _ref(self, sha: str, data: bytes, width: int, height: int) -> Any:
        from orion.core.bus.bus_schemas import AttachmentRefV1

        base = str(getattr(self.config, "PERCEPT_STORE_BASE_URL", "") or "").rstrip("/")
        return AttachmentRefV1(kind="percept", sha256=sha, mime="image/png", bytes=len(data), width=width,
                               height=height, source_url=f"{base}/{sha}")

    def _want(self, want_id: str, st: SittingWatch, cur: Any, *, timer: bool = False) -> Any:
        from orion.schemas.physiology_want import WantV1, WantWindowV1

        now = _utc(self.clock())
        gap = self.no_floor_look if timer else self.look_gap
        # Deferrable 0 in effect: the producer never holds. latest = one look interval. A timer look is the same
        # want (family, producer, verb, tier, lanes) as a floor look: only its trigger differs.
        return WantV1(want_id=want_id, family=W.FAMILY, producer=W.BINDING_PRODUCER, verb=W.VERB, tier=W.TIER,
                      object_ref=W.OBJECT_PREFIX + st.sitting_id, correlation_id=want_id, opened_at=now,
                      window=WantWindowV1(latest=now + timedelta(seconds=gap)), lanes=["chat"])

    @staticmethod
    def _object_text(cur: Any, st: SittingWatch) -> str:
        title = cur.game_title or cur.game_id or "a game"
        return f"{title}: {st.last_stored_note}" if st.last_stored_note else str(title)

    # --- default I/O --------------------------------------------------------------------------------
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

    async def _default_bus(self) -> Any:
        from orion.core.bus.async_service import OrionBusAsync

        bus = OrionBusAsync(str(self.config.ORION_BUS_URL), enabled=True)
        await bus.connect()
        return bus

    def _default_seat(self, bus: Any, want_id: str) -> Any:
        from orion.gpu_pool.client import gpu_lease

        return gpu_lease(bus, work_class=W.POOL_CLASS, holder=HOLDER, priority="system", kind="hold",
                         deadline_sec=self.seat_wait, turn_correlation_id=want_id, request_id=f"retro_watch:{want_id}")

    async def _default_call(self, *, bus: Any, want_id: str, prompt: str, refs: list, lease: Any) -> dict:
        from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
        from orion.llm.resource_lease import GPU_LEASE_OPTION
        from orion.schemas.gpu_pool import GpuLeaseRefV1

        if bus is None:
            raise LookFailed("no_bus")
        g = lease.grant
        ref = GpuLeaseRefV1(lease_id=g.lease_id, generation=g.generation, role=g.role, holder=HOLDER)
        reply = f"orion:exec:result:{self.config.LLM_GATEWAY_SERVICE_NAME}:{uuid.uuid4()}"
        payload = ChatRequestPayload(
            messages=[LLMMessage(role="user", content=prompt)], route=W.POOL_ROUTE, attachments=refs,
            options={"max_tokens": W.MAX_TOKENS, "temperature": 0.2, "purpose": "retro_watch_look",
                     "skip_spark_candidate_publish": True, "chat_template_kwargs": {"enable_thinking": False},
                     GPU_LEASE_OPTION: ref.model_dump(mode="json")})
        env = BaseEnvelope(kind="llm.chat.request", source=ServiceRef(name=SOURCE_NAME, version="1",
                                                                      node=getattr(self.config, "NODE_NAME", None)),
                           correlation_id=want_id, reply_to=reply, payload=payload.model_dump(mode="json"))
        try:
            msg = await bus.rpc_request(self.config.CHANNEL_LLM_INTAKE, env, reply_channel=reply,
                                        timeout_sec=CALL_TIMEOUT_SEC)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise LookFailed("gateway_timeout") from exc
        decoded = bus.codec.decode(msg.get("data") if isinstance(msg, dict) else None)
        if not decoded.ok or not decoded.envelope:
            raise LookFailed("gateway_reply_undecodable")
        out = decoded.envelope.payload
        return out if isinstance(out, dict) else {}

    def _default_post(self, data: bytes) -> str:
        base = str(getattr(self.config, "PERCEPT_STORE_BASE_URL", "") or "").strip()
        if not base:
            raise LookFailed("percept_store_unconfigured")
        return post_percept(data, base_url=base, token=str(getattr(self.config, "PERCEPT_STORE_TOKEN", "") or ""),
                            timeout_sec=_setting(self.config, "PERCEPT_STORE_TIMEOUT_SEC", 10.0))

    @staticmethod
    async def _default_publish_act(bus: Any, *, opened: bool, want_id: str, sitting_id: str, produced: bool = False,
                                   note: str = "") -> bool:
        from scripts.act_emit import publish_look_closed, publish_look_opened

        if opened:
            return await publish_look_opened(bus, want_id=want_id, sitting_id=sitting_id)
        return await publish_look_closed(bus, want_id=want_id, sitting_id=sitting_id, produced=produced, note=note)

    # --- the database -------------------------------------------------------------------------------
    async def _db(self, method: str, sql: str, *values: Any, quiet: Optional[str] = None) -> Any:
        """One bounded statement. With ``quiet``, a fault is logged under that name and None returned."""
        try:
            pool = self.pool_provider()
            if pool is None:
                raise LookupError("no_pool")

            async def run() -> Any:
                async with pool.acquire() as conn:
                    return await getattr(conn, method)(sql, *values)
            return await asyncio.wait_for(run(), OP_TIMEOUT_SEC)
        except LookupError:
            if quiet is None:
                raise
            return None
        except Exception:  # noqa: BLE001
            if quiet is None:
                raise
            self._warn(quiet)
            return None

    async def _write_counters(self, sitting_id: str, counters: dict) -> None:
        await self._db("execute", COUNTERS_SQL, sitting_id, *[int(counters.get(c, 0)) for c in W.COUNTERS],
                       quiet="retro_watch_counters_failed")

    async def flush(self) -> None:
        """Once a minute: the histograms, each touched game's floor, the counters, Orion's choices."""
        if self._flushing:
            return
        self._flushing = True
        try:
            pending, self.hist_pending = self.hist_pending, {}
            games = sorted({g for g, _i, _b in pending})
            if pending:
                ok = await self._db("executemany", HIST_UPSERT_SQL,
                                    [(g, i, b, n) for (g, i, b), n in sorted(pending.items())],
                                    quiet="retro_watch_hist_failed")
                if ok is None and self.pool_provider() is None:
                    for k, n in pending.items():          # no pool yet: keep the counts for the next flush
                        self.hist_pending[k] = self.hist_pending.get(k, 0) + n
            for g in games:
                await self.recalibrate(g)
            if self.state is not None:
                await self._write_counters(self.state.sitting_id, dict(self.state.counters))
                cur = self.sitting.tracker.current
                if cur is not None:
                    await self.refresh_mode(cur)
        finally:
            self._flushing = False

    async def recalibrate(self, game_id: str) -> Optional[W.Calibration]:
        rows = await self._db("fetch", HIST_READ_SQL, game_id, quiet="retro_watch_hist_read_failed")
        if rows is None:
            return None
        hi, hn = [0] * W.N_BINS, [0] * W.N_BINS
        for r in rows:
            b = int(r["bin"])
            if 0 <= b < W.N_BINS:
                (hi if r["input_in_window"] else hn)[b] += int(r["checks"])
        cal = W.calibrate(hi, hn, self.check_sec)
        histogram = {"edges_from": W.BIN_LOW, "bins_per_decade": W.BINS_PER_DECADE, "input": hi, "no_input": hn}
        await self._db("execute", FLOOR_UPSERT_SQL, game_id, cal.floor, cal.status, cal.checks_input,
                       cal.checks_no_input, cal.seconds, cal.noinput_above, cal.input_above, json.dumps(histogram),
                       quiet="retro_watch_floor_write_failed")
        before = self.calibrations.get(game_id)
        self.calibrations[game_id] = cal
        if before is None or before.floor != cal.floor or before.status != cal.status:
            logger.info("retro_watch_floor game=%s status=%s floor=%s checks_input=%d checks_no_input=%d",
                        game_id, cal.status, cal.floor, cal.checks_input, cal.checks_no_input)
        return cal

    async def refresh(self, cur: Any) -> None:
        """A sitting opened: its game's floor and the watch mode, read before its first possible look."""
        game = cur.game_id
        for _ in range(10):                       # the sitting record resolves the game in the background
            if game or self.sitting.tracker.current is not cur:
                break
            await asyncio.sleep(1.0)
            game = cur.game_id
        if game and str(game) not in self.calibrations:
            await self.recalibrate(str(game))
        await self.refresh_mode(cur)

    async def refresh_mode(self, cur: Any) -> W.Mode:
        """Copy Orion's new choices (act rows) into the table, then read both sides. On a read fault the
        last mode read stays (a database blip must not turn Orion's "off" back on)."""
        days = PREF_SYNC_DAYS if self._pref_synced_once else PREF_SYNC_FIRST_DAYS
        acts = await self._db("fetch", CHOICE_ACTS_SQL, days, quiet="retro_watch_choice_sync_failed")
        if acts is None:
            return self.mode                  # Orion's newest choice may be uncopied: keep the last mode read
        self._pref_synced_once = True
        for a in acts:
            row = choice_from_act(dict(a))
            if row is None:
                continue
            try:
                await self._db("execute", PREF_INSERT_SQL, *row)
            except Exception:  # noqa: BLE001 -- an uncopied choice: keep the last mode read
                self._warn("retro_watch_choice_copy_failed")
                return self.mode
        rows = await self._db("fetch", PREF_READ_SQL, cur.sitting_id, str(cur.game_id or ""),
                              quiet="retro_watch_pref_read_failed")
        if rows is None:
            return self.mode
        rows = [dict(r) for r in rows]
        self.orion_rows = [r for r in rows if r.get("set_by") == "orion"]
        jun = [r for r in rows if r.get("set_by") == "juniper"]
        self.juniper = max(jun, key=lambda r: r["set_at"]) if jun else None
        mine = W.orion_choice(self.orion_rows, cur.sitting_id, cur.game_id)
        self.mode = W.effective(mine.get("mode") if mine else None, self.juniper.get("mode") if self.juniper else None,
                                orion_why=mine.get("why") if mine else None,
                                juniper_why=self.juniper.get("why") if self.juniper else None)
        self._mode_known = True
        return self.mode

    # --- the page -----------------------------------------------------------------------------------
    async def set_juniper(self, mode: str, why: Optional[str] = None) -> bool:
        """Juniper's switch on the game page: a standing setting for every game (scope 'all')."""
        mode = W.valid_mode(mode)
        try:
            await self._db("execute", PREF_INSERT_SQL, f"juniper:{uuid.uuid4()}", "all", "", "juniper", mode,
                           (why or "")[:200] or None, None, None, _utc(self.clock()))
        except Exception:  # noqa: BLE001 -- the page says the switch did not hold
            self._warn("retro_watch_switch_failed")
            return False
        self.juniper = {"mode": mode, "why": why, "set_at": _utc(self.clock())}
        cur = self.sitting.tracker.current
        if cur is not None:
            await self.refresh_mode(cur)
        else:
            self.mode = W.effective(None, mode)
        return True

    def status(self) -> dict:
        cur = self.sitting.tracker.current
        st = self.state if (cur is not None and self.state is not None and self.state.sitting_id == cur.sitting_id) else None
        game = str(cur.game_id) if cur is not None and cur.game_id else None
        cal = self.calibrations.get(game) if game else None
        now = self.clock()
        line = W.status(
            enabled=self.enabled, sitting_open=cur is not None, mode=self.mode, calibration=cal, now=now,
            last_stored_at=st.last_stored_at if st else None, last_refusal=st.last_refusal if st else None,
            last_over_floor_at=st.last_over_floor_at if st else None,
            last_input_at=float(cur.last_input_at) if cur is not None else None,
            sitting_started_at=float(cur.started_at) if cur is not None else None,
            timer_sec=self._timer_sec(cal))
        return {
            "enabled": self.enabled, "mode": self.mode.mode, "set_by": self.mode.set_by, "why": self.mode.why,
            "juniper_mode": (self.juniper or {}).get("mode", "on"), "status": line,
            "counts": dict(st.counters) if st else None,
            "floor": {"game_id": game, "floor": cal.floor, "status": cal.status, "words": cal.words()} if cal else None,
            "last_note": st.last_stored_note if st else None,
            "looking": self.looking(),
        }

    async def close(self) -> None:
        """Process shutdown: the open sitting's counters and the pending histogram are written."""
        if self._loop_task is not None:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
        # A look in flight can hold a seat for ~90 s: cancel it (its finally blocks release the hold and the
        # want), then flush, all inside the gateway's 8 s shutdown bound.
        if self._look_task is not None and not self._look_task.done():
            self._look_task.cancel()
        if self._jobs:
            await asyncio.wait(list(self._jobs), timeout=3.0)
        self._flushing = False
        await self.flush()


def choice_from_act(act: dict) -> Optional[tuple]:
    """A ``retro_watch_preference`` row (PREF_INSERT_SQL values) from one ``orion.watch_choice`` act row, or None.
    The act (orion/dig/watch.py): object_ref ``retro_sitting:<id>`` or ``retro_game:<game id>``, test_before
    ``watch:<mode>``, Orion's words as the close note."""
    ref, test = str(act.get("object_ref") or ""), str(act.get("test_before") or "")
    if ref.startswith(W.OBJECT_PREFIX):
        scope, scope_id = "sitting", ref[len(W.OBJECT_PREFIX):]
    elif ref.startswith(W.GAME_PREFIX):
        scope, scope_id = "game", ref[len(W.GAME_PREFIX):]
    else:
        return None
    mode = test[len("watch:"):] if test.startswith("watch:") else ""
    if mode not in W.MODES or not scope_id:
        return None
    why = str(act.get("outcome_note") or "").strip() or None
    return (f"act:{act['act_id']}", scope, scope_id, "orion", mode, why[:200] if why else None, act["act_id"],
            act.get("session_id"), act.get("set_at"))


__all__ = ["HOLDER", "LINE_LOCK_KEY", "WATCH_KEY", "LookFailed", "SittingWatch", "WatchService", "choice_from_act", "post_percept"]
