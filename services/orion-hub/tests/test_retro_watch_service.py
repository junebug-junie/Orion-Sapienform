"""Orion's look at the game screen (scripts/retro_watch.py), with every outside thing faked: the market, the GPU
pool, the percept store, the model gateway, the act ledger and Postgres. Each test drives the checks by hand
(no sleeping loop) and waits on observable outcomes (the spawned jobs), never on one flag.

Not in tests/retro/: decoding a packet needs numpy, which the native-boundary CI job does not install."""
from __future__ import annotations

import asyncio
import json
import struct
import zlib
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from orion.retro import watch as W
from scripts import retro_sitting, retro_watch
from scripts.retro_watch import WatchService, choice_from_act

SID = "b" * 36
GAME = "a-link-to-the-past"
TITLE = "The Legend of Zelda: A Link to the Past"
FLOOR = W.Calibration(0.01, "set", 900, 900, 3600.0, 0.0, 1.0)


def packet(level: int, *, w: int = 256, h: int = 224) -> bytes:
    """An engine media packet (wire v1, XRGB8888) of a flat grey screen with one bright block whose place
    moves with ``level``: the same bytes the engine's packer writes (tests/test_retro_frames.py proves it)."""
    rgb = np.full((h, w, 3), 30, np.uint8)
    x = (level * 16) % (w - 32)
    rgb[96:128, x:x + 32] = 250
    r, g, b = (rgb[..., i].astype(np.uint32) for i in range(3))
    video = ((r << 16) | (g << 8) | b).astype("<u4").tobytes()
    header = json.dumps({"version": 1, "frame": level * 2, "width": w, "height": h, "format": 1,
                         "video_bytes": len(video), "sample_rate": 32040, "sample_end": 0, "audio_frames": 0},
                        separators=(",", ":")).encode()
    return zlib.compress(struct.pack("!I", len(header)) + header + video, 1)


class DB:
    """The few statements the watch runs, kept in memory."""

    def __init__(self):
        self.calls, self.hist, self.prefs, self.acts, self.looks, self.floors = [], {}, [], [], [], {}

    def acquire(self):
        db = self

        class Ctx:
            async def __aenter__(self): return Conn(db)
            async def __aexit__(self, *a): return False
        return Ctx()


class Conn:
    PREF_COLS = ("pref_id", "scope", "scope_id", "set_by", "mode", "why", "act_id", "correlation_id", "set_at")

    def __init__(self, db): self.db = db

    async def execute(self, sql, *a):
        self.db.calls.append((sql, a))
        if sql == retro_watch.PREF_INSERT_SQL and not any(p["pref_id"] == a[0] for p in self.db.prefs):
            self.db.prefs.append(dict(zip(self.PREF_COLS, a)))
        elif sql == retro_watch.LOOK_INSERT_SQL:
            self.db.looks.append(a)
        elif sql == retro_watch.FLOOR_UPSERT_SQL:
            self.db.floors[a[0]] = a

    async def executemany(self, sql, rows):
        self.db.calls.append((sql, rows))
        if sql == retro_watch.HIST_UPSERT_SQL:
            for g, i, b, n in rows:
                self.db.hist[(g, i, b)] = self.db.hist.get((g, i, b), 0) + n

    async def fetch(self, sql, *a):
        self.db.calls.append((sql, a))
        if sql == retro_watch.HIST_READ_SQL:
            return [{"input_in_window": i, "bin": b, "checks": n} for (g, i, b), n in self.db.hist.items() if g == a[0]]
        if sql == retro_watch.CHOICE_ACTS_SQL:
            done = {p["pref_id"] for p in self.db.prefs}
            return [x for x in self.db.acts if "act:" + x["act_id"] not in done]
        if sql == retro_watch.PREF_READ_SQL:
            return [p for p in self.db.prefs if
                    (p["set_by"] == "orion" and ((p["scope"] == "sitting" and p["scope_id"] == a[0])
                                                 or (p["scope"] == "game" and p["scope_id"] == a[1])))
                    or (p["set_by"] == "juniper" and p["scope"] == "all")]
        return []


class Seat:
    def __init__(self, log, fail=None):
        self.log, self.fail = log, fail

    async def __aenter__(self):
        if self.fail:
            exc = RuntimeError(self.fail)
            exc.reason = self.fail                   # orion.gpu_pool.client.LeaseUnavailable carries .reason
            raise exc
        self.log.append("granted")
        return SimpleNamespace(lease_id="lease-1", grant=SimpleNamespace(lease_id="lease-1", generation=1, role="chat"))

    async def __aexit__(self, *exc):
        self.log.append("released")
        return False


class Rig:
    """A sitting record and a watch on one fake clock, with every I/O recorded."""

    def __init__(self, *, decision="funded", seat_fail=None, reply=None, calibration=FLOOR):
        self.t = 1_000_000.0
        self.db = DB()
        self.log: list = []
        self.bids, self.releases, self.acts, self.posts, self.calls = [], [], [], [], []
        self.decision = decision
        self.reply = reply if reply is not None else {"content": "Link walks north through a forest clearing.",
                                                      "usage": {"prompt_tokens": 1086}}
        cfg = SimpleNamespace(HUB_RETRO_SITTING_ENABLED=True, HUB_RETRO_SITTING_GAP_SEC=300, HUB_RETRO_SITTING_MIN_SEC=120,
                              HUB_RETRO_WATCH_ENABLED=True, HUB_RETRO_LOOK_SEC=60, HUB_RETRO_WATCH_SEAT_WAIT_SEC=30,
                              PERCEPT_STORE_BASE_URL="http://percept.test/percepts")
        self.sitting = retro_sitting.SittingService(cfg, clock=lambda: self.t)
        self.watch = WatchService(cfg, sitting=self.sitting, pool_provider=lambda: self.db, clock=lambda: self.t,
                                  bid=self._bid, release=self._release, bus_factory=self._bus,
                                  seat=lambda bus, want_id: Seat(self.log, seat_fail),
                                  call_model=self._call, post_frame=self._post, publish_act=self._act)
        if calibration is not None:
            self.watch.calibrations[GAME] = calibration
        self.level = 0

    async def _bid(self, want, object_text):
        self.bids.append((want, object_text))
        if self.decision is None:
            return None
        return SimpleNamespace(outcome=self.decision, reason=None if self.decision == "funded" else "price", why="")

    async def _release(self, want_id): self.releases.append(want_id)

    async def _bus(self):
        self.log.append("bus")
        return SimpleNamespace(close=self._close)

    async def _close(self): self.log.append("bus_closed")

    async def _call(self, *, bus, want_id, prompt, refs, lease):
        self.calls.append(SimpleNamespace(want_id=want_id, prompt=prompt, refs=refs, lease=lease))
        return dict(self.reply)

    def _post(self, data: bytes) -> str:
        import hashlib
        assert data.startswith(b"\x89PNG")
        self.posts.append(data)
        return hashlib.sha256(data).hexdigest()

    async def _act(self, bus, *, opened, want_id, sitting_id, produced=False, note=""):
        self.acts.append(("opened" if opened else "closed", want_id, sitting_id, produced, note))
        return True

    def press(self, mask=None):
        """A button change (alternating masks: every call is a press or a release)."""
        self.mask = (0 if getattr(self, "mask", 0) else 128) if mask is None else mask
        self.sitting.tracker.on_input(SID, 0, self.mask, self.t)
        cur = self.sitting.tracker.current
        cur.game_id, cur.game_title = GAME, TITLE

    def frame(self, *, moving=True):
        if moving:
            self.level += 1
        self.watch.note_media(SID, packet(self.level))

    async def step(self, *, press=False, moving=True, sec=2.0):
        self.t += sec
        if press:
            self.press()
        self.frame(moving=moving)
        result = self.watch.check(self.t)
        await settle(self.watch)
        return result


async def settle(watch):
    for _ in range(50):
        if not watch._jobs:
            return
        await asyncio.gather(*list(watch._jobs), return_exceptions=True)


def run(coro):
    return asyncio.run(coro)


# --- the tap and the ring ------------------------------------------------------------------------------

def test_the_ring_keeps_one_packet_in_two_seconds_thirty_at_most_and_forgets_another_session():
    async def go():
        rig = Rig()
        for i in range(200):                      # 30 packets a second for ~6.6 s, then 90 s more
            rig.t += 1 / 30 if i < 200 else 0
            rig.watch.note_media(SID, b"x")
        assert len(rig.watch.ring) == 4           # 0, 2, 4, 6 s
        for _ in range(100):
            rig.t += 2.0
            rig.watch.note_media(SID, b"x")
        assert len(rig.watch.ring) == W.RING_MAX
        rig.watch.note_media("c" * 36, b"y")      # a new game session: the old frames are not its frames
        assert [e[1] for e in rig.watch.ring] == ["c" * 36]
        rig.watch.note_media(SID, "text frame")   # not a video packet: ignored, never raises
        rig.watch.note_media(SID, None)
        if rig.watch._loop_task:
            rig.watch._loop_task.cancel()
    run(go())


def test_no_sitting_means_no_check_and_the_watch_off_switch_keeps_nothing():
    async def go():
        rig = Rig()
        rig.frame()
        assert rig.watch.check(rig.t) is None and rig.watch.state is None
        off = WatchService(SimpleNamespace(HUB_RETRO_WATCH_ENABLED=False), sitting=rig.sitting)
        off.note_media(SID, packet(1))
        assert not off.ring and off.status()["status"]["code"] == "disabled"
        if rig.watch._loop_task:
            rig.watch._loop_task.cancel()
    run(go())


# --- gates and counters ------------------------------------------------------------------------------

def test_a_paused_game_gives_no_look_and_static_skips_rise():
    async def go():
        rig = Rig()
        rig.press()
        await rig.step(moving=False)
        for _ in range(300):                       # 10 minutes of a still screen
            assert await rig.step(moving=False) in (W.GATE_STATIC, W.GATE_INPUT_STALE)
        c = rig.watch.state.counters
        assert c["static_skips"] >= 299 and c["checks_over_floor"] == 0 and not rig.bids and not rig.db.looks
        assert rig.watch.status()["status"]["code"] == "static"
    run(go())


def test_a_changing_screen_with_no_press_for_two_minutes_gives_no_look_after_sixty_seconds():
    async def go():
        rig = Rig(decision="deferred")             # any look attempt would show as a bid
        rig.press()
        for _ in range(60):                        # 2 minutes, the screen moves, nobody presses
            await rig.step()
        c = rig.watch.state.counters
        bids_before_stale = len(rig.bids)
        assert c["input_stale_skips"] >= 29 and c["checks_over_floor"] >= 59
        # the bids that happened all came in the first 60 s after the press
        assert all(w.opened_at.timestamp() - (1_000_000.0) < 60 for w, _ in rig.bids)
        assert bids_before_stale >= 1 and rig.watch.status()["status"]["code"] in ("input_stale", "refused")
    run(go())


def test_a_game_with_no_floor_gets_no_look_and_the_page_says_so():
    async def go():
        rig = Rig(calibration=None)
        rig.press()
        for _ in range(40):
            await rig.step(press=True)
        assert not rig.bids and rig.watch.state.counters["ring_checks"] == 40
        assert rig.watch.status()["status"]["text"].startswith("No looks for this game yet")
        # every check with a check ~30 s before it lands in the game's histogram, labelled by input in those 30 s
        # (40 checks, 2 s apart: the first 14 have no 30-second base yet)
        assert sum(n for (g, i, _b), n in rig.watch.hist_pending.items() if g == GAME and i) == 26
    run(go())


def test_a_held_direction_counts_as_input_in_the_window():
    async def go():
        rig = Rig(calibration=None)
        rig.press(mask=128)                        # right held: no further mask change while walking
        for _ in range(25):
            await rig.step()
        labels = {i for (_g, i, _b) in rig.watch.hist_pending}
        assert labels == {True}
        rig.press(mask=0)                          # released
        for _ in range(25):                        # 50 s with nothing held: the last windows hold no input
            await rig.step(moving=False)
        assert any(not i for (_g, i, _b) in rig.watch.hist_pending)
    run(go())


# --- the look ----------------------------------------------------------------------------------------

def test_a_funded_look_takes_frames_at_the_grant_and_stores_one_note_with_its_want_and_lease():
    async def go():
        rig = Rig()
        rig.press()
        results = []
        for _ in range(20):
            results.append(await rig.step(press=True))
        assert W.GATE_READY in results
        assert len(rig.db.looks) == 1 and len(rig.bids) == 1
        want, object_text = rig.bids[0]
        assert (want.family, want.verb, want.tier, want.producer) == ("hub.retro_watch", "look", 1,
                                                                      "orion-hub.retro_watch.binding")
        assert want.object_ref == "retro_sitting:" + rig.watch.state.sitting_id and want.lanes == ["chat"]
        assert object_text == TITLE
        row = rig.db.looks[0]
        (look_id, sitting_id, game_id, looked_at, hashes, expire_at, note, outcome, failure, change, want_id,
         lease_id, lane_wait_ms, seconds, prompt_tokens, similar) = row
        assert look_id == want_id == want.want_id and lease_id == "lease-1" and outcome == "stored"
        assert note == "Link walks north through a forest clearing." and failure is None and prompt_tokens == 1086
        assert 2 <= len(hashes) <= 4 and len(rig.posts) == len(hashes) and change >= 0.01
        assert expire_at > datetime.fromtimestamp(rig.t, timezone.utc)          # last post + 1 h
        call = rig.calls[0]
        assert call.want_id == want.want_id and len(call.refs) == len(hashes)
        assert all(r.kind == "percept" and r.mime == "image/png" and r.width == 512 and r.height == 448
                   for r in call.refs)
        assert f"{len(hashes)} frames" in call.prompt and TITLE in call.prompt
        # the act: opened after the funded bid, closed 1, both on the want id; the want and the seat let go
        assert [a[0] for a in rig.acts] == ["opened", "closed"] and rig.acts[1][3] is True
        assert rig.acts[1][4].startswith("look stored: Link walks")
        assert rig.releases == [want.want_id] and rig.log[-2:] == ["released", "bus_closed"]
        c = rig.watch.state.counters
        assert (c["bids_funded"], c["model_calls"], c["notes_stored"], c["failed_looks"]) == (1, 1, 1, 0)
        assert rig.watch.status()["status"]["text"] == "Orion is watching."
        assert rig.watch.looks_for(rig.watch.state.sitting_id) == 1
        # the screen-change reference moved to the look's last frame; the ceiling holds the next look back
        assert rig.watch.state.max_change < 0.01 or rig.watch.state.last_look_at is not None
    run(go())


def test_a_refused_bid_opens_no_act_takes_no_seat_and_writes_no_row():
    async def go():
        rig = Rig(decision="deferred")
        rig.press()
        for _ in range(20):
            await rig.step(press=True)
        assert rig.bids and not rig.acts and not rig.db.looks and "granted" not in rig.log and not rig.calls
        assert rig.watch.state.counters["bids_funded"] == 0
        assert rig.watch.status()["status"]["text"] == "Orion is not looking: bid refused (deferred:price)."
        assert rig.releases                                   # a deferred want reserves nothing
    run(go())


def test_no_answer_from_the_market_is_no_look():
    async def go():
        rig = Rig(decision=None)
        rig.press()
        for _ in range(20):
            await rig.step(press=True)
        assert rig.bids and not rig.acts and not rig.calls
        assert rig.watch.status()["status"]["text"].endswith("(no_answer).")
    run(go())


def test_a_seat_the_pool_refuses_for_heat_closes_the_act_at_zero_and_writes_no_row():
    async def go():
        rig = Rig(seat_fail="shed:cabinet_hot")
        rig.press()
        for _ in range(20):
            await rig.step(press=True)
        assert [a[0] for a in rig.acts] == ["opened", "closed"] and rig.acts[1][3] is False
        assert "no_seat" in rig.acts[1][4] and not rig.calls and not rig.db.looks
        assert rig.watch.status()["status"]["code"] == "refused_heat"
    run(go())


def test_a_blind_lane_fails_the_look_and_stores_no_note():
    async def go():
        rig = Rig(reply={"content": "", "raw": {"error": "vision_unavailable"}})
        rig.press()
        for _ in range(20):
            await rig.step(press=True)
        row = rig.db.looks[0]
        assert row[7] == "failed" and row[6] is None and row[8].startswith("gateway:vision_unavailable")
        assert rig.acts[1][3] is False and rig.watch.state.counters["failed_looks"] == 1
        assert rig.watch.state.counters["notes_stored"] == 0
    run(go())


def test_empty_unclear_and_copied_notes_are_failed_looks():
    async def go():
        rig = Rig()
        rig.press()
        for _ in range(20):
            await rig.step(press=True)
        for content, outcome in (("Link walks north through a forest clearing.", "copied"), ("unclear", "unclear"),
                                 ("", "empty"), ("Link enters a cave.", "stored")):
            rig.reply = {"content": content}
            before = len(rig.db.looks)
            for _ in range(40):
                await rig.step(press=True)
                if len(rig.db.looks) > before:
                    break
            assert rig.db.looks[-1][7] == outcome
        c = rig.watch.state.counters
        assert c["notes_stored"] == 2 and c["failed_looks"] == 3
        assert [a[3] for a in rig.acts if a[0] == "closed"] == [True, False, False, False, True]
        assert rig.db.looks[-1][15] is not None                   # similar_to_previous of a stored note
    run(go())


def test_fewer_than_two_frames_never_reaches_the_model():
    async def go():
        rig = Rig()
        rig.press()
        for _ in range(20):
            rig.t += 2.0
            rig.press()
            rig.watch.ring.clear()                                  # the browser reconnected: one packet only
            rig.watch._last_kept = float("-inf")
            rig.frame()
            rig.watch.check(rig.t)
            await settle(rig.watch)
        assert not rig.calls and rig.db.looks and all(r[7] == "failed" for r in rig.db.looks)
        assert rig.db.looks[0][8].startswith("no_frames")
    run(go())


# --- the switch ----------------------------------------------------------------------------------------

def orion_act(act_id, object_ref, mode, words, at):
    return {"act_id": act_id, "object_ref": object_ref, "test_before": f"watch:{mode}",
            "outcome_note": f"watch {mode}: {words}", "session_id": "turn-1", "set_at": at}


def test_orions_off_stops_the_bid_and_juniper_cannot_turn_it_back_on():
    async def go():
        rig = Rig()
        rig.press()
        await rig.step(press=True)
        sid = rig.watch.state.sitting_id
        rig.db.acts.append(orion_act("a1", "retro_sitting:" + sid, "off", "I want to just keep company", datetime.now(timezone.utc)))
        assert await rig.watch.set_juniper("on")
        for _ in range(30):
            await rig.step(press=True)
        assert not rig.bids
        st = rig.watch.status()
        assert st["mode"] == "off" and st["set_by"] == "orion" and st["juniper_mode"] == "on"
        assert st["status"]["text"] == "Orion chose not to watch: watch off: I want to just keep company"
        # the copy carries Orion's act id and words; a second sync copies nothing twice
        assert [p["act_id"] for p in rig.db.prefs if p["set_by"] == "orion"] == ["a1"]
        await rig.watch.refresh_mode(rig.sitting.tracker.current)
        assert len([p for p in rig.db.prefs if p["set_by"] == "orion"]) == 1
    run(go())


def test_a_game_default_holds_from_the_first_press_and_juniper_off_holds_over_orion_on():
    async def go():
        rig = Rig()
        rig.db.acts.append(orion_act("a2", "retro_game:" + GAME, "off", "not this one", datetime.now(timezone.utc)))
        rig.press()
        for _ in range(30):
            await rig.step(press=True)
        assert not rig.bids and rig.watch.mode.mode == "off"
        rig.db.acts.append(orion_act("a3", "retro_sitting:" + rig.watch.state.sitting_id, "on", "ok, this time",
                                     datetime.now(timezone.utc)))
        await rig.watch.set_juniper("off")
        for _ in range(30):
            await rig.step(press=True)
        assert not rig.bids and rig.watch.mode == W.Mode("off", "juniper", None)
        await rig.watch.set_juniper("on")
        for _ in range(35):
            await rig.step(press=True)
        assert rig.bids                                            # both on: the look runs
    run(go())


def test_a_database_blip_keeps_the_last_mode():
    async def go():
        rig = Rig()
        rig.press()
        await rig.step(press=True)
        rig.db.acts.append(orion_act("a4", "retro_sitting:" + rig.watch.state.sitting_id, "off", "rest", datetime.now(timezone.utc)))
        cur = rig.sitting.tracker.current
        assert (await rig.watch.refresh_mode(cur)).mode == "off"
        rig.watch.pool_provider = lambda: None                      # the pool is gone
        assert (await rig.watch.refresh_mode(cur)).mode == "off"    # never back to "on" by a fault
    run(go())


def test_choice_from_act_reads_only_well_formed_choices():
    at = datetime.now(timezone.utc)
    row = choice_from_act(orion_act("x", "retro_game:tetris", "quiet", "words", at))
    assert row[:7] == ("act:x", "game", "tetris", "orion", "quiet", "watch quiet: words", "x")
    assert choice_from_act(orion_act("x", "camera:cam0", "off", "w", at)) is None
    assert choice_from_act({**orion_act("x", "retro_game:tetris", "off", "w", at), "test_before": "watch:loud"}) is None
    assert choice_from_act(orion_act("x", "retro_sitting:", "off", "w", at)) is None


# --- calibration through the service ----------------------------------------------------------------

def test_the_flush_adds_the_histogram_and_sets_the_floor_with_the_counts_it_came_from():
    async def go():
        rig = Rig(calibration=None)
        rig.press()
        for i in range(500):                                       # walking with presses, ~17 min
            await rig.step(press=True)
        rig.press(mask=0)                                          # hands off the pad
        for i in range(500):                                       # standing still, no press, ~17 min
            await rig.step(moving=False)
        await rig.watch.flush()
        cal = rig.watch.calibrations[GAME]
        assert cal.status == "set" and cal.floor is not None
        stored = rig.db.floors[GAME]
        assert stored[1] == cal.floor and stored[2] == "set" and stored[3] + stored[4] == cal.checks_input + cal.checks_no_input
        assert json.loads(stored[8])["input"] and sum(json.loads(stored[8])["no_input"]) == cal.checks_no_input
        assert not rig.watch.hist_pending
        # counters reach the sitting row, in report order
        counters = [a for s, a in rig.db.calls if s == retro_watch.COUNTERS_SQL]
        assert counters and counters[-1][0] == rig.watch.state.sitting_id and counters[-1][1] == 1000
    run(go())


# --- the sitting's play row --------------------------------------------------------------------------

def test_the_play_row_says_how_many_notes_orion_stored():
    s = retro_sitting.Sitting(sitting_id="s1", retro_session_id=SID, started_at=0.0, last_input_at=1800.0,
                              ended_at=1800.0, end_reason="idle", game_id=GAME, game_title=TITLE)
    assert retro_sitting.play_row(s, 3).summary.endswith("Orion watched: 3 notes.")
    assert retro_sitting.play_row(s, 3).looks == 3
    assert "watched" not in retro_sitting.play_row(s, 0).summary and retro_sitting.play_row(s, None).looks is None
    assert retro_sitting.play_row(s, 1).summary.endswith("Orion watched: 1 note.")
