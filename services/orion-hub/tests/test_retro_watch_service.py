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
SEEN = {"vision": {"status": "attached", "attachment_count": 4}}     # the gateway's raw.vision on a sighted call
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
                                                      "usage": {"prompt_tokens": 1086}, "raw": SEEN}
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


def test_a_game_with_no_floor_and_the_timer_off_gets_no_look_and_the_page_says_so():
    async def go():
        rig = Rig(calibration=None)
        rig.watch.no_floor_look = 0.0             # HUB_RETRO_WATCH_NO_FLOOR_LOOK_SEC=0: the old rule
        rig.press()
        for _ in range(40):
            await rig.step(press=True)
        assert not rig.bids and rig.watch.state.counters["ring_checks"] == 40
        assert rig.watch.status()["status"]["text"].startswith("No looks for this game yet")
        # every check with a check ~30 s before it lands in the game's histogram, labelled by input in those 30 s
        # (40 checks, 2 s apart: the first 14 have no 30-second base yet)
        assert sum(n for (g, i, _b), n in rig.watch.hist_pending.items() if g == GAME and i) == 26
    run(go())


NO_SEPARATION = W.Calibration(None, "no_separation", 900, 900, 3600.0)


def test_a_game_with_no_floor_looks_on_the_timer_while_buttons_are_pressed():
    """SimEarth, 2026-11-25: 302 ring checks, 0 looks. Now a still screen with presses gets one look a minute."""
    async def go():
        rig = Rig(calibration=NO_SEPARATION)
        rig.press()
        results = []
        for i in range(150):                       # 5 minutes, a still screen, a press every 10 s
            results.append(await rig.step(press=i % 5 == 0, moving=False))
        c = rig.watch.state.counters
        assert c["timer_looks"] == len(rig.bids) == len(rig.db.looks) == 4    # at 60, 120, 180, 240 s
        stamps = [w.opened_at.timestamp() for w, _ in rig.bids]
        assert [round(b - a) for a, b in zip(stamps, stamps[1:])] == [60, 60, 60]
        assert stamps[0] - 1_000_000.0 >= 60       # the timer starts at the first check, not at once
        # the same want as a floor look (family, producer, verb, tier, lanes) and the same GPU seat
        want, _ = rig.bids[0]
        assert (want.family, want.verb, want.tier, want.producer, want.lanes) == (
            "hub.retro_watch", "look", 1, "orion-hub.retro_watch.binding", ["chat"])
        assert rig.log.count("granted") == 4 and rig.log.count("released") == 4
        # the screen never changed: the model wrote the same words each time, and each is stored (a true look)
        assert [row[7] for row in rig.db.looks] == ["stored"] * 4 and c["notes_stored"] == 4
        assert rig.db.looks[-1][15] == 1.0         # similar_to_previous says it is the same note
        # each repeat is stored (fresh note) but its act closes 0: it said nothing new (Orion's review of #3995)
        closes = [a for a in rig.acts if a[0] == "closed"]
        assert [a[3] for a in closes] == [True, False, False, False]
        assert all(a[4].startswith("look repeat: ") for a in closes[1:])
        # no floor counter moved, and the page names the timer, never no_floor
        assert c["checks_over_floor"] == 0 and c["static_skips"] == 0 and c["input_stale_skips"] == 0
        assert rig.watch.status()["status"]["code"] == "timer"
        assert rig.watch.watch_snapshot()["status"] == "timer"
        assert rig.watch.watch_snapshot()["counts"]["timer_looks"] == 4
        # a note at most ~60 s old: world:play's 3-minute rule holds
        assert rig.t - rig.watch.state.last_stored_at < W.WATCHING_FRESH_SEC
    run(go())


def test_a_game_with_no_floor_and_no_button_press_gets_no_timer_look():
    async def go():
        rig = Rig(calibration=W.Calibration(None, "too_little_play", 10, 10, 40.0))
        rig.press()
        for i in range(150):                       # 5 minutes, the screen moves (a demo), one press at 42 s
            await rig.step(press=i == 20)
        c = rig.watch.state.counters
        assert len(rig.bids) == c["timer_looks"] == 1            # the one at 62 s, while the press was fresh
        assert rig.bids[0][0].opened_at.timestamp() - 1_000_000.0 == 62
        assert rig.watch.status()["status"]["code"] == "timer_input_stale"
        # the looks the timer missed are visible: a check where the look was due but no button changed in 60 s
        # (122 s .. 300 s, every 2 s), never a check that was only waiting for the interval
        assert c["input_stale_skips"] == 90 and c["static_skips"] == 0 and c["checks_over_floor"] == 0
        rig2 = Rig(calibration=W.Calibration(None, "one_sided", 900, 3, 1806.0))
        rig2.press()
        rig2.t += 61                                # the press is already stale when the watch starts
        for _ in range(60):
            await rig2.step()
        assert not rig2.bids and rig2.watch.state.counters["timer_looks"] == 0
    run(go())


def test_a_floored_game_never_looks_on_the_timer():
    async def go():
        rig = Rig()                                # FLOOR set
        rig.press()
        for i in range(150):                       # a still screen with presses: under the floor, no look
            assert await rig.step(press=i % 5 == 0, moving=False) in (W.GATE_STATIC,)
        assert not rig.bids and rig.watch.state.counters["timer_looks"] == 0
        assert rig.watch.status()["status"]["code"] == "static"
    run(go())


def test_the_timer_interval_is_one_env_key_and_zero_turns_it_off():
    async def go():
        rig = Rig(calibration=NO_SEPARATION)
        rig.watch.no_floor_look = 0.0
        rig.press()
        for i in range(60):
            await rig.step(press=i % 5 == 0, moving=False)
        assert not rig.bids and rig.watch.status()["status"]["code"] == "no_floor"
        rig = Rig(calibration=NO_SEPARATION)
        rig.watch.no_floor_look = 30.0
        rig.press()
        for i in range(60):                        # 2 minutes
            await rig.step(press=i % 5 == 0, moving=False)
        assert rig.watch.state.counters["timer_looks"] == 3       # at 32, 62, 92 s (first check at 2 s)
        cfg = SimpleNamespace(HUB_RETRO_WATCH_ENABLED=True, HUB_RETRO_WATCH_NO_FLOOR_LOOK_SEC=45)
        assert WatchService(cfg, sitting=rig.sitting).no_floor_look == 45.0
    run(go())


def test_a_timer_game_with_the_watch_off_counts_no_timer_look():
    async def go():
        rig = Rig(calibration=NO_SEPARATION)
        assert await rig.watch.set_juniper("off")
        rig.press()
        for i in range(100):
            await rig.step(press=i % 5 == 0, moving=False)
        assert not rig.bids and rig.watch.state.counters["timer_looks"] == 0
    run(go())


def test_an_unread_calibration_gives_no_timer_look():
    async def go():
        rig = Rig(calibration=None)
        rig.watch.pool_provider = lambda: None     # the floor table cannot be read: the game's floor is unknown
        rig.press()
        for i in range(90):                        # 3 minutes of presses
            await rig.step(press=i % 5 == 0, moving=False)
        assert not rig.bids and rig.watch.state.counters["timer_looks"] == 0
        assert rig.watch.status()["status"]["code"] == "no_floor"
    run(go())


def test_a_calibration_read_late_starts_the_timer_at_once():
    """A hub restart mid-sitting: the floor table answers only later (flush -> recalibrate)."""
    async def go():
        rig = Rig(calibration=None)
        rig.watch.pool_provider = lambda: None
        rig.press()
        for i in range(45):                        # 90 s, floor unknown: no look
            await rig.step(press=i % 5 == 0, moving=False)
        assert not rig.bids
        rig.watch.pool_provider = lambda: rig.db   # the database answers again
        rig.watch.calibrations[GAME] = NO_SEPARATION
        await rig.step(press=True, moving=False)   # more than 60 s since the first check: a look now
        assert len(rig.bids) == 1 and rig.watch.state.counters["timer_looks"] == 1
    run(go())


def test_a_refused_timer_bid_waits_the_interval_and_says_why():
    async def go():
        rig = Rig(decision="deferred", calibration=NO_SEPARATION)
        rig.press()
        for i in range(70):
            await rig.step(press=i % 5 == 0, moving=False)
        assert len(rig.bids) == 2 and not rig.db.looks and not rig.log      # no seat without funding
        assert rig.watch.status()["status"]["code"] == "refused"
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
        c = rig.watch.state.counters
        assert c["bids_funded"] == 1 and c["failed_looks"] == 1 and c["model_calls"] == 0
    run(go())


@pytest.mark.parametrize("reply,failure", [
    # the gateway's real refusal on a blind lane (orion-llm-gateway llm_backend.py): an "[Error: ...]" TEXT, no raw.error
    ({"content": "[Error: route 'retro_watch' cannot accept images (worker reports modalities.vision=false)]",
      "raw": {"vision": {"status": "refused"}}}, "vision:refused"),
    ({"content": "[Error: attachments could not be read: 404]", "raw": {"vision": {"status": "fetch_failed"}}},
     "vision:fetch_failed"),
    ({"content": "Link walks north.", "raw": {}}, "vision:unconfirmed"),        # no proof the model saw the frames
    ({"content": "[Error: something]", "raw": SEEN}, "gateway_error_text"),
    ({"content": "", "raw": {"error": "upstream_failed"}}, "gateway:upstream_failed"),
])
def test_a_look_the_model_could_not_see_stores_no_note(reply, failure):
    async def go():
        rig = Rig(reply=reply)
        rig.press()
        for _ in range(20):
            await rig.step(press=True)
        row = rig.db.looks[0]
        assert row[7] == "failed" and row[6] is None and row[8] == failure
        assert rig.acts[1][3] is False and rig.watch.state.counters["failed_looks"] == 1
        assert rig.watch.state.counters["notes_stored"] == 0
        assert rig.watch.status()["status"]["code"] != "watching"
    run(go())


def test_an_unexpected_error_mid_look_still_writes_the_row_and_closes_the_act():
    async def go():
        rig = Rig()
        async def boom(**kw): raise ConnectionError("bus dropped")
        rig.watch._call_model = boom
        rig.press()
        for _ in range(20):
            await rig.step(press=True)
        assert rig.db.looks[0][7] == "failed" and rig.db.looks[0][8] == "error:ConnectionError"
        assert [a[0] for a in rig.acts] == ["opened", "closed"] and rig.acts[1][3] is False
        assert rig.log.count("released") == 1
    run(go())


def test_no_look_until_both_sides_of_the_switch_were_read():
    async def go():
        rig = Rig()
        rig.watch.pool_provider = lambda: None          # the database is not up yet
        rig.press()
        for _ in range(20):
            await rig.step(press=True)
        assert not rig.bids and rig.watch.state.last_refusal[0] == "mode_unread"
    run(go())


def test_empty_unclear_and_copied_notes_are_failed_looks():
    async def go():
        rig = Rig()
        rig.press()
        for _ in range(20):
            await rig.step(press=True)
        for content, outcome in (("Link walks north through a forest clearing.", "copied"), ("unclear", "unclear"),
                                 ("", "empty"), ("Link enters a cave.", "stored")):
            rig.reply = {"content": content, "raw": SEEN}
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


# --- patch 3: frames for a turn, and one play call at a time ---------------------------------------------------

def test_frames_for_a_turn_post_the_newest_ring_frames_as_percepts():
    async def go():
        rig = Rig()
        rig.press()
        for _ in range(20):
            await rig.step(press=True)
        out = await rig.watch.frames_for_turn(4)
        return rig, out
    rig, out = run(go())
    assert out["status"] == "ok" and len(out["frames"]) == 4
    f = out["frames"][0]
    assert f["kind"] == "percept" and f["mime"] == "image/png" and f["frames_expire_at"] == out["frames_expire_at"]
    times = [x["kept_at"] for x in out["frames"]]
    assert times == sorted(times) and rig.t - times[0] <= 30.0
    assert any(sql == retro_watch.EXPIRY_SQL for sql, _a in rig.db.calls)    # the re-post moves every row's expiry


def test_no_ring_no_frames_and_no_sitting():
    async def go():
        rig = Rig()
        assert (await rig.watch.frames_for_turn())["status"] == "no_sitting"
        rig.press()
        assert (await rig.watch.frames_for_turn())["status"] == "no_frames"      # the browser shows nothing
        await rig.step(press=True)
        rig.t += 60                                                              # the ring went stale
        return await rig.watch.frames_for_turn()
    assert run(go())["status"] == "no_frames"


def test_frames_wait_for_a_look_in_flight():
    async def go():
        rig = Rig()
        rig.press()
        await rig.step(press=True)
        rig.watch._look_task = asyncio.get_running_loop().create_future()      # a look holds the seat
        out = await rig.watch.frames_for_turn()
        rig.watch._look_task.cancel()
        return out
    assert run(go())["status"] == "look_in_flight"


class LockRedis:
    def __init__(self, value): self.value = value
    async def get(self, key):
        assert key == retro_watch.LINE_LOCK_KEY
        return self.value


def test_no_look_is_bid_for_while_a_line_turn_runs():
    async def go():
        rig = Rig()
        rig.sitting.redis_provider = lambda: LockRedis(b"line-1")
        rig.press()
        for _ in range(40):
            await rig.step(press=True)
        return rig
    rig = run(go())
    assert rig.bids == [] and rig.watch.state.last_refusal[0] == "line_running"
    assert rig.watch.status()["looking"] is False
