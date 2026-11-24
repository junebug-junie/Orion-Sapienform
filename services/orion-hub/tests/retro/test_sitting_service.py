"""Hub's sitting record (scripts/retro_sitting.py): the relay hook, Redis, Postgres rows and the page routes."""
import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from orion.retro.sitting import SITTING_KEY
from scripts import retro_routes, retro_sitting, retro_transport
from scripts.retro_memory_export import _COLUMNS

PLAY_COLUMNS = _COLUMNS + ("player_stated", "looks")   # the exporter's columns, then the ones a sitting adds

SID = "a" * 36


class Redis:
    def __init__(self): self.data, self.writes = {}, 0
    async def set(self, key, value, ex=None): self.data[key] = value; self.writes += 1
    async def get(self, key): return self.data.get(key)
    async def delete(self, key): self.data.pop(key, None)


class Conn:
    def __init__(self, pool): self.pool = pool
    async def execute(self, sql, *args): self.pool.calls.append(("execute", sql, args))
    async def fetchval(self, sql, *args):
        self.pool.calls.append(("fetchval", sql, args)); return args[0]
    async def fetch(self, sql, *args):
        self.pool.calls.append(("fetch", sql, args))
        return self.pool.missing if sql == retro_sitting.RECOVER_MISSING_SQL else []


class Pool:
    def __init__(self): self.calls, self.missing = [], []
    def acquire(self):
        pool = self
        class Ctx:
            async def __aenter__(self): return Conn(pool)
            async def __aexit__(self, *a): return False
        return Ctx()
    def named(self, table):
        """The writes to one table (its INSERT statements), oldest first."""
        return [c for c in self.calls if c[1].lstrip().startswith("INSERT INTO " + table)]


def msg(mask, port=0, **extra):
    return json.dumps({"version": 1, "session_id": SID, "generation": "g", "port": port, "epoch": 1, "seq": 1, "mask": mask, **extra})


def service(clock, *, redis=None, pool=None, **cfg):
    async def resolve(sid): return "a-link-to-the-past", "The Legend of Zelda: A Link to the Past"
    config = SimpleNamespace(HUB_RETRO_SITTING_ENABLED=True, HUB_RETRO_SITTING_GAP_SEC=300, HUB_RETRO_SITTING_MIN_SEC=120, **cfg)
    return retro_sitting.SittingService(config, redis_provider=lambda: redis, pool_provider=lambda: pool,
                                        resolve_game=resolve, clock=lambda: clock["t"])


async def settle(svc):
    while svc._jobs:
        await asyncio.gather(*list(svc._jobs))


def test_a_press_opens_a_sitting_with_its_game_in_redis_and_postgres(monkeypatch):
    async def check():
        monkeypatch.setattr(retro_sitting, "TICK_SEC", 3600)
        clock, redis, pool = {"t": 1000.0}, Redis(), Pool()
        svc = service(clock, redis=redis, pool=pool)
        svc.note_input(SID, msg(0))                       # the page announcing itself
        assert svc.status()["sitting"] is None and not svc._jobs
        svc.note_input(SID, msg(128))                     # a press
        await settle(svc)
        open_ = json.loads(redis.data[SITTING_KEY])
        assert open_["game_title"] == "The Legend of Zelda: A Link to the Past" and open_["ended_at"] is None
        assert open_["player_stated"] is None and open_["mask_changes"] == 1
        assert len(pool.named("retro_sitting")) == 1 and not pool.named("retro_play_sessions")
        writes = redis.writes
        for _ in range(100): svc.note_input(SID, msg(128))   # repeats of the held state
        assert not svc._jobs and redis.writes == writes
        await svc.close()
    asyncio.run(check())


def test_a_long_sitting_ends_as_one_human_play_row_that_credits_orion_with_nothing(monkeypatch):
    async def check():
        monkeypatch.setattr(retro_sitting, "TICK_SEC", 0.01)
        clock, redis, pool = {"t": 1000.0}, Redis(), Pool()
        svc = service(clock, redis=redis, pool=pool)
        svc.note_input(SID, msg(128)); await settle(svc)
        assert await svc.state_player("juniper") is True
        svc.note_action(SID, "save"); svc.note_action(SID, "load"); svc.note_action(SID, "save")
        for i, t in enumerate(range(1100, 1700, 100)):            # ten minutes of play: a change every 100 s
            clock["t"] = float(t); svc.note_input(SID, msg(0 if i % 2 == 0 else 128))
        assert not svc._jobs                                      # no gap was crossed: nothing closed or opened
        clock["t"] = 1600.0 + 301                                 # five minutes with no change
        for _ in range(200):
            await asyncio.sleep(0.01)
            # The tick closes the sitting first, then deletes the Redis key, then writes the rows: wait for all.
            if (svc.tracker.current is None and SITTING_KEY not in redis.data
                    and pool.named("retro_play_sessions") and not svc._jobs): break
        await settle(svc)
        assert SITTING_KEY not in redis.data                      # nobody is playing: the key is gone
        (_, sql, args), = pool.named("retro_play_sessions")
        row = dict(zip(PLAY_COLUMNS, args))
        assert row["kind"] == "human_play" and row["session_id"].startswith("sitting:")
        assert row["orion_frames"] == 0 and row["human_frames"] == 0 and row["human_share"] is None
        assert row["control_basis"] == "human_input" and row["played_together"] is False
        assert row["duration_sec"] == 600.0 and row["end_reason"] == "idle" and row["player_stated"] == "juniper"
        assert row["summary"] == "Juniper played The Legend of Zelda: A Link to the Past for 10 min. 2 saves and 1 load."
        assert row["dispatch_id"] is None and row["decisions"] is None and row["predictions_met"] == 0
        last = pool.named("retro_sitting")[-1][2]
        assert last[6] is not None and last[7] == "idle"          # ended_at, end_reason on the sitting row
        await svc.close()
    asyncio.run(check())


def test_a_short_sitting_leaves_no_play_row_and_nobody_stated_is_someone(monkeypatch):
    async def check():
        monkeypatch.setattr(retro_sitting, "TICK_SEC", 3600)
        clock, pool = {"t": 0.0}, Pool()
        svc = service(clock, redis=Redis(), pool=pool)
        svc.note_input(SID, msg(1)); await settle(svc)
        clock["t"] = 60.0; svc.note_input(SID, msg(0))
        svc.note_action(SID, "stop"); await settle(svc)
        assert svc.status()["sitting"] is None and not pool.named("retro_play_sessions")
        assert pool.named("retro_sitting")[-1][2][7] == "session_stopped"
        # the same stretch, long enough, with nobody stated
        svc.note_input(SID, msg(1)); await settle(svc)
        for i, t in enumerate((160.0, 260.0, 360.0, 400.0)):
            clock["t"] = t; svc.note_input(SID, msg(0 if i % 2 == 0 else 1))
        await svc.close()
        (_, _, args), = pool.named("retro_play_sessions")
        row = dict(zip(PLAY_COLUMNS, args))
        assert row["player_stated"] is None and row["summary"].startswith("Someone played ") and row["end_reason"] == "shutdown"
    asyncio.run(check())


def test_a_stick_alone_opens_a_sitting_through_the_relay_hook(monkeypatch):
    # Review finding: the live path made the axes coarse twice, so a stick push never counted.
    async def check():
        monkeypatch.setattr(retro_sitting, "TICK_SEC", 3600)
        svc = service({"t": 5.0}, redis=Redis(), pool=Pool())
        svc.note_input(SID, msg(0, axes=[90, -200, 0, 0])); await settle(svc)       # a resting stick
        assert svc.status()["sitting"] is None
        svc.note_input(SID, msg(0, axes=[32767, 0, 0, 0])); await settle(svc)      # a full push, no button
        assert svc.status()["sitting"]["mask_changes"] == 1
        await svc.close()
    asyncio.run(check())


def test_recover_closes_dead_rows_and_writes_the_play_rows_a_close_could_not(monkeypatch):
    # Review finding: a killed gateway left its row open for good, and a database that was down at a
    # close dropped the play row with no retry.
    async def check():
        from datetime import datetime, timedelta, timezone
        monkeypatch.setattr(retro_sitting, "TICK_SEC", 3600)
        clock, pool = {"t": 10.0}, Pool()
        svc = service(clock, redis=Redis(), pool=pool)
        t0 = datetime(2026, 11, 17, 3, 0, tzinfo=timezone.utc)
        pool.missing = [dict(sitting_id="dead1", retro_session_id=SID, game_id="tetris", game_title="Tetris",
                             started_at=t0, last_input_at=t0 + timedelta(minutes=30), ended_at=t0 + timedelta(minutes=30),
                             end_reason="recovered", mask_changes=900, saves=0, loads=0, pauses=2, player_stated="guest")]
        assert await svc.recover() == 1
        (_, sql, args), = [c for c in pool.calls if c[1] == retro_sitting.RECOVER_OPEN_SQL]
        assert args == ("",)                                   # nothing is open here: every open row is dead
        row = dict(zip(PLAY_COLUMNS, pool.named("retro_play_sessions")[-1][2]))
        assert (row["session_id"], row["kind"], row["duration_sec"], row["end_reason"]) == ("sitting:dead1", "human_play", 1800.0, "recovered")
        assert row["summary"] == "Someone played Tetris for 30 min." and row["orion_frames"] == 0
        # the sitting open in THIS process is never closed by its own recovery, and a new sitting recovers once
        pool.calls.clear(); pool.missing = []
        svc.note_input(SID, msg(128)); await settle(svc)
        (_, _, args), = [c for c in pool.calls if c[1] == retro_sitting.RECOVER_OPEN_SQL]
        assert args == (svc.status()["sitting"]["sitting_id"],)
        # no pool, a broken pool, a disabled service: never an exception
        assert await service(clock).recover() == 0
        class Broken:
            def acquire(self): raise RuntimeError("down")
        assert await service(clock, pool=Broken()).recover() == 0
        off = service(clock, pool=pool); off.enabled = False
        assert await off.recover() == 0
        await svc.close()
    asyncio.run(check())


def test_an_answer_lands_on_the_sitting_it_was_asked_about_and_survives_a_close(monkeypatch):
    # Review finding: state_player read the open sitting again after an await; a sitting that closed
    # in between gave an AttributeError (HTTP 500), and an old question could name a new sitting.
    async def check():
        monkeypatch.setattr(retro_sitting, "TICK_SEC", 3600)
        clock, pool = {"t": 0.0}, Pool()

        class ClosingRedis(Redis):
            svc = None
            async def set(self, key, value, ex=None):
                await super().set(key, value, ex)
                if self.svc is not None and json.loads(value).get("player_stated"):
                    self.svc.tracker.close_all("session_stopped")      # the sitting ends during the write
        redis = ClosingRedis()
        svc = service(clock, redis=redis, pool=pool); redis.svc = svc
        svc.note_input(SID, msg(1)); await settle(svc)
        sid = svc.status()["sitting"]["sitting_id"]
        assert await svc.state_player("juniper", "another-sitting") is False
        assert svc.status()["sitting"]["player_stated"] is None
        assert await svc.state_player("juniper", sid) is True          # no exception, though it closed meanwhile
        assert svc.status()["sitting"] is None
        last = pool.named("retro_sitting")[-1][2]
        assert last[0] == sid and last[12] == "juniper"                # the answer is on that sitting's row
        assert await svc.state_player("guest", sid) is False           # nothing is open now
        await svc.close()
    asyncio.run(check())


def test_a_slow_database_never_holds_the_redis_copy_or_the_loop(monkeypatch):
    # Review finding: the wait for a pooled connection had no bound, and the Redis key was removed
    # only after the database writes. A chat turn was told "is playing" for minutes after Stop.
    async def check():
        monkeypatch.setattr(retro_sitting, "TICK_SEC", 3600)
        monkeypatch.setattr(retro_sitting, "OP_TIMEOUT_SEC", 0.05)

        class Stuck:
            def acquire(self):
                class Ctx:
                    async def __aenter__(self): await asyncio.sleep(3600)
                    async def __aexit__(self, *a): return False
                return Ctx()
        clock, redis = {"t": 0.0}, Redis()
        svc = service(clock, redis=redis, pool=Stuck())
        svc.note_input(SID, msg(1))
        await asyncio.wait_for(settle(svc), 3)                         # opening does not hang on the database
        assert SITTING_KEY in redis.data
        clock["t"] = 200.0; svc.note_input(SID, msg(0))
        svc.note_action(SID, "stop")
        await asyncio.sleep(0)                                         # one turn of the loop: the key is already gone
        await asyncio.sleep(0)
        assert SITTING_KEY not in redis.data
        await asyncio.wait_for(settle(svc), 3)
        await asyncio.wait_for(svc.close(), 3)
    asyncio.run(check())


def test_small_rules_zero_is_a_value_the_loop_comes_back_and_a_title_is_asked_again(monkeypatch):
    async def check():
        monkeypatch.setattr(retro_sitting, "TICK_SEC", 3600)
        clock = {"t": 0.0}
        cfg = SimpleNamespace(HUB_RETRO_SITTING_ENABLED=True, HUB_RETRO_SITTING_GAP_SEC=0, HUB_RETRO_SITTING_MIN_SEC=0)
        tuned = retro_sitting.SittingService(cfg)
        assert tuned.min_sec == 0.0 and tuned.tracker.gap_sec == 0.0   # 0 is not "unset"
        assert retro_sitting.SittingService(SimpleNamespace()).min_sec == 120.0
        # a loop that died comes back with the next input of the same sitting
        calls = {"n": 0}
        async def resolve(sid):
            calls["n"] += 1
            if calls["n"] == 1: raise RuntimeError("retro is starting")
            return "tetris", "Tetris"
        pool = Pool()
        svc = retro_sitting.SittingService(SimpleNamespace(), pool_provider=lambda: pool, resolve_game=resolve,
                                           clock=lambda: clock["t"])
        svc.note_input(SID, msg(1)); await settle(svc)
        assert svc.status()["sitting"]["game_title"] is None            # the first ask failed
        svc._loop_task.cancel(); await asyncio.gather(svc._loop_task, return_exceptions=True)
        clock["t"] = 30.0; svc.note_input(SID, msg(0))                  # a change inside the same sitting
        assert not svc._loop_task.done()
        clock["t"] = 200.0; svc.note_input(SID, msg(1))
        await svc.close()                                               # the close asks for the title again
        row = dict(zip(PLAY_COLUMNS, pool.named("retro_play_sessions")[-1][2]))
        assert row["game_title"] == "Tetris" and row["summary"] == "Someone played Tetris for 3 min."
        # a socket that dropped with a button held leaves no state: the reconnect's neutral message opens nothing
        again = service(clock); again.note_input(SID, msg(128)); await settle(again)
        again.note_disconnect(SID)
        again.tracker.close_all()
        again.note_input(SID, msg(0)); await settle(again)
        assert again.status()["sitting"] is None
        await again.close()
    asyncio.run(check())


def test_disabled_tracks_nothing_and_faults_never_reach_the_relay(monkeypatch):
    async def check():
        clock = {"t": 0.0}
        off = service(clock, redis=Redis(), pool=Pool()); off.enabled = False
        off.note_input(SID, msg(128)); off.note_action(SID, "save")
        assert off.status() == {"enabled": False, "sitting": None} and await off.state_player("juniper") is False

        class Broken:
            def __getattr__(self, name): raise RuntimeError("down")
        monkeypatch.setattr(retro_sitting, "TICK_SEC", 3600)
        svc = service(clock, redis=Broken(), pool=Broken())
        svc.note_input(SID, "not json"); svc.note_input(SID, msg(128)); await settle(svc)   # no exception
        assert svc.status()["sitting"]["mask_changes"] == 1
        assert await svc.state_player("a name from a camera") is False
        await svc.close()
    asyncio.run(check())


def test_the_prompt_line_says_what_is_known_and_only_that():
    open_ = {"sitting_id": "s", "started_at": 1000.0, "last_input_at": 1000.0 + 47 * 60, "ended_at": None,
             "game_title": "A Link to the Past", "player_stated": None}
    line = retro_sitting.sitting_line(open_, now=1000.0 + 47 * 60 + 5)
    assert line.startswith("Someone is playing A Link to the Past on your Retro service, for 47 min so far.")
    assert "You do not know who holds the controller." in line and "it holds nothing you saw on the screen" in line
    told = retro_sitting.sitting_line(dict(open_, player_stated="juniper"), now=1000.0 + 50 * 60)
    assert told.startswith("Juniper is playing") and "do not know who" not in told and "3 min ago" in told
    for not_open in (None, {}, "x", dict(open_, ended_at=2000.0), {"started_at": "?", "last_input_at": 1}):
        assert retro_sitting.sitting_line(not_open) is None

    async def read():
        r = Redis()
        assert await retro_sitting.read_sitting(r) is None
        r.data[SITTING_KEY] = json.dumps(open_).encode()
        assert (await retro_sitting.read_sitting(r))["game_title"] == "A Link to the Past"
        r.data[SITTING_KEY] = b"{broken"
        assert await retro_sitting.read_sitting(r) is None
        class Down:
            async def get(self, key): raise RuntimeError("down")
        assert await retro_sitting.read_sitting(Down()) is None
    asyncio.run(read())


def test_the_relay_hands_each_forwarded_input_message_to_the_observer_and_survives_its_faults():
    class Browser:
        def __init__(self): self.incoming = asyncio.Queue()
        async def receive_text(self):
            item = await self.incoming.get()
            if isinstance(item, Exception): raise item
            return item
        async def send_text(self, data): pass
        send_bytes = send_text

    class Native:
        def __init__(self): self.sent, self.incoming = [], asyncio.Queue()
        async def receive(self): return await self.incoming.get()
        async def send_str(self, m): self.sent.append(m)

    async def check():
        seen = []
        def observer(m):
            seen.append(m)
            if len(seen) == 2: raise RuntimeError("observer bug")
        ws, native = Browser(), Native()
        task = asyncio.create_task(retro_transport.relay_messages(ws, native, "input", SID, on_input=observer))
        for m in ("one", "two", "three"): ws.incoming.put_nowait(m)
        ws.incoming.put_nowait(WebSocketDisconnect(1000))
        assert await asyncio.wait_for(task, 1) == "browser_disconnected"
        assert native.sent == ["one", "two", "three"] and seen == ["one", "two", "three"]
        # the media channel never calls it: acknowledgments are not input
        ws2 = Browser(); ws2.incoming.put_nowait(WebSocketDisconnect(1000))
        assert await asyncio.wait_for(retro_transport.relay_messages(ws2, Native(), "media", SID, on_input=observer), 1) == "browser_disconnected"
        assert len(seen) == 3
    asyncio.run(check())


def test_the_page_routes(tmp_path, monkeypatch):
    monkeypatch.setattr(retro_sitting, "TICK_SEC", 3600)
    (tmp_path / "hub.token").write_text("y" * 48)
    cfg = SimpleNamespace(HUB_RETRO_ENABLED=True, HUB_RETRO_ORIGIN="http://testserver", HUB_RETRO_AUTH_DIR=str(tmp_path),
                          HUB_RETRO_SOCKET="/nonexistent", HUB_RETRO_SITTING_ENABLED=True)
    clock = {"t": 50.0}
    svc = retro_sitting.SittingService(cfg, clock=lambda: clock["t"])
    app = FastAPI(); app.include_router(retro_routes.create_router(cfg, sitting=svc))
    good = {"Origin": "http://testserver"}
    with TestClient(app) as c:
        first = c.get("/api/retro/sitting").json()
        assert first.pop("watch")["status"]["code"] == "no_sitting"   # patch 2: the watch rides along
        assert first == {"enabled": True, "sitting": None}
        assert app.state.retro_sitting is svc                         # the gateway's shutdown finds it here
        assert c.post("/api/retro/sitting/player", json={"player": "juniper"}).status_code == 403   # no origin
        assert c.post("/api/retro/sitting/player", json={"player": "juniper"}, headers=good).status_code == 409
        svc.tracker.on_input(SID, 0, 128, 50.0)
        for bad in ({"player": "Juniper"}, {"player": "a camera guess"}, {"player": "guest", "x": 1}, ["guest"]):
            assert c.post("/api/retro/sitting/player", json=bad, headers=good).status_code == 422
        # an answer names the sitting it was asked about: one for an older sitting is refused, not misplaced
        assert c.post("/api/retro/sitting/player", json={"player": "guest", "sitting_id": 7}, headers=good).status_code == 422
        assert c.post("/api/retro/sitting/player", json={"player": "guest", "sitting_id": "older"}, headers=good).status_code == 409
        assert svc.status()["sitting"]["player_stated"] is None
        open_id = svc.status()["sitting"]["sitting_id"]
        r = c.post("/api/retro/sitting/player", json={"player": "guest", "sitting_id": open_id}, headers=good)
        assert r.status_code == 200 and r.json()["sitting"]["player_stated"] == "guest"
        got = c.get("/api/retro/sitting")
        assert got.headers["cache-control"] == "no-store" and got.json()["sitting"]["retro_session_id"] == SID


def test_the_game_page_works_when_the_record_cannot_be_loaded(tmp_path, monkeypatch):
    # The record is an observer. A fault while it loads must leave the router (the relay, the proxy,
    # the page) as it was before patch 1.
    import sys
    # The module create_router will import at call time (another test file may have reloaded scripts.*,
    # so the name bound at the top of this file is not always that module).
    live = sys.modules["scripts.retro_sitting"]
    real = live.SittingService
    def boom(*a, **k): raise ImportError("a dependency of the record is broken")
    monkeypatch.setattr(live, "SittingService", boom)
    (tmp_path / "hub.token").write_text("y" * 48)
    cfg = SimpleNamespace(HUB_RETRO_ENABLED=True, HUB_RETRO_ORIGIN="http://testserver", HUB_RETRO_AUTH_DIR=str(tmp_path),
                          HUB_RETRO_SOCKET="/nonexistent")
    router = retro_routes.create_router(cfg)
    assert isinstance(router.retro_sitting, retro_routes.NoSitting)
    app = FastAPI(); app.include_router(router)
    with TestClient(app) as c:
        page = c.get("/api/retro/sitting").json()
        assert page.pop("watch")["enabled"] is False                  # no record, no watch: the page says so
        assert page == {"enabled": False, "sitting": None}
        assert c.post("/api/retro/sitting/player", json={"player": "guest"},
                      headers={"Origin": "http://testserver"}).status_code == 409
    none = router.retro_sitting
    none.note_input(SID, msg(1)); none.note_action(SID, "stop"); none.note_disconnect(SID)
    assert asyncio.run(none.recover()) == 0 and asyncio.run(none.close()) is None
    # every call the router makes on a real service exists on the stand-in
    used = {"note_input", "note_action", "note_disconnect", "status", "state_player", "recover", "close"}
    assert used <= set(dir(retro_routes.NoSitting)) and used <= set(dir(real))
