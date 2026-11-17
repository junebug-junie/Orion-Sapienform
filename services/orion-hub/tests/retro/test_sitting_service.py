"""Hub's sitting record (scripts/retro_sitting.py): the relay hook, Redis, Postgres rows and the page routes."""
import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from orion.retro.sitting import SITTING_KEY
from scripts import retro_routes, retro_sitting, retro_transport

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


class Pool:
    def __init__(self): self.calls = []
    def acquire(self):
        pool = self
        class Ctx:
            async def __aenter__(self): return Conn(pool)
            async def __aexit__(self, *a): return False
        return Ctx()
    def named(self, table): return [c for c in self.calls if table in c[1]]


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
            if svc.tracker.current is None and not svc._jobs: break
        await settle(svc)
        assert SITTING_KEY not in redis.data                      # nobody is playing: the key is gone
        (_, sql, args), = pool.named("retro_play_sessions")
        row = dict(zip(retro_sitting._PLAY_COLUMNS, args))
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
        row = dict(zip(retro_sitting._PLAY_COLUMNS, args))
        assert row["player_stated"] is None and row["summary"].startswith("Someone played ") and row["end_reason"] == "shutdown"
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
    assert "You do not know who holds the controller." in line and "You are not watching the screen" in line
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
        assert c.get("/api/retro/sitting").json() == {"enabled": True, "sitting": None}
        assert app.state.retro_sitting is svc                         # the gateway's shutdown finds it here
        assert c.post("/api/retro/sitting/player", json={"player": "juniper"}).status_code == 403   # no origin
        assert c.post("/api/retro/sitting/player", json={"player": "juniper"}, headers=good).status_code == 409
        svc.tracker.on_input(SID, 0, 128, 50.0)
        for bad in ({"player": "Juniper"}, {"player": "a camera guess"}, {"player": "guest", "x": 1}, ["guest"]):
            assert c.post("/api/retro/sitting/player", json=bad, headers=good).status_code == 422
        r = c.post("/api/retro/sitting/player", json={"player": "guest"}, headers=good)
        assert r.status_code == 200 and r.json()["sitting"]["player_stated"] == "guest"
        got = c.get("/api/retro/sitting")
        assert got.headers["cache-control"] == "no-store" and got.json()["sitting"]["retro_session_id"] == SID


def test_the_chat_prompt_gets_one_play_line_only_while_someone_plays():
    from orion.hub import turn_orchestrator as orch

    open_ = {"sitting_id": "s", "started_at": 0.0, "last_input_at": 60.0 * 12, "ended_at": None,
             "game_title": "A Link to the Past", "player_stated": "juniper"}
    on = SimpleNamespace(HUB_RETRO_SITTING_ENABLED=True)

    async def check():
        r = Redis(); bus = SimpleNamespace(redis=r)
        assert await orch._situation_with_game_sitting("Situation: midday.", bus, settings=on) == "Situation: midday."
        r.data[SITTING_KEY] = json.dumps(open_)
        text = await orch._situation_with_game_sitting("Situation: midday.", bus, settings=on, correlation_id="c1")
        assert text.startswith("Situation: midday.\nPlay: Juniper is playing A Link to the Past on your Retro service, for 12 min so far.")
        assert (await orch._situation_with_game_sitting(None, bus, settings=on)).startswith("Play: Juniper is playing")
        off = SimpleNamespace(HUB_RETRO_SITTING_ENABLED=False)
        assert await orch._situation_with_game_sitting("Situation: midday.", bus, settings=off) == "Situation: midday."
        # a closed sitting, an unconnected bus and a broken Redis all leave the prompt as it was
        r.data[SITTING_KEY] = json.dumps(dict(open_, ended_at=800.0))
        assert await orch._situation_with_game_sitting("S", bus, settings=on) == "S"

        class NotConnected:
            @property
            def redis(self): raise RuntimeError("bus not connected")
        assert await orch._situation_with_game_sitting("S", NotConnected(), settings=on) == "S"
        assert await orch._situation_with_game_sitting("S", None, settings=on) == "S"
    asyncio.run(check())
