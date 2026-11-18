"""Orion's lines while a person plays (scripts/retro_commentary.py), with every outside thing faked: the market,
the GPU pool, the gateway, Orion's now, the act ledger and Postgres (an in-memory table with the real rules for
running -> yielded / delivered). Delivery goes through the REAL EndogenousOutreach.deliver_line, so "no push, no
history row" is checked on the real door. No sleeping: each test waits on observable outcomes (events and tasks).

Not in tests/retro/: it imports scripts.endogenous_outreach (and through it the turn orchestrator)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import scripts.endogenous_outreach as eo
from orion.retro import commentary as C
from scripts import retro_commentary as rc
from scripts.retro_commentary import CommentaryService, prepare_retro_turn

NOW = 1_800_000_000.0
SID = "sitting-1"
SITTING = {"sitting_id": SID, "game_id": "alttp", "game_title": "A Link to the Past", "started_at": NOW - 1800,
           "last_input_at": NOW - 5, "saves": 1, "loads": 0, "player_stated": None, "ended_at": None}
FRAMES = {"status": "ok", "sitting_id": SID, "frames_expire_at": "2027-01-01T00:00:00+00:00",
          "frames": [{"kind": "percept", "sha256": f"{i:064x}", "mime": "image/png", "bytes": 10, "width": 512,
                      "height": 448, "source_url": f"http://store/percepts/{i:064x}", "kept_at": NOW - 30 + 10 * i,
                      "frames_expire_at": "2027-01-01T00:00:00+00:00"} for i in range(4)]}


def ts(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, timezone.utc)


class Table:
    """retro_line and retro_watch_look, with the statements the service runs."""

    def __init__(self, looks=None):
        self.lines: dict[str, dict] = {}
        self.looks = {lk["look_id"]: dict(lk) for lk in (looks or [])}
        self.events: list[str] = []
        self.replied: list[tuple] = []

    def acquire(self):
        db = self

        class Ctx:
            async def __aenter__(self): return Conn(db)
            async def __aexit__(self, *a): return False
        return Ctx()


class Conn:
    def __init__(self, db: Table):
        self.db = db

    async def execute(self, sql, *v):
        db = self.db
        if sql is rc.INSERT_SQL:
            db.lines[v[0]] = {"line_id": v[0], "sitting_id": v[1], "outcome": "running", "decided_at": v[5]}
            db.events.append("row_inserted")
            return "INSERT 0 1"
        if sql is rc.REPLIED_SQL:
            db.replied.append(v)
            return "UPDATE 0"
        raise AssertionError(f"unexpected execute: {sql[:60]}")

    async def executemany(self, sql, rows):
        assert sql is rc.SETTLE_SQL
        for look_id, checkable, verdict, at, line_id in rows:
            lk = self.db.looks.get(look_id)
            if lk is not None and lk.get("settled_at") is None:
                lk.update(checkable=checkable, verdict=verdict, settled_at=at, settled_by_line=line_id)
        self.db.events.append("settled")

    async def fetchval(self, sql, *v):
        db = self.db
        if sql is rc.YIELD_SQL:
            row = db.lines.get(v[0])
            db.events.append("yield_marked")
            if row and row["outcome"] == "running":
                row["outcome"] = "yielded"
                return v[0]
            return None
        if sql is rc.CLAIM_SQL:
            row = db.lines.get(v[0])
            db.events.append("claim")
            if row and row["outcome"] == "running":
                row["outcome"] = "delivered"
                return v[0]
            return None
        if sql is rc.FINISH_SQL:
            row = db.lines[v[0]]
            if row["outcome"] == "running":
                row["outcome"] = v[1]
            row.update(failure=v[3], text=v[4], look_ids=v[5], frame_hashes=v[6], now_cites=v[8], looks_shown=v[9],
                       looks_checkable=v[10], looks_contradicted=v[11], stance_sec=v[12], motor_sec=v[13],
                       finalize_sec=v[14], lease_priorities=v[17], message_id=v[18])
            return row["outcome"]
        if sql is rc.NEWEST_NOTE_SQL:
            return max((ts(lk["looked_at"]) for lk in db.looks.values()), default=None)
        if sql is rc.NEWEST_NOTE_TEXT_SQL:
            return max(db.looks.values(), key=lambda lk: lk["looked_at"])["note"] if db.looks else None
        if sql in (rc.LAST_ATTEMPT_SQL, rc.LAST_DELIVERED_SQL):
            return None
        raise AssertionError(f"unexpected fetchval: {sql[:60]}")

    async def fetch(self, sql, *v):
        if sql is rc.NOTES_SQL:
            since = v[1].timestamp()
            rows = [lk for lk in self.db.looks.values() if lk["looked_at"] > since]
            return sorted(rows, key=lambda r: -r["looked_at"])
        if sql is rc.PREVIOUS_LINES_SQL:
            return []
        raise AssertionError(f"unexpected fetch: {sql[:60]}")


def look(i: int, at: float, note: str, sim=None) -> dict:
    return {"look_id": f"look-{i}", "looked_at": at, "note": note, "similar_to_previous": sim, "settled_at": None,
            "verdict": "unchecked"}


class Lease:
    grant = SimpleNamespace(lease_id="hold-1", generation=1, role="chat-lane")


class Seat:
    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        self.log.append("hold_granted")
        return Lease()

    async def __aexit__(self, *a):
        self.log.append("hold_released")
        return False


def make_outreach(monkeypatch):
    rows: list[dict] = []
    import scripts.endogenous_outreach_decisions as decisions_mod

    monkeypatch.setattr(decisions_mod, "record_decision", lambda result, **kw: rows.append(result))
    outreach = eo.EndogenousOutreach(enabled=True, tick_interval_sec=300.0, min_cooldown_sec=0.0, daily_cap=0,
                                     quiet_start_hour=-1, quiet_end_hour=-1, timeout_sec=5.0,
                                     agent_lane_timeout_sec=5.0, notify_channel="orion:notify:in_app",
                                     fallback_session_id="orion_outreach", trigger_evaluator=lambda: None,
                                     timezone_name="UTC")
    queue: asyncio.Queue = asyncio.Queue()
    outreach._connections["c1"] = {"queue": queue, "session_id": "juniper-main"}
    outreach._connections["c2"] = {"queue": asyncio.Queue(), "session_id": "retro-guest-sitting-1"}
    history: list[dict] = []
    notified: list[dict] = []

    async def no_tension(self):
        return None

    async def fake_history(self, **kw):
        history.append(kw)

    async def fake_notify(self, **kw):
        notified.append(kw)

    monkeypatch.setattr(eo.EndogenousOutreach, "_fetch_fresh_tension_reason", no_tension)
    monkeypatch.setattr(eo.EndogenousOutreach, "_publish_history", fake_history)
    monkeypatch.setattr(eo.EndogenousOutreach, "_publish_notification", fake_notify)
    return outreach, queue, history, notified, rows


def make_service(monkeypatch, *, looks=None, bid_outcome="funded", frames=FRAMES, compose=None, sitting=SITTING):
    db = Table(looks if looks is not None else [look(1, NOW - 600, "Link enters a cave by the well."),
                                                 look(2, NOW - 15, "Link walks down a dark corridor.")])
    outreach, queue, history, notified, rows = make_outreach(monkeypatch)
    log: list[str] = db.events
    calls: dict = {"compose": [], "acts": [], "released": [], "uses": [], "bids": [], "motor_cancel": []}

    async def bid(want, object_text):
        calls["bids"].append((want, object_text))
        return SimpleNamespace(outcome=bid_outcome, reason="ok")

    async def release(want_id):
        calls["released"].append(want_id)

    def seat(bus, want_id):
        calls["seat"] = (want_id,)
        return Seat(log)

    async def gateway_get(path, *, auth, timeout):
        if path.startswith("/api/retro/sitting/frames"):
            assert auth is True
            return 200, frames
        return 200, {"watch": {"mode": "on", "looking": False}}

    async def read_now(line_id):
        return None

    async def publish_act(bus, **kw):
        calls["acts"].append(kw)
        return True

    async def cancel_motor(corr):
        calls["motor_cancel"].append(corr)
        log.append("motor_cancelled")

    async def default_compose(prompt, corr, **kw):
        calls["compose"].append((prompt, corr, kw))
        return "You found the cave behind the well.\nCHECKS:\nN2: holds", {"harness_fcc_elapsed_sec": 40.0}

    monkeypatch.setattr(outreach, "compose", compose(calls, log) if compose else default_compose)
    bus = SimpleNamespace(redis=FakeRedis(sitting))
    svc = CommentaryService(SimpleNamespace(HUB_RETRO_COMMENTARY_ENABLED=True, HUB_RETRO_LINE_MIN_GAP_SEC=120.0),
                            bus_provider=lambda: bus, pool_provider=lambda: db, outreach_provider=lambda: outreach,
                            clock=lambda: NOW, bid=bid, release=release, seat=seat, gateway_get=gateway_get,
                            read_now=read_now, publish_act=publish_act, cancel_motor=cancel_motor)
    return SimpleNamespace(svc=svc, db=db, outreach=outreach, queue=queue, history=history, notified=notified,
                           rows=rows, calls=calls, log=log, bus=bus)


class FakeRedis:
    def __init__(self, sitting):
        import json
        self.data = {"orion:retro:sitting": json.dumps(sitting) if sitting else None}
        self.sets: list = []

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.sets.append((key, value, ex))
        self.data[key] = value

    async def delete(self, key):
        self.sets.append((key, None, None))
        self.data.pop(key, None)


def drain(q: asyncio.Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


# --- a line is delivered ---------------------------------------------------------------------------------

def test_a_funded_line_runs_held_at_system_with_frames_and_is_delivered(monkeypatch):
    t = make_service(monkeypatch)

    async def go():
        return await t.svc.tick()
    assert asyncio.run(go()) == C.GATE_READY
    line_id = t.calls["bids"][0][0].want_id
    row = t.db.lines[line_id]
    assert row["outcome"] == "delivered" and row["text"] == "You found the cave behind the well."
    prompt, corr, kw = t.calls["compose"][0]
    assert kw["lane"] == "chat" and kw["source"] == "retro_commentary"
    assert kw["gpu_lease"].lease_id == "hold-1" and kw["session_id"] == "juniper-main"
    assert [a["kind"] for a in kw["attachments"]] == ["percept"] * 4
    assert "kept_at" not in kw["attachments"][0] and kw["attachments"][0]["frames_expire_at"]
    assert "Attached: 4 frames" in prompt and "CHECKABLE" in prompt
    # the line's own priorities: the hold and the route every held stage attaches with; never interactive
    assert "interactive" not in row["lease_priorities"] and row["lease_priorities"][0] == "system"
    # pushed once to every socket, with the line's keys, and written once as Orion's own row; no notification
    pushed = drain(t.queue)
    assert len(pushed) == 1 and pushed[0]["source"] == "retro_commentary" and pushed[0]["line_id"] == line_id
    assert pushed[0]["retro_sitting_id"] == SID and pushed[0]["kind"] == "orion_outreach"
    assert len(t.history) == 1 and t.history[0]["source_tag"] == "retro_commentary" and not t.notified
    assert t.calls["acts"] == [{"line_id": line_id, "sitting_id": SID, "text": row["text"]}]
    # the checkable note was judged; the old one was shown and settled unchecked
    assert t.db.looks["look-2"]["verdict"] == "checked_holds" and t.db.looks["look-2"]["checkable"] is True
    assert t.db.looks["look-1"]["verdict"] == "unchecked" and t.db.looks["look-1"]["checkable"] is False
    assert row["looks_shown"] == 2 and row["looks_checkable"] == 1 and row["looks_contradicted"] == 0
    assert row["motor_sec"] == 40.0
    assert t.calls["released"] == [line_id]
    keys = [k for k, *_ in t.bus.redis.sets]
    assert keys == [rc.LINE_LOCK_KEY, rc.LINE_LOCK_KEY]          # set, then cleared
    assert t.log.index("hold_granted") < t.log.index("claim") and "hold_released" in t.log


def test_an_unfunded_bid_writes_no_row_and_runs_no_turn(monkeypatch):
    t = make_service(monkeypatch, bid_outcome="deferred")
    asyncio.run(t.svc.tick())
    assert t.db.lines == {} and t.calls["compose"] == [] and len(t.calls["released"]) == 1


def test_no_ring_means_no_turn_and_outcome_no_frames(monkeypatch):
    t = make_service(monkeypatch, frames={"status": "no_frames", "sitting_id": SID})
    asyncio.run(t.svc.tick())
    (row,) = t.db.lines.values()
    assert row["outcome"] == "no_frames" and t.calls["compose"] == [] and drain(t.queue) == []


def test_silence_is_valid_a_pass_writes_a_decision_row_and_no_act(monkeypatch):
    def compose(calls, log):
        async def f(prompt, corr, **kw):
            return "PASS\nCHECKS:\nN2: contradicted", {}
        return f
    t = make_service(monkeypatch, compose=compose)
    asyncio.run(t.svc.tick())
    (row,) = t.db.lines.values()
    assert row["outcome"] == "passed" and row["looks_contradicted"] == 1
    assert t.db.looks["look-2"]["verdict"] == "contradicted"        # judged also when Orion passes
    assert drain(t.queue) == [] and t.history == [] and t.calls["acts"] == []
    assert [r["reason"] for r in t.rows] == ["orion_passed"] and t.rows[0]["source"] == "retro_commentary"
    assert len(t.calls["released"]) == 1


def test_a_line_never_runs_if_the_held_route_were_interactive(monkeypatch):
    t = make_service(monkeypatch)
    monkeypatch.setattr(rc, "held_route_priority", lambda: "interactive")
    assert asyncio.run(t.svc.run_line(SITTING)) is None
    assert t.calls["bids"] == [] and t.db.lines == {}


def test_the_held_route_in_the_pool_config_is_system():
    """Acceptance check (spec, patch 3): a line turn's stance, motor and finalize run at system. They attach to the
    line's hold through the held route (orion.llm.resource_lease.GPU_LEASE_ROUTE); its priority is read here from
    the real config/gpu_pool.yaml."""
    assert rc.held_route_priority() == "system"


# --- Juniper goes first -------------------------------------------------------------------------------

def blocking_compose(started: asyncio.Event, release: asyncio.Event, text: str):
    def factory(calls, log):
        async def f(prompt, corr, **kw):
            calls["compose"].append((prompt, corr, kw))
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                log.append("turn_cancelled")
                raise
            return text, {}
        return f
    return factory


def test_her_message_mid_turn_marks_yielded_before_the_kill_and_nothing_is_said(monkeypatch):
    """Also the mid-stance case: the fake turn has not reached any stage that could deliver (orion-thought has no
    cancel listener, #3658): no chat row, no push, no delivery."""
    started, release = asyncio.Event(), asyncio.Event()
    t = make_service(monkeypatch, compose=blocking_compose(started, release, "too late"))

    async def go():
        line = asyncio.ensure_future(t.svc.tick())
        await started.wait()
        t.svc.note_human_message(surface=None, player=None)
        await line
    asyncio.run(go())
    (row,) = t.db.lines.values()
    assert row["outcome"] == "yielded"
    assert t.log.index("yield_marked") < t.log.index("turn_cancelled")      # the row first, then the kill
    assert "claim" not in t.log and drain(t.queue) == [] and t.history == [] and t.calls["acts"] == []
    assert t.calls["motor_cancel"] == [row["line_id"]]
    assert "hold_released" in t.log


def test_a_draft_that_finishes_after_the_yield_is_refused_by_delivery(monkeypatch):
    """The hard order: Hub marks the line yielded, the cancel does not reach the turn, the draft finishes and
    delivery still tries. The claim refuses it: no push, no history row, no act."""
    started, release = asyncio.Event(), asyncio.Event()
    t = make_service(monkeypatch, compose=blocking_compose(started, release, "A finished draft."))

    async def go():
        line = asyncio.ensure_future(t.svc.tick())
        await started.wait()
        # The row is marked (as yield_line does) but the turn is NOT cancelled: it finishes its draft.
        conn = Conn(t.db)
        await conn.fetchval(rc.YIELD_SQL, t.svc.running.line_id, "juniper_spoke")
        release.set()
        await line
    asyncio.run(go())
    (row,) = t.db.lines.values()
    assert row["outcome"] == "yielded"
    assert "claim" in t.log and t.log.index("yield_marked") < t.log.index("claim")
    assert drain(t.queue) == [] and t.history == [] and t.calls["acts"] == []
    assert t.rows[-1]["reason"] == "yielded"


def test_a_message_with_no_line_running_only_restarts_the_gap(monkeypatch):
    t = make_service(monkeypatch)
    t.svc._sitting_id = SID

    async def go():
        t.svc.note_human_message(surface="retro", player="guest")
        await asyncio.gather(*list(t.svc._jobs))
    asyncio.run(go())
    assert t.svc._last_exchange_at == NOW and t.db.replied == [(SID, "guest", C.REPLY_WINDOW_SEC)]


# --- the gate --------------------------------------------------------------------------------------------

def test_quiet_mode_gives_no_line(monkeypatch):
    t = make_service(monkeypatch)

    async def gw(path, *, auth, timeout):
        return 200, {"watch": {"mode": "quiet", "looking": False}}
    t.svc._gateway_get = gw
    assert asyncio.run(t.svc.tick()) == C.GATE_MODE and t.calls["bids"] == []


def test_a_look_in_flight_holds_the_line(monkeypatch):
    t = make_service(monkeypatch)

    async def gw(path, *, auth, timeout):
        return 200, {"watch": {"mode": "on", "looking": True}}
    t.svc._gateway_get = gw
    assert asyncio.run(t.svc.tick()) == C.GATE_LOOK and t.calls["bids"] == []


def test_no_sitting_no_line(monkeypatch):
    t = make_service(monkeypatch, sitting=None)
    assert asyncio.run(t.svc.tick()) == C.GATE_NO_SITTING


# --- the chat box from the game page -----------------------------------------------------------------------

class FramesOnly:
    async def chat_frames(self):
        return [{"kind": "percept", "sha256": "a" * 64, "mime": "image/png"}]


def test_a_stated_juniper_gets_her_normal_turn_with_frames():
    bus = SimpleNamespace(redis=FakeRedis({**SITTING, "player_stated": "juniper"}))
    data = {"text_input": "hi", "source": "x", "surface_context": {"surface": "retro", "retro_sitting_id": SID}}
    out, origin, text, player = asyncio.run(prepare_retro_turn(data, "hi", bus=bus, commentary=FramesOnly()))
    assert origin == "juniper" and player == "juniper" and text == "hi"
    assert "chat_history_source" not in out and "source" not in out
    assert out["attachments"][0]["kind"] == "percept" and out["surface_context"]["player"] == "juniper"


def test_a_guest_or_nobody_stated_is_never_recorded_as_juniper():
    for stated in (None, "guest"):
        bus = SimpleNamespace(redis=FakeRedis({**SITTING, "player_stated": stated}))
        data = {"surface_context": {"surface": "retro"}, "chat_history_source": "hub_orion"}
        out, origin, text, player = asyncio.run(prepare_retro_turn(data, "jump!", bus=bus, commentary=FramesOnly()))
        assert origin is None and player == "guest" and out["chat_history_source"] == "retro_guest"
        assert text.startswith(rc.GUEST_PREFIX) and text.endswith("jump!")


def test_a_flip_mid_sitting_keeps_each_message_source_at_send_time():
    redis = FakeRedis({**SITTING, "player_stated": "guest"})
    bus = SimpleNamespace(redis=redis)
    first = asyncio.run(prepare_retro_turn({"surface_context": {"surface": "retro"}}, "a", bus=bus, commentary=None))
    import json
    redis.data["orion:retro:sitting"] = json.dumps({**SITTING, "player_stated": "juniper"})
    second = asyncio.run(prepare_retro_turn({"surface_context": {"surface": "retro"}}, "b", bus=bus, commentary=None))
    assert first[0]["chat_history_source"] == "retro_guest" and first[1] is None
    assert "chat_history_source" not in second[0] and second[1] == "juniper"


def test_unreadable_sitting_reads_as_a_guest():
    bus = SimpleNamespace()       # no redis attribute
    out, origin, _text, player = asyncio.run(prepare_retro_turn({"surface_context": {"surface": "retro"}}, "x",
                                                                 bus=bus, commentary=None))
    assert origin is None and player == "guest" and out["chat_history_source"] == "retro_guest"


# --- stage seconds -------------------------------------------------------------------------------------------

def test_stage_clock_reads_stance_motor_and_finalize_from_hops():
    times = iter([10.0, 21.0, 22.0])
    clock = rc.StageClock(clock=lambda: next(times))
    clock.note([{"hop": {"stage": "thought_rpc", "status": "started"}}])
    clock.note([{"hop": {"stage": "thought_rpc", "status": "ok"}}])
    clock.note([{"hop": {"stage": "harness_dispatch", "status": "started"}}])
    assert clock.seconds(80.0, 41.0) == {"stance_sec": 11.0, "motor_sec": 41.0, "finalize_sec": 17.0}


# --- the outreach door carries the line's hold, frames and source -------------------------------------------

def test_compose_puts_the_hold_frames_and_source_into_the_turn_payload(monkeypatch):
    seen: dict = {}

    async def fake_turn(**kw):
        seen.update(kw)
        return [{"type": "final", "llm_response": "a line", "harness_fcc_elapsed_sec": 12.5}]

    import orion.hub.turn_orchestrator as to
    monkeypatch.setattr(to, "execute_unified_turn", fake_turn)
    outreach, *_ = make_outreach(monkeypatch)
    outreach._bus = object()
    from orion.schemas.gpu_pool import GpuLeaseRefV1

    hold = GpuLeaseRefV1(lease_id="hold-1", generation=2, role="chat-lane", holder="orion-hub.retro_commentary")

    async def sink(frames):
        return None
    text, debug = asyncio.run(outreach.compose("p", "corr-1", lane="chat", attachments=[{"kind": "percept"}],
                                               gpu_lease=hold, source="retro_commentary", cockpit_sink=sink,
                                               session_id="juniper-main", timeout_sec=99.0))
    assert text == "a line" and debug["harness_fcc_elapsed_sec"] == 12.5
    p = seen["payload"]
    assert p["gpu_lease"]["lease_id"] == "hold-1" and p["source"] == "retro_commentary" and p["no_write"] is True
    assert p["attachments"] == [{"kind": "percept"}] and seen["cockpit_sink"] is sink
    assert seen["session_id"] == "juniper-main"


def test_compose_without_extras_is_the_old_call(monkeypatch):
    seen: dict = {}

    async def fake_turn(**kw):
        seen.update(kw)
        return [{"type": "final", "llm_response": "x"}]

    import orion.hub.turn_orchestrator as to
    monkeypatch.setattr(to, "execute_unified_turn", fake_turn)
    outreach, *_ = make_outreach(monkeypatch)
    outreach._bus = object()
    asyncio.run(outreach.compose("p", "corr-1", lane="chat"))
    assert seen["payload"] == {"no_write": True, "source": "endogenous_outreach"} and "cockpit_sink" not in seen


def test_a_held_compose_refuses_the_agent_first_lane(monkeypatch):
    outreach, *_ = make_outreach(monkeypatch)
    with pytest.raises(ValueError):
        asyncio.run(outreach.compose("p", "c", gpu_lease={"lease_id": "x"}))


def test_the_turn_ingress_carries_the_source_prime_reads():
    from orion.hub.turn_orchestrator import _cockpit_ingress_payload

    line = _cockpit_ingress_payload(user_message="m", session_id="s", payload={"source": "retro_commentary"})
    guest = _cockpit_ingress_payload(user_message="m", session_id="s",
                                     payload={"chat_history_source": "retro_guest",
                                              "surface_context": {"surface": "retro"}})
    assert line["source"] == "retro_commentary" and guest["chat_history_source"] == "retro_guest"
    assert "source" not in _cockpit_ingress_payload(user_message="m", session_id="s", payload={})


def test_deliver_line_refuses_when_the_claim_fails_and_writes_nothing(monkeypatch):
    outreach, queue, history, notified, rows = make_outreach(monkeypatch)

    async def claim():
        return False
    out = asyncio.run(outreach.deliver_line(text="hello", correlation_id="c", source_tag="retro_commentary",
                                            claim=claim))
    assert out["reason"] == "yielded" and drain(queue) == [] and history == [] and notified == []


def test_deliver_line_refuses_a_named_internal_signal_before_claiming(monkeypatch):
    outreach, queue, history, _n, _r = make_outreach(monkeypatch)
    monkeypatch.setattr(eo.EndogenousOutreach, "_named_ungrounded_terms", lambda self, *a: ["prediction_error"])
    claimed: list = []

    async def claim():
        claimed.append(1)
        return True
    out = asyncio.run(outreach.deliver_line(text="my prediction_error is up", correlation_id="c",
                                            source_tag="retro_commentary", claim=claim))
    assert out["reason"] == "named_ungrounded_signal" and claimed == [] and drain(queue) == []
