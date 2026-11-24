"""Game company patch 4, Hub side: the line's act carries Juniper's engagement, a finished sitting posts its event
want, and the gateway publishes its watch state for Prime's frame.

Not in tests/retro/: it imports scripts.retro_commentary (through it the outreach door and the turn orchestrator)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from orion.retro.sitting import WATCH_KEY, Sitting
from scripts import act_emit, retro_sitting

NOW = 1_800_000_000.0


class Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append(envelope.payload)


def _published(juniper: bool):
    bus = Bus()
    ok = asyncio.run(act_emit.publish_line_said(bus, line_id="11111111-2222-3333-4444-555555555555", sitting_id="S1",
                                                text="Nice jump.", player_stated="juniper" if juniper else None))
    return ok, [(p["event_kind"], p["act"]) for p in bus.published]


def test_a_line_said_to_juniper_opens_a_juniper_replied_act_that_the_resolver_grades():
    ok, events = _published(True)
    assert ok and [k for k, _ in events] == ["act_opened"]
    act = events[0][1]
    assert act["actor"] == "hub.retro_commentary" and act["verb"] == "say"
    assert act["expect"]["signal_id"] == "juniper_replied" and act["expect"]["horizon_sec"] == 300.0
    assert act["test_before"] == "retro_line:11111111-2222-3333-4444-555555555555"


@pytest.mark.parametrize("stated", [None, "guest"])
def test_a_line_said_to_a_guest_or_nobody_keeps_the_produced_act_and_is_no_engagement_record(stated):
    bus = Bus()
    asyncio.run(act_emit.publish_line_said(bus, line_id="11111111-2222-3333-4444-555555555555", sitting_id="S1",
                                           text="Nice jump.", player_stated=stated))
    kinds = [(p["event_kind"], p["act"]["expect"]["signal_id"]) for p in bus.published]
    assert kinds == [("act_opened", "artifact_produced"), ("act_closed", "artifact_produced")]


class DB:
    def __init__(self, minutes):
        self.minutes, self.calls = list(minutes), []

    async def __call__(self, method, sql, *values):
        self.calls.append((method, sql, values))
        if sql == retro_sitting.EVENT_HISTORY_SQL:
            return [{"minutes": m} for m in self.minutes]
        if method == "fetchval":
            return True
        return None


def _svc(minutes, *, enabled=True, decision=None):
    bids = []

    async def bid(want):
        bids.append(want)
        return decision or SimpleNamespace(outcome="funded", reason="net_positive")

    cfg = SimpleNamespace(HUB_RETRO_SITTING_ENABLED=True, HUB_RETRO_SITTING_GAP_SEC=300, HUB_RETRO_SITTING_MIN_SEC=120,
                          HUB_RETRO_SITTING_EVENT_WANT_ENABLED=enabled)
    svc = retro_sitting.SittingService(cfg, clock=lambda: NOW, bid=bid)
    db = DB(minutes)
    svc._db = db
    return svc, db, bids


def _sitting(minutes, *, backfilled=False, player="juniper"):
    return Sitting(sitting_id="S1", retro_session_id="r1", started_at=NOW - minutes * 60, last_input_at=NOW,
                   ended_at=NOW, end_reason="idle", game_id="crystalis", game_title="Crystalis",
                   player_stated=player, backfilled=backfilled)


def test_a_finished_sitting_posts_one_event_want_salient_against_its_last_30_days():
    svc, db, bids = _svc([10.0] * 15 + [90.0] * 10)
    assert asyncio.run(svc._post_event_want(_sitting(45.0))) == "funded"
    [w] = bids
    assert (w.family, w.verb, w.tier, w.event_ref) == ("hub.retro_sitting", "hold", 2, "retro_sitting:S1")
    assert w.producer == "orion-hub.retro_sitting.event_want" and w.salience == pytest.approx(15 / 25)
    assert w.object_ref == "retro_game:crystalis" and w.lanes == []
    # The want id is the event's: a re-post of the same close is the same want.
    svc2, _, bids2 = _svc([10.0] * 25)
    asyncio.run(svc2._post_event_want(_sitting(45.0)))
    assert bids2[0].want_id == w.want_id
    # The history read excludes the sitting itself and ends at its close.
    (_, sql, values), = [c for c in db.calls if c[1] == retro_sitting.EVENT_HISTORY_SQL]
    assert values[0] == "S1" and values[2] > values[1]


def test_no_want_without_twenty_past_sittings_for_a_backfilled_row_or_when_off():
    svc, _, bids = _svc([10.0] * 19)
    assert asyncio.run(svc._post_event_want(_sitting(45.0))) == "no_history" and bids == []
    svc, _, bids = _svc([10.0] * 30)
    assert asyncio.run(svc._post_event_want(_sitting(45.0, backfilled=True))) == "off" and bids == []
    svc, _, bids = _svc([10.0] * 30, enabled=False)
    assert asyncio.run(svc._post_event_want(_sitting(45.0))) == "off" and bids == []


def test_a_failing_history_read_or_bid_never_stops_a_close():
    svc, db, bids = _svc([10.0] * 30)

    async def boom(*a):
        raise RuntimeError("db down")

    svc._db = boom
    assert asyncio.run(svc._post_event_want(_sitting(45.0))) == "failed" and bids == []


def test_the_close_posts_the_want_only_for_a_sitting_worth_recording():
    svc, db, bids = _svc([10.0] * 30)

    async def noop(*a, **k):
        return None

    svc._resolve = noop
    asyncio.run(svc._finish(_sitting(1.0)))
    assert bids == []
    asyncio.run(svc._finish(_sitting(30.0)))
    assert len(bids) == 1


def test_the_settings_flag_ships_on():
    from app.settings import Settings

    assert Settings().HUB_RETRO_SITTING_EVENT_WANT_ENABLED is True


def test_the_gateway_publishes_its_watch_state_for_prime():
    from test_retro_watch_service import Rig, run, settle

    class Redis:
        def __init__(self):
            self.kv, self.ttl = {}, {}

        async def set(self, key, value, ex=None):
            self.kv[key], self.ttl[key] = value, ex

    async def go():
        rig = Rig()
        redis = Redis()
        rig.sitting.redis_provider = lambda: redis
        rig.press()
        for _ in range(20):
            await rig.step(press=True)
        await rig.watch.publish_watch()
        await settle(rig.watch)
        snap = json.loads(redis.kv[WATCH_KEY])
        assert snap["sitting_id"] == rig.watch.state.sitting_id and snap["mode"] == "on"
        assert snap["counts"]["notes_stored"] == 1 and snap["last_note"].startswith("Link walks north")
        assert snap["status"] == "watching" and 0 < redis.ttl[WATCH_KEY] <= 120
        # No sitting: nothing is written (the key expires on its own).
        rig.sitting.tracker.current = None
        redis.kv.clear()
        await rig.watch.publish_watch()
        assert redis.kv == {}

    run(go())
