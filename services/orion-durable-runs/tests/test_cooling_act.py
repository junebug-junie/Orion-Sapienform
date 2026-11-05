"""The cooling world act's loop (ladder retirement PR 5, app/cooling_act.py): the market decides, the pool acts.

Every store, market, bus and hardware-watch read is a fake: nothing here can reach a live GPU pool. The pure
rules are tested in tests/test_cooling_act_rules.py; this file tests the loop's order of operations -- read the
world, gate on eligibility and admission, post the event want, OBEY the decision (funded only), precommit the
episode before the RPC, settle from the pool's record -- and that the flag ships on.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import app.cooling_act as ca
from app.cooling_act import CoolingAct, cabinet_reading_at, sqlalchemy_url
from orion.autonomy import shed_act
from orion.autonomy.self_shed import HardwareWatchView
from orion.autonomy.shed_settlement import SHED_PENDING
from orion.hardware_watch.rules import TempPoint
from orion.schemas.gpu_pool import GpuPoolShedResultV1

NOW = datetime(2026, 11, 5, 18, 0, tzinfo=timezone.utc)


def _points(start_c: float, end_c: float, *, minutes: int = 20, now: datetime = NOW) -> list[TempPoint]:
    n = minutes * 2
    return [TempPoint(now - timedelta(seconds=30 * (n - i)), start_c + (end_c - start_c) * i / n) for i in range(n + 1)]


def _settings(**kw):
    base = dict(cooling_act_tick_sec=60.0, cooling_act_holdback_fraction=0.5, hardware_watch_health_url="http://x",
                shed_rise_threshold_c=0.5, gpu_pool_shed_rpc_timeout_sec=5.0, shed_ttl_sec=900.0, shed_min_gap_sec=900.0,
                shed_max_sec_per_day=3600.0, service_name="orion-durable-runs",
                action_outcome_channel="orion:autonomy:action:outcome")
    base.update(kw)
    return SimpleNamespace(**base)


class _Store:
    def __init__(self, points=None, *, in_flight=(), treated=(), history=None, insert_raises=False):
        self.points = points if points is not None else _points(29.6, 30.6)
        self.in_flight, self.treated = list(in_flight), list(treated)
        self.history = history if history is not None else [(NOW - timedelta(hours=h), 0.5 + 0.02 * (h % 30))
                                                               for h in range(1, 60)]
        self.insert_raises = insert_raises
        self.episodes: dict[str, dict] = {}
        self.settlements: list[dict] = []
        self.calls: list[str] = []
        self.pending: list[dict] = []

    def load_cabinet_points(self, *, since, until):
        return [p for p in self.points if since <= p.ts <= until]

    def load_warming_history(self, *, now, rise_threshold_c):
        return self.history

    def load_background_occupancy(self):
        return 1, 0

    def admission_inputs(self, *, now):
        return self.in_flight, self.treated

    def insert_episode(self, row):
        self.calls.append("insert")
        if self.insert_raises:
            raise RuntimeError("db down")
        if row["episode_id"] in self.episodes:
            return False
        self.episodes[row["episode_id"]] = dict(row)
        return True

    def episode_arm(self, episode_id):
        return (self.episodes.get(episode_id) or {}).get("arm")

    def record_settlement(self, **kw):
        self.calls.append("record_settlement")
        self.settlements.append(kw)

    def load_pending_settlements(self, *, limit=20):
        return self.pending

    def load_effect_posteriors(self):
        return {}

    def latest_surprise(self):
        return 0.2


class _Market:
    def __init__(self, decision=None, *, fail=False):
        self.decision, self.fail = decision, fail
        self.bids: list = []
        self.released: list[str] = []

    async def bid(self, want, now=None, *, reserve_at_open=False):
        self.bids.append(want)
        if self.fail:
            raise RuntimeError("market down")
        return self.decision

    def release(self, want_id):
        self.released.append(want_id)
        return 0


def _dec(outcome="funded", reason="net_positive", net=0.6, retry=None):
    return SimpleNamespace(outcome=outcome, reason=reason, net=net, retry_after_sec=retry, decision_id="dec-1",
                           binding_constraint=None, charge=0.0)


class _Bus:
    def __init__(self, store):
        self.store = store
        self.published: list = []
        self.requests: list = []
        self.codec = SimpleNamespace(decode=lambda raw: raw)

    async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
        self.store.calls.append("rpc")
        self.requests.append((channel, env))
        out = GpuPoolShedResultV1(ok=True, shed_id="oshed_1", dispatch_id=env.payload["dispatch_id"], state="active",
                                  ttl_sec=900.0, started_at=NOW, valid_until=NOW + timedelta(seconds=900))
        return SimpleNamespace(ok=True, envelope=SimpleNamespace(payload=out.model_dump(mode="json")))

    async def publish(self, channel, env):
        self.published.append((channel, env))


def _healthy(now):
    return HardwareWatchView.from_health({"enabled": True, "last_tick_ok": True,
                                          "last_tick_at": (now - timedelta(seconds=15)).isoformat(),
                                          "open_incidents": []}, now=now)


def _act(store=None, market=None, *, bus=True, hw=_healthy, **kw):
    store = store or _Store()
    market = market if market is not None else _Market(_dec())
    b = _Bus(store) if bus else None
    act = CoolingAct(settings=_settings(**kw), store=store, market_getter=lambda: market, bus_getter=lambda: b,
                     hardware_watch=hw, clock=lambda: NOW)
    return act, store, market, b


def _run(coro):
    return asyncio.run(coro)


def test_the_flag_ships_on_and_the_keys_moved_with_their_values():
    from app.settings import Settings

    s = Settings(POSTGRES_URI="postgresql://x:y@h/d")
    assert s.cooling_act_enabled is True and s.cooling_act_holdback_fraction == 0.5
    assert (s.shed_ttl_sec, s.shed_min_gap_sec, s.shed_max_sec_per_day) == (900.0, 900.0, 3600.0)
    assert s.hardware_watch_health_url == "http://hardware-watch:8131/health" and s.shed_rise_threshold_c == 0.5


def test_sqlalchemy_url_names_the_psycopg3_driver():
    assert sqlalchemy_url("postgresql://a:b@h:5432/d") == "postgresql+psycopg://a:b@h:5432/d"
    assert sqlalchemy_url("postgres://a@h/d") == "postgresql+psycopg://a@h/d"
    assert sqlalchemy_url("postgresql+psycopg2://a@h/d") == "postgresql+psycopg://a@h/d"


def test_a_cool_cabinet_posts_nothing():
    act, store, market, _ = _act(_Store(_points(25.0, 25.5)))
    assert _run(act.tick(NOW)) == "ineligible"
    assert market.bids == [] and store.episodes == {}


def test_warm_but_not_warming_posts_nothing():
    act, store, market, _ = _act(_Store(_points(30.5, 30.5)))
    assert _run(act.tick(NOW)) == "not_warming"
    assert market.bids == []


def test_the_reflex_covers_critical_and_orion_posts_nothing():
    act, _, market, _ = _act(_Store(_points(33.6, 34.4)))
    assert _run(act.tick(NOW)) == "ineligible" and market.bids == []


def test_in_flight_and_pool_admission_refusals_post_nothing():
    act, _, market, _ = _act(_Store(in_flight=["cooling_act:old"]))
    assert _run(act.tick(NOW)) == "ineligible" and market.bids == []           # self_shed: winner_loop_in_flight
    recent = [{"decided_at": NOW - timedelta(minutes=20), "settlement_state": "expired", "settlement": {}}]
    act, _, market, _ = _act(_Store(treated=recent))
    assert _run(act.tick(NOW)) == "not_admitted" and act.last["refusal"] == "pool_gap" and market.bids == []


def test_funded_treated_precommits_before_the_one_shed_rpc(monkeypatch):
    monkeypatch.setattr(shed_act, "arm_for", lambda eid, f=0.5: "treated")
    act, store, market, bus = _act()
    assert _run(act.tick(NOW)) == "acted:treated"
    want = market.bids[0]
    assert want.family == "self_regulate.shed" and want.event_ref.startswith("cabinet_reading:athena:")
    assert 0.0 <= want.salience <= 1.0 and want.producer == shed_act.PRODUCER
    eid = shed_act.episode_id_for(want.want_id)
    assert store.calls[:3] == ["insert", "rpc", "record_settlement"]            # precommit BEFORE the RPC
    row = store.episodes[eid]
    assert row["arm"] == "treated" and row["settlement"]["state"] == SHED_PENDING and row["decision_id"] == "dec-1"
    assert row["eligibility"]["eligible"] and row["eligibility"]["want"]["want_id"] == want.want_id
    assert row["expected_effect"]["signal_id"] == "cabinet_heat_pressure"
    channel, env = bus.requests[0]
    assert channel == "orion:gpu_pool:shed:request" and env.payload["dispatch_id"] == eid
    assert env.payload["reason"] == "orion_self_shed" and env.payload["ttl_sec"] == 900.0
    assert store.settlements[0]["state"] == "active" and store.settlements[0]["shed_id"] == "oshed_1"
    assert market.released == [want.want_id] and act._open is None


def test_funded_control_records_the_row_and_sends_nothing(monkeypatch):
    monkeypatch.setattr(shed_act, "arm_for", lambda eid, f=0.5: "control")
    act, store, _, bus = _act()
    assert _run(act.tick(NOW)) == "acted:control"
    (row,) = store.episodes.values()
    assert row["arm"] == "control" and bus.requests == [] and "rpc" not in store.calls


def test_a_deferral_holds_the_same_want_and_bids_again_at_its_retry_time(monkeypatch):
    monkeypatch.setattr(shed_act, "arm_for", lambda eid, f=0.5: "control")
    market = _Market(_dec("deferred", "sated", net=0.0, retry=300))
    act, store, _, _ = _act(market=market)
    assert _run(act.tick(NOW)) == "deferred:sated"
    assert store.episodes == {} and act._open is not None
    assert _run(act.tick(NOW + timedelta(seconds=60))) == "holding" and len(market.bids) == 1
    market.decision = _dec()
    later = NOW + timedelta(seconds=300)
    act._store.points = _points(29.6, 30.6, now=later)
    assert _run(act.tick(later)) == "acted:control"
    assert len(market.bids) == 2 and market.bids[0].want_id == market.bids[1].want_id     # the same event


@pytest.mark.parametrize("market", [_Market(None), _Market(fail=True), _Market(_dec("deferred", "physiology_unavailable"))])
def test_no_answer_never_sheds(market):
    act, store, _, bus = _act(market=market)
    out = _run(act.tick(NOW))
    assert out.startswith("no_answer") and store.episodes == {} and bus.requests == []


def test_no_market_at_all_never_sheds():
    store = _Store()
    act = CoolingAct(settings=_settings(), store=store, market_getter=lambda: None, bus_getter=lambda: None,
                     hardware_watch=_healthy, clock=lambda: NOW)
    assert _run(act.tick(NOW)).startswith("no_answer") and store.episodes == {}


def test_a_held_want_closes_when_its_event_is_over():
    market = _Market(_dec("deferred", "price", net=-0.1, retry=120))
    act, store, _, _ = _act(market=market)
    _run(act.tick(NOW))
    later = NOW + timedelta(seconds=180)
    act._store.points = _points(25.0, 25.0, now=later)                          # the cabinet cooled
    assert _run(act.tick(later)) == "ineligible" and act._open is None and store.episodes == {}


def test_a_failed_precommit_sends_nothing(monkeypatch):
    monkeypatch.setattr(shed_act, "arm_for", lambda eid, f=0.5: "treated")
    act, store, _, bus = _act(_Store(insert_raises=True))
    _run(act.tick(NOW))
    assert bus.requests == [] and "rpc" not in store.calls


def test_unknown_salience_posts_nothing():
    act, _, market, _ = _act(_Store(history=[]))
    assert _run(act.tick(NOW)) == "no_salience" and market.bids == []


def test_settlement_publishes_the_outcome_then_records_the_terminal():
    store = _Store(_points(25.0, 25.0))
    decided = NOW - timedelta(minutes=16)
    store.pending = [{"episode_id": "cooling_act:w", "dispatch_kind": "self_regulate",
                      "settlement": {"state": SHED_PENDING, "decided_at": decided.isoformat(), "ttl_sec": 900.0},
                      "pool_row": {"state": "expired", "shed_id": "oshed_9", "started_at": decided.isoformat(),
                                   "ended_at": (decided + timedelta(seconds=900)).isoformat(), "grants_withheld": 4}}]
    act, _, _, bus = _act(store)
    assert _run(act.reconcile(NOW)) == 1
    (channel, env), = bus.published
    assert channel == "orion:autonomy:action:outcome" and env.payload["action_id"] == "cooling_act:w"
    assert env.payload["kind"] == "self_regulate" and env.payload["success"] is True
    s = store.settlements[0]
    assert s["state"] == "expired" and s["settlement"]["state"] == "settled" and s["shed_id"] == "oshed_9"


def test_a_failed_outcome_publish_leaves_the_episode_pending_for_the_next_tick():
    store = _Store(_points(25.0, 25.0))
    store.pending = [{"episode_id": "cooling_act:w", "dispatch_kind": "self_regulate",
                      "settlement": {"state": SHED_PENDING, "decided_at": (NOW - timedelta(hours=1)).isoformat(),
                                     "ttl_sec": 900.0}, "pool_row": None}]
    act = CoolingAct(settings=_settings(), store=store, market_getter=lambda: None, bus_getter=lambda: None,
                     hardware_watch=_healthy, clock=lambda: NOW)
    assert _run(act.reconcile(NOW)) == 0 and store.settlements == []


def test_reading_at_is_the_newest_readings_time():
    reading = SimpleNamespace(age_sec=12.4)
    assert cabinet_reading_at(reading, NOW) == (NOW - timedelta(seconds=12.4)).replace(microsecond=0)


def test_main_starts_the_loop_only_with_a_market_and_a_bus():
    from pathlib import Path

    src = (Path(ca.__file__).parent / "main.py").read_text()
    assert "cooling_act_enabled" in src and "CoolingAct(" in src and "no_market" in src
    assert ca.BID_TIMEOUT_SEC <= 10.0


def test_a_deferral_clamped_to_the_events_end_never_bids_or_sheds_late(monkeypatch):
    """Review blocker 2026-11-05: deferred with a long retry -> retry_at == end; the next tick lands after it."""
    monkeypatch.setattr(shed_act, "arm_for", lambda eid, f=0.5: "treated")
    market = _Market(_dec("deferred", "price", net=-0.1, retry=3600))
    act, store, _, bus = _act(market=market)
    assert _run(act.tick(NOW)).startswith("deferred")
    end = act._open.end
    assert act._open.retry_at == end
    market.decision = _dec()                                   # would be funded now
    late = end + timedelta(seconds=40)
    act._store.points = _points(29.6, 30.6, now=late)
    assert _run(act.tick(late)) == "event_expired"
    assert len(market.bids) == 1 and store.episodes == {} and bus.requests == []


def test_no_bus_writes_no_row_and_sends_nothing(monkeypatch):
    monkeypatch.setattr(shed_act, "arm_for", lambda eid, f=0.5: "treated")
    act, store, _, _ = _act(bus=False)
    assert _run(act.tick(NOW)) == "acted:no_bus"
    assert store.episodes == {} and "insert" not in store.calls
