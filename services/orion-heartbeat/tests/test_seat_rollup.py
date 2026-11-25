"""Five-minute per-seat rollup (issue #3886): the accumulator, the service path, and the replay's
read path. The live exact-match check against grammar_events is evals/seat_rollup_match.py."""
from __future__ import annotations

import asyncio
import importlib.util
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from orion.schemas.heartbeat_seat_rollup import (
    HEARTBEAT_SEAT_ROLLUP_CHANNEL,
    HEARTBEAT_SEAT_ROLLUP_KIND,
    HeartbeatSeatRollupV1,
)

from app.seat_rollup import SeatRollupAccumulator, parse_emitted_epoch
from app.substrate import routing
from app.substrate.routing import SEAT_ORDER

REPO = Path(__file__).resolve().parents[3]
T0 = 1_764_000_000  # a 5-minute boundary (divisible by 300)
assert T0 % 300 == 0


def _acc(started: float = T0, **kw) -> SeatRollupAccumulator:
    return SeatRollupAccumulator(node="athena", seats=SEAT_ORDER, started_at_epoch=started, **kw)


def _by_seat(events: list[HeartbeatSeatRollupV1]) -> dict[tuple[int, str], HeartbeatSeatRollupV1]:
    return {(int(e.bucket_start.timestamp()), e.seat): e for e in events}


def _listen(acc: SeatRollupAccumulator, start: float, end: float, step: float = 15.0) -> None:
    """The channel talking every ``step`` seconds from start to end."""
    t = start
    while t <= end:
        acc.observe_receipt(t)
        t += step


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


# --- accumulator -------------------------------------------------------------------------------


def test_buckets_by_emitted_at_and_writes_every_seat() -> None:
    acc = _acc()
    _listen(acc, T0 + 10, T0 + 330)
    acc.observe_atom(site_index=0, level=0.5, emitted_epoch=T0 + 1, now_epoch=T0 + 10, event_id="a")
    acc.observe_atom(site_index=0, level=0.25, emitted_epoch=T0 + 299.999, now_epoch=T0 + 301, event_id="b")
    acc.observe_atom(site_index=0, level=0.5, emitted_epoch=T0 + 300, now_epoch=T0 + 301, event_id="c")
    acc.observe_skip(site_index=1, emitted_epoch=T0 + 5, now_epoch=T0 + 10, event_id="d")
    assert acc.flush_closed(T0 + 300 + 29, grace_sec=30) == []  # inside the grace
    events = acc.flush_closed(T0 + 330, grace_sec=30)
    assert [e.seat for e in events] == list(SEAT_ORDER)  # one row per seat, quiet seats included
    rows = _by_seat(events)
    hub = rows[(T0, "orion-hub")]
    assert (hub.level_sum, hub.atom_count, hub.skipped_count, hub.partial) == (0.75, 2, 0, False)
    assert hub.rollup_id == f"athena:orion-hub:{_iso(T0)}"
    assert hub.bucket_end.timestamp() == T0 + 300
    bio = rows[(T0, "orion-biometrics")]
    assert (bio.level_sum, bio.atom_count, bio.skipped_count) == (0.0, 0, 1)
    # A quiet seat in a complete bucket is a real zero row, not a missing one.
    assert rows[(T0, "orion-bus")].atom_count == 0 and rows[(T0, "orion-bus")].partial is False
    # The atom stamped exactly on the boundary is in the next bucket, still open.
    nxt = acc.flush_closed(T0 + 630, grace_sec=30)
    assert _by_seat(nxt)[(T0 + 300, "orion-hub")].level_sum == 0.5


def test_sum_does_not_depend_on_arrival_order() -> None:
    rng = random.Random(7)
    levels = [rng.random() * 10 ** rng.randint(-8, 0) for _ in range(500)]
    sums = set()
    for _ in range(5):
        rng.shuffle(levels)
        acc = _acc()
        for i, lv in enumerate(levels):
            acc.observe_atom(site_index=2, level=lv, emitted_epoch=T0 + 1, now_epoch=T0 + 1, event_id=None)
        sums.add(_by_seat(acc.flush_closed(T0 + 400))[(T0, "orion-cortex-exec")].level_sum)
    assert sums == {math.fsum(levels)}


def test_start_mid_bucket_is_partial_and_earlier_atoms_are_late() -> None:
    acc = _acc(started=T0 + 120)
    acc.observe_receipt(T0 + 121)
    assert acc.observe_atom(site_index=0, level=0.5, emitted_epoch=T0 - 1, now_epoch=T0 + 121) == "late"
    acc.observe_atom(site_index=0, level=0.5, emitted_epoch=T0 + 121, now_epoch=T0 + 121)
    for t in range(150, 601, 30):  # the channel keeps talking
        acc.observe_receipt(T0 + t)
    events = acc.flush_closed(T0 + 900)
    first = [e for e in events if e.bucket_start.timestamp() == T0]
    second = [e for e in events if e.bucket_start.timestamp() == T0 + 300]
    assert len(first) == 5 and all(e.partial for e in first)
    assert len(second) == 5 and not any(e.partial for e in second)
    assert acc.late == 1


def test_silence_gap_marks_spanned_buckets_partial_and_blind_buckets_have_no_rows() -> None:
    acc = _acc(silence_gap_sec=60)
    acc.observe_receipt(T0 + 10)
    acc.observe_receipt(T0 + 250)  # 240 s silence inside bucket 0
    acc.observe_receipt(T0 + 310)
    acc.observe_receipt(T0 + 1210)  # 900 s silence: buckets 1..4 blind
    acc.observe_receipt(T0 + 1220)
    starts = {}
    for e in acc.flush_closed(T0 + 1800):
        starts.setdefault(int(e.bucket_start.timestamp()) - T0, set()).add(e.partial)
    assert starts == {0: {True}, 300: {True}, 1200: {True}}  # 600, 900: no rows at all


def test_bucket_flushed_during_an_ongoing_silence_is_partial() -> None:
    acc = _acc(silence_gap_sec=60)
    for t in range(10, 300, 20):
        acc.observe_receipt(T0 + t)
    acc.observe_receipt(T0 + 310)  # then the subscription goes quiet with no error
    rows = acc.flush_closed(T0 + 1000)
    assert {int(e.bucket_start.timestamp()) - T0: e.partial for e in rows} == {0: False, 300: True}


def test_partial_marks_reach_a_bucket_that_only_gets_late_stamped_atoms() -> None:
    """The loop stalls: messages from inside the blind span arrive afterwards. The bucket
    they open must still read partial."""
    acc = _acc(silence_gap_sec=60)
    acc.observe_receipt(T0 + 10)
    acc.observe_receipt(T0 + 700)
    acc.observe_atom(site_index=3, level=0.1, emitted_epoch=T0 + 400, now_epoch=T0 + 700)
    rows = _by_seat(acc.flush_closed(T0 + 1200))
    assert rows[(T0 + 300, "orion-bus")].partial is True


def test_duplicate_event_ids_count_once_and_future_atoms_are_dropped() -> None:
    acc = _acc()
    assert acc.observe_atom(site_index=0, level=0.5, emitted_epoch=T0 + 1, now_epoch=T0 + 2, event_id="x") == "counted"
    assert acc.observe_atom(site_index=0, level=0.5, emitted_epoch=T0 + 1, now_epoch=T0 + 2, event_id="x") == "duplicate"
    assert acc.observe_atom(site_index=0, level=0.5, emitted_epoch=T0 + 900, now_epoch=T0 + 2) == "future"
    # One bucket ahead is tolerated (small clock skew between hosts).
    assert acc.observe_atom(site_index=0, level=0.5, emitted_epoch=T0 + 301, now_epoch=T0 + 2) == "counted"
    rows = _by_seat(acc.flush_closed(T0 + 900))
    assert rows[(T0, "orion-hub")].atom_count == 1
    assert (acc.duplicates, acc.future) == (1, 1)


def test_published_bucket_is_closed_for_late_atoms() -> None:
    acc = _acc()
    acc.observe_atom(site_index=0, level=0.5, emitted_epoch=T0 + 1, now_epoch=T0 + 1)
    acc.flush_closed(T0 + 330, grace_sec=30)
    assert acc.observe_atom(site_index=0, level=0.5, emitted_epoch=T0 + 299, now_epoch=T0 + 331) == "late"
    assert acc.flush_closed(T0 + 2000) == []


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2026-11-25T00:00:00Z", 1795564800.0),
        ("2026-11-25T00:00:00+00:00", 1795564800.0),
        ("2026-11-25T00:00:00", 1795564800.0),
        ("not a time", None),
        (None, None),
        (17, None),
    ],
)
def test_parse_emitted_epoch(value, expected) -> None:
    assert parse_emitted_epoch(value) == expected


# --- service path ------------------------------------------------------------------------------


def _payload(source: str, role: str, emitted: float, event_id: str, **fields) -> dict:
    return {
        "event_kind": "atom_emitted",
        "event_id": event_id,
        "emitted_at": _iso(emitted),
        "provenance": {"source_service": source},
        "atom": {"atom_type": "signal", "semantic_role": role, **fields},
    }


@pytest.fixture
def svc(monkeypatch):
    from app import service as service_mod

    clock = {"now": float(T0 + 1)}
    monkeypatch.setattr(service_mod.time, "time", lambda: clock["now"])
    s = service_mod.HeartbeatService()
    s.seat_rollup = SeatRollupAccumulator(node="athena", seats=SEAT_ORDER, started_at_epoch=float(T0))
    s._clock = clock
    return s


def test_service_routes_into_the_rollup(svc) -> None:
    payloads = [
        _payload("orion-hub", "repair_pressure_reading", T0 + 1, "e1", salience=0.3),
        _payload("orion-hub", "session_context", T0 + 2, "e2"),  # skip rule
        _payload("orion-cortex-orch", "route_arbitration_decided", T0 + 3, "e3"),  # presence, seat 2
        _payload("orion-biometrics", "gpu_pressure_signal", T0 + 4, "e4", salience=0.2),
        _payload("orion-biometrics", "telemetry_sample", T0 + 5, "e5"),  # clockwork skip
        _payload("orion-heartbeat", "summary", T0 + 6, "e6"),  # self: never counted
        _payload("orion-gpu-pool", "x", T0 + 7, "e7"),  # unrouted: never counted
        _payload("orion-hub", "repair_pressure_reading", T0 + 8, "e1", salience=0.3),  # duplicate
    ]

    async def run() -> None:
        for p in payloads:
            await svc._handle_grammar_message(p)

    asyncio.run(run())
    rows = _by_seat(svc.seat_rollup.flush_closed(T0 + 900))
    assert (rows[(T0, "orion-hub")].level_sum, rows[(T0, "orion-hub")].atom_count) == (0.3, 1)
    assert rows[(T0, "orion-hub")].skipped_count == 1
    assert (rows[(T0, "orion-cortex-exec")].level_sum, rows[(T0, "orion-cortex-exec")].atom_count) == (0.5, 1)
    bio = rows[(T0, "orion-biometrics")]
    assert (bio.level_sum, bio.atom_count, bio.skipped_count) == (0.2, 1, 1)
    assert sum(r.atom_count for r in rows.values()) == 3
    assert svc._seat_rollup_stats()["duplicate_atoms_ignored"] == 1


def test_service_counts_atoms_even_when_the_absorb_queue_is_full(svc) -> None:
    import asyncio as _a

    svc._absorb_queue = _a.Queue(maxsize=1)
    for i in range(3):
        asyncio.run(svc._handle_grammar_message(_payload("orion-hub", "user_utterance", T0 + i, f"q{i}")))
    assert svc.events_dropped_queue_full == 2
    rows = _by_seat(svc.seat_rollup.flush_closed(T0 + 900))
    assert rows[(T0, "orion-hub")].atom_count == 3


def test_service_without_emitted_at_falls_back_to_receipt_time(svc) -> None:
    p = _payload("orion-hub", "user_utterance", T0, "n1")
    p.pop("emitted_at")
    svc._clock["now"] = float(T0 + 42)
    asyncio.run(svc._handle_grammar_message(p))
    assert svc.seat_rollup_emitted_at_fallback == 1
    assert _by_seat(svc.seat_rollup.flush_closed(T0 + 900))[(T0, "orion-hub")].atom_count == 1


class _FakeBus:
    def __init__(self, fail: int = 0) -> None:
        self.sent: list[tuple[str, object]] = []
        self.fail = fail

    async def publish(self, channel, envelope) -> None:
        if self.fail:
            self.fail -= 1
            raise ConnectionError("bus down")
        self.sent.append((channel, envelope))


def test_flush_publishes_one_envelope_per_seat_and_retries_after_failure(svc) -> None:
    svc.bus = _FakeBus(fail=1)
    asyncio.run(svc._handle_grammar_message(_payload("orion-hub", "user_utterance", T0 + 1, "p1")))
    assert asyncio.run(svc._flush_seat_rollup_once(T0 + 400)) == 0
    assert svc.seat_rollup_publish_failed == 1 and len(svc._seat_rollup_pending) == 5
    assert asyncio.run(svc._flush_seat_rollup_once(T0 + 410)) == 5
    assert len(svc._seat_rollup_pending) == 0
    channels = {c for c, _ in svc.bus.sent}
    assert channels == {HEARTBEAT_SEAT_ROLLUP_CHANNEL}
    env = svc.bus.sent[0][1]
    assert env.kind == HEARTBEAT_SEAT_ROLLUP_KIND
    back = HeartbeatSeatRollupV1.model_validate(env.payload)
    assert back.seat == "orion-hub" and back.atom_count == 1
    assert svc._seat_rollup_stats()["published"] == 5


def test_rollup_disabled_leaves_service_unchanged(monkeypatch) -> None:
    from app import service as service_mod

    monkeypatch.setattr(service_mod.settings, "seat_rollup_enabled", False)
    s = service_mod.HeartbeatService()
    assert s.seat_rollup is None
    asyncio.run(s._handle_grammar_message(_payload("orion-hub", "user_utterance", T0, "d1")))
    assert s.events_queued == 1
    assert asyncio.run(s._flush_seat_rollup_once(T0 + 900)) == 0
    assert s._seat_rollup_stats()["enabled"] is False


# --- replay read path --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def replay():
    spec = importlib.util.spec_from_file_location("demon_network_replay_t", REPO / "scripts" / "demon_network_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_replay_aggregate_equals_service_rollup_on_the_same_atoms(replay, svc) -> None:
    """Same atoms, two paths: ledger-row tuples through the replay, bus payloads through the
    service. Every (bucket, seat) cell must be equal with ==, measured levels included."""
    rng = random.Random(3886)
    specs = [
        ("orion-hub", "repair_pressure_reading", {"salience": None}),
        ("orion-hub", "user_utterance", {}),
        ("orion-biometrics", "gpu_pressure_signal", {"salience": None}),
        ("orion-biometrics", "stability_signal", {"salience": None}),
        ("orion-bus", "bus_activity_zscore_computed", {"confidence": 0.9, "uncertainty": None}),
        ("orion-cortex-exec", "exec_step_started", {}),
        ("orion-harness-governor", "harness_fcc_step", {}),
    ]
    payloads, ledger_rows = [], []
    for i in range(600):
        svc_name, role, fields = rng.choice(specs)
        f = {k: (round(rng.random(), rng.randint(1, 9)) if v is None else v) for k, v in fields.items()}
        emitted = T0 + rng.randint(0, 1_500_000) / 1000  # ms: survives the ISO round trip
        payloads.append(_payload(svc_name, role, emitted, f"ev{i}", **f))
        ledger_rows.append(
            (str(int(emitted // 300)), svc_name, role, "signal",
             str(f.get("salience", "")), str(f.get("confidence", "")), str(f.get("uncertainty", "")))
        )
    rng.shuffle(payloads)
    svc._clock["now"] = float(T0 + 1500)

    async def run() -> None:
        for p in payloads:
            await svc._handle_grammar_message(p)

    asyncio.run(run())
    got = {
        (int(e.bucket_start.timestamp()) // 300, e.site_index): (e.level_sum, e.atom_count)
        for e in svc.seat_rollup.flush_closed(T0 + 3000)
        if e.atom_count
    }
    want, _stats = replay.aggregate_grammar_rows(ledger_rows)
    assert got == want
    assert len(want) >= 20


def _rows(b, levels, *, partial=False, node="athena", seats=range(5)):
    return [(str(b), node, str(s), str(levels[s]), "3", "t" if partial else "f") for s in seats]


def test_build_series_prefers_rollup_and_labels_sources(replay) -> None:
    b = T0 // 300
    grammar = {(b + t, s): (1.0, 2) for t in range(0, 8) for s in range(5)}
    rollup = (
        _rows(b - 2, [9.0] * 5)  # rollup history older than the ledger
        + _rows(b + 2, [7.0] * 5)  # overlaps the ledger: rollup wins
        + _rows(b + 3, [7.0] * 5, partial=True)  # partial: missing, no refill
        + _rows(b + 4, [7.0] * 5, seats=range(4))  # a seat row missing: partial
    )
    level, count, idx, source, stats = replay.build_series(grammar, rollup)
    assert idx[0] == b - 2 and idx[-1] == b + 7
    lab = dict(zip(idx.tolist(), source.tolist()))
    assert lab[b - 2] == "rollup" and lab[b - 1] == "none"
    assert lab[b] == "grammar_events" and lab[b + 1] == "grammar_events"
    assert lab[b + 2] == "rollup" and lab[b + 3] == "rollup_partial" and lab[b + 4] == "rollup_partial"
    t = {int(v): i for i, v in enumerate(idx)}
    assert level[t[b - 2]].tolist() == [9.0] * 5
    assert np.isnan(level[t[b - 1]]).all()  # no source: missing, not zero
    assert np.isnan(level[t[b]]).all()  # first ledger bucket: partial
    assert level[t[b + 1]].tolist() == [1.0] * 5
    assert level[t[b + 2]].tolist() == [7.0] * 5
    assert np.isnan(level[t[b + 3]]).all() and np.isnan(level[t[b + 4]]).all()
    assert np.isnan(level[t[b + 7]]).all()  # last ledger bucket: still filling
    assert stats["source_buckets"] == {"rollup": 2, "rollup_partial": 2, "grammar_events": 5, "none": 1}
    assert [s["source"] for s in stats["source_spans"]][:3] == ["rollup", "none", "grammar_events"]


def test_build_series_marks_clockwork_outage_missing_for_rollup_rows(replay) -> None:
    b = T0 // 300
    rows = [(str(b), "athena", str(s), "1.0", "0" if s == 3 else "4", "f") for s in range(5)]
    level, _count, _idx, source, stats = replay.build_series({}, rows)
    assert source.tolist() == ["rollup"] and np.isnan(level).all() and stats["outage_buckets"] == 1


def test_build_series_refuses_rows_from_two_nodes(replay) -> None:
    b = T0 // 300
    with pytest.raises(SystemExit, match="more than one node"):
        replay.build_series({}, _rows(b, [1.0] * 5) + _rows(b, [1.0] * 5, node="circe"))


def test_routing_seat_order_matches_schema_site_index() -> None:
    assert dict(enumerate(SEAT_ORDER)) == routing.SITE_ORGAN_MAP
