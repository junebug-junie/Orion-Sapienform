"""Temporal Self rev 4 patch 7: dream replay draws candidates from closed arcs.

Pinned: arc -> candidate mapping, window filter, closed-only, sleep excluded, cap,
read failure -> sleep unchanged, and sleep pressure / rest drive unchanged with arcs on.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone

import pytest

NOW = datetime(2026, 10, 12, 13, 0, tzinfo=timezone.utc)
SINCE = NOW - timedelta(hours=9)


def _arc(arc_id, kind="concern", *, ended=None, status="closed", dwell=1800.0, returns=2, label="gpu lane jam"):
    return {
        "arc_id": arc_id, "kind": kind, "subject_ref": f"ref:{arc_id}", "subject_label": label,
        "began_at": (ended or NOW - timedelta(hours=1)) - timedelta(minutes=30),
        "ended_at": ended if ended is not None or status != "closed" else NOW - timedelta(hours=1),
        "status": status, "attention_returns": returns, "cumulative_dwell_sec": dwell,
    }


# --- mapping -----------------------------------------------------------------


def test_arc_maps_to_a_citable_candidate_with_declared_weight():
    import math

    from app.replay import ARC_WEIGHT_MAX, candidate_from_arc

    c = candidate_from_arc(_arc("a1b2c3", dwell=1800.0, returns=2))
    assert c is not None
    assert c.ref_id == "arc:a1b2c3" and c.source_kind == "arc" and c.tags == ["concern"]
    expected = ARC_WEIGHT_MAX * (math.log1p(30) / math.log1p(60) + 2 / 3) / 2
    assert c.weight == pytest.approx(expected, abs=1e-4)
    assert "gpu lane jam" in c.text and "held 30 min" in c.text and "returned to 2 times" in c.text
    assert c.reason


def test_arc_weight_saturates_below_a_critical_metacog_and_zero_is_dropped():
    from app.replay import ARC_WEIGHT_MAX, arc_weight, candidate_from_arc

    assert arc_weight(74106, 53) == ARC_WEIGHT_MAX < 0.8  # a 20 h curiosity, 53 returns; under a 0.8 crystallization
    assert arc_weight(3600, 3) == ARC_WEIGHT_MAX
    assert arc_weight(0, 0) == 0.0
    assert arc_weight("junk", None) == 0.0
    assert candidate_from_arc(_arc("z", dwell=0.0, returns=0)) is None
    # a concern with no dwell but returns still carries weight
    assert candidate_from_arc(_arc("c", dwell=0.0, returns=6)).weight == pytest.approx(ARC_WEIGHT_MAX / 2)


def test_label_falls_back_to_subject_ref():
    from app.replay import candidate_from_arc

    c = candidate_from_arc(_arc("q", label=""))
    assert "ref:q" in c.text


# --- filters -----------------------------------------------------------------


def test_closed_only_and_sleep_and_strain_excluded():
    from app.replay import arc_candidates

    rows = [
        _arc("closed1"),
        _arc("open1", status="open"),
        _arc("susp1", status="suspended"),
        _arc("sleep1", kind="sleep"),
        _arc("strain1", kind="strain"),  # a body state lasting days would saturate every replay
        {**_arc("closed_no_end"), "ended_at": None},
    ]
    assert [c.ref_id for c in arc_candidates(rows, SINCE, NOW)] == ["arc:closed1"]


def test_window_filter_is_ended_in_since_until():
    from app.replay import arc_candidates

    rows = [
        _arc("before", ended=SINCE - timedelta(seconds=1)),
        _arc("at_since", ended=SINCE),
        _arc("inside", ended=NOW - timedelta(minutes=5)),
        _arc("at_until", ended=NOW),
        _arc("after", ended=NOW + timedelta(minutes=1)),
    ]
    got = {c.ref_id for c in arc_candidates(rows, SINCE, NOW)}
    assert got == {"arc:at_since", "arc:inside"}


def test_cap_keeps_the_heaviest_and_dedupes_arc_ids():
    from app.replay import ARC_CANDIDATES_MAX, arc_candidates

    rows = [_arc(f"a{i:03d}", dwell=60.0 * (i + 1), returns=0, ended=NOW - timedelta(minutes=i + 1)) for i in range(50)]
    rows.append(dict(rows[-1]))  # duplicate row of the heaviest arc
    got = arc_candidates(rows, SINCE, NOW)
    assert len(got) == ARC_CANDIDATES_MAX == 20
    assert len({c.ref_id for c in got}) == 20
    assert got[0].ref_id == "arc:a049"  # longest dwell first
    weights = [c.weight for c in got]
    assert weights == sorted(weights, reverse=True)
    assert arc_candidates(rows, SINCE, NOW, cap=3) == got[:3]


def test_store_sql_filters_closed_non_sleep_window_and_is_read_only():
    from app.cycle_store import ARC_ROWS_SQL, CYCLE_WRITE_TABLES

    sql = " ".join(ARC_ROWS_SQL.split())
    assert "status = 'closed'" in sql and "kind <> 'sleep'" in sql
    assert "ended_at >= :since AND ended_at < :until" in sql and "LIMIT :limit" in sql
    assert "temporal_self_arc" not in CYCLE_WRITE_TABLES
    assert not any(w in sql.upper() for w in ("INSERT", "UPDATE", "DELETE"))


def test_store_read_failure_is_empty(monkeypatch):
    from app import cycle_store

    def boom():
        raise RuntimeError("relation temporal_self_arc does not exist")

    monkeypatch.setattr(cycle_store, "_get_engine", boom)
    assert cycle_store.load_arc_rows(SINCE, NOW) == []


def test_flag_ships_on():
    from app.settings import Settings

    assert Settings.model_fields["DREAM_REPLAY_ARC_CANDIDATES_ENABLED"].default is True


# --- cycle -------------------------------------------------------------------


def _rows():
    return {
        "metacog": [
            {"id": "m1", "summary": "Recall returned empty for three turns in a row", "severity": "critical",
             "trigger_kind": "recall_empty", "tags": ["recall"]},
            {"id": "m2", "summary": "Latency spike on the chat lane", "severity": "degraded",
             "trigger_kind": "latency", "tags": ["latency"]},
        ],
        "compaction_request": [{"request_id": "r1", "theme": "juniper's sleep schedule", "reason": "recurs"}],
        "resonance": [{"alert_id": "a1", "theme_key": "being watched", "violation_count": 4}],
        "crystallization": [{"crystallization_id": "c1", "subject": "Juniper prefers plain English",
                             "summary": "dense answers get re-asked", "salience": 0.8, "tags": ["style"]}],
    }


def _live_arcs(now):
    return [_arc(f"arc{i}", kind=k, dwell=3600.0, returns=3, ended=now - timedelta(minutes=10 + i))
            for i, k in enumerate(["concern", "curiosity", "attention", "interoception", "reverie",
                                   "imagery", "conversation", "concern", "curiosity", "attention"])]


class _Fakes:
    def __init__(self, rows, arcs=None, arc_loader=None):
        self.rows, self.arcs, self.arc_loader = rows, arcs, arc_loader
        self.persisted, self.readings, self.arc_reads = [], [], []

    def deps(self, *, with_arcs):
        from app.cycle import CycleDeps

        def load(since, limit, until=None):
            return {} if until is not None else self.rows

        def load_arcs(since, until):
            self.arc_reads.append((since, until))
            return self.arcs(until) if callable(self.arcs) else (self.arcs or [])

        async def complete(_prompt):
            return json.dumps({"link": True, "claim": "These two recur together more often than chance allows",
                               "why": "w"})

        async def publish(reading):
            self.readings.append(reading)

        loader = None
        if with_arcs:
            loader = self.arc_loader or load_arcs
        return CycleDeps(
            load_source_rows=load, load_idle_minutes=lambda: 120.0,
            load_last_window_start=lambda: None, load_last_attempt_end=lambda: None,
            persist_cycle=lambda c: self.persisted.append(c) or True, complete=complete,
            publish_drive_reading=publish, load_arc_rows=loader,
        )




def test_pressure_and_rest_drive_unchanged_with_arcs_on():
    from app.cycle import run_cycle_once

    off, on = _Fakes(_rows()), _Fakes(_rows(), arcs=_live_arcs)
    c_off = asyncio.run(run_cycle_once(off.deps(with_arcs=False)))
    c_on = asyncio.run(run_cycle_once(on.deps(with_arcs=True)))
    assert c_off.status == c_on.status == "completed"
    assert c_on.pressure.pressure == c_off.pressure.pressure == pytest.approx(3.6)
    assert c_on.pressure.counts == c_off.pressure.counts and "arc" not in c_on.pressure.counts
    assert c_on.pressure.new_counts == c_off.pressure.new_counts
    # pre-sleep and post-sleep rest drive readings: same state, level, reasons
    assert len(on.readings) == len(off.readings) == 2
    for a, b in zip(on.readings, off.readings):
        assert (a.state, a.level, a.due_reason) == (b.state, b.level, b.due_reason)
    # and the arcs actually reached replay, read over the cycle's own window
    assert any(r.source_kind == "arc" for r in c_on.replay)
    assert not any(r.source_kind == "arc" for r in c_off.replay)
    (since, until), = on.arc_reads
    assert since == c_on.pressure.since and until == c_on.started_at
    assert "arc candidates=10" in c_on.note


def test_arcs_take_at_most_a_third_and_critical_metacog_leads():
    from app.cycle import run_cycle_once
    from app.settings import settings

    on = _Fakes(_rows(), arcs=lambda now: [
        _arc(f"x{i}", dwell=90000.0, returns=50, ended=now - timedelta(minutes=i + 1)) for i in range(40)])
    c = asyncio.run(run_cycle_once(on.deps(with_arcs=True)))
    k = settings.DREAM_REPLAY_MAX
    arcs = [r for r in c.replay if r.source_kind == "arc"]
    rest = [r for r in c.replay if r.source_kind != "arc"]
    assert len(arcs) == k // 3 == 4  # a third of the replay, not the plain ceil(k/2) cap
    assert len(rest) == 5  # every existing candidate still replayed
    assert c.replay[0].ref_id == "metacog:m1"  # critical metacog outranks every arc


def test_arcs_alone_never_start_a_sleep():
    from app.cycle import run_cycle_once

    f = _Fakes({k: [] for k in _rows()}, arcs=_live_arcs)
    c = asyncio.run(run_cycle_once(f.deps(with_arcs=True), trigger="manual", force=True))
    assert c.status == "empty" and c.replay == [] and f.arc_reads == []


@pytest.mark.parametrize("failure", ["raise", "garbage"])
def test_arc_read_failure_leaves_the_sleep_as_before(failure):
    from app.cycle import run_cycle_once

    def bad(since, until):
        if failure == "raise":
            raise RuntimeError("db down")
        return [{"arc_id": None}, "not a row"]

    off = _Fakes(_rows())
    on = _Fakes(_rows(), arc_loader=bad)
    c_off = asyncio.run(run_cycle_once(off.deps(with_arcs=False)))
    c_on = asyncio.run(run_cycle_once(on.deps(with_arcs=True)))
    assert c_on.status == "completed"
    assert [r.ref_id for r in c_on.replay] == [r.ref_id for r in c_off.replay]
    assert c_on.pressure.pressure == c_off.pressure.pressure


def test_arc_read_timeout_is_bounded(monkeypatch):
    from app import cycle
    from app.cycle import run_cycle_once

    monkeypatch.setattr(cycle, "ARC_READ_TIMEOUT_SEC", 0.05)

    def slow(since, until):
        time.sleep(0.5)
        return _live_arcs(until)

    on = _Fakes(_rows(), arc_loader=slow)

    async def timed():
        # Timed inside the loop: asyncio.run's shutdown joins the still-sleeping thread.
        t0 = time.monotonic()
        cycle_ = await run_cycle_once(on.deps(with_arcs=True))
        return cycle_, time.monotonic() - t0

    c, elapsed = asyncio.run(timed())
    assert c.status == "completed" and not any(r.source_kind == "arc" for r in c.replay)
    assert "arc candidates=0" in c.note
    assert elapsed < 0.4  # the loader sleeps 0.5 s: wait_for must return first


def test_arcs_about_one_subject_are_one_candidate_the_heaviest():
    from app.replay import arc_candidates

    def same(arc_id, dwell, returns, minutes_ago):
        return {**_arc(arc_id, kind="interoception", dwell=dwell, returns=returns,
                       ended=NOW - timedelta(minutes=minutes_ago)), "subject_ref": "node:substrate.bus_synaptic"}

    rows = [same("s1", 900, 11, 200), same("s2", 2280, 8, 100), same("s3", 1680, 7, 10),
            {**_arc("other", kind="concern"), "subject_ref": "node:substrate.bus_synaptic"}]
    got = arc_candidates(rows, SINCE, NOW)
    ids = [c.ref_id for c in got]
    assert len(ids) == 2 and "arc:other" in ids  # same subject, different kind: two things
    (intero,) = [c for c in got if c.tags == ["interoception"]]
    assert intero.ref_id == "arc:s2"  # 38 min, 8 returns: the heaviest of the three


def test_live_1012_mix_keeps_the_compaction_requests():
    """The 10-12 13:45 sleep replayed 6 metacog (1.0) + 6 compaction requests (0.5). Its
    window's top arcs weighed 0.76..0.56: under ceil(k/2) alone they displaced every request."""
    from app.replay import ReplayItemV1, arc_replay_slots, select_replay

    existing = [ReplayItemV1(ref_id=f"metacog:{i}", source_kind="metacog", text="t", weight=1.0, reason="r")
                for i in range(6)]
    existing += [ReplayItemV1(ref_id=f"compaction_request:{i}", source_kind="compaction_request", text="t",
                              weight=0.5, reason="r") for i in range(6)]
    arcs = [ReplayItemV1(ref_id=f"arc:{i}", source_kind="arc", text="t", weight=w, reason="r")
            for i, w in enumerate([0.755, 0.608, 0.579, 0.563, 0.562, 0.55, 0.53])]
    plain = select_replay(existing + arcs, 12)
    assert sum(r.source_kind == "compaction_request" for r in plain) == 0  # the failure mode
    mixed = select_replay(existing + arcs, 12, source_caps={"arc": arc_replay_slots(12)})
    kinds = [r.source_kind for r in mixed]
    assert kinds.count("metacog") == 6 and kinds.count("arc") == 4 and kinds.count("compaction_request") == 2


def test_source_caps_only_tighten_and_default_is_unchanged():
    from app.replay import ReplayItemV1, select_replay

    items = [ReplayItemV1(ref_id=f"metacog:{i}", source_kind="metacog", text="t", weight=1.0, reason="r")
             for i in range(10)]
    assert select_replay(items, 12) == select_replay(items, 12, source_caps={})
    assert len(select_replay(items, 12, source_caps={"metacog": 99})) == 6  # never above ceil(k/2)
    assert len(select_replay(items, 12, source_caps={"metacog": 2})) == 2


def _many_existing():
    rows = _rows()
    rows["compaction_request"] = [{"request_id": f"r{i}", "theme": f"theme {i}", "reason": "recurs"} for i in range(10)]
    return rows


def test_existing_sources_keep_two_thirds_with_many_existing():
    from app.cycle import run_cycle_once
    from app.settings import settings

    off = _Fakes(_many_existing())
    on = _Fakes(_many_existing(), arcs=_live_arcs)
    c_off = asyncio.run(run_cycle_once(off.deps(with_arcs=False)))
    c_on = asyncio.run(run_cycle_once(on.deps(with_arcs=True)))
    k = settings.DREAM_REPLAY_MAX
    on_existing = [r.ref_id for r in c_on.replay if r.source_kind != "arc"]
    assert len(c_on.replay) == k
    assert len(on_existing) == min(len(c_off.replay), k - k // 3) == 8
    # what stays is the top of the arc-free selection, in order: arcs only remove its tail
    assert on_existing == [r.ref_id for r in c_off.replay][: len(on_existing)]


def test_control_pool_sees_only_the_replayable_arcs(monkeypatch):
    from app import cycle
    from app.cycle import run_cycle_once
    from app.replay import arc_replay_slots
    from app.settings import settings

    seen = {}
    real = cycle.control_pairs

    def spy(pool, n, **kw):
        seen["pool"] = list(pool)
        return real(pool, n, **kw)

    monkeypatch.setattr(cycle, "control_pairs", spy)
    on = _Fakes(_rows(), arcs=lambda now: [_arc(f"y{i}", dwell=60.0 * (i + 1), returns=0,
                                               ended=now - timedelta(minutes=i + 1)) for i in range(20)])
    asyncio.run(run_cycle_once(on.deps(with_arcs=True)))
    arcs = [c for c in seen["pool"] if c.source_kind == "arc"]
    assert len(arcs) == arc_replay_slots(settings.DREAM_REPLAY_MAX) == 4
    assert {c.ref_id for c in arcs} == {f"arc:y{i}" for i in (19, 18, 17, 16)}  # the heaviest
