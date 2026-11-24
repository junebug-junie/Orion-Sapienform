"""Dream cycle v2: pressure, replay, recombination, cycle orchestration."""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

NOW = datetime(2026, 9, 25, 4, 0, tzinfo=timezone.utc)


_T = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)


def _rows():
    # Every source query returns the row's time as occurred_at; the dream arm pairs by it (#3664).
    t = lambda m: _T + timedelta(minutes=m)  # noqa: E731
    return {
        "metacog": [
            {"id": "m1", "summary": "Recall returned empty for three turns in a row", "severity": "critical",
             "trigger_kind": "recall_empty", "tags": ["recall", "memory"], "occurred_at": t(0)},
            {"id": "m2", "summary": "Latency spike on the chat lane", "severity": "degraded",
             "trigger_kind": "latency", "tags": ["latency"], "occurred_at": t(20)},
            {"id": "m3", "summary": "all fine", "severity": "nominal", "trigger_kind": "x", "tags": [],
             "occurred_at": t(30)},
        ],
        # compaction_request / resonance were sources until 2026-11-24 (the reverie chain, their producer,
        # was retired); two crystallizations at their weights keep the fixture's pressure the same.
        "crystallization": [
            {"crystallization_id": "r1", "subject": "juniper's sleep schedule", "summary": "recurs across days",
             "salience": 0.5, "tags": ["sleep"], "occurred_at": t(10)},
            {"crystallization_id": "a1", "subject": "being watched", "summary": "the camera feeling comes back",
             "salience": 0.7, "tags": ["camera"], "occurred_at": t(40)},
            {"crystallization_id": "c1", "subject": "Juniper prefers plain English",
             "summary": "dense answers get asked to be re-said", "salience": 0.8, "tags": ["style"],
             "occurred_at": t(50)},
            {"crystallization_id": "c2", "subject": "GPU5 contention", "summary": "metacog lane shares a GPU",
             "salience": 0.0, "tags": ["gpu"], "occurred_at": t(60)},
        ],
    }


# --- replay ------------------------------------------------------------------


def test_candidates_drop_unusable_rows_and_weight_by_declared_rules():
    from app.replay import build_candidates

    cands = {c.ref_id: c for c in build_candidates(_rows())}
    assert "metacog:m3" not in cands  # nominal is not surprise
    assert "crystallization:c2" not in cands  # zero salience carries nothing
    assert cands["metacog:m1"].weight == 1.0
    assert cands["metacog:m2"].weight == 0.6
    assert cands["crystallization:r1"].weight == 0.5
    assert cands["crystallization:a1"].weight == pytest.approx(0.7)
    assert cands["crystallization:c1"].weight == 0.8
    assert all(c.reason for c in cands.values())


def test_pressure_is_sum_of_weights_and_zero_when_nothing_new():
    from app.replay import compute_pressure, keyed_candidates

    total, counts, new_counts = compute_pressure(keyed_candidates(_rows()))
    assert total == pytest.approx(1.0 + 0.6 + 0.5 + 0.7 + 0.8)
    assert counts == new_counts == {"metacog": 2, "crystallization": 3}
    assert compute_pressure(keyed_candidates({k: [] for k in _rows()})) == (0.0, {}, {})


def _timeouts(n, severity="degraded"):
    # One event, n rows, model prose reworded every time (live: 222 rows/week of this key).
    return [{"id": f"t{i}", "summary": f"the gateway timed out again, take {i}", "severity": severity,
             "trigger_kind": "transport", "tags": [],
             "dedupe_key": "transport:rpc_timeout:orion:exec:request:llmgatewayservice"} for i in range(n)]


def test_repeats_of_one_thing_are_one_candidate_at_their_highest_weight():
    from app.replay import compute_pressure, keyed_candidates, select_replay

    rows = {"metacog": _timeouts(40) + _timeouts(1, "critical")}
    keyed = keyed_candidates(rows)
    assert len(keyed) == 1
    (item,) = keyed.values()
    assert item.weight == 1.0 and item.text == "the gateway timed out again, take 0"  # newest text, max weight
    assert compute_pressure(keyed)[0] == 1.0  # not 41 rows of pressure
    assert len(select_replay(list(keyed.values()), 12)) == 1


def test_a_thing_seen_before_the_window_adds_no_pressure_but_stays_replayable():
    from app.replay import compute_pressure, keyed_candidates, prior_keys

    keyed = keyed_candidates({**_rows(), "metacog": _rows()["metacog"] + _timeouts(5)})
    seen = prior_keys({"metacog": _timeouts(1)})
    total, counts, new_counts = compute_pressure(keyed, seen)
    assert counts["metacog"] == 3 and new_counts["metacog"] == 2  # the chronic timeout is not new
    assert total == pytest.approx(1.0 + 0.6 + 0.5 + 0.7 + 0.8)
    assert any(k.startswith("metacog:transport:") for k in keyed)


def test_rows_without_a_key_count_as_their_own_thing_never_one_shared_blank():
    """Guard for the dangerous failure: a null key collapsing everything into one
    item would pin pressure near 0 and Orion would stop sleeping."""
    from app.replay import keyed_candidates

    rows = {"metacog": [{"id": f"m{i}", "summary": "s", "severity": "critical", "trigger_kind": "x",
                         "dedupe_key": None} for i in range(3)]}
    assert len(keyed_candidates(rows)) == 3


def test_every_source_query_returns_a_key_and_honours_until():
    from app.cycle_store import SOURCE_QUERIES

    for kind, sql in SOURCE_QUERIES.items():
        assert "dedupe_key" in sql and ":until" in sql and ":since" in sql, kind
    # recall rewrites updated_at on ~100 crystallizations per retrieval: not new material
    assert "memory_crystallization_history" in SOURCE_QUERIES["crystallization"]
    assert "updated_at" not in SOURCE_QUERIES["crystallization"]
    assert "trigger_reason" in SOURCE_QUERIES["metacog"]


def test_read_pressure_reads_the_lookback_before_the_window():
    from app.cycle import read_pressure
    from app.settings import settings

    f = _Fakes({**_rows(), "metacog": _rows()["metacog"] + _timeouts(3)}, prior={"metacog": _timeouts(1)})
    last = NOW - timedelta(hours=7)
    pressure, candidates = read_pressure(f.deps(), NOW, last)
    assert f.reads == [(last, None), (last - timedelta(hours=settings.DREAM_LOOKBACK_HOURS), last)]
    assert pressure.new_counts["metacog"] == 2 and pressure.counts["metacog"] == 3
    assert len(candidates) == sum(pressure.counts.values())


def test_select_replay_caps_any_one_source():
    from app.replay import build_candidates, select_replay

    rows = {"metacog": [
        {"id": f"m{i}", "summary": f"thing {i}", "severity": "critical", "trigger_kind": "x", "tags": []}
        for i in range(10)
    ], "crystallization": _rows()["crystallization"]}
    picked = select_replay(build_candidates(rows), 4)
    kinds = [p.source_kind for p in picked]
    assert kinds.count("metacog") == 2
    assert "crystallization" in kinds


# --- recombination -----------------------------------------------------------


def _replay():
    from app.replay import build_candidates, select_replay

    return select_replay(build_candidates(_rows()), 12)


def test_dream_pairs_are_disjoint_and_prefer_cross_source():
    from app.recombine import dream_pairs

    pairs = dream_pairs(_replay(), 3, window_minutes=120)
    refs = [r for p in pairs for r in (p.a.ref_id, p.b.ref_id)]
    assert len(refs) == len(set(refs))
    assert all(p.a.source_kind != p.b.source_kind for p in pairs)
    assert all(p.arm == "dream" for p in pairs)


def _item(ref, kind, minute, weight=1.0, until=None):
    from orion.schemas.dream_cycle import ReplayItemV1

    return ReplayItemV1(
        ref_id=ref, source_kind=kind, text=f"fragment {ref}", weight=weight, reason="test",
        occurred_at=None if minute is None else _T + timedelta(minutes=minute),
        occurred_until=None if until is None else _T + timedelta(minutes=until),
    )


def test_dream_pairs_only_pair_coincident_fragments_from_different_organs():
    # #3664: the dream arm asks about things that happened near each other in different organs.
    from app.recombine import dream_pairs

    items = [
        _item("metacog:near", "metacog", 0),
        _item("compaction_request:near", "compaction_request", 30),
        _item("metacog:same_kind", "metacog", 2),             # 2 min from metacog:near, same organ: not a pair
        _item("walkway:far", "walkway", 600),                 # 9.5 h after anything else: never paired
        _item("crystallization:untimed", "crystallization", None),  # no time: never paired by this arm
    ]
    pairs = dream_pairs(items, 3, window_minutes=120)
    # One compaction fragment, so one pair: the nearer metacog (28 min) beats the farther (30 min).
    assert [(p.a.ref_id, p.b.ref_id) for p in pairs] == [("compaction_request:near", "metacog:same_kind")]
    assert all(p.a.source_kind != p.b.source_kind for p in pairs)
    # No fallback: with nothing coincident the arm asks nothing, rather than a distant pair.
    assert dream_pairs([items[3], items[4], items[0]], 3, window_minutes=120) == []


def test_dream_pairs_prefer_weight_then_nearness_and_use_each_item_once():
    from app.recombine import dream_pairs

    items = [
        _item("metacog:a", "metacog", 0, weight=1.0),
        _item("arc:b", "arc", 100, weight=0.5),
        _item("compaction_request:c", "compaction_request", 90, weight=0.9),
        _item("walkway:d", "walkway", 5, weight=0.9),
    ]
    pairs = dream_pairs(items, 3, window_minutes=120)
    keys = [frozenset((p.a.ref_id, p.b.ref_id)) for p in pairs]
    # a-c and a-d both weigh 0.9; d is nearer (5 min vs 90), so a-d wins; then c-b (0.45) over nothing else.
    assert keys[0] == frozenset(("metacog:a", "walkway:d"))
    assert keys[1] == frozenset(("compaction_request:c", "arc:b"))
    assert len(keys) == 2


def test_an_arc_span_overlapping_a_point_is_zero_minutes_apart():
    from app.recombine import time_gap_minutes

    arc = _item("arc:x", "arc", 0, until=300)      # 12:00 to 17:00
    inside = _item("metacog:y", "metacog", 200)    # 15:20, inside the span
    after = _item("metacog:z", "metacog", 330)     # 30 min after the span ends
    assert time_gap_minutes(arc, inside) == 0.0
    assert time_gap_minutes(arc, after) == 30.0
    assert time_gap_minutes(arc, _item("metacog:n", "metacog", None)) is None
    # Longer than the window: only its end counts, so it does not swallow everything inside it.
    assert time_gap_minutes(arc, inside, max_span_minutes=120) == 100.0
    assert time_gap_minutes(arc, after, max_span_minutes=120) == 30.0


def test_a_night_long_arc_does_not_take_every_pair():
    from app.recombine import dream_pairs

    night = _item("arc:night", "arc", 0, weight=1.0, until=480)     # 8 h body arc
    items = [night, _item("metacog:m", "metacog", 60, weight=0.9),
             _item("compaction_request:c", "compaction_request", 70, weight=0.9)]
    pairs = dream_pairs(items, 3, window_minutes=120)
    assert [frozenset((p.a.ref_id, p.b.ref_id)) for p in pairs] == [
        frozenset(("metacog:m", "compaction_request:c"))]


def test_every_builder_carries_the_rows_time():
    from app.replay import build_candidates

    for c in build_candidates(_rows()):
        assert c.occurred_at is not None and c.occurred_at.tzinfo is not None, c.ref_id


def test_control_pairs_are_seeded_and_exclude_dream_pairs():
    from app.recombine import control_pairs, dream_pairs
    from app.replay import build_candidates

    pool = build_candidates(_rows())
    d = dream_pairs(_replay(), 2, window_minutes=120)
    one = control_pairs(pool, 2, seed="dc-abc", exclude=d)
    two = control_pairs(list(reversed(pool)), 2, seed="dc-abc", exclude=d)
    assert [(p.a.ref_id, p.b.ref_id) for p in one] == [(p.a.ref_id, p.b.ref_id) for p in two]
    dream_keys = {frozenset((p.a.ref_id, p.b.ref_id)) for p in d}
    assert all(frozenset((p.a.ref_id, p.b.ref_id)) not in dream_keys for p in one)
    assert all(p.arm == "control" for p in one)


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"link": false}', "no_link"),
        ('{"maybe": 1}', None),
        ("no json at all", None),
        ('{"link": true, "claim": "too short"}', None),  # malformed, not a decline
        ('sure! {"link": true, "claim": "Recall empties cluster right after GPU5 contention", "why": "timing"}',
         ("Recall empties cluster right after GPU5 contention", "timing")),
    ],
)
def test_parse_link(text, expected):
    from app.recombine import parse_link

    assert parse_link(text) == expected


def test_recombine_counts_no_link_failures_and_rejects_echo():
    from app.recombine import Pair, recombine

    items = _replay()
    pairs = [Pair(items[0], items[1], "dream"), Pair(items[1], items[2], "dream"),
             Pair(items[2], items[3], "control"), Pair(items[0], items[3], "control")]
    echo = items[0].text[:60]
    answers = iter([
        json.dumps({"link": True, "claim": "Metacog criticals predict reverie rumination the next hour", "why": "w"}),
        json.dumps({"link": False}),
        RuntimeError("gateway down"),
        json.dumps({"link": True, "claim": echo}),
    ])

    async def complete(_prompt):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    res = asyncio.run(recombine(pairs, complete, cycle_id="dc-1", ttl_hours=72, now=NOW))
    assert len(res.hypotheses) == 1
    assert res.no_link == 1  # the explicit no only
    assert res.echo == 1  # the pasted-back fragment is counted apart (#3412)
    assert res.failures == 1
    assert res.unparseable == 0
    # Every pair asked leaves one stored answer, in order, with what it was counted as.
    assert [a.outcome for a in res.answers] == ["linked", "no_link", "failed", "echo"]
    assert [(a.ref_a, a.ref_b, a.arm) for a in res.answers] == [(p.a.ref_id, p.b.ref_id, p.arm) for p in pairs]
    assert res.answers[1].answer == json.dumps({"link": False})
    assert res.answers[2].answer == ""
    h = res.hypotheses[0]
    assert h.arm == "dream" and h.cycle_id == "dc-1"
    assert h.expires_at == NOW + timedelta(hours=72)


# --- cycle -------------------------------------------------------------------


class _Fakes:
    def __init__(self, rows, idle=120.0, last_end=None, answer=None, last_start=None, prior=None):
        self.rows, self.idle, self.last_end, self.last_start = rows, idle, last_end, last_start
        self.prior, self.reads = prior or {}, []
        self.persisted, self.prompts = [], []
        self.answer = answer or json.dumps(
            {"link": True, "claim": "These two recur together more often than chance would allow", "why": "w"}
        )
        self.seen_since = None

    def deps(self):
        from app.cycle import CycleDeps

        def load(since, limit, until=None):
            self.reads.append((since, until))
            if until is not None:  # the lookback before the window
                return self.prior
            self.seen_since = since
            return self.rows

        async def complete(prompt):
            self.prompts.append(prompt)
            return self.answer

        return CycleDeps(
            load_source_rows=load,
            load_idle_minutes=lambda: self.idle,
            load_last_window_start=lambda: self.last_start,
            load_last_attempt_end=lambda: self.last_end,
            persist_cycle=lambda c: self.persisted.append(c) or True,
            complete=complete,
        )


def test_cycle_not_due_when_not_idle():
    from app.cycle import run_cycle_once

    f = _Fakes(_rows(), idle=5.0)
    assert asyncio.run(run_cycle_once(f.deps())) is None
    assert f.persisted == [] and f.prompts == []


def test_cycle_not_due_when_idle_unknown():
    from app.cycle import run_cycle_once

    f = _Fakes(_rows(), idle=None)
    assert asyncio.run(run_cycle_once(f.deps())) is None


def test_cycle_not_due_when_last_cycle_too_recent():
    from app.cycle import run_cycle_once

    f = _Fakes(_rows(), last_end=datetime.now(timezone.utc) - timedelta(minutes=30))
    assert asyncio.run(run_cycle_once(f.deps())) is None


def test_cycle_runs_both_arms_and_persists():
    from app.cycle import run_cycle_once
    from app.settings import settings

    f = _Fakes(_rows())
    cycle = asyncio.run(run_cycle_once(f.deps()))
    assert cycle is not None and cycle.status == "completed"
    arms = [h.arm for h in cycle.hypotheses]
    assert arms.count("dream") >= 1 and arms.count("control") == settings.DREAM_CONTROL_PER_CYCLE
    assert f.persisted == [cycle]
    assert cycle.pressure.pressure == pytest.approx(3.6)
    # the arm never reaches the prompt
    assert all("control" not in p and "arm" not in p.lower() for p in f.prompts)


def test_forced_cycle_with_nothing_new_is_honestly_empty():
    from app.cycle import run_cycle_once

    f = _Fakes({k: [] for k in _rows()}, idle=0.0)
    cycle = asyncio.run(run_cycle_once(f.deps(), trigger="manual", force=True))
    assert cycle.status == "empty" and cycle.hypotheses == [] and f.prompts == []


def test_window_is_last_cycle_end_but_capped_by_lookback():
    from app.cycle import window_start
    from app.settings import settings

    recent = NOW - timedelta(hours=3)
    assert window_start(NOW, recent) == recent
    assert window_start(NOW, recent.replace(tzinfo=None)) == recent  # naive db value
    assert window_start(NOW, None) == NOW - timedelta(hours=settings.DREAM_LOOKBACK_HOURS)
    ancient = NOW - timedelta(days=30)
    assert window_start(NOW, ancient) == NOW - timedelta(hours=settings.DREAM_LOOKBACK_HOURS)


def test_all_llm_calls_failing_marks_cycle_failed():
    from app.cycle import run_cycle_once

    f = _Fakes(_rows())

    async def boom(_p):
        raise RuntimeError("gateway down")

    deps = f.deps()
    deps.complete = boom
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle.status == "failed" and cycle.llm_failures > 0 and cycle.hypotheses == []


def test_unparseable_answers_are_not_counted_as_declines():
    from app.cycle import run_cycle_once

    f = _Fakes(_rows(), answer="I think they are related, honestly")
    cycle = asyncio.run(run_cycle_once(f.deps()))
    assert cycle.status == "completed"
    assert cycle.no_link_count == 0 and cycle.unparseable_count > 0
    # The answers that produced those counts are saved with the sleep, cut to 300 characters.
    assert len(cycle.pair_answers) == cycle.unparseable_count
    assert {a.outcome for a in cycle.pair_answers} == {"unparseable"}
    assert cycle.pair_answers[0].answer == "I think they are related, honestly"


def test_pair_answers_are_capped_and_survive_the_stored_json():
    from app.cycle import run_cycle_once
    from orion.schemas.dream_cycle import PAIR_ANSWER_CHARS, DreamCycleV1

    f = _Fakes(_rows(), answer=json.dumps({"link": False}) + " " + "x" * 2000)
    cycle = asyncio.run(run_cycle_once(f.deps()))
    assert cycle.pair_answers and all(len(a.answer) == PAIR_ANSWER_CHARS for a in cycle.pair_answers)
    assert cycle.no_link_count == len(cycle.pair_answers) and cycle.echo_count == 0
    # cycle_json is model_dump(mode="json"): it must read back whole.
    back = DreamCycleV1.model_validate(json.loads(json.dumps(cycle.model_dump(mode="json"))))
    assert back.pair_answers == cycle.pair_answers


def test_old_cycle_json_without_pair_answers_still_loads():
    from orion.schemas.dream_cycle import DreamCycleV1

    f = _Fakes(_rows())
    from app.cycle import run_cycle_once
    raw = asyncio.run(run_cycle_once(f.deps())).model_dump(mode="json")
    raw.pop("pair_answers"); raw.pop("echo_count")
    old = DreamCycleV1.model_validate(raw)
    assert old.pair_answers == [] and old.echo_count == 0


def test_window_uses_last_good_cycle_start_not_attempt_end():
    from app.cycle import run_cycle_once

    start = datetime.now(timezone.utc) - timedelta(hours=10)
    f = _Fakes(_rows(), last_start=start, last_end=start + timedelta(hours=3))
    asyncio.run(run_cycle_once(f.deps()))
    assert f.seen_since == start


# --- write surface -----------------------------------------------------------


def test_cycle_store_writes_only_v2_tables():
    from app.cycle_store import CYCLE_WRITE_TABLES

    src = Path(__file__).resolve().parents[1].joinpath("app", "cycle_store.py").read_text()
    written = set(re.findall(r"(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+([a-z_]+)", src, re.I))
    assert written == set(CYCLE_WRITE_TABLES)


def test_process_floors_hold_when_persist_fails(monkeypatch):
    """Migration not applied: db reads 'never slept', floors must still hold."""
    from app import cycle_store, main
    from orion.schemas.dream_cycle import DreamCycleV1, SleepPressureV1

    main._CYCLE_STATE.clear()
    monkeypatch.setattr(cycle_store, "load_last_window_start", lambda **kwargs: None)
    monkeypatch.setattr(cycle_store, "load_last_attempt_end", lambda **kwargs: None)
    monkeypatch.setattr(cycle_store, "persist_cycle", lambda c: False)
    deps = main.build_cycle_deps()
    p = SleepPressureV1(since=NOW, pressure=0, threshold=1, idle_required_minutes=1)

    def cyc(status, start):
        return DreamCycleV1(cycle_id=f"dc-{status}", trigger="pressure", status=status,
                            started_at=start, ended_at=start + timedelta(minutes=5), pressure=p)

    deps.persist_cycle(cyc("completed", NOW))
    deps.persist_cycle(cyc("failed", NOW + timedelta(hours=1)))
    assert deps.load_last_window_start() == NOW  # failed cycle does not advance the window
    assert deps.load_last_attempt_end() == NOW + timedelta(hours=1, minutes=5)
    main._CYCLE_STATE.clear()


class _ReplyBus:
    """Bus fake for app.llm.complete: answers every rpc with one fixed gateway payload."""

    def __init__(self, payload):
        from types import SimpleNamespace

        self.codec = SimpleNamespace(
            decode=lambda _data: SimpleNamespace(ok=True, envelope=SimpleNamespace(payload=payload))
        )

    async def rpc_request(self, *_a, **_k):
        return {"data": b"-"}


# The gateway's own reply when the GPU pool sheds a call (orion-llm-gateway
# app/main.py _pool_unavailable_result): empty text, raw.error set.
_SHED_REPLY = {
    "text": "", "content": "", "spark_meta": {}, "route": "metacog", "served_by": None,
    "raw": {"error": "gpu_pool_unavailable", "details": {"reason": "shed:cabinet_hot"}},
}


def test_gateway_error_reply_raises_instead_of_returning_empty_text():
    from app import llm

    with pytest.raises(llm.GatewayRefused, match="shed:cabinet_hot"):
        asyncio.run(llm.complete(_ReplyBus(_SHED_REPLY), "p"))
    with pytest.raises(llm.GatewayRefused):
        asyncio.run(llm.complete(_ReplyBus({"content": "   "}), "p"))
    with pytest.raises(llm.GatewayRefused, match="upstream_error"):
        asyncio.run(llm.complete(_ReplyBus({"content": "[Error: llamacpp URL not configured]", "raw": {}}), "p"))
    assert asyncio.run(llm.complete(_ReplyBus({"content": '{"link": false}'}), "p")) == '{"link": false}'


def test_all_shed_sleep_is_stored_failed_not_completed():
    """Live 2026-10-08 06:27/18:27: every call shed for heat, cycle stored 'completed' with
    4 'unparseable', and the next sleep's replay window started after it (the window
    starts at the last non-failed cycle, cycle_store.LAST_WINDOW_START_SQL)."""
    from app import llm
    from app.cycle import run_cycle_once

    f = _Fakes(_rows())
    deps = f.deps()
    deps.complete = lambda p: llm.complete(_ReplyBus(_SHED_REPLY), p)
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle.status == "failed"
    assert cycle.unparseable_count == 0 and cycle.llm_failures > 0


def test_gateway_refusal_keeps_the_bus_but_a_transport_error_drops_it(monkeypatch):
    from app import llm, main

    drops = []

    async def bus():
        return object()

    async def drop():
        drops.append(1)

    monkeypatch.setattr(main, "_cycle_bus", bus)
    monkeypatch.setattr(main, "_drop_cycle_bus", drop)
    complete = main.build_cycle_deps().complete

    async def refused(_bus, _p):
        raise llm.GatewayRefused("gpu_pool_unavailable:shed:cabinet_hot")

    monkeypatch.setattr(llm, "complete", refused)
    with pytest.raises(llm.GatewayRefused):
        asyncio.run(complete("p"))
    assert drops == []

    async def timeout(_bus, _p):
        raise TimeoutError("rpc")

    monkeypatch.setattr(llm, "complete", timeout)
    with pytest.raises(TimeoutError):
        asyncio.run(complete("p"))
    assert drops == [1]


def _all_seen(last_start, idle=120.0):
    # Every thing in the window was also seen before it: pressure 0, candidates present.
    return _Fakes(_rows(), idle=idle, last_start=last_start, prior=_rows())


def test_overdue_backstop_sleeps_when_nothing_new_but_the_window_hit_its_reach():
    from app.cycle import run_cycle_once
    from app.settings import settings

    last = datetime.now(timezone.utc) - timedelta(hours=settings.DREAM_LOOKBACK_HOURS + 1)
    f = _all_seen(last)
    cycle = asyncio.run(run_cycle_once(f.deps()))
    assert cycle is not None and cycle.status == "completed"
    assert cycle.pressure.pressure == 0.0 and "overdue" in (cycle.note or "")


def test_no_backstop_before_the_window_reaches_back_its_full_lookback():
    from app.cycle import run_cycle_once

    f = _all_seen(datetime.now(timezone.utc) - timedelta(hours=10))
    assert asyncio.run(run_cycle_once(f.deps())) is None


def test_backstop_never_sleeps_through_a_conversation():
    from app.cycle import run_cycle_once
    from app.settings import settings

    last = datetime.now(timezone.utc) - timedelta(hours=settings.DREAM_LOOKBACK_HOURS + 1)
    assert asyncio.run(run_cycle_once(_all_seen(last, idle=5.0).deps())) is None


def test_pressure_endpoint_reports_the_overdue_backstop(monkeypatch):
    """The Hub gauge reads this to say "will sleep once quiet" below the line,
    instead of "not tired enough" right before a backstop sleep."""
    from app import main
    from app.settings import settings

    last = datetime.now(timezone.utc) - timedelta(hours=settings.DREAM_LOOKBACK_HOURS + 1)
    monkeypatch.setattr(main, "build_cycle_deps", lambda: _all_seen(last).deps())
    out = asyncio.run(main.cycle_pressure_endpoint())
    assert out["overdue"] is True and out["lookback_hours"] == settings.DREAM_LOOKBACK_HOURS
    assert out["pressure"]["pressure"] == 0.0 and out["candidates"] > 0

    recent = datetime.now(timezone.utc) - timedelta(hours=2)
    monkeypatch.setattr(main, "build_cycle_deps", lambda: _all_seen(recent).deps())
    assert asyncio.run(main.cycle_pressure_endpoint())["overdue"] is False


# --- sleep -> story -------------------------------------------------------------


def _story_fakes(**kw):
    f = _Fakes(_rows(), **kw)
    started = []

    async def start_story(trigger):
        started.append(trigger)

    deps = f.deps()
    deps.start_story = start_story
    return f, deps, started


def _material_text(item):
    return f"{item.source_kind}: {' '.join(item.text.split())}"[:300]


def test_a_completed_sleep_starts_one_story_about_what_it_worked_on():
    from app.cycle import run_cycle_once
    from orion.schemas.telemetry.dream import DreamInternalTriggerV1

    f, deps, started = _story_fakes()
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle.status == "completed" and len(started) == 1
    trigger = started[0]
    assert trigger.trigger_id == f"sleep:{cycle.cycle_id}" and trigger.source == "orion-dream.sleep"
    assert trigger.sleep.threshold == cycle.pressure.threshold and not trigger.sleep.overdue
    for r in cycle.replay:
        assert any(line.startswith(_material_text(r)[:40]) for line in trigger.sleep.material)
    # What cortex-orch does with the payload: the digest survives the round trip.
    dumped = DreamInternalTriggerV1.model_validate(trigger.model_dump(mode="json")).model_dump(mode="json")
    assert dumped["sleep"]["material"] == trigger.sleep.material


def test_the_story_exposes_both_arms_equally_and_never_the_hypotheses(monkeypatch):
    """Dream pairs come from replay, control pairs from the whole pool. Every item
    in either arm's hypotheses must be in the story, or the story primes one arm."""
    from app.cycle import run_cycle_once
    from app.replay import keyed_candidates
    from app.settings import settings

    # Replay of 2 = the one dream pair, which control may not reuse: every control
    # pair must then reach outside the replay, so this fails if only replay is told.
    monkeypatch.setattr(settings, "DREAM_REPLAY_MAX", 2)
    monkeypatch.setattr(settings, "DREAM_HYPOTHESES_PER_CYCLE", 1)
    f, deps, started = _story_fakes()
    cycle = asyncio.run(run_cycle_once(deps))
    assert {h.arm for h in cycle.hypotheses} == {"dream", "control"} and len(started) == 1
    replay_refs = {r.ref_id for r in cycle.replay}
    assert any({h.ref_a, h.ref_b} - replay_refs for h in cycle.hypotheses if h.arm == "control")
    material = started[0].sleep.material
    pool = {c.ref_id: c for c in keyed_candidates(_rows()).values()}
    for h in cycle.hypotheses:
        for ref in (h.ref_a, h.ref_b):
            assert _material_text(pool[ref]) in material, (h.arm, ref)
    assert set(started[0].model_dump(mode="json")["sleep"]) == {
        "cycle_id", "started_at", "pressure", "threshold", "overdue", "material"}
    blob = started[0].model_dump_json()
    for h in cycle.hypotheses:
        assert h.claim not in blob and h.hypothesis_id not in blob


def test_control_pair_items_outside_replay_reach_the_story_in_a_fixed_order():
    from app.cycle import run_cycle_once
    from app.story import story_trigger

    _, deps, _ = _story_fakes()
    cycle = asyncio.run(run_cycle_once(deps))
    outsider = cycle.replay[0].model_copy(update={"ref_id": "pool-only", "text": "a thing only the control arm drew"})
    trigger = story_trigger(cycle, control_items=[outsider])
    assert f"{outsider.source_kind}: a thing only the control arm drew" in trigger.sleep.material
    assert len(trigger.sleep.material) == len(cycle.replay) + 1
    assert story_trigger(cycle, control_items=[outsider]).sleep.material == trigger.sleep.material


def test_a_failed_empty_or_unsaved_sleep_starts_no_story():
    from app.cycle import run_cycle_once

    _, deps, started = _story_fakes()
    deps.complete = _always_fail
    assert asyncio.run(run_cycle_once(deps)).status == "failed"

    f = _Fakes({k: [] for k in _rows()})
    empty_deps = f.deps()
    empty_deps.start_story = deps.start_story
    assert asyncio.run(run_cycle_once(empty_deps, trigger="manual", force=True)).status == "empty"

    _, unsaved, _ = _story_fakes()
    unsaved.start_story = deps.start_story
    unsaved.persist_cycle = lambda c: False
    assert asyncio.run(run_cycle_once(unsaved)).status == "completed"
    assert started == []


async def _always_fail(prompt):
    raise RuntimeError("gateway down")


def test_a_story_that_fails_to_start_does_not_fail_the_sleep():
    from app.cycle import run_cycle_once

    f, deps, _ = _story_fakes()

    async def broken(trigger):
        raise ConnectionError("bus down")

    deps.start_story = broken
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle.status == "completed" and f.persisted == [cycle]


def test_an_overdue_sleep_tells_the_story_it_was_overdue():
    from app.cycle import run_cycle_once
    from app.settings import settings

    f, deps, started = _story_fakes(last_start=datetime.now(timezone.utc) - timedelta(hours=settings.DREAM_LOOKBACK_HOURS + 1),
                                    prior=_rows())  # everything seen before: pressure 0, backstop sleeps
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle is not None and "overdue" in cycle.note
    assert started[0].sleep.overdue


def test_story_material_clips_long_text():
    from app.story import ITEM_TEXT_CHARS, story_material
    from orion.schemas.dream_cycle import ReplayItemV1

    lines = story_material([ReplayItemV1(ref_id="a", source_kind="metacog", text="word\n" * 400, weight=0.4, reason="r")], [], "s")
    assert len(lines[0]) <= len("metacog: ") + ITEM_TEXT_CHARS and "\n" not in lines[0]


def test_story_after_sleep_switch_turns_the_link_off(monkeypatch):
    from app import main
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_STORY_AFTER_SLEEP_ENABLED", False)
    assert main.build_cycle_deps().start_story is None
    monkeypatch.setattr(settings, "DREAM_STORY_AFTER_SLEEP_ENABLED", True)
    assert main.build_cycle_deps().start_story is not None


# --- sleep -> carried dream (dream.carry) ----------------------------------------------------


class _CarryBus:
    """Stands in for the cycle bus: records cortex RPCs and publishes."""

    def __init__(self, fail=False):
        from orion.core.bus.codec import OrionCodec

        self.codec = OrionCodec()
        self.rpcs, self.published, self.fail = [], [], fail

    async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        self.rpcs.append((channel, env))
        if self.fail:
            raise TimeoutError("cortex-orch did not answer")
        durable = env.payload["context"]["metadata"]["durable_run"]
        return {"data": self.codec.encode(BaseEnvelope(
            kind="cortex.orch.result", source=ServiceRef(name="orion-cortex-orch"),
            payload={"status": "accepted", "metadata": {"durable_run": {
                "run_id": durable["run_id"], "workflow_kind": durable["workflow"],
                "requested_resource": durable["admission"]["resource"]}}}))}

    async def publish(self, channel, env):
        self.published.append((channel, env))


def _carry_deps(monkeypatch, *, carry_enabled, fail=False):
    from app import cycle as cycle_mod
    from app import main
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_STORY_AFTER_SLEEP_ENABLED", True)
    monkeypatch.setattr(settings, "DREAM_CARRY_ENABLED", carry_enabled)
    monkeypatch.setattr(settings, "ORION_BUS_ENABLED", True)
    bus, dropped, built = _CarryBus(fail=fail), [], []

    async def fake_bus():
        return bus

    async def fake_drop():
        dropped.append(True)

    real_story_trigger = cycle_mod.story_trigger

    def recording_story_trigger(*a, **kw):
        t = real_story_trigger(*a, **kw)
        built.append(t)
        return t

    monkeypatch.setattr(main, "_cycle_bus", fake_bus)
    monkeypatch.setattr(main, "_drop_cycle_bus", fake_drop)
    monkeypatch.setattr(cycle_mod, "story_trigger", recording_story_trigger)
    f, deps, _ = _story_fakes()
    deps.start_story = main.build_cycle_deps().start_story
    return f, deps, bus, dropped, built


def test_with_carry_on_a_completed_sleep_submits_one_dream_carry_and_no_story(monkeypatch):
    from app.cycle import run_cycle_once
    from app.settings import settings
    from orion.schemas.dream_carry import dream_carry_run_id
    from orion.schemas.durable_run import DurableRunRequestV1

    f, deps, bus, _, built = _carry_deps(monkeypatch, carry_enabled=True)
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle.status == "completed" and len(built) == 1
    assert bus.published == []  # no dream.trigger
    (channel, env), = bus.rpcs
    assert channel == settings.CHANNEL_CORTEX_REQUEST and env.kind == "cortex.orch.request"
    request = DurableRunRequestV1.model_validate(env.payload["context"]["metadata"]["durable_run"])
    assert request.workflow == "dream.carry"
    assert request.run_id == dream_carry_run_id(f"sleep:{cycle.cycle_id}")
    assert request.brief.trigger_id == f"sleep:{cycle.cycle_id}"
    assert request.brief.sleep == built[0].sleep  # the digest story_trigger built, unchanged
    assert request.admission.resource == "llm.route.metacog_background"


def test_with_carry_off_a_completed_sleep_still_publishes_the_story(monkeypatch):
    from app.cycle import run_cycle_once
    from app.settings import settings

    f, deps, bus, _, built = _carry_deps(monkeypatch, carry_enabled=False)
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle.status == "completed" and bus.rpcs == []
    (channel, env), = bus.published
    assert channel == settings.CHANNEL_DREAM_TRIGGER and env.kind == "dream.trigger"
    assert env.payload["sleep"]["material"] == built[0].sleep.material


def test_a_carry_that_fails_to_submit_falls_back_to_the_story_and_keeps_the_sleep(monkeypatch):
    """Transport failure: the bus is dropped for reconnect, and the sleep still gets its dream
    (the one-shot story), so a carry that cannot start never costs the sleep its dream."""
    from app.cycle import run_cycle_once
    from app.settings import settings

    f, deps, bus, dropped, built = _carry_deps(monkeypatch, carry_enabled=True, fail=True)
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle.status == "completed" and f.persisted == [cycle]
    assert len(bus.rpcs) == 2 and dropped == [True, True]  # resubmitted once before falling back
    (channel, env), = bus.published
    assert channel == settings.CHANNEL_DREAM_TRIGGER and env.payload["sleep"]["material"] == built[0].sleep.material


def test_a_carry_cortex_refuses_falls_back_to_the_story_without_dropping_the_bus(monkeypatch):
    from app.cycle import run_cycle_once

    f, deps, bus, dropped, _ = _carry_deps(monkeypatch, carry_enabled=True)

    async def refusing(channel, env, *, reply_channel, timeout_sec):
        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        bus.rpcs.append((channel, env))
        return {"data": bus.codec.encode(BaseEnvelope(kind="cortex.orch.result", source=ServiceRef(name="o"),
                                                      payload={"status": "fail", "error": {"type": "AdmissionUnconfirmed"}}))}

    bus.rpc_request = refusing
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle.status == "completed" and len(bus.rpcs) == 2 and dropped == []
    assert [env.kind for _, env in bus.published] == ["dream.trigger"]


def test_a_carry_submit_that_times_out_once_is_resubmitted_not_doubled_with_a_story(monkeypatch):
    """A timeout can land after durable-runs already took the run: resubmit (same run_id,
    dedupes) instead of also publishing the story, which would give the sleep two dreams."""
    from app.cycle import run_cycle_once

    f, deps, bus, dropped, _ = _carry_deps(monkeypatch, carry_enabled=True)
    accept = bus.rpc_request
    attempts = []

    async def flaky(channel, env, *, reply_channel, timeout_sec):
        attempts.append(env.payload["context"]["metadata"]["durable_run"]["run_id"])
        if len(attempts) == 1:
            bus.rpcs.append((channel, env))
            raise TimeoutError("receipt late")
        return await accept(channel, env, reply_channel=reply_channel, timeout_sec=timeout_sec)

    bus.rpc_request = flaky
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle.status == "completed"
    assert len(attempts) == 2 and attempts[0] == attempts[1]  # same deterministic run_id
    assert bus.published == [] and dropped == [True]


def test_the_carry_step_responder_runs_even_with_new_carries_off(monkeypatch):
    """DREAM_CARRY_ENABLED gates new carries only; in-flight carries must still get answers."""
    from app import carry_listener, main
    from app.settings import settings

    started = []

    class FakeListener:
        async def start(self):
            started.append("start")

        async def stop(self):
            started.append("stop")

    monkeypatch.setattr(carry_listener, "build_carry_listener", lambda: FakeListener())
    monkeypatch.setattr(settings, "DREAM_INTROSPECT_ENABLED", False)
    monkeypatch.setattr(settings, "ORION_DREAM_CYCLE_ENABLED", False)
    monkeypatch.setattr(settings, "ORION_BUS_ENABLED", True)
    for enabled in (False, True):
        monkeypatch.setattr(settings, "DREAM_CARRY_ENABLED", enabled)

        async def run():
            async with main.lifespan(main.app):
                pass
        asyncio.run(run())
    assert started == ["start", "stop", "start", "stop"]
    monkeypatch.setattr(settings, "ORION_BUS_ENABLED", False)
    asyncio.run(run())
    assert started == ["start", "stop", "start", "stop"]  # no bus, no responder


# --- retired reverie sources --------------------------------------------------


def test_reverie_chain_sources_are_retired():
    """compaction_request and resonance came only from the reverie narration chain (retired 2026-11-24):
    no source query reads them and no candidate is built from their rows."""
    from app.cycle_store import SOURCE_QUERIES
    from app.replay import build_candidates

    assert set(SOURCE_QUERIES) == {"metacog", "crystallization"}
    assert build_candidates({"compaction_request": [{"request_id": "r1", "theme": "t", "reason": "r"}],
                             "resonance": [{"alert_id": "a1", "theme_key": "t", "violation_count": 3}]}) == []


def test_curiosity_compaction_only_after_completed_saved_sleep():
    from app.cycle import run_cycle_once
    calls = []
    async def compact(cycle):
        calls.append(cycle.cycle_id)
    _, deps, stories = _story_fakes()
    deps.compact_curiosity = compact
    done = asyncio.run(run_cycle_once(deps))
    assert calls == [done.cycle_id] and stories
    _, unsaved, _ = _story_fakes()
    unsaved.compact_curiosity = compact
    unsaved.persist_cycle = lambda c: False
    asyncio.run(run_cycle_once(unsaved))
    empty = _Fakes({}).deps()
    empty.compact_curiosity = compact
    asyncio.run(run_cycle_once(empty, force=True))
    failed = _Fakes(_rows()).deps()
    async def no_model(prompt):
        raise RuntimeError("gateway unavailable")
    failed.complete = no_model
    failed.compact_curiosity = compact
    asyncio.run(run_cycle_once(failed))
    assert calls == [done.cycle_id]


def test_curiosity_failure_does_not_fail_sleep_and_switch_wires_hook(monkeypatch):
    from app import main
    from app.cycle import run_cycle_once
    async def fail(cycle):
        raise RuntimeError("store unavailable")
    _, deps, _ = _story_fakes()
    deps.compact_curiosity = fail
    assert asyncio.run(run_cycle_once(deps)).status == "completed"
    monkeypatch.setattr(main.settings, "DREAM_CURIOSITY_COMPACTION_ENABLED", False)
    assert main.build_cycle_deps().compact_curiosity is None
    monkeypatch.setattr(main.settings, "DREAM_CURIOSITY_COMPACTION_ENABLED", True)
    monkeypatch.setattr(main.settings, "DREAM_STORY_AFTER_SLEEP_ENABLED", False)
    monkeypatch.setattr(main.settings, "DREAM_CARRY_ENABLED", False)
    assert main.build_cycle_deps().compact_curiosity is not None


def test_an_answer_with_nul_or_lone_surrogate_is_stored_clean():
    # jsonb refuses \u0000 ("cannot be converted to text"), which would lose the whole sleep's
    # one-transaction persist; a lone surrogate cannot be encoded as UTF-8 at all.
    from app.recombine import Pair, recombine

    items = _replay()
    pairs = [Pair(items[0], items[1], "dream"), Pair(items[1], items[2], "control")]
    answers = iter(['garbled\x00answer', '{"link": false} \ud800 tail'])

    async def complete(_prompt):
        return next(answers)

    res = asyncio.run(recombine(pairs, complete, cycle_id="dc-1", ttl_hours=72, now=NOW))
    assert [a.outcome for a in res.answers] == ["unparseable", "no_link"]
    assert res.answers[0].answer == "garbledanswer"
    stored = json.dumps([a.model_dump(mode="json") for a in res.answers])
    assert "\\u0000" not in stored and "\\ud800" not in stored
    stored.encode("utf-8")  # must not raise


def test_the_pair_caps_fit_the_stored_lists():
    # pair_answers and hypotheses share MAX_HYPOTHESES; raising a settings cap past it would make
    # DreamCycleV1 refuse the sleep before it is saved.
    from app.settings import Settings
    from orion.schemas.dream_cycle import MAX_HYPOTHESES

    def le(name):
        return next(m.le for m in Settings.model_fields[name].metadata if hasattr(m, "le"))

    assert le("DREAM_HYPOTHESES_PER_CYCLE") + le("DREAM_CONTROL_PER_CYCLE") <= MAX_HYPOTHESES


def test_metacog_fragment_carries_its_time_and_evidence():
    # #3412: the summary alone is prose the model rightly declines to link; the facts and the time
    # are what make a claimed link checkable.
    from app.replay import candidate_from_metacog

    item = candidate_from_metacog({
        "id": "m1", "summary": "The motor burned out before the reply could form.", "severity": "critical",
        "trigger_kind": "chat_turn", "occurred_at": datetime(2026, 11, 17, 23, 2, 14, tzinfo=timezone.utc),
        "evidence": "compliance failed; grounding: finalize_reply_deadline",
    })
    assert item.text == ("[11-17 23:02 UTC] The motor burned out before the reply could form. "
                         "Evidence: compliance failed; grounding: finalize_reply_deadline.")
    # An ISO string (orion_metacog.timestamp is producer-written text) reads the same.
    same = candidate_from_metacog({"id": "m1", "summary": "s", "severity": "critical",
                                   "occurred_at": "2026-11-17T23:02:14Z"})
    assert same.text == "[11-17 23:02 UTC] s"
    # A row without them keeps the old text; a bad time is dropped, not guessed.
    assert candidate_from_metacog({"id": "m1", "summary": "s", "severity": "critical"}).text == "s"
    assert candidate_from_metacog({"id": "m1", "summary": "s", "severity": "critical",
                                   "occurred_at": "not a time"}).text == "s"
    assert candidate_from_metacog({"id": "m1", "summary": "", "severity": "critical", "evidence": "e"}) is None
    # Evidence that already ends in a full stop gets one, not two; long evidence fits the cap whole-ended.
    assert candidate_from_metacog({"id": "m1", "summary": "s", "severity": "critical",
                                   "evidence": "done."}).text == "s Evidence: done."
    from app.replay import TEXT_CAP
    fits = candidate_from_metacog({"id": "m1", "summary": "y" * 272, "severity": "critical",
                                   "occurred_at": "2026-11-17T23:02:14Z", "evidence": "z" * 300})
    assert len(fits.text) <= TEXT_CAP and fits.text.endswith("….")  # evidence cut, its close kept
    full = candidate_from_metacog({"id": "m1", "summary": "y" * 700, "severity": "critical", "evidence": "z"})
    assert len(full.text) == TEXT_CAP and "Evidence" not in full.text  # no room: summary only, capped


def test_source_queries_read_the_content_behind_the_label():
    from app.cycle_store import SOURCE_QUERIES

    meta = " ".join(SOURCE_QUERIES["metacog"].split())
    assert "what_changed->'evidence'" in meta and "AS occurred_at" in meta and "evidence FROM" in meta
