"""Dream cycle v2: pressure, replay, recombination, cycle orchestration."""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

NOW = datetime(2026, 9, 25, 4, 0, tzinfo=timezone.utc)


def _rows():
    return {
        "metacog": [
            {"id": "m1", "summary": "Recall returned empty for three turns in a row", "severity": "critical",
             "trigger_kind": "recall_empty", "tags": ["recall", "memory"]},
            {"id": "m2", "summary": "Latency spike on the chat lane", "severity": "degraded",
             "trigger_kind": "latency", "tags": ["latency"]},
            {"id": "m3", "summary": "all fine", "severity": "nominal", "trigger_kind": "x", "tags": []},
        ],
        "compaction_request": [
            {"request_id": "r1", "theme": "juniper's sleep schedule", "reason": "recurs across chains"},
        ],
        "resonance": [
            {"alert_id": "a1", "theme_key": "being watched", "violation_count": 4},
        ],
        "crystallization": [
            {"crystallization_id": "c1", "subject": "Juniper prefers plain English",
             "summary": "dense answers get asked to be re-said", "salience": 0.8, "tags": ["style"]},
            {"crystallization_id": "c2", "subject": "GPU5 contention", "summary": "metacog lane shares a GPU",
             "salience": 0.0, "tags": ["gpu"]},
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
    assert cands["compaction_request:r1"].weight == 0.5
    assert cands["resonance:a1"].weight == pytest.approx(0.7)
    assert cands["crystallization:c1"].weight == 0.8
    assert all(c.reason for c in cands.values())


def test_pressure_is_sum_of_weights_and_zero_when_nothing_new():
    from app.replay import compute_pressure, keyed_candidates

    total, counts, new_counts = compute_pressure(keyed_candidates(_rows()))
    assert total == pytest.approx(1.0 + 0.6 + 0.5 + 0.7 + 0.8)
    assert counts == new_counts == {"metacog": 2, "compaction_request": 1, "resonance": 1, "crystallization": 1}
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

    pairs = dream_pairs(_replay(), 3)
    refs = [r for p in pairs for r in (p.a.ref_id, p.b.ref_id)]
    assert len(refs) == len(set(refs))
    assert all(p.a.source_kind != p.b.source_kind for p in pairs)
    assert all(p.arm == "dream" for p in pairs)


def test_control_pairs_are_seeded_and_exclude_dream_pairs():
    from app.recombine import control_pairs, dream_pairs
    from app.replay import build_candidates

    pool = build_candidates(_rows())
    d = dream_pairs(_replay(), 2)
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
    assert res.no_link == 2  # explicit no + echo
    assert res.failures == 1
    assert res.unparseable == 0
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


def test_window_uses_last_good_cycle_start_not_attempt_end():
    from app.cycle import run_cycle_once

    start = datetime.now(timezone.utc) - timedelta(hours=10)
    f = _Fakes(_rows(), last_start=start, last_end=start + timedelta(hours=3))
    asyncio.run(run_cycle_once(f.deps()))
    assert f.seen_since == start


def test_rem_receives_window_since():
    from app.cycle import run_cycle_once

    seen = {}

    async def rem(cycle_id, since):
        seen["args"] = (cycle_id, since)
        return "compaction-delta:x"

    f = _Fakes(_rows())
    deps = f.deps()
    deps.rem_compaction = rem
    cycle = asyncio.run(run_cycle_once(deps))
    assert seen["args"] == (cycle.cycle_id, cycle.pressure.since)
    assert cycle.compaction_delta_id == "compaction-delta:x"


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

    async def start_story(cycle):
        started.append(cycle)

    deps = f.deps()
    deps.start_story = start_story
    return f, deps, started


def test_a_completed_sleep_starts_one_story_about_its_replay():
    from app.cycle import run_cycle_once
    from app.story import story_trigger
    from orion.schemas.telemetry.dream import DreamInternalTriggerV1

    f, deps, started = _story_fakes()
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle.status == "completed" and started == [cycle]

    trigger = story_trigger(cycle)
    assert trigger.trigger_id == f"sleep:{cycle.cycle_id}" and trigger.source == "orion-dream.sleep"
    assert len(trigger.sleep.replay) == len(cycle.replay) > 0
    # heaviest first, "source: text"
    weights = [r.weight for r in sorted(cycle.replay, key=lambda r: r.weight, reverse=True)]
    assert weights == sorted(weights, reverse=True)
    assert all(": " in line for line in trigger.sleep.replay)
    # What cortex-orch does with the payload: the digest survives the round trip.
    dumped = DreamInternalTriggerV1.model_validate(trigger.model_dump(mode="json")).model_dump(mode="json")
    assert dumped["sleep"]["cycle_id"] == cycle.cycle_id


def test_the_story_never_sees_the_blind_hypotheses():
    from app.cycle import run_cycle_once
    from app.story import story_trigger

    f, deps, _ = _story_fakes()
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle.hypotheses
    payload = story_trigger(cycle).model_dump_json()
    for h in cycle.hypotheses:
        assert h.claim not in payload and h.hypothesis_id not in payload
    assert '"arm"' not in payload and "control" not in payload


def test_a_failed_or_empty_sleep_starts_no_story():
    from app.cycle import run_cycle_once

    _, deps, started = _story_fakes()
    deps.complete = _always_fail
    assert asyncio.run(run_cycle_once(deps)).status == "failed"

    f = _Fakes({k: [] for k in _rows()})
    empty_deps = f.deps()
    empty_deps.start_story = deps.start_story
    assert asyncio.run(run_cycle_once(empty_deps, trigger="manual", force=True)).status == "empty"
    assert started == []


async def _always_fail(prompt):
    raise RuntimeError("gateway down")


def test_a_story_that_fails_to_start_does_not_fail_the_sleep():
    from app.cycle import run_cycle_once

    f, deps, _ = _story_fakes()

    async def broken(cycle):
        raise ConnectionError("bus down")

    deps.start_story = broken
    cycle = asyncio.run(run_cycle_once(deps))
    assert cycle.status == "completed" and f.persisted == [cycle]


def test_story_digest_clips_long_replay_text_and_marks_overdue():
    from app.story import REPLAY_TEXT_CHARS, story_trigger
    from orion.schemas.dream_cycle import DreamCycleV1, ReplayItemV1, SleepPressureV1

    now = datetime.now(timezone.utc)
    cycle = DreamCycleV1(
        cycle_id="dc-x", trigger="pressure", status="completed", started_at=now, ended_at=now,
        pressure=SleepPressureV1(since=now, computed_at=now, pressure=1.234, threshold=3.0, idle_required_minutes=45.0),
        replay=[ReplayItemV1(ref_id="a", source_kind="metacog", text="word\n" * 400, weight=0.4, reason="r"),
                ReplayItemV1(ref_id="b", source_kind="resonance", text="loud", weight=0.9, reason="r")],
        note="pairs dream=1 control=1 | overdue: 48 h without crossing threshold",
    )
    sleep = story_trigger(cycle).sleep
    assert sleep.overdue and sleep.pressure == 1.23
    assert sleep.replay[0] == "resonance: loud"
    assert len(sleep.replay[1]) <= len("metacog: ") + REPLAY_TEXT_CHARS and "\n" not in sleep.replay[1]


def test_story_after_sleep_switch_turns_the_link_off(monkeypatch):
    from app import main
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_STORY_AFTER_SLEEP_ENABLED", False)
    assert main.build_cycle_deps().start_story is None
    monkeypatch.setattr(settings, "DREAM_STORY_AFTER_SLEEP_ENABLED", True)
    assert main.build_cycle_deps().start_story is not None
