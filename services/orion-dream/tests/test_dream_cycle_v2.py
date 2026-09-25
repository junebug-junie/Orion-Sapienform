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
    from app.replay import build_candidates, compute_pressure

    total, counts = compute_pressure(build_candidates(_rows()))
    assert total == pytest.approx(1.0 + 0.6 + 0.5 + 0.7 + 0.8)
    assert counts == {"metacog": 2, "compaction_request": 1, "resonance": 1, "crystallization": 1}
    assert compute_pressure(build_candidates({k: [] for k in _rows()})) == (0.0, {})


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
        ('{"link": false}', None),
        ("no json at all", None),
        ('{"link": true, "claim": "too short"}', None),
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
    h = res.hypotheses[0]
    assert h.arm == "dream" and h.cycle_id == "dc-1"
    assert h.expires_at == NOW + timedelta(hours=72)


# --- cycle -------------------------------------------------------------------


class _Fakes:
    def __init__(self, rows, idle=120.0, last_end=None, answer=None):
        self.rows, self.idle, self.last_end = rows, idle, last_end
        self.persisted, self.prompts = [], []
        self.answer = answer or json.dumps(
            {"link": True, "claim": "These two recur together more often than chance would allow", "why": "w"}
        )
        self.seen_since = None

    def deps(self):
        from app.cycle import CycleDeps

        def load(since, limit):
            self.seen_since = since
            return self.rows

        async def complete(prompt):
            self.prompts.append(prompt)
            return self.answer

        return CycleDeps(
            load_source_rows=load,
            load_idle_minutes=lambda: self.idle,
            load_last_cycle_end=lambda: self.last_end,
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


# --- write surface -----------------------------------------------------------


def test_cycle_store_writes_only_v2_tables():
    from app.cycle_store import CYCLE_WRITE_TABLES

    src = Path(__file__).resolve().parents[1].joinpath("app", "cycle_store.py").read_text()
    written = set(re.findall(r"(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+([a-z_]+)", src, re.I))
    assert written == set(CYCLE_WRITE_TABLES)
