"""Each new philosophical curiosity finding in the top tenth by novelty posts one event want (step 3). No network."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from orion.physiology.event_wants import event_want_id
from scripts.curiosity_event_wants import (
    FAMILY,
    FINDINGS_SQL,
    POSTED_SQL,
    SALIENCE_BAND,
    CuriosityEventWants,
    plan_findings,
)

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


def _rows(recent=(0.50, 0.12)):
    # 40 older philosophical findings over the last 20 days, novelty 0.10 .. 0.139; then the recent ones.
    rows = [{"run_id": f"old{k}", "line": "investigate", "novelty": 0.10 + 0.001 * k,
             "created_at": NOW - timedelta(days=20) + timedelta(hours=10 * k)} for k in range(40)]
    rows += [{"run_id": f"new{i}", "line": "self_inquiry", "novelty": n,
              "created_at": NOW - timedelta(minutes=30 + i)} for i, n in enumerate(recent)]
    return rows


class _Conn:
    def __init__(self, state):
        self.state = state

    async def fetch(self, sql, *args):
        if sql == FINDINGS_SQL:
            return self.state["rows"]
        assert sql == POSTED_SQL
        self.state.setdefault("ledger_asked", []).append(list(args[0]))
        return [{"want_id": w} for w in self.state["ledger"] if w in args[0]]


class _Pool:
    def __init__(self, state):
        self.state = state

    def acquire(self):
        conn = _Conn(self.state)

        class _CM:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *a):
                return False

        return _CM()


def _loop(state, decision="funded"):
    bids = []

    async def bid(want):
        bids.append(want)
        if decision is None:
            return None
        reason = "physiology_unavailable" if decision == "unavailable" else "net_positive"
        return SimpleNamespace(outcome="deferred" if decision == "unavailable" else decision, reason=reason,
                               value_rank=0.8)

    loop = CuriosityEventWants(enabled=True, pool_provider=lambda: _Pool(state), bid=bid, clock=lambda: NOW)
    return loop, bids


def test_plan_ranks_each_recent_finding_against_its_30_days_and_bands_it():
    by = {p["run_id"]: p for p in plan_findings(_rows(), now=NOW, lookback=timedelta(hours=12))}
    assert set(by) == {"new0", "new1"}
    assert by["new0"]["salience"] == 1.0 and by["new0"]["in_band"]       # newer than everything before it
    # new1 (a minute before new0) is ranked against the 40 older ones only: 0.12 is above 0.100 .. 0.119.
    assert by["new1"]["salience"] == 20 / 40 and not by["new1"]["in_band"]
    assert SALIENCE_BAND == 0.9


def test_only_the_band_is_posted_with_its_event_and_salience():
    state = {"rows": _rows(), "ledger": set()}
    loop, bids = _loop(state)
    summary = asyncio.run(loop.tick())
    assert summary["posted"] == 1 and summary["below_band"] == 1
    (w,) = bids
    assert w.family == FAMILY and w.verb == "infer" and w.tier == 2 and w.salience == 1.0
    assert w.event_ref == "curiosity_run:new0" and w.want_id == event_want_id("curiosity_finding", "new0")
    assert w.opened_at == NOW - timedelta(minutes=30)
    assert w.cost is not None and not w.cost.gpu_slot_s_by_lane       # no GPU slot: the thought is Prime's
    # Below band is never retried, and the posted one is never posted twice.
    asyncio.run(loop.tick())
    assert len(bids) == 1
    # The ledger is asked only about in-band findings.
    assert state["ledger_asked"][0] == [event_want_id("curiosity_finding", "new0")]


def test_a_finding_already_in_the_ledger_is_not_reposted_after_a_restart():
    state = {"rows": _rows(), "ledger": {event_want_id("curiosity_finding", "new0")}}
    loop, bids = _loop(state)
    summary = asyncio.run(loop.tick())
    assert summary["already"] == 1 and summary["posted"] == 0 and bids == []


def test_no_answer_and_unpriced_decisions_are_bid_again():
    for decision in (None, "unavailable"):
        state = {"rows": _rows(), "ledger": set()}
        loop, bids = _loop(state, decision=decision)
        asyncio.run(loop.tick())
        asyncio.run(loop.tick())
        assert len(bids) == 2, decision
    assert "reason <> 'physiology_unavailable'" in POSTED_SQL


def test_too_little_history_posts_nothing_and_is_not_retried():
    state = {"rows": [r for r in _rows() if r["run_id"].startswith("new")], "ledger": set()}
    loop, bids = _loop(state)
    assert asyncio.run(loop.tick())["no_history"] == 2 and bids == []
    assert asyncio.run(loop.tick())["already"] == 2


def test_only_philosophical_findings_with_novelty_are_read():
    assert "register = 'philosophical'" in FINDINGS_SQL and "novelty IS NOT NULL" in FINDINGS_SQL
    # A null novelty that reaches the planner anyway is dropped, not ranked.
    rows = _rows() + [{"run_id": "nul", "line": "x", "novelty": None, "created_at": NOW - timedelta(minutes=5)}]
    assert "nul" not in {p["run_id"] for p in plan_findings(rows, now=NOW, lookback=timedelta(hours=12))}


def test_off_means_no_task():
    loop = CuriosityEventWants(enabled=False, pool_provider=lambda: None)
    asyncio.run(loop.start())
    assert loop._task is None


def test_family_is_registered_with_a_window_and_no_lanes():
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parents[3]
    fam = yaml.safe_load((repo / "config/physiology/families.yaml").read_text())["families"][FAMILY]
    assert fam["tier"] == 2 and fam["deferrable_window_s"] == 43200 and fam["lanes"] == []
    organs = yaml.safe_load((repo / "config/integration/organs.yaml").read_text())["organs"]
    assert FAMILY in organs["curiosity"]["families"]


def test_the_replay_eval_fixture_passes_its_band_gate():
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    out = subprocess.run([sys.executable, str(repo / "services/orion-hub/evals/run_curiosity_event_wants_replay.py")],
                         capture_output=True, text=True, env={"PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1"})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PASS" in out.stdout
