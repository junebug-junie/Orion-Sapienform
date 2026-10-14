"""Replay eval: 7 days of real stored history through the self-signal -> outcome association
(spec 2026-10-07 section E), checked truth-to-rows by an independent recount.

Inputs (both exported read-only from the live DB on 2026-10-14):
- fixtures/self_signal_pe_history_2026-10-07_2026-10-14.csv.gz -- the PE history the episode
  replay eval already uses (internal nodes only);
- fixtures/self_signal_outcomes_2026-10-06_2026-10-14.jsonl -- every harness run's
  (created_at, compliance_verdict, response_repair_ran) and transport_baseline_hourly summed per
  hour (gpu_pool_wait keys kept separate so the exclusion is exercised).

What it pins (live numbers, see the PR report):
- every association's counts and interval re-derive from the hour rows
  (scripts/analysis/self_signal_outcome_truth.py, not the reducer);
- on this week nothing clears min_hours=24 with an interval excluding 1, so nothing renders --
  the honest result for a week of data;
- forcing min_hours down to 1 still renders nothing a recount cannot back;
- circular pairs stay excluded; chat (event-written, never sustained) has no high hour.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "analysis"), str(HERE.parent)]

import orion.attention.self_signal_outcomes as sso  # noqa: E402
from app.self_signal_outcome_rollup import body_signal_ids  # noqa: E402
from replay_self_signal_episodes import rows_from_csv  # noqa: E402
from self_signal_outcome_truth import check_association, check_clause  # noqa: E402

PE = HERE / "fixtures" / "self_signal_pe_history_2026-10-07_2026-10-14.csv.gz"
OUT = HERE / "fixtures" / "self_signal_outcomes_2026-10-06_2026-10-14.jsonl"
COMPUTED = datetime(2026, 10, 14, 4, 0, tzinfo=timezone.utc)


def _ts(x: float) -> datetime:
    return datetime.fromtimestamp(float(x), tz=timezone.utc)


@pytest.fixture(scope="module")
def replay():
    hist: dict[str, list] = {}
    for node, t, v in rows_from_csv(PE):
        hist.setdefault(node, []).append((_ts(t), float(v)))
    turns, rpc = [], []
    for line in OUT.read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "turn":
            turns.append((_ts(r["t"]), r["verdict"], r["repaired"]))
        else:
            rpc.append((_ts(r["hour"]), r["key"], int(r["ok"]), int(r["to"])))
    first = min(r[0][0] for r in hist.values())
    last = max(r[-1][0] for r in hist.values())
    hours, h = [], sso.hour_floor(first) + sso.HOUR
    while h + sso.HOUR <= last:
        hours.append(h)
        h += sso.HOUR
    t_by_h, r_by_h = sso.turn_outcome_hours(turns), sso.rpc_outcome_hours(rpc)
    signals = body_signal_ids(hist)
    per = {s: sso.classify_signal_hours(s, hist[s], hours) for s in signals}
    rows = [
        sso.build_hour_row(
            h, signals=[per[s][h] for s in signals],
            outcomes=list(t_by_h.get(h, [])) + ([r_by_h[h]] if h in r_by_h else []),
            computed_at=COMPUTED, history_hours=(h - first).total_seconds() / 3600,
        )
        for h in hours
    ]
    return rows, signals, per, hours


def test_replay_covers_the_week_and_only_body_signals(replay) -> None:
    rows, signals, per, hours = replay
    assert len(hours) >= 160
    assert "node:substrate.harness_closure" not in signals  # placeholder
    assert "node:substrate.perception" not in signals  # the world
    assert set(signals) == {
        "node:substrate.biometrics", "node:substrate.bus_synaptic", "node:substrate.cabinet",
        "node:substrate.chat", "node:substrate.codebase", "node:substrate.execution", "node:substrate.route",
    }


def test_high_hours_are_the_sustained_minority(replay) -> None:
    _, signals, per, hours = replay
    for s in signals:
        conds = [per[s][h].condition for h in hours]
        assert conds.count("high") < 0.25 * len(hours), s  # never "most hours are high"
    chat = [per["node:substrate.chat"][h].condition for h in hours]
    assert chat.count("high") == 0 and chat.count("unknown") > 40  # quiet hours are unknown, not calm


def test_every_association_recounts_from_the_hour_rows(replay) -> None:
    rows = replay[0]
    for min_hours in (24, 1):
        assocs = sso.associate_all(rows, computed_at=COMPUTED, min_hours=min_hours)
        assert len(assocs) == 7 * 3
        fails = [f for a in assocs for f in check_association(a, rows)]
        by_id = {a.association_id: a for a in assocs}
        fails += [f for a in assocs if a.clause for f in check_clause(a.clause, by_id)]
        assert fails == []


def test_this_week_nothing_clears_so_nothing_renders(replay) -> None:
    assocs = sso.associate_all(replay[0], computed_at=COMPUTED, min_hours=24)
    assert [a for a in assocs if a.clause] == []
    verdicts = {(a.signal_id.removeprefix("node:substrate."), a.outcome): a.verdict for a in assocs}
    assert verdicts[("execution", "turn_failed")] == "excluded_circular"
    assert verdicts[("bus_synaptic", "rpc_timeout")] == "excluded_circular"
    assert verdicts[("chat", "response_repair")] == "excluded_circular"
    assert all(v in ("below_min_n", "excluded_circular", "no_clear_difference") for v in verdicts.values())


def test_checker_has_teeth(replay) -> None:
    rows = replay[0]
    a = next(x for x in sso.associate_all(rows, computed_at=COMPUTED) if x.n_high and x.n_normal)
    assert check_association(a.model_copy(update={"events_high": a.events_high + 1}), rows)
    assert check_association(a.model_copy(update={"verdict": "tends_more", "clause": "x"}), rows)
    fake = a.model_copy(update={
        "verdict": "tends_more", "association_id": "ssa-0000000000",
        "clause": "when this is high, my next hour had a failed turn 9.9x as often "
                  "(90% of 30 high hours vs 10% of 60 normal) [assoc:ssa-0000000000]",
    })
    assert check_clause(fake.clause, {fake.association_id: fake})
    assert check_clause(fake.clause, {})  # cites a row that does not exist
