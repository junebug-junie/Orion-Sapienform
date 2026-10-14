"""Replay eval: 7 days of real substrate_node_prediction_error_history (exported
2026-10-14 with scripts/analysis/replay_self_signal_episodes.py --write-fixture,
internal nodes only) folded through the live code path:
world_first.node_candidate -> judge_candidate -> self_signal_episode.step_signal.

What it pins (numbers are from this fixture; see the PR report):
- volume: ~9.5 episodes/day total, never more than 15 on one day;
- without the dwell (open on the first hot poll) it would be >100/day -- the
  dwell is what makes "high for itself" an episode rather than the top decile
  of every signal's week;
- the 10-08 06:00Z GPU shed (923 gpu_pool 'unavailable' refusals that hour)
  opens a bus_synaptic episode within minutes;
- the 10-07 cabinet heating run (06:00-15:00Z, ahead of the first GPU-pool
  refusal at 14:05Z) opens cabinet episodes;
- placeholders never open (harness_closure); every close is calm/faded,
  never silent, on a history with no outage of the table itself.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "analysis")]

from replay_self_signal_episodes import (  # noqa: E402
    check_fast_magnitude,
    episodes_between,
    replay,
    rows_from_csv,
)

FIXTURE = Path(__file__).parent / "fixtures" / "self_signal_pe_history_2026-10-07_2026-10-14.csv.gz"


@pytest.fixture(scope="module")
def rows():
    return rows_from_csv(FIXTURE)


@pytest.fixture(scope="module")
def result(rows):
    return replay(rows)


def _days(r) -> float:
    return (r["end"] - r["start"]) / 86400.0


def test_replay_magnitude_matches_the_real_magnitude_function(rows):
    assert check_fast_magnitude(rows, samples=15) == []


def test_daily_volume(result):
    total = len(result["episodes"])
    per_day = {d: sum(c.values()) for d, c in result["per_day"].items()}
    print("per-day episode opens:", per_day)
    print("per-signal:", {k: v["episodes"] for k, v in result["per_signal"].items()})
    assert 5.0 <= total / _days(result) <= 15.0, total
    assert max(per_day.values()) <= 15, per_day


def test_without_the_dwell_every_signal_opens_constantly(rows, result):
    no_dwell = replay(rows, open_dwell_sec=0.0)
    assert len(no_dwell["episodes"]) / _days(no_dwell) > 100
    assert len(no_dwell["episodes"]) > 8 * len(result["episodes"])


def test_placeholder_never_opens(result):
    hc = result["per_signal"]["node:substrate.harness_closure"]
    assert hc["episodes"] == 0 and set(hc["observations"]) == {"unknown"}


def test_closes_are_returns_not_silence(result):
    reasons = Counter(e["close_reason"] for e in result["episodes"])
    assert set(reasons) <= {"calm", "faded", "still_open"}, reasons


def test_10_08_gpu_shed_opens_an_episode(result):
    eps = episodes_between(result, "2026-10-08T05:55:00+00:00", "2026-10-08T06:30:00+00:00")
    assert any(e["signal_id"] == "node:substrate.bus_synaptic" for e in eps), eps


def test_10_07_cabinet_heating_opens_cabinet_episodes(result):
    eps = episodes_between(
        result, "2026-10-07T06:00:00+00:00", "2026-10-07T15:30:00+00:00", "node:substrate.cabinet"
    )
    assert len(eps) >= 4, eps


def test_episode_durations_are_bounded(result):
    durs = [e["duration_sec"] for e in result["episodes"] if e["duration_sec"] is not None]
    assert durs and min(durs) >= 600.0  # at least the calm gap
    assert max(durs) < 8 * 3600.0
