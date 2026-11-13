"""episodes + specificity on in-memory rows (no DB): a planted preference is found, none is not invented."""
from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measure_personality_situation_specificity import episodes, render, specificity  # noqa: E402

T0 = datetime(2026, 10, 17, tzinfo=timezone.utc)
BURST = timedelta(minutes=30)


def _row(node, lane, at, arousal, company=False, phase="morning"):
    return {"disposition_id": node, "lane": lane, "occurred_at": at,
            "meta": {"situation": {"arousal": arousal, "company": company, "day_phase": phase}}}


def _world(planted: bool, n_node=15, seed=1):
    rng = random.Random(seed)
    rows = []
    # Background: 3 days, reverie hourly on other nodes, arousal 50/50 idle/strained.
    for h in range(72):
        rows.append(_row(f"bg{h % 5}", "reverie", T0 + timedelta(hours=h), rng.choice(["idle", "strained"])))
    for i in range(n_node):
        at = T0 + timedelta(hours=i * 4, minutes=7)
        rows.append(_row("theme", "reverie", at, "strained" if planted else rng.choice(["idle", "strained"])))
    return rows


def _theme(results, dim="arousal"):
    return next(r for r in results if r["node"] == "theme" and r["dimension"] == dim)


def test_bursts_collapse_to_one_episode_per_lane():
    rows = [_row("d1", "reverie", T0 + timedelta(minutes=m), "idle") for m in (0, 5, 20, 90)]
    rows.append(_row("d1", "journal", T0 + timedelta(minutes=1), "idle"))
    eps = episodes(rows, burst=BURST)
    assert sorted((e["lane"], e["at"].minute) for e in eps) == [("journal", 1), ("reverie", 0), ("reverie", 30)]


def test_planted_preference_is_found_at_live_sample_sizes():
    res = specificity(episodes(_world(True), burst=BURST), min_episodes=8, permutations=500)
    r = _theme(res)
    assert r["testable"] and r["over"] == "strained" and r["p"] < 0.01


def test_no_preference_is_not_invented():
    ps = [_theme(specificity(episodes(_world(False, seed=s), burst=BURST), min_episodes=8,
                             permutations=200, seed=s))["p"] for s in range(20)]
    assert sum(p < 0.05 for p in ps) <= 3  # ~1 expected by chance


def test_strata_that_never_vary_make_a_node_untestable_not_null():
    rows = [_row("theme", "dream", T0 + timedelta(hours=h), "idle") for h in range(20)]
    (r, *_) = [x for x in specificity(episodes(rows, burst=BURST), min_episodes=8, permutations=50)
               if x["dimension"] == "arousal"]
    assert r["testable"] is False and r["informative"] == 0
    assert "untestable 3" in render(specificity(episodes(rows, burst=BURST), min_episodes=8, permutations=50),
                                    {}, alpha=0.1)
