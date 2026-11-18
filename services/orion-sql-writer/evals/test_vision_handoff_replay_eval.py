"""Eval: a synthetic street through the handoff replay (same pure chain as the loop).

Two weeks of sidewalk walkers leave the walkway on its left edge and reach the
doorbell's right edge after a gap of ~1.7 frame widths at their own speed; the
doorbell watches every window (a self-look each time). Checks:

1. After 10 met the band switches to the learned one, the learned gap
   recovers the true gap (median within 10%, spread well under 50%), the
   10-90% band covers 70-90% of arrivals, and early / late arrivals are
   found (none graded missed).
2. Met p-values are roughly uniform (the calm state): KS distance small, and
   a quiet hour (no late / missed) produces zero attention-gate passes beyond
   chance.
3. A doorbell window nobody watched is unscorable, never missed.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))

import run_vision_handoff_replay_eval as ev  # noqa: E402
from app import vision_handoff as vh  # noqa: E402
from orion.vision.topology import load_topology  # noqa: E402

TZ = ZoneInfo("America/Denver")
T0 = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)
TRUE_G = 1.7


def _story(sid, stream, start, end, **kw):
    base = dict(story_id=sid, stream_id=stream, kind="person", started_at=start, ended_at=end, dir_lateral=-1,
                dir_depth=0, exit_edge="inside", entry_edge="inside", exit_foot_x=0.3, entry_foot_x=0.8,
                exit_speed=0.05, created_at=end + timedelta(seconds=50))
    base.update(kw)
    return vh.StoryFacts(**base)


def _street(n=80, watched=True, seed=11):
    rng = random.Random(seed)
    stories, census = [], {"walkway": [], "doorbell": []}
    for i in range(n):
        t = T0 + timedelta(hours=4 * i, minutes=rng.randint(0, 50))
        speed = rng.uniform(0.03, 0.08)
        foot = rng.uniform(0.1, 0.4)
        stories.append(_story(f"w{i}", "walkway", t - timedelta(seconds=15), t, exit_speed=speed, exit_foot_x=foot))
        g = TRUE_G * rng.lognormvariate(0, 0.12)
        arrive = t + timedelta(seconds=(foot + g) / speed)
        stories.append(_story(f"d{i}", "doorbell", arrive, arrive + timedelta(seconds=8), entry_edge="right",
                              entry_foot_x=0.95))
        if watched:
            census["doorbell"].append((t.timestamp(), t.timestamp() + 600))
    return stories, census


def test_learned_gap_recovers_the_true_gap_and_met_p_is_calm() -> None:
    stories, census = _street()
    cfg = vh.HandoffConfig()
    graded, overlaps, unannounced, learned = ev.replay(stories, census, [], {}, {}, load_topology(), cfg, TZ)
    out = ev.summarize(graded, overlaps, unannounced, learned, TZ, cfg, True)
    w2d = out["outcomes"]["walkway_to_door"]
    # Spec acceptance: the 80% band covers 70-90% of arrivals; nobody who arrived is "missed".
    assert w2d.get("missed", 0) == 0 and 0.7 <= w2d["met"] / sum(w2d.values()) <= 0.9, w2d
    assert out["band_source"].get("learned", 0) > 0
    g = out["learned_g"]["walkway_to_door/person"]
    assert abs(g["median"] - TRUE_G) / TRUE_G < 0.1 and g["spread_over_median"] < 0.5
    assert out["met_p"]["ks_uniform"] < 0.25
    assert out["attention_gate"]["quiet_hour_passes"] <= max(1, 0.05 * out["met_p"]["n"])
    assert out["verdict"].startswith("PASS"), out["verdict"]


def test_unwatched_doorbell_windows_are_unscorable_never_missed() -> None:
    stories, census = _street(n=12, watched=False)
    stories = [s for s in stories if s.stream_id == "walkway"]   # and nobody arrived
    cfg = vh.HandoffConfig()
    graded, overlaps, unannounced, learned = ev.replay(stories, census, [], {}, {}, load_topology(), cfg, TZ)
    out = ev.summarize(graded, overlaps, unannounced, learned, TZ, cfg, True)
    assert out["outcomes"]["walkway_to_door"] == {"unscorable": 12}
    assert out["doorbell_target"]["missed"] == 0 and out["verdict"].startswith("TOO THIN")
