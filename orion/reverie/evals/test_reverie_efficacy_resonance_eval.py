"""Phase H eval harness — efficacy + resonance over synthetic corpora.

This is the evidence the plan names as "what licenses turning G's hot gate on":
the ouroboros tripwire provably fires on a runaway loop and stays silent on a
healthy one, and the pre/post recall + discharge metrics compute honestly.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from orion.reverie.efficacy import pressure_discharge_rate, recall_delta
from orion.reverie.resonance import ThemeEvent, detect_resonance

NOW = datetime(2026, 7, 6, tzinfo=timezone.utc)
REFRACTORY = 900.0


def _runaway(theme: str, n: int, gap_sec: float) -> list[ThemeEvent]:
    return [ThemeEvent(theme, NOW + timedelta(seconds=gap_sec * i)) for i in range(n)]


def _healthy(theme: str, n: int, gap_sec: float) -> list[ThemeEvent]:
    # gaps strictly exceed the refractory bound → damped
    return [ThemeEvent(theme, NOW + timedelta(seconds=gap_sec * i)) for i in range(n)]


def test_detector_fires_on_synthetic_runaway_loop():
    events = _runaway("loop:runaway", n=6, gap_sec=120)  # all inside 900s
    alert = detect_resonance(events, refractory_sec=REFRACTORY)
    assert alert is not None, "tripwire must fire on a runaway loop"
    assert alert.theme_key == "loop:runaway"
    assert alert.violation_count == 5  # 6 occurrences → 5 breaching gaps


def test_detector_silent_on_healthy_loop():
    events = _healthy("loop:calm", n=6, gap_sec=1200)  # every gap > refractory
    assert detect_resonance(events, refractory_sec=REFRACTORY) is None


def test_detector_isolates_runaway_amid_healthy_noise():
    corpus: list[ThemeEvent] = []
    rng = random.Random(11)
    # a pile of well-damped themes
    for k in range(8):
        corpus += _healthy(f"loop:calm-{k}", n=rng.randint(2, 5), gap_sec=1000 + rng.randint(0, 500))
    # one runaway hidden inside
    corpus += _runaway("loop:hot", n=4, gap_sec=90)
    rng.shuffle(corpus)
    alert = detect_resonance(corpus, refractory_sec=REFRACTORY)
    assert alert is not None
    assert alert.theme_key == "loop:hot"


def test_recall_metrics_recorded_pre_post_compaction():
    # Compaction should shrink the graph and not regress latency.
    d = recall_delta(
        latency_ms_before=140.0, latency_ms_after=95.0,
        graph_size_before=12000, graph_size_after=10800,
    )
    assert d.graph_size_delta < 0, "compaction should shrink the graph"
    assert d.latency_ms_delta <= 0, "recall should not get slower"


def test_pressure_discharge_rate_over_chain_batch():
    # 4 of 5 synthetic chains discharged their spawning pressure.
    before = [0.9, 0.85, 0.7, 0.6, 0.8]
    after = [0.3, 0.9, 0.2, 0.1, 0.4]
    rate = pressure_discharge_rate(before, after)
    assert rate == 4 / 5
