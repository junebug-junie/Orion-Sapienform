"""Backtest: real events through the real pressure code -> when would Orion sleep?

Fixture: fixtures/sleep_pressure_events_2026-10-09.psv, the four pressure sources
exported live for the 10 days before 2026-10-09 02:00 UTC, one row per (source,
15-minute bucket, key). Columns: source|epoch|dedupe_key|extra (metacog
severity, resonance violation_count, crystallization salience).

Before this change the same week slept on the 6 h minimum-interval clock: 28
sleeps, every gap 6 h, pressure 70-129 against a threshold of 3.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "sleep_pressure_events_2026-10-09.psv"
STEP_SEC = 1800
MIN_GAP_SEC = 6 * 3600
LOOKBACK_SEC = 48 * 3600
DAYS = 7


def _row(source: str, i: int, key: str, extra: str) -> dict:
    if source == "metacog":
        return {"id": f"m{i}", "summary": key, "severity": extra, "trigger_kind": key.split(":")[0],
                "tags": [], "dedupe_key": key}
    if source == "compaction_request":
        return {"request_id": f"r{i}", "theme": key, "reason": "", "dedupe_key": key}
    if source == "resonance":
        return {"alert_id": f"a{i}", "theme_key": key, "violation_count": int(extra or 0), "dedupe_key": key}
    return {"crystallization_id": key, "subject": key, "summary": "", "salience": float(extra or 0.5),
            "tags": [], "dedupe_key": key}


def _events():
    out = []
    for i, line in enumerate(FIXTURE.read_text().splitlines()):
        source, epoch, key, extra = line.split("|")
        out.append((int(epoch), source, _row(source, i, key, extra)))
    return out


def _rows_between(events, lo: int, hi: int) -> dict:
    rows = defaultdict(list)
    for t, source, row in events:
        if lo <= t < hi:
            rows[source].append(row)
    for source in rows:
        rows[source].reverse()  # newest first, as the SQL returns them
    return rows


def _simulate(threshold: float):
    from app.replay import compute_pressure, keyed_candidates, prior_keys

    events = _events()
    end = events[-1][0]
    last = end - DAYS * 86400
    t, sleeps = last, []
    while t < end:
        t += STEP_SEC
        if t - last < MIN_GAP_SEC:
            continue
        keyed = keyed_candidates(_rows_between(events, last, t))
        seen = prior_keys(_rows_between(events, last - LOOKBACK_SEC, last))
        pressure, _, _ = compute_pressure(keyed, seen)
        if pressure >= threshold:
            sleeps.append((t, pressure))
            last = t
    return sleeps


def test_threshold_3_is_no_longer_a_six_hour_clock():
    sleeps = _simulate(3.0)
    gaps_h = sorted((b[0] - a[0]) / 3600 for a, b in zip(sleeps, sleeps[1:]))
    print(f"threshold=3: sleeps/7d={len(sleeps)} gaps_h={[round(g, 1) for g in gaps_h]}")
    assert len(sleeps) < 28 - 7  # the old rule: 28, all at the minimum gap
    assert gaps_h[len(gaps_h) // 2] > 6.5  # median gap is not the clock
    assert max(gaps_h) > 12


def test_orion_still_sleeps_at_least_daily_on_average():
    """The dangerous failure is pressure pinned near 0: Orion would stop sleeping."""
    sleeps = _simulate(3.0)
    gaps_h = [(b[0] - a[0]) / 3600 for a, b in zip(sleeps, sleeps[1:])]
    assert len(sleeps) >= DAYS
    assert max(gaps_h) < 72
