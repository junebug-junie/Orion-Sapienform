"""Deterministic unit tests for measure_origination_gate.py.

No DB. The replay layer calls the REAL extract_tensions_from_self_state and
OriginationEngine (both pure, no I/O), so it is exercised directly with
synthetic SelfStateV1 fixtures -- same fixture pattern as
orion/spark/concept_induction/tests/test_endogenous_origination_wiring.py.
The sweep/summary layer is exercised separately on synthetic ReplayTick
lists, no engine involved.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_origination_gate.py"
_spec = importlib.util.spec_from_file_location("measure_origination_gate", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_origination_gate"] = mod
_spec.loader.exec_module(mod)

from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1  # noqa: E402

UTC = timezone.utc
BASE = datetime(2026, 7, 1, 0, 0, 0, tzinfo=UTC)


def _self_state_payload(*, agency=0.9, intensity=0.0, dwell=40, unresolved=None, coherence=0.5,
                        trajectory=None, ts=BASE) -> dict:
    dims = {
        "agency_readiness": SelfStateDimensionV1(dimension_id="agency_readiness", score=agency, confidence=1.0),
        "coherence": SelfStateDimensionV1(dimension_id="coherence", score=coherence, confidence=1.0),
    }
    state = SelfStateV1(
        self_state_id=str(uuid4()), generated_at=ts,
        source_field_tick_id="ft", source_field_generated_at=ts,
        source_attention_frame_id="af", source_attention_generated_at=ts,
        overall_intensity=intensity, overall_confidence=0.7, dimensions=dims,
        # Default trajectory mirrors test_endogenous_origination_wiring.py's own
        # fixture (drift must be nonzero for P to have any chance of crossing
        # threshold -- an empty trajectory silently zeroes the drift term).
        dimension_trajectory=trajectory if trajectory is not None else {"coherence": 0.9, "uncertainty": 0.8},
        attention_dwell_ticks=dwell, unresolved_pressures=unresolved or [],
    )
    return state.model_dump(mode="json")


def test_replay_quiet_stream_eventually_fires() -> None:
    """A repeated quiet, high-dwell/high-agency stream (no coherence deltas to
    mint exogenous tensions) should let origination fire within the ring
    window -- mirrors test_endogenous_origination_wiring.py's
    test_worker_flag_on_quiet_fires, replayed through this script's own code
    path instead of ConceptWorker directly."""
    rows = [
        (BASE + timedelta(seconds=5 * i), _self_state_payload(agency=0.9, dwell=40, coherence=0.5, ts=BASE + timedelta(seconds=5 * i)))
        for i in range(10)
    ]
    ticks = mod.replay_origination(rows)
    assert len(ticks) == 10
    assert any(t.fired_live_config for t in ticks), "quiet high-dwell/agency stream should fire at least once"


def test_replay_busy_stream_never_fires_at_live_floor() -> None:
    """A stream with a real coherence swing every tick mints an exogenous
    tension every tick (contradiction/distress-class), so Gate 1
    (exogenous_tension_count <= 0) should block every fire at the live
    floor, even though the same ticks may show a real P signal.

    Row 0 is a warm-up tick (no previous_self_state exists yet, so
    extract_tensions_from_self_state cannot compute a delta -- exo=0 by
    construction, same edge case the live worker's very first tick ever has).
    Only rows[1:], which all have a real prior state to swing against, are
    asserted never to fire. Coherence is monotonically decreasing (not
    alternating) because tensions.py's contradiction block only fires on a
    *drop* (coherence_delta < 0) -- an alternating swing would only mint a
    tension on every other tick, undermining "exogenous every tick"."""
    rows = []
    for i in range(11):
        ts = BASE + timedelta(seconds=5 * i)
        coherence = max(0.05, 0.95 - 0.08 * i)  # strictly decreasing -> contradiction tension every step
        rows.append((ts, _self_state_payload(agency=0.9, dwell=40, coherence=coherence, ts=ts)))
    ticks = mod.replay_origination(rows)
    assert len(ticks) == 11
    assert not any(t.fired_live_config for t in ticks[1:]), "busy stream must not fire under floor=0 once a prior state exists"
    # But a materially loosened floor should let some of these through --
    # this is exactly the floor-sweep question the script exists to answer.
    floor_sweep, _ = mod.sweep_gate(ticks[1:], floors=(0, 1, 2))
    assert floor_sweep[0] == 0


def test_sweep_gate_respects_cooldown() -> None:
    """Two ticks both eligible (P above threshold, exo<=floor) within the
    cooldown window should only fire once; a third tick after cooldown
    elapses should fire again."""
    ticks = [
        mod.ReplayTick(generated_at=BASE, exogenous_tension_count=0, drift=0.9, dwell=0.9, agency=0.9, P=0.9, fired_live_config=False),
        mod.ReplayTick(generated_at=BASE + timedelta(seconds=100), exogenous_tension_count=0, drift=0.9, dwell=0.9, agency=0.9, P=0.9, fired_live_config=False),
        mod.ReplayTick(generated_at=BASE + timedelta(seconds=1000), exogenous_tension_count=0, drift=0.9, dwell=0.9, agency=0.9, P=0.9, fired_live_config=False),
    ]
    floor_sweep, _ = mod.sweep_gate(ticks, floors=(0,), thresholds=(), cooldown_sec=900.0)
    assert floor_sweep[0] == 2  # tick 1 fires, tick 2 suppressed by cooldown, tick 3 (900s later) fires again


def test_sweep_gate_threshold_independent_of_floor() -> None:
    """A tick with exogenous_tension_count=5 should never fire regardless of
    threshold when floor stays at the live default (0); the threshold sweep
    (which holds floor at the live value) should report 0 fires at every
    threshold for this tick."""
    ticks = [
        mod.ReplayTick(generated_at=BASE, exogenous_tension_count=5, drift=0.9, dwell=0.9, agency=0.9, P=0.99, fired_live_config=False),
    ]
    _, threshold_sweep = mod.sweep_gate(ticks, floors=(), thresholds=(0.55, 0.1))
    assert all(count == 0 for count in threshold_sweep.values())


def test_exogenous_count_histogram_buckets_overflow() -> None:
    ticks = [
        mod.ReplayTick(generated_at=BASE, exogenous_tension_count=n, drift=0.0, dwell=0.0, agency=0.0, P=0.0, fired_live_config=False)
        for n in (0, 0, 3, 15)
    ]
    hist = mod.exogenous_count_histogram(ticks, cap=10)
    assert hist[0] == 2
    assert hist[3] == 1
    assert hist[11] == 1  # 15 > cap=10 buckets into cap+1


def test_frac_le_and_frac_p_ge() -> None:
    ticks = [
        mod.ReplayTick(generated_at=BASE, exogenous_tension_count=0, drift=0.0, dwell=0.0, agency=0.0, P=0.6, fired_live_config=False),
        mod.ReplayTick(generated_at=BASE, exogenous_tension_count=3, drift=0.0, dwell=0.0, agency=0.0, P=0.1, fired_live_config=False),
    ]
    assert mod.frac_le(ticks, 0) == 0.5
    assert mod.frac_p_ge(ticks, 0.55) == 0.5


def test_summarize_empty() -> None:
    dist = mod.summarize([])
    assert dist.count == 0
    assert dist.median is None


def test_replay_skips_malformed_rows_without_raising() -> None:
    rows = [(BASE, {"not": "a valid self state"})]
    ticks = mod.replay_origination(rows)
    assert ticks == []
