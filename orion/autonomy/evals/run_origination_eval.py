"""Eval: endogenous origination in quiet vs busy periods (spec Step 1).

Replays two scripted periods through the real OriginationEngine and asserts the
world-wins invariant:
  - QUIET (no exogenous tensions): Orion originates >=1 endogenous want within one
    cooldown span.
  - BUSY (exogenous tensions every tick): zero endogenous origination.

Reports origination rate and suppression ratio. No bus, no Docker.
Run: python orion/autonomy/evals/run_origination_eval.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from orion.autonomy.endogenous_origination import OriginationConfig, OriginationEngine
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

T0 = datetime(2026, 7, 8, 0, 0, 0, tzinfo=timezone.utc)


def _self_state(ts: datetime, *, drift: float, agency: float, intensity: float,
                dwell: int) -> SelfStateV1:
    return SelfStateV1(
        self_state_id=f"ss-{ts.isoformat()}", generated_at=ts,
        source_field_tick_id="ft", source_field_generated_at=ts,
        source_attention_frame_id="af", source_attention_generated_at=ts,
        overall_intensity=intensity, overall_confidence=0.7,
        dimensions={"agency_readiness": SelfStateDimensionV1(
            dimension_id="agency_readiness", score=agency, confidence=1.0)},
        dimension_trajectory={"coherence": drift, "uncertainty": drift * 0.9},
        attention_dwell_ticks=dwell, unresolved_pressures=[],
    )


def _run_period(*, exogenous_per_tick: int, ticks: int, cadence_sec: int) -> tuple[int, int]:
    cfg = OriginationConfig(window=8, threshold=0.55, cooldown_sec=900.0, mag_cap=0.5)
    eng = OriginationEngine(cfg)
    fired = 0
    suppressed = 0
    ts = T0
    for _ in range(ticks):
        # High internal churn: high drift + agency surplus (quiet mind wants).
        eng.observe(_self_state(ts, drift=0.9, agency=0.9, intensity=0.05, dwell=60))
        t = eng.maybe_originate(exogenous_tension_count=exogenous_per_tick, now=ts)
        if t is not None:
            fired += 1
        elif eng.last_signal.get("P", 0.0) >= cfg.threshold:
            suppressed += 1  # would have fired but for a gate
        ts = ts + timedelta(seconds=cadence_sec)
    return fired, suppressed


def run() -> int:
    # QUIET: 20 ticks, 60s apart (~20 min), no exogenous input.
    q_fired, q_suppressed = _run_period(exogenous_per_tick=0, ticks=20, cadence_sec=60)
    # BUSY: same, but 3 exogenous tensions every tick (world present).
    b_fired, b_suppressed = _run_period(exogenous_per_tick=3, ticks=20, cadence_sec=60)

    checks = [
        ("QUIET originates >=1 want", q_fired >= 1, f"quiet_fired={q_fired}"),
        ("BUSY originates 0 (world-wins)", b_fired == 0, f"busy_fired={b_fired}"),
        ("BUSY suppression observed", b_suppressed >= 1, f"busy_suppressed={b_suppressed}"),
    ]

    print("\n=== Endogenous origination eval ===")
    print(f"QUIET: fired={q_fired}  suppressed_by_gate={q_suppressed}")
    print(f"BUSY : fired={b_fired}  suppressed_by_exogenous={b_suppressed}")
    print("checks:")
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  ({detail})")
        ok = ok and passed
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
