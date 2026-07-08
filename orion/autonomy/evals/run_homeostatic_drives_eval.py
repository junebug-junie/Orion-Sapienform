"""End-to-end eval for homeostatic drives (spec/plan Task 8).

Replays a synthetic stream through the REAL pipeline (deviation gate ->
signal->tension adapter -> rate limiter -> tick attribution -> leaky
DriveEngine) and asserts four runtime properties:

  A. The 55/s scene_state flood contributes ZERO tensions.
  B. A real biometrics strain drop + an injected failure mint tensions.
  C. Pressures differentiate and rest toward zero in a quiet span (no 0.731 pin).
  D. dominant_drive reflects injected events (not alphabetical "autonomy", not
     constant None).

No bus, no Docker — pure replay. Run:
    python orion/autonomy/evals/run_homeostatic_drives_eval.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from orion.autonomy.deviation_gate import DeviationGate
from orion.autonomy.signal_drive_map import load_signal_drive_map
from orion.autonomy.signal_tension import failure_to_tension, signal_to_tension
from orion.autonomy.tension_ratelimit import TensionRateLimiter
from orion.core.schemas.drives import TensionEventV1
from orion.signals.models import OrganClass, OrionSignalV1
from orion.spark.concept_induction.drive_attribution import (
    compute_tick_attribution,
    dominant_drive_from_attribution,
)
from orion.spark.concept_induction.drives import DRIVE_KEYS, DriveEngine, DriveMathConfig

T0 = datetime(2026, 7, 8, 0, 0, 0, tzinfo=timezone.utc)


def _sig(kind: str, dims: dict, sid: str) -> OrionSignalV1:
    now = datetime.now(timezone.utc)
    return OrionSignalV1(
        signal_id=sid, organ_id="eval", organ_class=OrganClass.exogenous,
        signal_kind=kind, dimensions=dims, source_event_id=sid,
        observed_at=now, emitted_at=now,
    )


def run() -> int:
    gate = DeviationGate(alpha=0.1, z_threshold=1.5, sigma_floor=0.02, impulse_k=0.25, warmup=5)
    sdm = load_signal_drive_map()
    rl = TensionRateLimiter(cap=3, window_sec=60.0)
    engine = DriveEngine(DriveMathConfig(decay_tau_sec=300.0, leaky_math_enabled=True))

    pressures = {k: 0.0 for k in DRIVE_KEYS}
    activations = {k: False for k in DRIVE_KEYS}
    prev_ts = T0

    flood_signals_seen = 0
    flood_tensions = 0
    total_tensions = 0
    dominant_hits: dict[str, int] = {}
    trajectory: list[tuple[int, dict]] = []

    # 20-minute timeline at 1s. hrv steady ~0.8, drops during minutes 5-7,
    # recovers. scene_state floods at 55/s throughout. A failure fires at t=420s.
    for t in range(0, 1200):
        ts = T0 + timedelta(seconds=t)
        candidates: list[TensionEventV1] = []

        # scene_state flood: 55 unmapped signals every second.
        for i in range(55):
            flood_signals_seen += 1
            ft = signal_to_tension(_sig("scene_state", {"salience": 0.9, "confidence": 0.9},
                                         f"scene-{t}-{i}"), gate, sdm)
            if ft is not None:
                flood_tensions += 1
                candidates.append(ft)

        # biometrics hrv: steady 0.8, real dip during the strain window.
        hrv = 0.8
        if 300 <= t < 420:
            hrv = 0.45  # sustained strain
        bt = signal_to_tension(_sig("biometrics_state", {"hrv_level": hrv, "confidence": 0.9},
                                     f"bio-{t}"), gate, sdm)
        if bt is not None:
            candidates.append(bt)

        # injected failure at t=600.
        if t == 600:
            xt = failure_to_tension(severity=0.9, sdm=sdm, channel="orion:system:error",
                                    summary="exec_step_failed")
            if xt is not None:
                candidates.append(xt)

        kept = rl.bounded(candidates, now=float(t))
        total_tensions += len(kept)

        attribution = compute_tick_attribution(kept)
        lead = kept[0] if kept else None
        dom = dominant_drive_from_attribution(attribution, lead_tension=lead)
        if dom is not None:
            dominant_hits[dom] = dominant_hits.get(dom, 0) + 1

        pressures, activations = engine.update(
            previous_pressures=pressures, previous_activations=activations,
            tensions=kept, now=ts, previous_ts=prev_ts,
        )
        prev_ts = ts
        if t in (299, 419, 700, 1199):
            trajectory.append((t, {k: round(v, 4) for k, v in pressures.items()}))

    quiet_pressures = trajectory[-1][1]  # t=1199, long after all events

    # Assertions.
    checks = []
    checks.append(("A flood->0 tensions", flood_tensions == 0,
                   f"flood_signals={flood_signals_seen} flood_tensions={flood_tensions}"))
    checks.append(("B events mint tensions", total_tensions > 0,
                   f"total_tensions={total_tensions}"))
    distinct = len({round(v, 3) for v in quiet_pressures.values()})
    checks.append(("C pressures differentiate (not uniform pin)", distinct > 1,
                   f"quiet={quiet_pressures}"))
    rests = all(v < 0.2 for v in quiet_pressures.values())
    checks.append(("C rest toward zero in quiet span", rests, f"quiet={quiet_pressures}"))
    not_alpha = dominant_hits.get("autonomy", 0) < sum(dominant_hits.values() or [1])
    event_drives = {"capability", "continuity", "coherence"} & set(dominant_hits)
    checks.append(("D dominant reflects events (not alphabetical autonomy)",
                   bool(event_drives) and not_alpha, f"dominant={dominant_hits}"))

    print("\n=== Homeostatic Drives eval ===")
    print(f"flood signals seen : {flood_signals_seen}")
    print(f"flood tensions     : {flood_tensions}  (must be 0)")
    print(f"total tensions kept: {total_tensions}")
    print(f"dominant histogram : {dominant_hits}")
    print("pressure trajectory:")
    for t, p in trajectory:
        print(f"  t={t:>4}s  {p}")
    print("\nchecks:")
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  ({detail})")
        ok = ok and passed
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
