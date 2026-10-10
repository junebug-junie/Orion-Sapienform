"""Replay of the 2026-10-09/10 GPU controller incident through the GPU watch.

Real data: fixtures/2026-10-09_actuate_refused.json is every ``actuate_refused``
row in conjourney.gpu_pool_events (155 rows, 04:07Z 10-09 -> 05:51Z 10-10, all
role agent-gpu2, all ``config_unloadable:ValidationError``, one per ~10 min as
the pool retried). Zero such rows exist in the 16 days before. Through the
incident the controller's ``status`` answered ``refused
config_unloadable:ValidationError`` (verified live with
scripts/gpu_pool_actuator_probe.py). Before this patch: zero cards in 25.7 h.

Measures: seconds from the first refusal to the first card, active probe alone
(swept over every probe phase) and with the passive event watch, plus total cards
over the incident; and that the healthy baseline stays silent.
"""
from __future__ import annotations

import json
from pathlib import Path

from orion.gpu_pool.actuator_probe import classify

from app.gpu_watch import ActiveProbeTracker, RefusalWatch, RoleTarget
from app.settings import Settings
from app.stability import REALERT_AFTER_SEC, AlertGate

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "2026-10-09_actuate_refused.json").read_text())
REFUSALS = FIXTURE["generated_at_epoch"]
REASON = FIXTURE["reason"]
TARGETS = {"agent-gpu2": RoleTarget("agent-gpu2", "circe", "circe"), "diffusion": RoleTarget("diffusion", "circe", "circe")}
INTERVAL = float(Settings.model_fields["gpu_probe_interval_sec"].default)


def _replay(*, phase: float, passive: bool, active: bool = True, broken: bool = True) -> list[tuple[float, str]]:
    """Merged timeline of probe cycles (every INTERVAL from REFUSALS[0] + phase) and pool events.
    Returns (time, key) per admitted card."""
    start, end = REFUSALS[0], REFUSALS[-1]
    timeline: list[tuple[float, str]] = []
    if active:
        t = start + phase
        while t <= end:
            timeline.append((t, "probe"))
            t += INTERVAL
    if passive and broken:
        timeline += [(t, "event") for t in REFUSALS]
    timeline.sort()
    tracker, refusals, gate = ActiveProbeTracker(), RefusalWatch(), AlertGate()
    cards: list[tuple[float, str]] = []
    for now, what in timeline:
        if what == "probe":
            alerts = []
            for role, target in TARGETS.items():
                bad = broken and role == "agent-gpu2"
                status = classify("status", [{"status": "refused", "reason": REASON}] if bad
                                  else [{"status": "succeeded"}], role=role)
                digest = classify("digest", [{"status": "refused", "reason": REASON if bad else "profile_not_allowed"}],
                                  role=role)
                alerts += tracker.observe(target, status) + tracker.observe(target, digest)
        else:
            alerts = refusals.observe({"event": "actuate_refused", "role": "agent-gpu2", "reason": REASON}, now, TARGETS)
        cards += [(now, a.key) for a in gate.admit(alerts, now)]
    return cards


def test_fixture_is_the_real_incident() -> None:
    assert len(REFUSALS) == 155 and REASON == "config_unloadable:ValidationError"
    assert 25 * 3600 < REFUSALS[-1] - REFUSALS[0] < 26 * 3600


def test_time_to_first_card_is_within_one_probe_interval_at_every_phase() -> None:
    worst_active = 0.0
    for phase in range(0, int(INTERVAL), 5):
        active_only = _replay(phase=phase, passive=False)
        both = _replay(phase=phase, passive=True)
        assert active_only and both
        worst_active = max(worst_active, active_only[0][0] - REFUSALS[0])
        assert both[0][0] - REFUSALS[0] == 0.0  # the first refused swap pages at once
    print(f"\n10-09 GPU incident: first card active-only worst case {worst_active:.0f}s, "
          f"with pool events 0s (before this patch: never, 25.7 h)")
    assert worst_active <= INTERVAL


def test_whole_incident_is_one_card_per_gate_window_not_155() -> None:
    cards = _replay(phase=0, passive=True)
    span = REFUSALS[-1] - REFUSALS[0]
    assert {key for _, key in cards} == {"gpu_config_unloadable:agent-gpu2"}
    assert len(cards) == int(span // REALERT_AFTER_SEC) + 1
    print(f"cards over the incident: {len(cards)} (155 refusals, {span / 3600:.1f} h)")


def test_healthy_baseline_is_silent() -> None:
    assert _replay(phase=0, passive=True, broken=False) == []
