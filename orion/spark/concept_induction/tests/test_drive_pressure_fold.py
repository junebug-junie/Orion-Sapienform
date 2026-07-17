"""O2 (event-rate/decay mismatch fix): DriveEngine.update() is called once per
raw bus event (~13/min, ~4.6s apart), but decay_tau_sec=1800.0 means decay
between consecutive calls is negligible (~0.3%) -- repeated same-direction
impulses converge pressure toward 1.0 within seconds regardless of tau,
reproducing the cpu/gpu_pressure saturation bugs (PRs #1108-1111) from a
different mechanism (see docs/superpowers/pr-reports/2026-07-16-drive-economy-
desaturation-o1-o4-pr.md and orion/autonomy/drives_and_autonomy_retrospective.md
Section 5a).

These tests exercise ConceptWorker._update_drive_pressures directly: the new
seam that throttles how often NEW tension impulses get folded into the
persisted drive-pressure integrator (at most once per _DRIVE_FOLD_INTERVAL_SEC
== 900.0s) while every event still gets a fresh decay-only projection for
display/audit/goal-proposal.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.drives import TensionEventV1
from orion.spark.concept_induction.bus_worker import (
    _MAX_PENDING_DRIVE_TENSIONS,
    ConceptWorker,
)
from orion.spark.concept_induction.drives import DriveEngine, DriveMathConfig
from orion.spark.concept_induction.settings import ConceptSettings

NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)


def _tension(drive: str, magnitude: float, *, artifact_id: str = "t-test") -> TensionEventV1:
    return TensionEventV1(
        artifact_id=artifact_id,
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="tension.test.v1",
        magnitude=magnitude,
        drive_impacts={drive: 1.0},
        provenance={"intake_channel": "test:drive-fold"},
    )


def _worker(tmp_path) -> ConceptWorker:
    return ConceptWorker(
        ConceptSettings(orion_bus_enabled=False, store_path=str(tmp_path / "state.json"))
    )


def test_first_ever_call_always_folds(tmp_path) -> None:
    """No prior fold timestamp recorded for the subject -- there's nothing to
    wait for, so the very first tick always folds and persists."""
    worker = _worker(tmp_path)
    tension = _tension("coherence", magnitude=0.5)

    pressures, activations, folded = worker._update_drive_pressures("orion", [tension], NOW)

    assert folded is True
    # The impulse was actually applied, not just decay from a zero baseline.
    assert pressures["coherence"] > 0.0

    persisted = worker.store.load_drive_state("orion")
    assert persisted.get("updated_at") == NOW.isoformat()
    assert persisted.get("pressures", {}).get("coherence") == pressures["coherence"]


def test_second_call_within_fold_interval_does_not_fold_or_persist(tmp_path) -> None:
    """A second call arriving well within _DRIVE_FOLD_INTERVAL_SEC (900s) of the
    first fold must NOT apply its tension to the persisted integrator, and must
    NOT touch the store -- it only returns a decay-only projection."""
    worker = _worker(tmp_path)
    tension1 = _tension("coherence", magnitude=0.5, artifact_id="t-1")
    worker._update_drive_pressures("orion", [tension1], NOW)
    persisted_after_call1 = worker.store.load_drive_state("orion")

    now2 = NOW + timedelta(seconds=60)
    tension2 = _tension("coherence", magnitude=0.9, artifact_id="t-2")
    pressures2, activations2, folded2 = worker._update_drive_pressures("orion", [tension2], now2)

    assert folded2 is False

    # Expected: pure wall-clock decay of call 1's persisted pressure, computed
    # independently via a fresh DriveEngine with tensions=[] (no impulse) --
    # NOT influenced by tension2 at all, since it was buffered, not applied.
    expected_engine = DriveEngine(DriveMathConfig())
    expected_pressures, expected_activations = expected_engine.update(
        previous_pressures=persisted_after_call1["pressures"],
        previous_activations=persisted_after_call1["activations"],
        tensions=[],
        now=now2,
        previous_ts=NOW,
    )
    assert pressures2 == expected_pressures
    assert activations2 == expected_activations

    # The store must be untouched by call 2 -- still shows call 1's timestamp.
    persisted_after_call2 = worker.store.load_drive_state("orion")
    assert persisted_after_call2.get("updated_at") == NOW.isoformat()
    assert persisted_after_call2 == persisted_after_call1


def test_third_call_after_interval_folds_all_buffered_tensions_together(tmp_path) -> None:
    """Once _DRIVE_FOLD_INTERVAL_SEC has elapsed since the last fold, the next
    call folds EVERYTHING buffered since then (the skipped call 2's tension
    plus this call's own new tension) into a single DriveEngine.update() --
    not two separate applications."""
    worker = _worker(tmp_path)
    tension1 = _tension("coherence", magnitude=0.5, artifact_id="t-1")
    worker._update_drive_pressures("orion", [tension1], NOW)
    persisted_after_call1 = worker.store.load_drive_state("orion")

    now2 = NOW + timedelta(seconds=60)
    tension2 = _tension("coherence", magnitude=0.9, artifact_id="t-2")
    worker._update_drive_pressures("orion", [tension2], now2)  # buffered, not folded

    now3 = NOW + timedelta(seconds=901)  # past the 900s interval from the call-1 fold
    tension3 = _tension("coherence", magnitude=0.3, artifact_id="t-3")
    pressures3, activations3, folded3 = worker._update_drive_pressures("orion", [tension3], now3)

    assert folded3 is True

    # Expected: one single DriveEngine.update() call starting from call 1's
    # persisted state, applying BOTH tension2 and tension3 together.
    expected_engine = DriveEngine(DriveMathConfig())
    expected_pressures, expected_activations = expected_engine.update(
        previous_pressures=persisted_after_call1["pressures"],
        previous_activations=persisted_after_call1["activations"],
        tensions=[tension2, tension3],
        now=now3,
        previous_ts=NOW,
    )
    assert pressures3 == expected_pressures
    assert activations3 == expected_activations

    persisted_after_call3 = worker.store.load_drive_state("orion")
    assert persisted_after_call3.get("updated_at") == now3.isoformat()


def test_saturation_prevention_regression(tmp_path) -> None:
    """The actual point of O2: this is the test that would have caught the
    original per-event saturation problem. OLD (unthrottled) behavior for a
    repeated impulse of this exact shape (drive_impacts={"predictive": 1.0},
    magnitude=0.5, applied on every single DriveEngine.update() call) pins
    pressure above 0.95 within ~5 ticks / ~23 seconds (verified during the O3
    review). NEW (fold-gated) behavior must NOT do that.

    This simulates the STEADY-STATE bus cadence (~13 events/min, ~4.6s apart)
    that follows an already-recent fold, by seeding _last_drive_fold_at
    directly rather than letting the loop's own first call trigger a
    cold-start fold (that cold-start path is exercised separately by
    test_first_ever_call_always_folds above) -- none of these 30 calls should
    cross the 900s fold boundary, so every one of them must be a pure
    decay-only projection with no impulse applied.
    """
    worker = _worker(tmp_path)
    worker._last_drive_fold_at["orion"] = NOW - timedelta(seconds=1)

    now = NOW
    pressures: dict = {}
    for _ in range(30):
        tension = _tension("predictive", magnitude=0.5)
        pressures, _activations, folded = worker._update_drive_pressures("orion", [tension], now)
        assert folded is False  # steady-state: none of these 30 calls cross 900s
        now = now + timedelta(seconds=4.6)

    # ~138.6s of ~4.6s-cadence impulses, none folded -> pressure stays at the
    # decayed baseline (0, since nothing has ever been folded in) instead of
    # climbing toward 1.0 the way the old unthrottled per-event call would.
    assert pressures["predictive"] < 0.1


def test_pending_buffer_caps_and_drops_oldest(tmp_path) -> None:
    """Review finding: the pending buffer had no size cap, only a time-based
    fold gate -- a subject that never crosses a fold boundary could buffer an
    unbounded number of tensions (each carrying a full ArtifactProvenance).
    _MAX_PENDING_DRIVE_TENSIONS bounds it, dropping the oldest entries first
    (same "keep most recent" convention as _prune_window)."""
    worker = _worker(tmp_path)
    # Seed a recent fold so none of the following calls cross the 900s gate.
    worker._last_drive_fold_at["orion"] = NOW

    overflow_count = _MAX_PENDING_DRIVE_TENSIONS + 10
    for i in range(overflow_count):
        tension = _tension("coherence", magnitude=0.1, artifact_id=f"t-{i}")
        _, _, folded = worker._update_drive_pressures(
            "orion", [tension], NOW + timedelta(seconds=i)
        )
        assert folded is False

    pending = worker._pending_drive_tensions["orion"]
    assert len(pending) == _MAX_PENDING_DRIVE_TENSIONS
    # Oldest entries (t-0 .. t-9) were dropped; the most recent survive.
    kept_ids = {t.artifact_id for t in pending}
    assert "t-0" not in kept_ids
    assert f"t-{overflow_count - 1}" in kept_ids
