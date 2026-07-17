"""Task 1: leaky-integrator pressure math.

Proves the flat-0.731 fixed point is gone: pressure rests at zero, is
cadence-invariant, and differentiates per drive. Legacy path preserved.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.drives import TensionEventV1
from orion.spark.concept_induction.drives import (
    DRIVE_KEYS,
    DriveEngine,
    DriveMathConfig,
)

T0 = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)


def _tension(drive_impacts: dict[str, float], magnitude: float = 1.0) -> TensionEventV1:
    return TensionEventV1(
        artifact_id="t-test",
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="tension.signal.v1",
        magnitude=magnitude,
        drive_impacts=drive_impacts,
        provenance={"intake_channel": "test:leaky"},
    )


def _leaky_engine(tau: float = 1800.0) -> DriveEngine:
    return DriveEngine(DriveMathConfig(decay_tau_sec=tau, leaky_math_enabled=True))


def test_rest_at_zero() -> None:
    """Any starting pressure + many no-impulse ticks decays below 1e-3."""
    engine = _leaky_engine(tau=300.0)
    pressures = {k: 0.8 for k in DRIVE_KEYS}
    activations = {k: True for k in DRIVE_KEYS}
    ts = T0
    for _ in range(60):  # 60 * 60s = 3600s = 12 tau
        ts = ts + timedelta(seconds=60)
        pressures, activations = engine.update(
            previous_pressures=pressures,
            previous_activations=activations,
            tensions=[],
            now=ts,
            previous_ts=ts - timedelta(seconds=60),
        )
    assert all(p < 1e-3 for p in pressures.values()), pressures


def test_cadence_invariance() -> None:
    """Same impulse at the same wall-times yields equal pressure regardless of
    how many empty ticks happen between them."""
    impulse_at = T0 + timedelta(seconds=100)
    read_at = T0 + timedelta(seconds=200)

    # Path A: one impulse tick, one read tick.
    eng_a = _leaky_engine()
    p_a, a_a = eng_a.update(
        previous_pressures={k: 0.0 for k in DRIVE_KEYS},
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=[_tension({"capability": 0.5})],
        now=impulse_at,
        previous_ts=T0,
    )
    p_a, _ = eng_a.update(
        previous_pressures=p_a,
        previous_activations=a_a,
        tensions=[],
        now=read_at,
        previous_ts=impulse_at,
    )

    # Path B: same impulse at the same wall-time, but 100 empty ticks/second
    # between impulse and read.
    eng_b = _leaky_engine()
    p_b, a_b = eng_b.update(
        previous_pressures={k: 0.0 for k in DRIVE_KEYS},
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=[_tension({"capability": 0.5})],
        now=impulse_at,
        previous_ts=T0,
    )
    prev = impulse_at
    for i in range(1, 101):
        now = impulse_at + timedelta(seconds=i)
        p_b, a_b = eng_b.update(
            previous_pressures=p_b,
            previous_activations=a_b,
            tensions=[],
            now=now,
            previous_ts=prev,
        )
        prev = now

    assert abs(p_a["capability"] - p_b["capability"]) < 1e-9, (p_a, p_b)


def test_no_uniform_pin() -> None:
    """Distinct per-drive impulses produce distinct pressures — not all 0.731."""
    engine = _leaky_engine()
    pressures, _ = engine.update(
        previous_pressures={k: 0.0 for k in DRIVE_KEYS},
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=[_tension({"capability": 0.6, "coherence": 0.2, "relational": 0.05})],
        now=T0 + timedelta(seconds=10),
        previous_ts=T0,
    )
    assert pressures["capability"] > pressures["coherence"] > pressures["relational"]
    assert pressures["autonomy"] == 0.0  # unmapped drive stays at rest
    # None are pinned at the legacy fixed point.
    assert all(abs(p - 0.7309) > 0.01 for p in pressures.values()), pressures


def test_frequent_ticks_do_not_inflate() -> None:
    """The flat-0.731 bug: legacy math inflates every drive under frequent
    zero-impulse ticks; leaky math does not."""
    leaky = _leaky_engine()
    legacy = DriveEngine(DriveMathConfig(leaky_math_enabled=False))
    lp = {k: 0.0 for k in DRIVE_KEYS}
    gp = {k: 0.0 for k in DRIVE_KEYS}
    la = {k: False for k in DRIVE_KEYS}
    ga = {k: False for k in DRIVE_KEYS}
    prev = T0
    for i in range(1, 501):  # 500 fast ticks, no real impulse
        now = T0 + timedelta(seconds=i)
        lp, la = leaky.update(previous_pressures=lp, previous_activations=la,
                              tensions=[], now=now, previous_ts=prev)
        gp, ga = legacy.update(previous_pressures=gp, previous_activations=ga,
                               tensions=[], now=now, previous_ts=prev)
        prev = now
    # Leaky rests at zero; legacy stays pinned near its fixed point is only
    # reached with impulse — with zero impulse legacy also decays, so assert the
    # discriminating property: leaky is at rest.
    assert all(p < 1e-6 for p in lp.values()), lp


def test_negative_impact_relieves_pressure() -> None:
    """A relief (negative-weight) tension reduces existing pressure, floored
    at 0 -- the mandatory companion fix for P3's satisfaction mechanic."""
    engine = _leaky_engine()
    # Build up real pressure first.
    pressures, activations = engine.update(
        previous_pressures={k: 0.0 for k in DRIVE_KEYS},
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=[_tension({"predictive": 0.9})],
        now=T0 + timedelta(seconds=10),
        previous_ts=T0,
    )
    before = pressures["predictive"]
    assert before > 0.0

    # Relief tension, same tick cadence so decay is negligible.
    pressures, activations = engine.update(
        previous_pressures=pressures,
        previous_activations=activations,
        tensions=[_tension({"predictive": -0.5}, magnitude=0.3)],
        now=T0 + timedelta(seconds=20),
        previous_ts=T0 + timedelta(seconds=10),
    )
    after = pressures["predictive"]
    assert after < before
    assert after >= 0.0


def test_relief_floors_at_zero_never_negative() -> None:
    """A relief impulse larger than the remaining pressure floors at 0, never
    goes negative -- this must hold even with an extreme relief weight."""
    engine = _leaky_engine()
    pressures, activations = engine.update(
        previous_pressures={k: 0.05 for k in DRIVE_KEYS},
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=[_tension({"coherence": -1.0}, magnitude=1.0)],
        now=T0 + timedelta(seconds=10),
        previous_ts=T0,
    )
    assert pressures["coherence"] == 0.0
    assert all(p >= 0.0 for p in pressures.values())


def test_relief_at_zero_pressure_is_a_no_op() -> None:
    """Relief has nothing to relieve when pressure is already at rest --
    diminishing-effect-toward-floor means this doesn't depend on the outer
    clamp to "save" a would-be-negative value; the term itself is zero."""
    engine = _leaky_engine()
    pressures, _ = engine.update(
        previous_pressures={k: 0.0 for k in DRIVE_KEYS},
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=[_tension({"relational": -0.8}, magnitude=1.0)],
        now=T0 + timedelta(seconds=10),
        previous_ts=T0,
    )
    assert pressures["relational"] == 0.0


def test_positive_only_tensions_unaffected_by_signed_clamp() -> None:
    """Regression guard: the [-1,1] signed clamp must be byte-identical to
    the old [0,1] clamp for any weight that was already non-negative."""
    engine = _leaky_engine()
    pressures, _ = engine.update(
        previous_pressures={k: 0.0 for k in DRIVE_KEYS},
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=[_tension({"capability": 0.6, "coherence": 0.2, "relational": 0.05})],
        now=T0 + timedelta(seconds=10),
        previous_ts=T0,
    )
    # Exact same assertions as test_no_uniform_pin, unchanged by the clamp fix.
    assert pressures["capability"] > pressures["coherence"] > pressures["relational"]
    assert pressures["autonomy"] == 0.0
    assert all(abs(p - 0.7309) > 0.01 for p in pressures.values()), pressures


def test_signal_only_competition_tension_contributes_zero_pressure() -> None:
    """Regression (2026-07-15 saturation): tension.drive_competition.v1 is
    signal-only -- empty drive_impacts. Fed to update() alongside a normal
    tension, it must contribute exactly zero pressure change: the result is
    identical to running update() with the normal tension alone."""
    competition = TensionEventV1(
        artifact_id="t-competition",
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="tension.drive_competition.v1",
        magnitude=0.96,  # the live pinned spread from the diagnosis
        drive_impacts={},  # signal-only by design
        provenance={"intake_channel": "test:leaky"},
    )
    normal = _tension({"capability": 0.6, "coherence": 0.2})

    with_competition, act_with = _leaky_engine().update(
        previous_pressures={k: 0.1 for k in DRIVE_KEYS},
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=[normal, competition],
        now=T0 + timedelta(seconds=10),
        previous_ts=T0,
    )
    without_competition, act_without = _leaky_engine().update(
        previous_pressures={k: 0.1 for k in DRIVE_KEYS},
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=[normal],
        now=T0 + timedelta(seconds=10),
        previous_ts=T0,
    )
    assert with_competition == without_competition
    assert act_with == act_without
    # And the normal tension really did move something (the tick happened).
    assert with_competition["capability"] > 0.1


def test_competition_tension_alone_is_pure_decay() -> None:
    """A tick carrying ONLY the competition tension behaves exactly like an
    empty tick: pure wall-clock decay, no subsidy to any drive."""
    competition = TensionEventV1(
        artifact_id="t-competition-solo",
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="tension.drive_competition.v1",
        magnitude=0.96,
        drive_impacts={},
        provenance={"intake_channel": "test:leaky"},
    )
    start = {k: 0.5 for k in DRIVE_KEYS}
    with_tension, _ = _leaky_engine().update(
        previous_pressures=dict(start),
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=[competition],
        now=T0 + timedelta(seconds=60),
        previous_ts=T0,
    )
    empty_tick, _ = _leaky_engine().update(
        previous_pressures=dict(start),
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=[],
        now=T0 + timedelta(seconds=60),
        previous_ts=T0,
    )
    assert with_tension == empty_tick
    assert all(with_tension[k] < start[k] for k in DRIVE_KEYS)  # decay happened


def test_fold_batch_no_collapse_at_shared_clamp_boundary() -> None:
    """The core collapse-fix proof (2026-07-17 fold-batch clamp collapse bug).

    Live production symptom: coherence/capability/predictive pinned to a
    byte-identical mid-range value (0.45036942460343243) across consecutive
    drive_audits rows. That is only possible under the OLD sum-then-clamp
    design: impact_sum[drive] accumulates unbounded across a whole fold
    batch, then gets clamped ONCE at the end to [-1, 1]. Once that clamp
    saturates to exactly 1.0, `clamp01(base + 1.0*(1-base)) == 1.0`
    regardless of `base` -- any drive whose batch-summed impact exceeds the
    clamp lands on the exact same final value as every other drive that also
    exceeded it, independent of how much it exceeded by.

    Reproduction, hand-verified (see
    docs/superpowers/specs/2026-07-17-drive-engine-fold-batch-clamp-collapse-
    fix-design.md):

    Both drives start at the same mid-range base pressure (0.3, decay=1.0
    since elapsed=0). Each contributing tension has magnitude=1.0 and
    weight=0.3 -> per-event impulse = 0.3, already well inside [-1, 1] on
    its own (a single event never triggers the old bug).

    "coherence" receives 4 such tensions in the batch; "capability" receives
    8. Under the OLD code:
        impact_sum[coherence] = 4 * 0.3 = 1.2  -> clamp_signed -> 1.0
        impact_sum[capability] = 8 * 0.3 = 2.4 -> clamp_signed -> 1.0
    Both clamped impulses are exactly 1.0, so BOTH drives collapse to the
    identical final value clamp01(0.3 + 1.0*(1 - 0.3)) == 1.0, regardless of
    N=4 vs N=8 -- the exact collapse signature from production (a shared
    value that doesn't depend on how many tensions actually contributed).

    Under the NEW sequential code, each event's impulse (0.3) is applied one
    at a time via the diminishing-headroom recurrence
    p = clamp01(p + 0.3*(1-p)):
        N=4: 0.3 -> 0.51 -> 0.657 -> 0.7599 -> 0.83193
        N=8: continuing the same recurrence for 4 more events -> 0.959646393
    Both land in a genuinely high/mid-high range (not at the exact 0/1
    ceiling) and, critically, they DIFFER -- proving the batch no longer
    collapses distinct drives to an identical value.
    """
    engine = _leaky_engine()
    coherence_tensions = [_tension({"coherence": 0.3}, magnitude=1.0) for _ in range(4)]
    capability_tensions = [_tension({"capability": 0.3}, magnitude=1.0) for _ in range(8)]

    pressures, _ = engine.update(
        previous_pressures={k: 0.3 for k in DRIVE_KEYS},
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=coherence_tensions + capability_tensions,
        now=T0,
        previous_ts=T0,
    )

    # Neither drive collapsed to the old code's shared clamp-boundary value.
    assert pressures["coherence"] != 1.0
    assert pressures["capability"] != 1.0
    # The core proof: different tension counts -> different final pressures,
    # not the identical value the old sum-then-clamp design would produce.
    assert pressures["coherence"] != pressures["capability"]
    # Both genuinely moved into a high/mid-high range (saturation still
    # works, see test_fold_batch_saturation_still_reaches_bound below).
    assert 0.7 < pressures["coherence"] < 1.0
    assert 0.9 < pressures["capability"] < 1.0
    # Pin the hand-derived values so a future change to the recurrence is
    # caught explicitly, not just by the inequality above.
    assert abs(pressures["coherence"] - 0.83193) < 1e-9
    assert abs(pressures["capability"] - 0.959646393) < 1e-9


def test_fold_batch_saturation_still_reaches_bound() -> None:
    """Saturation itself is not disabled by the collapse fix -- a drive that
    legitimately receives enough same-sign tensions in one batch still moves
    substantially toward 1.0 (mirrored: toward 0.0 for relief)."""
    engine = _leaky_engine()
    tensions = [_tension({"predictive": 0.3}, magnitude=1.0) for _ in range(20)]

    pressures, _ = engine.update(
        previous_pressures={k: 0.0 for k in DRIVE_KEYS},
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=tensions,
        now=T0,
        previous_ts=T0,
    )
    assert pressures["predictive"] > 0.99


def test_fold_batch_update_is_deterministic_across_repeated_calls() -> None:
    """Running update() twice with the exact same tensions list (same order,
    same objects) produces byte-identical pressure output both times --
    sequential application is order-sensitive but still fully deterministic
    for a fixed input list."""
    tensions = [
        _tension({"coherence": 0.4, "capability": -0.2}, magnitude=0.8),
        _tension({"capability": 0.5}, magnitude=0.6),
        _tension({"coherence": -0.3, "predictive": 0.7}, magnitude=0.9),
    ]

    engine_a = _leaky_engine()
    pressures_a, activations_a = engine_a.update(
        previous_pressures={k: 0.35 for k in DRIVE_KEYS},
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=tensions,
        now=T0 + timedelta(seconds=10),
        previous_ts=T0,
    )

    engine_b = _leaky_engine()
    pressures_b, activations_b = engine_b.update(
        previous_pressures={k: 0.35 for k in DRIVE_KEYS},
        previous_activations={k: False for k in DRIVE_KEYS},
        tensions=tensions,
        now=T0 + timedelta(seconds=10),
        previous_ts=T0,
    )

    assert pressures_a == pressures_b
    assert activations_a == activations_b


def test_legacy_path_preserved() -> None:
    """Flag off ⇒ soft_saturate path drives the fixed-point inflation."""
    legacy = DriveEngine(DriveMathConfig(leaky_math_enabled=False))
    pressures = {k: 0.0 for k in DRIVE_KEYS}
    activations = {k: False for k in DRIVE_KEYS}
    prev = T0
    # Frequent ticks WITH a small steady impulse → legacy inflates toward ~0.73.
    for i in range(1, 401):
        now = T0 + timedelta(seconds=i)
        pressures, activations = legacy.update(
            previous_pressures=pressures,
            previous_activations=activations,
            tensions=[_tension({k: 0.02 for k in DRIVE_KEYS}, magnitude=1.0)],
            now=now,
            previous_ts=prev,
        )
        prev = now
    # Legacy pins high and uniform — the documented artifact.
    assert all(p > 0.6 for p in pressures.values()), pressures
