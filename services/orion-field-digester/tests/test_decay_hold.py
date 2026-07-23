"""Regression tests for the decay/injection-interval mismatch fix
(docs/superpowers/specs/2026-07-17-field-digester-decay-hold-fix-design.md).

Root cause: apply_decay() previously multiplied every NODE_DECAY_CHANNELS
entry by decay_rate unconditionally every 2s tick, regardless of whether
fresh data arrived that tick. Real biometrics publishes land roughly every
15-30s, so a channel lost ~44% of its value (0.92^7 ~= 0.56) between two real
publishes, then got snapped straight back up to the fresh measurement on the
next publish -- a mechanical sawtooth with zero connection to genuine
volatility (orion/autonomy/drives_and_autonomy_retrospective.md sec5b).

Fix: hold each channel's value by default; only decay once it has actually
gone stale (no fresh perturbation within staleness_threshold_sec). These
tests characterize the before/after behavior directly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.digestion.decay import NODE_DECAY_CHANNELS, apply_decay
from app.digestion.perturbation import apply_perturbations
from app.digestion.suppression import apply_suppression
from app.ingest.state_deltas import Perturbation

from orion.schemas.field_state import FieldStateV1

BASE = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
DECAY_RATE = 0.92
STALENESS_THRESHOLD_SEC = 90.0


def _state(node_vectors: dict[str, dict[str, float]] | None = None) -> FieldStateV1:
    return FieldStateV1(
        generated_at=BASE,
        tick_id="tick_decay_hold_test",
        node_vectors=node_vectors or {},
        edges=[],
    )


def test_new_fcc_motor_channels_are_registered_for_decay() -> None:
    """A channel only decays if listed in NODE_DECAY_CHANNELS (separate from being in
    NODE_CHANNELS) -- otherwise it holds forever. Written through the normal
    state_deltas.py -> apply_perturbations() path (same as execution_load), these get
    node_vector_updated_at stamped automatically, so no manual-stamping test is needed
    here -- this just confirms the registration itself, the miss this service's own
    CLAUDE.md warns has already happened once for a different channel."""
    for channel in (
        "harness_step_load",
        "tool_failure_streak_pressure",
        "avg_step_chars_pressure",
        "compliance_deficit",
    ):
        assert channel in NODE_DECAY_CHANNELS


def test_held_flat_within_staleness_window_then_decays_once_stale() -> None:
    """A channel perturbed once, then apply_decay called repeatedly at ~2s-tick
    increments with no further perturbation: value stays exactly flat for
    ticks within the staleness window, only begins decaying once elapsed
    time crosses staleness_threshold_sec."""
    state = _state()
    apply_perturbations(
        state,
        [Perturbation(node_id="node:atlas", channel="cpu_pressure", intensity=0.8, label="p1", mode="replace")],
        now=BASE,
    )
    assert state.node_vectors["node:atlas"]["cpu_pressure"] == 0.8

    # Ticks well within the 90s staleness window (2s cadence): value must
    # stay exactly 0.8, no sawtooth.
    tick_now = BASE
    for i in range(1, 40):  # up to ~78s elapsed, still under 90s
        tick_now = BASE + timedelta(seconds=2 * i)
        apply_decay(
            state,
            decay_rate=DECAY_RATE,
            now=tick_now,
            staleness_threshold_sec=STALENESS_THRESHOLD_SEC,
        )
        assert state.node_vectors["node:atlas"]["cpu_pressure"] == 0.8, (
            f"channel decayed at tick {i} ({(tick_now - BASE).total_seconds()}s elapsed), "
            "should have been held flat within the staleness window"
        )

    # Cross the threshold: next tick at 92s elapsed must decay.
    stale_now = BASE + timedelta(seconds=92)
    apply_decay(
        state,
        decay_rate=DECAY_RATE,
        now=stale_now,
        staleness_threshold_sec=STALENESS_THRESHOLD_SEC,
    )
    assert state.node_vectors["node:atlas"]["cpu_pressure"] == 0.8 * DECAY_RATE


def test_same_tick_perturb_then_decay_does_not_decay_fresh_value() -> None:
    """Perturb then immediately call apply_decay in the same tick (now identical
    to the perturbation's ts, mirroring run_digestion_tick's perturb -> decay
    order): the freshly-set value must NOT be decayed at all (elapsed = 0s)."""
    state = _state()
    apply_perturbations(
        state,
        [Perturbation(node_id="node:atlas", channel="memory_pressure", intensity=0.6, label="p1", mode="replace")],
        now=BASE,
    )
    apply_decay(
        state,
        decay_rate=DECAY_RATE,
        now=BASE,
        staleness_threshold_sec=STALENESS_THRESHOLD_SEC,
    )
    assert state.node_vectors["node:atlas"]["memory_pressure"] == 0.6


def test_missing_updated_at_entry_still_decays_every_call() -> None:
    """A channel with no node_vector_updated_at entry at all (simulating a
    never-perturbed channel, or a FieldStateV1 persisted before this fix)
    still decays every call -- proves the safe-default/migration path works
    unchanged from today's behavior."""
    state = _state(node_vectors={"node:atlas": {"gpu_pressure": 0.5}})
    assert state.node_vector_updated_at == {}

    apply_decay(
        state,
        decay_rate=DECAY_RATE,
        now=BASE,
        staleness_threshold_sec=STALENESS_THRESHOLD_SEC,
    )
    assert state.node_vectors["node:atlas"]["gpu_pressure"] == 0.5 * DECAY_RATE

    apply_decay(
        state,
        decay_rate=DECAY_RATE,
        now=BASE + timedelta(seconds=2),
        staleness_threshold_sec=STALENESS_THRESHOLD_SEC,
    )
    assert state.node_vectors["node:atlas"]["gpu_pressure"] == 0.5 * DECAY_RATE * DECAY_RATE


def test_decay_resumes_normally_once_genuinely_stale() -> None:
    """A channel goes stale (no perturbation for longer than the threshold)
    then decay resumes applying normally -- the post-threshold trajectory
    matches plain value * decay_rate per tick once stale."""
    state = _state()
    apply_perturbations(
        state,
        [Perturbation(node_id="node:atlas", channel="thermal_pressure", intensity=0.9, label="p1", mode="replace")],
        now=BASE,
    )

    # First stale tick, right at the threshold boundary (>= threshold decays).
    stale_start = BASE + timedelta(seconds=STALENESS_THRESHOLD_SEC)
    apply_decay(
        state,
        decay_rate=DECAY_RATE,
        now=stale_start,
        staleness_threshold_sec=STALENESS_THRESHOLD_SEC,
    )
    expected = 0.9 * DECAY_RATE
    assert state.node_vectors["node:atlas"]["thermal_pressure"] == expected

    # Note: node_vector_updated_at is only refreshed by a real perturbation,
    # not by decay itself, so every subsequent tick after going stale is
    # still "elapsed >= threshold" and decays normally each time -- matching
    # plain value * decay_rate per tick, same as today's unconditional loop,
    # for as long as the channel remains unperturbed.
    for i in range(1, 5):
        tick_now = stale_start + timedelta(seconds=2 * i)
        apply_decay(
            state,
            decay_rate=DECAY_RATE,
            now=tick_now,
            staleness_threshold_sec=STALENESS_THRESHOLD_SEC,
        )
        expected = expected * DECAY_RATE
        assert state.node_vectors["node:atlas"]["thermal_pressure"] == expected


def test_suppression_staleness_reset_records_node_vector_updated_at() -> None:
    """staleness (services/orion-field-digester/app/digestion/suppression.py)
    is a NODE_DECAY_CHANNELS entry but apply_suppression() writes it directly,
    outside apply_perturbations() -- code review (2026-07-17) found this
    bypassed node_vector_updated_at tracking, same class of gap as
    field_coherence_warning in worker.py. Currently inert (decaying 0.0 is a
    no-op) but fixed for consistency: verifies the reset value gets a
    matching node_vector_updated_at stamp."""
    state = _state(
        node_vectors={"node:circe": {"expected_offline_suppression": 1.0, "staleness": 0.5}}
    )
    apply_suppression(state)
    assert state.node_vectors["node:circe"]["staleness"] == 0.0
    assert state.node_vector_updated_at["node:circe"]["staleness"] == state.generated_at
