"""Tests for the vision liveness watcher.

Every threshold fixture below is hand-computed and the arithmetic written out,
so a test that passes is evidence the *formula* is right rather than evidence
the code agrees with itself.

The headline test is `test_replays_the_2026_08_20_outage`: it replays the shape
of the real incident (every task returning gpu_hard_floor, one every 5s, the
observed cadence from the live logs) and asserts an alert fires. That incident
ran ~21 hours undetected, so "would this have caught it" is the only acceptance
criterion that matters.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.liveness import (  # noqa: E402
    LivenessDecision,
    VisionLivenessWatcher,
    build_attention_request,
    build_watcher_or_default,
)


def _watcher(**kw):
    defaults = dict(
        window_sec=300.0, min_samples=10, arm_fail_rate=0.8,
        clear_fail_rate=0.2, sustain_sec=180.0, cooldown_sec=3600.0,
    )
    defaults.update(kw)
    return VisionLivenessWatcher(**defaults)


# -- construction ----------------------------------------------------------


def test_rejects_non_hysteretic_config() -> None:
    """clear >= arm is a flapping bug with extra config; refuse at construction."""
    with pytest.raises(ValueError):
        _watcher(arm_fail_rate=0.8, clear_fail_rate=0.8)
    with pytest.raises(ValueError):
        _watcher(arm_fail_rate=0.5, clear_fail_rate=0.9)


# -- the incident ----------------------------------------------------------


def test_replays_the_2026_08_20_outage() -> None:
    """Every task gpu_hard_floor, one per 5s -- the real observed cadence.

    Hand-computed, and the first version of this test got it wrong in a way
    worth recording. The sustain clock does not start at the first failure: it
    starts once min_samples is satisfied, because below that the rate is one or
    two points. At 5s spacing the 10th sample lands at t=45.0, so
    failing_since=45.0 and the alert needs t - 45.0 >= 180.0, i.e. t >= 225.0 --
    the 46th sample. Not t=180.0.

    Detection latency for the real incident is therefore 225s, under 4 minutes,
    against the ~21 hours it actually went unnoticed.
    """
    w = _watcher()
    fired_at = None
    for i in range(60):                      # 0s .. 295s
        t = i * 5.0
        d = w.record(ok=False, error_code="gpu_hard_floor", now=t)
        if d.alert:
            fired_at = t
            break

    assert fired_at == 225.0, f"expected first alert at t=225.0s, got {fired_at}"
    assert w.snapshot(now=225.0)["alerting"] is True


def test_does_not_alert_before_sustain_elapses() -> None:
    """45 consecutive failures at 5s spacing is t=220s -- still under t=225."""
    w = _watcher()
    for i in range(45):                      # 0s .. 220s
        d = w.record(ok=False, error_code="gpu_hard_floor", now=i * 5.0)
        assert not d.alert, f"alerted early at t={i * 5.0}s"


def test_does_not_alert_below_min_samples() -> None:
    """9 failures is total failure but only 9 samples; min_samples is 10.

    Spacing is 5s, not 60s: at 60s spacing with a 300s window the earliest
    samples are evicted before the 10th arrives, so the count never reaches the
    floor for a reason unrelated to what this test is checking.
    """
    w = _watcher()
    for i in range(9):
        d = w.record(ok=False, error_code="gpu_hard_floor", now=i * 5.0)
        assert not d.alert
        assert d.sample_count == i + 1


# -- normal operation ------------------------------------------------------


def test_healthy_traffic_never_alerts() -> None:
    w = _watcher()
    for i in range(200):
        assert not w.record(ok=True, now=i * 5.0).alert


def test_occasional_failures_never_alert() -> None:
    """1 failure in every 5 tasks = 0.2 fail rate, exactly the clear threshold."""
    w = _watcher()
    for i in range(200):
        d = w.record(ok=(i % 5 != 0), error_code="gpu_transient", now=i * 5.0)
        assert not d.alert
    assert w.snapshot(now=1000.0)["fail_rate"] == pytest.approx(0.2, abs=0.02)


# -- hysteresis ------------------------------------------------------------


def test_hysteresis_band_does_not_flap() -> None:
    """Sag into the band (0.5) and climb back: must not re-alert, must not clear.

    A bare threshold would clear at 0.5 < 0.8 and re-fire on the next failure.
    """
    w = _watcher()
    t = 0.0
    for _ in range(50):                       # arm it: needs t>=225, so 46+ samples
        d = w.record(ok=False, error_code="gpu_hard_floor", now=t); t += 5.0
    assert w.snapshot(now=t)["alerting"] is True

    # 50/50 for a while -- inside the band [0.2, 0.8), neither arm nor clear
    for i in range(60):
        d = w.record(ok=(i % 2 == 0), error_code="gpu_hard_floor", now=t); t += 5.0
        assert not d.alert, "re-alerted while inside the hysteresis band"
        assert not d.recovered, "cleared while inside the hysteresis band"
    assert w.snapshot(now=t)["alerting"] is True


def test_recovery_clears_and_reports() -> None:
    w = _watcher()
    t = 0.0
    for _ in range(50):                       # needs t>=225 to arm
        w.record(ok=False, error_code="gpu_hard_floor", now=t); t += 5.0
    assert w.snapshot(now=t)["alerting"] is True

    recovered = None
    for _ in range(80):                       # flush the window with successes
        d = w.record(ok=True, now=t); t += 5.0
        if d.recovered:
            recovered = d
            break
    assert recovered is not None, "never reported recovery"
    assert recovered.fail_rate <= 0.2
    assert w.snapshot(now=t)["alerting"] is False


def test_cooldown_blocks_immediate_refire() -> None:
    """After recovery, a fresh failure burst inside cooldown must not re-alert."""
    w = _watcher(cooldown_sec=3600.0)
    t = 0.0
    for _ in range(50):
        w.record(ok=False, error_code="gpu_hard_floor", now=t); t += 5.0
    for _ in range(80):
        w.record(ok=True, now=t); t += 5.0
    assert w.snapshot(now=t)["alerting"] is False

    for _ in range(60):                       # 300s of failure, well past sustain
        d = w.record(ok=False, error_code="gpu_hard_floor", now=t); t += 5.0
        assert not d.alert, f"re-alerted at t={t} inside the 3600s cooldown"


def test_alerts_again_after_cooldown_expires() -> None:
    w = _watcher(cooldown_sec=600.0)
    t = 0.0
    for _ in range(50):
        w.record(ok=False, error_code="gpu_hard_floor", now=t); t += 5.0
    for _ in range(80):
        w.record(ok=True, now=t); t += 5.0

    t += 700.0                                # past the 600s cooldown
    fired = False
    for _ in range(60):
        if w.record(ok=False, error_code="gpu_hard_floor", now=t).alert:
            fired = True
            break
        t += 5.0
    assert fired, "never re-alerted after the cooldown expired"


# -- window eviction -------------------------------------------------------


def test_old_failures_fall_out_of_the_window() -> None:
    """Failures older than window_sec must not keep the rate pinned high."""
    w = _watcher(window_sec=300.0)
    for i in range(20):
        w.record(ok=False, error_code="gpu_hard_floor", now=i * 5.0)   # t=0..95
    # Jump past the window, then succeed. The old failures are evicted.
    t = 500.0
    for _ in range(15):
        w.record(ok=True, now=t); t += 5.0
    assert w.snapshot(now=t)["fail_rate"] == 0.0


def test_thin_traffic_does_not_reset_the_sustain_clock() -> None:
    """Dipping below min_samples is not recovery.

    Traffic thinning must not restart the sustain clock, or a slow-but-broken
    stream never accumulates enough continuous failure to alert.
    """
    w = _watcher(min_samples=10, window_sec=300.0)
    for i in range(12):                        # arm the clock at t=0
        w.record(ok=False, error_code="gpu_hard_floor", now=i * 5.0)
    # long gap -> window empties -> below min_samples
    assert w.record(ok=False, error_code="gpu_hard_floor", now=1000.0).sample_count == 1
    # refill; the clock started at t=0, so sustain is long since satisfied
    fired = False
    for i in range(20):
        if w.record(ok=False, error_code="gpu_hard_floor", now=1000.0 + i * 5.0).alert:
            fired = True
            break
    assert fired, "sustain clock was reset by thin traffic"


# -- contract with the notify service --------------------------------------


def test_attention_body_matches_notify_schema() -> None:
    """Field names are pinned against the REAL model, not a copy of it.

    orion-vision-host posts a plain dict rather than importing the notify
    schema. This test closes that gap: a rename in ChatAttentionRequest breaks
    here instead of silently posting a body the notify service drops.
    """
    repo = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    from orion.schemas.notify import ChatAttentionRequest

    for decision in (
        LivenessDecision(alert=True, reason="90% of vision tasks failing (gpu_hard_floor)",
                         fail_rate=0.9, sample_count=42, failing_for_sec=600.0,
                         top_error_code="gpu_hard_floor"),
        LivenessDecision(recovered=True, reason="vision tasks succeeding again",
                         fail_rate=0.0, sample_count=42),
    ):
        body = build_attention_request(decision, node_name="athena")
        model = ChatAttentionRequest(**body)          # raises if the shape is wrong
        assert model.source_service == "vision-host"
        assert model.message

    warn = ChatAttentionRequest(**build_attention_request(
        LivenessDecision(alert=True, reason="r", fail_rate=0.9, sample_count=42,
                         failing_for_sec=600.0, top_error_code="gpu_hard_floor"),
        node_name="athena"))
    assert warn.severity == "warning" and warn.require_ack is True

    ok = ChatAttentionRequest(**build_attention_request(
        LivenessDecision(recovered=True, reason="r", fail_rate=0.0, sample_count=42),
        node_name="athena"))
    assert ok.severity == "info" and ok.require_ack is False


def test_alert_message_names_the_real_diagnosis() -> None:
    """The message has to be actionable at 2am by someone who lost 21 hours."""
    body = build_attention_request(
        LivenessDecision(alert=True, reason="100% of vision tasks failing (gpu_hard_floor)",
                         fail_rate=1.0, sample_count=60, failing_for_sec=1200.0,
                         top_error_code="gpu_hard_floor"),
        node_name="athena")
    msg = body["message"]
    assert "cannot see" in msg
    assert "VRAM" in msg and "card actually installed" in msg
    assert body["context"]["top_error_code"] == "gpu_hard_floor"


# =========================================================================
# Regression tests for PR #1805 review findings. Each reproduces the exact
# scenario the reviewer demonstrated, so a revert re-breaks a named test.
# =========================================================================


def test_thin_traffic_recovery_clears_and_the_next_outage_still_alerts() -> None:
    """Finding 2 (HIGH): the min_samples early return skipped the clear branch.

    Reproduced by the reviewer: arm the watcher, then recover fully at a 40s
    task cadence -- 7 samples per 300s window, below the floor of 10. The
    watcher stayed alerting=True forever, and because the arm branch
    short-circuits on `if self._alerting`, a brand-new 100%-failure outage
    produced NO alert at all.

    The margin is thin in production: measured healthy cadence is 13 samples
    per window against a floor of 10.
    """
    # Short cooldown on purpose: with the 3600s default the second-outage
    # assertion below would be blocked by the cooldown rather than by the wedge,
    # and would pass for the wrong reason. (It did, on the first run.)
    w = _watcher(cooldown_sec=60.0)
    t = 0.0
    for _ in range(50):                       # arm
        w.record(ok=False, error_code="gpu_hard_floor", now=t); t += 5.0
    w.note_alert_delivered(True)
    assert w.snapshot(now=t)["alerting"] is True

    # Full recovery, but slowly -- 40s apart keeps the window under min_samples.
    for _ in range(20):
        w.record(ok=True, now=t); t += 40.0
    snap = w.snapshot(now=t)
    assert snap["sample_count"] < 10, "test setup wrong: should be below the floor"
    assert snap["alerting"] is False, "wedged alerting=True after a thin-traffic recovery"

    # A genuinely new outage must still alert.
    fired = False
    for _ in range(80):
        if w.record(ok=False, error_code="gpu_hard_floor", now=t).alert:
            fired = True
            break
        t += 5.0
    assert fired, "deaf to the next outage after a thin-traffic recovery"


def test_failed_delivery_rolls_back_and_retries() -> None:
    """Finding 3 (MEDIUM): state was committed on decision, not on delivery.

    A notify outage meant the watcher believed it had alerted, so
    `if self._alerting` suppressed every retry for the whole incident.
    """
    w = _watcher()
    t = 0.0
    decision = None
    for _ in range(60):
        d = w.record(ok=False, error_code="gpu_hard_floor", now=t)
        if d.alert:
            decision = d
            break
        t += 5.0
    assert decision is not None

    w.note_alert_delivered(False)             # the POST failed
    assert w.snapshot(now=t)["alerting"] is False, "did not roll back a failed send"

    # Next record retries immediately -- the failure run is still real, so the
    # sustain clock is deliberately NOT restarted.
    t += 5.0
    assert w.record(ok=False, error_code="gpu_hard_floor", now=t).alert, "did not retry"


def test_successful_delivery_commits_and_suppresses() -> None:
    w = _watcher()
    t = 0.0
    for _ in range(60):
        if w.record(ok=False, error_code="gpu_hard_floor", now=t).alert:
            break
        t += 5.0
    w.note_alert_delivered(True)
    assert w.snapshot(now=t)["alerting"] is True
    for _ in range(20):
        t += 5.0
        assert not w.record(ok=False, error_code="gpu_hard_floor", now=t).alert


def test_stale_clock_does_not_defeat_the_sustain_gate() -> None:
    """Finding 4 (MEDIUM): _failing_since never expired below the arm rate.

    Reviewer's reproduction: a 60s blip, then ~2.7h parked at a 50% failure
    rate (inside the hysteresis band), then a fresh outage -- which alerted at
    175s against a 180s sustain requirement, reporting "failing for 169 min"
    for an outage under three minutes old.
    """
    w = _watcher()
    t = 0.0
    for _ in range(12):                       # blip: sets the clock
        w.record(ok=False, error_code="gpu_hard_floor", now=t); t += 5.0

    for i in range(2000):                     # ~2.7h parked at 50%
        w.record(ok=(i % 2 == 0), error_code="gpu_hard_floor", now=t); t += 5.0
    assert w.snapshot(now=t)["alerting"] is False

    outage_start = t
    fired_at = None
    for _ in range(120):
        d = w.record(ok=False, error_code="gpu_hard_floor", now=t)
        if d.alert:
            fired_at = t
            assert d.failing_for_sec < 600.0, (
                f"reported failing_for_sec={d.failing_for_sec:.0f}s from a stale clock"
            )
            break
        t += 5.0
    assert fired_at is not None
    assert fired_at - outage_start >= 180.0, (
        f"alerted {fired_at - outage_start:.0f}s into a new outage, under the "
        f"180s sustain gate -- the stale clock defeated it"
    )


def test_failing_for_sec_handles_a_zero_timestamp() -> None:
    """Finding 8 (LOW): `if self._failing_since` treated a real 0.0 as absent.

    `now` is injectable and time.monotonic() has no guaranteed epoch.
    """
    w = _watcher()
    for i in range(12):
        w.record(ok=False, error_code="gpu_hard_floor", now=float(i) * 0.0)
    snap = w.snapshot(now=100.0)
    assert snap["failing_for_sec"] == 100.0, (
        f"failing_since=0.0 was read as 'not failing'; got {snap['failing_for_sec']}"
    )


def test_bad_config_degrades_instead_of_crashing_the_service() -> None:
    """Finding 1 (HIGH): a ValueError at import crashlooped the whole service.

    VisionHostService is constructed at module import, so a typo'd
    VISION_LIVENESS_* value made the alerting subsystem a brand-new way to
    cause the exact blackout it exists to detect -- and the enable flag did not
    bypass it. The ValueError stays (correct for a library); the service must
    degrade to known-good defaults instead of dying.

    Tested against the real helper `app.main` calls. It was deliberately moved
    out of app/main.py for this: importing app.main pulls in torch/PIL and
    skips on the host venv, and the fix for a HIGH finding should not be
    covered only by a test that does not run.
    """
    class _BadCfg:
        VISION_LIVENESS_WINDOW_SEC = 300.0
        VISION_LIVENESS_MIN_SAMPLES = 10
        VISION_LIVENESS_ARM_FAIL_RATE = 0.2
        VISION_LIVENESS_CLEAR_FAIL_RATE = 0.9      # clear > arm
        VISION_LIVENESS_SUSTAIN_SEC = 180.0
        VISION_LIVENESS_COOLDOWN_SEC = 3600.0

    w = build_watcher_or_default(_BadCfg())
    snap = w.snapshot(now=0.0)
    assert snap["arm_fail_rate"] == 0.8 and snap["clear_fail_rate"] == 0.2, \
        "did not fall back to the known-good defaults"

    class _MissingCfg:                              # attribute error, not ValueError
        pass

    assert isinstance(build_watcher_or_default(_MissingCfg()), VisionLivenessWatcher)


def test_good_config_is_not_silently_replaced() -> None:
    """The fallback must not swallow a VALID non-default config."""
    class _Cfg:
        VISION_LIVENESS_WINDOW_SEC = 111.0
        VISION_LIVENESS_MIN_SAMPLES = 4
        VISION_LIVENESS_ARM_FAIL_RATE = 0.7
        VISION_LIVENESS_CLEAR_FAIL_RATE = 0.3
        VISION_LIVENESS_SUSTAIN_SEC = 22.0
        VISION_LIVENESS_COOLDOWN_SEC = 33.0

    snap = build_watcher_or_default(_Cfg()).snapshot(now=0.0)
    assert snap["window_sec"] == 111.0
    assert snap["arm_fail_rate"] == 0.7 and snap["clear_fail_rate"] == 0.3
