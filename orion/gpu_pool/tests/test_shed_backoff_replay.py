"""Eval gate for the shed backoff (orion/gpu_pool/evals/shed_backoff_replay.py): under a sustained
heat shed the pool sees a small fraction of the old attempts, and once the heat clears a steady
caller (vision-council, one window per 6 s) is served again within one capped window."""
from orion.gpu_pool.evals.shed_backoff_replay import replay, synthetic
from orion.gpu_pool.shed_backoff import CAP_SEC, JITTER


def test_sustained_shed_cuts_pool_attempts_per_hour():
    arrivals = synthetic(14.0)
    before = sum(1 for a in arrivals if a.caller == "vision-council") / 14.0
    after = replay(arrivals)["pool_attempts"]["vision-council"] / 14.0
    assert before > 500
    assert after <= 3600 / CAP_SEC * 1.5          # ~20/h at the cap, small slack for the ramp
    total_after = sum(replay(arrivals)["pool_attempts"].values()) / 14.0
    assert total_after < 0.05 * len(arrivals) / 14.0


def test_heat_clears_steady_caller_resumes_within_one_capped_window():
    worst = 0.0
    for clear in range(2 * 3600, 3 * 3600, 37):
        worst = max(worst, replay(synthetic(4.0), clear_at=clear)["backoff_delay_after_clear_sec"]["vision-council"])
    assert worst <= CAP_SEC * (1 + JITTER) + 6.0
