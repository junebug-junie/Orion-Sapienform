"""Pure timing/retry policy for the topic-foundry scheduler loop.

Lives apart from ``main.py`` so the behaviour can be tested without starting
the app. The loop in ``main.py`` used to sleep a full interval BEFORE its
first tick; at the 86400s production interval that required Hub to stay up 24
unbroken hours to fire even once, and it never had (confirmed live
2026-08-28, zero `_tick` lines ever emitted). Every function here is pure --
no I/O, no clock, no settings access.

See docs/superpowers/specs/2026-08-28-concept-induction-topic-model-rebuild-design.md.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

# Short grace period before the startup tick so it does not race the rest of
# app startup. Not an env knob: nothing an operator would tune, and the
# startup tick itself is already gated.
STARTUP_TICK_DELAY_SEC = 30.0

# How many times to re-arm the startup tick before falling back to the normal
# interval. On a full-stack `docker compose up`, Hub and orion-topic-foundry
# start together -- if topic-foundry is not answering yet, the first tick
# fails and without this the next attempt would be a whole interval away,
# reproducing the exact failure this patch exists to remove.
STARTUP_TICK_MAX_ATTEMPTS = 4

# Trigger outcomes that mean "topic-foundry was not reachable/ready", as
# opposed to "there was legitimately nothing to do". Only the former is worth
# retrying -- see trigger_topic_foundry_training_run()'s return shape.
RETRYABLE_TRIGGER_REASONS = frozenset(
    {
        "dataset_or_model_resolution_failed",
        "topic_foundry_base_url_not_configured",
        "topic_foundry_trigger_failed",
    }
)


def next_wait_seconds(
    *,
    startup_pending: bool,
    interval_sec: float,
    attempt: int = 0,
    startup_delay_sec: float = STARTUP_TICK_DELAY_SEC,
) -> float:
    """Seconds to wait before the next tick.

    The whole point of the patch: when a startup tick is pending, the wait is
    the short startup delay, NOT ``interval_sec``. Retries back off
    geometrically from the same base so a slow-starting topic-foundry gets a
    few widening chances well inside one interval.
    """
    if not startup_pending:
        return max(0.0, float(interval_sec))
    delay = max(0.0, float(startup_delay_sec)) * (2 ** max(0, int(attempt)))
    # Never let a backed-off startup retry exceed the normal cadence -- at
    # that point the interval tick is the better deal.
    return min(delay, max(0.0, float(interval_sec)))


def should_retry_startup_tick(summary: Optional[Mapping[str, Any]], attempt: int) -> bool:
    """Whether a failed startup tick should be re-armed rather than waiting a
    full interval.

    ``summary`` is ``trigger_topic_foundry_training_run``'s return dict, or
    ``None`` when the call raised. A tick that actually triggered is done; a
    tick that failed for a reason not in ``RETRYABLE_TRIGGER_REASONS`` (e.g.
    the scheduler is disabled) is not worth retrying either.
    """
    if attempt + 1 >= STARTUP_TICK_MAX_ATTEMPTS:
        return False
    if summary is None:
        return True  # the call raised -- almost always "not up yet"
    if summary.get("triggered"):
        return False
    return str(summary.get("reason") or "") in RETRYABLE_TRIGGER_REASONS
