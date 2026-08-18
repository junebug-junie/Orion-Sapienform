"""Field-deviation-tension trigger for endogenous outreach.

Replaces `endogenous_outreach.py::_should_roll()`'s coin flip -- see that
function's own docstring, and its module docstring, both of which have named
this as the sanctioned replacement seam since 2026-08-14: "Orion has no
endogenous 'I want to say something now' signal yet... fires on a randomized
timer instead of on a real motivational state."

WHAT THIS DOES NOT CLAIM
-------------------------
`orion.attention.tension.DeviationGate` is a change-detector, not a
level-detector: a channel that has been steadily overloaded for hours
re-centers its own EWMA baseline and reads as calm, by design (the same
"flood-starving" property that made it useful for the original attention-
starvation problem). This trigger inherits that limit honestly rather than
papering over it -- it can only ever claim "I noticed something change",
never "I am worried about the current state of things". A distress-shaped
outreach message needs a genuinely different, level-aware signal
(`orion.field.regime.channel_regime`, unwired to anything today) combined
honestly with this one -- real, separate follow-up work, not built here.

WHY PERSISTENCE, NOT A LEAKY INTEGRATOR
-----------------------------------------
A naive continuous-decay accumulator ("build up an urge, discharge on send")
was considered and rejected. `orion/substrate/attention/goal_context.py`
already tried exactly that shape for goal staleness and rejected it in its
own comment: a leaky-integrator decay "would risk the same saturation/floor
bugs CLAUDE.md's metric-quality-gate already names twice"
(`bus_synaptic_prediction_error`'s permanent 0.27 floor,
`node:substrate.route`'s decayed-to-zero-looks-calm). Instead: a bounded
*consecutive-run-length* count on the already-computed
`tension_borda_winner_target_id` -- no decaying state to get stuck at a
floor, because nothing decays. A run resets to zero the instant a different
target wins or nothing is admitted; there is no persisted accumulator to
saturate.

WHERE THE BAR CAME FROM
-------------------------
`MIN_RUN_LENGTH = 8` is not guessed. Replayed 2 real hours (2,374 ticks,
2026-08-16) of `substrate_field_state` through the real
`FieldTensionCompetition` and measured the natural distribution of
consecutive-same-winner run lengths: p50=3, p75=4, p90=5, p95=5, p99=8,
max=11 (455 total runs). A run of 8 or more happening by chance -- not
genuine persistence -- is roughly a 1st-percentile event against that one
snapshot of `FieldTensionCompetition`'s tuning. If that tuning drifts later,
the distribution this bar rests on drifts with it and nothing here would
notice -- so unlike the trigger's other internals, this one constant IS
operator-tunable (`HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH`, wired in
`scripts/main.py`) rather than baked in, precisely so it can be retuned from
real post-deploy firing-rate data without a code change/deploy. The module
constant below is the derived default, not a hardcoded floor.
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.pg_engine import get_engine as _engine

# Derived default -- see "WHERE THE BAR CAME FROM" above. Real callers get
# their value from HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH (settings.py) via
# `scripts/main.py`'s `functools.partial(current_run, min_run_length=...)`;
# this module-level constant is only the fallback for direct/test callers.
MIN_RUN_LENGTH = 8

# How far back to look for the current run. Deliberately NOT derived from
# orion-field-digester's actual poll cadence -- that is a separate service on
# its own env knob (`RECEIPT_POLL_INTERVAL_SEC`), and orion-hub has no
# business importing another service's internals to track it precisely. 10
# minutes is generously larger than any plausible MIN_RUN_LENGTH-tick run
# even at a much slower cadence than today's ~2s; if the digester's cadence
# ever changes, the failure mode of this margin being too small is a query
# that looks further back than strictly necessary (cheap), not one that
# silently truncates a real run's start.
LOOKBACK_MINUTES = 10.0


@dataclass(frozen=True)
class TensionTriggerReason:
    """A real, inspectable reason to reach out -- never fabricated.

    `target_id` is always the same node_id `orion.attention.tension`'s Borda
    competition already produces (e.g. "node:athena") -- never a generic
    placeholder. A caller with no real target gets `None` from `current_run`,
    not a reason with an empty/fake target.
    """

    target_id: str
    run_length: int
    # The RUN's peak deviation_pressure, not the latest tick's -- see
    # `current_run`'s loop below, which tracks a running max, not the last
    # value seen. Named `peak_`, not `latest_`, so a reader of this field
    # cannot mistake a several-ticks-old high-water mark for "the value right
    # now".
    peak_deviation_pressure: float


def _fetch_recent_winners(limit_minutes: float) -> list[tuple[str | None, float]]:
    """(winner_target_id, deviation_pressure) pairs, oldest first.

    Reads the already-computed columns `orion-field-digester` wrote once per
    real digestion tick -- does NOT replay `FieldTensionCompetition` itself
    (that machinery is for offline validation, like the measurement that
    derived `MIN_RUN_LENGTH` above; the live producer already did this work).
    """
    engine = _engine()
    if engine is None:
        return []
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT field_json->>'tension_borda_winner_target_id' AS winner,
                       field_json->>'tension_deviation_pressure' AS deviation
                FROM substrate_field_state
                WHERE generated_at > now() - make_interval(mins => :mins)
                ORDER BY generated_at ASC
                """
            ),
            {"mins": limit_minutes},
        ).fetchall()
    out: list[tuple[str | None, float]] = []
    for winner, deviation in rows:
        try:
            dev = float(deviation) if deviation is not None else 0.0
        except (TypeError, ValueError):
            dev = 0.0
        out.append((winner, dev))
    return out


def current_run(
    *, limit_minutes: float = LOOKBACK_MINUTES, min_run_length: int = MIN_RUN_LENGTH
) -> TensionTriggerReason | None:
    """Is Orion's field state, RIGHT NOW, in the middle of a sustained
    persistence episode on the same target?

    Walks backward from the most recent tick and stops at the first tick
    with a different (or missing) winner -- only the run ending AT THE
    LATEST TICK counts. An episode that already ended does not re-fire on a
    later poll just because it is still inside the lookback window.

    Never raises: a DB failure, an empty window, or a malformed row all
    degrade to `None` ("no reason to fire"), matching this module's own
    "never fabricate a reason" contract -- the failure mode of a broken
    trigger must be silence, not a false positive.
    """
    try:
        rows = _fetch_recent_winners(limit_minutes)
    except Exception:  # noqa: BLE001 - a DB hiccup must not crash the outreach tick
        return None
    if not rows:
        return None
    latest_winner, _latest_deviation = rows[-1]
    if not latest_winner:
        return None

    run_length = 0
    max_deviation_in_run = 0.0
    for winner, deviation in reversed(rows):
        if winner != latest_winner:
            break
        run_length += 1
        max_deviation_in_run = max(max_deviation_in_run, deviation)

    if run_length < min_run_length:
        return None
    return TensionTriggerReason(
        target_id=latest_winner,
        run_length=run_length,
        peak_deviation_pressure=max_deviation_in_run,
    )
