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
papering over it -- on its own it can only ever claim "I noticed something
change", never "I am worried about the current state of things".

COMBINED WITH THE LEVEL-AWARE SIGNAL (2026-08-19)
---------------------------------------------------
The level-aware half named above as future work now exists and is wired in:
`orion.field.significance.sustained_load_pressure` (PR #1718,
`orion.field.regime.channel_regime`'s `loaded_steady` regime, no adaptive
baseline). `TensionTriggerReason` below carries the LATEST tick's value of
it alongside the run's own peak deviation -- read straight off the same
`substrate_field_state` row this trigger already queries, not recomputed
here. This is honestly scoped GLOBAL, not per-target: `sustained_load_
pressure` is a `max()` over every `loaded_steady` channel/node in the
significance window, so a nonzero reading here means "something, somewhere
in the field is genuinely under sustained load right now", not "the SAME
node this run's target_id names is". `build_outreach_prompt`
(`endogenous_outreach.py`) states both numbers as separate real facts and
lets generation draw the connection -- it does not narrate a feeling on
Orion's behalf.

IDENTITY NOW CARRIED TOO (2026-09-07). `sustained_load_pressure` used to
ship no per-node identity -- `orion.field.significance` computed which
(channel, node_id) produced the max reading and then threw it away at the
line that collapsed it to a bare float. Root-caused live: Orion sent an
unprompted message naming a specific internal channel
("harness_closure prediction error") that was not in the context it was
given and had read 0.0/NULL for the prior 24h -- the real driver that tick
WAS `sustained_load_pressure` (`node:athena`), but the prompt could only say
"somewhere... a channel", and the generation model filled that gap with a
plausible, wrong, real-sounding invented name. `TensionTriggerReason` now
also carries `sustained_load_pressure_channel`/`_node_id`, read off the same
row, same LATEST-tick-not-run-tracked semantics as the scalar itself. Still
honestly scoped GLOBAL: naming the channel/node that is loaded does not
claim it is the SAME node `target_id` names -- `build_outreach_prompt`
states both as separate facts, same as before.

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
`MIN_RUN_LENGTH` is not guessed. Original derivation (2026-08-16): replayed
2 real hours (2,374 ticks) of `substrate_field_state` through the real
`FieldTensionCompetition` and measured the natural distribution of
consecutive-same-winner run lengths: p50=3, p75=4, p90=5, p95=5, p99=8,
max=11 (455 total runs) -- landed on 8, since a run that long happening by
chance, not genuine persistence, was roughly a 1st-percentile event against
that one snapshot of `FieldTensionCompetition`'s tuning.

That tuning DID drift, exactly as this paragraph warned it might, and
nothing here noticed until an operator asked "why hasn't Orion reached out"
a second time (2026-08-22) and the drift was checked by hand. RECALIBRATED
2026-08-22: same replay technique, re-run against the trailing 24h of live
`substrate_field_state` (3,625 raw runs, gaps-and-islands over the RAW tick
sequence -- a NULL or different-winner tick correctly breaks a run, matching
`current_run`'s own semantics below, not a same-winner-only filter that
would silently stitch across real gaps): p50=3, p75=4, p90=5, p95=5, p99=6,
max=10. The field is genuinely noisier now (more nodes actively competing
for the Borda win, including `node:rpc_timeout`; 55.96% of ticks in a
sampled trailing hour had no winner at all) -- the old bar of 8 had drifted
from "top 1%" to roughly "top 0.2%" (8 qualifying runs out of 3,625 that
same 24h). 6 is the new default, restoring the original ~1st-percentile
selectivity against TODAY's distribution rather than 2026-08-16's.

Unlike the trigger's other internals, this one constant IS operator-tunable
(`HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH`, wired in `scripts/main.py`)
rather than baked in, precisely so it can be retuned from real post-deploy
firing-rate data without a code change/deploy -- exactly what happened here.
The module constant below is the derived default, not a hardcoded floor;
re-derive it again the same way if `FieldTensionCompetition`'s tuning drifts
further. See docs/superpowers/pr-reports/2026-08-22-outreach-min-run-length-
recalibration-pr.md for the full before/after numbers and the query used.
"""

from __future__ import annotations

from dataclasses import dataclass

# Reused, not re-copied -- `_safe_float` already implements exactly the
# "None/malformed degrades to a caller-chosen default" contract this module
# needs, and this would otherwise be the THIRD hand-rolled copy of it in this
# codebase (that docstring names the other two: attention_broadcast.py's
# `_f()`, dynamics.py). Not promoted to a public name / moved to a shared
# module in this patch -- that would touch `falkor_codec.py` for no other
# reason this patch already has, same scope call this module's own design doc
# already made for its "3 near-duplicate hand-rolled SQL fetches" finding.
# `orion/` is the shared package (CLAUDE.md section 5's acceptable
# cross-service seam), not another service's internals -- Hub already
# imports `orion.field.significance`/`orion.attention.tension.*` the same
# way.
from orion.substrate.falkor_codec import _safe_float
from scripts.pg_engine import get_engine as _engine

# Derived default -- see "WHERE THE BAR CAME FROM" above. Real callers get
# their value from HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH (settings.py) via
# `scripts/main.py`'s `functools.partial(current_run, min_run_length=...)`;
# this module-level constant is only the fallback for direct/test callers.
MIN_RUN_LENGTH = 6

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
    # The LATEST tick's (not the run's) sustained_load_pressure -- see the
    # module docstring's "COMBINED WITH THE LEVEL-AWARE SIGNAL" section. No
    # "peak_" tracking here: unlike deviation_pressure this is already a
    # ~30s-throttled, carried-forward gauge read off one field_json column,
    # not a per-tick series worth taking a running max over.
    #
    # 0.0 here means EITHER "no channel field-digester's significance
    # producer sees is loaded_steady right now" (a real calm reading) OR "the
    # field_json row that produced this reason predates PR #1718 and has no
    # `sustained_load_pressure` key at all" (SQL NULL, collapsed to 0.0 by
    # `current_run` -- see its own comment). The two cases are NOT
    # distinguishable on this field once constructed; `_fetch_recent_winners`
    # keeps them apart as `float | None` for any caller that needs to. This
    # is a narrow, self-healing ambiguity (LOOKBACK_MINUTES=10.0 caps how
    # long a pre-migration row can still be "latest") -- CLAUDE.md's own
    # metric-quality-gate names exactly this failure shape
    # (missing-looks-like-calm) by incident, so it is disclosed here rather
    # than silently assumed away.
    sustained_load_pressure: float = 0.0
    # Identity of the (channel, node_id) that produced
    # `sustained_load_pressure` above (2026-09-07) -- see the module
    # docstring's "IDENTITY NOW CARRIED TOO" section for the incident this
    # closes. Both `None` in EXACTLY the same two cases `sustained_load_
    # pressure`'s own 0.0 already covers: a genuine "nothing loaded_steady
    # right now" reading, or a pre-migration `field_json` row with no such
    # keys at all -- this dataclass does not distinguish those two cases any
    # more for identity than it already declines to for the scalar (see that
    # field's comment). A caller that needs the distinction reads
    # `_fetch_recent_winners`'s raw `float | None` / `str | None` tuple
    # instead, same precedent already set for the scalar.
    sustained_load_pressure_channel: str | None = None
    sustained_load_pressure_node_id: str | None = None


def _fetch_recent_winners(
    limit_minutes: float,
) -> list[tuple[str | None, float, float | None, str | None, str | None]]:
    """(winner_target_id, deviation_pressure, sustained_load_pressure,
    sustained_load_pressure_channel, sustained_load_pressure_node_id)
    tuples, oldest first. `sustained_load_pressure` is `None` when the row's
    `field_json` has no such key (a pre-`PR #1718` row) or the value is
    malformed -- kept distinct from a genuine `0.0` reading here; `current_run`
    is where that distinction collapses (see `TensionTriggerReason`'s own
    comment for why collapsing it there, not here, is deliberate). The two
    identity strings follow the exact same None-means-either-quiet-or-
    pre-migration convention (2026-09-07) -- a row with a real
    `sustained_load_pressure` but a missing identity key (pre-identity-
    migration, post-#1718) reads as `(value, None, None)`, same shape as a
    genuinely quiet 0.0 row; nothing here needs to tell those apart, since
    the only real consumer (`current_run`) already treats "no identity to
    report" the same way regardless of which caused it.

    Reads the already-computed columns `orion-field-digester` wrote once per
    real digestion tick -- does NOT replay `FieldTensionCompetition` or
    `orion.field.significance.compute_tick` itself (that machinery is for
    offline validation, like the measurement that derived `MIN_RUN_LENGTH`
    above; the live producers already did this work).
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
                       field_json->>'tension_deviation_pressure' AS deviation,
                       field_json->>'sustained_load_pressure' AS sustained_load,
                       field_json->>'sustained_load_pressure_channel' AS sustained_load_channel,
                       field_json->>'sustained_load_pressure_node_id' AS sustained_load_node_id
                FROM substrate_field_state
                -- secs, not mins: make_interval's mins arg is int, so a
                -- float LOOKBACK_MINUTES raises UndefinedFunction -- secs is
                -- double precision and takes it directly. Confirmed LIVE,
                -- 2026-08-18: this call had been silently failing on every
                -- invocation since deploy, swallowed by this module's own
                -- "never raise, degrade to None" contract -- the trigger has
                -- never actually been able to fire. Same convention
                -- services/orion-hub/scripts/chat_history_rehydrate.py's own
                -- comment already documents for this exact pitfall.
                WHERE generated_at > now() - make_interval(secs => :secs)
                ORDER BY generated_at ASC
                """
            ),
            {"secs": limit_minutes * 60.0},
        ).fetchall()
    out: list[tuple[str | None, float, float | None, str | None, str | None]] = []
    for winner, deviation, sustained_load, sustained_load_channel, sustained_load_node_id in rows:
        # deviation's 0.0 default is intentional arithmetic, not an honesty
        # claim -- an absent/malformed reading contributes nothing to the
        # run's max() below, same as a genuine 0.0 would.
        dev = _safe_float(deviation, default=0.0)
        # sustained_load's default is None, not 0.0 -- see this function's
        # own docstring and `TensionTriggerReason.sustained_load_pressure`'s
        # comment for why the None/0.0 distinction is kept this far and no
        # further.
        load = _safe_float(sustained_load, default=None)
        # Plain strings, not `_safe_float` -- `->>'...'` already returns
        # `None` for a missing/JSON-null key, and an empty string never
        # occurs (the producer writes either a real node_id/channel string
        # or omits the key), so no coercion is needed here.
        channel = sustained_load_channel
        node_id = sustained_load_node_id
        out.append((winner, dev, load, channel, node_id))
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
    (
        latest_winner,
        _latest_deviation,
        latest_sustained_load,
        latest_sustained_load_channel,
        latest_sustained_load_node_id,
    ) = rows[-1]
    if not latest_winner:
        return None

    run_length = 0
    max_deviation_in_run = 0.0
    for winner, deviation, _sustained_load, _channel, _node_id in reversed(rows):
        if winner != latest_winner:
            break
        run_length += 1
        max_deviation_in_run = max(max_deviation_in_run, deviation)

    if run_length < min_run_length:
        return None
    # Identity belongs to the LATEST tick, same as the scalar itself -- not
    # accumulated/tracked across the run (see TensionTriggerReason's own
    # comment). A quiet or pre-migration latest tick collapses both to
    # None/None, same convention the scalar already uses for 0.0.
    is_quiet_or_missing = latest_sustained_load is None
    return TensionTriggerReason(
        target_id=latest_winner,
        run_length=run_length,
        peak_deviation_pressure=max_deviation_in_run,
        # None (missing key / pre-#1718 row) collapses to 0.0 here, same as
        # genuine calm -- both mean "nothing to add to the prompt" to every
        # real caller, so the type-level distinction `_fetch_recent_winners`
        # keeps has done its job by the time it reaches this object.
        sustained_load_pressure=(
            0.0 if latest_sustained_load is None else latest_sustained_load
        ),
        sustained_load_pressure_channel=(
            None if is_quiet_or_missing else latest_sustained_load_channel
        ),
        sustained_load_pressure_node_id=(
            None if is_quiet_or_missing else latest_sustained_load_node_id
        ),
    )
