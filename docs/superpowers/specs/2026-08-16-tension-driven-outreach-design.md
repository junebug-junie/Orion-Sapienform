# Tension-driven endogenous outreach — replacing the coin flip

Status: implemented, tested. Ships with its code.

## Arsonist summary

`services/orion-hub/scripts/endogenous_outreach.py` ("Orion speaks first") has fired on a
coin flip since it shipped 2026-08-14. Its own docstring said so outright: *"Orion has no
endogenous 'I want to say something now' signal yet... fires on a randomized timer instead
of on a real motivational state,"* and `_should_roll()`'s docstring named itself the
sanctioned replacement seam: *"STUB. Replace with a real endogenous trigger when one
exists."* This patch is that replacement, using the deviation-tension package shipped in
PR #1699/#1701.

## What this deliberately does NOT claim

Raised directly by Juniper before any code was written, and it's correct: `DeviationGate`
is a change-detector, not a level-detector. A channel steadily overloaded for hours
re-centers its own EWMA baseline and reads as calm — the same "flood-starving" property
that made it useful for the original attention-starvation problem makes it structurally
blind to sustained-bad-but-stable states. So this trigger can only ever honestly claim
**"I noticed something change,"** never **"I am worried about the current state of
things."** A distress-shaped trigger needs a genuinely different, level-aware signal
(`orion.field.regime.channel_regime`, already built, unwired to anything — see the
companion design doc for Path 2) combined honestly with this one, not stapled onto it.
`build_outreach_prompt()`'s tension block is worded to this exact honest scope, and a test
(`test_tension_reason_prompt_never_claims_distress`) asserts the prompt never contains
"worried"/"concerned"/"distress"/"alarmed".

## Why persistence, not a leaky integrator

A continuous-decay accumulator ("build up an urge, discharge on send") was the first
design considered and was rejected — not on vibes, on precedent. `orion/substrate/
attention/goal_context.py` already tried exactly that shape for goal staleness and
rejected it in its own comment: a leaky-integrator decay *"would risk the same
saturation/floor bugs CLAUDE.md's metric-quality-gate already names twice"*
(`bus_synaptic_prediction_error`'s permanent 0.27 floor, `node:substrate.route`'s
decayed-to-zero-looks-calm). Existing-mechanism check, not skipped.

Instead: a bounded **consecutive-run-length count** on the already-computed
`tension_borda_winner_target_id`. No decaying state to get stuck at a floor, because
nothing decays — a run resets to zero the instant a different target wins or nothing is
admitted.

## Metric quality gate (CLAUDE.md 0A)

1. **Provenance.** `tension_borda_winner_target_id` traced to `orion.attention.tension.
   competition.TickResult.winner`, written once per digestion tick by
   `services/orion-field-digester/app/digestion/tension.py::update_tension_pressure()`.
2. **Independence.** Not a transform of `tension_deviation_pressure` (the scalar) —
   identity, not magnitude. Needed because a scalar-only trigger can't honestly say what
   changed; "deviation_pressure=0.6" is a number, "node:athena kept winning" is a reason.
3. **Theory anchor.** Consecutive-run persistence is a standard non-parametric way to
   distinguish a sustained pattern from noise without assuming a distribution — reuses the
   Borda winner identity `orion.attention.tension` already computes, no new statistic
   invented.
4. **Live-data sanity**, real, not guessed: replayed 2 hours (2,374 real ticks,
   2026-08-16) of `substrate_field_state` through the actual
   `FieldTensionCompetition`. Natural consecutive-same-winner run-length distribution
   (455 total runs): p50=3, p75=4, p90=5, p95=5, p99=8, max=11. `MIN_RUN_LENGTH=8` is the
   ~1st-percentile bar — a run this long happening by chance, not genuine persistence, is
   rare. Winner dominance itself (node:athena 42.9%, node:atlas 22.1%, node:circe 21.6% of
   1,330 non-none winners) was checked precisely because a naive "any repeat" bar would
   have fired constantly just from the dominant node winning by default; the run-length
   distribution (not raw win-share) is what the bar is set against.
5. **Existing mechanism.** Searched: no other Hub module computes anything like this; the
   Borda winner mechanism itself is reused verbatim from the already-shipped tension
   package, not reimplemented.
6. **Reversibility.** Deleting `tension_outreach_trigger.py` and reverting
   `_should_roll()`'s body to the coin flip is a two-file revert. `HUB_ENDOGENOUS_OUTREACH_
   PROBABILITY` was removed (not left dead) per CLAUDE.md's kill-means-kill rule — a genuine
   revert restores that key too, disclosed as a real (small) reversibility cost.

## The design

- `services/orion-hub/scripts/tension_outreach_trigger.py` (new): `current_run()` reads the
  already-computed `tension_borda_winner_target_id`/`tension_deviation_pressure` columns
  directly from recent `substrate_field_state` rows (does NOT replay
  `FieldTensionCompetition` live — that's for the offline measurement only; the field
  digester already did this work once per tick). Walks backward from the latest tick,
  stops at the first different/missing winner, returns `None` below `MIN_RUN_LENGTH=8`.
  Never raises — a DB failure degrades to "no reason to fire", the honest failure mode for
  a broken trigger.
- `FieldStateV1.tension_borda_winner_target_id` (new field): the missing piece from
  PR #1699 — that patch deliberately didn't persist Borda-winner identity because it had no
  consumer yet (CLAUDE.md 0A keyword-cathedral rule). This patch is the consumer.
- `EndogenousOutreach.__init__` drops `probability`/`rng`, gains `trigger_evaluator:
  Callable[[], Optional[TensionTriggerReason]]` (defaults to the real `current_run`,
  lazily imported to keep a hard Postgres dependency out of this module's import time,
  matching the file's existing convention for `curiosity_hint`/`hub_presence`). Tests
  inject a fake evaluator directly — no probability/rng seam to fake through anymore.
- `_should_roll()` (now `async`, running the evaluator via `asyncio.to_thread` so a slow
  Postgres call cannot stall Hub's single event loop — a review fix, see below) calls the
  evaluator, stores the result on `self._last_tension_reason` (readable from `status()` for
  operator visibility), returns whether it fired. A `force=True` debug trigger, and any
  earlier gate (quiet_hours/daily_cap/cooldown/turn_in_flight) blocking before the evaluator
  ever runs, both explicitly clear `self._last_tension_reason` first, so `status()` never
  misattributes a stale organic episode to right now.
- `OutreachContext` gains `tension_reason`, counted as real grounding by `is_empty()` (a
  tension reason alone is enough to generate a prompt, same as curiosity summaries alone
  already were). `build_outreach_prompt()` renders it first, honestly scoped as above.
- `services/orion-hub/scripts/pg_engine.py` (new): one cached SQLAlchemy engine shared by
  `current_run()` and `endogenous_outreach._fetch_recent_turns` — this patch's own two new
  same-tick DB reads, not a repo-wide migration of `orion-hub`'s ~20 other pre-existing
  `create_engine` call sites (review fix, see below).

## Non-goals

- No level-aware ("I'm worried") trigger. Real, separate follow-up — see the Path 2 design
  doc, not built here.
- No target-binding into `orion/proposals/builder.py`'s attention path — the Borda winner
  identity is consumed here (Hub outreach) and nowhere else yet.
- `MIN_RUN_LENGTH`/`LOOKBACK_MINUTES` are disclosed, derived-but-uncalibrated constants
  (real replay data, not a guess, but not yet validated against real post-deploy firing
  rate). Revisit once outreach has actually fired for real, same discipline as every other
  constant in this arc.

## Acceptance checks

1. `services/orion-hub/tests/test_tension_outreach_trigger.py` (12 tests): empty window,
   no-winner tick, run shorter than bar, run meeting the bar, a stale different-winner run
   not extending the current one, a quiet-tick break resetting the run, magnitude reported
   as the run's max not just the latest value, malformed-value degradation, DB-failure
   degradation, `min_run_length` override, and the schema-field contract check (review fix).
2. `services/orion-hub/tests/test_endogenous_outreach.py`: updated factory (`trigger_
   evaluator` replaces `probability`/`rng`), new tests for no-trigger/broken-evaluator/
   forced-call-clears-stale-reason, tension-reason-alone-is-grounding, the
   never-claims-distress wording guard, event-loop-not-blocked, and stale-status-after-a-
   blocked-tick (the last two are review fixes). 88 tests pass total in this file + the new
   one.
3. `services/orion-field-digester/tests/test_tension_pressure_baseline.py`: 3 new tests for
   `tension_borda_winner_target_id` (None on quiet tick, names the real winner on a real
   spike, tracks whichever of two nodes actually admits). 12 tests pass.
4. Metric definition drift gate re-locked and passing.
5. `check_service_env_compose_parity.py orion-hub`: N/A (Hub uses `env_file:` wholesale) —
   confirmed, not assumed.

## Review findings fixed (2026-08-18)

8 finder angles ran against the diff; the top finding was independently
surfaced by 4 of 8.

- **`_should_roll()` blocked the event loop**: the default evaluator is a
  synchronous Postgres round trip, called with no `asyncio.to_thread` wrap on
  Hub's single uvicorn worker -- a slow query would have frozen every
  connected websocket and in-flight chat turn for its duration. Fixed:
  `_should_roll()` is now `async` and runs the evaluator via
  `asyncio.to_thread`, same pattern `_gather_context`'s `_safe()` already
  uses two methods down. Regression test: `test_trigger_evaluator_does_not_
  block_the_event_loop`.
- **`status()`'s `last_tension_reason` went stale** when an earlier gate
  (quiet_hours/daily_cap/cooldown/turn_in_flight) blocked a tick before
  `_should_roll()` ever ran, misreporting a possibly-hours-old episode as
  live. Fixed: the blocked branch now clears `_last_tension_reason`, mirroring
  the `force` branch's existing precedent. Test: `test_status_does_not_
  report_a_stale_tension_reason_after_a_blocked_tick`.
- **Duplicate connection pools on one tick**: this patch's own two new
  same-tick DB reads (`tension_outreach_trigger.current_run` and
  `endogenous_outreach._fetch_recent_turns`) each built/cached a separate
  engine against the identical database. Fixed: `scripts/pg_engine.py` (new,
  narrow -- one cached engine, not a repo-wide migration of the ~20 other
  pre-existing `create_engine` call sites in `orion-hub/scripts/*.py`, which
  is real, disclosed, pre-existing debt out of scope for this patch).
- **`latest_deviation_pressure` misnamed**: the field holds the run's *peak*
  value (a running max), not the latest tick's. Renamed to
  `peak_deviation_pressure` everywhere (dataclass, `status()`, tests).
- **`MIN_RUN_LENGTH` was a hardcoded constant** with no way to retune it from
  real post-deploy firing data without a code deploy. Made operator-tunable:
  `HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH` (settings.py, default 8, wired via
  `functools.partial` in `main.py`). Test: `test_min_run_length_is_
  overridable_per_call`.
- **No contract test on the raw `field_json->>'...'` SQL keys**: nothing
  caught a future rename of `tension_borda_winner_target_id`/`tension_
  deviation_pressure` on `FieldStateV1` breaking this query silently at
  runtime. Added `test_raw_sql_json_keys_match_the_real_schema_fields`.
- **`LOOKBACK_MINUTES`'s comment overstated coupling** to field-digester's
  actual poll cadence (a separate service's own env knob). Reworded to
  disclose this is an intentionally generous, decoupled margin, not a
  precise derivation.
- **Stale `settings.py` section header** still called this "stub random
  trigger" after the diff replaced it. Updated.
- **`.env` sync gap**: review correctly flagged no evidence in-diff that the
  live `.env` was kept in sync. Re-verified directly: the live shared-checkout
  `.env` still had `HUB_ENDOGENOUS_OUTREACH_PROBABILITY=0.15` (an earlier
  session's removal had not survived a later sync run reading the still-old
  `.env_example` from the primary checkout -- see `feedback_env_sync_reads_
  example_from_primary_checkout` in this repo's own memory). Fixed directly:
  removed the stale key, added `HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH=8`.

Not fixed, disclosed as real debt: the ~20 other pre-existing `POSTGRES_URI`/
`create_engine` call sites elsewhere in `orion-hub/scripts/*.py` still don't
use `scripts/pg_engine.py`. Migrating them is an unrelated, invasive change
across files this patch has no other reason to touch -- real follow-up work,
not silently dropped.

## What this does and does not establish

It establishes that Orion's outreach can now fire on a real, inspectable, honestly-scoped
signal instead of chance, with a persistence bar derived from real history rather than
guessed. It does **not** establish that `MIN_RUN_LENGTH=8` is well-calibrated for real
outreach cadence, or that a change-noticed message is what Juniper actually wants to
receive unprompted — both are real open questions to watch against live firing data, not
resolved by this patch.
