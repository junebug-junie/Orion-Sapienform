# Attention with stakes: two fixes, and curiosity starts paying for what it learns

Design: `docs/superpowers/specs/2026-09-25-attention-with-stakes-design.md` (gaps
G1-G10, defects D1-D2, proposals P1-P7). This PR ships the design, both defects,
and phase 1 of P1.

## Summary

- **Design doc.** How Orion's existing stakes (daily caps, GPU leases, Claude
  quota, Juniper's attention) could drive real attention, what is missing, and
  a phased plan. It went through adversarial review before any code was written.
- **D1: attention "novelty" no longer flickers on steady input.** Host and
  capability novelty compared this tick's pressure with last tick's *novelty*,
  so anything steady scored 0, p, 0, p forever. It now compares pressure with
  pressure. Targets over their per-kind cap used to vanish from the frame and
  read as brand new next tick. They are now kept in `suppressed_targets`,
  strongest first.
- **D2: the curiosity peer's Claude spend is gated on the Claude meter.** A
  local variable hid the Claude meter, so a clear *Cursor* reading was enough
  to spend Claude quota Juniper shares. In the container the Claude meter
  cannot be read, so the fallback now refuses (`claude_budget_unobserved`).
  That is fail-closed by design.
- **P1 phase 1: the curiosity spend log.** Every investigation run now records
  which beliefs it was shown, in what order, and at what expected value. It
  also records what the run changed, in nats of belief change, measured from
  Hub's own before and after snapshots.
  - That covers the "tested, nothing moved" case Orion's own revision nodes
    never capture.
  - A run that cannot be scored is recorded as unknown, with the reason.
- **P1 phase 1: value-ordered offer, off by default.** Beliefs are shown by
  expected belief change (uncertainty × measured learning yield) on a per-run
  random arm, so the two orders can be compared on real outcomes. It stays off
  until the replay script says the measurement is usable. The prompt describes
  the order in the same words on both arms.

## Outcome moved

- **Failure mode closed (D1).** Manufactured novelty on steady host and
  capability inputs. It fed dominant targets, proposal binding and the self
  model. Repro with the real functions, before the fix: `0.000, 0.800, 0.000,
  0.800, ...`. After: `0.800, 0.000, 0.000, ...`.
- **Failure mode closed (D2).** Claude quota spent on a Cursor reading.
- **New capability (P1).** Curiosity's most expensive act now leaves a record
  of what it bought. Before, nothing recorded the choice set, and a test that
  moved nothing left no trace (21 revisions across 94 journaled runs).
- **Simulation eval, not live evidence.** Over 24 simulated runs with the
  count rule off, value order spent 8 runs on flip-flopping beliefs where
  uncertainty order spent 20. It resolved 4 of 4 learnable beliefs; uncertainty
  order resolved 0. Under the live count rule (`stale_after=3`) the two orders
  tie (12 and 12). That is a finding about the count rule, recorded in the
  design doc, not a win.

## Current architecture

- The attention frame builder (`orion/attention/field_attention/`) ran
  Candidate B novelty for host and capability targets, as `|proxy_t −
  salience_{t−1}|`. Active targets past a per-kind cap were dropped from every
  bucket.
- The curiosity peer's `handle_help_request` bound a local `observe` (the
  Cursor meter) over the module-level Claude meter.
- Hub offered Orion its live priors most-uncertain first (`select_priors`). The
  prompt said so, and said "nothing here says which one is worth your time".
  Hub recorded neither the offer nor what the run changed. Orion writes
  `:PriorRevision` only when a confidence moves.

## Architecture touched

- **Attention builder.** Novelty formula; over-cap targets recorded; suppressed
  targets sorted. Runs in `orion-attention-runtime` and `orion-cortex-exec`.
- **Curiosity peer.** Fallback gate.
- **Hub curiosity loop.**
  - Offer arm, start and end snapshots, spend-log writes.
  - `valid_confidence` used for offer ordering.
  - New module `scripts/curiosity_offer_decisions.py`.
  - One prompt sentence, shared with self-inquiry.
- **Postgres.** Two new tables, via a manual migration, with Hub as the only
  writer.
- **Shared library.** `orion/curiosity/value.py` (new, pure); `worldview.py`
  gains an optional ordering hook and `Prior.fork_rank`.
- **Hub operator panel.** The field-attention suppressed-targets subtitle.

## Files changed

- `docs/superpowers/specs/2026-09-25-attention-with-stakes-design.md`: the design, as built, with both code-review rounds.
- `orion/attention/field_attention/scoring.py`, `selectors.py`, `candidate_society_of_mind.py`: D1 novelty diffs pressure against prior `pressure_score`.
- `orion/attention/field_attention/builder.py`: D1 over-cap targets kept in `suppressed_targets`; suppressed sorted strongest first.
- `orion/sentience_striving_program/instruments.yaml`, `README.md`: `rpt_lamme_recurrence` no longer claims `novelty_for_target` "needed no correction".
- `docs/architecture/attention-salience/README.md`: D1 history, including the one-tick transition exception.
- `scripts/analysis/measure_candidate_b_novelty_alternation.py` (+ tests): read-only replay of persisted frames that tells old-formula frames from new ones.
- `services/orion-curiosity-peer/app/worker.py` (+ `tests/test_fallback_once.py`, `README.md`): D2.
- `orion/curiosity/value.py` (+ `orion/curiosity/tests/test_curiosity_value.py`):
  - KL scoring, and entropy that validates its own input;
  - snapshot diff and attribution, with unknown reasons;
  - learning yield (credited progress, trajectory straightness, shrinkage);
  - the per-run arm.
- `orion/curiosity/worldview.py` (+ `tests/test_curiosity_worldview.py`):
  - optional `expected_nats_for` ordering, with a rounded key and uncertainty tie-break so the cold start is exactly today's order;
  - `Prior.uncertainty` uses `valid_confidence`;
  - `Prior.fork_rank`.
- `orion/curiosity/kickoff_prompt.py`: the sentence describing the order, now the same on both arms.
- `orion/autonomy/ask_claude_trigger.py`: comment only. It states why this gate still reads a broken confidence as settled.
- `services/orion-hub/scripts/curiosity_offer_decisions.py` (new): spend-log SQL and graph reads; the snapshot keeps the offer's fork copy.
- `services/orion-hub/scripts/curiosity_investigation.py`: arm, snapshots, outcome with `turn_ok` and `unknown_reason`.
- `services/orion-hub/scripts/main.py`, `app/settings.py`, `.env_example`: five `HUB_CURIOSITY_*` settings.
- `services/orion-hub/README.md`: "What each run bought: the spend log".
- `services/orion-hub/static/js/field-attention.js` (+ `tests/test_field_attention_operator_panel.py`): subtitle names both ways into "suppressed".
- `services/orion-hub/tests/test_curiosity_spend_log.py`, `tests/test_curiosity_investigation.py`, `evals/test_curiosity_value_order_eval.py`: loop tests on the real code, and the simulation eval.
- `services/orion-sql-db/manual_migration_curiosity_spend_v1.sql` (new): `curiosity_offer_decisions`, `curiosity_run_outcomes`.
- `scripts/analysis/replay_curiosity_realized_nats.py` (+ tests): read-only gate that decides whether value order may be switched on.
- `tests/test_attention_field_selectors.py`, `tests/test_attention_frame_builder.py`: D1 regressions.

## Schema / bus / API changes

- **Added.** Postgres tables `curiosity_offer_decisions` and
  `curiosity_run_outcomes`, including `turn_ok` and `unknown_reason` (manual
  migration). No bus channels, no registry entries, no HTTP routes.
- **Removed.** None.
- **Renamed.** None.
- **Behavior changed.**
  - `FieldAttentionFrameV1.suppressed_targets` now also holds active targets
    past a per-kind cap, with an explicit reason, strongest first. The schema is
    unchanged.
  - The substrate lattice's attention gate now finds an over-cap
    `capability:transport` in `suppressed_targets` and reads *pass* where it
    read *blocked*. The target was observed, so pass is the accurate reading.
  - A prior whose confidence is outside [0, 1] or not a number now counts as no
    confidence. It is offered as maximally uncertain, where before a 1.7 sank
    to the bottom and a NaN sorted by read order, and it drops to the bottom of
    the chat context's "current beliefs" list. The ask-Claude trigger
    deliberately still reads it as settled. Only broken-confidence priors move.
  - One sentence of the curiosity prompt changes on every run, including
    self-inquiry, which shares it:
    - Was: "the ones you were least sure about come first ... nothing here says
      which one is worth your time."
    - Now: "first come the ones where the code estimates a test could change
      your mind the most, starting from how unsure you are ... which one, if
      any, is worth your time is your call."
    - This is true of both orders and the same on both arms. It was the value
      arm's blocker.
- **Compatibility notes.**
  - Frames persisted before D1 store the same proxy in `pressure_score`, so
    the first post-deploy tick diffs like against like. The one exception is a
    target that was over cap in the last pre-fix frame: it reads fresh novelty
    for exactly one tick.
  - The migration has never been applied anywhere. It ships complete, and
    re-applying it upgrades an earlier copy (`ADD COLUMN IF NOT EXISTS`).

## Env/config changes

- **Added keys** (`services/orion-hub/.env_example`):
  - `HUB_CURIOSITY_SPEND_LOG_ENABLED=true`
  - `HUB_CURIOSITY_VALUE_ORDER_ENABLED=false`
  - `HUB_CURIOSITY_VALUE_ORDER_PROPENSITY=0.5`
  - `HUB_CURIOSITY_YIELD_WINDOW=3`
  - `HUB_CURIOSITY_YIELD_PSEUDO_TESTS=2.0`
- **Removed keys.** None.
- **Renamed keys.** None.
- **`.env_example` updated.** Yes. `settings.py` has matching fields, and Hub's
  compose `env_file` passes them through with no compose edit.
- **Local `.env` synced** with `python scripts/sync_local_env_from_example.py`:
  **not possible in this container.** The script reports "no .env found for any
  of the 35 services", because the live `.env` files exist only on the host.
  Run it on the host (restart section). Proven on a simulated `.env`: the
  default sync adds all five keys.
- **Skipped keys requiring operator action.** None.

## Tests run

```text
repo-level (every test importing a touched module, incl. situational +
            ask_claude_trigger, which read Prior.uncertainty)            615 passed
services/orion-hub  curiosity*, spend log, eval, field-attention panel,
                    cabinet, outreach, world_pulse_read*, substrate lattice 714 passed
services/orion-cortex-exec  attention_frame, curiosity reverie context    32 passed
services/orion-curiosity-peer  tests                                      45 passed
services/orion-durable-runs  tests, CI's venv + Postgres 16 DSN          142 passed
services/orion-durable-runs  evals: admission_fairness, gateway_capacity,
                             elastic_fairness                        3/3 exit 0
services/orion-hub/tests/test_graph_workbench.py (hub-graph-workbench CI) 18 passed
node --check services/orion-hub/static/js/field-attention.js              ok
git diff --check                                                          clean
static CI gates (mirror of .github/workflows/orion-static-gates.yml)      20/20 ok
```

Every regression test from both review rounds was checked against the old code
by putting the old line back, and each fails:

- The old sort key diverges from today's order on 95 of 2,000 random
  populations.
- Summing only recorded moves scores undone progress at 0.306.
- These old lines each fail their new test: the snapshot fork rule, the prompt
  sentence, the unvalidated entropy, the missing stamp check, the SQL gate,
  offer parsing, error matching, `turn_ok`, and the row cap.

## Evals run

```text
services/orion-hub/evals/test_curiosity_value_order_eval.py    3 passed
  no stale_after:  runs on flip-floppers  uncertainty 20 / value 8
                   learnable resolved     uncertainty 0/4 / value 4/4
  stale_after=3:   12 / 12, 0/4 / 0/4 (tie; finding recorded)
```

This is a simulation. The live comparison is Acceptance check 5, and it needs
`curiosity_run_outcomes` rows.

## Docker/build/smoke checks

```text
Real Postgres 16 (scratch cluster), the module's own functions:
  PASS missing table raises sqlstate 42P01; warns once, with the error text
  PASS migration idempotent; re-applying it adds turn_ok/unknown_reason to the
       first build's tables (CREATE TABLE IF NOT EXISTS alone would not)
  PASS decision row with NaN / 1.7 priors lands; NaN stored as null at ln2 entropy
  PASS control: jsonb rejects NaN (InvalidTextRepresentationError) -- the bug was real
  PASS unreadable start recorded; a retry fills the snapshot, keeps the first
       start time, and a stamped start scores unknown(start_stamped_by_this_run)
  PASS a later attempt cannot overwrite a taken snapshot
  PASS turn_ok / unknown_reason round-trip; a retry's upsert clears the reason
  PASS 20-day-old snapshot pruned (start time kept); fresh snapshot kept
  PASS missing column is NOT silenced as a missing table
replay_curiosity_realized_nats.py --pg against rows written by the Hub module:
  failed turn left out and counted per arm; unknown counted by reason; arms compared
```

- **Docker builds.** Not run: no Docker daemon in this cloud container. The
  change adds no dependencies, and no Dockerfile or compose file changed.
- **Live runtime.** `UNVERIFIED` for everything below; this container cannot
  reach the host's Postgres, FalkorDB or bus.
  - The D1 replay on persisted frames.
  - The first spend-log rows.
  - The D2 refusal in the running peer.

## Review findings fixed

**Design review (before code).** Nine findings, all addressed. The list is in
the design doc under "Review findings addressed".

**Code review of the build (subagent, with scratch-Postgres experiments):**

- **Finding: yield counted belief movement no recorded test made.** A jump
  between two scored tests read as progress, and straightness reached 4.0.
  - *Fix:* superseded by credited progress (next list).
  - *Evidence:* `test_yield_counts_only_moves_a_recorded_test_made`.
- **Finding: "no history = today's order" was false.** Float asymmetry, the
  entropy clamp, and a zero pool yield let the rotation hash decide.
  - *Fix:* the key is `(-round(expected, 9), uncertainty, times_tested, rotation)`.
  - *Evidence:* the 2,000-population test, mirror/clamp/broken cases, and the zero-value case.
- **Finding: a retry after an unreadable start turned unknown into a partial
  number.**
  - *Fix:* the diff refuses a start already stamped by the run (refined in the
    second round, below).
  - *Evidence:* `test_a_retry_after_an_unreadable_start_stays_unknown_not_partial`, and the Postgres smoke.
- **Finding: every "does not exist" error was silenced as a missing table, and
  a missing pool was silent.**
  - *Fix:* SQLSTATE 42P01; a missing pool warns once.
  - *Evidence:* unit tests, and the Postgres smoke.
- **Finding: 0.0 was ambiguous.**
  - *Fix:* `turn_ok`, which the replay filters on; an unscorable run is `NULL`.
  - *Evidence:* the failed-turn, unscorable and replay tests.
- **Finding: offer time and outcome time parsed differently, and a NaN made
  jsonb drop the whole row.**
  - *Fix:* `valid_confidence` everywhere, `allow_nan=False`, and `int(float(...))`.
  - *Evidence:* the broken-confidence tick test, and the Postgres control.
- **Finding: over-cap targets were unsorted, the subtitle was stale, and the
  README was off by one tick.**
  - *Fix:* sorted; subtitle and README reworded.
  - *Evidence:* the builder and panel tests.
- **Finding: the arm draw could equal 1.0.**
  - *Fix:* divided by 2³².
  - *Evidence:* `test_offer_arm_at_propensity_one_is_always_the_value_arm`.
- **Finding: forks, the 2000-row cap, and snapshots never pruned.**
  - *Fix:* a snapshot at the cap is unknown; snapshots are dropped after 14
    days; forks are fixed properly in the second round.
  - *Evidence:* the row-cap and prune tests, and the Postgres smoke.

**Second review, of the fixes.** It ran an exhaustive cold-start check over
100,006 confidences (zero divergences) and end-to-end retries on Postgres 16:

- **Finding (should-fix, introduced by the first fix): progress undone between
  tests scored as learning.** Tests pushing 0.5 → 0.7, knocked back off the
  record, scored 0.306 (4x a genuine learner) and lifted the pool 23x.
  - *Fix:* credited progress, meaning the smaller of the recorded and observed
    nets when they agree in sign, else 0. Jumps count toward path length, so
    straightness covers the whole trajectory.
  - *Evidence:* `test_progress_that_was_undone_between_tests_is_not_credited` (now 0.0079), and a 5,000-window property test (end point inside the observed range, straightness in [0, 1]).
- **Finding (should-fix, from the first build): the value arm told Orion
  "least sure first ... nothing here says which one is worth your time" above a
  list ordered by expected value.**
  - *Fix:* one sentence, true of both orders and the same on both arms.
  - *Evidence:* `test_both_arms_read_the_same_words_about_the_order`.
- **Finding: a tied fork was scored from a copy the offer never showed**
  (0.311 nats recorded for a 0.005 move).
  - *Fix:* the snapshot keeps the offer's copy (`Prior.fork_rank`).
  - *Evidence:* `test_a_forked_prior_is_scored_from_the_copy_the_offer_showed`.
- **Finding: `entropy_nats(1.7)` read as near-certain.**
  - *Fix:* it validates its own input.
  - *Evidence:* the parametrized entropy test.
- **Finding: the NULL and 0.0 wording was inexact, and NULL gave no reason.**
  - *Fix:* `unknown_reason` (five values), exact wording in the migration and
    README, and the replay counts unknowns by reason.
  - *Evidence:* reason assertions in the value, Hub and replay tests, and the Postgres round-trip.
- **Finding: the ask-Claude comment claimed a rule it no longer shares.**
  - *Fix:* the comment states the divergence. The behaviour is unchanged, so a
    broken number never spends Claude quota.
  - *Evidence:* `test_ask_claude_trigger*` pass.
- **Finding: gating the start on its time made every retry after a graph blip
  unknown.**
  - *Fix:* a retry may take the snapshot, and the stamp check alone keeps
    partial numbers out.
  - *Evidence:* `test_a_retry_after_an_unreadable_start_that_wrote_nothing_is_scored`, and the Postgres smoke.
- **Finding: a 42P01 query bug would read as "apply the migration".**
  - *Fix:* the warning carries the error text.
  - *Evidence:* the Postgres smoke.
- **Finding: the replay's graph summary dropped "2.0" counts and ignored the
  row cap.**
  - *Fix:* both handled.
  - *Evidence:* `test_graph_summary_reads_float_string_counts_and_flags_the_row_cap`.
- **Finding: the migration could not upgrade an earlier copy.**
  - *Fix:* `ADD COLUMN IF NOT EXISTS`.
  - *Evidence:* the Postgres smoke (the first build's DDL, then this migration).

## Restart required

```bash
# On the host, from the merged main checkout.
# 1. Tables for the spend log (idempotent), then confirm:
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_curiosity_spend_v1.sql
python3 scripts/check_sql_migrations_applied.py --file manual_migration_curiosity_spend_v1.sql
# 2. Local .env gets the five HUB_CURIOSITY_* keys:
python scripts/sync_local_env_from_example.py
# 3. Rebuild + restart what bakes the changed code:
scripts/safe_docker_build.sh orion-hub up -d --build              # spend log, value arm (off), prompt sentence, panel subtitle
scripts/safe_docker_build.sh orion-curiosity-peer up -d --build   # D2
scripts/safe_docker_build.sh orion-attention-runtime up -d --build  # D1 (persisted frames)
scripts/safe_docker_build.sh orion-cortex-exec up -d --build      # D1 (chat-scoped frames); Prior.uncertainty in the chat context
# 4. After ~24h: confirm D1 is what is live, from persisted frames:
POSTGRES_URI=... python scripts/analysis/measure_candidate_b_novelty_alternation.py --hours 24
# 5. After a week or more of runs: is the spend log usable, and how many runs does the arm test need?
python scripts/analysis/replay_curiosity_realized_nats.py --pg --graph
```

Leave `HUB_CURIOSITY_VALUE_ORDER_ENABLED=false` until step 5 reports USABLE with
a sample size you can reach, and its unknown-by-reason count looks healthy.

## Risks / concerns

- **Medium: the Claude fallback in the curiosity peer now refuses in
  production.**
  - *Concern:* this is the fix working. The container cannot read the Claude
    meter, so it fails closed. Tasks that used to fall back to Claude on a
    Cursor token failure will not.
  - *Mitigation:* this is intended and documented in the peer README.
    Re-enabling needs a readable Claude meter in the container, which is a
    separate decision.
- **Medium, pre-existing, not fixed here: the unconfirmed-dispatch fallback can
  hang a curiosity tick.**
  - *Concern:* only with `HUB_CURIOSITY_DURABLE_ADMISSION_ENABLED=false`. That
    is the settings default, but `.env_example` sets it true. If the runner's
    turn request registers before the tick's in-process fallback, the fallback
    joins that in-flight turn. The turn waits for `_run_lock`, which the tick
    holds.
    - Found by the second review and reproduced here with the Hub fakes (the
      tick never finishes).
    - Code: `curiosity_investigation.py`, the fallback near the
      `curiosity_durable_dispatch_fell_back` log and the in-flight join in
      `_turn_result_for`.
  - *Mitigation:* dormant if the host runs with durable admission on. The
    proposed follow-up is its own PR: the fallback should not join a turn that
    waits on the lock it holds.
- **Low: one prompt sentence changes on every curiosity and self-inquiry run.**
  - *Concern:* Orion reads slightly different words about the order.
  - *Mitigation:* the new sentence is accurate for both orders and still says
    the choice is Orion's. It is the precondition for a valid arm comparison.
- **Low: D1 changes live attention numbers.**
  - *Concern:* host and capability novelty drops on steady inputs, so
    dominant targets can change. That is the point, but downstream consumers
    will see different winners.
  - *Mitigation:* the measure script confirms the formula live. Consumers read
    the same schema.
- **Low: the spend log is on by default and writes two small rows per run,
  plus one snapshot.**
  - *Concern:* the snapshot is up to about 2,000 priors per run.
  - *Mitigation:* snapshots are pruned at 14 days. `HUB_CURIOSITY_SPEND_LOG_ENABLED=false`
    stops all writes, and every write is best-effort.
- **Low: the live comparison may be underpowered.**
  - *Concern:* the base rate says most tests move nothing.
  - *Mitigation:* the replay's verdict gates the switch. The design doc also
    records that the live count rule (`stale_after=3`) can erase value
    ordering's effect; replacing it is Missing question 8.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2355

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01QWbB2HV5XfJzpLbS3WfLn4
