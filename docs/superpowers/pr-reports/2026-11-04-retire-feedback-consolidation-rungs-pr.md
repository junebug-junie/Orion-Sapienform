# Retire the feedback (L10) and consolidation (L11) rungs, with no orphans

## Summary

- **Two ladder services go:** the feedback runtime and the consolidation runtime, along with
  their packages, schemas, config, tests, and smoke and analysis scripts. That is about 15,600
  lines deleted.
- **Orion's one live world act keeps its learning.** That act is "pause background GPU when the
  cabinet warms", run as a trial with a held-back control group. Its settle-time scorer moves,
  with the same logic, into orion-sql-writer, beside the act-ledger resolver.
- **The retention trap is cut.** sql-writer stops reading the "feedback pending" mark. Nothing
  clears that mark any more, so if sql-writer kept reading it, proposal-frame pruning would
  silently stop forever.
- **Every reader is cut or repointed:**
  - Spark intake;
  - Hub routes, and the lattice page, which shows the two rungs as "retired";
  - ladder liveness, which lists them as retired;
  - metrics liveness;
  - the schema registry, bus channels and the inner-state registry;
  - the physiology contract;
  - the env sync script;
  - CI;
  - the READMEs.
- **The reverie keeps its episode grounding.** Its switch is renamed to what it now does
  (`ORION_REVERIE_GROUND_EPISODES`, on) and loses the consolidation half.
- **Two things retire with their only data source:** the feedback-credit watch
  (`credit_integrity`) and the reverie "action usefulness" metric.

Spec: PR #3108 (`docs/superpowers/specs/2026-11-04-retire-feedback-and-consolidation-rungs.md`).
Juniper's decisions (11-04):
- drop the tables 2 days after deploy, with a backup first;
- dispatch is the next rung to retire;
- the Hub shows "retired";
- no orphans.

## Outcome moved

- **About 43,000 feedback frames a day stop.** The table holds 9.7 GB, with no retention.
- **23 consolidation frames a day and their side tables stop.**
- **About 86,000 empty Spark bus artifacts a day stop:** two for every feedback frame.
- **Two always-running containers go.**
- **Nothing Orion uses to decide changes:**
  - the act ledger still grades every real act;
  - the world act still learns;
  - the reverie still grounds in episodes.

## Current architecture (before)

- **Feedback.** It graded every ladder dispatch frame. It wrote outcome rows for four ladder kinds
  (summarize, maintain, inspect, self_regulate), about 390 in 14 days, and scored world-action
  episodes. The world-action episodes are the only part of its work that is still live.
- **Consolidation.** It learned motifs from feedback frames. Its only live reader was the
  reverie's motif references.
- **The sql-writer retention floor.** It probed `feedback_pending`, a column DEFAULT TRUE that
  only the feedback runtime cleared.

## Architecture touched

| Piece | Change |
|---|---|
| `orion/autonomy/world_settlement.py` | Moved from `orion/feedback/` with `git mv`. It now owns `claim_upheld` and `PRESSURE_DELTA_EPSILON`. |
| `orion/autonomy/world_settlement_store.py` | New: the store and the pass, moved from the feedback runtime's worker/store with the same SQL and guards. |
| `services/orion-sql-writer/app/world_settlement_loop.py` | New loop. It runs first and then sleeps. Setting: `SQL_WRITER_WORLD_SETTLEMENT_INTERVAL_SEC=30`. |
| `services/orion-sql-writer/app/grammar_truth.py` | The `dispatch->feedback` floor probe is removed. |
| `orion/substrate_ladder_liveness.py` + `scripts/check_substrate_ladder_liveness.py` | The two rungs, and the empty-motif check, are removed. `RETIRED_RUNGS` is printed and added to the JSON. |
| Hub `substrate_lattice_routes.py`, `static/js/substrate-lattice.js`, `substrate-lattice.html` | L10 and L11 show `retired` with what replaced them, and their tables are never queried. |
| `services/orion-thought` | The grounding switch is renamed and `motif_refs` is removed. Old stored thoughts still parse, because extra keys are ignored. |
| `orion-spark-concept-induction` | `orion:feedback:frame` is removed from its intake, its compose file and its env template. |

## Files changed

- **Deleted:**
  - `services/orion-feedback-runtime/`, `services/orion-consolidation-runtime/`;
  - `orion/feedback/`, `orion/consolidation/`, `orion/schemas/{feedback,consolidation}_frame.py`;
  - `config/{feedback,consolidation}/`, `orion/field/credit_integrity.py`;
  - Hub `substrate_{feedback,consolidation}_routes.py`;
  - their tests, `smoke_{feedback_frame,consolidation}_v1.sh`,
    `measure_proposal_feedback_correlation.py`, `eval_action_value_contrast.py`.
- **Stripped of feedback-only parts:**
  - tests: `tests/test_{randomized_holdback,action_value_contrast}.py`,
    `test_custom_route_result_status.py`;
  - scripts: `scripts/{check_merge_domination,eval_capability_unmeasured_replay,eval_field_decisions_replay}.py`.
- **Contracts:**
  - `orion/bus/channels.yaml`, `orion/schemas/registry.py`;
  - `orion/inner_state_registry.py`, `scripts/check_inner_state_registry.py`;
  - `config/physiology/contract.yaml`;
  - `config/metrics/metric_definitions.lock.json` (re-locked);
  - `orion/schema_skew_discovery.py`;
  - the header of `manual_migration_feedback_control_reduction.sql` (now names dispatch only).
- **CI:** `attend-act-loop.yml` (now runs the moved scorer test in sql-writer) and
  `visual-baseline.yml`.
- **Comments and docs:** the READMEs (root, sql-writer, dispatch, field-digester) and the comments
  that pointed at the deleted code.

## Schema / bus / API changes

- **Removed:**
  - channel `orion:feedback:frame`;
  - the schemas `FeedbackFrameV1`, `OutcomeObservationV1`, `ConsolidationFrameV1`,
    `MotifObservationV1`, `ExpectationV1`, `SparseTensorSliceV1`, `SchemaCandidateV1`;
  - the Hub routes `/api/substrate/feedback/*` and `/api/substrate/consolidation/*`;
  - `ReverieThoughtV1.motif_refs`.
- **Behavior changed:**
  - L10 and L11 report `retired`;
  - world scoring runs in sql-writer.
- **Compatibility:** the reverie schema ignores extra keys, so stored thoughts that carry
  `motif_refs` still parse. No forbid-model consumer reads the removed schemas.

## Env/config changes

- **Added:** `SQL_WRITER_WORLD_SETTLEMENT_INTERVAL_SEC=30` (sql-writer).
- **Renamed:** `ORION_REVERIE_GROUND_CONSOLIDATION` becomes `ORION_REVERIE_GROUND_EPISODES`
  (thought, on).
- **Removed:**
  - `orion:feedback:frame` from Spark's `BUS_INTAKE_CHANNELS`;
  - the `FEEDBACK_*` and `ORION_WORLD_SETTLEMENT_*` keys, which went with their service.
- **Sync script:** its prefixes now cover `SQL_WRITER_WORLD_SETTLEMENT_`. A test proves the key
  reaches the local `.env`.
- **Local `.env`:** synced by hand in the main checkout. The new sql-writer and thought keys are
  added, and the Spark channel is removed. The sync script cannot reach those files from a
  worktree.
- **Skipped keys:** the old thought key is left in `.env` until deploy, then removed (see
  Restart).

## Tests run

```text
static-gates CI job, every step run locally from the workflow file: 42/42 PASS
tests/scripts/test_substrate_ladder_liveness.py: 63 passed
orion-sql-writer tests: 1190 passed, 183 skipped (main: 1 failed, 1184 passed)
orion-thought tests: 811 passed, 32 skipped
orion-dream 338 passed; orion-execution-dispatch-runtime 128 passed; orion-spark 12 passed; orion-proposal-runtime 13 passed
hub: test_substrate_lattice_routes 65 passed, test_route_engine_reuse 10 passed
tests/test_world_settlement_scoring.py + test_world_action_dispatch.py: 26 passed
orion/autonomy agency episode tests 25 passed; orion/reverie efficacy 9 passed
root tests + orion/ collection: 51 collection errors, the same set as main (environment), none new
full root tests + orion/ (continue-on-collection-errors): 14391 passed, 113 failed.
  All 113 failing files re-run on main: 92 of the 113 fail there too.
  The other 21 re-run on this branch in isolation:
    - 19 pass (they only fail in the full suite: test-order effects, which main has too);
    - 1 fails on main as well (test_self_study_analysis_wiring, proposal template count);
    - 1 was mine: test_metric_liveness expected 5 ladder stages; now 3. Fixed (56 passed).
```

## Evals run

```text
orion/autonomy/evals/run_agency_episode_eval.py: 10/10 (old capture still parses)
orion/reverie/evals/test_reverie_efficacy_resonance_eval.py: pass
```

## Docker/build/smoke checks

```text
docker build -f services/orion-sql-writer/Dockerfile .: OK
in the image: import app.main, app.world_settlement_loop, orion.autonomy.world_settlement_store: OK
in the image: orion.feedback absent (find_spec None)
live read-only, WorldSettlementStore against conjourney:
  due 0, posterior cells 87, control cells 71, cabinet points (30 min) 58, incidents (7 d) 0
scripts/check_sql_migrations_applied.py --service orion-execution-dispatch-runtime: all 3 required applied
scripts/check_migration_required_by.py: OK
```

## Review findings fixed

The code review ran in a subagent at high effort. It found no blocker. It confirmed the moved world
scorer is line-for-line the old behaviour (same SQL, guards and idempotency), and that the dropped
NULL-dedup branch is unreachable.

- **Finding:** dispatch still withheld 20% of eligible ticks (`ORION_DISPATCH_HOLDBACK_FRACTION=0.2`)
  for a control arm only the feedback runtime folded. That made it orphaned behaviour that delays
  real acts.
  - **Fix:** the per-tick holdback branch and `tick_can_yield_control` are removed. The key now sets
    only the proven-effect act holdback, which still has a reader (`substrate_act_episodes`). Its
    comments in `.env_example`, settings and the README say so.
  - **Evidence:** `test_no_tick_is_withheld_after_the_feedback_rung_retired`, fraction 1.0, draw 0:
    the tick is sent. Dispatch suites: 75 + 128 passed.
- **Finding:** the world settle loop logged only when it scored something, so a dead loop and an
  idle loop looked the same. No episode has been due since 10-25.
  - **Fix:** a `[WORLD_SETTLEMENT] alive passes=.. scored=.. failed=..` line once an hour.
  - **Evidence:** `test_the_loop_says_it_is_alive_even_when_nothing_is_due`.
- **Finding:** stale text pointed at the retired services. It was in the root README (service list),
  the substrate-runtime README (ladder table, which also claimed a dream consumer that does not
  exist), `orion_shed_store.py` and `contrast.py`.
  - **Fix:** all four now name the new home.
- **Finding:** the two containers are still up, the leftover local `.env` files remain, and the
  thought `.env` still has the old key.
  - **Fix:** each has an exact command in Restart required.
- **Not changed:** the reviewer suggested `REQUIRED-BY: none` for
  `manual_migration_feedback_control_reduction.sql`. That is wrong: the same file creates
  `substrate_execution_dispatch_inflight`, which dispatch writes (`store.py:376,513`). So
  dispatch-only is the correct header.

Found and fixed during build, before review:
- **A syntax error that would have crashed sql-writer at boot.** The new loop's start line sat
  between an `if` and its `else` in `app/main.py`. It was caught by
  `tests/test_boot_ddl_guarded.py`.
  - Fix: the line is moved out.
  - Guard: the wiring test now parses `main.py` with `ast`. The earlier text match passed on the
    broken file.
- **The new key would never have reached the local `.env`.** The sync script copies only listed
  prefixes.
  - Fix: the `SQL_WRITER_WORLD_SETTLEMENT_` prefix is added, with a test.
- **A vacuous test.** The lattice "never queries the retired tables" test first checked mock
  calls, which don't record SQL.
  - Fix: it now spies on the query helper, and it asserts that the spy did see other tables.
- **The deploy gate would have refused a service.** The migration-header gate flagged a header
  that named the deleted services.
  - Fix: the header now names dispatch, which needs that file's in-flight table.

## Restart required

Deploy from the clean primary `main` after merge, when no chat turn is running.
1. Run the post-pull rebuild as usual (sql-writer, thought, spark, hub, dispatch, dream, field-digester and proposal pick up the changes).
2. Stop and remove the two retired containers (no compose file is left for them):
```bash
docker stop orion-athena-feedback-runtime orion-athena-consolidation-runtime && docker rm orion-athena-feedback-runtime orion-athena-consolidation-runtime
```
3. Remove the old thought key from the local `.env`:
```bash
sed -i '/^ORION_REVERIE_GROUND_CONSOLIDATION=/d' services/orion-thought/.env
```
4. Delete the two orphaned local `.env` files the deleted service directories leave behind (needs Juniper's OK):
```bash
rm services/orion-feedback-runtime/.env services/orion-consolidation-runtime/.env && rmdir services/orion-feedback-runtime services/orion-consolidation-runtime
```
5. In 2 or more days: the drop PR (DROP_PR_LINK). Follow its header: check that nothing has written since, take the backup, then apply.

## Risks / concerns

- **Medium: dispatch's learning freezes for four ladder kinds.**
  - What happens: the posterior cells for summarize, maintain, inspect and self_regulate stop
    updating.
  - Why it is acceptable: dispatch is the next rung to retire (Juniper). Its summarize calls are
    boilerplate: 516 LLM calls in 14 days.
  - Mitigation: the dispatch-retirement proposal is next.
- **Low: the ladder check loses its "running on nothing" symptom check.** That check counted empty
  consolidation motifs. The freshness checks on every rung below still caught the 09-20 incident,
  and that replay test still passes.
- **Low: the world act is UNVERIFIED in its new home.**
  - What is not verified yet: the first live score from sql-writer. No episode has been due since
    10-25.
  - What is verified: the loaders work live, and the pass has tests.
  - How to check: watch for a `[WORLD_SETTLEMENT] scored=` line in the sql-writer logs after the
    next shed.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
