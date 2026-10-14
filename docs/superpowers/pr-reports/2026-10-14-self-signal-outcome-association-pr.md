# feat(self-calibration): what a body signal tends to lead to (spec section E)

## Summary

- Orion can now learn, from its own records, what each of its body signals tends to lead to: "when this is high, my next hour had a failed turn 3.1x as often at equal activity (31% of 42 high hours vs 10% of 120 normal) [assoc:...]". Deterministic, no LLM.
- An hourly rollup in orion-equilibrium-service marks each body signal's hour as high / normal / unknown (high = the episode gate's own "unusual for longer than one event can explain" rule was met), counts that hour's failed turns, repaired replies and RPC timeouts, and stores one row per hour in `self_signal_hour` -- so the history outlives the 7-day prediction-error prune.
- From those hours it computes, per signal and outcome, the next-hour rate when high vs normal and an activity-adjusted ratio with a deliberately cautious 95% interval (adjusted for busy hours, for hours not being independent, and for testing 17 pairs at once), stored in `self_signal_outcome_association`.
- The clause renders in the self-reading line (stance_react / reverie) and in the episode open row's evidence **only** when both sides have at least 24 hours and the interval excludes 1. Pairs where the signal is built from the outcome itself are excluded.
- **Live today: nothing clears.** Only 23 of the last 167 hours have a full enough week of history to count, so every pair is below the minimum. Even counting all 167 hours with no minimum, no pair's interval excludes 1. Orion's prompts are unchanged until real evidence accumulates (about a month at today's rate of high hours).

## Outcome moved

Before: a self-reading said "this signal is high for me" with nothing about consequences. After: the same line can carry a cited, row-backed statement of what that state has historically preceded -- and stays silent when the data cannot support one. The persisted hourly rows start accumulating the evidence now.

## Current architecture

- `orion/attention/self_readings.py` (PR #2664) renders body signals against their own 7 days; `services/orion-thought/app/self_readings_cue.py` feeds stance_react / reverie.
- `orion/metacog/self_signal_episode.py` + `services/orion-equilibrium-service/app/self_signal_episode_gate.py` (PR #2662) open/close metacog rows when a signal stays unusual (> 300 s hot within 600 s).
- Calibration is the attention seam (`orion/attention/world_first.py` `node_candidate` + `judge_candidate`) over `substrate_node_prediction_error_history`, which is pruned at 7 days (live: min observed_at 2026-10-07 04:26, i.e. exactly 7 days back).
- Nothing linked a signal's state to what happened next.

## Architecture touched

- New pure reducer `orion/attention/self_signal_outcomes.py` (hour classification, outcome counts, activity-adjusted family-wise interval, circular-pair exclusion, clause).
- orion-equilibrium-service: `app/self_signal_outcome_rollup.py` rides the existing self_signal poll loop and its PE cache; reads `harness_turn_trace`, `transport_baseline_hourly`, `self_signal_hour` read-only; publishes on two new channels.
- orion-sql-writer: two new tables (created by `create_all` at boot, no manual migration).
- orion-thought: reads the newest association per (signal, outcome) within 3 h and passes it to the self-reading render.
- `orion/metacog/evidence_map.py`: the open row's `upstream.tends_to.clause` becomes one evidence line.

## Files changed

- `orion/attention/self_signal_outcomes.py`: the reducer.
- `orion/schemas/self_signal_outcome.py`: `SelfSignalHourV1`, `SelfSignalOutcomeAssociationV1`.
- `orion/schemas/registry.py`, `orion/bus/channels.yaml`: contract registration (both registry maps).
- `orion/inner_state_registry.py`, `orion/metrics/semantics.py`, `config/metrics/metric_definitions.lock.json`: semantic layer entry for the association (`ratio`, `rate_high`, `rate_normal` with rest/sparsity/absent_means and prompt sites), prompt inventory pin, re-lock.
- `orion/attention/self_readings.py`, `orion/schemas/self_reading.py`: optional clause on a reading (`tends_to`, `outcome_association_id`); render selects readings exactly as before, then adds clauses only while the block fits (a clause never costs a reading).
- `orion/metacog/evidence_map.py`: clause as an evidence line on the open row.
- `services/orion-equilibrium-service/app/{self_signal_outcome_rollup.py,service.py,settings.py,self_signal_episode_gate.py}`, `.env_example`, `docker-compose.yml`, `README.md`.
- `services/orion-thought/app/{self_readings_cue.py,settings.py}`, `.env_example`, `README.md`, `evals/self_readings_truth.py`.
- `services/orion-sql-writer/app/{models/self_signal_outcome.py,models/__init__.py,settings.py,worker.py}`, `.env_example`, `README.md`.
- `scripts/analysis/self_signal_outcome_truth.py`: independent truth-to-rows checker + live table (`--live`, `--persisted`).
- Tests/evals: `tests/test_self_signal_outcomes.py`, `tests/test_attention_self_readings.py`, `services/orion-equilibrium-service/tests/test_self_signal_outcome_rollup.py`, `.../tests/test_self_signal_episode_gate.py` (flag-off test now also turns the rollup off), `.../evals/test_self_signal_outcome_replay_eval.py` + fixture, `services/orion-thought/tests/test_stance_self_readings.py`, `.../evals/test_self_readings_truth_eval.py`, `services/orion-sql-writer/tests/test_self_signal_outcome_sql_shape.py`, `.github/workflows/self-signal-outcome-tests.yml`.

## Schema / bus / API changes

- Added: `SelfSignalHourV1` (`self_signal.hour.v1`) on `orion:equilibrium:self_signal:hour`; `SelfSignalOutcomeAssociationV1` (`self_signal.outcome_association.v1`) on `orion:equilibrium:self_signal:outcome_association`. Producer orion-equilibrium-service, consumer orion-sql-writer.
- Added (additive, in-process only): `SelfReadingV1.tends_to`, `SelfReadingV1.outcome_association_id`. SelfReadingV1 is built and consumed inside orion-thought only (not on the bus), so the extra=forbid consumer-first concern does not apply.
- Added (free-form dict): `MetacogTriggerV1.upstream.tends_to` on self_signal_episode **open** rows only, when a clause exists.
- Removed / renamed: none.
- Compatibility: flag off on both sides = no rows, no clause, prompts byte-identical (tested).

## Env/config changes

- Added keys: `EQUILIBRIUM_SELF_SIGNAL_OUTCOME_ASSOCIATION_ENABLED=true`, `EQUILIBRIUM_SELF_SIGNAL_OUTCOME_MIN_HOURS=24`, `EQUILIBRIUM_SELF_SIGNAL_OUTCOME_GRACE_SEC=600`, `CHANNEL_EQUILIBRIUM_SELF_SIGNAL_HOUR`, `CHANNEL_EQUILIBRIUM_SELF_SIGNAL_OUTCOME_ASSOCIATION` (orion-equilibrium-service); `SELF_SIGNAL_OUTCOME_ASSOCIATION_ENABLED=true` (orion-thought).
- Changed: `SQL_WRITER_SUBSCRIBE_CHANNELS` and `SQL_WRITER_ROUTE_MAP_JSON` (orion-sql-writer) gained the two channels / kinds (code also force-subscribes them).
- `.env_example` updated: yes (all three services).
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-equilibrium-service orion-thought`: yes (6 keys added to the primary checkout's local .env).
- skipped keys requiring operator action: none left -- the two sql-writer structured values were reported "diverged" by the sync script and were hand-added to the primary checkout's `services/orion-sql-writer/.env` after a backup (only the two entries appended; local ordering preserved). `check_env_template_parity.py`: PASS.

## Metric quality gate (outcome measures)

Every number the clause can show was traced: signals through the glossary (`check_metric_lineage.py --metric node:substrate.<x>.prediction_error` -> registered, value_kind/rest/polarity on record); outcomes are **not** in the semantic layer today (`--metric compliance_verdict`, `response_repair_ran`, `timeout_count` -> UNREGISTERED); the association itself is now registered (`self_signal_outcome_association.v1`) and the prompt-semantics gate passes.

| Outcome | 1. Provenance | 2. Independence | 3. Theory anchor | 4. Live sanity (rest state) | Verdict |
|---|---|---|---|---|---|
| Failed turn | `orion/harness/runner.py` compliance_verdict='failed' (motor failed: error with no partial, or cut short with nothing recorded) -> `harness_turn_trace.run_artifact` | **Circular with execution**: execution's surprise is a z-score of `execution_friction` (failed/started steps) and `failure_pressure` (run status fail) from the same runs (lineage: `execution_friction` -> `prediction_error.py:528`). Excluded. Independent of biometrics, bus_synaptic, cabinet, codebase, route, chat (chat = word count + repair pressure). | A failed turn is Orion's main action failing outright; "did the next hour have one" is the unit. | 14 days: 205 failed of 1,127 runs with a verdict; 23-28% of normal hours are followed by >=1 failure; most hours have none -> a genuine rest state. Not flat, not saturated. | PASS (execution pair excluded) |
| Repaired reply | `orion/harness/finalize.py` `needs_response_repair(reflection)` -> `run_artifact.response_repair_ran` | **Shared upstream with chat**: chat's surprise includes `repair_pressure_level`, which also forces the reply judge and the repair overlay (`finalize.py sensitive_turn_reason`). Excluded. **Circular with execution** (same runs). Excluded. | Orion's own post-draft judge decided the reply was misaligned / strained / uncertain and rewrote it. | 243 repaired of 1,127 runs; base rate high: 52-58% of normal hours are followed by one. Has a rest state (the other ~45%), but discriminates less. Note: it inherits the judge LLM's noise (the count itself is deterministic). | PASS with caveat |
| RPC timeout | `transport_baseline_hourly.timeout_count` (equilibrium's per-hop hourly accumulator over rpc_health snapshots) | **Shared upstream with bus_synaptic**: both read bus traffic; a timed-out RPC is a missing reply on a bus-synaptic edge (inter-arrival gap z-scores). Excluded. `#gpu_pool_wait` keys excluded: a GPU queue wait times out by design when the GPU is busy (321 of 1,684 timeouts all-time in the table). | A peer service failed to answer inside its deadline. | Bursty: one hop (`current_turn_probe`) logged 984 of 1,684 timeouts, so the unit is the hour (>=1 timeout), not the count. 10-12/10-13 had 2 and 0 timeouts all day -> genuine rest. | PASS (bus_synaptic pair excluded) |

5. Existing mechanism: none associated signal state with later outcomes; the transport gate and episode gate are reused, not copied. 6. Reversibility: two flags; two append/upsert tables with no consumers outside this patch; the semantic entry and lock lines are cheap to remove.

## How "high" is decided, and why (live data)

- The first version called an hour high when most readings were hot; it found 0 high execution hours. The second used "more than 300 s of hot time"; but a signal's top decile is ~10% of every hour by construction (~360 s), so a perfectly typical hour could be "high" (test `test_scattered_top_decile_readings_are_a_normal_hour`).
- Shipped rule: replay the episode gate's 30 s polls; an hour is high when the gate's own open dwell (> 300 s hot within 600 s) is met in it. Same constants as the gate (pinned by a test). Live 167 hours: biometrics 8 high, bus_synaptic 27, cabinet 16 (trigger fired), codebase 5, execution 16, route 10, chat 0 (event-written, two hot turns within 10 minutes never happened); unknown hours (no readable reading) are never counted as normal: chat 81, codebase 47, route 51.

## Minimum n (24 hours each side) and the interval

At the live base rates (failed turn ~26%, timeout ~28% of normal hours) 24 high hours give >= 6 expected events in the high group, the usual floor for a log-ratio approximation; one hour then moves the rate by ~4 points instead of 8-12 at n=8-12. On top of that (review fixes), the interval:

- uses the Mantel-Haenszel ratio over log2 bins of next-hour activity, so "busy hour after busy hour" is not read as an association (test `test_busy_next_hours_alone_do_not_make_an_association`), with the SE never narrower than the crude one;
- is widened for serial correlation (AR(1) of condition x outcome over consecutive hours; `se_inflation` on the row, 1.0-1.45 live);
- uses the Bonferroni z for every renderable pair (17 today: z = 2.97, family-wise 5%).

High hours accrue at ~1-4 per day per signal, so the first clause cannot appear for weeks, and only for a large effect.

## Live association table (2026-10-14)

Counted hours only (>= 144 h of PE history; the 7-day prune leaves 23 such hours today) -- what the producer will compute after deploy:

```text
replayed 167 hours 2026-10-07 06Z..2026-10-14 04Z, min_hours=24, min_history_hours=144 (23 hours eligible), Bonferroni z=2.97382
signal        outcome          high hrs normal hrs rate hi rate no MH ratio   adj. interval  infl  verdict
biometrics    turn_failed           0/0       4/16       -     25%        -               -  1.00  below_min_n
biometrics    response_repair       0/0       4/16       -     25%        -               -  1.00  below_min_n
biometrics    rpc_timeout           0/0       3/22       -     14%        -               -  1.00  below_min_n
bus_synaptic  turn_failed           2/3       2/13     67%     15%     5.50  (0.25, 118.80)  1.06  below_min_n
bus_synaptic  response_repair       0/3       4/13      0%     31%     0.00   (0.00, 44.44)  1.16  below_min_n
bus_synaptic  rpc_timeout           2/3       1/19     67%      5%        - (0.13, 1201.28)  1.45  excluded_circular
cabinet       turn_failed           0/0       4/16       -     25%        -               -  1.00  below_min_n
cabinet       response_repair       0/0       4/16       -     25%        -               -  1.00  below_min_n
cabinet       rpc_timeout           0/0       3/22       -     14%        -               -  1.00  below_min_n
chat          turn_failed           0/0       3/11       -     27%        -               -  1.00  below_min_n
chat          response_repair       0/0       4/11       -     36%        -               -  1.00  excluded_circular
chat          rpc_timeout           0/0       1/15       -      7%        -               -  1.00  below_min_n
codebase      turn_failed           1/1       3/15    100%     20%     3.50   (0.59, 20.70)  1.00  below_min_n
codebase      response_repair       0/1       4/15      0%     27%     0.00   (0.02, 40.99)  1.00  below_min_n
codebase      rpc_timeout           1/3       2/19     33%     11%     6.00  (0.14, 255.66)  1.20  below_min_n
execution     turn_failed           0/1       4/15      0%     27%     0.00   (0.02, 40.99)  1.00  excluded_circular
execution     response_repair       1/1       3/15    100%     20%        -   (1.08, 23.22)  1.00  excluded_circular
execution     rpc_timeout           0/1       3/21      0%     14%        -   (0.03, 79.44)  1.00  below_min_n
route         turn_failed           1/1       3/11    100%     27%     2.00    (0.45, 8.85)  1.00  below_min_n
route         response_repair       1/1       3/11    100%     27%     4.00   (0.30, 52.55)  1.00  below_min_n
route         rpc_timeout           0/2       2/15      0%     13%     0.00   (0.02, 71.85)  1.00  below_min_n
would render today: 0
truth-to-rows: PASS
```

Diagnostic (every replayed hour, no history floor -- these hours are stored but never counted):

```text
replayed 167 hours 2026-10-07 06Z..2026-10-14 04Z, min_hours=24, min_history_hours=0 (167 hours eligible), Bonferroni z=2.97382
signal        outcome          high hrs normal hrs rate hi rate no MH ratio   adj. interval  infl  verdict
biometrics    turn_failed           2/4     31/117     50%     26%     1.77   (0.26, 11.82)  1.22  below_min_n
biometrics    response_repair       2/4     64/117     50%     55%     0.86    (0.17, 4.37)  1.07  below_min_n
biometrics    rpc_timeout           3/8     45/157     38%     29%     1.15    (0.21, 6.29)  1.21  below_min_n
bus_synaptic  turn_failed          6/22      27/99     27%     27%     1.03    (0.33, 3.26)  1.01  below_min_n
bus_synaptic  response_repair      9/22      57/99     41%     58%     0.67    (0.30, 1.51)  1.00  below_min_n
bus_synaptic  rpc_timeout         10/27     38/138     37%     28%     1.24    (0.52, 2.92)  1.01  excluded_circular
cabinet       turn_failed          6/15     28/107     40%     26%     1.72    (0.52, 5.66)  1.11  below_min_n
cabinet       response_repair      9/15     57/107     60%     53%     1.19    (0.58, 2.44)  1.05  below_min_n
cabinet       rpc_timeout          6/19     43/147     32%     29%     1.02    (0.31, 3.41)  1.12  below_min_n
chat          turn_failed           0/0      16/67       -     24%        -               -  1.00  below_min_n
chat          response_repair       0/0      38/67       -     57%        -               -  1.00  excluded_circular
chat          rpc_timeout           0/0      27/86       -     31%        -               -  1.00  below_min_n
codebase      turn_failed           1/2      14/81     50%     17%     4.86   (0.53, 44.88)  1.00  below_min_n
codebase      response_repair       0/2      41/81      0%     51%     0.00   (0.01, 15.52)  1.00  below_min_n
codebase      rpc_timeout           1/5     28/114     20%     25%     0.81   (0.05, 14.17)  1.06  below_min_n
execution     turn_failed          1/11     29/104      9%     28%     0.25    (0.01, 5.23)  1.05  excluded_circular
execution     response_repair      8/11     55/104     73%     53%     1.42    (0.76, 2.66)  1.02  excluded_circular
execution     rpc_timeout          6/15     37/143     40%     26%     1.31    (0.45, 3.78)  1.03  below_min_n
route         turn_failed           4/9      21/79     44%     27%     1.16    (0.29, 4.63)  1.06  below_min_n
route         response_repair       6/9      44/79     67%     56%     0.87    (0.40, 1.88)  1.02  below_min_n
route         rpc_timeout          1/10     35/106     10%     33%     0.25    (0.01, 4.78)  1.04  below_min_n
would render today: 0
truth-to-rows: PASS
```

## Tests run

```text
tests/test_self_signal_outcomes.py tests/test_attention_self_readings.py tests/test_attention_pe_history_cache.py  64 passed
orion/metacog/tests/{test_evidence_map,test_evidence_map_producer_contract,test_self_signal_episode}.py  passed
services/orion-equilibrium-service: tests/test_self_signal_outcome_rollup.py tests/test_self_signal_episode_gate.py
  tests/test_docker_compose_env_parity.py evals/                                       48 passed
services/orion-thought: tests/test_stance_self_readings.py tests/test_reverie_thin_import_boundary.py
  evals/test_self_readings_truth_eval.py                                               33 passed, 3 skipped (pre-existing origin/main template skips)
services/orion-sql-writer: tests/test_self_signal_outcome_sql_shape.py tests/test_transport_baseline_hourly_sql_shape.py  10 passed
scripts/check_metric_lineage.py --gate / --prompt-semantics                            PASS / PASS
scripts/check_definition_drift.py --gate                                               PASS (after --update)
scripts/check_inner_state_registry.py                                                  OK (24 entries)
scripts/check_env_template_parity.py / check_env_key_single_source.py                  PASS / OK
scripts/check_service_env_compose_parity.py orion-equilibrium-service                  OK
repo tests/test_*catalog*,*registry*,*channel*,test_metric_*: 3 pre-existing failures unrelated to this branch
  (GoalProposalV1.drive_origin, PadRpc exec-result wildcard, gpu_pool single_consumer annotations)
```

## Evals run

```text
services/orion-equilibrium-service/evals/test_self_signal_outcome_replay_eval.py   5 passed
  7 days of stored PE history + exported outcomes -> every association recounted independently
  (scripts/analysis/self_signal_outcome_truth.py); nothing renders this week; circular pairs excluded;
  checker mutation cases fail as they should.
services/orion-thought/evals/test_self_readings_truth_eval.py   clause numbers resolve to their association row
python scripts/analysis/self_signal_outcome_truth.py --live      truth-to-rows PASS, 0 would render
```

## Docker/build/smoke checks

```text
docker compose --env-file .env --env-file services/orion-equilibrium-service/.env -f services/orion-equilibrium-service/docker-compose.yml config   rc=0 (new keys present)
orion-thought / orion-sql-writer compose files unchanged (env_file: services). No image build or `up` from the worktree (deploy is from the primary checkout).
Live runtime path UNVERIFIED until deployed: after deploy, `python scripts/analysis/self_signal_outcome_truth.py --persisted` checks the persisted rows.
```

## Review findings fixed

Code review ran in a subagent over `origin/main...HEAD`; no blockers. Every should-fix and the nits are fixed:

- Finding: 17 pairs re-tested hourly at 95% -> a false clause was likely over time.
  - Fix: Bonferroni z over every renderable pair (`family_z`, `n_comparisons`/`ci_z` on the row). Repeated looks over time remain (concern below).
  - Evidence: `test_bonferroni_widens_the_interval_for_the_whole_family`, `test_associate_all_counts_every_renderable_pair_as_one_family`.
- Finding: adjacent hours are not independent; Katz too narrow.
  - Fix: AR(1) SE inflation from consecutive hours (`serial_inflation`, `se_inflation` on the row).
  - Evidence: `test_clustered_hours_inflate_the_interval`; live 1.00-1.45.
- Finding: exposure confound (busy hours have more chances for one failure).
  - Fix: Mantel-Haenszel over log2 next-hour activity bins, SE floored at the crude Katz SE (Greenland-Robins collapsed on sparse strata in the replay); clause says "at equal activity".
  - Evidence: `test_busy_next_hours_alone_do_not_make_an_association` (crude 1.6x -> adjusted 1.0, no clause).
- Finding: cold-start backfill judged hours against a truncated week and kept them for 90 days.
  - Fix: `associate` ignores hours with `history_hours` < 144; rows stay stored and labelled; docstring corrected.
  - Evidence: `test_hours_judged_on_a_short_history_never_count`; live table above (23 eligible hours).
- Finding: orion-thought's association read sat inside the 0.5 s self-readings budget and the history lock.
  - Fix: background refresh thread (and startup warm); the turn path only reads the in-memory index.
  - Evidence: `test_a_stale_index_never_reads_on_the_callers_thread`.
- Nits fixed: event-written replay drops an event at age >= 300 s like the seam (`test_one_event_on_the_poll_grid_is_not_a_high_hour`); "never had" instead of "0.0x"; Haldane only on zero cells (`test_katz_only_corrects_zero_cells`); stale docstrings; a trigger still on from the previous hour is high; open rows cite only published associations (`published_ids`); the rollup runs after the gate step, skips history copies when nothing is due (`NodePeHistoryCache.earliest`), and rate-limits repeated failures; association table growth (~21 rows/hour, no retention) documented.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-sql-writer orion-equilibrium-service orion-thought; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build || break; done
```

Consumer-first: sql-writer creates the tables and subscribes, then equilibrium publishes, then orion-thought reads. No SQL migration. cortex-orch / orion-actions pick up the evidence-line change on their next rebuild (the extra upstream key is ignored until then).

## Risks / concerns

- Severity: medium. Concern: the association is re-tested every hour as data grows (repeated looks); Bonferroni covers the 17 pairs per look, not the looks. Mitigation: wide adjusted interval and min 24 hours; a follow-up could require the verdict to hold for N consecutive refreshes before rendering.
- Severity: low. Concern: the AR(1) inflation is a first-order approximation to the hours' dependence; a day-block bootstrap would be stricter.
- Severity: low. Concern: chat can essentially never be "high" (two hot chat turns within ~5 minutes); its hours without a turn are unknown. Live: 0 high, 86 normal, 81 unknown. Not degenerate as an input (it reads normal), but it will not produce a clause.
- Severity: low. Concern: the first deploy stores up to 7 days of backfilled hours that never count (short history); counting starts with the hours after deploy.
- Severity: low. Concern: `harness_turn_trace` holds every harness run (chat and self-initiated), so "a failed turn" is any harness turn, not only Juniper's chat turns.
- Severity: low. Concern: outcome fields (`compliance_verdict`, `response_repair_ran`, `timeout_count`) are not in the semantic layer; documented in the association's registry notes.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
