# feat(self-calibration): what a body signal tends to lead to (spec section E)

## Summary

- Orion can now learn, from its own records, what each of its body signals tends to lead to: "when this is high, my next hour had a failed turn 3.1x as often (31% of 42 high hours vs 10% of 120 normal) [assoc:...]". Deterministic, no LLM.
- An hourly rollup in orion-equilibrium-service marks each body signal's hour as high / normal / unknown (high = the episode gate's own "unusual for longer than one event can explain" rule was met), counts that hour's failed turns, repaired replies and RPC timeouts, and stores one row per hour in `self_signal_hour` -- so the history outlives the 7-day prediction-error prune.
- From those hours it computes, per signal and outcome, the next-hour rate when high vs normal, the ratio, and a 95% interval, stored in `self_signal_outcome_association`.
- The clause renders in the self-reading line (stance_react / reverie) and in the episode open row's evidence **only** when both sides have at least 24 hours and the interval excludes 1. Pairs where the signal is built from the outcome itself are excluded.
- **Live today: nothing clears.** One week of hours, no signal x outcome pair has an interval that excludes 1 (even ignoring the minimum). Orion's prompts are unchanged until real evidence accumulates.

## Outcome moved

Before: a self-reading said "this signal is high for me" with nothing about consequences. After: the same line can carry a cited, row-backed statement of what that state has historically preceded -- and stays silent when the data cannot support one. The persisted hourly rows start accumulating the evidence now.

## Current architecture

- `orion/attention/self_readings.py` (PR #2664) renders body signals against their own 7 days; `services/orion-thought/app/self_readings_cue.py` feeds stance_react / reverie.
- `orion/metacog/self_signal_episode.py` + `services/orion-equilibrium-service/app/self_signal_episode_gate.py` (PR #2662) open/close metacog rows when a signal stays unusual (> 300 s hot within 600 s).
- Calibration is the attention seam (`orion/attention/world_first.py` `node_candidate` + `judge_candidate`) over `substrate_node_prediction_error_history`, which is pruned at 7 days (live: min observed_at 2026-10-07 04:26, i.e. exactly 7 days back).
- Nothing linked a signal's state to what happened next.

## Architecture touched

- New pure reducer `orion/attention/self_signal_outcomes.py` (hour classification, outcome counts, Katz interval, circular-pair exclusion, clause).
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

## Minimum n (24 hours each side)

At the live base rates (failed turn ~26%, timeout ~28% of normal hours) 24 high hours give >= 6 expected events in the high group, the usual floor for the log-ratio normal approximation; one hour then moves the rate by ~4 points instead of 8-12 at n=8-12. With 24 high vs ~140 normal hours, the interval only excludes 1 when the ratio is above ~2x. Today the most high hours any non-excluded pair has is 21 (bus_synaptic), so nothing can render yet -- and even at min 1 no interval excludes 1.

## Live association table (2026-10-14, 167 hours, 2026-10-07 05Z..2026-10-14 03Z)

`python scripts/analysis/self_signal_outcome_truth.py --live` (counts are "next hours with >=1 event / hours"):

```text
signal        outcome          high hrs normal hrs rate hi rate no  ratio    95% interval  verdict
biometrics    turn_failed           2/4     31/117     50%     26%   1.89    (0.68, 5.26)  below_min_n
biometrics    response_repair       2/4     64/117     50%     55%   0.91    (0.34, 2.47)  below_min_n
biometrics    rpc_timeout           3/8     45/157     38%     29%   1.31    (0.52, 3.31)  below_min_n
bus_synaptic  turn_failed          5/21     28/100     24%     28%   0.85    (0.37, 1.94)  below_min_n
bus_synaptic  response_repair      9/21     57/100     43%     57%   0.75    (0.45, 1.27)  below_min_n
bus_synaptic  rpc_timeout          9/26     39/139     35%     28%   1.23    (0.68, 2.23)  excluded_circular
cabinet       turn_failed          6/14     27/108     43%     25%   1.71    (0.86, 3.41)  below_min_n
cabinet       response_repair      8/14     58/108     57%     54%   1.06    (0.65, 1.73)  below_min_n
cabinet       rpc_timeout          6/16     42/150     38%     28%   1.34    (0.68, 2.65)  below_min_n
chat          turn_failed           0/0      15/66       -     23%      -               -  below_min_n
chat          response_repair       0/0      38/66       -     58%      -               -  excluded_circular
chat          rpc_timeout           0/0      26/85       -     31%      -               -  below_min_n
codebase      turn_failed           1/2      13/81     50%     16%   3.12   (0.71, 13.59)  below_min_n
codebase      response_repair       0/2      42/81      0%     52%   0.00    (0.03, 4.07)  below_min_n
codebase      rpc_timeout           1/5     27/114     20%     24%   0.84    (0.14, 5.03)  below_min_n
execution     turn_failed          1/12     29/103      8%     28%   0.30    (0.04, 1.98)  excluded_circular
execution     response_repair      9/12     54/103     75%     52%   1.43    (0.98, 2.08)  excluded_circular
execution     rpc_timeout          6/16     37/142     38%     26%   1.44    (0.72, 2.87)  below_min_n
route         turn_failed           4/9      20/78     44%     26%   1.73    (0.76, 3.94)  below_min_n
route         response_repair       6/9      44/78     67%     56%   1.18    (0.72, 1.95)  below_min_n
route         rpc_timeout          1/10     34/105     10%     32%   0.31    (0.05, 2.02)  below_min_n
would render today: 0
truth-to-rows: PASS
```

## Tests run

```text
tests/test_self_signal_outcomes.py tests/test_attention_self_readings.py               50 passed
orion/metacog/tests/{test_evidence_map,test_evidence_map_producer_contract,test_self_signal_episode}.py  passed
services/orion-equilibrium-service: tests/test_self_signal_outcome_rollup.py tests/test_self_signal_episode_gate.py
  tests/test_docker_compose_env_parity.py evals/                                       48 passed
services/orion-thought: tests/test_stance_self_readings.py tests/test_reverie_thin_import_boundary.py
  evals/test_self_readings_truth_eval.py                                               32 passed, 3 skipped (pre-existing origin/main template skips)
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

(filled in after review)

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-sql-writer orion-equilibrium-service orion-thought; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build || break; done
```

Consumer-first: sql-writer creates the tables and subscribes, then equilibrium publishes, then orion-thought reads. No SQL migration. cortex-orch / orion-actions pick up the evidence-line change on their next rebuild (the extra upstream key is ignored until then).

## Risks / concerns

- Severity: medium. Concern: hours are not independent (high hours cluster, failures cluster), so a Katz interval assuming independent hours is somewhat too narrow. Mitigation: hour-level binary outcome (not raw counts) and min 24 hours each side; a later patch could block-bootstrap by day.
- Severity: low. Concern: the first deploy backfills up to 7 days of hours whose percentile window is shorter than a week for the earliest hours (`history_hours` records it on each row). Mitigation: those rows are labelled; new hours always have a full week.
- Severity: low. Concern: `harness_turn_trace` holds every harness run (chat and self-initiated), so "a failed turn" is any harness turn, not only Juniper's chat turns.
- Severity: low. Concern: outcome fields (`compliance_verdict`, `response_repair_ran`, `timeout_count`) are not in the semantic layer; documented in the association's registry notes.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
