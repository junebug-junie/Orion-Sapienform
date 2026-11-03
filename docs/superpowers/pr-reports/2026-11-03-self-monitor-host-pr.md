# Self-monitor organ, patch 2: host and publish (no effects)

## Summary

This is step 2 of the self-monitor organ. The organ is Orion's track record of their own faculties.
Patch 1 built it as a pure library and a report script. This patch runs it live, every hour.

- Orion-durable-runs now evaluates the organ once an hour. This happens in a new `self_monitor` step of
  the Temporal Self thread, after the `immune` step.
- Each evaluation reads 7 days of Orion's settled bets. It then works out a status for each faculty.
  A new status must hold for 3 evaluations in a row before it counts. This stops one bad hour from
  flipping a faculty.
- The result goes to Redis (`orion:self_monitor:latest`, kept for 3 hours) and to a new endpoint,
  `GET /self_monitor/state`. One copy per hour also goes to a new history table (kept for 30 days).
- When a faculty's status changes, the step writes one row to Orion's event table. The chronology
  turns these rows into "faculty" arcs, for example "walkway grading was broken from 09:00 to 14:00".
  The Day letter, recent arcs and the citation check can now name these arcs.
- Nothing acts on the state yet. There is no act gate, no self-question and no message to Juniper.
  Those are later patches.

## Outcome moved

Before this patch, Orion's track record existed only when someone ran the report by hand. Now it
exists at runtime, every hour, with a history. Orion's own chronology records when a faculty broke and
when it recovered.

## Current architecture

- Patch 1 (PR #3041, merged) has the contracts (`orion/schemas/self_monitor.py`), the adapters, the
  reducer (`orion/self_monitor/record.py`), the reader contract (`view.py`) and a read-only report.
- The Temporal Self thread in orion-durable-runs runs every 120 s:
  `ingest -> regulate -> physiology -> immune -> chronicle -> done`.
- Strain arcs already use the same pattern: the regulate step writes its own rows to
  `temporal_self_event`, and the chronicle folds them into arcs.

## Architecture touched

- **One read path.** The source SQL and the adapt step moved to `orion/self_monitor/collect.py`. The
  report and the live step both use it. There is one copy of the SQL, not two.
- **Compact hysteresis memory.** `record.advance_carry` takes a small per-faculty memory (status, since,
  candidate, how long the candidate held). The checkpoint stores only that, plus the status last
  written as an event. `advance` now calls `advance_carry`. The `reduce_faculty` constructor call is
  unchanged, so patch 3's new fields there merge cleanly.
- **One evaluation, pure.** `orion/self_monitor/host.py` turns the candidate state and the memory into
  the held state and the event rows. It does no I/O.
- **The step.** `services/orion-durable-runs/app/temporal_self_graph.py` gets a `self_monitor` node with
  its own interval and its own timeout. A failure is a warning. It never touches the regulation reading.
- **Faculty arcs.** `orion/temporal_self/sources.py` (adapter), `arcs.py` (fold, staleness, midnight
  carry) and the readers (`day_block.py`, `recent_arcs.py`, `citation_truth.py`). Faculty arcs carry no
  body summary (`body.py`), fill only leftover frame slots (`frame.py`) and stay out of dream replay
  (`services/orion-dream/app/replay.py`).

## Files changed

- `orion/self_monitor/collect.py`: new. The shared source list, the adapt step and `build_state`.
- `orion/self_monitor/report.py`: uses `collect.py` instead of its own copy.
- `orion/self_monitor/record.py`: `carry_of`, `advance_carry`; `advance` delegates to them.
- `orion/self_monitor/host.py`: new. `evaluate`, `mark_recorded`, `event_row`, `summary`.
- `orion/self_monitor/tests/test_self_monitor_host.py`: new pure tests.
- `orion/schemas/temporal_self.py`: source kind `faculty_status_changed`, arc kind `faculty`.
- `orion/temporal_self/{sources,arcs,day_block,recent_arcs,citation_truth,frame,body}.py`: faculty arcs.
- `orion/temporal_self/tests/test_faculty_arcs.py`: new. Rows are made by the producer's own builder.
- `orion/temporal_self/evals/export_fixture_day.sql`, `orion/temporal_self/README.md`: the new source.
- `services/orion-durable-runs/app/self_monitor_store.py`: new. Reads, history write and prune, event write.
- `services/orion-durable-runs/app/temporal_self_graph.py`, `temporal_self_driver.py`: the node, the
  carried keys, `/health`.
- `services/orion-durable-runs/app/main.py`: wiring, `GET /self_monitor/state`, chronicle source switch.
- `services/orion-durable-runs/app/temporal_self_sources.py`: the chronicle reads the new rows.
- `services/orion-durable-runs/{settings.py (app/), .env_example, docker-compose.yml, README.md}`: 4 keys.
- `services/orion-durable-runs/requirements.txt`: `numpy==2.2.6` (the shared grader needs it).
- `services/orion-durable-runs/tests/test_self_monitor_node.py`: new node and store tests.
- `services/orion-sql-db/manual_migration_self_monitor_history_v1.sql`: new table.
- `services/orion-dream/app/replay.py`: faculty arcs excluded from replay.
- `orion/inner_state_registry.py`, `config/metrics/metric_definitions.lock.json`: the new signal.
- `.github/workflows/{orion-durable-runs-tests,temporal-self-tests,self-monitor-tests}.yml`: paths and
  the numpy install.

## Schema / bus / API changes

- Added: `GET /self_monitor/state` on orion-durable-runs (state plus `view`: fresh or stale; 404 when
  no evaluation landed in 3 hours).
- Added: Redis key `orion:self_monitor:latest` (TTL 3 h, `DEFAULT_MAX_AGE_SEC`).
- Added: table `substrate_self_monitor_history (computed_at pk, state_json, ingested_at)`.
- Added: `temporal_self_event` rows with `source_kind = faculty_status_changed`,
  `source_table = self_monitor`. This is a new value, not a new column.
- Added: `TemporalSourceKind` value `faculty_status_changed` and `ArcKind` value `faculty`.
- Added: inner-state signal `self_monitor.state.v1` (SHADOW; its one reader is the chronology).
- Behavior changed: none for existing readers.
- Compatibility: the arc and frame models forbid unknown values. Orion-thought and orion-hub must run
  the new code before orion-durable-runs folds a faculty arc, the same as for strain and street.
  The post-pull rebuild rebuilds all three in one run, because they all import the changed schema.
  The first faculty arc appears at least 5 minutes after durable-runs starts (the chronicle's read lag).
- No bus channel. Same as regulation: a channel is added when a bus reader exists.

## Env/config changes

- Added keys (orion-durable-runs): `SELF_MONITOR_ENABLED=true`, `SELF_MONITOR_INTERVAL_SEC=3600`,
  `SELF_MONITOR_TIMEOUT_SEC=120`, `SELF_MONITOR_HISTORY_RETENTION_DAYS=30`. All flags ship ON.
- Removed / renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced: the four keys are in
  `/mnt/scripts/Orion-Sapienform/services/orion-durable-runs/.env`. The sync script reads templates from
  the primary checkout, which is on main and does not have these keys yet. So I added the four missing
  keys with their template values by hand. `python scripts/sync_local_env_from_example.py orion-durable-runs`
  then ran clean.
- Skipped keys requiring operator action: none. One old divergence was reported and left alone:
  `DURABLE_RUNS_GRAPH_HOST` (local value differs from the template on purpose).

## Metric quality gate

The published numbers are the same numbers patch 1 put through the gate. Its findings still hold:

1. **Provenance.** Every bet traces to a source row via `evidence_ref`. The SQL is in
   `orion/self_monitor/collect.py`. Each adapter names its producer.
2. **Independence.** Each faculty is reduced only from its own bets, and a bet counts once. The organ
   adds no sensor: forecast coverage is `prediction_forecast.in_80`, and act outcomes are
   `grammar_acts`' own. Act "calibration" measures whether the act posterior learns. It is not an
   independent judgement. This is written in the registry notes.
3. **Theory anchor.** Wilson intervals with Kish's effective sample size, Brier score and skill, and
   calibration-in-the-large.
4. **Live sanity.** Patch 1's live report (7 days to 2026-11-03 03:40Z) shows 1 broken, 1 unreliable,
   8 miscalibrated, 13 reliable and 23 unknown faculties out of 46. Nothing is pinned, and the
   reliable/unknown rest state exists (a test pins recovery). The live read took about 1 s.
5. **Existing mechanism.** It reuses the shared grader and copies the regulation host pattern.
6. **Reversibility.** `SELF_MONITOR_ENABLED=false` stops it. The table can be dropped. The arc kind is
   one Literal value.

**Follow-up the stance-cue PR (patch 3) depends on.** The prompt-semantics gate works on top-level float
fields of a registered schema. The fields the cue will render (`status`, `reliability`,
`reliability_hi`, `stated_mean`, `calibrated_share/lo/hi`, `n_failed_to_execute`, `n_graded`) sit
inside the `faculties` list, so this registry entry cannot name them one by one. This patch registers
the signal-level node only, with its rest and absent meaning. Patch 3 must add the per-field rest
semantics for whatever its cue puts in a prompt (for example a `FacultyRecordV1` registry entry, or
semantics keyed on the nested fields if the gate is extended), and re-lock.

## Tests run

```text
orion/self_monitor/tests                         31 passed
orion/temporal_self/tests + orion/thought/tests  398 passed (with self_monitor tests)
services/orion-durable-runs/tests                756 passed, 1 skipped (with a throwaway local
                                                 Postgres 16 as ORION_ADMISSION_TEST_DSN, so the new
                                                 store test and the existing Postgres tests ran)
scripts/check_env_template_parity.py             PASS (94 services)
scripts/check_inner_state_registry.py            OK (27 entries)
scripts/check_metric_lineage.py --gate           PASS
scripts/check_metric_lineage.py --prompt-semantics PASS
scripts/check_definition_drift.py --gate         PASS (re-locked: 2 added URNs)
tests/test_metric_lineage.py, tests/test_metric_prompt_semantics.py  82 passed
migration allowlist: orion.sql_migration_drift.auto_apply_refusals(...) == [] (new and existing table)
mutation check: disabling mark_recorded's "recorded" update fails 3 node tests
```

## Evals run

```text
orion/temporal_self/evals/run_arc_precision_eval.py      passed: true
orion/temporal_self/evals/run_play_cue_eval.py           PASS
orion/temporal_self/evals/run_day_citation_truth_eval.py truth_rate 1.000 (W44, W45)
```

The self-monitor has no eval harness of its own yet. The live check after deploy (below) is the first
real run. Acceptance check 8 in the spec (24 h of history rows, a real status change within 72 h) needs
the deploy.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-durable-runs build   Image orion-durable-runs-durable-runs Built
docker run --rm --entrypoint python <image> -c "import app.temporal_self_graph, app.self_monitor_store,
  orion.self_monitor.host, numpy"                       imports ok 2.2.6
```

No `up` was run (worktree).

## Review findings fixed

A review subagent read the diff against main. Fixed:

- Finding (material): a change row was stamped at the time its status began. A row retried an hour
  later then sat behind the chronicle's read point and was never folded, so its arc never opened.
  - Fix: every row is stamped when it is written. The status start stays in the payload and becomes
    the arc's `episode_began_at`. The event id still uses the status start, so a retry adds no copy.
  - Evidence: `test_a_retried_change_row_is_stamped_at_its_write_time_so_the_chronicle_still_reads_it`,
    and the node retry test now checks the stamp.
- Finding (material): each source read could take 45 s. Five reads (225 s) did not fit in the 120 s
  step limit, so two slow tables could blank the whole evaluation. The writes had no timeout.
  - Fix: 15 s per read (75 s total). The writes get a 15 s statement timeout and a 5 s lock timeout.
  - Evidence: the Postgres store test passes with the new settings.
- Finding (should): a faculty whose source failed this hour still counted the partial evidence toward
  a new status.
  - Fix: its held status is published unchanged and the hour does not count.
  - Evidence: `test_a_faculty_whose_source_erred_keeps_its_held_status_and_counts_nothing`.
- Finding (nit): a retried "dropped" row got a new id each time.
  - Fix: its id uses the held status start. Evidence: `test_a_retried_drop_row_keeps_its_id`.
- Finding (rollout): the new arc kind needs readers first. Fix: the restart order below lists
  orion-thought, orion-hub and orion-dream before orion-durable-runs.

Not changed: the arc's status is read back from its label (`<faculty> <status>`). A label is cut only
past 300 characters, and faculty names are far shorter. The reviewer agreed with the keepalive choice.
One more effect of it: a faculty whose source stays unreadable for more than 30 h gets its arc closed
with "then no word from the self-monitor". That is correct: there was no evidence for that time.

## Restart required

The migration is additive and has a valid `REQUIRED-BY` header, so the post-pull rebuild applies it
before it rebuilds orion-durable-runs. If it is not applied, the deploy gate refuses `up` and prints
the exact apply command. By hand, from the primary checkout on main:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_self_monitor_history_v1.sql
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-dream up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Live check after deploy:

```bash
curl -fsS http://localhost:8121/self_monitor/state | head -c 400
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select count(*), max(computed_at) from substrate_self_monitor_history"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select label, count(*) from temporal_self_event where source_kind='faculty_status_changed' group by 1"
```

## Risks / concerns

- Severity: low. Concern: the first evaluation writes one row for every faculty that is already notable
  (about 10 on today's data), so about 10 faculty arcs open at once. Mitigation: faculty arcs only fill
  leftover frame slots, and the Day letter line is capped at 6 items.
- Severity: low. Concern: the brief said "no change writes no row". I added a keepalive row every 12 h
  while a notable status holds. Without it, a stopped organ would leave "broken since" in Orion's
  chronology with no evidence behind it. This is the same rule strain arcs use. Within 12 h, no change
  writes no row.
- Severity: low. Concern: consumer-first rollout (see Compatibility). Mitigation: restart order above.
- Severity: low. Concern: a timeout after the event write but before the step returns can write a
  change row twice with different ids. The fold treats a repeated open row as an extend, so no second
  arc opens.
- UNVERIFIED: the live path (Redis key, history rows, event rows, arcs). It needs the deploy.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3066

🤖 Generated with [Claude Code](https://claude.com/claude-code)
