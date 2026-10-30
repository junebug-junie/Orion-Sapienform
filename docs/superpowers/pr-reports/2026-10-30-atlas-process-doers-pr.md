## Summary

The body atlas groups Orion's parts into organs. Until now every ongoing process was tied only to the service that schedules it, so all 20 processes piled into one big organ. Now each process is tied to the services that actually did its work, read from the run's own record.

- **Who did the work.** A durable run's record holds its correlation id and the id of the turn it started (`detail.turn_correlation_id`). Rows in trace tables that carry either id, and name the service that wrote them, tie that service to the process (`does` flow).
- **Runs that start runs.** `detail.child_run_ids` ties a run to the runs it started (`spawns` flow). Live: dream.carry starts reverie.visual.
- **Reducers.** Each reducer now ties to the one service whose events it reads, taken from orion-substrate-runtime's `REDUCER_SPECS` by reading the code (`feeds` flow).
- **No hand-typed "plumbing" list.** The scheduler and the GPU lease show up in almost every process. Each process still adds one unit of weight in total, split by `log(1 + runs) × log(1 + N / df)`: how often the service showed up, times how rare it is across processes. A service in every process keeps a small share. A process with only its scheduler keeps the whole unit on it.
- **Loud gaps.** A trace table the read-only role cannot read shows as `UNREAD` with the exact `GRANT` it needs. Nothing is granted from here.

## Outcome moved

Live (2026-10-30, inside orion-athena-harness-governor):

| | before | after |
|---|---|---|
| largest organ | 75 parts, 34% | 68 parts, 31% |
| modularity Q | 0.350 | 0.349 |
| organs | 18 (8 singletons) | 18 (8 singletons) |

The gate is 35%; there is now 4 points of room instead of 1. Seven of the eight reducers left the big organ for the organ of the service they read (the vision reducer now sits with the eyes, the execution and route reducers with cortex). All 12 durable workflows stay in the big organ: with the readable sources, their doers are the harness and the scheduler, which live there. dream.carry is now tied to reverie.visual, the run it starts.

The big organ is still big, and most of that is real. Its core is orion-hub, orion-gpu-pool, orion-harness-governor, orion-durable-runs and orion-substrate-runtime. Those five signal each other on the bus, and that is what holds 25 services and 27 tables together. Only 20 of its 75 parts were processes. Most durable runs really are harness turns that the scheduler starts (reading.turn: harness 67%, scheduler 33%; curiosity.investigate the same). So the organ is partly an executive spine, not only an artifact. I did not force it apart, and I did not change any gate.

Three trace tables cannot be read yet (see Risks). An offline projection that adds the brief's 7-day join counts as if those tables were readable gives 67 parts (31%), Q 0.349, 19 organs. That projection is UNVERIFIED: the real numbers need the grant.

## Current architecture

`orion/body_atlas/sources.py` read each durable workflow's run count and gave it one runner (the producer of the run-state channel, orion-durable-runs). Reducers got orion-substrate-runtime, labeled as an assumption. `build.py` linked each process to that one runner with the full unit of weight.

## Architecture touched

- `sources.py`: `ProcessIn` gains `doers` (service, evidence, runs) and `children` (child workflow, runs). New live reads, one per source so each fails alone: `grammar_events`, `cognition_traces`, `cockpit_turn_sighting` (joined by run id and turn id), `harness_turn_trace` (turn records; the service is that table's declared writer from sql-writer's route map), and `child_run_ids`. New declared read: `REDUCER_SPECS` by AST, following `from x import NAME` re-exports up to 3 modules deep. `psql_rows` now names the missing `GRANT` when the role is denied.
- `build.py`: process parts are added first, then their flows, split by the rarity-weighted rule. Each process gets a `work_split` fact (top three shares).
- `model.py`: relations `does`, `spawns`, `feeds`; evidence `observed_trace`, `observed_turn`, `declared_turn_writer`, `observed_child_run`, `declared_reducer_source`. File-artifact fields only, not a bus payload.

## Files changed

- `orion/body_atlas/sources.py`: doer, child-run and reducer-source reads; GRANT hint.
- `orion/body_atlas/build.py`: rarity-weighted process flows, `work_split` fact.
- `orion/body_atlas/model.py`: new relation and evidence labels.
- `orion/body_atlas/evals/run_organ_structure_eval.py`, `evals/fixtures/whole_body_2026-10-30.json`: third frozen fixture (live export with doers).
- `orion/body_atlas/tests/test_process_doers.py`: new tests. `tests/test_sources.py`: fake DB answers the new queries.
- `orion/body_atlas/README.md`: new flows, the weighting rule, live results, grants and the run-products gap.
- `.github/workflows/body-atlas-tests.yml`: also runs when `services/orion-substrate-runtime/app/worker.py` changes (the atlas reads its `REDUCER_SPECS`).

## Schema / bus / API changes

- Added: atlas file values only (`FlowRelation` `does`/`spawns`/`feeds`, five `FlowEvidence` values, `ProcessIn.doers`/`children`, part fact `work_split`).
- Removed: none. Renamed: none.
- Behavior changed: a process's flow weight is split over its doers instead of all going to its runner. Organ ids change where membership changed (expected). Older fixtures still validate.
- Compatibility notes: no bus or DB schema change. `test_names_still_carry_from_the_runner_only_body_onto_the_doers_body` proves names given on the old shape carry onto the new one.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. python -m pytest -q -p no:cacheprovider orion/body_atlas/tests orion/dig/tests orion/evals/model_replay/tests
215 passed
Mutation check: removing the rarity term from build._scores fails
  test_a_service_in_every_process_gets_a_small_share_and_each_process_adds_one_unit and
  test_reducers_lean_toward_the_service_they_read.
```

## Evals run

```text
python orion/body_atlas/evals/run_organ_structure_eval.py
organ structure (bus_graph_2026-10-23.json): PASS   Q=0.3335, largest 26%
organ structure (whole_body_2026-10-29.json): PASS  Q=0.3503, largest 32%
organ structure (whole_body_2026-10-30.json): PASS  Q=0.349,  largest 31%
run_organ_structure_eval.py --live (inside orion-athena-harness-governor, branch code): PASS, Q=0.349, largest 31%
```

## Docker/build/smoke checks

Live smoke inside `orion-athena-harness-governor` with this branch's code, no image rebuild. I tarred the worktree's `orion/`, `config/`, `mesh-utilities/common/`, `services/orion-sql-writer/app/`, `services/orion-substrate-runtime/app/worker.py`, and every service's `.env_example` and `docker-compose*.yml` into `/tmp/atlas-doers` in the container, then ran twice:

```text
env PYTHONPATH=/tmp/atlas-doers ORION_DIG_NOW_ROOT=/tmp/atlas-doers ORION_DIG_HOME=/tmp/atlas-dig-home python -P -m orion.dig body
```

Organ ids were identical on both runs. 5.3 s. Nothing written to any DB, graph or bus; dig output went to `/tmp/atlas-dig-home` in the container.

```text
parts 261: 99 svc, 4 endpoint, 4 camera, 92 store, 20 process, 7 hosts, 35 transient · organs 18 (8 singletons)
flows 832 signals, 98 writes, 78 runs_on, 36 uses, 20 runs, 11 sees_through, 8 feeds, 3 does, 1 spawns · Q=0.349 · largest organ 31%
UNREAD 3 source(s), this body is missing what they hold:
  live:postgres grammar_events (run + turn correlation ids): RuntimeError: needs GRANT SELECT ON grammar_events TO orion_readonly; ...
  live:postgres cognition_traces (run + turn correlation ids): RuntimeError: needs GRANT SELECT ON cognition_traces TO orion_readonly; ...
  live:postgres cockpit_turn_sighting (run + turn correlation ids): RuntimeError: needs GRANT SELECT ON cockpit_turn_sighting TO orion_readonly; ...
 1. organ:e063739906 [68: 25 svc, 3 endpoint, 27 store, 13 process; 60 live] on athena 15 circe 1: orion-hub, orion-gpu-pool, orion-harness-governor, orion-durable-runs, orion-substrate-runtime +63
 2. organ:2ad84a72c8 [67: 39 svc, 24 store, 4 process; 59 live] on athena 28 circe 1: orion-cortex-exec, orion-sql-writer, orion-cortex-orch, ...
 3. organ:995cb31c70 [25: 13 svc, 10 store, 2 process; 20 live] on athena 9: orion-llm-gateway, orion-vector-host, orion-bus, ...
 4. organ:096895fd81 [20: 13 svc, 4 camera, 2 store, 1 process; 17 live] on athena 11 carbon-x1 1 circe 1: orion-vision-council, ...
```

What each evidence source gave, live:

- `REDUCER_SPECS` (declared, code): all 8 reducers got their source service.
- `harness_turn_trace` (readable): reading.turn, curiosity.investigate, situation.update → orion-harness-governor (3 `does` flows).
- `child_run_ids` (readable): dream.carry → reverie.visual (1 `spawns` flow).
- `grammar_events`, `cognition_traces`, `cockpit_turn_sighting`: UNREAD (permission denied).

## Review findings fixed

A review subagent read the branch diff (`origin/main...HEAD`). It found no HIGH issues, nothing that writes, and no secrets in the fixture. It checked the SQL live as `orion_readonly` (`detail` is `json`, so the `json_*` calls are right; child-run join returns `dream.carry,reverie.visual,32`).

- Finding (MEDIUM): the trace-table join had no time bound, and `grammar_events` (1.4 M rows) has no `correlation_id` index. After the grant it could time out.
  - Fix: every trace and turn-record join is bounded to 31 days on the row's own time column. The README says these two evidence kinds have never run live and need an `EXPLAIN` after the grant.
  - Evidence: `test_the_join_query_follows_both_the_run_and_its_turn` asserts the bound.
- Finding (MEDIUM): no test read the real `REDUCER_SPECS`, so a refactor of `worker.py` would pass CI and only show up live as a partial read.
  - Fix: `test_the_real_reducer_specs_resolve_in_this_checkout`; CI also triggers on `orion/substrate/*_loop/constants.py` and `orion/schemas/*_projection.py`.
  - Evidence: the test passes on this checkout (8 reducers).
- Finding (LOW): the constant reader matched nested assignments, ignored `X: T = "..."`, and turned `None` into `"None"`.
  - Fix: module-level `Assign`/`AnnAssign` with `str` values only.
  - Evidence: `test_constants_are_module_level_strings_only`.
- Finding (LOW): a child run outside the 30-day window read as "no such process".
  - Fix: the child row is bounded by the same window.
- Finding (LOW): a self-loop could leave an empty flow entry.
  - Fix: guarded (`src == dst` skipped).
- Finding (LOW): `work_split` on runner-only processes just repeated the runner.
  - Fix: only set with two or more candidates. Evidence: `test_a_runner_only_process_has_no_work_split_fact`.
- Finding (LOW): a test comment contradicted its assertion. Fix: reworded.
- Not changed (LOW): run vs turn counts use the larger, not the union (listed in Risks). `link()` keeps the last relation on a key; only this loop creates process keys today.

After the fixes: 218 tests pass, all three fixtures and `--live` PASS, live organ ids unchanged and stable over two runs.

## Restart required

No restart is needed for config. The atlas code is baked into the harness image, so it goes live after the normal post-pull rebuild of orion-harness-governor. By hand, from the clean primary `main` checkout:

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: medium. Concern: `orion_readonly` cannot read `grammar_events`, `cognition_traces` or `cockpit_turn_sighting`, so the main doer evidence (cortex-exec for reverie.visual, hub and cortex-exec for situation.update, GPU lease use) is missing. Mitigation: each is UNREAD with the exact line. It needs Juniper's approval (a GRANT does not fit the post-pull auto-apply allowlist): `GRANT SELECT ON grammar_events, cognition_traces, cockpit_turn_sighting TO orion_readonly;`
- Severity: low. Concern: once granted, the `grammar_events` join scans about 1.4 M rows with no `correlation_id` index. Mitigation: it runs inside a 60 s read-only statement window; if it times out it shows as UNREAD, not as a wrong body.
- Severity: low. Concern: run products (`entry_id`, `dream_id`, `episode_id` in detail) give no `writes` flow, because no declared source says which table holds each id. Listed in `known_gaps`.
- Severity: low. Concern: for `observed_trace` and `observed_turn` on the same service, the run count is the larger of the two, not a distinct union. It can undercount a little; it never double-counts.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2990

🤖 Generated with [Claude Code](https://claude.com/claude-code)
