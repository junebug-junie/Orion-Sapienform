## Summary

- Once a night (01:00-03:00 Denver), Orion now has one long sleep that reads the whole Denver day that just closed and keeps up to 5 memories of it. This is Patch 2a of the day-dream spec (`docs/superpowers/specs/2026-11-17-nightly-day-dream-design.md`, approved in #3605).
- Code lays out part of the day: the day's sleeps (`dream_cycle` rows with their pair answers and hypotheses), the concerns Prime holds (read-only), and Orion's now (`GET /temporal-self/now?organ=dream`). The turn reads the rest itself with a new dig command, `dig dream-day <day>`. That command prints the organ roundup and every dig Orion ran that day, and leaves a read receipt.
- The turn answers in a fixed JSON shape: memories, a keep/drop/new plan, and one paragraph about the day. Code drops any item whose refs do not all resolve and counts it. It writes at most 5 memories (`episode_memory`, `channel='dream'`) through the episode validator and store. The memories and the new `dream_day` row go in one transaction.
- The plan is parsed, ref-checked and stored in `dream_day.plan`. It is **not** handed to Prime. Patch 2b does that, on top of #3610.
- The night shows up in three places: a new experience kind `dream_day` in Temporal Self, cites of the now recorded with `record_use`, and the act `dream.cycle make`. The act is graded by the run: 1 only when a memory was written, else 0 with the note "no lasting trace".

## Outcome moved

Before this patch, nothing took in the whole day for Orion's own sake. The `dream` memory channel had 0 rows. After it, each night can add up to 5 code-checked memories of yesterday to Orion's recallable memory. Each one cites records that resolve, so a memory with no record behind it cannot be written.

## Current architecture

- `orion-dream` runs pressure-driven sleeps. Each one replays 12 items and asks whether pairs of them link. No sleep ever saw a whole day.
- `dig organs <date>` (the organ roundup) existed, but only the morning letter read it. `dig digs <date>` (#3675) existed, but nothing used it. Its background half can only be read inside the governor.
- `episode_memory` accepted `channel='dream'`, but nothing wrote it.
- The daily-run pattern already lived in orion-durable-runs: the reminiscence tick (window, poll, `run_request`) and the single-turn day letter graph.

## Architecture touched

- **orion-durable-runs.** A new night tick (`app/dream_day_poll.py`). A new workflow graph (`app/dream_day_graph.py`: gather -> hold -> one turn -> persist). Reads and the one write live in `app/dream_day_store.py`, which is also the dream organ's `reads_now`. New routes: `GET /dream-day/plan` and `POST /dream-day/run`. `/health` now shows `dream_day_tick`. The act family is registered in `app/run_acts.py`.
- **Pure logic.** `orion/memory/dream_day.py` holds the parse, the ref check, the memories (through `orion.memory.episode.validate`) and the grade.
- **Memory store.** `orion/memory/episode/store.py` now has a shared `_insert_memories`, used by the distiller and the dream with identical SQL. The new `persist_dream_memories` runs inside the caller's transaction.
- **Dig kit.** `dig dream-day <date>`. `cmd_digs`'s read is factored into `_read_digs`; its output is unchanged.
- **Temporal Self.** New `TemporalSourceKind` `dream_day` (experience only). Its adapter is in `orion/temporal_self/sources.py`; its reader is in `temporal_self_sources.py`.
- **Registry.** In `config/integration/organs.yaml`, the `dream` organ gains workflow `dream.day`, source `dream_day`, and `reads_now`.

## Deviations from the spec (and why)

1. **Orion-durable-runs ticks, not orion-dream.** The spec says orion-dream's sleep loop submits through `carry_submit.py`. Instead, orion-durable-runs ticks the way the reminiscence and the field notes already do (window, poll, deterministic run id, store dedup). The run lives in that service, so a submit from there needs no cortex-orch hop and no receipt round-trip. orion-dream is unchanged.
2. **Most of the brief is read inside the turn.** The background half of the digs is readable only inside the governor (#3675). So the organ roundup and the digs come from `dig dream-day`, which the turn runs first. Code proves the read through a receipt (`DreamDayReadV1`) in the self dir that orion-durable-runs already mounts. `DreamDayBriefV1` is the run's brief (day, route, timeout). The half that code gathers is `DreamDayMaterialV1`. Both are stored on the `dream_day` row (`material`, `read`).
3. **The voice is `orion_thought`, not `orion`.** `episode_memory_voice_chk` has no `orion` value. `orion_thought` is Orion's own voice.
4. **`follow_up` is not a dream purpose.** It becomes `happened`. Carrying something into tomorrow is the plan's job, and a follow_up with no due date never comes due.
5. **There is no new body-economy family.** The run's act is the existing `dream.cycle` (verb `make`, keyed by run id). It shadow-bids as `dream.cycle`. That family lists lanes `[fast]`, while this run holds `memory_distill` (gpu1). The bid is shadow only, never obeyed.
6. **The eval has no judge yet.** It prints each statement beside its refs' text and re-resolves every ref now. The local-lane judge and the 90% gate need 7 nights of rows first.

## Files changed

- `orion/schemas/dream_day.py`: the contracts (`DreamDayBriefV1`, `DreamDayMaterialV1`, `DreamDayReadV1`, `DreamDayAnswerV1`, `DreamDayV1`) and the ref kinds.
- `orion/schemas/durable_run.py`: workflow `dream.day`, its brief union member, and the pairing check.
- `orion/schemas/registry.py`: registers the three named contracts in both registries (checked with `resolve()`).
- `orion/schemas/temporal_self.py`, `orion/temporal_self/sources.py`, `services/orion-durable-runs/app/temporal_self_sources.py`: the `dream_day` source kind, its adapter and its reader.
- `orion/memory/dream_day.py`: parse, ref check, memories, grade.
- `orion/memory/episode/store.py`: the shared memory insert and `persist_dream_memories`.
- `orion/dig/__main__.py`, `orion/dig/README.md`: `dig dream-day`.
- `services/orion-durable-runs/app/{dream_day_poll,dream_day_graph,dream_day_store}.py`: the tick, the run, and the store.
- `services/orion-durable-runs/app/{admission_runtime,main,run_acts,settings}.py`: wiring, the routes, the act family, and the flags.
- `services/orion-durable-runs/{.env_example,docker-compose.yml,README.md}`: the `DREAM_DAY_*` keys and the docs.
- `services/orion-sql-db/manual_migration_dream_day.sql`: the `dream_day` table (additive, REQUIRED-BY orion-durable-runs).
- `config/integration/organs.yaml`: dream organ: `dream.day`, `dream_day`, `reads_now`.
- `scripts/sync_local_env_from_example.py`: `DREAM_DAY_` prefix.
- Tests: `services/orion-durable-runs/tests/test_dream_day.py`, `orion/dig/tests/test_dream_day_cmd.py`, and `orion/temporal_self/tests/test_now.py` (updated).
- Eval: `services/orion-durable-runs/evals/run_dream_day_eval.py`.
- CI: `.github/workflows/{orion-durable-runs-tests,dig-kit-tests}.yml` path filters.

## Schema / bus / API changes

- Added: `DurableWorkflowV1` `dream.day`. `DurableRunRequestV1.brief` member `DreamDayBriefV1`. `TemporalSourceKind` `dream_day`. Table `dream_day`. Routes `GET /dream-day/plan` and `POST /dream-day/run`. Dig command `dream-day`.
- Removed: none.
- Renamed: none.
- Behavior changed: `persist_episode` now calls the shared `_insert_memories`. Its SQL and order are the same.
- Compatibility notes: these are additive values on `extra="forbid"`/`Literal` models, so consumers must update before producers. orion-durable-runs is the only producer of `dream.day` runs and of `dream_day` events, so it must deploy after orion-sql-writer, orion-hub, orion-actions, orion-cortex-orch and orion-thought. The post-pull rebuild rebuilds them together, and the first run cannot happen before 01:00 Denver. No new bus channel.

## Env/config changes

- Added keys (orion-durable-runs): `DREAM_DAY_ENABLED=true`, `DREAM_DAY_START_HOUR_LOCAL=1`, `DREAM_DAY_END_HOUR_LOCAL=3`, `DREAM_DAY_ROUTE=memory_distill`, `DREAM_DAY_TURN_TIMEOUT_SEC=900`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (`services/orion-durable-runs/.env_example`), and compose too.
- Local `.env` synced with `python3 scripts/sync_local_env_from_example.py --all-keys orion-durable-runs`: yes. It added the 5 keys above.
- Skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-durable-runs && pytest tests/test_dream_day.py           22 passed
ORION_ADMISSION_TEST_DSN=<disposable postgres:16> pytest tests/test_dream_day_postgres.py   1 passed
cd services/orion-durable-runs && pytest tests                             1046 passed, 150 skipped (after review fixes);
    a second run had 1 timing failure in test_physiology_market_replay_off_loop.py (passes alone and on main;
    host load, unrelated)
pytest orion/dig/tests/test_dream_day_cmd.py orion/dig/tests/test_digs.py  55 passed
pytest orion/temporal_self/tests orion/dig/tests orion/memory              all passed (after test_now.py update)
cd services/orion-sql-writer && pytest tests/test_act_downstream_use.py tests/test_durable_run_state_sql_shape.py   14 passed
pytest tests/test_check_integration_seam.py tests/test_report_integration_coverage.py tests/test_organ_names_partition.py tests/scripts/test_sync_local_env_from_example.py   49 passed
python3 scripts/check_integration_seam.py --base origin/main               OK (16 organs; 19 workflows; 30 source kinds)
python3 scripts/check_env_template_parity.py                               PASS
python3 scripts/check_service_env_compose_parity.py orion-durable-runs     OK (177 keys)
python3 scripts/check_migration_required_by.py                             OK
auto_apply_refusals(manual_migration_dream_day.sql, live tables)           [] (on the post-pull allowlist)
```

## Evals run

```text
python3 services/orion-durable-runs/evals/run_dream_day_eval.py --day 2026-11-17
  UNREAD dream_day: the table does not exist here (apply ...manual_migration_dream_day.sql)
  channel='dream' memories: 0
```

The eval runs live and read-only (`default_transaction_read_only=on`). It has nothing to show until the first night runs.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-durable-runs build                    Image built
docker run --rm --entrypoint python <image> -c "import app.dream_day_graph, app.dream_day_poll, app.dream_day_store, ..."   imports ok
Disposable postgres:16 (deleted after): applied the episode_memory migrations + manual_migration_dream_day.sql twice (idempotent);
  persist_dream_day: written (2 memories, produced) -> replay: same row, nothing twice -> a second run: already_written;
  empty night: produced=False, note "no lasting trace"; the chronicle's dream_day query + adapter -> event dream_day:2026-11-17
Live read-only (no writes):
  Prime concerns: status ok, 3 concerns parsed from orion:prime:state
  dream_cycle 2026-11-17: 2 rows read (one with 4 pair answers)
  resolve_refs: experience / act / dream_cycle / harness_turn_trace / chat_turn resolve; act:act_nope, chat_turn:does-not-exist do not
  governor: `dig digs 2026-11-17` (deployed code; dig dream-day composes the same reader): 373 calls, 121 background, no UNREAD
```

## Live proof

UNVERIFIED. Deploys go only from clean main through the post-pull rebuild, so this branch cannot run live. Post-merge checks are below.

## Review findings fixed

An independent review subagent read the diff and ran it on a throwaway Postgres, with no live writes. It found no blockers. Every finding is fixed:

- Finding (should-fix): the receipt check measured freshness from when the run was requested. So a retry could reuse attempt 1's receipt, and a hand or chat run of `dig dream-day` after the request would also pass.
  - Fix: `check_receipt` now requires a receipt newer than THIS attempt's start, minus 120 s of clock slack (`dream_day_graph.py`).
  - Evidence: `test_a_retry_needs_its_own_fresh_read`, `test_receipt_must_be_this_runs_read`.
- Finding (should-fix): the failure paths had no tests, and none of the SQL was tested against Postgres.
  - Fix: three new graph tests, plus `tests/test_dream_day_postgres.py`. That test is gated on `ORION_ADMISSION_TEST_DSN`, which CI provides. It resolves all six ref kinds plus a missing id and a quote-injection id, then persists, replays with the same run id, and tries a second run, which gets `already_written`.
  - The three graph tests: a persist retry with no second turn, cites giving up after 3 attempts while the run still completes, and a run that lost the day grading 0.
  - Evidence: `22 passed` (`test_dream_day.py`). `1 passed` (`test_dream_day_postgres.py` on a disposable postgres:16).
- Finding (nit): the row's `memories_rejected` left out the memories past the cap.
  - Fix: `row_fields` passes `memories_over_cap`, and the store adds it.
- Finding (nit): adding `dream_day` to `TemporalSourceKind` is an additive value. Readers of the now on old images would reject it.
  - Fix: a deploy-order note on the literal. The post-pull rebuild rebuilds those readers with this merge, and the first row cannot appear before 01:00 Denver.
- Finding (nit): a run whose persist ended `already_written` took `produced` from the winning run's row.
  - Fix: that run's own act now grades 0, with the note `already_written by <run>`.
  - Evidence: `test_a_run_that_lost_the_day_grades_zero`.
- Finding (nit): a 900 s GPU turn now pools with the cheap sleep acts in `dream.cycle`'s measured cost.
  - Fix: no code change. It is noted under Risks, so the first `dream.cycle` cost readout after deploy is not misread.

## Restart required

The post-pull rebuild on clean main rebuilds the affected services. It applies `manual_migration_dream_day.sql` first, because the file is additive with a REQUIRED-BY header. No manual restart is needed. To do it by hand on clean main:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_dream_day.sql
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Post-merge live check (one hand run for 2026-11-17, approved in the spec):

```bash
curl -fsS "http://localhost:8124/dream-day/plan?day=2026-11-17" | jq '.decision, .run_id'
curl -fsS -X POST "http://localhost:8124/dream-day/run?day=2026-11-17" | jq .
curl -fsS "http://localhost:8124/runs/dream-day-2026-11-17" | jq '.status, .detail'
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "SELECT day, run_id, produced, note, memories_written, items_dropped, refs_dropped, kept, dropped, new, read->'unread', read->>'digs_total', read->>'digs_background' FROM dream_day WHERE day='2026-11-17'"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "SELECT memory_id, statement FROM episode_memory WHERE channel='dream' AND episode_id='dream_day:2026-11-17'"
python3 services/orion-durable-runs/evals/run_dream_day_eval.py --day 2026-11-17
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "SELECT event_id, label FROM temporal_self_event WHERE source_kind='dream_day'"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "SELECT event_id, citing_ref FROM experience_use WHERE citing_ref='dream_day:2026-11-17'"
docker exec orion-athena-harness-governor cat /mnt/orion-fcc/self/dream_day/2026-11-17.json
```

Pass when all of these hold:

- The `dream_day` row exists with `read->'unread' = {}`.
- At least 1 `episode_memory` row has `channel='dream'`, and the eval prints `resolves_now` for every ref.
- One `temporal_self_event` row `dream_day:2026-11-17` exists. It appears after the chronicle's next window.
- `experience_use` has a row. It appears only if Orion cited one of the shown experiences.

## Risks / concerns

- Severity: medium. Concern: a confabulated memory, one that sounds right but does not match its records. Mitigation: code refuses any unresolved ref, and the validator checks the names in a statement against the refs' text. The eval prints each statement beside its refs, but there is no judge yet.
- Severity: low. Concern: the shadow bid runs as `dream.cycle`, whose lanes are `[fast]`, while the run holds gpu1. Mitigation: the bid is shadow only and never obeyed. The act is graded correctly.
- Severity: low. Concern: `dig digs` on a host whose claude home is readable but is not the governor's reads 0 background calls, not UNREAD. This bug predates this patch. Mitigation: the receipt records the transcript count in the page, and the run only ever executes in the governor.
- Severity: low. Concern: the dream turn's GPU cost (up to about 900 s) now pools with the cheap sleep-cycle acts in the `dream.cycle` family's measured costs and bids. Mitigation: read the first `dream.cycle` cost after deploy with this in mind. The spec chose no new family.
- Severity: low. Concern: in the full durable-runs suite, `test_physiology_market_replay_off_loop.py` is a timing flake under host load. It passes alone and on main.
- Issues opened: #3702 (verify-live: the first night). Closed: none.


🤖 Generated with [Claude Code](https://claude.com/claude-code)
