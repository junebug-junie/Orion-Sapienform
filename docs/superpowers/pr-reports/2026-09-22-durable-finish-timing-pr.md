# Durable runs: carry harness timing and turn correlation on finish/failed detail (2026-09-22)

Patch 1b of the curiosity-tab redesign
(`docs/superpowers/specs/2026-09-22-curiosity-tab-redesign-design.md`, section
"Timing fields the runner already knows"). Single service: `orion-durable-runs`.

## Summary

- When a curiosity run finishes, its `completed` event now says which correlation
  the harness turn actually ran under and how long the runner itself waited on
  it: `turn_correlation_id`, `harness_elapsed_sec`, `harness_started_at`,
  `harness_finished_at` are added to `finish_detail` (`app/graph.py`).
- The source is a new `harness_turn_meta` state key that `harness_turn` writes
  right after the Hub RPC returns (`timed_turn`), so the numbers survive a
  checkpoint/resume the same way `text` does.
- The `failed` event's detail now names the `node` and, for curiosity, the
  `turn_correlation_id` (`DurableRunner._failed_detail`). The admitted graph
  stashes the fenced per-lease correlation on failure *before* it clears the
  lease, so the terminal projection (`AdmissionRuntime._terminal`) can report
  the turn that actually failed rather than the run's lineage.
- `self_sense_eval`'s finish detail gains a per-question `turns` map with the
  same four fields (one self-sense run is several harness turns).
- Every new key is additive and optional on `DurableRunStateV1.detail` (a free
  `dict[str, Any]`). A checkpoint written before the key existed finishes with
  the fields absent -- not null, not an error.
- README gains a "Finish detail fields" section listing every detail key per
  workflow.

## Outcome moved

Today the `harness_turn_trace` Postgres row for a durable run (PK
`correlation_id`; holds `fcc_elapsed_sec`, `step_count`, `fcc_served_model`,
`grounding_status`) can only be found by text-searching `final_text`, because
nothing stored the turn's correlation structurally. After this patch the run's
`completed` event carries that key, so the Hub read side (sibling patch) can
join run -> harness row directly. The runner's own wall-clock around the turn is
also now visible per run, independent of Hub's in-process `debug.elapsed_sec`.

## Current architecture

- `services/orion-durable-runs/app/graph.py`: `harness_turn` calls
  `deps.run_turn` and returns `{text, debug, attempt}`; `finish_detail` builds
  the `completed` payload (line, self_definition, lived_answer,
  self_question_family, reach_out, reach_out_why, continue_line, finding_text,
  journal_entry_id, attempts). `turn_correlation_id(state)` derives the
  per-lease id when a lease is held, else returns the run's.
- `app/admitted_graph.py`: wraps the same node under admission; on failure
  returns a state with `lease: None`.
- `app/runner.py:_drive`: on a raised node, emits `failed` with
  `{"error": ...}` only.
- `app/admission_runtime.py:_terminal`: failed projection detail is
  `{"error": last_error}` only.
- `orion/schemas/durable_run.py:200`: `DurableRunStateV1.detail: dict[str, Any]`
  -- free dict.
- `orion/bus/channels.yaml` `orion:durable:run:state` description does not
  enumerate detail keys.

## Architecture touched

- Service: `orion-durable-runs` only. No Hub changes.
- Contracts: none. `detail` is a free dict; no schema, registry, or channel
  edit. `channels.yaml` checked -- it does not list detail keys, so nothing to
  keep accurate there.
- Runtime seam: the `completed` / `failed` payloads on `orion:durable:run:state`
  gain optional keys; `substrate_durable_run_state` (sql-writer) stores
  `detail` as JSON, so the new keys land there without a migration.

## Files changed

- `services/orion-durable-runs/app/graph.py`: `timed_turn` (runner-side
  monotonic timing + ISO-UTC stamps around the Hub RPC),
  `harness_turn_meta` state key, `recorded_turn_correlation_id`,
  `failed_turn_correlation_id`, `harness_meta_detail`; `finish_detail` spreads
  the optional keys in.
- `services/orion-durable-runs/app/self_sense_graph.py`: `ask_questions` uses
  `timed_turn`; per-answer timing keys; `finish_detail` gains `turns`.
- `services/orion-durable-runs/app/admitted_graph.py`: the admitted
  `harness_turn` wrapper captures `turn_correlation_id(state)` before the
  lease can be cleared and stashes it as `harness_turn_meta` on every failure
  return (deadline, max-attempts, retrying).
- `services/orion-durable-runs/app/admission_runtime.py`: `_terminal` failed
  detail carries the *recorded* correlation only -- never re-derived, since
  the lease is gone by then. The worker-recovery fence stashes the in-flight
  attempt's fenced id before clearing its lease (review finding).
- `services/orion-durable-runs/app/runner.py`: `WorkflowSpec` gains optional
  `failed_turn_correlation_id` (curiosity sets it; self-sense and reflect
  leave it None); `_drive` failure path uses `_failed_detail` ->
  `{error, node, turn_correlation_id?}`, degrading to the bare shape if the
  helper raises.
- `services/orion-durable-runs/tests/test_finish_timing.py`: 17 tests (below).
- `services/orion-durable-runs/README.md`: "Finish detail fields" section.

## Schema / bus / API changes

- Added: none (no schema/registry/channel files touched).
- Removed: none.
- Renamed: none.
- Behavior changed: `orion:durable:run:state` `completed` detail (curiosity)
  may now include `turn_correlation_id`, `harness_elapsed_sec`,
  `harness_started_at`, `harness_finished_at`; `self_sense_eval` completed
  detail includes `turns`; `failed` detail includes `node` and optionally
  `turn_correlation_id`.
- Compatibility notes: `DurableRunStateV1.detail` is `dict[str, Any]`, so
  older consumers see extra keys they ignore. **No deploy-order rule applies
  here**: the design doc's "deploy durable-runs before hub" note was written
  for a `forbid` model field (PR #2158's `line`), but these keys live inside
  the free `detail` dict, so an older Hub reading a newer event, or a newer
  Hub reading an older event, both work. The sibling Hub patch must treat the
  keys as optional (they are absent on pre-key checkpoints regardless of
  deploy order).

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not
  needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
cd services/orion-durable-runs && PYTHONPATH=<worktree> python -m pytest tests -q -p no:cacheprovider
82 passed, 41 skipped   (skips = Postgres-DSN tests; CI runs them with its own Postgres)

python -m pytest tests/test_finish_timing.py -q
17 passed

New tests (tests/test_finish_timing.py):
- completed detail carries all four fields; elapsed is runner-measured (not
  Hub's debug.elapsed_sec=900.0); stamps are tz-aware UTC and ordered; meta is
  checkpointed on the thread
- completed detail via the runner's state event carries the fields
- leased turn records the derived per-lease id, unleased records the run's;
  clearing the lease afterwards does not flip finish_detail back to lineage
- failed at harness_turn: detail names node + run correlation
- failed at journal (after a good turn): detail names node=journal + recorded
  correlation
- failed detail for a leased thread is the derived identity
- failed detail never raises (helper None / helper raises / bare state) and
  drops an empty correlation instead of emitting ""
- admitted failure keeps the fenced correlation after lease is cleared, and
  AdmissionRuntime._terminal reports it
- admitted success carries full timing to finish detail
- pre-key checkpoint finishes with the fields absent; leased pre-key
  checkpoint still yields turn_correlation_id from debug.turn_correlation_id
- finish_detail tolerates bare/malformed meta (None, str, int, list, all-None)
- self_sense finish detail names each question's turn + timing, joinable to
  the real request correlation
- self_sense tolerates answers recorded before timing existed
- retry under lease generation 2 replaces generation 1's stashed id (success
  and second-failure variants)
- failed_turn_meta is empty for a malformed lease (wrapper keeps its own
  failure path)
- worker-recovery fence records the fenced generation before clearing the lease

python scripts/check_env_template_parity.py      -> PASS (88 services)
python scripts/check_bus_reply_channels.py       -> 12 prefixes resolved, 0 uncovered
python -m pytest tests/test_agent_trace_schema_registry.py -q  -> 2 passed
scripts/check_single_consumer_channels.py        -> needs live ORION_BUS_URL; not run
                                                     (no channel added/changed)
git diff --check                                  -> clean
```

## Evals run

```text
None run. services/orion-durable-runs/evals/ holds admission_fairness.py,
gateway_capacity.py, elastic_fairness.py -- all Postgres admission evals, none
covers finish-detail content. CI runs them on its isolated Postgres. No eval
gap created by this patch: the new behavior is deterministic payload shaping,
fully covered by the gate tests above.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-durable-runs build
 -> Image orion-durable-runs-durable-runs Built (exit 0)
    (first attempt failed only because the fresh worktree had no gitignored
     .env; symlinked the main checkout's root and service .env in -- git
     check-ignore confirms both ignored, nothing staged)

docker run --rm --entrypoint python orion-durable-runs-durable-runs -c "<import check>"
 -> app.graph.timed_turn / failed_turn_correlation_id present,
    app.self_sense_graph.HARNESS_META_DETAIL_KEYS present,
    DurableRunner._failed_detail present,
    WorkflowSpec signature includes failed_turn_correlation_id=None
    (the patched code is in the image, not just on disk)

Not brought up (task said build only). Live emission of the new keys on
orion:durable:run:state is UNVERIFIED until the service is restarted and a
run completes.
```

## Review findings fixed

Code review ran in a subagent against the branch diff vs `main`. No must-level
findings; every should and nit was fixed in the follow-up commit.

- Finding (should): `admitted_graph.py` computed `turn_correlation_id(state)`
  for the failure stash *outside* the `try`, so a malformed lease dict would
  escape the wrapper instead of taking its release/attempt-count path.
  - Fix: new guarded `failed_turn_meta(state)` in `graph.py` returns `{}` on
    `KeyError`/`TypeError` (and on an empty id); the wrapper spreads that.
  - Evidence: `test_failed_turn_meta_is_empty_for_a_malformed_lease_so_the_wrapper_still_takes_its_failure_path`.
- Finding (should): no test executed the `retrying` return that carries the
  stash, nor the overwrite by a new lease generation -- the exact "could a
  stale generation's id be reported?" question had no direct test.
  - Fix: two admitted-graph tests: attempt 1 fails under generation 1
    (`harness_turn_meta` = gen-1 id), retry under generation 2 succeeds
    (recorded id = gen-2, full timing) / fails again (recorded id = gen-2).
  - Evidence: `test_retry_under_a_new_lease_generation_replaces_the_stale_generations_id`,
    `test_retry_that_fails_again_under_generation_two_reports_generation_two`.
- Finding (should, pre-existing gap): the worker-recovery fence in
  `admission_runtime.py` cleared an in-flight attempt's lease without
  recording its fenced id, so a later deadline/cancel terminal would report
  an *older* attempt's id (or nothing) while Hub's `harness_turn_trace` row
  sits under the fenced one -- the one real stale-id path.
  - Fix: that arm's `aupdate_state` payload now includes
    `failed_turn_meta(state)` before `lease: None`.
  - Evidence: `test_worker_recovery_fence_records_the_fenced_generation_before_clearing_the_lease`
    (pins the payload shape and that the runtime's arm uses the helper; the
    full recovery loop is exercised by `test_running_recovery_postgres.py`
    in CI's Postgres lane, unchanged).
- Finding (nit): `failed_turn_correlation_id` documented `None` but returned
  `""` for an empty lineage, and the test locked that in.
  - Fix: `turn_correlation_id(state) or None`; test asserts `is None`.
  - Evidence: `test_failed_detail_never_raises_and_is_bare_for_workflows_that_cannot_name_a_turn`.
- Finding (nit): README said `_terminal` served only `failed` (it also serves
  `cancelled`) and overstated "the correlation the turn actually ran under"
  for an attempt that failed before the RPC left.
  - Fix: README wording -- "issued, or would have been issued, under"; join
    returns no row, never a wrong row; fence behavior documented.
  - Evidence: README diff.
- Finding (nit): `harness_elapsed_sec < 5.0` upper bound could flake on a
  starved runner.
  - Fix: bound raised to 30.0 (floor of 0.01 kept).
- Finding (nit): `_drive` never drives admitted threads, so its leased derive
  branch cannot occur live; the test proving it needed a comment.
  - Fix: comment added in the test.

Confirmed-correct by the review (no change needed): no path re-derives the
correlation after the lease is cleared; `harness_turn_meta` is a plain
`LastValue` channel with one writer per superstep in both graphs and absent
(not broken) on pre-key checkpoints; no `null` is emitted where the contract
says absent; self-sense's failure path is byte-for-byte unchanged;
`DurableRunStateV1.detail` is a free dict so no contract file needed a change.

## Restart required

```bash
# From a worktree (never the shared checkout), after merge + pull:
scripts/safe_docker_build.sh orion-durable-runs up -d --build
docker compose -f services/orion-durable-runs/docker-compose.yml logs --tail=50
```

Deploy order: independent of `orion-hub`. See "Compatibility notes" -- the keys
are additive on a free dict, so durable-runs and the Hub read side can restart
in either order.

## Risks / concerns

- Severity: low
  - Concern: `harness_elapsed_sec` is measured by the runner around the Hub
    RPC, so it includes bus transit and reply decode -- it is not the same
    number as Hub's `debug.elapsed_sec` or `harness_turn_trace.run_artifact.
    fcc_elapsed_sec`. Readers should not treat them as interchangeable.
  - Mitigation: README documents the distinction; the field name is the
    runner's, and the join key (`turn_correlation_id`) lets a reader fetch the
    FCC-side numbers from the trace row when it wants those.
- Severity: low
  - Concern: on the admitted path a failed turn stores only
    `turn_correlation_id` (no elapsed), because the wrapped node raised before
    returning. A retried attempt overwrites `harness_turn_meta` with the new
    attempt's full record, so the final detail always describes the attempt
    that completed.
  - Mitigation: documented; tests cover both the failure stash and the
    success overwrite.
- Severity: low
  - Concern: live emission not verified (build only, per task).
  - Mitigation: restart + one completed run; check
    `substrate_durable_run_state.detail` for the new keys.

## PR link

PR_LINK_PENDING
