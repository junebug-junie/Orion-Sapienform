# PR: never strand a topic-foundry run in a non-terminal status

Branch: `fix/topic-foundry-enrichment-lifecycle`
Follows PR #1943 (`fix/topic-foundry-enrichment-contract`).

## Summary

- Live outage found 2026-08-29 while checking on PR #1943's deploy: six runs sat in
  `status='running'` (five `stage='enriching'`, one `'training'`), the oldest ~22
  hours, and **zero** runs were `complete` for the Orion model.
- `fetch_latest_completed_run` filters on `status='complete'`, so the concept-atlas
  ingest returned `{"available": false, "reason": "topic_foundry_no_completed_run"}`
  and the graph had no source run at all. The atlas kept rendering the last
  successful ingest, frozen.
- **Cause:** `_run_enrichment` had no `try/finally`. It wrote `status="running"` up
  front and restored the previous status only on the success path, so any raise --
  or any container restart -- left the run `running` permanently. Two of the six were
  stranded by ordinary redeploys of this service.
- A second, **latent** defect found while fixing it: `_run_enrichment` restored the
  status it had read *at entry*, so a pass starting while another was in flight read
  `"running"` and wrote it back as the terminal state.
- New `app/services/run_recovery.py` closes stranded runs at startup, on a real
  invariant rather than a staleness threshold.

## Outcome moved

| | before | after |
|---|---|---|
| runs stuck non-terminal | 6 (oldest ~22h) | 0 |
| `complete` runs for the Orion model | **0** | 5 |
| concept-atlas ingest | `topic_foundry_no_completed_run` | `available: true`, 34 participation edges, 170 edges |

## The invariant

Runs execute in-process as FastAPI `BackgroundTasks`. Verified, not assumed:
`app/storage/repository.py` is the only writer of `topic_foundry_runs` anywhere in
the repo; the Dockerfile CMD is a bare `uvicorn app.main:app` with no `--workers`;
compose declares no `deploy.replicas`; the sibling `chat-corpus-builder` container
shares the image but runs a different entrypoint against a different table.

    at process start, no run can legitimately be `running` or `queued`

There is no worker anywhere that could still be advancing it, so such a row is
restart residue and will sit there forever. That is why the reaper can act on the
first row it sees instead of guessing at an age threshold.

The precondition is precisely **one service instance per database**, which is
stronger than "one replica" -- `container_name` is node-scoped and the DSN is plain,
so a second node pointed at the same database would reap the first node's live runs.
Stated explicitly in the module docstring.

## Files changed

- `app/services/run_recovery.py` (new): `recovery_decision` (pure),
  `terminal_status_for_enrichment`, `enrich_refusal_reason`, and the
  `recover_stranded_runs` reaper.
- `app/main.py`: reaper in `lifespan`, best-effort, before anything can read a run.
- `app/services/enrichment.py`: `try/finally` starting immediately after the
  `status="running"` write, a precomputed terminal status, a reentrancy guard with an
  `owns_run` exemption for the inline caller, and a guarded `finally` body.
- `app/services/training.py`: `artifact_paths`/`stats` assigned before the enrichment
  step.
- `app/routers/runs.py`: `POST /runs/{id}/enrich` refuses a run that is not complete.
- `app/storage/repository.py`: `list_non_terminal_runs`, predicate derived from
  `NON_TERMINAL_STATUSES`.
- `tests/test_run_recovery.py` (new): 31 tests.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Behavior changed: `POST /runs/{run_id}/enrich` now returns **409** for a run that is
  not `complete` (previously it enqueued for any existing run). The Hub scheduler is
  unaffected -- it resolves its target through `fetch_latest_completed_run`, which
  only ever yields a completed run. Training's inline enrichment does not go through
  the endpoint.
- Compatibility: `RunRecord.status` is already `Literal["queued","running","complete","failed"]`;
  every status this code writes validates.

## Env/config changes

None. No env key added, removed or renamed; no `.env_example` touched; no
`orion/bus/channels.yaml` or `orion/schemas/registry.py` change needed
(`TopicFoundryEnrichCompleteV1` is unchanged).

## Tests run

```text
$ pytest tests/test_run_recovery.py -q
31 passed

$ pytest tests -q --ignore=tests/test_drift_reducer_loading.py \
    --ignore=tests/test_heartbeat_chassis.py --ignore=tests/test_training_umap_reduction.py
1 failed, 98 passed

$ pytest tests -q -k "concept or atlas or topic_foundry"   # orion-hub
160 passed, 1664 deselected
```

Pre-existing, not introduced here: the three ignored files fail to **collect**
(`No module named 'sklearn'` / `'joblib'` in the local venv; they run in the
container only), and `test_chat_corpus_builder_stages.py::test_stage_pipeline_outputs_expected_records`
fails on a `COMMAND_RE` regex assertion in a package that imports nothing this branch
touches.

### Mutation testing

Nine mutations, each asserted present in the file before the run so a no-op
replacement cannot read as a pass.

| mutation | result |
|---|---|
| drop the `finally` (restore only on success) | 1 test fails |
| drop the reentrancy guard | 1 fails |
| remove the `owns_run` exemption | 1 fails |
| `terminal_status_for_enrichment` echoes the row back | 2 fail |
| reaper marks a segment-bearing run `failed` | 4 fail |
| reaper lets the scan exception escape | 1 fails |
| reaper wipes `stats`/`artifact_paths`/`completed_at` | 1 fails *(was uncaught before review)* |
| drop `queued` from `NON_TERMINAL_STATUSES` | 2 fail |
| repository hardcodes the statuses again | 1 fails |
| `finally` claims `stage="enriched"` with nothing enriched | 1 fails |
| unguarded `finally` masks the real exception | 1 fails |
| enrich predicate always allows | 4 fail |
| route drops the 409 gate | 1 fails |

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-topic-foundry build   -> Built
$ scripts/safe_docker_build.sh orion-topic-foundry up -d topic-foundry

# first boot, against the six live stranded runs:
run_recovery_closed_stranded_run f9443362 status=complete stage=enriched segments=394 enriched=296
run_recovery_closed_stranded_run 68f0a3d1 status=complete stage=enriched segments=394 enriched=30
run_recovery_closed_stranded_run 9160b074 status=complete stage=enriched segments=2894 enriched=51
run_recovery_closed_stranded_run d3adedab status=complete stage=enriched segments=375 enriched=157
run_recovery_closed_stranded_run 76f2f163 status=failed   stage=failed   segments=0
run_recovery_closed_stranded_run 0fc63e9e status=complete stage=enriched segments=375 enriched=2
run_recovery_complete stranded=6 recovered=6

# second boot:
run_recovery_scan_clean stranded=0

$ curl -X POST .../api/substrate/concepts/ingest-topic-foundry
   {"available": true, "segments_fetched": 375, "participation_edges": 34,
    "edges_written": 170}
```

Backfill protocol (CLAUDE.md §14): the six affected rows were snapshotted to
`/tmp/topic-foundry-run-recovery/before.csv` before the reaper ran, and the result to
`after.csv`. Six rows, status/stage only; no row deleted, no content rewritten.

**The final redeploy (review-fix commit) is NOT yet verified live** -- see Risks.

## Review findings fixed

Two review rounds. Round 1 found the reentrancy-guard regression as a must-fix; I had
already self-caught and fixed it (`ff8f23b80`) before the review landed.

- **Finding (must-fix, self-caught): the reentrancy guard blocked training's own
  enrichment.** `_run_training` sets `status="running", stage="enriching"` on its own
  run and then calls `run_enrichment_sync` -- byte-for-byte the state the guard
  rejects. As first deployed, the enrichment step of every training run with
  `enable_enrichment=true` was a silent no-op.
  - Fix: the inline caller passes `owns_run=True`; `enqueue_enrichment` does not.
  - Evidence: two tests, one failing without the exemption and one failing if
    `enqueue_enrichment` ever grants itself the same exemption.
  - Live reachability: **zero runs**. Every live run has `enable_enrichment: false`,
    so this was latent, not live-active. Recording that rather than the scarier
    framing, per the runtime-truth rule.

- **Finding: I claimed the Hub scheduler could re-enter the same run. It cannot.**
  The scheduler resolves its target through `GET /runs?status=complete&limit=1`, so
  the moment a first pass writes `status="running"` the run leaves that result set
  and can never be handed to a second pass. Corrected in three places (module
  docstring, guard comment, test comment). The guard stays -- it is reachable from
  training's inline path and from a concurrent manual/smoke `POST /enrich`.

- **Finding: I claimed two defects caused the incident. One did.** All six stranded
  rows carry a pre-existing `completed_at` and real `topics_summary` artifacts, i.e.
  they entered enrichment already `complete`, so defect 2 would have restored
  `complete` for every one of them, harmlessly. Defect 1 alone explains all six.
  Corrected in the docstring and commit message.

- **Finding: a run reaped during training's inline enrichment would be `complete`
  with no artifacts.** `training.py` assigned `artifact_paths`/`stats` only *after*
  `run_enrichment_sync` returned, so the row written right before it carried
  `artifact_paths={}`. The reaper closes a segment-bearing run as `complete`; that
  run is newest by `created_at`, wins `fetch_latest_completed_run`, and then
  `_load_topic_labels` finds no `topics_summary` and serves **every topic label as
  None** -- a complete run pointing at nothing.
  - Fix: assign both before the enrichment step. One line each.

- **Finding: `POST /enrich` had no status guard, and enrichment now writes a terminal
  status.** Enriching a `queued` zero-segment run promoted it to `complete`, and the
  Hub picks the latest completed run by `created_at DESC`, so that empty run would
  have won. Fix: `enrich_refusal_reason` + a 409. Extracted as a pure predicate so it
  is testable without importing the sklearn/joblib training stack, with a
  source-reading test pinning the wiring.

- **Finding: a raise inside the `finally` re-creates the original bug and hides it
  better.** `_build_run_record_for_update` subscripts `specs[...]` bare and
  `update_run` does real I/O; an unguarded failure there would replace the original
  exception *and* re-strand the run, in code that now looks protected. Fix: the
  `finally` body is itself wrapped, logging `enrichment_terminal_write_failed`.

- **Finding: the `finally` wrote `stage="enriched"` even on the raise path**, where
  nothing was read at all. Fix: `"enriched"` only when `enriched_count` is non-zero.

- **Finding: the reaper had no test that it preserves what `update_run` overwrites.**
  `update_run` is a full-row `SET`, not a partial patch. Mutating the reaper to write
  `stats={}`, `artifact_paths={}`, `completed_at=now` left the suite **green** while
  destroying the training artifacts and real completion times of every recovered run
  -- the exact rows this PR cites as proof it worked. The fixture used empty values,
  so nothing could fail. Fix: the fixture carries real values and three tests assert
  preservation.

- **Finding: `test_every_non_terminal_status_is_recovered` was self-referential** --
  parametrized over the same frozenset the implementation branches on, so dropping
  `queued` shrank the test with it and still reported green. Fix: hardcoded literals,
  plus a test asserting the constant equals the exact set and that
  `list_non_terminal_runs` does not hardcode the statuses alongside it. The
  repository SQL now derives its predicate from the constant.

- **Finding (nit): the invariant is "one instance per database", not "one replica".**
  Stated explicitly in the docstring, with what would have to replace it (a lease or
  heartbeat) if that ever stops holding.

- **Finding (nit): vacuous and dead assertions, unused fixtures, a test-only
  constant.** Removed; `TERMINAL_STATUSES` and `NON_TERMINAL_STATUSES` now both have
  production consumers.

## Restart required

```bash
scripts/safe_docker_build.sh orion-topic-foundry build
scripts/safe_docker_build.sh orion-topic-foundry up -d topic-foundry
```

## Risks / concerns

- Severity: **medium**. Concern: the review-fix commit built and started, but the
  container has not reached healthy. It is blocked on something unrelated to this
  branch -- `ensure_tables()`'s `ALTER TABLE topic_foundry_runs ADD COLUMN IF NOT
  EXISTS spec_hash` is waiting on a relation lock held by a running `pg_dump`
  (`pg_blocking_pids` confirms it). The earlier boots of this same code succeeded, so
  the reaper and the lifecycle fix are verified; the 409 gate is verified only by
  test. Mitigation: the service starts on its own when the backup completes; a
  background watcher is polling `/health`.
- Severity: low, **pre-existing, not introduced here**. Concern: topic-foundry runs
  DDL on every boot with no `lock_timeout`, so any redeploy overlapping a backup
  hangs startup indefinitely -- and the queued `ACCESS EXCLUSIVE` request then blocks
  every other reader of `topic_foundry_runs` behind it. Proposed follow-up: a
  `lock_timeout` plus a startup log line naming the blocker.
- Severity: low. Concern: the reaper closes a segment-bearing run as `complete` on
  the strength of segments existing. `insert_segments` is a single `execute_values` in
  one transaction, so there is no partial-segment case, and the artifact fix above
  closes the no-artifacts case. Mitigation: `run_recovery_closed_stranded_run` logs
  the segment and enriched counts for every row it touches.

## Status

DONE_WITH_CONCERNS -- final deploy not yet health-verified (external lock).
