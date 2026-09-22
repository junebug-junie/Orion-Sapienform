# PR report: durable run lifecycle rows quiet since 2026-09-14 (finding, no code change)

## Summary

- Answers the "Live lifecycle rows have gone quiet" question from the curiosity tab redesign doc.
- Verdict: the transitions are still emitted, on a different channel and shape; the only bridge back to the old table forwards `completed` alone. That was written into the resource-admission PR (2026-09-13) and turned on by default 2026-09-14.
- Nothing is being dropped by sql-writer; the runner's per-node emitter is simply not on the admitted code path any more.
- Finding doc with evidence, consumer blast radius and a recommended consumer-first patch: `docs/superpowers/specs/2026-09-22-durable-state-nonterminal-rows-finding.md`.
- Side finding recorded: one 09-21 run is in a live hot failure loop (`run demand is immutable`, one event row every ~4.6 s, 3,232 rows so far).
- No code, schema, env or bus change in this PR.

## Outcome moved

The redesign doc's root cause is no longer UNVERIFIED. Whoever builds the run-story page now knows the true start time and failure timeline live in `durable_resource_events` / `durable_admission_runs`, not in `substrate_durable_run_state`.

## Current architecture

Curiosity runs are submitted by Hub with an `admission` block, handled by `AdmissionRuntime` in orion-durable-runs. That runtime records every transition in `durable_resource_events` (Postgres, own connection) and publishes them as `durable.resource.event.v1` on `orion:durable:resource:event`. Its outbox drain mirrors only `run.completed` into `DurableRunStateV1` on `orion:durable:run:state`, which sql-writer, Hub and the runtime-activity page consume. The pre-admission `DurableRunner._emit_state` path (one `DurableRunStateV1` plus one `AttentionSchemaV1` per node transition) is never reached for admitted runs.

## Architecture touched

None. Docs only.

## Files changed

- `docs/superpowers/specs/2026-09-22-durable-state-nonterminal-rows-finding.md`: the finding, evidence, and recommended patch.
- `docs/superpowers/pr-reports/2026-09-22-durable-state-nonterminal-rows-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
git diff --check  -> clean
No code changed; no test lane applies.
```

## Evals run

```text
None applicable (docs-only).
```

## Docker/build/smoke checks

```text
Read-only live checks used as evidence (no builds, no deploys):
docker exec orion-athena-sql-db psql ... substrate_durable_run_state by day/status
docker exec orion-athena-sql-db psql ... durable_resource_events by day/event
docker logs --since 48h orion-athena-durable-runs | grep -c durable_run_state  -> 0
docker logs --since 72h orion-athena-sql-writer | grep -i "durable|unknown kind|validation|reject"  -> retention lines only
md5sum of admission_runtime.py / store.py / admitted_graph.py inside the container == checkout
checkpoint_blobs channel=admission decoded for the stuck runs
```

## Review findings fixed

- Finding: (filled after code-review subagent returns)
  - Fix:
  - Evidence:

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: should
- Concern: run `54537b5b5ccc` is writing a `run.checkpoint_resume_failed` event row every ~4.6 s and will keep doing so until its checkpoint and stored `alternatives` agree or it is cancelled. Cause of the stored-row rewrite is UNVERIFIED.
- Mitigation: `POST /runs/54537b5b5ccc/cancel` on orion-durable-runs (port 8124) stops the loop; the `register_demand` equality carve-out recommended in the finding doc prevents recurrence. Neither done here (admission loop is under active sibling work).

## PR link

(filled after `gh pr create`)
