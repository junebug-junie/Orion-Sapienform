# PR report: durable cognition runs kicked off from cortex (the spike)

**Date:** 2026-09-06
**Branch:** `feat/durable-runs`
**Design:** `docs/superpowers/specs/2026-09-06-durable-cognition-runs-from-cortex-design.md` (PR #2127), option A, scoped to "the spike, and only the spike"
**Service name:** `orion-durable-runs` (Juniper: not "workflow", cortex already owns that word)

## Summary

- A curiosity investigation is Orion's longest single act of cognition (10-40 minutes) and, until now, lived entirely in Hub's process memory: every Hub redeploy killed it, burned a daily-cap slot, and left nothing to resume. Today, three runs started and none finished.
- New service `orion-durable-runs` owns the run's state machine as a LangGraph graph under a Postgres checkpointer. Every node's result is checkpointed; on boot and every two minutes it re-invokes any thread whose graph still has a next node; a thread older than a day is abandoned, never resumed into different material.
- Cortex is the kickoff: Hub's tick still owns scheduling, material, worldview and the prompt, but hands the run to `orion:cortex:request` (`context.metadata.durable_run`); cortex-orch dispatches it to the runner and answers `accepted`. Cortex sees every run start.
- Hub keeps the one node only Hub can execute, the harness turn, behind an RPC the runner calls; if Hub goes away mid-turn the checkpoint keeps the run and the turn is re-issued under the same `run_id` when Hub is back. Outreach stays in Hub, driven by the runner's `completed` event.
- Every node transition is a `DurableRunStateV1` row in a new table and a row on the attention surface (`process=durable_run`), so the sequencing is observable on the same table as the processes it belongs to.
- All behind `HUB_CURIOSITY_KICKOFF_VIA_CORTEX` (default `false`, which is today's direct path exactly).

## Outcome moved

(filled after the live restart test -- the design doc's Acceptance Check 1: a run survives a container restart on the real rail with the same `run_id`, one journal, one `TurnOutcome`, and a state row with `resumed_from_node` set.)

## Current architecture

Hub tick -> `execute_unified_turn` (a Hub-owned saga importing a dozen Hub-internal `scripts.*` modules) -> harness governor runs one FCC subprocess per run, cancellable, no checkpoint -> graph read -> attention row -> journal -> outreach. The daily cap and cooldown are stamped before the turn and refunded only on a clean `CancelledError`. Nothing outside Hub can execute the saga.

## Architecture touched

```
Hub tick --(cortex.orch.request, metadata.durable_run)--> cortex-orch --(orion:durable:run:request)--> orion-durable-runs
   ^                                                                                                     |
   |<------------------- orion:curiosity:turn:request (RPC, reply on orion:curiosity:turn:reply:<corr>) --|  harness_turn
   |                                                                                                     v
   |<------------------- orion:durable:run:state (completed + reach_out -> outreach) <-------------------|  every transition
                                        sql-writer -> substrate_durable_run_state; attention surface -> process=durable_run
```

- **Why the turn stays in Hub.** `orion/hub/turn_orchestrator.py` imports `scripts.settings`, `scripts.harness_governor_client`, `scripts.thought_client`, `scripts.pre_turn_appraisal_client` and more -- Hub-internal, and CLAUDE.md section 5 forbids reaching into them from another service. Re-implementing the saga in the runner would be the second implementation the design doc names as a non-goal. So the runner owns the state machine and Hub executes one node on request. What survives a Hub death is the run's identity, cap slot, continuation, and everything after the turn; the FCC minutes of an interrupted turn are re-spent (design doc MQ2, stated plainly).
- **Journal builders moved, verbatim.** `build_investigation_journal_entry`, `format_footprint`, `format_evidence` and their constants now live in `orion/curiosity/journal.py` (extracted with `ast`, byte-for-byte); Hub imports them from there. The runner writes the identical journal entry from a `MaterialCounts` object carrying the five facts the entry reads.
- **Consumer-first, by construction.** No existing schema field changed. `AttentionSchemaV1.process` gains the value `durable_run`, which the live sql-writer validates against, so sql-writer is deployed *before* the runner (the 2026-09-06 `FieldGoalProvenanceV1` incident, PR #2126, is why this order is written down).

## Files changed

- `services/orion-durable-runs/` (new): `app/graph.py` (the StateGraph, testable with fakes), `app/runner.py` (checkpointer, deps, state events, resume sweep), `app/main.py`, `app/settings.py`, `Dockerfile`, `docker-compose.yml`, `requirements.txt` (langgraph 1.2.11, langgraph-checkpoint-postgres 3.1.2), `.env_example`, `README.md`, `tests/test_curiosity_graph_resume.py`.
- `orion/schemas/durable_run.py` (new): four schemas, three channel names, the node list. `orion/schemas/registry.py`: both maps. `orion/bus/channels.yaml`: three channels (request and turn are `single_consumer`).
- `orion/schemas/attention_schema.py`: `durable_run` lane.
- `orion/curiosity/journal.py` (new, moved verbatim from Hub).
- `services/orion-cortex-orch/app/durable_runs.py` (new) + `app/main.py` branch + `tests/test_durable_run_dispatch.py`.
- `services/orion-hub/scripts/curiosity_investigation.py`: `_dispatch_durable_run`, `_turn_request_loop`/`_handle_turn_request`, `_run_state_loop`/`_handle_run_state`, `start`/`stop` wiring; `app/settings.py`, `scripts/main.py`, `.env_example`; six tests appended.
- `services/orion-sql-writer/app/models/durable_run_state.py` (new) + worker/settings/retention/env/README + `tests/test_durable_run_state_sql_shape.py`.
- `scripts/check_env_template_parity.py`: a branch-new service compares its worktree-local `.env` instead of refusing to deploy at all.
- `orion/sentience_striving_program/instruments.yaml`: `durable_runs` instrument (`runs_resumed`, `runs_abandoned`).
- `.github/workflows/orion-durable-runs-tests.yml`: CI for the runner's tests.
- `config/metrics/metric_definitions.lock.json`: re-locked (three new bus channels).

## Schema / bus / API changes

- Added: `DurableRunRequestV1` (`durable.run.request.v1`), `DurableRunStateV1` (`durable.run.state.v1`), `CuriosityTurnRequestV1` / `CuriosityTurnResultV1`; channels `orion:durable:run:request` (cortex-orch -> runner, single consumer), `orion:durable:run:state` (runner -> sql-writer, Hub), `orion:curiosity:turn:request` (runner -> Hub, single consumer, reply on `orion:curiosity:turn:reply:<corr>`); table `substrate_durable_run_state`; `AttentionSchemaV1.process` value `durable_run`.
- Removed / Renamed: none.
- Behavior changed: with the Hub flag on, a curiosity run's tail (graph read, attention row, journal) is executed by the runner instead of Hub; the journal entry is byte-identical. With the flag off, nothing.
- Compatibility: the new `process` value requires sql-writer to be rebuilt first (deploy order in the README).

## Env/config changes

- Added keys: `HUB_CURIOSITY_KICKOFF_VIA_CORTEX=false` (orion-hub); `SUBSTRATE_DURABLE_RUN_STATE_RETENTION_DAYS=90` (+ channel and route entries, orion-sql-writer); the whole `services/orion-durable-runs/.env_example`.
- `.env_example` updated: yes (three services). Local `.env`: Hub and sql-writer synced with `--all-keys`; the runner's `.env` was bootstrapped by hand from its example (the sync script cannot create one) with the real `POSTGRES_URI` and `DURABLE_RUNS_GRAPH_HOST=orion-athena-falkordb` (the runner is on the bridge network, Hub is host-networked and uses 127.0.0.1:6380), and pre-positioned in the primary checkout for after the merge.
- Skipped keys: none.

## Tests run

```text
services/orion-durable-runs/tests        5 passed  (full run; crash-after-turn resumes without re-issuing the turn;
                                                    failed turn resumable at harness_turn with attempt 2; sweep finds
                                                    exactly the unfinished threads)
services/orion-hub/tests/test_curiosity_investigation.py   127 passed (6 new: kickoff via cortex dispatches and does
                                                    not run the turn; falls back when cortex is down; flag off is the
                                                    direct path; turn RPC runs _generate and replies; empty turn replies
                                                    not-ok; completed+reach_out triggers outreach, others ignored)
services/orion-cortex-orch/tests/test_durable_run_dispatch.py   4 passed; rest of the dir 25 passed + 1 pre-existing
                                                    failure identical on main (test_concept_profile_config_adapter)
services/orion-sql-writer shape suites   13 passed
root: registry agreement, surface, inner-state gate, single-consumer gate, drift, worldview   233 passed
Static gates: env single-source OK, inner-state OK, journal dispatch OK, health producers OK, compose parity
  orion-durable-runs OK (18/18), definition drift PASS (re-locked), sentience instruments --static-only OK,
  git diff --check OK, no .env staged.
```

## Evals run

```text
No eval harness for this seam; the eval is the live restart test below (Acceptance Check 1) and the
instrument's `runs_resumed` claim.
```

## Docker/build/smoke checks

All from this worktree via `scripts/safe_docker_build.sh`, consumer-first.

```text
2026-09-06 20:44Z  sql-writer REFUSED by the env-drift gate: live SUBSCRIBE_CHANNELS / ROUTE_MAP_JSON lists
                   were missing the new members (the gate does what it says; fixed the live .env by hand,
                   deterministically: json-append the example's missing members).
                   runner: "network orion-athena-net not found" -> the external network is plainly `app-net`;
                   host port 8121 is orion-execution-dispatch-runtime -> 8124.
       20:46Z  sql-writer + runner up. LangGraph created checkpoints/checkpoint_blobs/checkpoint_writes/
                   checkpoint_migrations; sql-writer created substrate_durable_run_state and subscribed to
                   orion:durable:run:state; runner listening on orion:durable:run:request; /health ok.
       20:48Z  cortex-orch + Hub (flag on). Hub's bus FAILED to init: my wiring referenced
                   settings.CORTEX_REQUEST_CHANNEL (the env alias; the attribute is CORTEX_ORCH_REQUEST_CHANNEL)
                   inside Hub's bus-init block. Hub degraded ~2.5 min; hot-fixed and redeployed 20:51:17Z
                   ("curiosity_durable_listeners started", "curiosity_investigation started").
       20:52Z  POST /curiosity/api/run-now -> run ff8a379217d8:
                   Hub: curiosity_durable_dispatched status=no_reply -> fell back in-process   (bug 1: undecoded reply)
                   cortex-orch: durable_run_dispatched run=ff8a... channel=orion:durable:run:request   (cortex saw it)
                   runner: durable_run_request run=ff8a...; turn RPC published; Hub: curiosity_turn_request run=ff8a...
                   -> two turns for one run; the runner's RPC timed out at 3600s and the runner then froze
                   (bug 2: single-connection saver). Hub's tick started c67b1a10fb93 at 21:09Z the same way and
                   finished it in-process (journaled) -- the old path still works with the flag on.
2026-09-07 01:14Z  fixes deployed (pool, decoded replies, shared dedup). Runner froze at boot with zero Postgres
                   activity (bug 3: aget_state nested inside alist -> saver lock). Probed connect/pool/setup
                   inside the container with timeouts: all instant, so the hang was the caller's nesting.
       01:21Z  runner redeployed with the listing materialised -> resume on boot, below.
```

## Live restart test

Acceptance Check 1: a run survives a container restart on the real rail, same `run_id`.

```text
run ff8a379217d8   kicked off through cortex 2026-09-06 20:52:13Z; checkpoint at harness_turn (ts 20:52:13Z)
runner redeployed  2026-09-07 01:21:07Z (a full container recreate, 4.5h after the checkpoint)
runner boot        durable_run_resume run=ff8a379217d8 from=harness_turn age_h=4.5
                   durable_run_state run=ff8a379217d8 node=harness_turn status=resumed next=read_turn_result resumed_from=harness_turn
                   durable_runs_resume_on_boot {'resumed': 1, 'abandoned': 0, 'active': 0}
                   [rpc] publish success corr_id=f7a34fe9-... request_channel=orion:curiosity:turn:request
Hub                curiosity_turn_request run=ff8a379217d8 attempt=1
substrate_durable_run_state   ff8a379217d8 | harness_turn | resumed | next read_turn_result | resumed_from harness_turn | 01:21:26Z
substrate_attention_schema    durable-ff8a379217d8-harness_turn-resumed | harness_turn:resumed | predicted_next read_turn_result
/runs/unfinished   [{thread_id: ff8a379217d8, next_node: harness_turn, checkpoint_ts: 2026-09-06T20:52:13Z}]
```

Completion of the resumed run (journal, TurnOutcome, remaining node rows): see the addendum at the end of this
report, written when the turn finished.

## Review findings fixed

(filled after review)

## Restart required

Consumer-first order, all from this worktree via `scripts/safe_docker_build.sh <svc> up -d --build`:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
# then set HUB_CURIOSITY_KICKOFF_VIA_CORTEX=true in services/orion-hub/.env and
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium
  Concern: a resumed run re-issues the *turn*, not the FCC minutes; an interrupted turn is paid for twice. That is the honest limit of "resumable" here (design doc MQ2) and it is still strictly better than losing the run and the cap slot.
  Mitigation: stated in the README and the runner's log (`attempt=2`); `runs_resumed` on the instrument counts it.
- Severity: medium
  Concern: the Hub redeploy that kills a turn also kills the turn-RPC listener; the runner's RPC times out after `DURABLE_RUNS_TURN_RPC_TIMEOUT_SEC` (3600s) unless the bus reports the disconnect sooner, so a resume can lag up to an hour after Hub is back.
  Mitigation: the sweep re-issues on the next pass after the timeout; lowering the timeout below Hub's own 3500s budget would cancel healthy turns. A liveness ping from Hub during a turn is the right follow-up, not done here.
- Severity: low
  Concern: outreach after a resumed run depends on Hub receiving the `completed` event; if Hub is down at that moment the outreach is skipped (the journal and graph writes are not).
  Mitigation: acceptable for a message; the state row records `reach_out=true` for anyone who wants to backfill.
- Severity: low
  Concern: LangGraph is a new dependency, confined to one small service.
  Mitigation: the fallback (design doc option B) reuses every contract here except the checkpointer.

## PR link

(filled on open)
