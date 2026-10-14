# Curiosity: weekly/monthly topic and overall chronology

## Problem and behavior

Curiosity's individual run journals lack a weekly/monthly representation of what Orion
explored and how their understanding changed. After a completed persisted **sleep cycle**,
dream now checks due calendar windows and submits `curiosity.compaction`. Durable-runs
owns topic passes, hierarchical reduction, overall synthesis, GPU admission and recovery.
The narrative story/carry flag does not control this activity.

The sources are persisted run write-ups, not a claim that every admitted run wrote one.
Topic areas are generated from content. References must cover every input with no invented
IDs. Source-derived UTC date ranges survive every stage; dated citations remain readable
in the journal. Monthly summaries reuse complete weeks fully inside the month and fill
uncovered days from original journals. No source is removed or rewritten.

## Current architecture

- Services: `orion-dream` (calendar admission/readout); `orion-durable-runs` (execution);
  cortex-orch/exec (existing verb rail); sql-writer (existing journal consumer).
- Entry points: successful persisted `run_cycle_once`; admitted graph in
  `app/curiosity_compaction_graph.py`; `/dreams/curiosity-compactions` readout.
- Config: dream `.env_example`, local ignored `.env`, `app/settings.py`, compose.
  New flag `DREAM_CURIOSITY_COMPACTION_ENABLED=false`; local env synced.
- Bus: existing cortex request, durable request/state and journal write channels.
  Journal catalog now declares dream/durable-runs producers. No new event kind/channel.
- Schemas: new typed `CuriosityCompactionBriefV1`; additive DurableWorkflowV1 value.
  Existing registered DurableRunRequest/Receipt/State and JournalEntryWrite models carry it.
- Persistence: shared explicit `orion.curiosity.compaction_store` API; additive
  `curiosity_compaction` table; immutable input snapshot plus progress/artifact state.
- Docker: existing dream, durable-runs and cortex-exec compose files; SQLAlchemy added
  to durable-runs using its existing psycopg driver. No top-level service `settings.py`
  exists in dream/durable-runs; their settings live under `app/`.
- Tests/evals: dream pure/scheduling/Postgres tests; real LangGraph restart tests;
  actual cortex prompt rendering/token budget; offline chronology replay.

## Scheduling, recovery and boundaries

Completed Denver calendar weeks and months, with DST-aware UTC bounds. First activation
uses the previous closed period; subsequent sleeps process the oldest unfinished/gap
window. One window per cadence per sleep. Duplicate submissions use the same durable ID.
A terminal failure advances the generation while retaining saved work; old writes are
fenced. Cancelled/abandoned windows stay cancelled. Empty input is explicit, no LLM.

The source snapshot is immutable: later backdated journal inserts require an explicit
future rebuild. There is no automatic historical backfill. Sources remain internal
Curiosity journal data; no external distribution or notification is added. Stored
summaries are derived interpretations, not new facts or claims of sentience. Six broad
topic areas per pass and bounded prose keep synthesis tractable; source journals retain
the details. Live summary fidelity remains UNVERIFIED until a real run is evaluated.

No new cognition metric or sensor channel is introduced. Counts in the readout/lifecycle
are ordinary operational evidence of actual records and saved topics.

## Verification

- Dream tests + offline evals: 257 passed, 2 optional integration skips.
- Durable-runs suite: 412 passed, 86 environment-dependent skips (before the additional
  large-window test; see PR check results for final count).
- Final focused dream scheduling/core/Postgres/chronology checks: 79 passed, zero skips.
  These checks and the cortex prompt tests now run in the existing durable CI job.
- Cortex-exec prompt/token-budget smoke: 3 passed.
- New isolated Postgres tests exercise migration idempotence, advisory locks, checkpoints,
  missing delivery acknowledgements, retries, generation fencing, cancellation and
  month-boundary weekly reuse. Production SQL is read-only during this task.
- Read-only live corpus (2026-10-14): last closed week has 98 journals / 524,687 source
  characters / 52 initial topic calls; September has 238 / 1,673,071 / 182. Zero empty
  source bodies. This led to a regression test beyond LangGraph's default 25-step limit.
- Existing compactors: actual actions logs show chat and GitHub completed on 2026-10-12,
  under independent daily schedules, not dreams.
- Docker builds passed for dream, durable-runs and cortex-exec. Network-isolated image
  import/route/driver smokes passed. CI results are recorded in the PR/handoff.
- New production rail: UNVERIFIED; feature remains disabled pending approved deployment.

## Review findings fixed

- Finding: dates could disappear between topic/reduction/overall stages and citations
  could sort by ID.
  - Fix: source-derived UTC spans on every topic/material record, chronological sorting
    and explicit date-preservation instructions.
  - Evidence: date propagation/reordered-ID regression and offline chronology replay.
- Finding: cancelling a thread-backed DB acquisition could leak an advisory lock; cancelling
  preparation could close the connection while its worker was still using it.
  - Fix: shield acquisition, await the worker, then close; drain DB workers on cancellation.
  - Evidence: controlled thread-event regression for both paths; independent reviewer
    reproduced `worker-finished` before `closed` after the fixes.
- Independent requesting-code-review skill/subagent: no remaining material findings.

## CI dependencies

The implementation commit's durable graph/contract CI passed (498 passed, one optional
skip, plus the Postgres-backed fairness, deploy-order and chronology evals). Full PR status is blocked
by the existing personal-term gate: all 11 flagged paths are unchanged by this branch.
The independent fix is PR #2653 (its static gate is green); merge that fix before rebasing
this PR. Do not bypass or weaken the privacy gate. The substrate v4 matrix job also failed
before checkout because its Postgres image mirror returned `toomanyrequests: Data limit
exceeded`; a retry reproduced the infrastructure failure. The v6 matrix passed.
Current check results are authoritative in PR #2654.

## Deployment (requires Juniper's production approval)

The additive workflow literal requires consumer-first deployment. From this worktree:

```bash
docker exec -i orion-athena-sql-db sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < services/orion-sql-db/manual_migration_curiosity_compaction.sql
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
# Set DREAM_CURIOSITY_COMPACTION_ENABLED=true in the ignored live dream .env.
scripts/safe_docker_build.sh orion-dream up -d --build
```

Keep `ORION_BUS_URL=redis://100.92.216.81:6379/0` (verified local config). Smoke the next
natural sleep: `curiosity_compaction_admitted` must name the window/run; durable state
must progress through compact/finalize, the readout must contain topics+overall with dated
sources, and the stable journal entry must exist. Inspect actual prose for chronological
fidelity, contradictions and open questions before declaring live quality verified.

Rollback: disable the dream flag; pause/cancel admitted compaction runs through existing
durable controls; restore images only after new runs stop. Preserve tables, checkpoints,
original journals and graphify artifacts.
