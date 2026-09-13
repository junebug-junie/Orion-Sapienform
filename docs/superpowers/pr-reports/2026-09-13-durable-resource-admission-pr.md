# Durable Curiosity resource admission

A busy preferred LLM lane now leaves an admitted Curiosity study in a persisted
LangGraph resource wait. Cortex receives a short Postgres-backed receipt; the
queue does not hold its original RPC, a run worker or an inference timeout open.
Feature defaults remain off. No production deployment/configuration/database
changes were made.

## Repository evidence and ownership

The repository already had a LangGraph Graph API service with a pooled
`AsyncPostgresSaver`, stable Curiosity run IDs, Cortex dispatch, Hub turn RPC and
Gateway routing/overrides. This patch extends those seams. The broker owns only
FIFO capacity, immutable demands, fenced leases and their event outbox;
LangGraph owns workflow position, waits, bounded retries and recovery.

The actual primary Curiosity model path runs through Hub, governor and FCC's
Gateway HTTP adapter, with Exec participating in stance/finalize. Preserving
this existing path is the intentional difference from the requested direct
Exec primary-inference arrow. Graph API is retained because existing nodes and
explicit wait/retry branches are already the durable contract. Actions and
ordinary synchronous requests remain on their existing paths.

## Changes and exact flow

Hub investigation/self-inquiry -> existing Cortex request -> short durable
receipt RPC -> committed Postgres inbox -> idempotent resource demand -> graph
interrupt -> atomic broker grant/event -> bus wakeup or SQL reconciliation ->
fenced Hub turn -> governor/FCC -> Gateway -> existing Exec finalization and
graph result/attention/journal tail -> atomic terminal projection, completion
outbox and lease release.

File groups:

- `orion/durable_admission/`: Postgres store, capacity broker and pure lane policy.
- `orion/schemas/resource_admission.py`, existing durable/Harness schemas,
  registry and bus channels: receipt, demand, fence and lifecycle contracts.
- `services/orion-durable-runs/`: admitted graph/runtime, HTTP controls/history,
  receipt handler, recovery isolation, tests, fairness eval and configuration.
- Cortex Orch, Hub, governor, shared Harness and Exec: admission receipt and
  scoped lease/timeout/lane transport through existing execution.
- Gateway: optional authoritative lease checks for bus and HTTP inference,
  periodic validation and stale output rejection without changing unleased calls.
- `services/orion-sql-db/manual_migration_durable_resource_admission_v1.sql`:
  additive inbox/demand/lease/event tables, uniqueness indexes and fence sequence.
  Apply manually to the existing checkpoint database before activation.
- Existing durable CI workflow: real Postgres service, deterministic tests and
  separate fairness eval. Metric definition lock updated for registered channels.

## Compatibility and activation

Exclusive background leases for the whole run are fully supported; other modes
are rejected instead of implying unsupported guarantees. Widening defaults off,
starts at 1200 seconds when enabled and needs 120 seconds of estimated advantage
plus switching cost. Actual Gateway routes and explicit compatibility determine
candidates. No lane equivalence is inferred. Overrides/pins disable widening;
hard capabilities still apply. The first assignment survives later attempts.

Install the additive migration and updated consumers with flags off. Test runner
admission in shadow using dedicated requests. Then enable Gateway validation,
runner enforcement, Orch admission and finally Hub admission with the existing
Cortex kickoff flag. Enable widening only after auditing lane declarations.
Pause/cancel/drain admitted runs before disabling the validator/runtime.

Exact settings, defaults, endpoint contract, metric provenance, rollout and
rollback are in the [ADR](../../architecture/durable-resource-admission.md) and
[runner README](../../../services/orion-durable-runs/README.md).

## Validation

- Durable service: **41 passed**, including isolated real Postgres checkpoints,
  wait/restart recovery, running recovery, fencing/expiry, pause/cancel races,
  retries, duplicate receipts/grants, tail recovery, queue timing and attempt IDs.
- Gateway focused routing, override, upstream admission, Anthropic and lease
  regressions: **98 passed** with `LLM_LANE_ROUTING_ENABLED=false`, the legacy
  routing fixture setting expected by those tests.
- Cortex Orch durable dispatch: **11 passed**.
- Hub Curiosity/self-inquiry: **169 passed**; unified turn: **42 passed**.
- Harness governor: **30 passed**; Exec forwarding/override: **14 passed**;
  shared motor/MCP: **34 passed**.
- Separate real Postgres fairness eval: **20/20 unique runs served**, maximum
  two distinct fixture backends, five alternative assignments, final queued and
  active lists empty. Fake clock advances past 20 minutes; no real waiting.
- A built runner container against isolated Postgres accepted HTTP 202 in
  **0.043 s**, persisted `resource_wait`, reported no active workers and cancelled
  cleanly (`http-smoke-6506f451`). No live model or production bus was invoked.
- Docker builds passed for runner, Hub, Orch, Exec, governor and Gateway using
  the repository safety wrapper and isolated image project.
- Focused env/compose parity, service-hostname references, reply-channel coverage,
  metric lineage and definition drift were checked. Worktree env files remain
  ignored; the primary checkout remains clean.

Commands for the durable tests/eval:

```bash
ORION_ADMISSION_TEST_DSN=postgresql://athena@127.0.0.1:55439/orion_admission_test PYTHONPATH=. /tmp/orion-admission-venv/bin/python -m pytest services/orion-durable-runs/tests -q --tb=short
ORION_ADMISSION_TEST_DSN=postgresql://athena@127.0.0.1:55439/orion_admission_test PYTHONPATH=. /tmp/orion-admission-venv/bin/python services/orion-durable-runs/evals/admission_fairness.py
```

The broader existing `test_harness_runner_surfaces_fcc_error_code` fails its
timeout-text assertion on both this branch and the original baseline. It was
not changed to hide an unrelated failure. GitHub CI status is tracked on the PR.

## Review findings fixed

- Finding: durable terminal publication could be lost or race cancellation.
  - Fix: atomic terminal projection/release and replayable completion outbox;
    cancellation is irreversible and wins the transaction.
  - Evidence: independent real Postgres completion/control regressions.
- Finding: post-inference failures and recovery could replay completed inference
  or leave an expired lease usable at a later node.
  - Fix: guarded, bounded tail retries and phase-aware checkpoint recovery.
  - Evidence: independent heartbeat, tail, deadline and recovery tests.
- Finding: renewable lease TTL understated long inference estimates and retry
  decisions could forget the original lane assignment.
  - Fix: declared inference budgets for reservation estimates; persisted first
    lease locks all later attempts to that lane.
  - Evidence: fake-clock widening and multiple-waiting-tick regression.
- Finding: real Curiosity supplied no alternative list.
  - Fix: derive only explicitly declared candidates at acceptance, freeze them
    for duplicate receipts and revalidate against current routes at grant time.
  - Evidence: Postgres policy-derived-candidate regression.
- Finding: late cancellation with a reused correlation could stop a successor
  attempt, and queue timing could include previous inference.
  - Fix: lease-generation-scoped attempt correlation with parent lineage; first
    grant freezes initial wait duration.
  - Evidence: actual FCC registry cancellation test and Postgres timing test.

Independent subagent review used the available requesting-code-review skill;
material findings were fixed with regression evidence.

## Limits and next patch

The full production Curiosity-to-model rail is **UNVERIFIED**, intentionally:
this task forbids deployment and production changes. The exact live Curiosity
acceptance scenario must be run after authorized activation.

Exclusive capacity is strict among managed leases on the same canonical URL.
Legacy unleased traffic can race sampled `/slots` occupancy; blocking HTTP work
already running can retain hardware after cancellation, while stale results are
rejected. Interrupted FCC tool effects have existing at-least-once semantics.
Shared/opportunistic modes, global legacy capacity reservations, DNS alias
canonicalization and learned runtime estimates are not implemented.

The next smallest follow-up is one authoritative Gateway reservation seam shared
by its bus and HTTP traffic, followed by the live Curiosity smoke. No new
self-concept cognition or competing scheduling service was introduced.
