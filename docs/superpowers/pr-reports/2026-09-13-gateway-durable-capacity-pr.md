# Durable runs: shared Gateway capacity

Ordinary Gateway requests could start between the durable broker's upstream idle
sample and its Postgres lease grant. This patch makes request admission and
workflow reservation use the same Postgres transaction lock. The owning workflow
can make its stance, reflection, retry and conditional response-repair calls on the same lease;
foreign requests wait until that reservation ends.

## Implementation

- Add typed request permits and an additive `durable_gateway_permits` migration
  to the existing durable service. No new service or Gateway database dependency.
- Gate bus chat, Anthropic Messages and OpenAI chat, including streaming, behind
  the shared authority. Ordinary requests retain the configured upstream limit;
  a durable reservation permits one owning request at a time.
- Retain permits until actual thread/stream completion, including disconnected
  callers. Renew request occupancy independently of the revoked workflow lease,
  while fencing further dispatch and accepted stale output.
- Make acquisitions idempotent, share the caller budget across admission and
  inference, and prevent context-overflow escalation to an unreserved backend.
- Carry the typed owner through Hub/Thought stance and governor finalization,
  including conditional response repair, re-reflection and internal broker-assigned routes in Exec.
- Add an isolated Postgres contention eval and run the full Gateway suite in CI.

The architecture, API, evidence provenance and rollout limits are documented in
[the ADR](../../architecture/durable-gateway-capacity.md). No new cognitive
metric or detector is introduced. Permit inspection reads persisted ownership
facts, and the contention eval verifies that those facts return to zero at rest.

## Validation

- Gateway complete suite: 345 passed, including all bus and HTTP transport
  regressions and the original routing/admission tests.
- Durable runner suite: 51 passed against isolated Postgres, including concurrent
  acquisition, owner serialization, stale identity rejection, expiry, API
  contracts, capacity-only startup behavior and the auxiliary-call chain.
- Admission fairness eval: 20/20 served, queue and leases drained.
- Gateway contention eval: 40 races, two competing clients, exactly one resource
  owner per race; ordinary and durable callers both win, final occupancy zero.
- After integrating PR #2208: 332 Harness tests, 15 governor RPC tests,
  23 Exec route/lease tests, and 51 Hub tests passed. Thought lease-route tests
  passed; existing Thought/Mind enrichment evals: 3 passed. Harness layer
  attribution and unified-turn grounding evals: 4 passed.
- All static workflow checks passed, including env sync, metric lineage and
  definition drift, schema registration, graph preservation, hostname/compose
  checks, nonblocking routes and Hub JavaScript tests.
- Six affected service builds passed through `safe_docker_build.sh`: durable
  runner, Gateway, Thought, Hub, governor and Exec. Images used the isolated
  `orion-capacity-review` project and safe example env files.

An isolated Docker HTTP smoke passed eight concurrent mixed Anthropic/OpenAI
streaming and nonstreaming requests through the real authority and Postgres.
Fixture upstream maximum concurrency was one; the broker waited then granted
after drain, final permits were empty, and an authority outage returned HTTP 503
without another upstream invocation. No model or production bus was involved.
Production Curiosity-to-model execution remains **UNVERIFIED**; these checks do
not claim that an actual cognition run has completed.

## Review findings fixed

- Disabled/shadow admission could retain stale drain priority.
  Fix: capacity-only operation ignores pending demand decisions; shadow and
  unknown-slot candidates do not advertise drain priority. Real Postgres tests
  cover these states and the original schema with capacity disabled.
- Owner validation omitted resource and demand identity.
  Fix: check every ownership field; forged resource/demand regressions pass.
- Starlette disconnect could interrupt stream cleanup before release.
  Fix: retain a single independent cleanup task and close upstream/client before
  releasing. ASGI disconnect and pre-body failure regressions cover the lifecycle.
- Ordinary bus waiters could exhaust the local semaphore while waiting for a
  durable owner, blocking that owner's continuation.
  Fix: shared admission precedes the local thread gate; admitted background
  owners bypass the ordinary background gate. Owner-starvation regressions pass.
- Shutdown could cancel a supervisor before its `finally` ran.
  Fix: the actual concurrent future retains a completion callback; request
  occupancy survives until backend completion or process-loss TTL recovery.
- Final HTTP ownership checks could complete after the caller deadline.
  Fix: include validation and stream lease checks within the remaining budget.
- Two existing Gateway tests inherited lane routing while asserting route-table
  only behavior, then attempted real upstream calls.
  Fix: pin that mode in those tests; production fallback behavior is preserved.

Independent review found no remaining material issue after these corrections.
The subsequent PR #2208 merge was inspected locally and checked with the suites
below; an additional subagent review was unavailable because its usage limit
was reached.

## Current-main integration

PR #2208 replaced mandatory voice finalization while this patch was being
verified. The branch incorporates that change and carries the lease through
reflection, re-reflection, and conditional response repair. Accepted drafts
remain unchanged and do not make an extra repair call.

## Configuration and rollout

The following new keys were synced from this branch's templates into the primary
checkout's ignored `.env` files. All 106 existing runner/Gateway values were
preserved:

- `DURABLE_RUNS_CAPACITY_ENABLED=false`
- `LLM_GATEWAY_CAPACITY_ENABLED=false`
- `LLM_GATEWAY_CAPACITY_URL=http://durable-runs:8121/capacity`

Env/compose parity passes for both changed services. The definition lock was
regenerated against the current base; no metric definition changed. No secrets
or local `.env` files are committed. No production migration, restart or
admission activation was performed.

For rollout, apply `manual_migration_gateway_capacity_v1.sql` after the original
admission migration in the checkpoint database, enable/restart the runner's
capacity authority, then all Gateway replicas, then updated auxiliary consumers,
then admitted Curiosity. Widening remains disabled pending compatibility audit.
The new flag values take effect only after the corresponding services restart
through `scripts/safe_docker_build.sh <service> up -d --build` from a worktree.

Before disabling the authority, drain/pause admitted work and disable Gateway
capacity enforcement; keep additive SQL history. Direct backend callers, DNS
aliases beyond trailing-slash normalization, and upstream work surviving process
loss past the permit TTL remain outside the ownership guarantee. Expiry alone
is not evidence that GPU generation stopped.
