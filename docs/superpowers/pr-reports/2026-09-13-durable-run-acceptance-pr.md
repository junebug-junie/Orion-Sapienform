# Durable-run acceptance and recovery completion

A durably admitted Curiosity turn could still be cut off by the runner's legacy
Hub RPC ceiling and could accept a result with the wrong run/attempt identity.
This patch fixes those two failures and completes the connected acceptance
scenario on top of #2205, #2206 and #2209, including #2208's conditional response
repair. An accepted draft passes unchanged; repair runs only when required.

## Evidence and ownership

The existing Cortex dispatcher obtains a committed receipt; LangGraph owns
workflow position, waits, retries and terminal state; the thin Postgres broker
owns demand ordering and fenced grants. Gateway's shared permit authority covers
participating bus and HTTP inference. Actions retains calendar scheduling.
Discovery and design decisions are recorded in the
[admission ADR](../../architecture/durable-resource-admission.md) and
[capacity ADR](../../architecture/durable-gateway-capacity.md).

The connected acceptance follows:

`Cortex receipt → persisted LangGraph wait → grant bus handler → recovered graph → Hub Curiosity/unified turn → Thought + Governor/FCC → Cortex Exec auxiliary calls + Gateway → checkpointed result → attention/journal/completion → release`.

Four cases cover preferred versus explicitly compatible widened lanes, each
with accepted draft versus conditional repair. The widening clock advances
1201 seconds; the preferred lease remains occupied at the alternative grant.
A separate test kills the actual runner child process during a loopback backend
request. The independent request permit prevents a replacement lease until that
backend drains; the recovered run then completes once with a new generation.

## Changes

- `services/orion-durable-runs/app/runner.py`: honor admitted inference budgets
  while preserving the runtime's deadline owner; validate admitted reply kind,
  envelope correlation, payload run and attempt correlation.
- Service tests: connected real service adapters over typed in-process envelopes
  and isolated ASGI HTTP, a real Postgres saver/store, process-kill regression,
  and six focused RPC contract checks. Only SQLAlchemy is added as a test-only
  dependency to match the existing Hub/Exec pin.
- Durable CI: install acceptance dependencies and trigger on the connected
  Orch/Hub/Thought/Governor/Exec/Harness/Gateway seams.
- Service README and [acceptance contract](../../architecture/durable-run-acceptance.md):
  coverage map, reproducible commands, compatibility and verification limits.
- Metric definition lock: refresh required merge-base metadata; no definition
  changes, new metrics, runtime env keys, schemas or migrations in this patch.

The existing additive migrations remain
`manual_migration_durable_resource_admission_v1.sql` and
`manual_migration_gateway_capacity_v1.sql`; neither is applied to production.
No env template changed. Build-only ignored env files in the disposable worktree
were copied from safe templates; operator configuration was not changed.

## Validation

- Clean CI-style durable suite with real isolated Postgres: **62 passed**.
- Additional hostile-config acceptance run: **4 passed**, with dotenv reads
  forbidden and no inherited FCC/operator directory created.
- Separate admission fairness eval: **20/20 served**, empty final queue/leases.
- Separate Gateway capacity eval: **40 contended rounds**, two clients, no final
  permits, queued demands or leases.
- Runner Docker build: **PASS**, isolated `orion-acceptance-review` project.
- Every check in `orion-static-gates.yml`: **PASS** locally, including Hub JS,
  env-sync regressions, metric gates and Graphify preservation checks.
- GitHub CI and mergeability: checked on the published PR before final handoff.

The fixture model/FCC outputs and external knowledge reads are explicit test
substitutions. Actual service handlers, envelope serialization, durable waits,
bus wakeup, lease checks, capacity acquisition and emitted artifacts are exercised.
Journal/attention SQL-writer persistence and production cognition are **UNVERIFIED**;
nonempty fixture output is not presented as Orion's own finding.

## Review findings fixed

- Test settings could inherit private graph or FCC configuration.
  Fix: explicit Pydantic aliases, no dotenv reads during service imports,
  disposable policy/FCC paths, and a transport that rejects nonfixture hosts.
  Evidence: hostile-config acceptance run passes all four cases.
- Direct graph driving did not prove grant-driven wakeup.
  Fix: deliver the persisted grant through the installed bus handler, assert the
  wakeup, then reconcile; deliberately replay the unacknowledged grant.
  Evidence: all four cases finish once with one terminal history entry.
- The report claimed attention without checking the emitted artifact.
  Fix: assert one attention envelope and its run/correlation alongside journal
  and completion. Evidence: connected acceptance passes.
- Completed test tasks could spin the fixture bus drain.
  Fix: drain a task snapshot and remove it explicitly after awaiting completion.
  Evidence: connected acceptance passes, with no leftover RPC or subscription.
- CI lacked imports and triggers for the expanded acceptance rail.
  Fix: pin the existing SQLAlchemy test dependency and include connected paths.
  Evidence: clean virtualenv with only declared CI dependencies passes 62 tests.

Independent subagent review found no remaining material code issues.

## Compatibility, activation and limits

The existing Graph API and Hub/governor/FCC primary inference path are preserved
because they are the actual repository architecture. Stance, reflection and
conditional repair continue through Cortex Exec. The broker never invokes
inference. Ordinary synchronous traffic retains its existing path. Exclusive,
run-scoped background admission is the explicitly allowed first slice;
shared/opportunistic admission and a new self-concept algorithm are not added.

Admission, widening, Gateway validation and Gateway capacity flags remain opt-in.
Follow the capacity ADR's consumer-first sequence: additive migrations and runner
authority, all Gateway replicas with capacity enforcement, updated consumers,
then selected Curiosity admission. Use runner shadow evaluation before admitted
cognition and leave widening off until real compatibility has been audited.

No deployment or activation was performed. The next operational step is the
already documented selected-workflow rollout and an observed production artifact
chain. Fencing cannot undo prior FCC tool effects; replay of interrupted inference
is at least once. Simultaneous Gateway loss, direct backend callers and arbitrary
DNS aliases remain outside the guarantee tested here.
