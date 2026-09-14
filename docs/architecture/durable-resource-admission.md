# Durable resource admission: implementation evidence and ADR

## Evidence before implementation (2026-09-12)

The existing service `services/orion-durable-runs` already depends on
`langgraph==1.2.11` and `langgraph-checkpoint-postgres==3.1.2`. Its
`app/main.py::_open_checkpointer` uses `AsyncPostgresSaver` with a psycopg pool.
`app/graph.py::build_curiosity_graph` checkpoints the existing Curiosity phases;
`app/runner.py::resume_unfinished` recovers unfinished threads. Replacing this
with a new service or Functional API would disrupt existing checkpoints without
adding a capability. Keep the Graph API and existing node names.

`orion/schemas/durable_run.py` defines stable run/thread/correlation IDs and
Curiosity `investigate` / `self_inquiry` briefs. Cortex Orch's existing dispatch
publishes the run over Redis pubsub before replying accepted: subscriber presence
is not durable acceptance. Replace this optionally with a short persistence
receipt RPC. The timeout covers acceptance only, never queue waiting.

The actual execution path is runner -> Hub Curiosity turn RPC -> unified turn ->
harness governor/FCC -> Gateway Anthropic passthrough, with Cortex Exec handling
the existing stance/finalize legs. Preserve that established bridge; do not claim
that the primary study already executes directly through Cortex Exec.
`services/orion-hub/scripts/curiosity_investigation.py::_turn_result_for` currently
ignores per-request model/timeout and serializes all runs under one lock. These
must honor admitted requests for lane selection to have real runtime behavior.

Gateway already owns routing, overrides, shared-backend admission and `/routes`.
Its bus admission ledger omits primary FCC HTTP traffic, so it cannot be treated
as complete backend occupancy. Route names are not capability declarations.
Alternative lanes require explicit operator compatibility AND a matching real
route catalog entry. No alternative lane is enabled by default.

Actions' `app/main.py::_dispatch_scheduled_workflow` already sends due work back
to Cortex. Its calendar scheduler is unchanged. Existing SQL conventions use
idempotent `services/orion-sql-db/manual_migration_*.sql`; new admission tables
follow that convention, while LangGraph owns its checkpoint migrations.

## Ownership and proposed seam

- Cortex: policy front door and durable acceptance receipt.
- Existing LangGraph service: checkpoint position, interruption, retry timing,
  recovery, controls and terminal result. Queue waiting releases its graph task.
- Thin shared admission module: persisted demands, FIFO capacity arbitration,
  exclusive fenced leases, expiry, release and transactional event outbox.
- Hub/governor/Exec/Gateway: existing execution path, carrying the granted token
  and checking it before protected inference; routing remains Gateway-owned.
- Postgres: checkpoints, submission inbox, demands, leases and lifecycle history.
- Redis: receipt RPC and lifecycle/grant wakeups; database reconciliation recovers
  dropped wakeups. There is no assumption of RedisJSON or RediSearch.

New files: `orion/schemas/resource_admission.py`, `orion/durable_admission/`,
one manual SQL migration, runtime admission adapter and focused tests/eval.
Existing files: durable graph/runner/main/settings/config, durable schemas and
registry/channels, Cortex intake, Hub Curiosity, Gateway and lease propagation.

## Metric quality gate

Operational counts/timings come from durable admission rows and transition
timestamps, not cognitive signals. Queue depth, age and wait duration are
causally related views of the same queue and are explicitly redundant, not
independent model inputs. Their anchor is queue conservation and elapsed time;
lease counts obey capacity invariants. Empty queues have exact zero depth/age.
There is no live production data for this new seam yet: production sanity is
UNVERIFIED. Expose inspectable operational facts and test them on isolated
Postgres; do not wire them into any cognition detector or model. No estimated
quality/health telemetry will be fabricated. All policy instrumentation is
removable without changing existing cognition manifests.

## Acceptance and rollout boundary

Verify immediate persisted receipt, interrupted waiting with no live run task,
restart recovery, FIFO exclusive admission, fake-clock widening/hysteresis,
single winner across aliasing lanes, fenced renewal/expiry, bounded retries,
control operations and terminal release. Run existing sync routing regressions.
Feature defaults remain off; no deployment or production configuration writes.
Private prompts remain in the already-private checkpoint/inbox database; lifecycle
events carry identifiers and bounded operational facts, never prompt/material.

LangGraph interrupt behavior was checked against the official documentation:
https://docs.langchain.com/oss/python/langgraph/interrupts
https://docs.langchain.com/oss/python/langgraph/persistence
Pre-interrupt effects must be idempotent because a resumed node starts again.

## Decision and implemented flow

Use the existing Graph API: explicit `resource_request -> resource_wait ->
run_started -> harness_turn` and `retry_wait` branches are inspectable and retain
the existing post-turn nodes. `AsyncPostgresSaver` stores thread position. The
small `durable_admission_runs` inbox stores the immutable submission, a control
request and a terminal projection; it is not another workflow engine. The broker
never inspects a workflow node, drives a graph or invokes inference.

```text
Hub investigation/self_inquiry tick
  -> existing Cortex request with DurableRunRequestV1 + admission
  -> Orch short receipt RPC -> durable runner -> Postgres commit -> receipt
  -> idempotent {run_id}:harness_turn:{resource} demand
  -> LangGraph resource_wait interrupt; no per-run task/connection/RPC remains
  -> broker transaction grants one fenced physical reservation + event outbox
  -> Redis resource event wakes runtime; SQL reconciliation recovers lost wakeups
  -> graph validates lease and starts inference timeout
  -> existing Hub turn RPC -> unified turn -> governor/FCC -> Gateway HTTP
       (existing Cortex Exec stance/finalize calls remain part of the turn)
  -> existing result read, attention row, journal, finish
  -> atomic terminal projection + lease release + durable completion outbox
```

The global broker transaction lock is held only during capacity arbitration;
partial unique indexes prevent active duplicates by backend, run and demand.
Per-run Postgres session locks serialize graph writers across replicas and are
released at interruption. A process can hold a connection during an executing
attempt, never while queued. An inbox scan recovers death before checkpointing.
On recovery after dispatch, the old generation is revoked before replay. A
persisted completed turn is retained when a later node needs retry/recovery.
At most four graph drivers run concurrently per service process.

Cancellation is irreversible and wins a concurrent completion transaction.
Pause revokes the lease and retains the checkpoint; resuming reacquires capacity.
Retry backoff is a checkpointed interrupt with defaults of three total attempts,
30 seconds base and 300 seconds cap. Waiting consumes no attempts. An optional
timezone-aware `admission.deadline_at` bounds the whole workflow. Queue waiting
does not inherit the legacy stale-thread age limit.

## Lane policy and measured inputs

The first slice supports `exclusive`, `lease_scope=run`, `priority=background`;
unsupported modes fail schema validation. Shared/opportunistic capacities are
deferred until a concrete caller needs them.

Runtime discovery reads Gateway `/routes`: configured upstream URL, `status=up`,
measured context size and vision support. It probes each distinct upstream's
`/slots`; unknown/invalid occupancy or unavailable route discovery grants no
capacity. The broker receives this catalog as data and performs no HTTP calls.
Trailing-slash URL aliases share one capacity reservation.

No compatibility is inferred from lane names. `DURABLE_RUNS_LANE_POLICY_JSON`
maps actual route IDs to operator declarations. For example, **only after an
operator has verified the models**, an existing `metacog` route could be
declared compatible with `agent`:

```json
{
  "agent": {"capabilities": {"structured_output": true}},
  "metacog": {
    "compatible_with": ["agent"],
    "capabilities": {"structured_output": true},
    "quality_drop": 1,
    "max_quality_drop": 1,
    "switching_cost_seconds": 30
  }
}
```

This is an example declaration, not a shipped assertion about those models.
The shipped policy is `{}`. For a minimal real Hub demand, compatible candidates
are derived from this declaration and frozen at acceptance. An explicit caller
list restricts candidates further. Each grant still checks the current real
catalog and declarations; disappearing compatibility never authorizes a grant.
Hard requirements belong in `admission.requirements`, for example
`{"structured_output":true,"minimum_context_tokens":32768}`. Context size comes
from Gateway, and a declaration cannot inflate it.

Routing selection is operator override, capability validation, experimental pin,
widening, then preferred lane. Override/pin disable widening but never bypass
hard requirements. After 1200 seconds the preferred lane remains eligible;
an alternative must beat its estimated start by more than 120 seconds plus
declared switching cost. Original demand age survives widening and retry. A
single atomic grant wins; there are no separate losing lane jobs to clean up.
The first assigned lane is retained on later attempts, so restart/pause cannot
silently migrate a study.

Estimates are conservative reservations, **not learned runtime predictions**:
active declared inference budget minus elapsed time (with a nonzero lease floor
while held), plus declared budgets of older eligible work ahead. Renewable lease
TTL alone is not an inference estimate. No cold-load or health penalty is
fabricated; unhealthy routes are ineligible. Busy or unknown external occupancy
uses the declaration's `busy_budget_seconds` (default 3600) as a conservative
reservation assumption. It is not a measurement of remaining inference time and
can overestimate a nearly finished legacy request; inspect shadow decisions
before enabling widening. Unknown occupancy never permits a grant on that
backend. FIFO across all demands prevents younger native or widening
requests overtaking eligible older work; the finite-load eval covers both kinds.
This does not promise fairness under arbitrary infinite arrivals.

## Operational instrumentation and privacy

`PostgresAdmissionStore.queue_snapshot()` produces `GET /admission` directly
from demand rows, lease rows and committed `durable_resource_events`:

| Surface | Producer / interpretation |
| --- | --- |
| Queue depth / oldest age by preferred lane | pending demands, `created_at` |
| Active leases by assigned lane | active rows with future `expires_at` |
| Wait histogram / admission latency | first lease `granted_at - run.created_at`; one sample per run |
| Alternative assignments | first assigned lane differs from requested lane |
| Widening / suppression | changed broker decision events, suppression counts by reason |
| Expiry / release / retries / failures | committed lifecycle event counts |
| Checkpoint resume failures | runtime caught exception plus checkpoint node in event |

First-admission latency and wait duration are the same instrument, not two
independent signals. Suppression counts count changed decisions, not every poll.
Retry events are transition counts, not an inferred model-error rate. All are
redundant operational projections of the underlying queue/lease/event facts,
anchored in elapsed time, FIFO ordering and capacity conservation. They feed
debug/API output and admission policy only; no cognitive detector consumes them.
The existing metric lineage and definition-drift gates cover channel registration.
The two added registered bus channels and the existing governor-cancel producer
change are recorded in `metric_definitions.lock.json`.

Isolated real Postgres data confirmed nondegenerate waits, widening/suppression,
renewal/expiry and return to empty queues/zero active leases. The 20-request eval
served every unique request with at most two physical fixture backends occupied;
its final active and queued lists were empty. There is no production observation
for this new seam: **UNVERIFIED**. No metric is admitted as a cognitive input.
Removing these views is cheap and does not change a training/schema default.

Events preserve run/thread/correlation and session identifiers. Existing turn
artifacts preserve the current mind/trace lineage. Prompts and generated content
stay in the existing private checkpoint/inbox and completion-artifact boundary.
Resource transition events contain operational details, never prompt text; the
existing durable completion event still carries its existing result detail for
Hub's consumer. Tokens are per-request/process, never global environment or model
content. The new HTTP control endpoints share the existing trusted internal
network boundary and do not add public authentication.

## Activation and rollback

The checked-in operator templates now select the admitted Curiosity path. Code
and compose fallbacks remain false when a deployment supplies no env contract;
`DURABLE_RUNS_ADMISSION_SHADOW` also remains false because shadow mode does not
execute admitted work. Relevant operator-template values:

| Service | Settings |
| --- | --- |
| Runner | `DURABLE_RUNS_ADMISSION_ENABLED=true`, `DURABLE_RUNS_CAPACITY_ENABLED=true`, `DURABLE_RUNS_ADMISSION_SHADOW=false` |
| Runner policy | `DURABLE_RUNS_WIDENING_ENABLED=true`, `DURABLE_RUNS_WIDENING_AFTER_SEC=1200`, `DURABLE_RUNS_WIDENING_HYSTERESIS_SEC=120`, `DURABLE_RUNS_LANE_POLICY_JSON={}` |
| Runner timing | tick 5s, lease 90s, heartbeat 15s, retry attempts 3/base 30s/cap 300s |
| Runner discovery | `DURABLE_RUNS_GATEWAY_URL=http://llm-gateway:8210` |
| Orch | `CORTEX_DURABLE_ADMISSION_ENABLED=true`, receipt timeout 10s |
| Hub | `HUB_CURIOSITY_DURABLE_ADMISSION_ENABLED=true`, lease validator `http://127.0.0.1:8124/leases/validate` |
| Gateway | `LLM_GATEWAY_CAPACITY_ENABLED=true`, `LLM_GATEWAY_LEASE_VALIDATION_ENABLED=true`, validator `http://durable-runs:8121/leases/validate`, timeout 2s, interval 5s |

Before restarting with those values, apply the additive manual admission
migration to the checkpoint database, followed by the Gateway capacity migration.
The runner fails startup rather than silently operating without either contract.
The remaining consumer-first order is:

1. Apply both migrations and retain LangGraph's existing saver setup/migrations.
   Back up as usual.
2. On a fresh rollout, install consumers first: Gateway, Thought, governor, Exec
   and Hub with explicit false overrides until the authority is ready.
   Admitted FCC requests use existing `HARNESS_LLM_GATEWAY_URL` directly, with the
   scoped lease header, avoiding an unverified external proxy stripping the token.
   Legacy FCC requests keep their current proxy and authentication path.
3. For a separate dry run, temporarily enable runner shadow and submit dedicated
   requests through `/runs`; inspect would-be decisions. Shadow never invokes
   the graph's cognition nodes, takes a lease or alters legacy routing. It is
   not a duplicated tap of live cognition. Cancel shadow fixtures before rollout.
4. Restore runner shadow to false and verify `/routes`, `/slots` and authority
   connectivity. Restart every Gateway replica, Thought, governor and Exec before
   Orch and Hub. Start only the selected Curiosity workflow.
5. Verify durable receipt, pending checkpoint, grant, fenced Gateway request,
   actual turn artifact and terminal release on the real rail. Only then declare
   production verified. Approve alternative-lane policy only after auditing its
   capability and compatibility declarations.

Use `scripts/safe_docker_build.sh <service> up -d --build` from an isolated
worktree for each affected service when rollout is authorized. Bus URLs remain
`redis://100.92.216.81:6379/0`; container service names are not Redis addresses.
Local `.env` files must be synchronized with changed examples and remain ignored.

To disable, stop new Hub admission submissions first, pause/cancel or drain
existing admitted runs while the runner/validator are still available, then
disable runner admission and Gateway validation. Keep additive SQL/checkpoint
tables for inspection and future resume; do not drop them to roll back.

## Limits and next smallest patch

Strict exclusivity covers managed leases on the same canonical backend URL.
Existing unleased synchronous traffic remains compatible and can race the sampled
idle probe. DNS aliases to one backend are not canonicalized beyond trailing
slashes. This slice does not claim global physical exclusivity across legacy
callers. Next: share an authoritative Gateway capacity reservation across both
HTTP and bus traffic before broadening strict exclusive workloads.

That implementation is documented in
[Gateway capacity and durable leases](durable-gateway-capacity.md): an optional
shared Postgres authority closes this race for participating Gateway transports,
with owner propagation through stance and finalization.

Lease fencing prevents stale dispatch/result acceptance. A blocking upstream
HTTP thread or already-issued model inference may continue consuming hardware
until it exits; its existing Gateway permit is retained. Tool effects already
performed by an interrupted FCC attempt cannot be rolled back; replay is at
least once at that phase. Existing graph tail IDs make journal/attention writes
idempotent, and bounded node retries preserve completed inference.

The full production Curiosity-to-model path is **UNVERIFIED**: it was deliberately
not triggered. Real Postgres recovery, a running Docker HTTP receipt/wait/cancel
smoke, transport regressions and the actual route adapter are verified separately.
No self-concept cognition logic was invented. The supplied self-inquiry brief is
the existing study abstraction. The intended direct Exec primary-inference arrow
was changed to preserve the repository's actual Hub/governor/FCC ownership.
