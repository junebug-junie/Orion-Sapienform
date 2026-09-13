# Gateway capacity and durable leases

## Evidence and next slice (2026-09-13)

PRs #2205 and #2206 are merged and redeployed. A read-only check of the live
runner reports healthy, two active legacy runs, and admission disabled. All
admission flags remain false; this patch does not activate production cognition.

`app/upstream_admission.py` owns process-local bus request semaphores. Anthropic
and OpenAI HTTP paths bypass that gate, while the durable broker arbitrates
leases using Postgres and sampled upstream `/slots`. A legacy request can start
between the idle sample and durable grant. Another local semaphore cannot make
that check atomic with the broker transaction.

## Contract and ownership

Extend the existing Postgres admission authority with short request permits.
Both a durable lease grant and a Gateway permit acquisition use the existing
`ADMISSION_LOCK` transaction. A durable reservation excludes foreign requests;
its owning run may make one model request at a time. An ordinary request permits
up to Gateway's configured upstream capacity, but prevents a durable grant until
the actual request completes. Request IDs make acquisition retries idempotent.
Lease revocation does not release a still-running request's physical reservation.

The runner exposes typed internal `/capacity/acquire`, `/capacity/renew`,
`/capacity/release` and `/capacity` inspection endpoints. Capacity can initialize
without enabling durable cognition. Gateway remains responsible for route
resolution, caller deadlines and actual inference; the authority makes no model
calls. The existing local bus semaphore still protects the thread pool.

Hub/Thought stance and governor reflection/voice calls carry the owning lease
and assigned lane, preventing a study from waiting against its own reservation.
Gateway bus chat, Anthropic Messages and OpenAI chat use the same authority when enabled. Backend
escalation after planning must not bypass the permit's resolved backend.

## Acceptance checks

1. Race ordinary acquisition with a durable grant on real Postgres: exactly one
   resource owner wins, including normalized route aliases.
2. Serialize owner calls; exclude foreign/unleased calls; release, expiry,
   duplicate acquisition and stale renewal preserve capacity invariants.
3. Keep request permits alive until actual thread/stream completion, including
   cancellation of the caller and revocation of its durable lease.
4. Preserve existing unprotected behavior when the new flags are off. Fail closed
   on unavailable authority when enforcement is enabled.
5. Test the stance/reflection/voice producer-consumer chain with the assigned
   lease. Run separate transport checks, an isolated real-Postgres smoke/eval,
   affected Docker builds and independent review before publishing the PR.

## Rollout boundary

An additive migration and explicit defaults-off flags gate this change. First
bring up the capacity authority, then every Gateway replica, then admitted
Curiosity. The shared authority covers participating Gateway calls; direct
backend access and already-running uncancellable upstream work remain outside
that software guarantee. No production activation or migrations are performed
as part of this implementation.

## Implemented API and failure behavior

Apply `services/orion-sql-db/manual_migration_gateway_capacity_v1.sql` after the
existing admission migration, in the checkpoint database. The new
`durable_gateway_permits` table stores request ownership, timestamps and status;
it contains no prompts, results, retries or graph position. Capacity mode fails
startup without its migration. Admission without capacity mode continues using
the original schema only.

`DURABLE_RUNS_CAPACITY_ENABLED=false` gates the authority independently of
`DURABLE_RUNS_ADMISSION_ENABLED`. Gateway opt-in is
`LLM_GATEWAY_CAPACITY_ENABLED=false`, with
`LLM_GATEWAY_CAPACITY_URL=http://durable-runs:8121/capacity`.

| Internal endpoint | Request / result |
| --- | --- |
| `POST /capacity/acquire` | Stable request ID, correlation, resolved lane/backend, max inflight, caller budget, optional lease; returns acquired/reason/permit |
| `POST /capacity/renew` | Request ID + opaque permit ID; returns valid and current permit |
| `POST /capacity/release` | Same token; idempotent release result |
| `GET /capacity` | Unexpired request permits with lane/backend/owner and timestamps |

The server uses `DURABLE_RUNS_LEASE_SECONDS` (default 90s) for ticket expiry.
Gateway renews at most every 15s and within one third of that duration. The
request permit has a separate identity from its durable owner's lease generation.
Released/expired request IDs cannot acquire another ticket; changed payloads
under the same ID conflict. Acquisition retries preserve the original payload.

Ordinary requests contend up to `LLM_GATEWAY_UPSTREAM_MAX_INFLIGHT`; owner
requests serialize at one per exclusive lease. If deployment revisions disagree
on ordinary capacity, the smallest limit among active requests and the new caller
wins. This remains a trusted internal API, with the same network boundary as the
existing runner controls.

A waiting durable demand advertises only physical backends it can actually
select now. New ordinary calls yield that grant window while existing calls
drain. Hysteresis-suppressed alternatives, unknown `/slots` occupancy,
paused/cancelled runs and shadow decisions do not reserve that window.
Capacity-only operation ignores old wait decisions because no admission driver
can fulfill them.

Losing a workflow lease prevents further dispatch and accepted output. A
still-running thread separately retains and renews its request permit until
actual completion. Authority outages reject protected dispatch/results;
unconfirmed releases recover through expiry. Gateway bus calls retain local
thread isolation and gain the shared ticket. Anthropic Messages and OpenAI chat
HTTP also acquire tickets, including streaming. Acquisition and inference share
the caller budget. Streams close their upstream before releasing capacity.
Context-overflow escalation to another backend is disabled in capacity mode;
the original error returns instead of invoking an unreserved destination.

Stance/Thought and reflection/rereflection/voice contexts carry the typed owning
lease and assigned route. Exec accepts a route from this contract even when the
public model picker does not list it. Conflicting explicit route overrides are
preserved and rejected by Gateway fencing. Legacy requests omit the new optional
stance field from their wire payload, preserving strict older consumers.

## Evidence, instrumentation and limits

`PostgresCapacityStore.acquire`, `renew`, `release` and `expire` produce request
ownership facts. `snapshot()` exposes unexpired active rows; no cognitive signal
or new detector is introduced. Permit counts and durable occupancy are dependent
capacity views, anchored in conservation of exclusive ownership. An isolated
real-Postgres eval exercised 40 rounds with two competing clients, observed both
ordinary and durable winners, and returned to zero permits, demands and leases.
The operational view is removable without changing execution or model defaults.

This closes the race for participating Gateway traffic and trailing-slash URL
aliases. It cannot prevent direct backend calls, canonicalize arbitrary DNS
aliases, or force issued generation to stop after a Gateway dies or partitions
past its ticket TTL. A cancelled handler whose process remains alive retains its
ticket through actual thread completion. Expiry is not proof the GPU stopped;
upstream `/slots` remains a conservative occupancy check for durable grants.

Roll out the migration and runner authority first, then all Gateway replicas with
capacity enforcement, then updated Thought/Hub/governor/Exec consumers, then
admitted Curiosity. Keep widening off until compatibility is audited. Before
disabling the authority, drain/pause admitted work and disable Gateway capacity
enforcement. Retain additive SQL tables for history. The production
Curiosity-to-model scenario remains **UNVERIFIED** until activation is performed.
