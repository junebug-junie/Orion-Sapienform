# GPU pool stage 4.4 — callers carry the GPU lease ref; the gateway attaches

## Summary

In stage 4.5 a durable run will hold one GPU pool "hold" for its whole life. This PR teaches every
service that makes or forwards an LLM call for that run to carry a reference to the hold, and
teaches the LLM gateway to run such a call *inside* the hold instead of queueing for a second
lease. Nothing issues a hold yet, so nothing changes at runtime until 4.5.

- **The gateway attaches, never re-leases.** A call with the ref (bus `options.gpu_lease`, HTTP
  `X-Orion-Gpu-Lease`) is placed with the pool's `attach` verb. The agent role has one slot and the
  run's hold sits in it; a plain lease for the run's own call would wait behind the run forever.
  A refused attach (hold gone, stale generation) returns `gpu_pool_unavailable` and never falls
  back to a plain lease. A context overflow under a hold is returned as is (the child cannot leave
  the hold's role). The ref is never forwarded to llama.cpp.
- **Every hop carries it.** FCC (header, sent straight to the gateway like the old token),
  harness finalize (reflect/repair), harness-governor (admits a hold-carrying turn outside the
  legacy lock, dedupes by hold fence), cortex-exec (`options.gpu_lease`), orion-thought (stance),
  Hub turn orchestrator and thought client. The old `resource_lease` token rides alongside and is
  still checked until 4.6.
- **Hub fences a carried hold with the pool.** `CuriosityTurnRequestV1.gpu_lease` (consumer first)
  is checked with the pool's `status` verb before any generation: wrong holder, gone, or re-granted
  (stale generation) refuses the turn. Door-A outreach picks the ref up from the finish detail and
  runs the same check before composing; a bad or malformed ref records an outreach skip and still
  releases the grant, and never composes without the ref. A held turn always names the `agent`
  route (`llamacpp/agent` for FCC), whatever label the caller sent.
- **Held FCC calls may wait one interleaved inference.** Under a hold the HTTP passthrough waits the
  class's bus budget (`LLM_GATEWAY_POOL_[BACKGROUND_]WAIT_SEC`), not the 60s passthrough budget,
  so interleave (spec Decision 1 rule 3) cannot become a mid-turn 503.
- **Field: one waiting-run count across the cutover.** `durable_demand_pending` is now legacy
  pending demands UNION waiting durable-run pool holds. A frozen demand whose run has any pool
  hold is superseded, so a resumed run counts once and a finished one is not resurrected as
  waiting. `gpu_pool_waiting` counts everything else (request leases and operator holds), so no
  lease is counted in both.
- **Client helpers** (`orion/gpu_pool/client.py`): `gpu_lease(..., hold=ref)` sends `attach`;
  `lease_status()` and `validate_hold_ref()` for the `status` read; `durable_run_holder(run_id)`.

## Outcome moved

- The single most important correctness rule of stage 4 ("the run must never wait behind itself")
  is enforced in code and proven end to end before any hold exists: an acceptance test runs a whole
  curiosity turn under a hold through the real Hub, Thought, Governor, Cortex Exec and Gateway
  adapters, and every model call (stance, FCC over HTTP, finalize) is an attach. Dropping the ref
  at the gateway fails it (mutation-checked).
- The queue-contention field reading keeps meaning "durable runs waiting for a GPU" through the
  4.5 cutover, including the window where a resumed run has both a frozen demand and a new hold.

## Current architecture

- The durable token (`ResourceLeaseV1`) rides every call as `X-Orion-Resource-Lease` /
  `options.resource_lease`; the gateway's `LeaseGuard` checks it against durable-runs
  `/leases/validate`, then takes its **own** pool request lease for placement.
- Path of a durable turn: durable-runs → `CuriosityTurnRequestV1` → Hub `_turn_result_for` →
  `_generate` → `execute_unified_turn` → stance (`StanceReactRequestV1` → thought → cortex-exec →
  gateway bus) and harness (`HarnessRunRequestV1` → governor → runner → FCC over HTTP, finalize
  via cortex-exec → gateway bus).
- `durable_demand_pending` read `durable_resource_demands WHERE status='pending'`;
  `gpu_pool_waiting` read every queued/backlogged `gpu_pool_leases` row.

## Architecture touched

- Contracts (additive, optional): `GpuLeaseRefV1` now rides `HarnessRunRequestV1.gpu_lease`,
  `StanceReactRequestV1.gpu_lease`, `CuriosityTurnRequestV1.gpu_lease` (extra="forbid" — Hub must
  deploy before durable-runs 4.5 sends it), bus `options.gpu_lease`, HTTP `X-Orion-Gpu-Lease`
  (constants from 4.1). `GPU_LEASE_ROUTE = "agent"` (`orion/llm/resource_lease.py`).
- No new channel, no registry change, no env key.

## Files changed

- `orion/gpu_pool/client.py`: `hold=` → `attach`; `lease_status`, `validate_hold_ref`,
  `durable_run_holder`, `DURABLE_RUN_HOLDER_PREFIX`, `HOLD_LIVE_STATUSES`.
- `orion/llm/resource_lease.py`: `GPU_LEASE_ROUTE`.
- `services/orion-llm-gateway/app/{resource_lease,pool_placement,main,passthrough_proxy,anthropic_passthrough,openai_passthrough,llm_backend}.py`:
  parse the ref, attach, no overflow re-lease under a hold, lane routing keeps a held call's route.
- `orion/harness/{fcc_motor,runner,finalize}.py`, `orion/schemas/harness_finalize.py`: carry the ref.
- `services/orion-harness-governor/app/bus_listener.py`: admit + dedupe hold-carrying turns.
- `services/orion-cortex-exec/app/executor.py`: forward `gpu_lease`; route under a hold.
- `orion/schemas/thought.py`, `services/orion-thought/app/bus_listener.py`: stance carries the ref.
- `orion/hub/turn_orchestrator.py`, `services/orion-hub/scripts/{thought_client,curiosity_investigation}.py`,
  `orion/schemas/durable_run.py`: Hub side (status fence, turn, Door-A).
- `services/orion-field-digester/app/store.py`, `orion/field/queue_contention.py`, README: union source.
- Tests: `orion/gpu_pool/tests/test_client_attach.py`, `services/orion-llm-gateway/tests/test_gpu_lease_attach.py`
  (+ `conftest.py` fake pool holds), `orion/harness/tests/test_gpu_lease_transport.py`,
  `services/orion-hub/tests/test_curiosity_gpu_lease.py`, `services/orion-field-digester/tests/test_durable_waiting_sql_postgres.py`,
  `services/orion-durable-runs/tests/test_gpu_lease_turn_acceptance.py` (+ `acceptance_turn.py`/`acceptance_bus.py`
  fixture pool: holds, attach, status), additions in governor/thought/cortex-exec/hub test files.
- CI: `orion-gpu-pool-tests.yml` new `consumers` job (with Postgres); `orion-durable-runs-tests.yml` path triggers.

## Schema / bus / API changes

- Added: optional `gpu_lease: GpuLeaseRefV1` on `HarnessRunRequestV1`, `StanceReactRequestV1`,
  `CuriosityTurnRequestV1`; bus `options.gpu_lease`; HTTP `X-Orion-Gpu-Lease` honoured by the gateway.
- Removed / renamed: none.
- Behavior changed: `gpu_pool_waiting` excludes `kind='hold'` (no holds exist today: identical);
  `durable_demand_pending` includes waiting holds (none today: identical).
- Compatibility: an absent ref keeps every old wire shape (thought client pops it like `resource_lease`).
  `CuriosityTurnRequestV1` is extra="forbid": Hub must deploy before durable-runs 4.5 sets it.

## Env/config changes

- Added / removed / renamed keys: none. `.env_example` updated: no. Local `.env` sync: not needed.

## Metric gate — `durable_demand_pending` / `gpu_pool_waiting` re-point (CLAUDE.md 0A)

1. **Provenance.** `services/orion-field-digester/app/store.py` `DURABLE_WAITING_SQL`, read by
   `count_durable_demand_pending` / `oldest_durable_demand_pending_age_sec`; pool side
   `count_gpu_pool_waiting` / `oldest_gpu_pool_waiting_age_sec` (`AND kind = 'request'`). Consumed by
   `app/digestion/queue_contention.py` readers, unchanged.
2. **Independence.** Not a new signal: same source keys, same meaning. The two sources now read
   disjoint row sets (holds vs request leases), so no hold is counted in both. They stay causally
   correlated through the shared agent card, exactly as before.
3. **Theory anchor.** Unchanged: queue depth and head-of-line age of runs waiting for a GPU
   (gate doc `docs/superpowers/specs/2026-09-20-queue-contention-metric-gate.md`). Caveat: a hold's
   wait clock is `queued_since`, which restarts when an expired hold re-queues, while a demand's
   was its `created_at`. The 12h durable anchor was fitted on demands; re-fit it from a week of hold
   data after 4.5 (a knob, not a finding).
4. **Live data (2026-09-25, `orion-athena-sql-db`, read only).** Legacy: 12 pending demands,
   oldest 377,170 s. The new union query returns **12 / 377,170 s — identical**. Re-checked after
   the review fixes: legacy 11 / 378,519 s, union 11 — identical; `gpu_pool_waiting` 0 old and new. `gpu_pool_leases`
   holds 4,031 rows, all `kind='request'`, 0 with a `durable-runs:` holder, so the `kind` filter
   changes nothing today. Rest state: an empty queue reads 0 and 0.0 (tested); the legacy half has
   been non-zero for days (the stuck durable backlog), so it is not flat. EXPLAIN: anti-join on the
   `durable_resource_demands_fifo` index + `gpu_pool_leases_live_idx`, cost ~100.
5. **Existing mechanism.** Extends the existing source; nothing new built.
6. **Reversibility.** One SQL string in one reader; no schema, manifest or training default.

## Tests run

```text
services/orion-llm-gateway: pytest tests -q                              -> 322 passed (11 new)
orion/gpu_pool/tests -q                                                  -> 146 passed (15 new)
orion/harness/tests -q                                                   -> 365 passed (8 new)
services/orion-harness-governor: pytest tests -q                         -> 55 passed (1 new)
services/orion-cortex-exec: test_resource_lease_forwarding + overloaded  -> 29 passed (7 new)
services/orion-thought: pytest tests -q                                  -> 447 passed, 1 failed (pre-existing,
    env-dependent: test_settings_mind_enrichment expects http://orion-mind:6611, host env gives http://mind:6611;
    settings untouched here)
services/orion-hub: curiosity_gpu_lease/admission/investigation/thought_client/turn_orchestrator_ws_frames -> 207 passed
services/orion-field-digester: pytest tests -q (throwaway Postgres)      -> 257 passed, 5 skipped (6 new incl. 5 on Postgres)
services/orion-durable-runs: pytest tests -q (throwaway Postgres)        -> 145 passed (existing acceptance unchanged + 3 new held-turn)
static: check_metric_lineage --gate, check_definition_drift --gate, check_inner_state_registry,
        check_async_routes_not_blocking, check_chat_route_poachers, check_system_health_producers,
        check_sentience_instruments --static-only, check_gpu_pool_config, schema-registry/topology/grammar pytest gates -> all pass
fresh venv with only the new CI job's pins: every consumers-job step passes
```

Mutation check: removing `hold=hold` from the gateway bus path fails the held-turn acceptance (2 failed).

## Evals run

```text
services/orion-durable-runs/evals/{admission_fairness,gateway_capacity,elastic_fairness}.py (throwaway Postgres) -> exit 0
```

No new eval: 4.4 changes no scheduling behavior (no holds are issued). The hold policy eval belongs
to 4.3 (pool engine) per the spec.

## Docker/build/smoke checks

```text
Not run. No dependency, compose or env change; not deploying (per task).
```

## Review findings fixed

Code review ran in a subagent over the whole diff; no blockers. Fixed:

- Finding: a frozen pending demand whose run's hold had already *finished* was counted as waiting
  again (with its days-old `created_at`) until the Juniper-gated withdrawal.
  - Fix: a demand is superseded by any hold row of its run, whatever the status (no durable hold
    exists before 4.5 and no demand is written after it).
  - Evidence: Postgres test case "e" (held, released) now reads not-waiting.
- Finding: operator holds (`operator:<actor>`, `kind='hold'`) would have moved into
  `durable_demand_pending` under the 12h anchor.
  - Fix: the pool half requires a `durable-runs:` holder; `gpu_pool_waiting` counts
    `kind='request' OR holder NOT LIKE 'durable-runs:%'`.
  - Evidence: `test_operator_holds_stay_in_gpu_pool_waiting` on Postgres.
- Finding: a held FCC child waited at most 60s, so interleave would 503 a turn.
  - Fix: the class's bus wait budget under a hold.
  - Evidence: `test_anthropic_header_attaches_and_is_not_forwarded_upstream` asserts the deadline.
- Finding: Door-A did not validate the ref, and a malformed ref composed *without* it (self-deadlock).
  - Fix: `status` check with holder before composing; malformed → skip + release, never compose.
  - Evidence: two new Hub tests.
- Finding: a caller's chat FCC label could reach a held turn (chat-class attach).
  - Fix: `execute_unified_turn` forces `llamacpp/agent` once when a ref (and no old token) is present.
  - Evidence: orchestrator test sends `llamacpp/chat`, asserts `llamacpp/agent`.
- Finding: a refused attach was sent twice (withdraw re-sent it because the reply had no lease_id).
  - Fix: a reply without a lease_id means nothing was created; only a *lost* reply re-sends.
  - Evidence: the test now asserts the exact sequence `["attach"]`.
- Finding: heartbeat interval keyed off the caller's `kind`, not the sent request's. Fix: `req.kind`.
- Finding: an explicit route under a ref was forwarded raw. Fix: alias + normalize, else `agent`.
  Evidence: `AGENT` → `agent`, `not-a-route` → `agent` cases.
- Finding: thought-client comment claimed consumers reject extra fields. Fix: reworded (the risk
  is a silent drop by an un-upgraded consumer).

Not fixed (noted):

- Governor drops a malformed-payload message without replying (pre-existing, same as the old token).
- 4.5 hazard: durable-runs `runner.py` `_call_reflect_llm` calls the LLM with neither lease; if a
  reflect run holds an agent hold in 4.5 it would queue behind itself. For 4.5 to resolve.

## Deploy order (not deployed)

Consumers first; no behavior change, because nothing issues a hold until 4.5:

1. `orion-llm-gateway`, `orion-cortex-exec`, `orion-thought`, `orion-harness-governor`, `orion-hub`,
   `orion-field-digester` — any order among them.
2. All of them before durable-runs 4.5 (which sends `CuriosityTurnRequestV1.gpu_lease`, extra="forbid").

`orion-thought` and `orion-harness-governor` are not in the spec's 4.4 deploy list but carry the
ref (stance, harness admission): they must be deployed too. Their request models have no
extra="forbid", so an un-upgraded one silently *drops* the ref and its call takes a plain lease
behind the run's own hold — nothing fails loudly. The 4.5 runbook should check, before step 5,
that each of the six containers has this code (e.g. `docker exec <c> grep -c gpu_lease <file>`).

## Restart required

```text
No restart required for this PR on its own (not deployed). When deployed: rebuild + restart the six
services above via scripts/safe_docker_build.sh <service> up -d --build from a worktree.
```

## Assumptions about 4.3's pool API

- `attach` answers like `acquire`: `granted` (child `lease_id` + `grant` on the hold's role),
  `queued` (then a `granted` event with `detail.grant` on `orion:gpu_pool:event`), or a refusal
  (`unknown_lease` / `unavailable` + reason). The child is then a normal request lease: heartbeat,
  recall, release by the child's `lease_id`.
- `attach` is idempotent on `request_id`: when an attach reply is *lost* (RPC timeout / cancel),
  the client's withdraw path re-sends the same attach to learn the child `lease_id`, then cancels
  it. A reply without a `lease_id` is taken to mean nothing was created.
- `status` answers `granted` or `recall` with `grant.generation` equal to the hold's current
  generation for a live hold; anything else is treated as not held.
- The child's `work_class` is whatever the call's route maps to (`agent` by default under a hold);
  placement follows the hold's role, not the class. The child's `holder` is the gateway's
  (`llm-gateway` / `http:anthropic`); the run link is the pool's `hold_lease_id` column.
- `min_ctx_tokens` is sent on attach; a `min_ctx_exceeds_class` refusal still triggers the existing
  one clamp retry.

## Spec problems found

- **A hold's role is not a route.** Spec says durable-runs emits `run.lane_assigned` with
  lane = role. Hub used `llamacpp/<lane>` as FCC's model and cortex-exec/thought used the lane as
  the route; for `agent-gpu2` that is a 404. Fixed here with `GPU_LEASE_ROUTE` ("agent") for held
  calls; 4.5 must not feed a role into `assigned_lane`/`fcc_model_label`.
- **The field row assumes no overlap.** "Before cutover the second half is 0; after migration the
  first half is 0" misses the window between 4.5 runbook steps 5 and 6, where a resumed run has a
  frozen pending demand *and* a hold. Handled: such a run counts once, as its hold.
- **Acceptance check 2 still names `parent_lease_id IS NULL`.** Correction 2 moved "child of hold"
  to `hold_lease_id`; the check should read `hold_lease_id IS NULL`.
- **Door-A validation.** Hub's only lease validation today is the per-turn check in
  `_turn_result_for`; Door-A composition does not validate at Hub (the gateway fences it). The ref
  is validated with `status` at the turn check; Door-A relies on the gateway's attach, same as today.
- **4.4 deploy list** omits `orion-thought` and `orion-harness-governor` (see Deploy order).

## Risks / concerns

- Severity: low. Concern: the pool side of attach/status (4.3) is being built in parallel; a
  mismatch with the assumptions above would show up only when 4.5 issues holds. Mitigation: the
  assumptions are listed; the client/gateway tests pin the wire request shape.
- Severity: low. Concern: a hold's oldest-wait clock (`queued_since`) differs slightly from a
  demand's (`created_at`). Mitigation: re-fit the anchor after a week of hold data.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
