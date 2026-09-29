# GPU pool stage 4.6 — the old durable lease path is gone; the pool hold is the only run lease

## Summary

- **One run lease.** The old durable-runs lease token is deleted everywhere it was carried or
  checked: `ResourceLeaseV1`, the `X-Orion-Resource-Lease` header, bus `options.resource_lease`,
  the gateway's `LeaseGuard` and its `LLM_GATEWAY_LEASE_VALIDATION_*` broker check. That covers
  the FCC motor, harness runner/finalize, thought, cortex-exec, harness-governor, Hub's turn
  orchestrator and curiosity loop, and durable-runs' runner. A GPU pool hold ref (`GpuLeaseRefV1`:
  `X-Orion-Gpu-Lease`, `options.gpu_lease`) is the only way a call runs under a run's reservation.
- **gpu2 has one boss.** circe's gpu-lane-controller loses its `durable` authority branch:
  `GPU2_AUTHORITY`, `GPU2_AUTHORITY_URL`, the `/elastic/status` callback and
  `POST /v1/gpu-slots/activate` are gone. Pool actuation over the bus is unconditional.
- **The durable-waiting signal reads one table.** field-digester's `durable_demand_pending` counts
  only queued/backlogged durable-runs pool holds; the frozen `durable_resource_demands` half is
  dropped. The oldest-wait reading (#2337) is kept.
- **Broker-only admission fields deleted.** `ResourceRequirementV1` loses `allow_elastic_activation`,
  `alternatives`, `pinned_lane`, `operator_override`; producers (Hub, cortex-exec self-study) stop
  sending them. `HUB_CURIOSITY_ELASTIC_ACTIVATION_ENABLED` is gone.
- **Door-A has its own URL key.** `HUB_CURIOSITY_LEASE_VALIDATION_URL` (Hub derived the Door-A
  release URL from it) is replaced by `HUB_CURIOSITY_DURABLE_RUNS_URL`, same default base.

## Outcome moved

Before this, two lease mechanisms coexisted: every admitted call could carry both the dead broker
token and the pool ref, the gateway still had code (and a live `.env` flag set to `true`) to
validate the token against a `/leases/validate` endpoint that 4.5 already deleted, and the controller
still had a switch that could put gpu2 back under a decider that no longer exists. Now there is one
path, so nothing can quietly fall back to the dead one. A real bug surfaced while deleting: the
durable-runs runner decided "admitted" (long RPC timeout) from `request.lease or request.gpu_lease`;
it now reads the hold only, and removing the field made the stale check fail loudly in tests
instead of silently.

## Current architecture (before this PR)

- Stage 4.5 (live since 2026-09-26) moved durable runs onto pool holds, but left the old token in
  every schema, every forwarder and the gateway guard, "accepted and ignored until 4.6".
- The controller ran `GPU2_AUTHORITY=pool` on circe, with the `durable` branch still compiled in.
- `durable_demand_pending` = union of pending `durable_resource_demands` (frozen, 0 since cutover)
  and queued/backlogged durable-runs holds.

## Architecture touched

- Contracts: `orion/schemas/{resource_admission,durable_run,thought,harness_finalize}.py`,
  `orion/llm/resource_lease.py` (now only the pool-ref wire helpers).
- Services: llm-gateway, gpu-lane-controller (circe), field-digester, Hub, thought, cortex-exec,
  harness-governor, durable-runs (runner one-liner + tests/README only).
- Not touched on purpose: `orion/durable_admission/{capacity,capacity_client,store}.py`, `/capacity`,
  the `durable_resource_*` tables (world-model/diffusion permits, stage 5); `pool_hold.py` / admitted
  graphs (another PR); the harness-governor `fcc:<...>` hop key.

## Files changed

- `orion/schemas/resource_admission.py`: `ResourceLeaseV1` and the four broker fields deleted;
  `CapacityAcquireV1.lease` is `None`-only (capacity.py reads it until stage 5; no caller sends one).
- `orion/schemas/{durable_run,thought,harness_finalize}.py`: `lease` / `resource_lease` fields deleted.
- `orion/llm/resource_lease.py`: `LEASE_HEADER`, `encode/decode_lease_header`, `validate_resource_lease` deleted.
- `orion/harness/{fcc_motor,runner,finalize}.py`, `orion/hub/turn_orchestrator.py`: token plumbing deleted;
  owner lane = hold route (`agent`) when held.
- `services/orion-llm-gateway/app/{resource_lease,main,passthrough_proxy,openai_passthrough,anthropic_passthrough,llm_backend,settings}.py`,
  compose, `.env_example`, README: `LeaseGuard` and settings deleted.
- `services/orion-gpu-lane-controller/app/{gpu2,main,actuator_bus,pool_fence,settings}.py`, compose,
  `.env_example`, README: durable authority, activate route, `GPU2_AUTHORITY*` deleted.
- `services/orion-field-digester/app/store.py`, README, `orion/field/queue_contention.py`: pool-holds-only SQL.
- `services/orion-hub/{app/settings.py,scripts/main.py,scripts/curiosity_investigation.py,scripts/thought_client.py}`,
  compose, `.env_example`, README: token path + two keys deleted, `HUB_CURIOSITY_DURABLE_RUNS_URL` added.
- `services/orion-thought/app/bus_listener.py`, `services/orion-cortex-exec/app/{executor,self_study}.py`,
  `services/orion-harness-governor/app/bus_listener.py` (+ READMEs/.env_example comments).
- `services/orion-durable-runs/app/runner.py`: admitted = hold present.
- `orion/world_pulse_read/durable.py` (+ test): stored pre-4.6 bindings are read without the deleted keys.
- `docs/architecture/durable-resource-admission.md`: superseded note on the old settings table.
- `scripts/sync_local_env_from_example.py`: dead `LLM_GATEWAY_LEASE_`/`LLM_GATEWAY_CAPACITY_` prefixes removed.
- `services/orion-world-model/app/settings.py`, `services/orion-gpu-pool/.env_example`: stale comments.
- `docs/runbooks/2026-09-25-gpu-pool-stage4-cutover.md`: 4.6 done; old-image rollback no longer applies.
- `.github/workflows/orion-gpu-pool-tests.yml`: renamed test file, path triggers.
- Tests deleted with their code: `orion/harness/tests/test_resource_lease_transport.py`,
  `services/orion-llm-gateway/tests/test_resource_lease.py`. Renamed/rewritten:
  `test_finalize_resource_lease.py` -> `test_finalize_chain_owner_lane.py`,
  cortex-exec `test_resource_lease_forwarding.py` -> `test_gpu_lease_forwarding.py`; rewritten
  thought, governor, Hub, controller, field-digester and durable-runs tests. New regression tests pin
  the absence (`test_the_hold_ref_is_the_only_run_lease_after_4_6`, gateway/thought/cortex-exec
  "stray legacy token is ignored" tests, controller "switch and activate route are gone").

## Schema / bus / API changes

- Removed: `ResourceLeaseV1` (never in either schema registry; `resolve("ResourceLeaseV1")` raises),
  `CuriosityTurnRequestV1.lease`, `StanceReactRequestV1.resource_lease`,
  `HarnessRunRequestV1.resource_lease`, `ResourceRequirementV1.{allow_elastic_activation,
  alternatives, pinned_lane, operator_override}`; header `X-Orion-Resource-Lease`; bus
  `options.resource_lease`; controller `POST /v1/gpu-slots/activate`.
- Behavior changed: `CapacityAcquireV1.lease` accepts only null. A call carrying a stray
  `resource_lease` is no longer validated or forwarded; it runs as if it carried nothing.
- Compatibility: `ResourceRequirementV1` is `extra="forbid"` — an old producer still sending the
  broker fields is refused by an upgraded durable-runs/cortex-orch, so producers deploy first (below).
  `CuriosityTurnRequestV1` is also forbid, but durable-runs dumps it with `exclude_none=True` and
  never sent `lease`, so Hub is order-free. Stance/harness requests are `extra="ignore"`.
  Rows stored before 4.6 still compare cleanly on duplicate receipt (`store.py`'s
  `IGNORED_ADMISSION_FIELDS`, untouched), and the world-pulse reader strips the same keys from its
  own stored bindings before validating (review blocker, fixed).

## Env/config changes

- Added: `HUB_CURIOSITY_DURABLE_RUNS_URL=http://127.0.0.1:8124` (Hub).
- Removed: `LLM_GATEWAY_LEASE_VALIDATION_ENABLED`, `LLM_GATEWAY_LEASE_VALIDATION_URL`,
  `LLM_GATEWAY_LEASE_VALIDATION_TIMEOUT_SEC`, `LLM_GATEWAY_LEASE_CHECK_INTERVAL_SEC` (gateway);
  `HUB_CURIOSITY_ELASTIC_ACTIVATION_ENABLED`, `HUB_CURIOSITY_LEASE_VALIDATION_URL` (Hub);
  `GPU2_AUTHORITY`, `GPU2_AUTHORITY_URL` (gpu-lane-controller).
- `.env_example` updated: yes. Local `.env` synced with `python scripts/sync_local_env_from_example.py`:
  yes — `+HUB_CURIOSITY_DURABLE_RUNS_URL` landed in athena's Hub `.env`.
- Keys the sync does not remove, still in live `.env` files (harmless — every settings model is
  `extra="ignore"` — but dead; drop by hand):
  - athena `services/orion-llm-gateway/.env`: lines 136-139 `LLM_GATEWAY_LEASE_VALIDATION_ENABLED`,
    `LLM_GATEWAY_LEASE_VALIDATION_URL`, `LLM_GATEWAY_LEASE_VALIDATION_TIMEOUT_SEC`,
    `LLM_GATEWAY_LEASE_CHECK_INTERVAL_SEC`; lines 142-143 `LLM_GATEWAY_CAPACITY_ENABLED`,
    `LLM_GATEWAY_CAPACITY_URL` (already dead before this PR).
  - athena `services/orion-hub/.env`: line 702 `HUB_CURIOSITY_LEASE_VALIDATION_URL`, line 705
    `HUB_CURIOSITY_ELASTIC_ACTIVATION_ENABLED`.
  - athena `services/orion-gpu-lane-controller/.env`: line 37 `GPU2_AUTHORITY_URL`.
  - circe `services/orion-gpu-lane-controller/.env` (not readable from here): `GPU2_AUTHORITY`,
    `GPU2_AUTHORITY_URL`.
- Skipped keys requiring operator action: none.

## Metric gate: `durable_demand_pending` (source change)

1. Provenance: `FieldDigesterStore.count_durable_demand_pending()` /
   `oldest_durable_demand_pending_age_sec()` (`services/orion-field-digester/app/store.py`), both over
   `DURABLE_WAITING_SQL` = `gpu_pool_leases WHERE kind='hold' AND holder LIKE 'durable-runs:%' AND
   status IN ('queued','backlogged')`, age from `coalesce(queued_since, created_at)`; called from
   `app/digestion/queue_contention.py:77,96`.
2. Independence: unchanged — `gpu_pool_waiting` still excludes durable holds (`NOT_DURABLE_HOLD_SQL`),
   so no hold is counted twice.
3. Anchor: unchanged (a durable run waiting for its GPU is queue contention; same source key/meaning).
4. Live data (read-only, 2026-09-29): `durable_resource_demands` = 158 `withdrawn`, 0 pending, newest
   row 2026-09-26 00:17 UTC (before the first hold at 06:03) — the dropped half was structurally 0.
   Pool-only value today = 0 = old union, so the change moves no number. Not stuck: 176 durable-runs
   holds since cutover, 108 waited >10 s, median wait 354 s, max 31,664 s, and it is back at a real 0.
   The digester's 72 published values since its 03:55 UTC restart are all 0.0 (logs don't reach the
   nonzero periods; those come from the lease rows).
5. Existing mechanism: this is the existing source, narrowed.
6. Reversibility: one SQL string.

## Tests run

```text
services/orion-llm-gateway: pytest tests -q                           -> 304 passed
orion/harness/tests -q                                                -> 367 passed
orion/gpu_pool/tests -q                                               -> 236 passed
services/orion-harness-governor: pytest tests -q                      -> 55 passed
services/orion-thought: pytest tests -q                               -> 492 passed, 28 skipped, 1 failed
  (test_settings_mind_enrichment::test_mind_enrichment_defaults_off; fails identically on main, env-sensitive)
services/orion-cortex-exec: per-file (whole-dir collection collides on verb registration, same on main)
  -> identical failing set to origin/main (6 files: chat_kids_story/chat_quick plumbing,
     cognition_trace_verb_runtime, main_autonomy_graph_probe, situation_prompt_integration,
     story_weave_smoke); test_gpu_lease_forwarding.py 11 passed
services/orion-hub: pytest tests -q -> 38 failed, 3004 passed, 73 skipped
  (the same 38 fail on origin/main: UI smoke, memory-graph routes, etc.); touched files 236 passed
  after merging main: gateway+harness+gpu_pool+world_pulse_read+controller 1085 passed; durable-runs 240 passed; Hub 38 failed / 3118 passed (same 38)
services/orion-durable-runs (throwaway postgres:16, ORION_ADMISSION_TEST_DSN): pytest tests -q -> 213 passed
services/orion-gpu-lane-controller: pytest tests -q                   -> 77 passed
services/orion-field-digester (throwaway postgres:16, GPU_POOL_TEST_POSTGRES_URI): pytest tests -q -> 263 passed
  mutation: re-adding the legacy UNION fails 3 tests
static gates (orion-static-gates.yml): env sync/parity pytest, schema registry agree, substrate ladder,
  grammar producer catalog, field topology, substrate requests, check_metric_lineage --gate,
  check_definition_drift --gate, inner_state_registry, scripts stdlib shadow, hostname refs, compose
  mounts, claude.json mounts, journal registry, sentience instruments --static-only, system_health
  producers, control surface parity, async routes, chat route poachers, check_gpu_pool_config,
  check_env_template_parity -> all PASS
```

## Evals run

```text
services/orion-durable-runs/evals/hold_fairness.py     -> PASS (failed_checks: [])
services/orion-durable-runs/evals/deploy_order_skew.py -> PASS (failed_checks: [])
services/orion-durable-runs/evals/gateway_capacity.py  -> verdict PASS
```

## Docker/build/smoke checks

```text
Not built or deployed (task: do not deploy).
```

## Review findings fixed

Code-review subagent (read-only) on the branch: 1 blocker, 3 should, 5 nit. It confirmed no live
reference to any removed name remains, the gateway/controller/governor/thought/cortex-exec paths
behave as before for every live caller, and deleted tests are covered elsewhere.

- Finding (blocker): the world-pulse reading loop reads its stored `reading_durable_turn.request_json`
  back through `DurableRunRequestV1.model_validate`. Every row written before 4.6 carries the deleted
  broker fields, so the forbid model would refuse it and the seed would sit in
  `reading_binding_unavailable` forever, with no loud failure. Live: 39 of 72 rows unconsumed, all
  carrying the keys; 4 belong to live seeds. Deploy order cannot fix it (Hub is both the reader and
  a first-wave producer).
  - Fix: `orion/world_pulse_read/durable.py` `_stored_request` strips the four keys from a stored
    row before validating, at both read sites. The stored prompt/run id stay authoritative; the
    resubmit matches durable-runs' stored row because `store.py` ignores the same keys.
  - Evidence: `orion/world_pulse_read/tests/test_durable.py::test_a_binding_stored_before_4_6_still_binds`
    (text + jsonb forms); mutation (no strip) fails both.
- Finding (should): deploy order must put Hub and cortex-exec before cortex-orch and durable-runs.
  - Fix: already the order below; the blocker fix removes the one hazard it could not cover.
- Finding (should): dead keys in live `.env` files, including `LLM_GATEWAY_CAPACITY_*`.
  - Fix: exact lines listed under Env/config changes.
- Finding (should): `docs/architecture/durable-resource-admission.md` still documented the Hub
  validator and gateway lease settings.
  - Fix: "Superseded" note above the settings table.
- Nits fixed: sync-script comment (capacity keys were already dead), `resolve("ResourceLeaseV1")`
  assertion labelled as a re-registration guard, Hub `.env_example` comment split per key.
- Nits not changed: durable-runs README does list `alternatives` (the reviewer read the other
  paragraph); `ResourceLeaseRejected` / `resource_lease_rejected` stay because the pool-ref path and
  cortex-exec still use that error name; the `orion-resource-lease` placeholder token literal is
  cosmetic.

## Restart required

Deploy order (producers of the removed `ResourceRequirementV1` fields before the models that refuse them):

```bash
cd <worktree at merged main>
python3 scripts/sync_local_env_from_example.py
# 1. producers stop sending the broker fields (Hub also: new Door-A key, token path gone)
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
# 2. the pass-through validator, then the consumer
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
# 3. order-free (nobody sends the old token any more)
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-field-digester up -d --build
# 4. circe, any time (live durable-runs 4.5 no longer calls the activate route):
#    drop GPU2_AUTHORITY and GPU2_AUTHORITY_URL from circe's services/orion-gpu-lane-controller/.env, then
scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build   # on circe
```

Service names above are the directory names the wrapper takes; check each compose file's service
name if a build argument is required. world-model changed a comment only: no rebuild.

## Risks / concerns

- Severity: medium.
  - Concern: a `DurableRunRequestV1` still in flight from an un-rebuilt Hub/cortex-exec carrying
    `allow_elastic_activation` is refused by a rebuilt cortex-orch/durable-runs.
  - Mitigation: deploy order above; producers are resubmit-safe (duplicate receipts compare clean).
- Severity: low.
  - Concern: `CapacityAcquireV1.lease` survives as a None-only field because capacity.py (frozen) reads it.
  - Mitigation: stage 5 deletes capacity.py and the field together.
- Severity: low.
  - Concern: `pool_hold.is_hold_ref` still tolerates a pre-cutover `ResourceLeaseV1`-shaped dict in an
    old checkpoint; left alone because another PR is editing `pool_hold.py`.
  - Mitigation: data-shape guard only, no type import; delete with that PR or stage 5.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2401

🤖 Generated with [Claude Code](https://claude.com/claude-code)
