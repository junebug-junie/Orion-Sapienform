# Durable Curiosity admission activation

## Result

The checked-in operator env templates now select the resource-admitted Curiosity
path implemented and accepted in PRs #2205, #2209 and #2210. The host-local env
files were updated to the same values without changing unrelated settings.

Enabled gates:

- `HUB_CURIOSITY_KICKOFF_VIA_CORTEX=true` (already enabled)
- `HUB_CURIOSITY_DURABLE_ADMISSION_ENABLED=true`
- `CORTEX_DURABLE_ADMISSION_ENABLED=true`
- `DURABLE_RUNS_ENABLED=true` (already enabled)
- `DURABLE_RUNS_ADMISSION_ENABLED=true`
- `DURABLE_RUNS_CAPACITY_ENABLED=true`
- `DURABLE_RUNS_WIDENING_ENABLED=true`
- `LLM_GATEWAY_CAPACITY_ENABLED=true`
- `LLM_GATEWAY_LEASE_VALIDATION_ENABLED=true`

`DURABLE_RUNS_ADMISSION_SHADOW=false` remains false intentionally: true computes
decisions without acquiring a lease or executing the graph, so it would disable
the requested live path. Code and compose fallbacks remain false when an env
contract is absent.

## Prerequisite found

A read-only query against the live `conjourney` database on 2026-09-14 found
`durable_admission_runs`, `durable_resource_demands`,
`durable_resource_leases`, and `durable_gateway_permits` all missing. The runner
is designed to fail startup when enabled without those contracts. Apply, in
order:

1. `services/orion-sql-db/manual_migration_durable_resource_admission_v1.sql`
2. `services/orion-sql-db/manual_migration_gateway_capacity_v1.sql`

No migration, container restart, or production cognition run is performed in
this patch. The local env is staged for activation, so do not restart the runner
or Gateway before applying both migrations.

## Compatibility and lane behavior

Ordinary Cortex requests retain their synchronous path. Missing lease tokens
remain valid for existing synchronous Gateway traffic, while participating
ordinary requests acquire shared capacity permits. Curiosity admission remains
the only selected durable workflow.

Widening is enabled at 1200 seconds, but `DURABLE_RUNS_LANE_POLICY_JSON={}`
approves no compatible alternative lane. This is deliberate: an alternative
must be backed by an explicit capability and compatibility audit. The preferred
agent lane therefore remains the only eligible lane until that separate policy
change lands.

## Activation and rollback

After both migrations, restart in this order: durable-runs authority, every LLM
Gateway replica, Thought, governor, Cortex Exec, Cortex Orch, then Hub. Thought
must propagate the owning lease before Hub submits an admitted study. Verify
`/health`, `/admission`, `/capacity`,
one committed receipt, one resource grant, the fenced Gateway request, nonempty
attention/journal artifacts and terminal release.

To roll back, stop new Hub submissions first, drain or pause admitted work while
the authority is available, then set Hub/Cortex admission and Gateway enforcement
false before disabling the runner authority. Retain the additive tables.

## Validation

- Env-template parity for Hub, Cortex Orch, durable-runs and LLM Gateway.
- Service hostname references and compose rendering for all four services.
- Durable-run tests: 29 passed, 33 Postgres-dependent tests skipped locally.
- Gateway tests: 345 passed.
- Env sync/parity regression tests: 30 passed.
- Independent code/config review before publication.
