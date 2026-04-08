# Unified Cognitive Substrate — Phase 18: Durable Policy Store and Policy-Backed Cache/Runtime Wiring

## Why this phase exists

Phase 17 established manual policy lifecycle semantics (stage/activate/rollback) but remained intentionally in-memory and left some modeled knobs only partially wired (`query_cache_enabled` in particular).

Phase 18 hardens the policy control plane for operational durability and behavior fidelity.

## What Phase 18 adds

### 1) Durable policy store behavior

`SubstratePolicyProfileStore` now supports optional persistence via a durable JSON store path.

- profiles, active-scope mappings, and audit history are persisted,
- restart-style reconstruction restores active/staged/rolled-back state,
- baseline remains valid when no active profile matches scope,
- lifecycle semantics remain explicit and manual.

### 2) Runtime policy resolution remains deterministic

Scope matching remains deterministic over:

- invocation surface,
- target zone,
- operator mode.

No staged profile is implicitly activated during reload.

### 3) `query_cache_enabled` is now real behavior

Phase 16 query coordinator now accepts `cache_enabled`.

- `cache_enabled=true`: bounded reuse path is active.
- `cache_enabled=false`: reuse is bypassed for that execution path.

Execution metadata explicitly records cache posture to preserve source honesty.

### 4) Policy-backed runtime/query integration

Runtime policy resolution now wires safe knobs into live behavior:

- `query_limit_nodes` / `query_limit_edges` into consolidation semantic-region bounds,
- `query_cache_enabled` into consolidation query coordinator behavior,
- `frontier_followup_allowed` gating in runtime.

Scheduler policy overrides remain scope-bound and deterministic.

## Safety posture

- Manual control model remains unchanged (no auto-adoption).
- Strict-zone protections remain intact.
- Baseline/no-profile remains a safe mode.
- Rollback remains explicit and auditable.

## Follow-on work

- optional SQL-backed policy durability for multi-instance coordination,
- richer operator diff tooling over profile transitions,
- bounded migration tooling for policy history lifecycle management.
