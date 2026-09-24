# PR: System One substrate appraisal shadow reducer

Branch: `feat/system-one-substrate-appraisal` → `main`

## Summary

Adds a Kev / TypeSafe-System-One-compatible **shadow reducer** inside
`orion-substrate-runtime`.

It consumes the existing attention broadcast plus a fresh FieldAttention frame, sends a bounded
provider-neutral state to `/v1/systemone`, persists the typed result as
`SystemOneAppraisalFrameV1`, and emits a normal `GrammarProjectionV1` causal trace.

No output drives behavior in this PR.

## Why this seam

This follows the existing substrate doctrine:

```text
event -> schema -> trace -> reducer -> projection -> eval -> UI/debug
```

It does not create a parallel StateFacet/SelfState graph. Kev is an inference operator over
existing substrate artifacts.

## Files

### New

- `orion/schemas/system_one_appraisal.py`
- `orion/substrate/system_one_appraisal.py`
- `services/orion-sql-db/manual_migration_system_one_appraisal_v1.sql`
- `tests/test_system_one_appraisal.py`
- `services/orion-substrate-runtime/tests/test_worker_system_one_appraisal.py`
- `services/orion-substrate-runtime/tests/test_store_system_one_appraisal.py`
- `services/orion-substrate-runtime/tests/test_system_one_appraisal_endpoint.py`
- `scripts/smoke_system_one_appraisal.py`
- `scripts/analysis/eval_system_one_appraisal.py`
- `docs/superpowers/specs/2026-09-23-system-one-substrate-appraisal-shadow-design.md`

### Updated

- substrate runtime worker/settings/store/main
- substrate runtime `.env_example` / `docker-compose.yml`
- schema registry
- substrate-runtime README

## Initial shadow questions

- reverie_fit
- curiosity_pull
- deliberation_need
- attention_interrupt

All are three-level `score` questions. The complete probability distribution is persisted.
There is no hidden conversion into a 0..1 "propensity."

## Safety / behavior

- default off;
- fail-open on endpoint/model failure;
- stale field frame omitted;
- malformed or partial provider result rejected entirely;
- isolated append-only table;
- no StateDelta;
- no FieldState write;
- no behavioral consumer.

## Data boundary

No raw conversation body or raw graph snapshot is intentionally sent. The bounded state can still
contain derived attention descriptions originating from private conversation, so a remote provider
would cross the privacy boundary. Default config is off and the intended first deployment is local
Kev.

## Verification

Focused unit tests are included for:

- provider wire shape and probability preservation;
- URL normalization;
- malformed/partial response rejection;
- grammar projection emission;
- worker fail-open behavior;
- stale field-frame omission;
- store insert/load;
- debug endpoint.

A read-only smoke and live-distribution evaluator are included.

**Live deployment proof: UNVERIFIED in this PR authoring environment.** Migration, local `.env`
sync, container restart, real Kev call, persisted row, grammar-event observation, and 24h
distribution report must be performed on Athena after merge/deploy.

## Deploy

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_system_one_appraisal_v1.sql
python scripts/sync_local_env_from_example.py orion-substrate-runtime
# set SUBSTRATE_SYSTEM_ONE_BASE_URL and SUBSTRATE_SYSTEM_ONE_APPRAISAL_ENABLED=true
# rebuild/restart orion-substrate-runtime
python scripts/smoke_system_one_appraisal.py
python scripts/analysis/eval_system_one_appraisal.py --hours 24
```

No downstream promotion belongs in this PR.
