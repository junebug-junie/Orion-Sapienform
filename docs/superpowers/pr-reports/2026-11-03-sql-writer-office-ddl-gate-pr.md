## Summary

- Prevent SQL writer from subscribing to `office.env.sample.v1` unless the three CCS811 columns
  exist in `office_env_sample`.
- Move the required office DDL out of the long warning-only bootstrap transaction and verify the
  resulting catalog shape in the same isolated transaction.
- Make a lock timeout fail startup instead of silently diverting every office sample to
  `bus_fallback_log`.

## Incident evidence

Athena's 2026-11-03 deploy hit PostgreSQL `LockNotAvailable` on
`office_env_sample.office_eco2_ppm`. The shared bootstrap transaction rolled back, logged only a
warning, and started Hunter. From 04:06:06Z through 06:17:41Z, 264 office samples failed with
`UndefinedColumn` and landed in fallback.

Live recovery completed before this patch:

- Added `office_eco2_ppm`, `office_tvoc_ppb`, and `air_warming` in one bounded transaction.
- Verified new live samples reached `office_env_sample` with eCO2/TVOC values.
- Revalidated and replayed all 264 preserved payloads through the production schema/normalizer in
  one transaction, deleting their fallback rows only after insert success.
- Verified 264 distinct recovered timestamps, zero destination overlaps before replay, and zero
  fallback rows in the 24-hour window afterward.

## Current architecture

`orion-office-env` → `orion:office:env:sample` → `orion-sql-writer` →
`office_env_sample`. `Base.metadata.create_all()` creates a fresh table but cannot add columns to an
existing one, so boot DDL owns additive upgrades.

## Files changed

- `services/orion-sql-writer/app/main.py`: isolated, fail-closed office schema gate.
- `services/orion-sql-writer/tests/test_office_env_sample_sql_shape.py`: success and lock-timeout
  coverage.
- `services/orion-sql-writer/README.md`: startup contract.

## Schema / bus / env changes

None. This patch changes enforcement of the already-shipped table shape. `.env_example` is
unchanged, so no local env sync is required.

## Tests and evals

```text
pytest -q services/orion-sql-writer/tests/test_boot_ddl_guarded.py
  3 passed

Athena throwaway container, production SQL-writer image with patched files mounted read-only:
pytest -q services/orion-sql-writer/tests/test_office_env_sample_sql_shape.py
  11 passed

Runtime failure-mode smoke in the same image:
  missing columns -> all three established and verified
  simulated lock timeout -> propagated; startup gate did not swallow it
  current exact shape -> no ALTER issued
  wrong type/nullability -> startup rejected
```

No latent/model eval applies: this is deterministic schema enforcement. The live incident replay
is the end-to-end persistence eval.

## Acceptance checks

- Existing current schema: guarded DDL skips all three statements and startup continues.
- Missing column: additive DDL runs and exact type/nullability verification passes before subscription.
- Current schema: all DDL is skipped without taking a table lock.
- Wrong type or nullability: startup fails before subscription.
- Lock timeout or missing table: startup fails; no office events are consumed into fallback.
- No channel, schema model, payload meaning, env key, or retention behavior changes.

## Risk and rollback

Low. A database lock can now keep SQL writer down instead of allowing partially functional startup;
that is intentional because serving the affected route loses its destination writes. Roll back the
commit to restore warning-only behavior. The live additive columns and recovered rows require no
rollback.

## Deployment

After merge, deploy from clean synchronized primary `main` only:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

Expected startup evidence:

```text
office_env_sample required columns ensured; boot DDL ran=0 skipped=3
```
