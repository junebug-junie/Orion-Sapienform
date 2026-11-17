Fixes #3567

## Summary

- The mesh-edge check stops logging in as the database superuser. It now logs in as `orion_readonly`. This is the check that stops Orion quoting core pages and lived memories on GitHub. Orion's command line inherits this setting, so a superuser login was a wider key than the check needs.
- A new hand-applied GRANT file, `scripts/sql/2026-11-17_grant_orion_readonly_mesh_edge.sql`, gives `orion_readonly` read access to the tables the check reads: `episode_memory`, `person_fact`, `person`, `self_concept_history`. It grants no write access.
- `memory_crystallizations` and `episode_memory_referent` are deliberately NOT granted. The approved command included both. See "Review findings fixed" for why.
- Fail-closed still holds. If the login cannot read a table, every GitHub write is refused. New tests prove it, and a live run on the current database shows it.
- The DSN key is protected from `sync_local_env_from_example.py --force`. The template carries a `CHANGE_ME` password, and a forced sync must not overwrite the real one.

## Outcome moved

The mesh-edge DSN that Orion's FCC turns inherit no longer carries superuser rights. Its read scope is now limited to what the guard needs.

## Current architecture

- `orion/memory/mesh_edge.py` `load_guard()` reads the guarded sentences through `resolve_dsn()`. The first key it tries is `ORION_MESH_EDGE_PG_DSN`. If loading fails for any reason, it raises `MeshEdgeUnavailable`.
- `orion/fcc/mesh_edge_hook.py` `decide()` turns `MeshEdgeUnavailable` into a block on any GitHub write.
- The governor `.env_example` and the compose default set the DSN to `postgres:postgres`.

## Architecture touched

- `orion-harness-governor`: config only (`.env_example`, `docker-compose.yml` default).
- Postgres role `orion_readonly`: the GRANT is applied by hand.
- The env sync script: one key added to `NEVER_SYNC_KEYS`.

## Files changed

- `services/orion-harness-governor/.env_example`: DSN is `orion_readonly` with a `CHANGE_ME` password.
- `services/orion-harness-governor/docker-compose.yml`: the default DSN is `orion_readonly` with no password, so an unset key cannot log in and fails closed.
- `scripts/sql/2026-11-17_grant_orion_readonly_mesh_edge.sql`: the GRANT, plus undo and verify steps.
- `scripts/sync_local_env_from_example.py`: `ORION_MESH_EDGE_PG_DSN` added to `NEVER_SYNC_KEYS`.
- `orion/memory/mesh_edge.py`: comment only. The guard's bare table name resolves to the `orion_scope` view for this role.
- `orion/fcc/tests/test_mesh_edge_hook.py`: fail-closed tests, a control case, and a config guard.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the mesh-edge guard reads as `orion_readonly`. AI Town stance crystallizations are no longer in the guard (107 of 719 stance rows are visible). #3597 asks Juniper whether that is fine.
- Compatibility notes: if the governor restarts before the GRANT is applied, every Orion GitHub write is refused. This is the safe direction.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- Changed meaning: `ORION_MESH_EDGE_PG_DSN` is now the `orion_readonly` login.
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: it ran. The key is now in `NEVER_SYNC_KEYS` because it holds a password, so the sync skips it. I set the live value in the primary checkout's `services/orion-harness-governor/.env` by hand. The password was copied from `ORION_CURIOSITY_PG_DSN` in `/root/.fcc/.env` and never printed.
- Skipped keys requiring operator action: `ORION_MESH_EDGE_PG_DSN` on any other host.

## Tests run

```text
pytest orion/fcc/tests/test_mesh_edge_hook.py orion/memory/tests/test_mesh_edge.py -q   -> 59 passed
python3 scripts/check_env_template_parity.py                                           -> PASS
python3 scripts/check_migration_required_by.py                                         -> OK
git diff --check                                                                       -> clean
```

## Evals run

```text
None. This is a credential/config change. The eval-equivalent is the live check below.
```

## Docker/build/smoke checks

```text
Before the GRANT, live in orion-athena-harness-governor, load_guard() with the orion_readonly DSN:
  UNAVAILABLE mesh-edge guard load failed: InsufficientPrivilege (permission denied for table episode_memory)
  -> fail closed confirmed on the real rail.
Before the GRANT, privileges for orion_readonly: self_concept_history SELECT=t; episode_memory, person_fact, person SELECT=f; INSERT=f on all.

After the GRANT: UNVERIFIED (Juniper applies it). Check:
  docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "SELECT t, has_table_privilege('orion_readonly', t, 'SELECT'), has_table_privilege('orion_readonly', t, 'INSERT') FROM unnest(ARRAY['episode_memory','person_fact','person','self_concept_history','public.memory_crystallizations']) t;"
  expect: t|f for the first four rows, f|f for public.memory_crystallizations.
  After the governor restart, an Orion `gh issue comment` that quotes a core-page sentence is still blocked with "[orion-mesh-edge] ... quotes".
```

## Review findings fixed

- Finding (must): granting `public.memory_crystallizations` would undo `scope_orion_readonly_off_aitown.sql`, which deliberately took the raw AI Town rows away from the curiosity sandbox. The grant would not even help: the role's `search_path` resolves the guard's bare name to the filtered view.
  - Fix: removed it from the GRANT. Documented the effect in the SQL file and in `mesh_edge.py`. Filed #3597 (needs-juniper).
  - Evidence: live `rolconfig = search_path=orion_scope, public`. Stance rows: 719 raw, 107 through the view.
- Finding (should): the scope widening was not named, and `episode_memory_referent` had no reader.
  - Fix: added a privacy-boundary paragraph to the SQL header. Dropped `episode_memory_referent`.
- Finding (should): the fail-closed test could pass with a broken fake driver.
  - Fix: added a control case and a check that the cause is the permission error. This exposed a real bug in the fake, which is now fixed.
  - Evidence: the control failed with `KeyError` before the fix and passes after it.
- Finding (should): nothing pinned the DSN fallback chain.
  - Fix: a test now asserts that no `postgresql://postgres:` string and no `DSN_ENV_KEYS[1:]` key appears in the governor `.env_example` or compose.
- Finding (nit): the `REQUIRED-BY` header in `scripts/sql/` is not enforced.
  - Kept as requested, as a record. The file says plainly that the gate cannot see a grant.

## Restart required

Apply the GRANT first. Then restart the governor from the clean primary main checkout:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "GRANT SELECT ON episode_memory, person_fact, person, self_concept_history TO orion_readonly;"
scripts/safe_docker_build.sh orion-harness-governor up -d
```

## Risks / concerns

- Severity: medium. Concern: if the governor restarts before the GRANT, Orion's GitHub writes are refused. Mitigation: apply the GRANT first. Fail closed is the safe direction.
- Severity: low. Concern: AI Town stances are unguarded at the mesh edge. Mitigation: #3597 asks Juniper.
- Issues opened: #3597. Issues closed: #3567.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3593

🤖 Generated with [Claude Code](https://claude.com/claude-code)
