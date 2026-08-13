# Untangle DriveAuditSQL's write-path wiring from orion-sql-writer

## Summary

- Fully removed `DriveAuditSQL`'s write-path wiring from `orion-sql-writer`: the model class (`app/models/drive_audit.py`, deleted), its `MODEL_MAP`/`INSERT_ONLY_MODELS`/route-map entries, its channel subscription (`orion:memory:drives:audit`) from `settings.py`'s default list and `.env_example`, and the `_apply_drive_audit_derivations()` helper + call site.
- Also removed the dead `DRIVE_AUDITS_RETENTION_DAYS` startup-prune job in `app/main.py` and its settings field — it targeted a table that no longer exists, guarded only by a try/except.
- Corrected a scope note left over from the earlier PR (#1614): that PR claimed `DriveAuditSQL` "shares a `_JSONB` type declaration with other live sql-writer models" as the reason to leave the write path wired. Checked directly — every other model file (`causal_geometry_snapshot.py`, `metacog_entry.py`, `thought_decision.py`, `repair_pressure_appraisal.py`, `harness_turn_trace.py`) declares its own private copy of that one-line type alias, not a shared import. Nothing was actually entangled.
- Updated `orion/bus/channels.yaml`'s `orion:memory:drives:audit` entry to `producer_services: []`, `consumer_services: []` with a full trail comment.
- Replaced the old shape-test file (`test_drive_audit_sql_shape.py`) with absence-guard tests asserting the model, its registrations, and the channel are actually gone.
- Added doc notes to the two remaining downstream readers (`scripts/drive_history_reflection_synthesis.py`, `scripts/analysis/measure_autonomy_gate.py`) clarifying their existing "missing table → degrade to empty/insufficient" behavior is now permanent, not transient.

## Outcome moved

`orion-sql-writer` no longer carries dead write-path code for a channel with zero producers and (as of this patch) zero consumers. No more risk of a stray message on `orion:memory:drives:audit` triggering an INSERT against a table that doesn't exist, and no more startup DELETE against that same nonexistent table.

## Current architecture

Before this patch: the Hub Drives Analytics tab and the `drive_audits` table were already removed (PR #1614, 2026-08-13, earlier same day) — table dropped, snapshotted first, boot DDL removed. But `orion-sql-writer`'s write-path wiring for `DriveAuditSQL` was left in place, deliberately scoped out under a since-corrected belief that it was entangled with other live models via a shared `_JSONB` type.

## Architecture touched

`services/orion-sql-writer` (models, worker, settings, env, README), `orion/bus/channels.yaml`, two standalone analysis/batch scripts (doc notes only, no behavior change — both already degrade safely).

## Files changed

- `services/orion-sql-writer/app/models/drive_audit.py`: deleted (the `DriveAuditSQL` model + its private `_JSONB` alias).
- `services/orion-sql-writer/app/models/__init__.py`: removed the `DriveAuditSQL` import and `__all__` entry.
- `services/orion-sql-writer/app/worker.py`: removed `DriveAuditSQL`/`DriveAuditV1` imports, the `INSERT_ONLY_MODELS` entry, the `MODEL_MAP` entry, `_apply_drive_audit_derivations()`, and its call site in `_write_row()`.
- `services/orion-sql-writer/app/settings.py`: removed `DEFAULT_ROUTE_MAP["memory.drives.audit.v1"]`, removed the channel from the default `sql_writer_subscribe_channels` list, removed the `drive_audits_retention_days` field.
- `services/orion-sql-writer/app/main.py`: removed the `DRIVE_AUDITS_RETENTION_DAYS` startup-prune block (dead: targeted a dropped table); corrected a stale comment that had prematurely claimed the worker.py wiring was "removed/updated in the same patch" as the boot-DDL removal (it wasn't, until now).
- `services/orion-sql-writer/.env_example`: removed `orion:memory:drives:audit` from `SQL_WRITER_SUBSCRIBE_CHANNELS`, removed `DRIVE_AUDITS_RETENTION_DAYS`.
- `services/orion-sql-writer/.env` (local, not committed): synced by hand to match — the sync script didn't flag either key as diverged, so this was a manual edit per the env-parity rule. Side-finding: local `.env` had `DRIVE_AUDITS_RETENTION_DAYS=90`, not the `0` the earlier PR's `.env_example` had intended — the prune job could have been live-pruning `drive_audits` all along in this environment. Moot now (table and job both gone), but flagged here since it wasn't caught before.
- `services/orion-sql-writer/README.md`: replaced the "table is frozen/historical" section with a "REMOVED 2026-08-13" trail note, removed the channel row from the channel table.
- `services/orion-sql-writer/tests/test_drive_audit_sql_shape.py`: replaced entirely with absence-guard tests (model file gone, not exported, not in `MODEL_MAP`/`INSERT_ONLY_MODELS`, not in route map, channel not in settings default or `.env_example`, no stray `DriveAuditSQL`/`DriveAuditV1` references in `worker.py`).
- `orion/bus/channels.yaml`: `orion:memory:drives:audit` entry's `consumer_services` changed from `["orion-sql-writer", "*"]` to `[]`; comment rewritten to document the full removal trail.
- `scripts/drive_history_reflection_synthesis.py`, `scripts/analysis/measure_autonomy_gate.py`: doc-only — added a STATUS note that the "missing table" degrade branch each already had is now permanent, not transient. No behavior change (both were already fail-open on a missing table, confirmed by reading `fetch_drive_history_events`/`fetch_drive_stats_postgres` before touching anything).

## Schema / bus / API changes

- Removed: `orion:memory:drives:audit` channel's sole remaining consumer (`orion-sql-writer`). Channel now has zero producers and zero consumers declared.
- Removed: `DriveAuditSQL` SQLAlchemy model, its `MODEL_MAP`/route-map registrations.
- Removed: `DRIVE_AUDITS_RETENTION_DAYS` config key and its startup job.
- `DriveAuditV1` (`orion/core/schemas/drives.py`) and its `orion/schemas/registry.py` registration were deliberately left in place — that's a schema/type definition, not live wiring, and other code still references the kind string (`orion/spark/concept_induction/dossier.py`'s `drive_audit_ref` lookup, already permanently `None` since concept_induction stopped publishing this kind).
- Compatibility notes: no live producer has existed since 2026-07-30 (PR #1486); no live consumer has existed since this patch. Removing this is safe by construction — confirmed via repo-wide grep for every reference to the channel, the model, and `DriveAuditV1` before touching anything.

## Env/config/changes

- Removed keys: `DRIVE_AUDITS_RETENTION_DAYS` (both `.env_example` and local `.env`).
- Removed values: `SQL_WRITER_SUBSCRIBE_CHANNELS`'s `orion:memory:drives:audit` entry, and — found only by the review pass, see below — `SQL_WRITER_ROUTE_MAP_JSON`'s `"memory.drives.audit.v1":"DriveAuditSQL"` entry (both `.env_example` and local `.env`).
- `.env_example` updated: yes — all four of the above.
- local `.env` synced: manually for all four (the sync script did not flag any of them as diverged, for reasons not further investigated here — verified and edited by hand instead of trusting the script's silence).
- Skipped keys requiring operator action: none.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-sql-writer/tests/ --ignore=tests/test_dream_model_constraints.py -q
18 failed, 204 passed, 3 errors  (204 after review-driven fixes; 202 before)
```
All 18 failures + 3 errors match the exact pre-existing failure set confirmed against an unmodified `main` checkout in the prior session (test_grammar_truth.py x7, test_journal_entry_payload_boundary.py x1, test_notify_attention_ack.py x3, test_notify_attention_escalate.py x2, test_phase21_wiring_verification.py x5, test_grammar_ledger_integration.py x3 errors) — zero new regressions.

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-sql-writer/tests/test_drive_audit_sql_shape.py -q
9 passed  (7 before review fixes, +2 route-map-JSON guard tests added in response to the review finding)
```

Also verified directly (not just via pytest): imported `app.worker`, `app.settings`, `app.models` in-process and confirmed `DriveAuditSQL` is absent from `MODEL_MAP`, the route map, the model namespace, and the default subscribe-channel list. `ast.parse` on every touched Python file. `yaml.safe_load` on `channels.yaml` (261 channels parse cleanly).

## Evals run

No dedicated eval harness exists for this seam — pure removal of dead wiring, no new behavior to eval.

## Docker/build/smoke checks

Not run — no local Docker daemon access in this session. The pytest suite already exercises `app/worker.py`'s and `app/settings.py`'s module-level construction (`MODEL_MAP`, `Settings`), which would fail loudly on a broken import or malformed dict; that's the meaningful smoke surface for this change.

## Review findings fixed

- Finding (MUST): `SQL_WRITER_ROUTE_MAP_JSON` in `.env_example` (and the deployed local `.env`) still contained `"memory.drives.audit.v1":"DriveAuditSQL"`. `Settings.route_map` does `{**DEFAULT_ROUTE_MAP, **json.loads(sql_writer_route_map_json)}` — the env override wins over the cleaned Python default, so the effective runtime route map still resolved `memory.drives.audit.v1` even after `DEFAULT_ROUTE_MAP` was cleaned. Degraded safely today (worker.py's `MODEL_MAP` guard would have fallen through to the unknown-kind fallback log, not crashed), but contradicted this same patch's own `channels.yaml` comment and PR-report claims.
  - Fix: removed the stale entry from `SQL_WRITER_ROUTE_MAP_JSON` in both `.env_example` and local `.env`.
  - Evidence: `grep -c DriveAuditSQL services/orion-sql-writer/.env_example` → 0; same on the live `.env`.
- Finding (SHOULD): the original `test_route_map_no_longer_points_at_drive_audit_sql` only asserted against `DEFAULT_ROUTE_MAP` (the Python fallback), never against the actual runtime-effective `Settings().route_map` or `.env_example`'s `SQL_WRITER_ROUTE_MAP_JSON=` line — which is exactly how the finding above slipped through the first pass.
  - Fix: added `test_route_map_json_in_env_example_no_longer_has_drive_audit_entry` (parses `.env_example`'s JSON directly) and `test_settings_effective_route_map_no_longer_has_drive_audit_entry` (instantiates `Settings()` with an explicit override string and asserts the merged `route_map` property).
  - Evidence: `pytest services/orion-sql-writer/tests/test_drive_audit_sql_shape.py -q` → 9 passed (was 7).
- Finding (SHOULD): `scripts/analysis/measure_phase3_biometrics_drive_shadow_comparison.py` does its own direct `SELECT ... FROM drive_audits` with its own pre-existing degrade-safe handling, but received no "permanent, not transient" disclosure note, unlike the two sibling scripts this patch did annotate.
  - Fix: added the same STATUS 2026-08-13 note to its module docstring.
  - Evidence: `grep -n "STATUS 2026-08-13" scripts/analysis/measure_phase3_biometrics_drive_shadow_comparison.py`.
- Finding (process note): the PR report file itself was untracked (`??`), not staged with the rest of the changeset.
  - Fix: staged with `git add -A` before commit.
  - Evidence: `git status --short` after staging shows no untracked files in the diff.
- The review agent's environment lacked `pytest`/`sqlalchemy` (no venv under the worktree) and could not run the test suite itself — it reviewed statically (`py_compile`, manual trace of `route_map`/`MODEL_MAP`/`INSERT_ONLY_MODELS`) and flagged this as a limitation. I ran the actual suite separately (see Tests run above) using the repo-root `.venv`, both before and after applying its findings.

## CI failure fixed post-push

`orion-static-gates` CI's "Metric lineage gate" failed: `scripts/check_metric_lineage.py --gate` ratcheted `bus_channel` orphans 18 → 19. Zeroing `orion:memory:drives:audit`'s `consumer_services` (removing the `"*"` wildcard along with the real `orion-sql-writer` entry) made the channel a genuine orphan by this gate's own definition — a registered `bus_channel` node with no code consumer and no surviving declared consumer (the gate special-cases `"*"` as "an unverifiable but real claim of consumption," so it had been the only thing keeping this entry off the ratchet). The gate's own failure message: "Wire it to a real consumer, or retire the registry entry." Restoring `"*"` would have satisfied the gate cosmetically without being true — the channel really does have zero producers and zero consumers now. Fix: removed the channel entry from `orion/bus/channels.yaml` entirely (matches CLAUDE.md 0A's "kill means kill, no fallback to the thing being killed" — a `producer_services: [], consumer_services: []` entry is a partial exclusion, not a retirement). `DriveAuditV1`'s schema definition and its `orion/schemas/registry.py` entry were left in place (a type definition, not live wiring). Verified locally: `check_metric_lineage.py --gate` → PASS (`bus_channel: 18`, back to baseline), all 5 other static gates still pass, sql-writer test suite unchanged (204 passed / 18 failed / 3 errors, same pre-existing set).

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-sql-writer/.env \
  -f services/orion-sql-writer/docker-compose.yml \
  up -d --build orion-sql-writer
```

## Risks / concerns

- Severity: LOW
- Concern: `scripts/drive_history_reflection_synthesis.py` and `scripts/analysis/measure_autonomy_gate.py` still reference `DriveAuditV1`/`drive_audits` conceptually and will now permanently report "insufficient/missing data" rather than ever producing real output again.
- Mitigation: confirmed both scripts are manual-invocation only (not on any cron/scheduler/systemd timer in this repo) and both already fail open safely on a missing table (no crash, honest degrade) — this was true before this patch too, since the table was already dropped in the earlier PR #1614. This patch changes nothing about their runtime behavior, only documents that the degrade is now permanent.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1628
