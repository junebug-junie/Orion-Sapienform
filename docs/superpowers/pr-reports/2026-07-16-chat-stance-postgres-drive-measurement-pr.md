# PR report: chat stance → Postgres drive measurement (graph snapshot SoR off)

Branch: `feat/chat-stance-postgres-drive-measurement`
Series: Cypher-native substrate + Postgres-via-bus split (after Falkor adapter #1120)

## Summary

- Chat stance now fills `ctx["chat_drive_state"]` from a bounded fail-open latest-row
  fetch of Postgres `drive_audits` (`subject='orion'`), same bus→sql-writer rail Mind uses.
- `_project_autonomy_from_beliefs` no longer projects measurement from graph
  `snapshot_source="drive_state"` (returns `drive_state: None`).
- Substrate-runtime `DRIVE_STATE_SUBSTRATE_MATERIALIZATION_ENABLED` defaults **false**;
  writer remains behind the flag as a rollback ladder only.
- No Hub→HTTP→sql-writer path; no new measurement table (`activations` approximated from
  `active_drives`).

## Outcome moved

Before: chat stance depended on substrate graph drive snapshots for pressures /
activations / dominant / summary / tension_kinds, while Mind already trusted Postgres.
After: one measurement SoR for stance + Mind; graph materialization off by default.

## Current architecture

DriveEngine → bus `orion:memory:drives:audit` → sql-writer → `drive_audits`.
Mind/Thought already read latest row. Stance read unified-beliefs graph snapshots fed by
substrate-runtime `_materialize_drive_state_to_substrate` (default on).

## Architecture touched

- `orion-cortex-exec` (Postgres fetch + stance wiring)
- `orion-substrate-runtime` (materialization default off + compose passthrough)
- `scripts/sync_local_env_from_example.py` (`CHAT_STANCE_` / `DRIVE_STATE_SUBSTRATE_` prefixes)
- `orion/self_state/inner_state_registry.py` (consumer notes)

## Files changed

- `services/orion-cortex-exec/app/drive_state_postgres.py`: new fail-open fetch + stance mapping
- `services/orion-cortex-exec/app/chat_stance.py`: wire Postgres; stop graph SoR projection
- `services/orion-cortex-exec/tests/test_drive_state_postgres.py`: fetch/mapping/schema guard
- `services/orion-cortex-exec/tests/test_chat_stance_drive_state_projection.py`: Postgres wiring tests
- `services/orion-cortex-exec/.env_example`, `docker-compose.yml`, `README.md`: timeout + docs
- `services/orion-substrate-runtime/{settings,.env_example,docker-compose,worker,tests}`: default off
- `docs/superpowers/plans/2026-07-16-chat-stance-postgres-drive-measurement.md`: slice plan
- design acceptance checkboxes for drive items marked done

## Schema / bus / API changes

- Added: none
- Removed: none
- Behavior changed: stance `chat_drive_state` source = Postgres; graph drive snapshot SoR off
- Compatibility: `activations` is `{d: True for d in active_drives}` (lossy vs full bool map)

## Env/config changes

- Added keys: `CHAT_STANCE_DRIVE_STATE_FETCH_TIMEOUT_SEC` (default `0.4`)
- Behavior changed: `DRIVE_STATE_SUBSTRATE_MATERIALIZATION_ENABLED` default `false`
- `.env_example` updated: yes (cortex-exec + substrate-runtime)
- local `.env` synced: operator shared-checkout `.env` updated
  (`CHAT_STANCE_DRIVE_STATE_FETCH_TIMEOUT_SEC=0.4`, materialization flipped to `false`)
- skipped keys requiring operator action: `PUBLISH_CORTEX_EXEC_GRAMMAR` (NEVER_SYNC; unchanged)

## Tests run

```text
pytest services/orion-cortex-exec/tests/test_drive_state_postgres.py \
       services/orion-cortex-exec/tests/test_chat_stance_drive_state_projection.py -q
# 17 passed

pytest services/orion-substrate-runtime/tests/test_drive_state_substrate_materialization.py -q
# 9 passed

pytest services/orion-cortex-exec/tests/test_autonomy_slice.py \
       services/orion-cortex-exec/tests/test_router_autonomy_payload_export.py -q
# 38 passed
```

## Evals run

```text
No new eval harness for this seam; Mind already covered by cortex-orch/thought drive facet tests.
Follow-up: optional live smoke that chat_drive_state_diagnostics.reason=success on a real turn.
```

## Docker/build/smoke checks

```text
Not rebuilt in this session. Restart commands below pick up env default flip + stance code.
```

## Review findings fixed

- Finding: unused settings fields for timeout/visibility (dual config source)
  - Fix: removed from `settings.py`; runtime stays on `os.getenv` (matches existing visibility flag)
  - Evidence: settings.py no longer defines those fields
- Finding: missing `no_meaningful_content` + SELECT schema drift tests
  - Fix: added both in `test_drive_state_postgres.py`
  - Evidence: tests pass
- Finding: materialization key missing from substrate compose
  - Fix: added `DRIVE_STATE_SUBSTRATE_MATERIALIZATION_ENABLED` (+ drive channels) to compose
  - Evidence: docker-compose.yml

## Restart required

```bash
# From a worktree (not shared checkout), after merge/deploy:
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build

# Or equivalent compose up with --env-file for those two services.
# Confirm substrate-runtime env shows DRIVE_STATE_SUBSTRATE_MATERIALIZATION_ENABLED=false
# and cortex-exec has ORION_ACTION_OUTCOME_DB_URL pointing at conjourney.
```

## Risks / concerns

- Severity: Low
- Concern: `activations` loses explicit `False` entries vs old graph map
- Mitigation: autonomy_slice does not use activations; router preview only; document approximation
- Severity: Medium (ops)
- Concern: if `ORION_ACTION_OUTCOME_DB_URL` blank in a deploy, stance fail-opens with empty drive state
- Mitigation: `.env_example` + compose already set conjourney DSN; diagnostics on ctx

## PR link

(filled after `gh pr create`)
