## Summary

- Treat empty hub `social_artifact_*` placeholders in `client_meta` as absent instead of validating them into hard failures.
- Wrap `orion:chat:social:stored` consumption in try/except so one bad turn cannot silently kill the subscription task.
- Add regression test for empty artifact placeholder dicts on real hub-direct turn shape.
- Completes the hub-direct social continuity pipeline together with sql-writer post-commit `publish_with_reconnect` (already on `main` in `291a45ca`).

## Outcome moved

Hub-direct social-room turns now have a live path from Postgres write → `orion:chat:social:stored` → social-memory synthesis without manual backfill or consumer restart.

## Current architecture

Before this patch, sql-writer could persist `social_room_turns` and (after `publish_with_reconnect`) emit stored events, but social-memory’s bus consumer died on the first hub turn carrying empty `social_artifact_proposal: {}` keys. Continuity tables stayed stale even though `/summary` prefetch still worked from old projections.

## Architecture touched

- `services/orion-social-memory/app/synthesizer.py` — `_optional_artifact_record()` guard
- `services/orion-social-memory/app/service.py` — consumer error isolation
- Related upstream (already on main): `services/orion-sql-writer/app/worker.py` — `_publish_post_commit()` via `publish_with_reconnect`

## Files changed

- `services/orion-social-memory/app/synthesizer.py`: skip empty/invalid artifact dialogue records
- `services/orion-social-memory/app/service.py`: log and continue on stored-turn processing failures
- `services/orion-social-memory/tests/test_artifact_dialogue_records.py`: regression for hub empty placeholders

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: social-memory no longer crashes on `{}` artifact placeholders in stored-turn payloads
- Compatibility notes: consumer now survives malformed artifact fragments; invalid non-empty payloads are ignored per-field

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=services/orion-social-memory:. pytest services/orion-social-memory/tests/test_artifact_dialogue_records.py -q
# 1 passed

PYTHONPATH=services/orion-sql-writer:. pytest services/orion-sql-writer/tests/test_social_turn_stored_emit.py -q
# 1 passed (upstream fix already on main)
```

## Evals run

```text
Not run — no social-memory eval harness for bus consumer path; regression covered by unit test + live smoke on athena.
```

## Docker/build/smoke checks

```text
docker compose -f services/orion-social-memory/docker-compose.yml up -d --build
# rebuilt and restarted orion-athena-social-memory

Live smoke (athena):
- Full client_meta turn from social_room_turns no longer crashes process_social_turn
- Bus publish to orion:chat:social:stored updates evidence_count after consumer restart
- Hub-direct turn 02811559 backfilled manually pre-patch
```

## Review findings fixed

- Finding: empty `{}` artifact keys from hub are not valid Pydantic records
  - Fix: `_optional_artifact_record()` treats empty/invalid dicts as absent
  - Evidence: `test_artifact_dialogue_records_ignores_empty_placeholder_dicts`
- Finding: unhandled exception in consumer loop kills subscription with no log
  - Fix: try/except with `social_turn_stored_process_failed` log
  - Evidence: consumer survived subsequent bus publishes in live smoke

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-social-memory/.env \
  -f services/orion-social-memory/docker-compose.yml up -d --build

# sql-writer already patched on main; restart if not yet redeployed:
docker compose --env-file .env --env-file services/orion-sql-writer/.env \
  -f services/orion-sql-writer/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
- Concern: silently dropping malformed non-empty artifact payloads could hide producer bugs
- Mitigation: log on handler failure; valid populated artifact records still validate strictly via `_optional_artifact_record` only after non-empty check

## PR link

https://github.com/junebug-junie/Orion-Sapienform/compare/main...fix/social-stored-consumer-resilience
