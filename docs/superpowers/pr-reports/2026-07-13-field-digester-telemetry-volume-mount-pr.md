## Summary

- Fixes a gap in `feat/field-channel-raw-corpus-collector` (merged, PR #1022): that PR added `FIELD_CHANNEL_CORPUS_PATH` defaulting under `/mnt/telemetry`, but `orion-field-digester`'s `docker-compose.yml` never mounted `/mnt/telemetry` into the container. Without this mount, the path resolves inside the container's ephemeral filesystem — the sink would appear to work (no error, no crash) but the data would silently vanish on the next container restart/rebuild, never reaching the host at all.
- Adds a `volumes:` section mirroring `orion-spark-introspector`'s existing, already-working mount exactly: `${TELEMETRY_ROOT:-/mnt/telemetry}:/mnt/telemetry`.
- Found while walking through actually deploying the collector for the first time (setting a real `FIELD_CHANNEL_CORPUS_PATH` value and preparing to restart) — caught before the restart happened, not after data loss.

## Outcome moved

`FIELD_CHANNEL_CORPUS_PATH` will now actually persist to the host once this merges and the service restarts, instead of silently writing into a throwaway container layer.

## Current architecture

`orion-field-digester`'s compose file had no `volumes:` section at all before this patch — every other path this service reads/writes (Postgres URI, lattice config) either goes through the database or a config file baked into the image, so this is the first time the service needed a host bind mount.

## Architecture touched

`services/orion-field-digester/docker-compose.yml` only. No application code, schema, or settings changes.

## Files changed

- `services/orion-field-digester/docker-compose.yml`: added `volumes: - ${TELEMETRY_ROOT:-/mnt/telemetry}:/mnt/telemetry`.

## Schema / bus / API changes

None.

## Env/config changes

None new — reuses the existing `TELEMETRY_ROOT` variable convention already established by `orion-spark-introspector`'s compose file (defaults to `/mnt/telemetry` if unset, same as that service).

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -c "import yaml; yaml.safe_load(open('services/orion-field-digester/docker-compose.yml')); print('OK')"
=> OK
```
No application code changed — no pytest suite applicable to this patch.

## Evals run

Not applicable.

## Docker/build/smoke checks

Not run this session (compose YAML validated as above). Real validation is the pending restart on the live host, which is what this fix unblocks.

## Review findings fixed

None — single-line, mirrors an existing working pattern exactly, no review subagent spawned for a 7-line infra-config diff.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-field-digester/.env \
  -f services/orion-field-digester/docker-compose.yml up -d --build
```
This is the same restart already needed to deploy `feat/field-channel-raw-corpus-collector`'s code — this fix should land before or together with that restart, not after, so the first restart is the one that actually persists data correctly.

## Risks / concerns

- Severity: low
- Concern: none identified — this is a corrective fix for an already-merged PR's own gap, caught pre-deployment.

## PR link

`gh` is unauthenticated in this environment — open manually at:
https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/field-digester-telemetry-volume-mount
