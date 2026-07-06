# PR: fix(orion-thought) — reverie reads attention broadcast via direct DSN, not heavy substrate

Branch: `fix/reverie-thought-db-read-recover` → `main`
Commit: `91f400aa`

## Summary

Reverie was **inert in production**. `_default_broadcast_reader` imported
`orion.substrate.felt_state_reader`, whose package `__init__` drags
`materializer → graphdb_store → import requests`. orion-thought is a thin bus
service that ships neither `requests` nor the graph engine, so **every 90s
reverie tick raised `ModuleNotFoundError: No module named 'requests'`**, read no
coalition, and produced zero thoughts (the code looked wired; the runtime was
dead — config-truth ≠ runtime-truth, §0A).

- Add `services/orion-thought/app/broadcast_reader.py`: minimal fail-open direct
  query of `substrate_attention_broadcast_projection` (jsonb `projection_json` +
  tz-aware `generated_at`), sqlalchemy + psycopg2 only, 300s staleness guard.
- `reverie.py` / `store.py` now use a direct `POSTGRES_URI` DSN (→ `conjourney`)
  instead of `felt_state_reader`.
- Add `SQLAlchemy==2.0.36` + `psycopg2-binary==2.9.10` pins.
- Add `POSTGRES_URI` to `.env_example`.
- Rewrite the three reader unit tests onto the new path + add an empty-table
  fail-open test.

## Outcome moved

Reverie went from **0 thoughts/tick (crash-looping)** to live: post-rebuild logs
show consecutive `reverie thought published ... salience=0.720` with no
`No module` errors; `substrate_reverie_thought` grew 24 → 27 within a minute of
restart.

## Current architecture (before)

- `services/orion-thought/` — thin bus service (fastapi/uvicorn/pydantic/redis/
  PyYAML). Evoked path gets its coalition from the request payload
  (`request.association.broadcast`).
- Reverie is self-driven with no incoming payload, so it must fetch the coalition
  itself — it did so via `orion.substrate.felt_state_reader.hydrate_felt_state_ctx`,
  which pulls the heavy `orion.substrate` package.

## Architecture touched

- New thin seam `app/broadcast_reader.py` isolates the broadcast read behind a
  direct DSN, so the service never imports `orion.substrate.*`.
- `store.py` `_database_url()` switched to the same direct DSN.

## Files changed

- `services/orion-thought/app/broadcast_reader.py`: new fail-open direct reader.
- `services/orion-thought/app/reverie.py`: `_default_broadcast_reader` → direct reader.
- `services/orion-thought/app/store.py`: `_database_url` → direct DSN, no substrate import.
- `services/orion-thought/requirements.txt`: add SQLAlchemy + psycopg2-binary pins.
- `services/orion-thought/.env_example`: add `POSTGRES_URI`.
- `services/orion-thought/tests/test_reverie_spontaneous_thought.py`: reader tests → new path + empty-table test.

## Schema / bus / API changes

- None. No channel, schema, or payload changes. Reads an existing table.

## Env/config changes

- Added key: `POSTGRES_URI` (orion-thought `.env_example`), form
  `postgresql://postgres:postgres@${PROJECT}-sql-db:5432/conjourney`.
- Local `.env` already carries the resolved `orion-athena-sql-db` value + reverie
  flags (`ORION_REVERIE_ENABLED=true`, resonance flags). `.env` remains gitignored.
- No skipped keys.

## Tests run

```
.venv/bin/python -m pytest services/orion-thought/tests -q
55 passed, 13 warnings in 2.83s
```

## Runtime / smoke checks

```
# broadcast projection table confirmed
\d substrate_attention_broadcast_projection
  generated_at    | timestamp with time zone | not null
  projection_json | jsonb                    | not null
# latest broadcast ~4s old (substrate ticking)

# in-container reader, no requests
docker exec orion-athena-thought python -c "from app.broadcast_reader import read_latest_broadcast; print(read_latest_broadcast())"
  read ok, broadcast: present substrate.attention.broadcast.v1

# live logs after rebuild
reverie thought published corr=95bfd496-... salience=0.720 channel=orion:reverie:thought
reverie thought published corr=ec515c69-... salience=0.720 channel=orion:reverie:thought
reverie thought published corr=475ec9b7-... salience=0.720 channel=orion:reverie:thought
# zero "No module" / "broadcast read failed" lines post-restart

# substrate_reverie_thought count: 24 → 27 within a minute
```

## Review findings fixed

- Code review (high effort) over the 6-file diff: no material findings. Imports
  (`Any`, `os`, `AttentionBroadcastProjectionV1`) all still used after edits; all
  paths fail-open to None/False, never raise; static SQL (no injection).

## Restart required

```bash
docker compose -p orion-thought \
  --env-file .env \
  --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml \
  up -d --build thought
```

Already applied — reverie is live.

## Risks / concerns

- Severity: low. `broadcast_reader` keeps its own sqlalchemy engine (separate pool
  from `store.py`) — negligible at 90s cadence.
- Durability depends on this branch being merged to local `main`; a full `--build`
  from unmerged `main` would revert reverie to the broken state.

## Status

DONE_WITH_CONCERNS — landed, pushed, runtime-verified. Open items: browser PR +
merge to local `main`.
