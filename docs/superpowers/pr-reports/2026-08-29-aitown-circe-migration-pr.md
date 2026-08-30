# PR report: AI Town Atlas → Circe migration + Hub-owned convex config

PR (code): https://github.com/junebug-junie/Orion-Sapienform/pull/1970  
Follow-up (this doc + operator env): `chore/aitown-circe-pr-report`

## Summary

- Deprecated Atlas; AI Town Convex stack now runs on **Circe** (`100.112.254.99`).
- Convex data bind-mount moved to `/mnt/telemetry/orion-circe/ai-town/convex-data` (not under `/mnt/scripts`).
- Hub AI Town panel and `/aitown-convex` proxy read **`HUB_AITOWN_*` from `services/orion-hub/.env`** — not `~/.fcc/.env`.
- Embodiment / harness MCP still read **`AITOWN_*` from `~/.fcc/.env`** for Orion's in-world body.
- LLM route gate blocks **`chat` / `circe-worker-1` only**; `quick_background` on `circe-worker-fast-1` is allowed post-Atlas.
- Live data compacted on Circe (~16GB → ~200MB db); compaction cron moved Atlas → Circe.

## Outcome moved

| surface | before | after |
|---|---|---|
| AI Town host | Atlas (deprecated) | Circe `100.112.254.99` |
| Convex data path | Atlas telemetry | `/mnt/telemetry/orion-circe/ai-town/convex-data` |
| Hub status/proxy config | `~/.fcc/.env` or localhost fallback | `HUB_AITOWN_CONVEX_URL` / `HUB_AITOWN_UI_URL` in hub `.env` |
| Hub panel timeout | stale Atlas URL in running container | Circe URLs after hub recreate |
| `db.sqlite3` (Circe) | ~16GB bloat | ~212MB after export/import |

Live endpoints:

- Town UI: `http://100.112.254.99:5173/ai-town/`
- Convex: `http://100.112.254.99:3210`
- World ID: `m174spk0rd4namch9qvt53fs4x8a4f62`

## Current architecture

Hub embeds AI Town via reverse proxy (`/aitown/`, `/aitown-convex/*`). Status probe calls Convex directly using server-side settings. Orion's embodied presence and harness MCP tools reach Convex through operator `~/.fcc/.env` (`AITOWN_*`), a deliberate split so the Hub panel does not depend on FCC secrets layout.

AI Town service is self-hosted Convex + Vite frontend on the mesh node named in `URL_BASE`. LLM for NPC dialogue is wired through orion-llm-gateway (`quick_background` lane → `circe-worker-fast-1`), not the interactive `chat` lane.

## Architecture touched

- `services/orion-ai-town/` — compose bind mount, README evacuation, LLM route check
- `services/orion-hub/` — settings, `aitown_status.py`, `api_routes.py`, compose env passthrough
- `config/fcc.env_example` — Circe `AITOWN_CONVEX_URL`; documents Hub vs embodiment split (this follow-up PR)

## Files changed

### PR #1970 (merged `c01bc389c`)

- `services/orion-ai-town/docker-compose.yml`, `.env_example`, `README.md`
- `services/orion-ai-town/scripts/check_llm_route_not_circe.py` + tests
- `services/orion-hub/app/settings.py`, `scripts/aitown_status.py`, `scripts/api_routes.py`
- `services/orion-hub/.env_example`, `docker-compose.yml`, `README.md`, tests

### This follow-up PR

- `docs/superpowers/pr-reports/2026-08-29-aitown-circe-migration-pr.md` — this report
- `config/fcc.env_example` — Circe cutover URL + Hub/embodiment env split comment
- `services/orion-harness-governor/.env_example` — `HARNESS_AITOWN_CONVEX_URL` → Circe mesh IP

## Schema / bus / API changes

- Hub `/api/aitown/status` and `/aitown-convex/*` return HTTP 400 when `HUB_AITOWN_CONVEX_URL` unset (no silent localhost fallback).
- No bus/schema contract changes.

## Env/config changes

### Hub (`services/orion-hub/.env` — gitignored, operator sync)

```bash
HUB_AITOWN_ENABLED=true
HUB_AITOWN_UI_URL=http://100.112.254.99:5173
HUB_AITOWN_CONVEX_URL=http://100.112.254.99:3210
HUB_AITOWN_WORLD_ID=m174spk0rd4namch9qvt53fs4x8a4f62
HUB_AITOWN_ADMIN_KEY=
```

Do **not** put Hub panel URLs in `~/.fcc/.env`.

### Embodiment (`~/.fcc/.env`)

```bash
AITOWN_CONVEX_URL=http://100.112.254.99:3210
# AITOWN_WORLD_ID, AITOWN_ORION_PLAYER_ID, etc. as before
```

### Circe AI Town (`services/orion-ai-town/.env`)

- `URL_BASE=http://100.112.254.99`
- `TELEMETRY_ROOT=/mnt/telemetry` (default)
- LLM wired: `LLM_API_URL=http://100.92.216.81:8210`, `LLM_MODEL=quick_background`

## Live migration steps (already executed)

1. Rsync convex-data Atlas → Circe (via Athena staging).
2. Update compose bind mount + Circe `.env`; `docker compose up -d --build`.
3. Compact: `scripts/compact_convex_data.sh` (export → reset sqlite → import).
4. Wire LLM gateway; verify `check_llm_route_not_circe.py`.
5. Cutover Athena hub `.env` + `~/.fcc/.env`; recreate `orion-athena-hub`.
6. Move weekly compaction cron from Atlas to Circe (fnm/Node on PATH).

## Tests run

```text
pytest services/orion-ai-town/tests/test_check_llm_route_not_circe.py -q  # 9 passed
pytest services/orion-hub/tests/test_aitown_status_api.py services/orion-hub/tests/test_aitown_proxy.py -q  # 5 passed
curl -fsS http://127.0.0.1:8080/api/aitown/status  # ok, convex_reachable: true (post hub recreate)
```

## Evals run

```text
None (config/migration)
```

## Docker/build/smoke checks

```text
Circe: AI Town stack up, Convex health OK after compact
Athena: docker compose -f services/orion-hub/docker-compose.yml up -d --build hub-app
```

## Review findings fixed

- Hub no longer reads AI Town convex URL from `~/.fcc/.env`.
- `.env_example` documents `/ai-town/` suffix belongs to Hub proxy, not `HUB_AITOWN_UI_URL`.
- `config/fcc.env_example` updated so operators do not leave stale Athena Convex URL for embodiment.

## Restart required

```bash
# Athena Hub (after hub .env sync):
python3 scripts/sync_local_env_from_example.py
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d hub-app

# Circe AI Town (only if .env or compose changed):
docker compose --env-file services/orion-ai-town/.env \
  -f services/orion-ai-town/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
- Concern: Operators who only update `~/.fcc/.env` will fix embodiment but not the Hub panel.
- Mitigation: Split documented in hub `.env_example`, `config/fcc.env_example`, and this report.

## PR link

- Code: https://github.com/junebug-junie/Orion-Sapienform/pull/1970 (merged)
- Report + env follow-up: (this branch)
