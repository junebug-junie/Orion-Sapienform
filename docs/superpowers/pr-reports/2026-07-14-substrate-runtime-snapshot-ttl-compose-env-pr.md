# PR report: `SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC` missing from `orion-substrate-runtime`'s compose environment

## Summary

- Post-deploy verification of `fix/substrate-graph-store-snapshot-cache` (#1040, merged and deployed) found the new env var never actually reaches the running `orion-substrate-runtime` container: `os.getenv("SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC")` returned `None` inside the live container, confirmed via direct Python invocation, not just static file inspection.
- Root cause: unlike `orion-hub`/`orion-cortex-exec` (both use an `env_file:` directive, so every `.env` key reaches them regardless of the `environment:` list), `orion-substrate-runtime` has no `env_file:` directive — `docker-compose.yml`'s explicit `environment:` block is the only path in, and it never listed this key.
- Currently harmless by coincidence: the code's own fallback default (`2.0`) matches the intended value, so the fix is functionally working today. But tuning this value via `.env` later would silently do nothing — the exact "flag set, dependency not wired" footgun already found and fixed twice this session (`orion-recall`, `CONCEPT_RELATION_RESOLUTION_ENABLED`).
- Found using `scripts/check_service_env_compose_parity.py` (built earlier this session for exactly this class of bug), which also surfaced 21 other pre-existing gaps in this same file — reported, not fixed, out of scope for this patch.

## Outcome moved

`SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC` is now genuinely tunable for `orion-substrate-runtime`, not just accidentally correct at its default value.

## Current architecture (before this patch)

`orion-substrate-runtime/docker-compose.yml` has an `environment:` block (no `env_file:` directive) that didn't list `SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC`, added to `.env_example`/live `.env` in the parent PR.

## Architecture touched

`services/orion-substrate-runtime/docker-compose.yml` only.

## Files changed

- `services/orion-substrate-runtime/docker-compose.yml`: added `- SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC=${SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC:-2.0}` to the `environment:` block, next to the other `SUBSTRATE_GRAPH_*` entries.

## Schema / bus / API changes

None.

## Env/config changes

No new keys — this closes a pass-through gap for a key that already existed in `.env_example`/live `.env` from the parent PR.

## Tests run

Not applicable — pure config passthrough, no Python logic changed. Verified via live runtime inspection instead (see below), which is the actual proof this class of bug needs.

## Evals run

Not applicable.

## Docker/build/smoke checks

```text
# Before this fix, live in the running container:
$ docker exec orion-athena-substrate-runtime python -c "
import os
print(os.getenv('SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC'))
from orion.substrate.graphdb_store import build_substrate_store_from_env
store = build_substrate_store_from_env()
print(store._cfg.snapshot_cache_ttl_sec)
"
None
2.0   # only correct by coincidence (code's own fallback default)

# After this fix, docker compose config against real live .env:
$ docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
    -f services/orion-substrate-runtime/docker-compose.yml config
SUBSTRATE_SNAPSHOT_CACHE_TTL_SEC: "2.0"
```

## Review findings fixed

N/A — this is itself a review/verification finding from live post-deploy checking, not a patch that went through a separate review pass. Self-contained, one-line, low-risk change.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Low — 21 other pre-existing keys are also missing from this same `environment:` block (including `BRAIN_FRAME_INTERVAL_SEC` and `SUBSTRATE_BRAIN_FRAME_ENABLED` themselves — the exact config controlling the root-cause tick loop from the parent incident). Deliberately not fixed here — this patch scopes to only the key this session's own patch introduced. Whether to do a full parity sweep on this file is a real, separate scope decision, not something to fold in silently.

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...fix/substrate-runtime-snapshot-ttl-compose-env?expand=1`
