# PR report: chore/harness-route-env-sync

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1769

## Summary

- PR #1761 shipped the `harness` route split in code (gateway, Hub picker filter, actions/cortex-exec normalization) but the config activation step was never applied — `LLM_GATEWAY_ROUTE_TABLE_JSON` had no `harness` key and `~/.fcc/.env`'s `MODEL*` still said `chat`.
- Verified live before this patch: a real FCC turn run after the #1761 merge resolved `model=chat route=chat` in the gateway's `anthropic_passthrough` log — functionally unchanged, still sharing `circe-worker-1`'s single slot with live Hub chat traffic.
- `services/orion-llm-gateway/.env_example`: added the `harness` route table entry (interim alias of `chat`'s own worker/URL, `"priority":"system"`), with a comment block mirroring the existing `agent`-split documentation.
- `config/fcc.env_example`: `MODEL`/`MODEL_SONNET`/`MODEL_OPUS` changed from `chat` → `harness` (`MODEL_HAIKU` untouched). Documented that this shared file drives BOTH Hub Agent Claude mode and Orion's harness motor together (confirmed via `services/orion-harness-governor/tests/test_harness_governor_rpc.py`'s "MODEL_SONNET/MODEL_OPUS share one route" comment) — by design, this moves both away from plain `chat`, not just one of them.
- Live `services/orion-llm-gateway/.env` and `~/.fcc/.env` updated to match (gitignored, not part of this diff) and applied: `orion-llm-gateway` rebuilt/restarted, `orion-athena-fcc` restarted.

## Outcome moved

FCC/Claude Code CLI harness turns and Hub Agent Claude turns now resolve to the dedicated `harness` route instead of the shared `chat` route — closing the config gap #1761 left open.

## Current architecture

Before this patch: `harness` existed only as code (route type, gateway logic, Hub filter) with no route-table entry and no consumer pointed at it. Every FCC-driven turn (Hub Agent Claude and Orion's own harness motor) still resolved to `chat`, sharing its single-slot worker with live Hub chat traffic — exactly the contention #1761 was meant to remove.

## Architecture touched

- `services/orion-llm-gateway/.env_example` (route table template)
- `config/fcc.env_example` (shared FCC model-routing template, consumed by both `orion-fcc` and `orion-harness-governor` containers)
- Live-only, not in this diff: `services/orion-llm-gateway/.env`, `~/.fcc/.env`

## Files changed

- `services/orion-llm-gateway/.env_example`: harness route table entry + doc comment
- `config/fcc.env_example`: MODEL/MODEL_SONNET/MODEL_OPUS → harness, doc comment on shared-file semantics

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: FCC/harness traffic now resolves to the `harness` route instead of `chat` once the synced `.env`/`~/.fcc/.env` values are live (already applied on this host).
- Compatibility notes: `harness` is an interim alias of `chat`'s own worker (same url/served_by) — a labeling/observability seam, not physical isolation yet.

## Env/config changes

- Added keys: none (no new key names — existing keys' values changed)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes — `services/orion-llm-gateway/.env_example`, `config/fcc.env_example`
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not applicable — this was a JSON-blob-internal-key addition and a shared-file value change, not suited to the sync script's key-add/report model; `services/orion-llm-gateway/.env` and `~/.fcc/.env` were edited by hand to match and applied live (see Docker/build/smoke checks below).
- skipped keys requiring operator action: none

## Tests run

```text
python3 scripts/check_service_env_compose_parity.py orion-llm-gateway
# orion-llm-gateway declares env_file: -- all 69 .env_example keys reach the container. N/A.
```

No code changed in this patch (config/example files only) — #1761's own test suite (356 tests) already covers the code paths this activates.

## Evals run

None — config/example-only change, no eval-relevant code touched.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-llm-gateway --env-file .env --env-file services/orion-llm-gateway/.env \
  -f services/orion-llm-gateway/docker-compose.yml up -d --build
# built + recreated, GET /health -> {"status":"ok",...}

curl -fsS http://localhost:8210/routes | jq '.routes[] | select(.id=="harness")'
{
  "id": "harness", "served_by": "circe-worker-1", "backend": "llamacpp",
  "status": "up", "model": "/models/gguf/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf",
  "priority": "system", ...
}
# previously: "status": "not_configured"

scripts/safe_docker_build.sh orion-fcc --env-file services/orion-fcc/.env -f services/orion-fcc/docker-compose.yml up -d --build
docker restart orion-athena-fcc   # compose saw no config diff (only mounted file contents changed); forced restart to pick up new ~/.fcc/.env
# GET /health -> {"status":"healthy"}

curl -X POST http://127.0.0.1:8082/v1/messages -d '{"model":"claude-sonnet-4-20250514", ...}'
# gateway log:
[LLM-GW] INFO - anthropic_passthrough corr=- model=harness route=harness \
  upstream=http://100.112.254.99:8011/v1/messages served_by=circe-worker-1 \
  stream=True tools=0
# Before this patch, the same call logged model=chat route=chat.
```

`scripts/smoke_llm_gateway_routes.py` was attempted but hits a pre-existing, already-documented issue unrelated to this patch: `DEFAULT_ROUTE_SERVERS["chat"]`/`["agent"]` are stale `atlas-worker-1` placeholders (the script's own comments call this out as "NOT fixed here" debt) — the RPC call itself succeeded (`served_by=circe-worker-1`), only the smoke's hardcoded expectation is wrong. Not touched here; out of scope.

## Review findings fixed

N/A — config/example-only change, no code touched; skipped the code-review subagent for this reason (nothing but literal env values and comments changed, verified by direct diff against origin/main above).

## Restart required

Already performed as part of this patch:

```bash
docker restart orion-athena-fcc
# orion-llm-gateway was rebuilt via safe_docker_build.sh above
```

No further restart required.

## Risks / concerns

- Severity: low
- Concern: `~/.fcc/.env` carries the comment "Managed by Free Claude Code /admin. Edit in the server UI when possible." — a future admin-UI save could overwrite this manual edit back to stale values.
- Mitigation: none automated yet; worth a follow-up smoke/gate that periodically checks `~/.fcc/.env`'s `MODEL*` values match `config/fcc.env_example`'s intent, but out of scope for this thin patch.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1769
