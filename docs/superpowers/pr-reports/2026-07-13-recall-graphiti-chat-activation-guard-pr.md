# PR report: activate real chat-time graph search + permanently guard it from the sync script

## Summary

- `RECALL_GRAPHITI_IN_CHAT` (default `false`, and previously entirely absent from this host's live `.env`) gates whether real chat/conversation turns — both `chat_general` and unified-turn (`stance_react`, confirmed to share the same PCR/recall machinery) — ever use the graph-search rail proven working earlier today. It was off; every bit of that earlier work was invisible to real conversations.
- Flipped on, live: `RECALL_GRAPHITI_IN_CHAT=true` + `RECALL_GRAPHITI_ADAPTER_URL=http://orion-athena-graphiti-adapter:8000` (both were missing from the live `.env` entirely — added, not just toggled).
- `.env_example` updated to match — this is now the shipped default, not just a live-only override, following the same "prove it, then make it the default" pattern already used for the graphiti_core backend default flip earlier today.
- `RECALL_GRAPHITI_IN_CHAT`/`RECALL_GRAPHITI_ADAPTER_URL` added to `sync_local_env_from_example.py`'s `NEVER_SYNC_KEYS` — an explicit, permanent guarantee (verified even under `--all-keys --force`) that these two keys can never be silently desynced from each other, closing off the exact failure mode that broke two other features earlier this session (`CONCEPT_RELATION_RESOLUTION_ENABLED`, `GRAPHITI_BACKEND`: flag enabled, dependency left empty).
- Live-verified via the actual running `Settings` object and the real gating function, not `docker exec env` (which doesn't reflect what this service actually reads — see below) — confirmed a real `GraphitiAdapter` is constructed for `retrieval_intent="contradiction"` and correctly `None` for every other intent.

## Outcome moved

The graph-search work from earlier today (activation, the RELATES_TO fix, default-flip, driver caching) now actually affects real conversations, not just the adapter's own API and Hub's manual calls. And the specific footgun that bit this session twice already (flag on, dependency empty, silent no-op) is now structurally prevented for this feature specifically, and self-documented for the next person who touches it.

## Current architecture (before this patch)

`orion-recall/app/collectors/active_packet.py::_graphiti_adapter()` already existed and correctly gated graph search to `retrieval_intent == "contradiction"` only — this logic was never broken. What was missing was the config to actually turn it on: `RECALL_GRAPHITI_IN_CHAT` wasn't even present in the live `.env`, so it silently defaulted to `False` in Python.

## Architecture touched

`services/orion-recall` (env only, no code change — the gating logic was already correct), `scripts/sync_local_env_from_example.py` (tooling).

## A real gotcha found and worked around during verification

`docker exec orion-athena-recall env | grep RECALL_GRAPHITI...` returned nothing even after the container was rebuilt with the new `.env` — this looked like the change hadn't taken effect. Root cause: `orion-recall`'s `Dockerfile` does `COPY services/orion-recall /app` at build time, baking `.env` directly into the image, and `settings.py`'s pydantic-settings `Config` reads `env_file=".env"` directly — so the app never relies on docker-compose's `environment:` list at all (which, separately, doesn't even list these two keys — a pre-existing gap in `docker-compose.yml`, not touched by this patch since it isn't the actual read path). `docker exec ... env` only shows OS-level environment variables, not what pydantic-settings loads from the baked-in file. Verified correctly instead by asking the running app's actual `Settings` object directly (see Docker/build/smoke checks below) — this is the "runtime truth beats config truth" standard this whole session has been held to, applied to distinguish "looks unset" from "the app's actual view."

## Files changed

- `services/orion-recall/.env_example`: `RECALL_GRAPHITI_IN_CHAT` default `false`→`true`; `RECALL_GRAPHITI_ADAPTER_URL` filled in with a real, container-DNS-reachable value (matching the exact precedent already used for `CRYSTALLIZER_EMBED_HOST_URL` when the graphiti_core backend's own default flipped earlier today); comment documents the `NEVER_SYNC_KEYS` protection and the must-change-together requirement.
- `services/orion-recall/.env` (live, not committed — gitignored): same two keys, added (were entirely absent, not just `false`).
- `scripts/sync_local_env_from_example.py`: `NEVER_SYNC_KEYS` gains `RECALL_GRAPHITI_IN_CHAT`, `RECALL_GRAPHITI_ADAPTER_URL`. Belt-and-suspenders on top of an existing, weaker protection: neither key matches any `SYNC_PREFIXES` entry (verified directly via `should_sync_key()`), so they were already excluded from the script's default scan — this makes that exclusion an explicit, permanent decision rather than an accident of the prefix list, and extends it to hold even under `--all-keys --force`.
- `tests/scripts/test_sync_local_env_from_example.py`: two new tests, mirroring the file's existing `test_orion_bus_url_never_synced` style — `should_sync_key()` returns `False` for both keys in both modes; a real `sync_file()` round-trip with `--force`+`--all-keys` (the strongest possible attempt to override) leaves both keys completely untouched.
- `docs/superpowers/specs/2026-07-13-recall-followups-loop-retirement-saturation-gate-spec.md`: carried over from an earlier uncommitted state this session, included here since it was sitting untracked and this PR is the natural place for it to land (it documents unrelated prior work from the same day, not part of this patch's diff otherwise).

## Schema / bus / API changes

None. No new keys — both `RECALL_GRAPHITI_IN_CHAT` and `RECALL_GRAPHITI_ADAPTER_URL` already existed in `.env_example`; only their default values and sync-protection changed.

## Env/config changes

- Changed defaults: `RECALL_GRAPHITI_IN_CHAT` (`false`→`true`), `RECALL_GRAPHITI_ADAPTER_URL` (empty→`http://orion-athena-graphiti-adapter:8000`) in `.env_example`.
- Live `.env` updated to match (both keys were entirely absent before this session; added, not committed — gitignored).
- `python scripts/sync_local_env_from_example.py orion-recall --dry-run` confirmed clean (`No changes needed`) after these edits — no unexpected divergence introduced elsewhere in the file.
- Both changed keys are now permanently excluded from any future sync-script run, by design.

## Tests run

```text
$ source venv/bin/activate && python -m pytest tests/scripts/test_sync_local_env_from_example.py -v
8 passed in 0.10s
```
All existing tests plus the 2 new ones. No regression.

## Evals run

Not applicable — this is config + a tooling guard, not application logic with a meaningful eval surface.

## Docker/build/smoke checks

```text
$ docker compose --env-file .env --env-file services/orion-recall/.env \
    -f services/orion-recall/docker-compose.yml up -d --build
Container orion-athena-recall Started

$ docker ps --format '{{.Names}}\t{{.Status}}' | grep recall
orion-athena-recall   Up About a minute (healthy)

$ curl -sS http://localhost:<port>/health   (via docker exec, internal)
{"ok":true,"service":"recall","version":"0.1.0","node":"athena-DEBUG-CONTAINER"}

# Real verification -- NOT docker exec env (see gotcha above), the app's own Settings object:
$ docker exec orion-athena-recall python -c "
from app.settings import settings
print(settings.RECALL_GRAPHITI_IN_CHAT, settings.RECALL_GRAPHITI_ADAPTER_URL)
"
True http://orion-athena-graphiti-adapter:8000

# The actual gating function, exercised directly, both branches:
$ docker exec orion-athena-recall python -c "
from app.settings import settings
from app.collectors.active_packet import _graphiti_adapter
a = _graphiti_adapter(settings, intent='contradiction')
print(a.enabled, a.url)   # -> True http://orion-athena-graphiti-adapter:8000
b = _graphiti_adapter(settings, intent='semantic')
print(b)                  # -> None (narrow gate confirmed correct)
"
```

## Review findings fixed

None from a separate review pass — this is a config activation + tooling-guard patch, self-reviewed by directly exercising the real code path (the `_graphiti_adapter()` call above) rather than trusting the config values alone, catching the `docker exec env` red herring along the way rather than reporting a false negative.

## Restart required

Already executed live during this session:

```bash
docker compose --env-file .env --env-file services/orion-recall/.env \
  -f services/orion-recall/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Low — `docker-compose.yml`'s `environment:` block still doesn't list `RECALL_GRAPHITI_*`/`RECALL_ACTIVE_PACKET_ENABLED`/`ACTIVATION_RECALL_FLOOR` at all. Harmless today (the baked-in `.env` file is the actual read path, confirmed), but worth knowing: if this service's Dockerfile ever stops baking `.env` into the image (e.g., switches to a bind-mount-only pattern), these keys would silently stop being read unless also added to the compose file. Not fixed here — out of scope for this patch, flagging for visibility.
- Severity: Low — `RECALL_GRAPHITI_FALKORDB_URI` remains unset/empty; not required for the `contradiction`-intent search path (`GraphitiAdapter` only needs the adapter URL), left as-is.

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...chore/recall-graphiti-chat-activation-guard?expand=1`
