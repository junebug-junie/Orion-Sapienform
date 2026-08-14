# llm-gateway: split agent route off chat, point it at Muse Glimmer

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1636
Branch: `fix/llm-gateway-agent-route`

## Summary

- `services/orion-llm-gateway`'s `LLM_GATEWAY_ROUTE_TABLE_JSON` default had `"agent"` aliasing `"chat"` (both pointed at `circe-worker-1:8011`) — a merged-mode default from before any distinct agent-lane model existed.
- Now that Muse Glimmer (`config/llm_profiles.yaml`: `muse-glimmer-30b-udq4kxl-v100-32gb-agent-vision`, PRs #1627/#1634) is live on Atlas's dedicated agent-lane worker (`atlas-worker-agent-1`, port 8014), `"agent"` points there instead.
- README's route-table examples were also stale (backwards default labeling + wrong `chat` URL) — a `/code-review medium` pass caught it, fixed in the same PR.
- Considered and ruled out: whether this silently changes FCC's backing model too (README documents an FCC config using `MODEL=llamacpp/agent`). Confirmed with the operator their actual FCC usage ("orion chat unified") uses Mode: Orion + Compute: Chat, not Compute: Agent — unaffected.

## Outcome moved

Hub's "Compute: Agent" selector (and anything else selecting the `agent` logical lane via `orion-llm-gateway/app/lane_routes.py`) now actually reaches Muse Glimmer instead of silently landing on Circe's chat worker running an unrelated model.

## Current architecture

`orion-llm-gateway` is the single-subscriber LLM routing service. `LLM_GATEWAY_ROUTE_TABLE_JSON` maps logical route keys (`chat`/`agent`/`metacog`/`quick`/`quick_background`) to backend `{url, served_by, backend}` entries. `app/lane_routes.py:resolve_llm_lane_route()` resolves a caller's requested lane (`chat`/`spark`/`background`/`agent`) to one of those route-table keys — for `agent` specifically, it looks up `_AGENT_ROUTE_KEYS = ("agent",)` directly in the route table. Hub's UI exposes this as a "Compute" selector; a separate "Mode" selector (e.g. "Quick") drives a different code path entirely (`body_route` for the `chat` logical lane), which is why an operator hitting a 400 on Mode: Quick and a stale Compute: Agent mapping were two separate, independently-diagnosed problems in the same debugging session.

## Architecture touched

- `services/orion-llm-gateway/.env_example`: default route table's `agent` entry.
- `services/orion-llm-gateway/README.md`: two example route-table blocks + the callout describing them.
- Live local `.env` in the primary checkout (`/mnt/scripts/Orion-Sapienform/services/orion-llm-gateway/.env`), hand-synced to match — not part of this PR's diff (gitignored), but required for this repo's own dev environment to reflect the new default.

## Files changed

- `services/orion-llm-gateway/.env_example`: `agent`'s `url`/`served_by` changed from `circe-worker-1:8011` to `atlas-worker-agent-1:8014`; comment above the block rewritten to explain the split and how to re-merge if needed.
- `services/orion-llm-gateway/README.md`: swapped which route-table example is labeled "default" (split-agent, now default) vs. "legacy" (merged mode); corrected `chat`'s URL in both example blocks from a stale `atlas-worker-1:8011` to the real `circe-worker-1:8011` default.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: any caller resolving the `agent` logical lane through `orion-llm-gateway` now reaches a different backend (`atlas-worker-agent-1:8014` instead of `circe-worker-1:8011`). `chat`, `metacog`, `quick`, and `quick_background` routes are unchanged.
- Compatibility notes: this is a live-routing behavior change, not a schema change — takes effect only once whichever host actually runs the live `orion-llm-gateway` process is restarted with the updated env (see "Restart required").

## Env/config changes

- Added keys: none — `LLM_GATEWAY_ROUTE_TABLE_JSON` already existed; only its default *value* changed.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes, `agent`'s url/served_by within the existing `LLM_GATEWAY_ROUTE_TABLE_JSON` key.
- local `.env` synced: **not** via `python scripts/sync_local_env_from_example.py` — that script's diverged-key protection intentionally does not overwrite a key that already exists locally with a different value (to avoid clobbering intentional operator overrides), so `LLM_GATEWAY_ROUTE_TABLE_JSON` wasn't touched by running it. Hand-edited the primary checkout's `services/orion-llm-gateway/.env` directly to match, per this repo's rule that the checked-in default must reflect the live intended default.
- skipped keys requiring operator action: none beyond the above — but the host actually running the live gateway process (likely not this dev machine) needs the same manual `.env` update and a restart; this repo's local `.env` sync doesn't reach that host.

## Tests run

```text
$ pytest services/orion-llm-gateway/tests/test_lane_routes.py services/orion-llm-gateway/tests/test_route_catalog.py -q
11 passed

$ pytest services/orion-hub/tests/test_llm_route_selector.py -q
21 passed, 3 failed -- reproduces identically against unmodified origin/main
(test_should_use_context_exec_agent_lane_when_enabled,
test_build_context_exec_request_sets_llm_profile,
test_run_hub_agent_via_context_exec_uses_quick_profile) -- confirmed pre-existing,
unrelated to this change.
```

Both test files exercise route-resolution *logic* (which route-table key a given lane/body_route resolves to), not the specific URL a route entry points at — so they pass regardless of which physical worker `agent` targets, as expected for a pure default-value change.

## Evals run

No eval harness applicable — this is a routing-config change, not a model-quality change.

## Docker/build/smoke checks

Not run from this dev environment (no access to whichever host runs the live `orion-llm-gateway` process). See "Restart required" for the exact commands to run there.

## Review findings fixed

- `/code-review medium` against `fix/llm-gateway-agent-route`: README.md's route-table examples labeled merged mode (agent aliasing chat) as "default" and the split table as "optional" — backwards relative to the `.env_example` default this PR's first commit shipped. CLAUDE.md section 16 lists README.md as a surface to update in the same changeset when affected.
  - Fix: swapped the labels (split-agent is now "default", merged mode is now "legacy" with a note on when to use it), and while in that section, also corrected both example blocks' stale `chat` URL (`atlas-worker-1` → the real `circe-worker-1` default) — the same class of copy-paste-the-wrong-default risk the review flagged for `agent`, just for `chat` too.
  - Evidence: second commit on this branch (`396950cf0`).

## Restart required

```bash
# Wherever the live orion-llm-gateway instance actually runs (not this dev environment):
# 1. Pull this change / update that host's services/orion-llm-gateway/.env to match
#    the new LLM_GATEWAY_ROUTE_TABLE_JSON default (agent -> atlas-worker-agent-1:8014).
# 2. Restart:
docker compose -f services/orion-llm-gateway/docker-compose.yml up -d --build llm-gateway
curl -s http://127.0.0.1:8210/v1/models | jq
# 3. Verify from Hub: select Compute: Agent (not Mode: Quick) and confirm the response
#    comes from Muse Glimmer (e.g. ask it to identify itself, or check response timing
#    against the DFlash-speedup expectation).
```

## Risks / concerns

- Severity: low
  - Concern: this repo's local `.env` for orion-llm-gateway was hand-synced (not via the sync script) to match the new default. Whichever host actually runs the live gateway process needs the same manual update — this repo's `.env` isn't automatically that host's runtime config, and there's no automated propagation.
  - Mitigation: exact restart/verify commands above; flagged explicitly rather than assumed done.
- Severity: low
  - Concern: FCC's documented `MODEL=llamacpp/agent` config (README lines ~96-119) would have been affected if live, but operator confirmed their actual FCC usage path doesn't select the `agent` compute lane.
  - Mitigation: none needed for the operator's current usage; worth re-checking if FCC's config or usage pattern changes later.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1636
