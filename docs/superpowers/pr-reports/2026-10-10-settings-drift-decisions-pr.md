## Summary

- Follow-up to #2592, which merged before Juniper's decisions on the two keys it had refused to guess at. This PR applies those decisions.
- Proposal review is now off everywhere (`HUB_PROPOSAL_REVIEW_ENABLED=false`). orion-context-exec is retired: it was a completed experiment with no production use, and nothing listens on :8096. Hub's "Pending Decisions" panel was pointed at a dead API.
- Orion's own GPU self-shed (`GPU_POOL_ORION_SHED_ENABLED`) now defaults to true in code and in the compose fallback, matching the template and production. Juniper turned it on.
- Both keys come off the drift check's `REASONED_DRIFT` exemption list.

## Outcome moved

Hub stops offering a review panel backed by a service that doesn't exist. A GPU pool booted without the key now behaves like production. The drift check has one exemption left (`CHAT_HISTORY_LOG_CHANNEL`, equivalent by construction).

## Current architecture

After #2592, both keys were exempted in `scripts/check_settings_defaults.py` `REASONED_DRIFT` while waiting for Juniper. Hub's `.env_example` and live `.env` had proposal review on; the GPU pool's code default for Orion self-shed was off while the template and live were on.

## Architecture touched

orion-hub config template; orion-gpu-pool settings, compose and template; the drift check's exemption list. No bus, schema or API changes.

## Files changed

- `services/orion-hub/.env_example`: `HUB_PROPOSAL_REVIEW_ENABLED=false`; comment says context-exec is retired.
- `services/orion-gpu-pool/app/settings.py`, `docker-compose.yml`: `GPU_POOL_ORION_SHED_ENABLED` defaults to true.
- `services/orion-gpu-pool/.env_example`: comment says Juniper turned it on (live since 2026-10-01, code default since 2026-10-10).
- `scripts/check_settings_defaults.py`: both entries removed from `REASONED_DRIFT`.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: Hub proposal review is off once Hub restarts with the updated `.env`.
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: orion-hub (value plus comment), orion-gpu-pool (comment only)
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran from the primary checkout root; nothing added and nothing skipped. The sync script never overwrites an existing value, so the primary checkout's live `services/orion-hub/.env` was edited by hand for `HUB_PROPOSAL_REVIEW_ENABLED=false` only. A line-by-line comparison confirmed line 42 was the only line changed. Until this merges, the sync script reports that key as "diverged" because main's template still says true.
- skipped keys requiring operator action: none

## Tests run

```text
python scripts/check_settings_defaults.py orion-hub --example-drift      -> OK (exit 0), 1 reasoned exemption left
python scripts/check_settings_defaults.py orion-gpu-pool --example-drift -> OK (exit 0), no exemptions
pytest tests/test_check_settings_defaults.py -q                         -> 24 passed
pytest services/orion-gpu-pool/tests -q                                   -> 169 passed, 15 skipped
```

## Evals run

```text
None. This is a config-only change; the drift check is the gate.
```

## Docker/build/smoke checks

```text
Not run (not deploying). The hub container still has HUB_PROPOSAL_REVIEW_ENABLED=true until it restarts.
```

## Review findings fixed

- The config values were decided by Juniper; no new logic was added, and the drift-check logic itself was reviewed in #2592. No new review findings.

## Restart required

Hub only, to pick up `HUB_PROPOSAL_REVIEW_ENABLED=false` (already in the live `.env`). Run from the primary checkout on main after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-hub up -d --build orion-hub
```

The GPU pool needs no restart: it already runs `GPU_POOL_ORION_SHED_ENABLED=true`.

## Risks / concerns

- Severity: low
- Concern: context-exec is retired, but code and config still wire it in. Follow-up only; nothing removed here.
  - In Hub, the flags are off live: `HUB_AGENT_CONTEXT_EXEC_ENABLED=false` and `CONTEXT_EXEC_INVESTIGATION_V2_ENABLED=false`. The supporting pieces are still present:
    - `HUB_CONTEXT_EXEC_API_URL` / `_TIMEOUT_SEC` / `_EVENT_CHANNEL`
    - `HUB_PROPOSAL_REVIEW_API_URL` (:8096)
    - `scripts/context_exec_agent_bridge.py`, `context_exec_client.py`, `agent_step_relay.py`
    - the `api_routes.py` Agent-lane branch
  - orion-cortex-exec runs `CONTEXT_EXEC_ENABLED=true` and `CONTEXT_EXEC_LEGACY_FALLBACK=true` live, even though its code default is false. It still routes over the bus to a service that doesn't exist and then falls back.
  - Elsewhere: the `services/orion-context-exec/` tree itself; channel entries in `orion/bus/channels.yaml`; `orion/schemas/context_exec.py`, `proposal_ledger.py`, `proposal_lifecycle.py` and their `orion/schemas/registry.py` entries; `orion/cognition/verbs/context_exec_trace_autopsy.yaml`.
- Mitigation: separate retirement PR, with Juniper's go-ahead.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2596

🤖 Generated with [Claude Code](https://claude.com/claude-code)
