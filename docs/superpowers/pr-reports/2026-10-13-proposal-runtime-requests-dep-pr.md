## Summary

- orion-proposal-runtime's image now installs `requests`.
- Regression test pins it: the import chain `orion.proposals.builder` -> `orion.attention.world_first` -> `orion.substrate.graphdb_store` needs `requests`.

## Outcome moved

About half of proposal-runtime ticks were crashing with `ModuleNotFoundError: No module named 'requests'` (`proposal_runtime_tick_failed`). Live 2026-10-13 19:48-19:54 UTC: 23 failed ticks, 22 saved frames. After this patch every tick can build.

## Current architecture

`_is_internal_target` (orion/proposals/builder.py, world-first seam #2583) calls `field_target_source_kind` (orion/attention/field_attention/selectors.py:502), which imports `orion.attention.world_first`, which imports `orion.substrate` -> `graphdb_store.py:16 import requests`. The proposal-runtime requirements never listed `requests`. The break surfaced when the image was rebuilt on 2026-10-13 for the arousal readers (#2635).

## Architecture touched

`services/orion-proposal-runtime/requirements.txt` only.

## Files changed

- `services/orion-proposal-runtime/requirements.txt`: add `requests==2.32.3` (the repo's most common pin) with a comment naming the import chain.
- `services/orion-proposal-runtime/tests/test_runtime_dependencies.py`: fails if `graphdb_store` still imports `requests` and the requirements lack it.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
.venv/bin/python -m pytest -q services/orion-proposal-runtime/tests/test_runtime_dependencies.py  -> 1 passed
old requirements: 0 'requests' lines; new: 1 (test would fail on the old file)
```

## Evals run

No eval harness change; this is a packaging fix.

## Docker/build/smoke checks

```text
docker build -f services/orion-proposal-runtime/Dockerfile .  -> ok
docker run ... python -c "from orion.attention.field_attention.selectors import field_target_source_kind; import orion.attention.world_first"  -> import ok
```

## Review findings fixed

Two-line dependency fix plus a static test; self-reviewed, no subagent review run.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
```

## Risks / concerns

- Severity: low. Concern: other services that import `orion.proposals.builder` may have the same gap. Mitigation: none found running; follow-up if seen.

## PR link

(filled after creation)
