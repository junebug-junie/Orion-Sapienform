## Summary

- `SUBSTRATE_PE_HISTORY_ENABLED` now defaults **true** in `settings.py`, `docker-compose.yml` and `.env_example`.
- README heading says "default on"; the `.env_example` comment no longer tells operators to wait before turning it on.
- New test pins the default in both settings and `.env_example`.

## Outcome moved

#2480 deployed 2026-10-02 08:25 with the flag off, so `substrate_node_prediction_error_history` stayed at 0 rows. With the flag on, every attention-broadcast tick records fresh prediction-error readings and attaches size/range/trend to broadcast loops. Standing rule (Juniper, 2026-10-02): flags ship on; the flag is the rollback lever, not a rollout gate.

## Current architecture

#2480 merged + deployed: migration applied (table exists, empty), consumers (thought, hub) rebuilt first, substrate-runtime running the new code with the flag off. Thought logged 0 `attention broadcast read failed` since rebuild, so turning the producer on is consumer-safe.

## Files changed

- `services/orion-substrate-runtime/app/settings.py`: default True
- `services/orion-substrate-runtime/docker-compose.yml`: `:-true`
- `services/orion-substrate-runtime/.env_example`: `=true`, comment updated
- `services/orion-substrate-runtime/README.md`: "default on"
- `services/orion-substrate-runtime/tests/test_worker_pe_history.py`: default-on test

## Schema / bus / API changes

None.

## Env/config changes

- Changed default: `SUBSTRATE_PE_HISTORY_ENABLED` false -> true
- local/prod `services/orion-substrate-runtime/.env` set to `true` in the same change

## Tests run

```text
pytest services/orion-substrate-runtime/tests/test_worker_pe_history.py -q  -> 13 passed
mutation: settings default back to False -> default test fails (1 failed)
```

## Evals run

```text
None yet. 24 h after enable: python services/orion-substrate-runtime/evals/run_pe_magnitude_live_sanity.py
```

## Review findings fixed

- Config-only flag flip; no separate review findings.

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d
```

## Risks / concerns

- Severity: low. Concern: history table growth (<30k rows/day, pruned at 168 h). Mitigation: set the flag false to stop writes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
