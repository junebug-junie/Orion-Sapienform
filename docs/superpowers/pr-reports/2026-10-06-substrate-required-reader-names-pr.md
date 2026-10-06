## Summary

- After the Stage 2 deploy (#2515/#2520), the referent and assertion projectors' readiness gate could never open. `SUBSTRATE_ASSERTION_REQUIRED_READERS` listed `orion-hub`, `orion-recall`, `orion-cortex-exec`, `orion-cortex-orch`, `orion-spark-concept-induction` and `orion-meta-tags`, but those services advertise their `SERVICE_NAME`: `hub`, `recall`, `cortex-exec`, `cortex-orch`, `spark-concept-induction`, `meta-tags`.
- The list now uses the names each service really advertises.
- New test: every required reader name must be a `SERVICE_NAME` (or `SUBSTRATE_READER_NAME`) some service declares, and the compose default must match `.env_example`. It fails on the old list.
- CI path triggers now include `services/*/.env_example` and `services/*/docker-compose.yml`, so a service rename re-runs the check.

## Outcome moved

The gate opens once every reader has really been rebuilt with #2515, instead of never.

## Live evidence (2026-10-06)

Capability keys present in Falkor Redis: `orion-substrate-runtime`, `hub`, `orion-memory-consolidation`. The projector waited for `orion-hub`, so the gate stayed shut by name, not by missing code.

## Files changed

- `services/orion-memory-consolidation/.env_example`, `docker-compose.yml`: corrected the default list.
- `orion/substrate/tests/test_required_reader_names_match_services.py`: new regression test.
- `.github/workflows/substrate-neighborhood.yml`: runs the test; path triggers.

## Env/config changes

- Changed default: `SUBSTRATE_ASSERTION_REQUIRED_READERS`.
- The local `.env` value was edited by hand in the primary checkout (sync does not overwrite an existing key).

## Tests run

```text
test_required_reader_names_match_services.py: 2 passed (1 fails on the previous list)
check_env_template_parity: PASS; check_definition_drift --gate: PASS
```

## Restart required

Merge, then rebuild memory-consolidation so it reads the corrected list (one line, from the primary checkout on main):

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-memory-consolidation up -d --build
```

The 6 readers not yet rebuilt with #2515 (cortex-exec, cortex-orch, spark-concept-induction, field-digester, world-pulse, meta-tags) still need rebuilding before the gate opens.

## Review findings fixed

None. Review skipped: a config-default correction plus a regression test.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
