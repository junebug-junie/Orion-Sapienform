## Summary

- Pause `orion-affectgpt-worker` on circe's post-pull rebuild allowlist until a GPU is chosen for it.
- Correct the allowlist header: the rebuild planner does read it (it is not documentation-only).
- New static gate: a host allowlist may not name a service whose compose requires (`${VAR:?}`) a key that `.env_example` leaves empty.

## Outcome moved

After PR #2894 (the worker's GPU is set by UUID, and the key ships empty on purpose), every circe post-pull rebuild that picked the worker ended with `FAILED orion-affectgpt-worker (exit 1)`. After this patch the planner no longer selects it on circe, and CI catches the same mismatch for any service.

## Current architecture

`scripts/rebuild_affected_services.py` (`_include_file_for_host` / `load_host_allowlist`, ~L691-711) filters affected services to `mesh-utilities/common/include_services_<host>.txt` when that file exists. `up_all_services*.sh` ignore it and read `exclude_services.txt` (which already excludes the worker).

## Files changed

- `mesh-utilities/common/include_services_circe.txt`: worker commented out with a re-add note; header corrected.
- `tests/scripts/test_host_allowlist_buildable.py`: the new gate. It merges the root `.env_example` then the service's own (the same order safe_docker_build uses), ignores `ORION_INCLUDE_SERVICES_FILE`, and covers both allowlist layouts.
- `.github/workflows/orion-static-gates.yml`: own step for the gate.

## Schema / bus / API changes

None.

## Env/config changes

None. The local `.env` warning for `AFFECTGPT_GPU_DEVICE_ID`/`AFFECTGPT_SCRATCH_DIR` on circe is moot while the worker is paused.

## Tests run

```text
pytest tests/scripts/test_host_allowlist_buildable.py              -> 2 passed
  same, with ORION_INCLUDE_SERVICES_FILE=/nonexistent               -> 2 passed
  same, against the pre-patch allowlist                             -> 1 failed (orion-affectgpt-worker requires ['AFFECTGPT_GPU_DEVICE_ID'])
pytest tests/scripts/test_sync_local_env_from_example.py scripts/tests/test_check_env_template_parity.py -> 40 passed
pytest tests/scripts/test_rebuild_affected_services.py              -> 88 passed
```

## Evals run

None. This is a deploy-tooling gate with no behaviour to eval; the gate is its own regression test.

## Docker/build/smoke checks

None; no image changes.

## Review findings fixed

- Finding: the re-add note said to set the UUID in `.env`, but the gate reads `.env_example`.
  - Fix: the note now says the UUID goes in `.env_example` (as kev pins `KEV_GPU_DEVICE_ID`).
- Finding: the root `.env_example` was ignored, a possible false positive.
  - Fix: the root values are merged first.
- Finding: the test inherited an operator's `ORION_INCLUDE_SERVICES_FILE`.
  - Fix: `monkeypatch.delenv`.
- Nits fixed: the gate matches only `${VAR:?}` (not `${VAR?}`); it globs both allowlist layouts; the header notes that the `up_all_services` scripts ignore the file.
  - Evidence: the test runs above.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low.
  - Concern: the worker stays off circe rebuilds until someone re-adds it.
  - Mitigation: the re-add note in the allowlist, and the gate fails until a real UUID is set.

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
