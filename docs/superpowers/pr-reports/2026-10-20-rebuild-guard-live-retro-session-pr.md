## Summary

- The post-pull rebuild no longer recreates orion-retro or orion-hub while a retro game is live. It skips those two stacks, logs `SKIPPED <svc> reason=retro_session_live`, prints the exact command to run later, and rebuilds everything else.
- New read-only helper `scripts/check_retro_session_live.py` decides "live": the retro service's `status.json` says starting/running/paused AND the `orion-athena-retro` container is running.
- Skipped stacks count as deferred, not failed: the run exits 0 when nothing else failed. The `DONE` log line gains ` deferred=<svcs>` only when something was deferred.
- Override: `ORION_REBUILD_IGNORE_LIVE_RETRO=1`.

## Outcome moved

Live 2026-10-20 22:13Z: a post-pull rebuild recreated `orion-athena-retro`, `orion-athena-hub` and `orion-athena-hub-retro-gateway` while Juniper was playing. The game lived only in the retro worker's memory and was lost. With this patch that run would have deferred both stacks. Live check from this branch at 22:37Z (Juniper's new game): `session state=paused game=crystalis in running container orion-athena-retro`, exit 3 (live).

## Current architecture

`rebuild_one` in `scripts/rebuild_services_from_git_diff.sh` rebuilt every affected stack with no knowledge of in-memory sessions. orion-retro persists its session to `<RETRO_EVIDENCE_DIR>/status.json` (`services/orion-retro/app/session.py` `persist()`), and on boot rewrites any starting/running/paused file to `failed` (`service_restart`).

## Why the signal is trustworthy

- `status.json` alone can be stale: a container that died without restarting leaves "running" behind. So the container must be running (`docker inspect`).
- A running container cannot carry a stale live state: boot rewrites it to `failed` before serving.
- The evidence dir comes from the running container's `/evidence` mount source; `services/orion-retro/.env` `RETRO_EVIDENCE_DIR` is only a fallback.
- Fail-safe: docker errors, a missing evidence dir, or an unreadable `status.json` with the container running = UNKNOWN, which also skips (`reason=retro_session_unknown`). A late rebuild is cheap; a killed game is not. No container, a stopped container, or no `status.json` = not live (nothing in memory to lose).
- A missing helper file warns and does not block: a real pull always brings both files together, and test fixtures that copy only the rebuild script keep working.

## Architecture touched

Deploy tooling only. No service code, bus, schema, or env-template changes.

## Files changed

- `scripts/check_retro_session_live.py`: new read-only live-game check (exit 0/3/4).
- `scripts/rebuild_services_from_git_diff.sh`: `retro_guard_skips()` called at the top of `rebuild_one`, and again after a separate `build` step for orion-retro/orion-hub; deferred summary on stderr and in the `DONE` log line. Does not touch the lines PR #2818 edits (77-93); `git merge-tree` with #2818's branch: no conflict.
- `tests/scripts/test_rebuild_retro_session_guard.py`: helper fixtures (each state, stale file + dead container, no container, no file, corrupt file, docker error, .env fallback) and end-to-end script runs (skip + continue, unknown skips, stale file rebuilds, override, dry-run).
- `.github/workflows/post-pull-migrations.yml`: runs the new tests.
- `services/orion-retro/README.md`: one-line note.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: post-pull rebuild defers orion-retro/orion-hub during a live game.
- Compatibility notes: `DONE` log line unchanged unless something was deferred.

## Env/config changes

- Added keys: none in any `.env_example`. `ORION_REBUILD_IGNORE_LIVE_RETRO` is a per-command operator override, not a service config key.
- `.env_example` updated: no. Local `.env` sync: not needed. Skipped keys: none.

## Tests run

```text
pytest -q tests/scripts/test_rebuild_retro_session_guard.py tests/scripts/test_rebuild_services_from_git_diff.py tests/test_post_pull_apply_migrations.py
193 passed
First version of the new tests against the pre-fix rebuild script: 4 failed (skip, unknown, override, dry-run), 12 passed (helper-only + stale-file cases).
bash -n scripts/rebuild_services_from_git_diff.sh: ok
Live: python3 scripts/check_retro_session_live.py -> "session state=paused game=crystalis in running container orion-athena-retro (status.json last written 0.1h ago)", rc=3
```

## Evals run

```text
None. Deploy tooling with no eval harness; behavior is covered by the deterministic tests above.
```

## Docker/build/smoke checks

```text
No build or deploy run (task said do not deploy). Live helper run above reads the real container and status.json.
```

## Review findings fixed

Code review ran in a subagent. No must-fix findings.

- Finding (should): the check ran once before a multi-minute `up -d --build`; a game started during the build would still be killed.
  - Fix: for orion-retro/orion-hub the script now runs `build` first, checks again, and only then runs `up` (which hits the build cache).
  - Evidence: `test_game_started_during_build_is_not_recreated` (the fake build writes a live status.json; only `build` runs, the stack is SKIPPED).
- Finding (should): helper stderr (a traceback) went multi-line into the one-record-per-line log.
  - Fix: output collapsed to one line, max 400 chars.
  - Evidence: `test_guard_crash_is_unknown_and_skips_with_one_line_log`.
- Finding (should): no bound on how long a live state holds; a stale `running` could survive if the container restarts with `RETRO_ENABLED=false`.
  - Fix: the reason now reports `status.json last written <N>h ago` so the operator can judge. No auto-expiry: the file is written on state changes, not per frame, so age alone is not proof of staleness. Listed under risks.
  - Evidence: `test_live_reason_reports_status_age`; live output `(status.json last written 0.1h ago)`.
- Finding (nit): `Path.exists()` could hide a permission error as "not live" on newer Python.
  - Fix: `stat()` with FileNotFoundError = not live, any other OSError = UNKNOWN.
- Test gaps filled: guard crash (rc 1) maps to unknown, docker binary missing = UNKNOWN, DONE line with both failed and deferred.
- Not changed: `.env` fallback does not resolve relative paths (only used without a mount; result is UNKNOWN, fail-safe); missing helper fails open (documented; a pull brings both files).

## Restart required

```text
No restart required. The next `git pull` on the primary main checkout uses the new script.
```

## Risks / concerns

- Severity: low. Concern: a game left paused for days defers retro/hub rebuilds for days. Mitigation: every skipped run logs it and prints the exact command; `ORION_REBUILD_IGNORE_LIVE_RETRO=1` overrides.
- Severity: low. Concern: if orion-retro crashes mid-game and comes back with `RETRO_ENABLED=false`, the boot rewrite to `failed` may not run, and the stale `running` reads as live until someone clears it. Mitigation: the skip line shows the file age; override with `ORION_REBUILD_IGNORE_LIVE_RETRO=1`.
- Severity: low. Concern: guarded stacks now run `build` then `up -d --build` as two steps (the second build is a cache hit). Mitigation: none needed; adds seconds.
- Severity: low. Concern: only the post-pull rebuild is guarded; a manual `scripts/safe_docker_build.sh orion-hub up` still recreates the hub mid-game. Mitigation: out of scope; follow-up if wanted.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
