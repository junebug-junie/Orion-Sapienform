## Summary

- The post-pull rebuild no longer recreates `orion-hub` or `orion-harness-governor` while one of Juniper's chat turns is in flight. It builds first, waits until Hub shows no chat turn running or queued (max 600 s), and then restarts.
- New `scripts/wait_for_chat_turns_idle.py` reads Hub's existing `/api/runtime-activity` tracker. It always exits 0 and prints one verdict line.
- `safe_docker_build.sh` gains a check-only mode (`ORION_SAFE_BUILD_PREFLIGHT_ONLY=1`). It runs every refusal check and stops before compose, so the rebuild never waits for an `up` that would be refused.
- Design doc for the full fix: chat as a durable run (`chat.orion_turn`) through cortex-orch's durable front door, with retries and a resumed `claude` session.

## Outcome moved

Before: a pull during a chat killed the reply, and nothing retried it. After: the restart waits for the chat to finish, up to 10 minutes. Each wait leaves a `CHAT_DRAIN` line in `~/.orion/post_pull_rebuild.log`.

## Current architecture

`rebuild_one` ran `up -d --build` for both stacks. Compose recreated the container whenever the build finished, whatever was running in it.

## Architecture touched

Deploy scripts only. No service code, schema, bus, or env template changes.

## Files changed

- `scripts/rebuild_services_from_git_diff.sh`: drain branch for Hub and the governor (build, check-only, wait, `up --no-build`). The wait is logged as its own START/END step.
- `scripts/safe_docker_build.sh`: check-only mode.
- `scripts/wait_for_chat_turns_idle.py`: the bounded wait.
- `tests/scripts/test_wait_for_chat_turns_idle.py`, `tests/scripts/test_rebuild_services_from_git_diff.py`, `tests/scripts/test_check_rebuild_log.py`: coverage.
- `CLAUDE.md` section 8: documents the drain and its knobs.
- `docs/superpowers/specs/2026-11-25-durable-chat-turns-design.md`: design for Patches 1–2.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: Hub and governor rebuilds are now two compose steps (`build`, then `up -d --no-build`), with a bounded wait between them.
- Compatibility notes: none.

## Env/config changes

- Added keys: shell-only knobs, not in `.env_example`. Same pattern as `ORION_SANDBOX_SYNC_WAIT_S`.
  - `ORION_REBUILD_CHAT_DRAIN` (default 1)
  - `ORION_REBUILD_CHAT_DRAIN_WAIT_S` (default 600)
  - `ORION_SAFE_BUILD_PREFLIGHT_ONLY`
  - `ORION_HUB_URL` (optional override; the default is 127.0.0.1 on `HUB_PORT`)
- `.env_example` updated: no. Local `.env` sync: not needed. Skipped keys: none.

## Tests run

```text
pytest tests/scripts/test_wait_for_chat_turns_idle.py tests/scripts/test_rebuild_services_from_git_diff.py \
  tests/scripts/test_rebuild_deployed_head_catchup.py tests/scripts/test_check_rebuild_log.py tests/test_deploy_migration_gate.py
150 passed
bash -n on both shell scripts: ok
```

## Evals run

None. This is deploy plumbing with no quality dimension. The live check is the first real `CHAT_DRAIN` line in the rebuild log.

## Docker/build/smoke checks

```text
python3 scripts/wait_for_chat_turns_idle.py --wait-s 0   -> idle   (against live Hub)
ORION_SAFE_BUILD_PREFLIGHT_ONLY=1 scripts/safe_docker_build.sh orion-hub up -d --no-build  (worktree) -> refused, rc=1
ORION_SAFE_BUILD_PREFLIGHT_ONLY=1 scripts/safe_docker_build.sh orion-hub config -> "preflight only ... allowed", rc=0, no compose run
```

A live drain during a real pull is UNVERIFIED. It needs a pull that touches Hub while a chat is running.

## Review findings fixed

- Finding: a SIGKILL during the up-to-10-minute wait left Hub or the governor out of `check_rebuild_log.py`'s finish command. The service already had a START line, but nothing was in flight.
  - Fix: the wait is logged as its own step (`START/END <svc> chat-drain`, with `IN_FLIGHT` set).
  - Evidence: `test_a_run_killed_during_the_chat_drain_still_lists_the_service` and `test_drain_is_logged_as_its_own_step`.
- Finding: the build-then-wait ran even when the `up` would be refused (a migration deferred behind a backup, or the deploy pin moved). That could add about 20 minutes inside the rebuild lock and block `git pull`.
  - Fix: a check-only `safe_docker_build.sh` call runs first, and the wait is skipped when it refuses.
  - Evidence: `test_drain_is_skipped_when_the_up_would_be_refused` and the real-script smoke above.

## Restart required

```text
No restart required. Takes effect on the next post-pull rebuild.
```

## Risks / concerns

- Severity: low
  - Concern: a chat turn that starts in the seconds between the check and the restart is still killed. So is Hub's short pre-handoff phase (stance and recall).
  - Mitigation: Patch 1 of the design (durable chat turns) retries those turns.
- Severity: low
  - Concern: `git pull` can block up to 10 minutes longer while a chat runs.
  - Mitigation: lower it with `ORION_REBUILD_CHAT_DRAIN_WAIT_S`, or switch it off with `ORION_REBUILD_CHAT_DRAIN=0`.
- Issues opened: #4001 (consults and round-ups bypass cortex's durable front door). Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/4002

🤖 Generated with [Claude Code](https://claude.com/claude-code)
