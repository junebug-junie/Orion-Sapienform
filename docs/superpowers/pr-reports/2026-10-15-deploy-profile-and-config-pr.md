## Summary

- After a `git pull`, the auto-deploy now rebuilds every service whose image bakes in a changed repo file outside its own folder (for example `config/vision_zones.yaml`), not just services matched by Python imports.
- Services that bind-mount a changed repo file are force-recreated, because an unchanged image would not restart them and a single-file mount keeps serving the old file.
- Compose services hidden behind a `profiles:` gate (the walkway camera, `orion-vision-retina-walkway`) are now rebuilt too, but only if they are already running on this host from this same checkout's compose file. Stopped profile services are never started, and ones started from a worktree or with an overlay compose file are left alone with a warning.
- The `orion-analytics` whole-profile special case still works and is not doubled up.

## Outcome moved

PR #2677 changed `config/vision_zones.yaml`. The post-pull deploy rebuilt none of the three images that copy it (`orion-vision-host`, `orion-sql-writer`, `orion-vision-council`). They kept running the old zones. Now `python3 scripts/rebuild_affected_services.py --list-only --base bfb966b46^1` lists all three. The walkway retina container also used to keep its old image after every pull. A dry run now shows it being rebuilt with `--profile walkway`.

## Current architecture

`scripts/git_hooks/post-merge` (opt-in) runs `scripts/rebuild_services_from_git_diff.sh`, which asks `scripts/rebuild_affected_services.py` for affected service folders. Before this patch, the classifier matched changes to services in only three ways: direct `services/<svc>/` paths, an `orion/` import graph, and an explicit `scripts/` map. `config/` was skip-listed outright. For each affected service, the wrapper ran `safe_docker_build.sh <svc> up -d --build`, which only touches default-profile compose services. Analytics was the one hardcoded exception.

## Architecture touched

- Classifier: new dependency-source index built from each `services/*/docker-compose.yml`. It reads build context and Dockerfile `COPY`/`ADD` sources (handling continuations, flags, JSON form and globs, and skipping `--from` stages), plus repo bind mounts (short and long syntax, `${ORION_HOST_REPO_ROOT:-...}` and relative paths). This check runs before the skip rules. Three kinds of source are deliberately excluded: `orion/` (so the narrow `orion_import_packages` policy is unchanged), whole-repo sources, and whole top-level folders such as `COPY config`, because copying the whole folder does not show that a service reads any particular file in it. A service's own folder gets no force-recreate for directory mounts, only for single-file mounts.
- Classifier CLI: `--plan` prints `<svc>\t<build|recreate>`. `--running-profile-services <svc>` (with `--skip-profile` and `--docker-ps-file`) lists running profile-gated compose services. It matches on docker's `com.docker.compose.project.config_files` label ending in `services/<svc>/docker-compose.yml`, and if docker is unavailable it starts nothing.
- Wrapper: consumes `--plan`, adds `--force-recreate` for recreate services, and runs a per-profile `up -d --build <name>` pass for each running profile service. Dry-run prints both.

## Files changed

- `scripts/rebuild_affected_services.py`: dependency-source index, recreate set, profile detection, new CLI flags.
- `scripts/rebuild_services_from_git_diff.sh`: plan parsing, `--force-recreate`, running-profile pass, `run_safe_build` helper.
- `tests/scripts/test_rebuild_affected_services.py`: #2677 regression, mount/recreate, parser edge cases, fake docker-ps profile tests.
- `tests/scripts/test_rebuild_services_from_git_diff.py`: arg-aware fake classifier; profile pass, recreate, analytics, dry-run tests.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: post-pull rebuild set is larger for `config/` (and other baked/mounted repo path) changes; running profile services are rebuilt.
- Compatibility notes: `--list-only` and `--json` keep their shape; `--json` gains a `recreate` key.

## Env/config changes

- Added keys: none (optional test-only `ORION_REBUILD_DOCKER_PS_FILE` override read by the classifier CLI; not an operator setting)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest tests/scripts/test_rebuild_affected_services.py tests/scripts/test_rebuild_services_from_git_diff.py
  -> 36 passed
Old classifier (origin/main) on --paths config/vision_zones.yaml -> empty list (bug reproduced);
new classifier -> orion-sql-writer, orion-vision-council, orion-vision-host
Mutation check: removing `</dev/null` from the profile loop fails
  test_build_step_reading_stdin_cannot_swallow_later_profile_services.
pytest tests/scripts/ -> 2 failures in test_eval_vision_organ_replay.py, which
  are already broken: they fail without this PR's test files and pass when run alone
  (order-dependent, files untouched here).
```

## Evals run

```text
No eval harness for deploy scripts; behavior covered by deterministic tests above plus a dry run against the real #2677 diff.
```

## Docker/build/smoke checks

```text
scripts/rebuild_services_from_git_diff.sh --dry-run --base bfb966b46^1
  -> 13 services (orion-vision-frame-router with --force-recreate for the mounted
     config/vision_frame_router.yaml); run from the worktree, so the walkway
     container (started from the primary checkout) is correctly left alone with a warning
python3 scripts/rebuild_affected_services.py --repo-root /mnt/scripts/Orion-Sapienform \
    --running-profile-services orion-vision-retina   (live docker ps)
  -> walkway	orion-vision-retina-walkway
CI static gates (.github/workflows/orion-static-gates.yml, run locally): all 35 run steps passed
No real deploy was run.
```

## Review findings fixed

A fresh reviewer subagent read the diff. It found nothing that must be fixed, and five things that should be. All five are fixed:

- Finding: six images do `COPY config` (the whole folder), so every config edit would rebuild mesh-guardian and the llama hosts.
  - Fix: whole top-level folder sources are excluded, the same way `orion/` is.
  - Evidence: `test_whole_config_folder_copy_does_not_fan_out`.
- Finding: a service's own directory mounts forced `--force-recreate` on every change. For analytics, which is mounted whole, that meant recreating stateful containers.
  - Fix: own-folder mounts recreate only when the mount is a single file (the stale-inode case).
  - Evidence: `test_own_directory_mount_is_not_force_recreated`.
- Finding: a build step that reads stdin could swallow the rest of the profile list.
  - Fix: `</dev/null` on the per-profile build.
  - Evidence: `test_build_step_reading_stdin_cannot_swallow_later_profile_services` (fails when the fix is removed).
- Finding: `$` in a COPY source left a dead prefix.
  - Fix: `$` is treated like a glob character.
  - Evidence: the `services/${SVC}/x config/${NAME}.yaml` case in the parser test.
- Finding: profile matching by path suffix could take over a container that was started from a worktree or with an overlay compose file.
  - Fix: the container's compose file list must be exactly this checkout's file, otherwise it is left alone with a warning.
  - Evidence: `test_profile_service_started_from_another_checkout_is_left_alone`.
- Nits fixed: tab-separated COPY, comments inside continuations, a dry-run `set -e` trap, a strict `type: bind` check, and a test for docker being unavailable.

## Restart required

```text
No restart required. The scripts run from the primary checkout on the next pull.
```

## Risks / concerns

- Severity: low
  - Concern: images that `COPY config` whole (mesh-guardian, the llamacpp and ollama hosts) are still not auto-rebuilt for a specific config file they may read.
  - Mitigation: this is deliberate, so one config edit does not rebuild six images. The better fix is to narrow those Dockerfiles to the files they actually use; it is a follow-up.
- Severity: low
  - Concern: changes under `orion/` that are bind-mounted live (retina, and the others that mount `orion/`) still do not trigger a restart.
  - Mitigation: this is deliberate. It keeps the existing narrow `orion_import_packages` policy, and changing that is a separate decision.
- Severity: low
  - Concern: only `docker-compose.yml` is inspected. Alternate compose files such as `services/orion-vision-host/docker-compose.circe-qwen.yml` are not covered.
  - Mitigation: `safe_docker_build.sh` only drives `docker-compose.yml` anyway.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2680

🤖 Generated with [Claude Code](https://claude.com/claude-code)
