## Summary

- The post-pull rebuild now redeploys a running lane that lives in a service's second compose file (`docker-compose.<name>.yml`). Before, it only ever ran `docker-compose.yml`. The Qwen caption lane (`orion-vision-host/docker-compose.circe-qwen.yml`) therefore ran its 2026-10-16 image and env for days, and dropped every doorbell caption that PR #2747 had enabled.
- A running profile service that was started from a since-deleted worktree of this repo is now adopted into main. Before, it was "left alone" on every pull, forever. Live case: `orion-energy-portal`, started from the pruned `-orion-energy-portal-autologin` worktree (its PR, #2377, is merged).
- `scripts/safe_docker_build.sh --compose-file docker-compose.<name>.yml` targets such a file instead of `docker-compose.yml`. It never merges the two files, and every deploy check still applies.
- Safety rails. The rebuild:
  - redeploys only the services running now (`up --no-deps <names>`);
  - never takes over a live worktree's deploy;
  - never touches a GPU-pool launch file (`config/gpu_pool.yaml`);
  - never redeploys under a different compose project;
  - never redeploys a file whose header names its own env file.

## Outcome moved

The failure mode is closed: "merged and synced, but the running container never got it." On athena today this covers the Qwen caption lane and the energy portal.

## Current architecture

`rebuild_services_from_git_diff.sh` ran `safe_docker_build.sh <svc> up -d --build`, plus `--profile <p> up <name>` for running profile services. The wrapper hard-coded `-f services/<svc>/docker-compose.yml`. Second compose files had no deploy path except a hand-run raw `docker compose`. Profile services started from any other checkout were always skipped.

## Audit (live, 2026-10-20)

- **athena:** `orion-athena-vision-host-qwen` was the only container started from a second compose file. `orion-athena-orion-energy-portal` ran from a deleted worktree. `sweet_murdock` is a GitHub MCP server started by hand (not Orion; out of scope).
- **circe:** the `atlas-*` llama.cpp lanes run from `orion-llamacpp-host/docker-compose.atlas-workers.yml`. The GPU pool's lane controller owns them by design (a `launch:` block), so this PR deliberately leaves them alone.

## Architecture touched

These deploy scripts only: `scripts/rebuild_affected_services.py`, `scripts/rebuild_services_from_git_diff.sh` and `scripts/safe_docker_build.sh`. Also the docs for the Qwen lane. No service code, bus, schema or env change.

## Files changed

- `scripts/rebuild_affected_services.py`:
  - new `--running-extra-compose` / `running_extra_compose_files()` and `pool_launch_composes()`;
  - `_orphaned_copy_of()`, which treats a path as orphaned only when it is gone **and** sits under this repo's worktree prefix with a separator;
  - `_expected_project()`;
  - orphan adoption in `running_profile_services()`.
- `scripts/rebuild_services_from_git_diff.sh`: redeploys running extra compose files with `--compose-file <file> up -d --build --no-deps [--force-recreate] <names>`. A failure marks the stack FAILED, and dry-run reports each file.
- `scripts/safe_docker_build.sh`: the `--compose-file` flag. It accepts basenames only and refuses `/` or `..`; the usage line and heartbeat summary are updated.
- `services/orion-vision-host/{README.md,docker-compose.circe-qwen.yml}`: these said the lane could not use the wrapper. They now give the wrapper command and say the rebuild covers the lane.
- Tests:
  - `tests/scripts/test_rebuild_affected_services.py`: 10 new;
  - `tests/scripts/test_rebuild_services_from_git_diff.py`: 5 new;
  - `tests/scripts/test_safe_docker_build_compose_file.py`: new, 4 tests.

## Schema / bus / API changes

- Added: `--running-extra-compose` (rebuild helper CLI) and `--compose-file` (wrapper flag).
- Removed: none.
- Renamed: none.
- Behavior changed: the post-pull rebuild redeploys running extra-compose lanes, and adopts orphaned profile deploys.
- Compatibility notes: calling the wrapper without `--compose-file` is unchanged.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced: n/a (no template changed)
- skipped keys requiring operator action: none

## Tests run

```text
.venv/bin/python -m pytest tests/scripts/test_rebuild_affected_services.py \
  tests/scripts/test_rebuild_services_from_git_diff.py \
  tests/scripts/test_safe_docker_build_compose_file.py \
  tests/scripts/test_safe_docker_build_heartbeat.py -q          -> 131 passed
services/orion-vision-host/tests/test_circe_qwen_lane_compose.py -> 6 passed
POST_PULL_MIGRATIONS_TEST_POSTGRES_URI=<throwaway postgres:16> pytest tests/test_post_pull_apply_migrations_postgres.py -> 9 passed
  (CI caught this one: its fake helper did not know --running-extra-compose and fell through to
   its default answer, which the rebuild read as an extra compose file; the fake now no-ops it)
Mutation check: removing the "path still exists" guard in _orphaned_copy_of
  -> test_live_sibling_worktree_deploys_are_never_taken_over FAILS; restored -> passes
```

## Evals run

```text
No eval harness for deploy tooling. Live replay against real docker state below.
```

## Docker/build/smoke checks

```text
athena live, --repo-root /mnt/scripts/Orion-Sapienform:
  --running-extra-compose orion-vision-host   -> docker-compose.circe-qwen.yml<TAB>vision-host-circe-qwen
  --running-extra-compose orion-llamacpp-host / orion-bus / orion-cortex-exec / orion-vllm-host -> (nothing)
  --running-profile-services orion-energy     -> portal<TAB>orion-energy-portal (adopted, WARNING names the deleted path)
circe docker ps replayed:
  --running-extra-compose orion-llamacpp-host -> (nothing; pool launch file, and /repo is not an orphan)
python3 scripts/rebuild_affected_services.py --paths services/orion-vision-host/docker-compose.circe-qwen.yml --plan
  -> orion-vision-host build   (a change to only the extra file triggers the rebuild)
```

## Review findings fixed

- Finding (medium): an extra file was brought up whole, so it could start a stopped sibling service.
  - Fix: `up --no-deps <running services>`.
  - Evidence: `test_only_the_running_services_of_an_extra_file_are_named`, plus the shell tests.
- Finding (medium): no test covered "never take over a live worktree's deploy". A mutation that broke the rule passed every test.
  - Fix: `test_live_sibling_worktree_deploys_are_never_taken_over`.
  - Evidence: the mutation check above.
- Finding (low): a lane started under another compose project would clash on `container_name`.
  - Fix: the running project must equal `_expected_project`.
  - Evidence: two tests.
- Finding (low): `orion-bus/docker-compose.atlas-standby.yml` needs its own env file.
  - Fix: a file whose header names `services/<svc>/.env.*` is left alone.
  - Evidence: `test_extra_file_that_needs_its_own_env_file_is_left_alone`.
- Finding (low): the orphan prefix had no separator after the repo name.
  - Fix: `/` or `-` is required.
  - Evidence: a test.
- Nits fixed: the usage line, and the heartbeat summary without a session ID.

Not fixed, accepted:
- Adoption trusts that main supersedes a deleted worktree. The WARNING names the path.
- A crash in the resolver skips the lane silently, the same as the existing profile path.
- Symlinked checkout paths miss, but only on the safe side.

## Restart required

```text
No restart required. The scripts take effect on the next post-pull rebuild. The Qwen lane
was already recreated by hand today (2026-10-20 18:05 UTC). The energy portal is adopted the
next time services/orion-energy changes.
```

## Risks / concerns

- Severity: low.
  - Concern: adopting the energy portal recreates it from main on its next rebuild. That is the intended main code, but it is a restart of a live service.
  - Mitigation: its PR (#2377) is merged, and it has a fixed `container_name` and the same project.
- Severity: low.
  - Concern: the live Qwen-lane redeploy path is UNVERIFIED until the next pull that touches `services/orion-vision-host/`.
  - Mitigation: the resolver was replayed against live docker state, and the wrapper flag is tested.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2808

🤖 Generated with [Claude Code](https://claude.com/claude-code)
