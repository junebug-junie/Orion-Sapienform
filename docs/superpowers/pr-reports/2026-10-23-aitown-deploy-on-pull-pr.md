## Summary

- A `git pull` now really deploys AI Town. Before this, the post-pull rebuild rebuilt only the frontend, from an unpatched `upstream/`, and never pushed the Convex functions. The live town ran 2026-08-30 code until 2026-10-23. It had none of #2230 (self-repeat), #2228 (plans), or #2871 (echo).
- `rebuild_services_from_git_diff.sh` gets two per-stack hooks:
  - `services/<svc>/scripts/pre_rebuild.sh` runs before the build. If it fails, the stack is skipped.
  - `services/<svc>/scripts/post_rebuild.sh` runs only after every compose step passed. If it fails, the stack is marked failed.
- `orion-ai-town/scripts/pre_rebuild.sh` backs up `upstream/`, resets it to the pin, and applies the chain from scratch. Plain re-apply fails on a patched tree.
- `orion-ai-town/scripts/post_rebuild.sh` pushes the functions with `npx`, or with `node:20-alpine` on circe.

## Outcome moved

- Failure mode: merged town patches never reached the live world. Now any pull that touches `services/orion-ai-town` deploys them.
- Failure mode: deploying without the local cards replaced Juniper's real join text with the redacted example. Now that refuses. This happened on circe during the 2026-10-23 manual deploy, and was fixed by hand before the push.

## Current architecture

The post-merge hook runs `rebuild_services_from_git_diff.sh`, which calls `safe_docker_build.sh <svc> up -d --build` for each changed stack. AI Town's compose builds only the frontend, from `./upstream`. The Convex functions were deployed by hand.

## Architecture touched

`scripts/rebuild_services_from_git_diff.sh` (generic hooks) and `services/orion-ai-town/scripts/`. No bus, schema, or env changes.

## Files changed

- `scripts/rebuild_services_from_git_diff.sh`: pre/post hooks; dry-run names them.
- `services/orion-ai-town/scripts/pre_rebuild.sh` (new): back up, reset to the pin, apply the chain.
- `services/orion-ai-town/scripts/post_rebuild.sh` (new): push the Convex functions.
- `services/orion-ai-town/tests/test_pre_rebuild.py` (new): runs both scripts on a real pinned clone.
- `tests/scripts/test_rebuild_services_from_git_diff.py`: hook order, failure and dry-run behavior.
- `.github/workflows/aitown-patch-chain.yml`: runs `test_pre_rebuild.py`; triggers on `.env_example`.
- `services/orion-ai-town/README.md`: deploy section.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: the post-pull rebuild of `orion-ai-town` now deploys the patched upstream and the Convex functions.
- Compatibility notes: stacks without hook files are unchanged.

## Env/config changes

- Added keys: none. There are script-only overrides: `AITOWN_ALLOW_EXAMPLE_CARDS`, `AITOWN_BACKUP_ROOT`, and the test-only `AITOWN_TEST_CONVEX_PUSH`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest tests/scripts/test_rebuild_services_from_git_diff.py services/orion-ai-town/tests (AITOWN_REQUIRE_TOOLS=1, node 24, local pinned clone) -> 188 passed, 1 skipped (pre-existing)
bash -n on all three scripts -> ok
git diff --check -> ok
pytest tests/scripts (full) -> 2 failures in test_eval_vision_organ_replay.py: order-dependent; they pass alone and on main; unrelated
```

## Evals run

None. This is deploy plumbing; the tests above exercise the real scripts.

## Docker/build/smoke checks

The manual equivalent ran on circe 2026-10-23:
1. Backup to `~/.orion/aitown-upstream-backup-20261023T184424Z`.
2. Reset and apply: all 17 patches "Applied".
3. Copied the local cards; the Juniper text then matched the old live text byte for byte.
4. `convex dev --once`: "Convex functions ready".
5. `safe_docker_build.sh orion-ai-town up -d --build frontend`.
6. `testing:resume`: world `running`.

The hook path itself runs on the next pull that touches `services/orion-ai-town`. UNVERIFIED until then.

## Review findings fixed

- Finding (medium): the push ran before `up`, so it could never heal a stopped stack, and a failed build left new functions behind an old frontend.
  - Fix: the push moved to `post_rebuild.sh`, which runs only after every compose step passed.
  - Evidence: `test_failed_build_never_runs_post_rebuild` and `test_pre_and_post_rebuild_wrap_the_build`.
- Finding (medium): the push skipped `safe_docker_build.sh`'s main-checkout and migration gates.
  - Fix: same move; post runs only after the gated `up` succeeded.
  - Evidence: same tests.
- Finding (low-medium): the backup missed staged edits and the contents of untracked files.
  - Fix: `git diff HEAD`, plus untracked files copied into `untracked/`.
  - Evidence: `test_rebuilds_a_hand_edited_tree_to_the_fresh_chain` checks both.
- Finding (low): `checkout REF -- .` behaves badly when upstream HEAD isn't the pin.
  - Fix: refuse with the exact commands to run.
  - Evidence: `test_refuses_when_upstream_head_is_not_the_pin`.
- Finding (low): an exported override silently replaced the deploy.
  - Fix: renamed `AITOWN_TEST_CONVEX_PUSH`; it prints "NOT pushing functions".
  - Evidence: `test_post_rebuild_pushes_from_upstream`.
- Finding (low): `head -n -10` is GNU-only.
  - Fix: `sort -r | tail -n +11`.
  - Evidence: script.
- Finding (test weakness): no proof the hand edit is gone, the bare-apply claim was unchecked, and dry-run was uncovered.
  - Fix: the test compares the tree to the fresh chain, runs the bare apply script, and checks dry-run.
  - Evidence: tests.

## Restart required

No restart. The next pull on circe that touches `services/orion-ai-town` deploys through the hooks.

## Risks / concerns

- Severity: low. Concern: a hand edit in `upstream/` is discarded on every town rebuild. Mitigation: the last 10 backups are kept under `~/.orion/`, and the README says patches are the only source.
- Severity: low. Concern: `pre_rebuild.sh` rewrites `upstream/` even when `up` is later refused. Mitigation: `upstream/` is generated, and the running containers are unaffected until a build succeeds.

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
