## Summary

- The town's post-pull deploy on circe failed after #2978 merged. The image build type-checks `convex/crons.ts`, and the new cron calls `internal.sim.tick.run`. The generated API types were still the pin's copy, which has no `sim` module.
- `pre_rebuild.sh` now runs `convex codegen` after the patch chain. Codegen rewrites `convex/_generated/` for the patched tree and does not change the deployment.
- `pre_rebuild.sh` and `post_rebuild.sh` share a new `scripts/run_convex.sh`. It runs `npx convex`, or a `node:20-alpine` container on a host without Node (circe).
- Every future patch that adds a Convex module would have failed the deploy in the same way.

## Outcome moved

- Failure mode: post-pull rebuild `failed=orion-ai-town` (circe `~/.orion/post_pull_rebuild.log`, 2026-10-29T23:41:46Z, `rc=1` in 9 s). Because the build failed, `post_rebuild.sh` correctly did not run. The live town kept its old frontend and old functions, and the simulation core from #2978 never reached Convex.

## Current architecture

`rebuild_services_from_git_diff.sh` runs `pre_rebuild.sh`, which resets `upstream/` to the pin and applies the chain. It then runs the compose build (`npm run build` = `tsc && vite build`). Only after the build does it run `post_rebuild.sh` (`npx convex dev --once`). `convex/_generated/` is tracked in upstream's git, so the reset restored the pin's copy, and only `convex dev` regenerated it, which was too late for `tsc`.

## Architecture touched

AI Town deploy hooks only. No bus, schema, env or service-code changes.

## Files changed

- `services/orion-ai-town/scripts/run_convex.sh` (new): runs `npx convex <args>` in `upstream/`, with the docker fallback; test hook `AITOWN_TEST_CONVEX`.
- `services/orion-ai-town/scripts/pre_rebuild.sh`: starts the `backend` container (`safe_docker_build.sh ... up -d --wait backend`, a no-op when it runs), then runs `codegen --typecheck disable` after the chain.
- `services/orion-ai-town/scripts/post_rebuild.sh`: uses `run_convex.sh` (same push).
- `services/orion-ai-town/tests/test_pre_rebuild.py`: asserts that codegen runs on the patched tree (sim overlay present), on every run, and that a codegen failure fails the hook. It also asserts that a failed chain runs no codegen, and that post still pushes.
- `.github/workflows/aitown-patch-chain.yml`: triggers on the post hook and the runner too.
- `services/orion-ai-town/README.md`: step 4 of `pre_rebuild.sh`.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: test hook `AITOWN_TEST_CONVEX_PUSH` → `AITOWN_TEST_CONVEX` (tests only).
- Behavior changed: `pre_rebuild.sh` now starts the Convex backend container if it is down, because codegen reads it.
- Compatibility notes: none.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest services/orion-ai-town/tests/test_pre_rebuild.py (AITOWN_UPSTREAM_GIT=local clone) -> 6 passed
pytest tests/scripts/test_rebuild_services_from_git_diff.py -> 50 passed
mutation: codegen line replaced with `true` -> 2 failed
git diff --check -> clean
```

## Evals run

None. This is a deploy-ordering fix; the live check is the post-merge rebuild below.

## Docker/build/smoke checks

```text
circe, by hand, on the real upstream/:
  safe_docker_build.sh orion-ai-town build  -> FAIL: crons.ts(19,53) Property 'sim' does not exist
  npx convex codegen --typecheck disable   -> api.d.ts now lists sim/ (12 lines)
  npx convex run sim/state:snapshot        -> still "Could not find function" (codegen pushed nothing)
  safe_docker_build.sh orion-ai-town build  -> Image orion-ai-town-frontend Built
```

## Review findings fixed

- Finding (medium): if the backend was down, codegen failed, so `pre_rebuild.sh` failed, the stack was skipped, and `up` never started the backend. Every later pull would fail the same way.
  - Fix: `pre_rebuild.sh` runs `up -d --wait backend` through `safe_docker_build.sh` before codegen. The post-pull run exports `ORION_DEPLOY_PINNED_HEAD`, so the nested `up` passes the deploy gate.
  - Evidence: `test_rebuilds_a_hand_edited_tree_to_the_fresh_chain` asserts backend-up before each codegen. 6 passed.
- Finding (not a bug): README line "It runs after `up`" was ambiguous after the runner sentence.
  - Fix: names `post_rebuild.sh`.

## Restart required

None by hand. Merging this PR changes `services/orion-ai-town/`, so the post-pull rebuild on circe reruns `pre_rebuild.sh`, then the build and `up`, then `post_rebuild.sh`, which pushes the sim functions. Check:

```bash
tail -8 ~/.orion/post_pull_rebuild.log   # on circe: POST-END rc=0 orion-ai-town
```

## Risks / concerns

- Severity: low. Concern: the backend-up step was not run by hand on circe (it is a production `up`). Mitigation: the post-merge rebuild runs it; `PRE-END rc=0` in the log proves it.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2982

🤖 Generated with [Claude Code](https://claude.com/claude-code)
