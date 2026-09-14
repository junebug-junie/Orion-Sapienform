# Orion Analytics post-merge rebuild profile fix

## Status

`READY_FOR_REVIEW`

## Summary

The post-merge rebuild runner called every ordinary service as:

```text
safe_docker_build.sh <service> up -d --build
```

Every long-running `orion-analytics` service is intentionally gated behind the
Compose `analytics` profile, so that invocation selected no services and failed
with `no service selected` after PR #2214 merged.

The runner now gives `orion-analytics` its required profile:

```text
safe_docker_build.sh orion-analytics --profile analytics up -d --build
```

The separate `analytics-dbt` one-shot tool remains behind the `tools` profile
and is not started as a daemon.

## Regression coverage

A behavioral shell-runner test executes the rebuild script inside a temporary
fake repository, selects `orion-analytics`, captures the safe-build invocation,
and requires the exact profile-aware argument order. This tests execution, not
only a source-code substring.

## Live proof

- Reproduced before the fix: Compose exited `1` with `no service selected`.
- Ran the fixed command through `scripts/safe_docker_build.sh`: exited `0`.
- Lightdash was recreated and reached Docker health `healthy`.
- Final recreation was run from the primary post-merge checkout; the live dbt
  mount resolves to `/mnt/scripts/Orion-Sapienform/services/orion-analytics`,
  not a temporary worktree.
- `GET /api/v1/health`: `healthy=true`, `requiresMigration=false`.
- The deployed Curiosity dashboard route returned HTTP `200` after rebuild.

## Checks

- focused rebuild/safe-wrapper tests: `4 passed`
- `bash -n scripts/rebuild_services_from_git_diff.sh`: pass
- `git diff --check`: pass

## Review findings fixed

The adversarial review found no material issues. It confirmed the Compose
argument order, shared-checkout escape propagation, and unchanged command path
for every other service. Its non-blocking suggestion was broader coverage of
the already-existing ordinary-service/shared-checkout branches; this patch
keeps its regression focused on the reproduced analytics failure.

## Existing unrelated test debt

The broader `tests/scripts/test_rebuild_affected_services.py` suite has five
failures on current `main` because its expected import/path classifications no
longer match the live repository graph. This patch does not change the
classifier. The new command-routing regression and the existing safe-wrapper
tests pass.
