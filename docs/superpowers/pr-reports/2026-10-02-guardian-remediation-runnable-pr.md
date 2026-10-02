# fix(mesh-guardian): make auto-remediation actually runnable

## Summary

- **Nothing could run.** The guardian image had no `docker` CLI: Debian trixie's `docker.io` package stopped shipping it. Every remediation (`docker compose up --force-recreate` / `build`) would fail with `docker: not found`. This was dormant only because `MESH_GUARDIAN_AUTO_REMEDIATE=false`.
- **The image now installs `docker-cli`, `docker-compose` (the v2 plugin), and `docker-buildx`.** The image build fails if compose or buildx is missing.
- **The repo is now mounted at its host path, and `ORION_REPO_ROOT` points there.** Compose talks to the host daemon, so from `/repo` it would have handed the daemon `/repo/...` paths and relabelled every recreated container's `working_dir`.
- **Remediation now refuses services that were deployed from a worktree.** It checks the container's `com.docker.compose.project.working_dir` label. Without this, a recreate or rebuild from the shared checkout would silently revert the worktree deploy, which is the incident `scripts/safe_docker_build.sh` exists to prevent. The check fails closed if Docker is unreadable, and the card names the deploy origin.
- **The guardian checks Docker at startup when auto-remediation is enabled.** If the CLI can't reach the daemon (for example a client/daemon API-version gap), it logs the error at boot instead of mid-incident.

## Outcome moved

Before, remediation could not run at all. After, every check below was verified live from inside the deployed guardian.

## Current architecture

`app/remediator.py` shells `docker compose` against the host daemon through the mounted socket. Tier 1 recreates the container; tier 2 builds the image, then brings the service up.

## Files changed

- `services/orion-mesh-guardian/Dockerfile`: docker CLI, compose plugin, and buildx, verified at build.
- `services/orion-mesh-guardian/docker-compose.yml`: host-path repo mount and `ORION_REPO_ROOT`.
- `services/orion-mesh-guardian/app/remediator.py`: `deployed_from_elsewhere` guard and `docker_cli_selfcheck`.
- `services/orion-mesh-guardian/app/service.py`: self-check at startup; stale comment fixed.
- `services/orion-mesh-guardian/tests/test_remediation_runtime.py`, `tests/test_remediator_exec.py`: regression tests.

## Schema / bus / API changes

None.

## Env/config changes

None. `ORION_HOST_REPO_ROOT` already exists in the root `.env`. The compose default covers the case where it is unset.

## Tests run

```text
pytest services/orion-mesh-guardian/tests services/orion-mesh-guardian/evals -q -> 78 passed
```

## Evals run

`evals/test_stability_incident_replay.py` passes. It is unchanged; this patch adds no new eval.

## Docker/build/smoke checks

```text
image: Docker 26.1.5, Docker Compose 2.26.1, buildx present (build-time check)
startup selfcheck inside deployed guardian: ok
compose --dry-run tier-1 for all 5 auto_remediate roster services (cortex-gateway, cortex-orch,
  cortex-exec, recall, llm-gateway): exit 0, each "Recreate"s its existing container
real `docker buildx build --load` of a throwaway image from inside the guardian: ok (image removed)
deploy-origin guard: 5/5 roster services -> allowed (deployed from main);
  orion-mesh-guardian itself (deployed from this worktree) -> refused, origin named
```

## Review findings fixed

- **Finding (high):** remediation from the shared checkout would revert worktree deploys.
  - **Fix:** the `deployed_from_elsewhere` guard, which fails closed.
  - **Evidence:** unit tests, plus the live refusal of the worktree-deployed guardian.
- **Finding (medium):** the tier-2 build was unverified because buildx was missing.
  - **Fix:** `docker-buildx` installed.
  - **Evidence:** a real throwaway build from inside the container.
- **Finding (low):** a client/daemon API gap would go unnoticed until a remediation failed.
  - **Fix:** startup self-check.
  - **Evidence:** `test_selfcheck_reports_an_unreachable_daemon`, and the live self-check reads `ok`.
- **Finding (none):** a stale `/repo` comment.
  - **Fix:** updated.

## Restart required

Deployed from this worktree. After merge:

```bash
scripts/safe_docker_build.sh orion-mesh-guardian up -d --build
```

Auto-remediation stays off (`MESH_GUARDIAN_AUTO_REMEDIATE=false`) until Juniper turns it on.

## Risks / concerns

- **Medium: llm-gateway's live container is named `orion-llm-gateway`.** It was deployed with `PROJECT=orion`, not `orion-athena`, which predates this change. Any recreate, whether by remediation or by hand, renames it to `orion-athena-llm-gateway`. Check that nothing addresses it by the old name before enabling remediation.
- **Low: tier 2 builds from the shared checkout.** If main has uncommitted dirt, the build bakes it in. The deploy-origin guard does not cover that case.
- **Low: the docker CLI is 26.1.5 (API 1.45) and the host daemon is 29.1.3 (minimum API 1.44).** That is a narrow margin. The self-check logs the error if a future daemon upgrade breaks it.
