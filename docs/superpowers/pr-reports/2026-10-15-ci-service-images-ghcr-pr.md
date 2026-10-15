## Summary

- The CI failures on 2026-10-15 were **not** GHCR. Every failing job died pulling `public.ecr.aws/docker/library/postgres:15/16` with `toomanyrequests: Data limit exceeded` (AWS ECR Public's anonymous data cap). In the substrate job the private GHCR falkordb image pulled fine in the same job, seconds earlier.
- Copied `postgres:15` and `postgres:16` once to the private package `ghcr.io/junebug-junie/postgres` (linked to this repo, same as the falkordb copy from 2026-10-09), and switched all 12 postgres service containers to it, pulled with `GITHUB_TOKEN`.
- Added `permissions: {contents: read, packages: read}` to each job that pulls the private copy.
- New static gate `tests/test_ci_service_image_sources.py` (run in `orion-static-gates`): every service image must come from `ghcr.io/`, and any `ghcr.io/junebug-junie/*` image must have credentials and `packages: read`.

## Outcome moved

Postgres-backed CI jobs no longer depend on an anonymous pull quota that is shared with every other GitHub runner on the same IP. This is the second time in a week (Docker Hub on 10-09, ECR Public on 10-15) that an anonymous public registry failed jobs before any test ran.

## Current architecture

10 workflows run postgres as a job service container. Commit `017a0c892` (10-09) moved them off Docker Hub to ECR Public because of Docker Hub's anonymous rate limit. Commit `5a6a35af2` moved the falkordb images to a private GHCR copy for the same reason. ECR Public then hit its own anonymous limit.

## Evidence (attempt-1 job logs)

| Job id | Workflow | Failing pull |
|---|---|---|
| 116239688611 | orion-attention-runtime-tests | `public.ecr.aws/docker/library/postgres:15` -> `Data limit exceeded` |
| 116239688697 | Substrate neighborhood reads | `ghcr.io/junebug-junie/falkordb:v4.18.11` **succeeded**, then `public.ecr.aws/.../postgres:16` -> `Data limit exceeded` |
| 116267100633 | orion-durable-runs tests | `public.ecr.aws/.../postgres:16` -> `Data limit exceeded` |
| 116274835767 | orion-gpu-pool tests (stage 4.4) | `public.ecr.aws/.../postgres:16` -> `Data limit exceeded` |

## Image inventory (service containers; no workflow uses `docker pull`/`docker run`)

| Image (after) | Was | Owner / visibility | Jobs pulling it per triggering push |
|---|---|---|---|
| `ghcr.io/junebug-junie/postgres:16` | ECR Public `postgres:16` | ours, private (GHCR default for new packages) | 9: memory-schema-bootstrap 1, memory-episode 1, sql-writer-integration 1, durable-runs unit 1, substrate-neighborhood 2 (matrix), exo-exploration-integration 1, gpu-pool pool + consumers 2 |
| `ghcr.io/junebug-junie/postgres:15` | ECR Public `postgres:15` | ours, private | 3: attention-runtime 1, regulation-history 1, visual-baseline 1 |
| `ghcr.io/junebug-junie/falkordb:v4.18.11`, `:6.0.1` | unchanged | ours, private | 1 each (substrate-neighborhood matrix) |
| `ghcr.io/falkordb/falkordb:latest` | unchanged | FalkorDB's own public GHCR | 1 (memory-episode) |

All ten workflows already have `paths:` filters. Recent volume (since 10-14): about 7-11 runs per postgres-backed workflow, i.e. tens of ~150 MB pulls a day, nowhere near ECR Public's 500 GB/month anonymous cap. The cap was exhausted by the **shared runner IP**, not by this repo, so cutting our own pull count (concurrency cancellation, dedupe) would not have prevented these failures. Not done, to keep the patch thin. Expected pull reduction per push: zero; the change is which registry the pulls go to, not how many there are.

Copy provenance: `FROM docker.io/library/postgres@sha256:c961aa28...842c` (15) and `@sha256:ca0bd484...641d` (16), amd64, plus an `org.opencontainers.image.source` label linking the package to this repo. Pushed tags: `ghcr.io/junebug-junie/postgres:15` (`sha256:40127a3e...`), `:16` (`sha256:54bfba50...`).

## Files changed

- `.github/workflows/{memory-schema-bootstrap,orion-attention-runtime-tests,orion-durable-runs-tests,orion-exo-exploration-tests,orion-gpu-pool-tests,orion-memory-episode-tests,orion-sql-writer-tests,regulation-history,substrate-neighborhood,visual-baseline}.yml`: postgres image -> GHCR copy + credentials + job `packages: read`.
- `.github/workflows/orion-static-gates.yml`: runs the new gate.
- `tests/test_ci_service_image_sources.py`: the gate.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none. Behavior changed: CI only. Compatibility notes: none.

## Env/config changes

- None. No `.env_example` touched.

## Tests run

```text
pytest tests/test_ci_service_image_sources.py -q            -> 3 passed
negative check (scratch copy: one image back to ECR Public, one job's permissions removed) -> 2 failed, naming both
python -c "yaml.safe_load(...)" .github/workflows/*.yml     -> all parse
actionlint                                                  -> not installed on host
```

## Evals run

```text
None; CI-config change. The live check is the PR's own CI: every postgres-backed job that its path filters trigger must pull from GHCR.
```

## Docker/build/smoke checks

```text
docker build + push of both copies to ghcr.io/junebug-junie/postgres: succeeded (digests above).
```

## Review findings fixed

- See PR conversation.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: the copies are pinned. Upstream postgres patch releases no longer arrive automatically. Mitigation: re-push the tag when a bump is wanted (same as falkordb).
- Severity: low. Concern: a job that pulls the private copy with no `packages: read` fails at "Initialize containers". Mitigation: the new static gate refuses that.
- Severity: decision for Juniper. Private GHCR packages count against the account's package storage (two images, a few hundred MB). Pulls from Actions using `GITHUB_TOKEN` are free and do not use the data-transfer allowance. Making `postgres`/`falkordb` public would remove even the storage accounting, but would publish them under the account. Left private, matching the falkordb precedent.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
