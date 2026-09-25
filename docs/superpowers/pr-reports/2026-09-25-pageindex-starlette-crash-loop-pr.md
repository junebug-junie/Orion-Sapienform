## Summary

- The PageIndex journal-search service had been crash-looping since 2026-08-30 (about 4,800 restarts) and was never up. It now starts and stays up.
- Cause: the Docker build installed our service's pinned packages, then installed the upstream PageIndex project's own requirements into the same Python on top. Upstream PageIndex (cloned from its unpinned `main`) pulls in `mcp>=1.19,<3`, which upgraded `starlette` from 0.41 to 1.6 underneath our pinned `fastapi==0.115.6`. starlette 1.x removed the `on_startup` argument that this fastapi version passes, so the app died at import with `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'`.
- Fix: the service now runs from its own virtualenv (`/opt/orion-venv`), so PageIndex's packages cannot reach it. PageIndex itself still runs from the system `python3`, which is how the service already calls it (as a subprocess).
- Isolating the venv exposed one more accidental dependency: `pyyaml` (needed by `orion.core.bus`) had only ever arrived via PageIndex's requirements. It is now pinned explicitly.
- The build now imports `app.main` from the service venv at the end. A future version conflict fails `docker build` instead of producing a restart loop.

## Outcome moved

`orion-athena-pageindex`: was crash-looping (RestartCount 4,793); now `running`, RestartCount 0, `/healthz` answers 200, and SystemHealthV1 heartbeats show up on `orion:system:health`.

## Current architecture

A single-stage `python:3.11-slim` image. It ran `pip install -r requirements.txt`, then `git clone PageIndex@main` and `pip install -r /opt/PageIndex/requirements.txt` into the same interpreter, then `uvicorn app.main:app`. The service calls PageIndex only through a subprocess (`PAGEINDEX_PYTHON_BIN=python3 run_pageindex.py`, `app/pageindex_cli.py`) and never imports it in-process.

## Architecture touched

Only the `orion-pageindex` build. No contract, bus, schema, or env changes.

## Files changed

- `services/orion-pageindex/Dockerfile`: installs service deps into `/opt/orion-venv` and runs `pip check`; CMD uses `/opt/orion-venv/bin/uvicorn`; PATH is left unchanged so `python3` still resolves to the system interpreter that has PageIndex; adds a build-time `import app.main` guard.
- `services/orion-pageindex/requirements.txt`: adds `pyyaml==6.0.2`.
- `services/orion-pageindex/tests/test_dockerfile_dependency_isolation.py`: new static regression tests for the isolation, CMD, PATH, guard placement, PageIndex CLI guard, `PAGEINDEX_PYTHON_BIN`, and pyyaml.
- `.github/workflows/orion-pageindex-build-tests.yml`: runs those tests on PRs that touch the service.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none (the service now actually runs)
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none. Note: the build wrapper warns that the local `services/orion-pageindex/.env` is missing `HEARTBEAT_INTERVAL_SEC`, `ORION_BUS_ENABLED`, `ORION_BUS_URL`. This was already true before this patch, and the compose defaults cover all three.

## Tests run

```text
PYTHONPATH=services/orion-pageindex:. .venv/bin/python -m pytest services/orion-pageindex/tests -q
  -> 27 passed, 1 failed (test_service.py::test_chat_episodes_rebuild_and_query)
The single failure predates this patch: it fails the same way on main without these changes
("PageIndex query is not supported by current runner configuration; set PAGEINDEX_QUERY_ARGS
with a {query} placeholder"). This patch touches no Python code under app/.

New regression tests against the OLD Dockerfile: 4 of 5 fail (the pyyaml test was added later).
All 8 pass against the new one.
```

## Evals run

```text
No eval harness exists for orion-pageindex. This is a build/packaging fix; the live smoke below
is the behavioral proof. Follow-up: a retrieval eval for the journals corpus once the DB config
gap (Risks) is resolved.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-pageindex build
  #14 RUN /opt/orion-venv/bin/python -c "...; import app.main; ..."
  app.main import ok: fastapi 0.115.6 starlette 0.41.3
  PageIndex CLI ok under /usr/local/bin/python3
venv:   fastapi==0.115.6 starlette==0.41.3 pydantic==2.10.3
system: starlette==1.7.0 mcp==2.2.0 litellm==1.97.0  (PageIndex's, isolated)

Guard proven to catch the original bug: installing starlette==1.6.0 into the venv and running
the same guard command reproduces
  TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'

scripts/safe_docker_build.sh orion-pageindex up -d --build
  docker inspect: running restarts=0; Mounts = named volume orion-pageindex_pageindex-data only (no bind mounts)
  curl localhost:8360/healthz -> HTTP 200
  logs: "Application startup complete", "system_health_heartbeat_started", 0 errors/tracebacks
  redis SUBSCRIBE orion:system:health -> orion-pageindex heartbeats observed
make substrate-ladder-check (primary checkout) -> GREEN, no CANNOT CHECK lines
```

## Review findings fixed

The review subagent found no must-fix issues. It independently confirmed in the built image that the venv imports every runtime module (including the lazy RPC-timeout imports), that `python3` resolves to the system interpreter and cannot import fastapi, that `run_pageindex.py --help` works there, and that `PAGEINDEX_PYTHON_BIN=python3` is the value on every config surface.

- Finding (should): the new regression tests ran in no CI workflow.
  - Fix: added `.github/workflows/orion-pageindex-build-tests.yml`, path-filtered to `services/orion-pageindex/**`. It runs only the stdlib static test file, because `test_service.py` has the pre-existing failure.
  - Evidence: the workflow ran on this PR (see CI).
- Finding (should): nothing verified at build time that the PageIndex side still runs.
  - Fix: added `RUN cd /opt/PageIndex && python3 run_pageindex.py --help` after the app import guard.
  - Evidence: build log `PageIndex CLI ok under /usr/local/bin/python3`.
- Finding (should): `PAGEINDEX_REF=main` is unpinned, and the `sed` edit on the dotenv pin fails silently.
  - Fix: not changed. Recorded under Risks. It can no longer crash the service, only fail a build or change PageIndex's own behavior.
- Finding (nit): the PATH test missed a multi-variable `ENV A=1 PATH=...` line.
  - Fix: now checks every `ENV` line for `PATH`.
  - Evidence: mutating the Dockerfile to `ENV ORION_VENV=/opt/orion-venv PATH=/opt/orion-venv/bin:$PATH` fails the test (1 failed, 7 passed).
- Finding (nit): nothing pinned `PAGEINDEX_PYTHON_BIN=python3`.
  - Fix: a test now asserts it in `.env_example`, compose, and `settings.py`.
- Finding (nit): a garbled test name.
  - Fix: renamed to `test_service_reqs_only_installed_via_venv_pip`.
- Not changed (nits, pre-existing): `loguru` is absent from the venv, so the chassis falls back to stdlib logging (boot logs confirm it works). `pytest` still ships in the runtime image. The subprocess inherits `PYTHONPATH=/app`.

## Restart required

```text
No restart required. Already redeployed from this worktree (no bind mounts, so the running
container does not depend on the worktree). Once this merges, redeploying from main just refreshes
the compose working_dir label:
scripts/safe_docker_build.sh orion-pageindex up -d --build
```

## Risks / concerns

- Severity: medium (pre-existing, not caused by this patch)
  - Concern: `/healthz` returns `ok:false` because none of `PAGEINDEX_SQL_DATABASE_URL`, `ENDOGENOUS_RUNTIME_SQL_DATABASE_URL`, `SQL_DATABASE_URL` is set in the local `.env`. `JOURNAL_PG_DSN` is set, but `service._resolve_db_dsn()` never reads it. The process is up, but journal rebuilds will fail with "database URL missing" until an operator sets one of those keys, or a follow-up makes the resolver fall back to `JOURNAL_PG_DSN`.
  - Mitigation: left for Juniper to decide. Turning on DB access for a service that has been down for a month changes live recall behavior, so it should not ride along with a packaging fix.
- Severity: low
  - Concern: `PAGEINDEX_REF=main` still clones an unpinned upstream at build time, so PageIndex's own behavior can drift between builds.
  - Mitigation: it can no longer break the service process. Pinning a commit is a sensible follow-up.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
