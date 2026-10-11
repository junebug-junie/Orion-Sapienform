## Summary

Container logs now go to the host journal, so they are still there after `docker compose up` replaces a container. That is what was missing when the 10-08 reading failure could not be diagnosed.

- Every service in every `services/*/docker-compose*.yml` (122 services, 99 files) gets the same 4-line `logging:` block: `driver: journald`, `tag: "{{.Name}}"`.
- New static gate `tests/test_compose_logging_journald.py`, run in `orion-static-gates` CI, fails if any service in any git-tracked compose file lacks the block.
- Host drop-in `deploy/systemd/journald.conf.d/orion-container-logs.conf` raises the journal cap to 24G. Juniper installs it; nothing on the host was changed.
- `docs/operations/container-logs.md` explains how to read a recreated container's old logs; one line added to `services/orion-hub/README.md`.

## Outcome moved

Before: recreating a container deleted its whole log history. After: `journalctl CONTAINER_NAME=<name>` shows lines from every earlier instance of that name, up to the journal's size cap.

## Current architecture

No compose file set `logging:`. The daemon default is `json-file` with no `max-size` (`/etc/docker/daemon.json` sets only data-root and the nvidia runtime). Each log lives in `/mnt/docker/containers/<id>/<id>-json.log` and is deleted with the container.

Verified live on athena (Docker 29.1.3):

- `docker restart`: same container ID, log kept (`first-boot-json` printed twice).
- `docker rm` + `run` with the same name (what a compose recreate does): only `second-boot-json` remained.
- Same test with `--log-driver journald`: `docker logs` showed only the current boot, and `journalctl CONTAINER_NAME=orion-logtest-jd` showed both `first-boot-journald` and `second-boot-journald`.

Also found: json-file logs were unbounded. They held about 60 GB in total on 2026-10-10. One orion-signal-gateway container had 32 GB in 16 days, from a per-message `Hunter intake` INFO line.

## Why journald

- It survives a recreate, which `json-file` and `local` do not.
- Docker ships the driver, and journald was already running with persistent storage (`/var/log/journal`, 2.3G, history back to 2026-07-23).
- Its size is capped.
- `docker logs` and `docker compose logs` still work, because journald is a driver Docker can read back natively. Dual logging (Docker 20.10+) covers drivers that can't. Hub's `service_logs.py` and the scripts that call `docker logs` keep working.

No log-shipping stack was built.

## Architecture touched

- Compose `logging:` on all services. Takes effect per container at its next recreate.
- CI static gate.
- Host config templates (not installed).

## Files changed

- `services/*/docker-compose*.yml` (99 files): journald logging block on each service.
- `tests/test_compose_logging_journald.py`: the gate.
- `.github/workflows/orion-static-gates.yml`: runs the gate.
- `deploy/systemd/journald.conf.d/orion-container-logs.conf`: `SystemMaxUse=24G`, `SystemKeepFree=20G`.
- `docs/operations/container-logs.md`: how to read the logs, plus the evidence.
- `services/orion-hub/README.md`: one-line pointer.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: container log storage only.
- Compatibility notes: a running container keeps json-file until its next recreate. On that recreate its existing json history is deleted, the same as before this change.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest tests/test_compose_logging_journald.py -q                    2 passed
  red check: hub compose reverted to origin/main -> 1 failed naming
  services/orion-hub/docker-compose.yml::hub-app; restored -> passed
docker compose -f <each of 99 files> config --no-interpolate --format json
  -> 99/99 render, every service logging == journald/{{.Name}}
compose-touching tests (ai-town, cortex-exec bringup, falkordb, recall,
  dsv41, bonsai, kev, bus, equilibrium, field-digester, root check_* tests): pass
tests/scripts/test_rebuild_affected_services.py::test_sample_pull_diff: fails
  identically on main (pre-existing, unrelated)
```

## Evals run

```text
No eval harness applies (deployment config). Live recreate experiment above stands in.
```

## Docker/build/smoke checks

```text
Throwaway containers (python:3-alpine) for json-file vs journald restart/recreate; removed afterward.
No Orion service was rebuilt, restarted or recreated.
```

## Review findings fixed

- Finding: the docs claimed the journald rate limit was 10000 lines / 30s and that bursts would drop lines. journald scales the burst with free disk space, so on athena it is about 50000, and the fleet averages about 4000. The proposed docker.service drop-in also needed a Docker daemon restart, and that stops every container (live-restore is off).
  - Fix: deleted the drop-in. The docs now give the real limit and say how to spot suppressed lines.
  - Evidence: `deploy/systemd/docker.service.d/` removed; see `docs/operations/container-logs.md`, section "Retention".
- Finding: the 24G cap is shared by the whole fleet, so signal-gateway's noise evicts every other container's history first. journald may also take more space per line than json-file.
  - Fix: both caveats are now in the docs and in the risks below.
  - Evidence: `docs/operations/container-logs.md`.
- Finding: the gate globbed the filesystem under `services/*/` only. An untracked local override could fail it, and a compose file outside `services/` would never be checked.
  - Fix: the gate now uses `git ls-files` across the whole repo and excludes vendored `*/upstream/` code.
  - Evidence: the gate still finds 99 files and passes 2/2.
- Finding (doc nits): lines over 16 KB are split into several journal entries; containers started with plain `docker run` still use json-file; other hosts need the persistent-journal drop-in too.
  - Fix: one note for each added to the docs.
- Finding: the PR report was untracked.
  - Fix: committed with this push.
- The reviewer counted 100 compose files. That number includes the hub README in the diff; `git ls-files` shows 99. No change needed.

## Restart required

Nothing changes until containers are recreated. Juniper, when convenient:

```bash
sudo install -Dm644 deploy/systemd/journald.conf.d/orion-container-logs.conf /etc/systemd/journald.conf.d/orion-container-logs.conf && sudo systemctl restart systemd-journald
```

After merge, a normal per-service `scripts/safe_docker_build.sh <service> up -d` recreate switches that container to journald.

## Risks / concerns

- Severity: medium
  - Concern: retention. The fleet wrote about 2.4 MB/min of json-file output (~3.4 GB/day). At journald's default cap (4G) that is about one day of history, and it would push system logs out early.
  - Mitigation: install the 24G drop-in, which gives roughly a week. That figure is estimated from json-file growth; check `journalctl --disk-usage` after a day. Separate follow-up: signal-gateway's per-message INFO line is about 60% of the volume.
- Severity: low
  - Concern: rate limit. All containers share docker.service's journal limit, which is about 50000 lines / 30s at athena's free disk. The fleet averages about 4000.
  - Mitigation: none needed today. Suppressed lines would show up in `journalctl -u systemd-journald`.
- Severity: low
  - Concern: circe, atlas and hecate overlays use the same block. Those hosts were not checked (no SSH from athena). A host without systemd-journald would refuse to start the container.
  - Mitigation: run `systemctl is-active systemd-journald` on each host before recreating there. UNVERIFIED.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2602

🤖 Generated with [Claude Code](https://claude.com/claude-code)
