# orion-docker-door

A filtering proxy in front of the Docker socket for Orion's turns.
Design: `docs/superpowers/specs/2026-11-24-orion-safe-autonomy-in-place-design.md`.

## What it does

- **Passes looking:** every `GET`/`HEAD` (`docker ps`, `logs`, `inspect`, `stats`, `info`, `events`).
- **Passes exec** (`docker exec`), unless the target container leads back to the host: it mounts
  the Docker socket, runs privileged, uses `pid: host`, or bind-mounts `/`, `/run`, `/var/run` or
  `/var/lib/docker`. The door inspects the target live on every exec, so no list to keep.
- **Refuses changing containers:** every other `POST`/`PUT`/`DELETE` (run, create, start, stop,
  restart, rm, build, pull, compose up, prune, ...) and every attach. The CLI prints
  "Orion's Docker door: changing containers is off".
- **Refuses callers** other than `DOCKER_DOOR_ALLOWED_CALLERS` (default: the harness governor).
- **Records every request** to `/mnt/storage-warm/orion-record/docker-YYYY-MM-DD.jsonl` on the
  host (time, caller, method, path, container, exec command and user, decision, reason). Orion's
  turns cannot exec into this container (it holds the socket), so they cannot change the record.

One request per connection: the door forwards with `Connection: close`, so a second request
cannot ride an already-checked connection.

## Canary

Orion's turns use the door only inside a window set in the governor:
`ORION_DOCKER_DOOR_CANARY_WINDOWS=2026-11-26T15:00Z/2026-11-26T19:00Z`
(`services/orion-harness-governor/.env_example`). Outside the window, turns use the real socket. If
the door does not answer, turns keep the socket and the governor logs `DOCKER_DOOR_CANARY ...
door=unreachable`. The GitHub MCP server is pinned to the real socket (it starts a container per
turn, which the door would refuse); that is harness code, not Orion's shell.

What broke in a window:

```bash
python3 services/orion-docker-door/app/report.py --since 2026-11-26T15:00Z --until 2026-11-26T19:00Z
```

Governor side: `docker logs orion-athena-harness-governor 2>&1 | grep DOCKER_DOOR_CANARY`.

## Run

Stdlib only, no requirements. `scripts/safe_docker_build.sh orion-docker-door up -d --build`
(from the clean primary main checkout).

## Tests

`python3 -m pytest services/orion-docker-door/tests -q`
