# orion-docker-door

A filtering proxy in front of the Docker socket for Orion's turns.
Design: `docs/superpowers/specs/2026-11-24-orion-safe-autonomy-in-place-design.md`.

## What it does

- **Passes looking:** every `GET`/`HEAD` (`docker ps`, `logs`, `inspect`, `stats`, `info`, `events`).
  Copying files out (`docker cp` from a container, `docker export`) reads through binds, so it is
  checked like an exec. `docker save` (`/images/get`) is refused.
- **Passes exec** (`docker exec`), unless the target container leads back to the host:
  - privileged; host PID, network, IPC, user, UTS or cgroup namespace; or a `container:<x>`
    namespace where x leads back to the host;
  - added capabilities, devices, device cgroup rules, or an `unconfined` security option;
  - a runtime socket (`docker.sock`, `containerd.sock`, any `*.sock` under `/run` or `/var/run`),
    or any other bind under `/run`/`/var/run` except `/run/orion-audio` and `/run/orion-sensors`;
  - a bind of `/`, or of anything under `/proc`, `/sys`, `/dev`, `/etc`, `/root`, or Docker's data
    root (read from the daemon; `/mnt/docker` here) other than named volumes;
  - a **writable** bind under `/mnt/scripts`, `/home`, `/usr`, `/var`, `/opt`, `/boot`, `/srv`,
    `/lib*`, `/bin`, `/sbin` or the record folder (code the host runs, or the record itself);
  - a local volume made with `o=bind` counts as a bind of its `device`.

  The door inspects the target live on every exec, so there is no list to keep. An exec that asks
  for `privileged` (any letter case, as Docker reads it) is refused too.
- **Refuses changing containers:** every other `POST`/`PUT`/`DELETE` (run, create, start, stop,
  restart, rm, build, pull, compose up, prune, ...) and every attach. The CLI prints
  "Orion's Docker door: changing containers is off".
- **Refuses callers** other than `DOCKER_DOOR_ALLOWED_CALLERS` (default: the harness governor).
- **Records every request** to `/mnt/storage-warm/orion-record/docker-YYYY-MM-DD.jsonl` on the
  host (time, caller, method, path, container, exec command and user, decision, reason). Orion's
  turns cannot exec into this container (it holds the socket), so they cannot change the record.

Strict by design (code review, 2026-11-24): the door reads the whole request before it decides,
and refuses (400) anything it cannot read exactly: a bare LF or stray CR in the head, folded
headers, any version but HTTP/1.0 and 1.1, `Transfer-Encoding`, `Expect`, a missing or bad
`Content-Length` on an exec create, a body over 1 MB, an encoded or dotted path. It forwards
exactly that one request, always with `Connection: close`. Later bytes from the client reach the
daemon only after a non-detached exec start, where Docker has already turned the connection into
that exec's stdin. Slow or broken clients are dropped by read timeouts. If the record cannot be
written, nothing passes (503).

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
