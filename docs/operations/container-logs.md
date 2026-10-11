# Container logs that survive a recreate

Every service in every `services/*/docker-compose*.yml` logs to the host
journal instead of Docker's default `json-file`:

```yaml
    logging:
      driver: journald
      options:
        tag: "{{.Name}}"
```

`tests/test_compose_logging_journald.py` (run by `orion-static-gates` CI)
fails if any service in any tracked compose file is missing this block, so new services cannot
drift.

## Why

The 2026-10-08 reading failure could not be diagnosed: a fleet restart
recreated orion-hub and its logs were gone.

With `json-file`, a container's log lives inside that container's own
directory (`/mnt/docker/containers/<id>/<id>-json.log` on athena). Checked live
on athena, 2026-10-10, Docker 29.1.3:

- `docker restart` keeps the same container, so the log survives.
- Recreating it (`docker compose up` after an image/config change,
  `down`/`up`, `--force-recreate`, `rm` + `run`) makes a new container ID and
  deletes the old directory, log included.
- With `journald`, the same recreate test kept both boots' lines:
  `journalctl CONTAINER_NAME=<name>` showed `first-boot` and `second-boot`.

journald is the simplest option that survives a recreate: Docker ships the
driver, journald is already running with persistent storage
(`/var/log/journal`), and its size is capped. The `local` driver also rotates
but deletes its files with the container, same as `json-file`. Anything else is
a log-shipping stack.

`json-file` was also unbounded here (no `max-size`): on 2026-10-10 the fleet
held about 60 GB of json logs, 32 GB of it from one orion-signal-gateway
container. Moving to journald puts the whole fleet under a single cap.

## Reading logs

`docker logs` / `docker compose logs` still work: journald is a driver Docker
can read back natively (Docker 29.1.3 here; dual logging, added in 20.10,
covers drivers that can't). They only show the *current* container, though.
For history that crosses a recreate, use journalctl:

```bash
# Everything a container name ever logged, across recreates
journalctl CONTAINER_NAME=orion-athena-hub --since "2026-10-08" --until "2026-10-09"

# Same thing by tag (tag = container name)
journalctl -t orion-athena-hub --since "2 hours ago"

# Follow live
journalctl -f CONTAINER_NAME=orion-athena-hub

# One specific (old) container instance
journalctl CONTAINER_ID=<12-char id>

# Search
journalctl CONTAINER_NAME=orion-athena-hub --since today | grep -i reading
```

Docker splits lines longer than 16 KB into several journal entries
(`CONTAINER_PARTIAL_MESSAGE=true`). `docker logs` joins them back together;
`journalctl | grep` does not, so a match in a very long line may be cut.

The user needs to be in `adm` or `systemd-journal` to read the journal
without sudo (athena's `athena` user is in `adm`).

Container names come from each compose file's `container_name:`
(usually `${PROJECT}-<service>`). Run `docker ps --format '{{.Names}}'` to
list them.

## Effect on existing containers

The logging driver is set when a container is created. A running container
keeps `json-file` until its next recreate (`docker compose up -d` after
pulling this change). Once recreated, its old `json-file` history is deleted
as before; from then on its lines go to the journal.

## Retention (host config)

`deploy/systemd/journald.conf.d/orion-container-logs.conf` sets
`SystemMaxUse=24G` (needs sudo to install). journald's default cap is
min(10% of `/`, 4G). On 2026-10-10 the fleet wrote about 2.4 MB/min of
json-file output (~3.4 GB/day). Under the default cap that is about a day of
history, and it would push system logs out early. 24G is roughly a week.

Two caveats on that week:

- It is an estimate from json-file growth. journald adds per-entry metadata,
  so short lines may take *more* space, not less. Check
  `journalctl --disk-usage` after a day.
- The cap is shared by the whole fleet, and orion-signal-gateway is about 60%
  of the volume (a per-message `Hunter intake` INFO line). Its spikes evict
  every other container's history first. Lowering that line's level is the
  real fix (separate follow-up).

Rate limiting: dockerd writes every container's lines itself, so the fleet
shares docker.service's journal rate limit. journald scales its default burst
(10000 / 30s) up with free disk space; with ~113G free on athena that is about
50000 / 30s, well above the fleet's ~4000 / 30s average. No change needed. If
lines do get dropped you will see `Suppressed N messages from docker.service`
in `journalctl -u systemd-journald`.

Containers started outside compose (`docker run` in scripts and evals) still
use the daemon default, `json-file`. Setting
`"log-driver": "journald", "log-opts": {"tag": "{{.Name}}"}` in
`/etc/docker/daemon.json` would cover them too; the compose block and its gate
stay the portable source of truth either way.

## Other hosts

The compose files for circe, atlas and hecate use the same block. Those hosts
need a running systemd-journald for the driver to start a container. Not
checked from athena (no SSH access) -- confirm with
`docker info --format '{{.LoggingDriver}} {{.ServerVersion}}'` and
`systemctl is-active systemd-journald` on each before recreating there.
Install the same journald drop-in there too: without persistent storage the
journal lives in memory and is lost on reboot.
