# orion-power-guard

Watches a UPS and, if utility power stays out too long, shuts the host down
before the battery runs dry.

Today this runs a single instance, on **athena**, watching the APC
Smart-UPS 1500 plugged into it over USB.

## What it does

1. Polls the UPS every `POWER_GUARD_POLL_INTERVAL_SEC` (default 5s) for its
   status: online / on-battery, charge %, line voltage, time left.
2. Publishes `power.guard.on_battery` / `power.guard.grace_elapsed` /
   `power.guard.restored` events to the Orion bus (`orion:power:events`), so
   the rest of Orion can see a power event happen, not just infer it from a
   service going dark.
3. If the UPS stays on battery for longer than
   `POWER_GUARD_ONBATTERY_GRACE_SEC` (default 300s / 5 min), and
   `POWER_GUARD_ENABLE_SHUTDOWN=true`, it shuts the host down.
4. It also publishes a bus-native `SystemHealthV1` heartbeat
   (`orion:system:health`) so a dead power-guard container is itself
   visible, independent of the power events above.

## How it talks to the UPS (USB, via apcupsd)

The UPS is USB-attached to athena, not network-attached. power-guard doesn't
speak to the UPS driver directly — it talks to
[`apcupsd`](https://www.apcupsd.org/), which owns the USB device and already
runs as a systemd service on athena:

```text
UPS (USB) -> apcupsd (host, systemd) -> NIS protocol, TCP 3551 -> power-guard (container)
```

`apcupsd`'s NIS server is what `app/ups_nis_client.py` connects to
(`POWER_GUARD_UPS_HOST=host.docker.internal`, reached over the docker bridge
via the `extra_hosts: host.docker.internal:host-gateway` entry in
`docker-compose.yml`). Confirm it's up with `apcaccess status` on the host,
or `sudo systemctl status apcupsd`.

There's also an older SNMP client (`app/ups_snmp_client.py`,
`POWER_GUARD_UPS_HOST=192.168.0.50` in the commented-out example) for a
network-card-equipped UPS reached over the LAN instead of USB. `app/main.py`
is wired to the NIS/USB client — the SNMP client is unused code kept for a
different UPS setup, not a toggle.

## Shutdown wiring

This is the part worth reading closely if you're touching it.

**power-guard runs in a container. A local `shutdown` command inside that
container does not shut down the host** — it has no effect on athena at
all. The container mounts a dedicated, purpose-built SSH key
(`docker-compose.yml`: `/root/.ssh/powerguard_shutdown` ->
`/etc/powerguard/ssh_key`, read-only) so it can reach out to the real host
and shut *that* down:

```text
power-guard (container) --ssh, root, key--> host.docker.internal (athena) --> shutdown -h now
```

`POWER_GUARD_SHUTDOWN_CMD` (default, in `.env_example`):

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 \
  -i /etc/powerguard/ssh_key root@host.docker.internal \
  'shutdown -h now "Orion PowerGuard: UPS on battery beyond grace period"'
```

Requires on the host:
- `sshd` listening and reachable from the docker bridge (already true on
  athena — confirmed `0.0.0.0:22`).
- `PermitRootLogin` allowing key auth for root (already true — athena's
  `/etc/ssh/sshd_config` has `PermitRootLogin yes`).
- The public half of `/root/.ssh/powerguard_shutdown` in root's
  `authorized_keys`.

`StrictHostKeyChecking=no` / `UserKnownHostsFile=/dev/null`: this link never
leaves the docker bridge to the same physical host, and a persistent
`known_hosts` file has nowhere reliable to live across container restarts —
disabling host-key pinning here is a deliberate tradeoff for that narrow
path, not a general recommendation.

**`POWER_GUARD_ENABLE_SHUTDOWN` defaults to `false`.** Arming a real,
unattended host shutdown is a per-deployment decision, not something a
checked-in template should turn on for you. To arm it: set
`POWER_GUARD_ENABLE_SHUTDOWN=true` in the service's `.env`, then restart the
container (`scripts/safe_docker_build.sh orion-power-guard up -d`).

### apcupsd's own failsafe (separate, already there, lower-level)

Independent of all of the above, `apcupsd` itself will shut athena down
directly (no container, no bus event, `/etc/apcupsd/apccontrol` ->
`shutdown -h now`) if battery charge drops below `BATTERYLEVEL` (5%) or
remaining runtime drops below `MINUTES` (3 min) — see
`/etc/apcupsd/apcupsd.conf` on athena. It also has a flat "on battery for N
seconds regardless of charge" trigger (`TIMEOUT`), currently `0`
(disabled) — the closest native equivalent of power-guard's grace timer, but
silent: no bus event, nothing else in Orion sees it happen. power-guard's
grace timer exists to give that same kind of protection *with* visibility,
not to duplicate apcupsd's job.

## Config

See `.env_example` for the full list with defaults. The ones that matter
most:

| Key | Default | What it does |
|---|---|---|
| `POWER_GUARD_UPS_HOST` | `host.docker.internal` | Where `apcupsd`'s NIS server is (USB mode). |
| `POWER_GUARD_POLL_INTERVAL_SEC` | `5.0` | How often to poll the UPS. |
| `POWER_GUARD_ONBATTERY_GRACE_SEC` | `300.0` | How long on battery before the grace-elapsed event (and shutdown, if enabled) fires. |
| `POWER_GUARD_ENABLE_SHUTDOWN` | `false` | Arm/disarm the real host shutdown. |
| `POWER_GUARD_SHUTDOWN_CMD` | (see above) | The command run when the grace period elapses. |
| `CHANNEL_POWER_EVENTS` | `orion:power:events` | Bus channel for on_battery/grace_elapsed/restored events. |

Env parity: if you change `.env_example`, sync your local `.env`
(`python scripts/sync_local_env_from_example.py` from repo root) — see
`AGENTS.md` section 7.

## Verifying it's actually working (not just configured)

```bash
# UPS status directly from apcupsd on the host
apcaccess status

# power-guard's own poll loop -- should show raw=ONLINE/ONBATT matching apcaccess
docker logs --tail 20 orion-athena-power-guard

# Prove the shutdown path can reach the host (harmless -- does NOT shut down)
docker exec orion-athena-power-guard sh -c \
  "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 \
   -i /etc/powerguard/ssh_key root@host.docker.internal 'echo SSH_OK && hostname'"
```

Do not test the real `POWER_GUARD_SHUTDOWN_CMD` end-to-end against a host
you care about — it shuts the host down. Verify SSH reachability (above)
and trust the grace-timer logic via `tests/`, not a live fire.

## Tests

```bash
pytest services/orion-power-guard/tests -q
```

No `evals/` directory — this service's output is a poll loop and a shutdown
trigger, not generated content; the gate tests above are the coverage that
applies here.
