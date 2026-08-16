# B3: the disk and network sensors now measure the node, not the container

## Summary

- Fixed two independent collector bugs that made `disk_pressure` and `net_pressure` report
  quantities that were not the node's: `/proc/net/dev` read from inside a bridged container
  (its own veth), and `/proc/diskstats` summed whole disks *and* their partitions.
- Replaced the `NET_BW_MBPS` guess with a measurement: the summed link speed of the node's up
  physical NICs, read from the kernel. The constant is now a fallback.
- Un-parked `disk_bytes_per_sec`, `net_bytes_per_sec` and `net_link_mbps` into `measurements`
  and the fleet aggregate — they were deliberately withheld *because* of the two bugs above.
- Did **not** solve the disk denominator. Explained below rather than papered over.

## Outcome moved

`net_pressure` was **degenerate on every node in the fleet** — athena 3.2e-05, atlas 2.2e-05,
circe 2.1e-05. Not "low": structurally incapable of ever being anything else, because the
numerator was this service's own veth traffic against a 125 MB/s denominator. It contributed a
silent zero to `strain` on all three nodes.

Live on athena, before → after the same deploy:

```text
net_pressure    0.000032  ->  0.001529      (numerator is now the node's uplink)
disk_pressure   0.130226  ->  0.069019      (1.956x double-count removed)
```

and the raw quantities are published for the first time:

```json
"disk_bytes_per_sec": 13803773.5, "net_bytes_per_sec": 191073.1, "net_link_mbps": 1000.0
```

`net_pressure` is still small, and that is the honest reading — athena's 1 GbE genuinely is not
its bottleneck at rest. The difference is that it can now reach 1.0 during a model pull, which
it previously could not.

## Current architecture

`BiometricsCollector` ran entirely inside the container and read `/proc` as if it were the
host's. That is true for `/proc/diskstats` (not namespaced) and false for `/proc/net/dev` and
`/sys` (both namespaced). Nothing in the payload distinguished the two cases, so a consumer had
no way to tell a node reading from a container reading.

The plant work on 2026-08-13 caught both errors and B1 explicitly withheld the raw disk/net
numbers from `measurements` rather than emit them, parking the fix as "a live behaviour change
that belongs in its own patch". This is that patch.

## Architecture touched

```text
docker-compose.yml   /proc -> /host_proc:ro,  /sys -> /host_sys:ro      (new, read-only)
metrics.py           _physical_interfaces()   up NICs with a real `device` symlink
                     _link_speed_mbps()       summed sysfs `speed`, the new denominator
                     _read_netdev(path, only) host namespace, physical NICs only
                     _is_disk_device()        whole block devices, never partitions
                     network payload gains    scope, interfaces, link_mbps
pipeline             net scale = link_mbps * 1e6 / 8, else NET_BW_MBPS
                     extract_measurements()   + disk_bytes_per_sec, net_bytes_per_sec,
                                                net_link_mbps  (net only when scope == host)
                     FLEET_SUM_KEYS           + the three above (all extensive)
```

## Files changed

- `services/orion-biometrics/app/metrics.py`: both sensor fixes, host-namespace resolution,
  link-speed measurement, `scope` reporting, and a scope-change guard on the rate baseline.
- `services/orion-biometrics/app/settings.py`: `HOST_PROC_PATH`, `HOST_SYS_PATH`; both
  bandwidth constants re-documented for what they now are.
- `services/orion-biometrics/docker-compose.yml`: the two read-only host mounts and the two
  new env keys.
- `orion/telemetry/biometrics_pipeline.py`: measured net denominator; the three new
  measurements; the stale "deliberately not here" docstring replaced.
- `services/orion-biometrics/tests/test_io_sensor_scope.py`: new, 34 tests.
- `tests/test_io_measurements.py`: new, 20 tests.
- `services/orion-biometrics/.env_example`: the two new keys, plus honest documentation of
  what each denominator is and is not.

## Why three filters, not one

Each of these was load-bearing and each would have been a silent wrong number:

- **Physical NICs only, for the denominator.** athena's docker bridges and `docker0` all report
  a fabricated `speed` of 10000 Mb/s. Summing every interface with a speed gives 31,000 Mb/s of
  capacity on a box with one 1 Gb link — understating `net_pressure` 31-fold.
- **Physical NICs only, for the numerator.** A packet leaving a container traverses
  veth → bridge → eno1 and is counted on each. Summing all interfaces multiply-counts container
  traffic. Measured live: 132,575 B/s all-interfaces vs 126,568 B/s physical-only at rest, and
  the gap widens with inter-container traffic — of which athena has a great deal.
- **Up interfaces only.** athena has eno1..eno6 and one cable. Counting the five dark ports
  claims 6 Gb/s of capacity that does not exist.
- **`speed` of -1 is unknown, not a capacity.** `tailscale0` reports -1; using it would make the
  denominator negative and invert the pressure.

## What I did NOT fix, and why

**The disk denominator.** `DISK_BW_MBPS=200.0` is still one constant fleet-wide. Two reasons,
both real:

1. The kernel does not report block-device throughput the way it reports link speed. There is
   nothing to measure it from without benchmarking, which is not a thing to do to a live host
   mid-arc.
2. It is the wrong **shape**, not just the wrong value. athena has ten devices spanning a 10k
   SAS spinner and a Samsung 990 PRO — roughly 150 MB/s to 7000 MB/s. No single scalar is right
   for that array, so hand-picking one for athena while atlas and circe keep 200 would make
   cross-node `disk_pressure` incomparable in a *new and undocumented* way. That is worse than
   the status quo.

What I did instead: publish `disk_bytes_per_sec` in bytes/sec, which is real, node-scale, and
comparable across hosts. The 0-1 band keeps its documented meaning — "is this node's storage
unusually busy for itself" — and the `.env_example` now says so instead of implying a ceiling.

**I could not read atlas's or circe's NIC speeds.** No SSH from athena. This is exactly why the
denominator is measured per node by the collector rather than configured: each node reports its
own on redeploy, and until then it is `measurements_missing`, not a guess.

## Schema / bus / API changes

- Added: `network.scope`, `network.interfaces`, `network.link_mbps` on the biometrics sample;
  `disk_bytes_per_sec`, `net_bytes_per_sec`, `net_link_mbps` in `measurements` and
  `FLEET_SUM_KEYS`.
- Removed / renamed: none.
- Behavior changed: `disk_pressure` and `net_pressure` change value on redeploy — disk roughly
  halves, net rises by orders of magnitude. `strain` moves with them. This is the point, but it
  is a live change to a shipped field signal and is called out under Risks.
- Compatibility notes: all additive and optional. A consumer reading `measurements` gets the new
  keys only from redeployed nodes; the others appear in `measurements_missing`. `network.scope`
  absent is treated as *not* host, so a producer predating this patch never has its veth traffic
  read as node traffic.

## Env/config changes

- Added keys: `HOST_PROC_PATH=/host_proc`, `HOST_SYS_PATH=/host_sys`.
- Removed / renamed: none.
- `.env_example` updated: yes.
- local `.env` synced: **yes, by hand, and worth recording why.**
  `scripts/sync_local_env_from_example.py` resolves `.env_example` from the *primary checkout*,
  not from the worktree it is run in — so a worktree's new example keys are invisible to it and
  it reported nothing to add. I added both keys to the live `.env` directly and verified them
  through `docker compose config`. Anyone adding an env key from a worktree hits this.
- skipped keys requiring operator action: none.

## Tests run

```text
$ .venv/bin/python -m pytest services/orion-biometrics/tests/test_io_sensor_scope.py -q
34 passed

$ .venv/bin/python -m pytest tests/test_io_measurements.py tests/test_fleet_measurements.py \
    services/orion-cortex-exec/tests/test_metacog_biometrics_fleet_watts.py -q
46 passed

$ .venv/bin/python -m pytest services/orion-biometrics/tests -q
59 passed, 2 failed
```

The two failures are **pre-existing and unrelated** — `test_circe_expected_offline` and
`test_circe_node_availability_reflects_expected_offline` both assert circe is expected offline,
while `config/biometrics/node_catalog.yaml` says `expected_online=True`. Verified by running
both tests in the unmodified primary checkout, where they fail identically. Raised below.

Fixture values are the live 2026-08-14 readings and the arithmetic is hand-computed — the
10,240,000-byte diskstats total and the 125,000,000 B/s link scale are both derived by hand in
the test bodies, not copied from the implementation.

## Evals run

```text
None. services/orion-biometrics has no evals/ directory (unchanged from the B1 report).
The behaviour this patch changes is a sensor reading with a ground truth available on the
host, so it is verified by direct comparison under Docker checks below rather than by an
eval harness.
```

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-biometrics config     # mounts + env resolve
  HOST_PROC_PATH: /host_proc      source: /proc -> /host_proc  read_only: true
  HOST_SYS_PATH:  /host_sys       source: /sys  -> /host_sys   read_only: true

$ scripts/safe_docker_build.sh orion-biometrics up -d --build
  Container orion-athena-biometrics Recreated / Started

# live collector output, athena:
scope=host  ifaces=['eno1']  link=1000.0  rx=59428 tx=131645 | disk r=128815 w=13674958

# ground truth, measured on the host over the same period:
  host /proc/net/dev physical-only      126,568 B/s     container /proc/net/dev  1,357 B/s
  diskstats whole-disks-only          9,905,766 B/s     with partitions     19,378,995 B/s
                                                        ratio 1.956

# pressures moved as predicted:
  net  0.000032 -> 0.001529      disk  0.130226 -> 0.069019

# reached the fleet aggregate (orion-state-service), mid-rollout:
  measurements:  disk_bytes_per_sec 13,803,773  net_bytes_per_sec 191,073  net_link_mbps 1000
  missing:       disk_bytes_per_sec [atlas, circe]   net_bytes_per_sec [atlas, circe]
                 net_link_mbps      [atlas, circe]
```

That last block is the invariant working live: athena is redeployed and in the total, atlas and
circe are not and are **named**, not counted as zero.

## Review findings fixed

- Finding: a namespace change between ticks would subtract a container baseline from a host
  counter, reporting a one-tick burst of the host's entire lifetime traffic.
  - Fix: `_prev_net_scope` guards the delta; the baseline is dropped and the tick reports 0.
  - Evidence: `test_a_scope_change_drops_the_baseline_instead_of_spiking` — asserts 0.0 across
    the transition and a correct 500.0 B/s on the following tick.
- Finding: `link_mbps` is Mb/s (bits) and the pipeline's scale is bytes/sec. A missing factor of
  8 is invisible inside a 0-1 band — the same class of error that hid the original bug.
  - Fix: explicit `/ 8.0` with the reasoning in a comment.
  - Evidence: `test_one_gigabit_link_is_one_hundred_twenty_five_megabytes_per_second` pins
    1000 Mb/s and the legacy 125 MB/s constant to the identical scale, hand-derived.
- Finding: a bad `link_mbps` (0, -1, `True`, a string) would divide by zero, invert the
  pressure, or read every node as saturated.
  - Fix: explicit positive-numeric guard with a `bool` exclusion, falling back to the constant.
  - Evidence: parametrised over all four, plus `None`.

## Restart required

Already applied on athena. **atlas and circe still run the old collector** and will keep
reporting container-scoped network and double-counted disk until redeployed. They appear in
`measurements_missing` until then, so nothing silently averages them in.

```bash
# on atlas, and on circe:
scripts/safe_docker_build.sh orion-biometrics up -d --build
curl -fsS http://localhost:8100/raw/recent?limit=1 | grep -o '"scope":"[a-z]*"'   # want: host
```

## Risks / concerns

- Severity: medium. Concern: `disk_pressure` and `net_pressure` change value on every node as it
  is redeployed, and `strain` moves with them. Anything holding a learned baseline over these
  channels sees a step change. Mitigation: the direction is a correction toward truth in both
  cases, and the fleet rolls node-by-node so the change is observable rather than simultaneous.
  Worth watching the field-channel anomaly detector for a burst attributable to this.
- Severity: low. Concern: mounting host `/proc` and `/sys` into a container widens what that
  container can read about the host. Mitigation: both are `:ro`, nothing is written through
  either, and the collector reads exactly three paths under them (`1/net/dev`,
  `class/net/*/operstate`, `class/net/*/speed`). This is a telemetry service whose entire job is
  reading host state, and it already holds five read-only host mounts for disk capacity.
- Severity: low. Concern: `net_bytes_per_sec` counts only physical NICs, so purely
  container-to-container traffic on athena — of which there is a lot — is invisible to it. That
  is correct for "is the uplink saturated" and wrong for "how much network work is this node
  doing". The current name leans toward the latter. Follow-up: rename or split if a consumer
  ever needs intra-node traffic; nothing does today.
- Severity: informational. Concern: `config/biometrics/node_catalog.yaml` marks circe
  `expected_online=True` while two tests assert it is expected offline; both fail on main. Given
  the arc's premise that circe is *off by choice* and expensive to admit, which of the two is
  wrong is a real question, not a test-fixture typo. Left for Juniper rather than silently
  changed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1667
