# Capacity you cannot route to is not capacity; and a fleet total that sheds a machine

## Summary

- **Fixed a bug I shipped this morning.** `net_pressure`'s denominator summed every up physical
  NIC. Bringing athena's dark 10 GbE port up made that denominator 11,000 Mb/s against traffic
  that can only ever leave over the 1,000 Mb/s port — understating `net_pressure` **11-fold**,
  silently, at the exact moment someone thought they had improved the network.
- Added `nodes_absent` to the cluster payload. `measurements_missing` closes the gap for a node
  that reported but lacked a key; nothing closed the gap for a node that stopped reporting
  entirely.
- Found in the process that **`extra="ignore"` makes schema rollout silent** — a new field can
  be produced correctly and discarded by a stale consumer with no error anywhere.

## Outcome moved

**The bug, live on athena before this patch:**

```text
interfaces: ['eno1', 'eno6']    link_mbps: 11000.0
```

`eno6` links at 10 Gb but holds only a SLAAC IPv6 address and has **no IPv4 route** — the host
route table lists exactly one physical NIC, `eno1`. It carries nothing. After:

```text
interfaces: ['eno1']            link_mbps: 1000.0
```

**The roster gap, observed live while circe was down:**

```text
chassis_watts: 646.0        measurements_missing: null
```

A two-machine sum presenting as the whole fleet. After:

```text
sources:       ['atlas', 'athena']
nodes_absent:  ['circe', 'prometheus']
chassis_watts: 653.0
```

## Current architecture

`_physical_interfaces()` answered "is there a link" — a `device` symlink plus `operstate=up`.
That was correct for every state the fleet had been in, because athena had exactly one live NIC
and the question "which links exist" and "which links carry traffic" had the same answer.

They came apart the moment a second port was brought up. The rule was never wrong about link
existence; it was wrong to treat link existence as capacity.

Separately, `aggregate_fleet_measurements` receives `per_node` — the nodes the hub currently
holds. A node that stops publishing is not in that dict at all, so it cannot appear in
`missing`. The B1 docstring reasons carefully about circe's *absent BMC* and never about
circe's *absent self*.

## Files changed

- `services/orion-biometrics/app/metrics.py`: `_routed_interfaces()` reads the host IPv4 route
  table; `_carrying_interfaces()` intersects it with the physical set and is now what
  `_collect_network` uses for both numerator and denominator.
- `orion/schemas/telemetry/biometrics.py`: `nodes_absent` on `BiometricsClusterV1`.
- `services/orion-biometrics/app/main.py`: `publish_cluster` computes it from the node catalog,
  resolving aliases.
- `services/orion-biometrics/tests/test_io_sensor_scope.py`: 8 more tests.
- `tests/test_fleet_roster_coverage.py`: new, 10 tests.

## Why the route table

`/proc/net/route` is the kernel's own answer to "can traffic leave by this NIC". An interface
with no address has no row. It is cheap, already reachable through the `/host_proc` mount, and
it needs no heuristic about what an operator *meant* by bringing a port up.

Falls back to every up physical NIC when the route table is unreadable, and again when no
physical NIC appears in it — "I could not check" must not become "this node has no network",
and a v6-only host must not read as zero capacity and divide the pressure by nothing.

The physical filter still applies on top: `docker0` and the bridges have routes *and* a
fabricated 10 Gb/s speed, so routing is an extra constraint, not a replacement.

## Why `nodes_absent` carries no verdict

It states who is absent. It does not say whether that is a problem, because that is not the
aggregator's call:

- circe is run intermittently to save cost — its silence is expected.
- The same silence from athena would be an outage.

The catalog's `expected_online` flag cannot settle it either. I was about to set circe to
`false`, and the entry's own comment stopped me: it was flipped **to** `true` on 2026-07-18
because `false` made `pressure_organ.py` suppress circe's real strain as "expected staleness".
The flag is already committed to a different question — *should I trust this node's readings
when it is up* — and overloading it with *should I alarm when it is down* would have re-broken
a three-week-old fix.

So the flag is untouched, the two `circe expected_offline` tests stay failing (they encode an
assumption the catalog deliberately contradicts — not mine to resolve by picking a side), and
the aggregate reports coverage as fact.

## Schema / bus / API changes

- Added: `nodes_absent: Optional[List[str]]` on `biometrics.cluster.v1`.
- Removed / renamed: none.
- Behavior changed: `network.interfaces` and `network.link_mbps` now reflect routable NICs only.
  On any node with exactly one live NIC — which was every node until today — the value is
  unchanged.
- Compatibility notes: `None` means "this producer does not report coverage", **not** "nobody is
  absent". A consumer must not read the field's absence as a completeness guarantee; tested.

## The deployment lesson

`nodes_absent` was produced correctly and arrived as `None` at the consumer for twenty minutes
of fresh 15-second publishes. Not stale, not cached. `orion-state-service` was running an older
image whose `BiometricsClusterV1` predates the field, and `model_config = ConfigDict(extra="ignore")`
**silently discards it during validation.**

There is no error, no warning, and no log line anywhere. The symptom is indistinguishable from
"my producer is broken", and I spent four checks on the producer before thinking to interrogate
the consumer's schema:

```text
$ docker exec orion-athena-state-service python3 -c "..."
state-service schema has nodes_absent: False
extra policy: ignore
```

Adding a field to a shared schema is a **two-service deploy**. The producer alone proves
nothing.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: not needed.
- local `.env` synced: no change required.

## Tests run

```text
$ .venv/bin/python -m pytest services/orion-biometrics/tests/test_io_sensor_scope.py -q
42 passed

$ .venv/bin/python -m pytest tests/test_fleet_roster_coverage.py \
    tests/test_io_measurements.py tests/test_fleet_measurements.py -q
53 passed
```

The two pre-existing `circe expected_offline` failures are unchanged and deliberately not
addressed — see above.

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-biometrics up -d --build
$ scripts/safe_docker_build.sh orion-state-service up -d --build     # required, see above

# denominator corrected on the node where the bug was live:
  before:  interfaces: ['eno1', 'eno6']   link_mbps: 11000.0
  after:   interfaces: ['eno1']           link_mbps:  1000.0

# host truth it now matches:
$ awk 'NR>1 {print $1}' /proc/net/route | sort -u
  br-6d7d917f145a  br-8a71dda15b30  br-b62121b0a59e  docker0  eno1
$ ip -br addr show eno6
  eno6  UP  fd61:...:b0c1/64  fe80::...  <- IPv6 only, no IPv4, no route

# roster coverage reaching the consumer:
  sources:       ['atlas', 'athena']
  nodes_absent:  ['circe', 'prometheus']
  chassis_watts: 653.0     net_link_mbps: 11000.0    (athena 1000 + atlas 10000)
```

## Review findings fixed

- Finding: numerator and denominator could use different interface sets, making the ratio
  meaningless.
  - Fix: `_collect_network` derives both from the single `_carrying_interfaces()` call.
  - Evidence: `test_the_numerator_uses_the_same_interface_set_as_the_denominator`.
- Finding: an empty route table (parsed but with no rows) would yield an empty carrying set and
  report zero capacity.
  - Fix: `_routed_interfaces` returns `None` for an empty result, and `_carrying_interfaces`
    falls back on an empty intersection.
  - Evidence: `test_an_empty_route_table_is_treated_as_unreadable`,
    `test_no_physical_nic_routed_falls_back_rather_than_claiming_zero`.
- Finding: comparing raw node strings against canonical catalog ids would report an actively
  reporting node as absent, since nodes publish under names like `circe.tail348bbe.ts.net`.
  - Fix: resolve through the catalog's alias map before differencing.
  - Evidence: `test_absence_is_resolved_through_aliases_not_raw_strings`.

## Restart required

athena done (both services). atlas and circe need the biometrics redeploy to pick up the
routable-NIC fix — until then their `link_mbps` counts any unrouted NIC they may have:

```bash
scripts/safe_docker_build.sh orion-biometrics up -d --build
```

Any other service validating `biometrics.cluster.v1` needs a redeploy to see `nodes_absent` at
all — `orion-hub`'s biometrics cache is the other subscriber.

## Risks / concerns

- Severity: low. Concern: a NIC that is a bond slave or bridge member has no route of its own,
  so its capacity would be excluded even though it carries traffic. Not present on this fleet
  (no bonds), but it is the obvious next way this rule comes apart, exactly as the single-NIC
  assumption did. Mitigation: none needed today; revisit if a bond is introduced.
- Severity: low. Concern: `/proc/net/route` is IPv4-only. A genuinely v6-only node falls back to
  counting every link — the old, wrong-in-this-case behaviour. Mitigation: the fallback is
  explicit and tested; no such node exists here.
- Severity: informational. Concern: the roster names **`prometheus`**, a fourth node in
  `node_catalog.yaml` that is not contributing and that this arc has never mentioned. Either it
  is decommissioned and the catalog is stale, or it is a real node with no biometrics. Worth a
  look — the roster check surfaced it on its first run, which is roughly the point.
- Severity: informational. Concern: athena's 10 GbE port is up, cabled, and unusable — no
  address, no route. Whether to give it one is a real infrastructure decision and deliberately
  out of scope here. Measured evidence says 1 Gb is not currently binding: athena moves ~196 KB/s
  at rest against 125 MB/s of link.

## PR link

<to be filled after push>
