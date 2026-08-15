# B6: circe gets a power reading, from the PDU instead of the BMC it doesn't have

## Summary

- Added a read-only SNMP poller for the rack PDU's **per-outlet** active power, and wired it to
  `chassis_watts` for nodes with no BMC.
- **circe now has a chassis figure — ~419 W.** It is the only machine in the fleet that has
  never reported power, and every fleet total in this arc has carried
  `measurements_missing: {"chassis_watts": ["circe"]}` because of it.
- Cross-validated iLO for the first time: atlas read **291 W on its PDU outlets and 291 W on its
  iLO at the same instant**. Two independent instruments, one at the wall and one at the PSU.
- The PDU never overwrites a BMC reading. On a node with both they are two meters on the same
  watts, and they are kept separate rather than summed or raced.

## Outcome moved

```text
before   circe   chassis_watts: absent          fleet_watts_partial: ["circe"]
after    circe   chassis_watts: 419 W           from outlets 19 + 25 + 31
```

Read live from inside the running container:

```text
circe outlets: watts=419.0 per_outlet={19: 128.0, 25: 127.0, 31: 164.0}
```

That is the first power number circe has ever produced, and it closes the last hole in the
fleet total.

## Current architecture

`chassis_watts` came from one place: `IloPoller`, a RedFish call to the node's own BMC. circe's
BMC is unreachable (`192.168.1.150`, no route — confirmed from athena and from circe's own
subnet), so `_ilo_poller` is permanently `not_configured` there and no code path could produce
a chassis figure for it. B6 was recorded in the roadmap as "blocked on the NIC."

It was not blocked on the NIC. The rack PDU meters every outlet individually and was reachable
the whole time — SNMP was simply switched off on the device.

## Getting there — what the investigation actually found

Recorded because most of it invalidated an assumption:

1. **The UPS path is a dead end for power, permanently.** athena's Smart-UPS 1500 is on USB and
   apcupsd exposes 28 fields, none of which are `LOADPCT`, `LINEV` or `NOMPOWER`. Battery-side
   only — it is the unit's USB report set (firmware ID=1027), not a configuration problem. No
   amount of work on power-guard would have produced watts.
2. **`services/orion-power-guard/app/ups_snmp_client.py` is dead code** — a full APC SNMP client,
   imported nowhere, left behind when the UPS's management card became unreachable after a move
   and the service fell back to local USB. Its `#POWER_GUARD_UPS_HOST=192.168.0.50` sits
   commented out in `.env` as the scar. It contributed the `puresnmp` dependency this patch
   reuses, and nothing else.
3. **The PDU had SNMP disabled entirely**, not restricted. Juniper enabled it with a read
   community scoped to athena.
4. **The first metered table was the wrong one.** `10.1.2.2` looked like per-outlet data with 3
   rows; it is per-**bank**, and its single populated row carries atlas and circe together. The
   real per-outlet table is `10.1.5.1.1` with 36 rows.

## The OID decode, and how it was verified

```text
1.3.6.1.4.1.19536.10.1.5.1.1.<col>.1.<outlet>        enterprise 19536 = Panduit
    col 2   outlet name (generic, not editable on this firmware)
    col 12  apparent power, VA
    col 13  ACTIVE POWER, WATTS      <- the one that matters
    col 14  energy, Wh
```

Verified three independent ways:

- **Against the device's own admin panel, per outlet.** `OUTLET34`: panel 156 W / 159 VA /
  339.4 kWh, SNMP col13 156 / col12 159 / col14 339274.
- **By summation.** Per-outlet watts across all five powered outlets summed to **678 W** against
  a unit-level total of **680 W**.
- **Against an independent instrument.** atlas's outlets read 291 W with its iLO reading 291 W.

## The outlet map, and how it was got wrong first

```text
circe   outlets 19, 25, 31      3 PSUs    ~419 W
atlas   outlets 34, 35          2 PSUs    ~266 W
```

**I initially had these swapped.** I inferred 34+35 were circe from a power spike that appeared
while atlas's iLO stayed flat — but iLO polls on a **60 s** cadence and I sampled every 15 s, so
a ~30 s ramp on atlas was invisible to it. I used a stale value as evidence of absence. That is
the same sampling-skew trap flagged as a risk in the RAPL PR (#1669) one day earlier.

I also leaned on the per-outlet energy accumulators as an age signal, which was wrong twice
over: they all count from the same date (`2021/11/22`), and outlet 19's 15,308 kWh against
outlet 31's 78.9 kWh is real history from whatever occupied that outlet before circe.

Juniper traced the cabling physically. The corrected map then verified cleanly against the
steady-state iLO match. **A physical trace beat two telemetry inferences**, and the write-up
keeps that rather than presenting the answer as though it fell out of the data.

## Files changed

- `services/orion-biometrics/app/pdu.py`: new. `PduPoller` (mirrors `IloPoller`'s cadence and
  caching), `fetch_pdu_snapshot`, `parse_outlets`.
- `services/orion-biometrics/app/main.py`: construct/start/stop the poller, inject `pdu` into
  the pipeline input beside `ilo`.
- `services/orion-biometrics/app/settings.py`: `PDU_HOST`, `PDU_OUTLETS`, `PDU_SNMP_COMMUNITY`,
  `PDU_SNMP_PORT`, `PDU_POLL_INTERVAL_SEC`, `PDU_REQUEST_TIMEOUT_SEC`.
- `orion/telemetry/biometrics_pipeline.py`: `pdu_watts` in `extract_measurements` and
  `FLEET_SUM_KEYS`; `chassis_watts` fallback for BMC-less nodes.
- `services/orion-biometrics/{docker-compose.yml,.env_example,requirements.txt}`: config surface
  and `puresnmp==2.0.0` (same pin as orion-power-guard).
- `tests/test_pdu_outlet_power.py`: new, 22 tests.

## Design decisions worth stating

**Per-node polling, not hub-side.** Each node polls for its own outlets and reports its own
`chassis_watts`, exactly as `ILO_HOST` already works. Keeping `chassis_watts` a self-report
means the absent-is-not-zero machinery needs no changes: a node that cannot reach the PDU simply
has no value and gets named in `measurements_missing`, which is today's circe behaviour and
therefore not a regression.

**iLO wins where both exist**, and the ordering is tested. Under moving load the two disagree
instantaneously — measured 737 W (PDU) against 484 W (iLO) mid-burst on atlas — because one is a
60 s poll and the other is a live read. Without a fixed precedence, whichever wrote last would
win at random.

**All-or-nothing over a node's outlets.** A 3-PSU machine read from 2 outlets understates by a
third while looking like a valid number — the same error `_sum_of` prevents for GPU power. A
partial read returns an error, not a smaller sum.

**Read-only, and the write community is deliberately not used.** The device is outlet-*switched*:
a write community can power off circe and atlas. `.env_example` says so explicitly.

## Metric quality gate

1. **Provenance.** `1.3.6.1.4.1.19536.10.1.5.1.1.13.1.<outlet>`, the Panduit per-outlet monitor
   table, summed over the node's own outlets. Decode verified against the device's admin panel.
2. **Independence — YES, genuinely.** This is the first measurement in the arc not derived from
   the machine's own sensors. It is independent of iLO, RAPL and nvidia-smi, which is exactly
   what makes it able to validate them. On a node with both it is redundant *with iLO* and is
   documented and tested as such.
3. **Theory anchor.** Active power at the outlet is the quantity the electricity meter bills.
   Nothing about it is inferred.
4. **Live-data sanity.** circe 419 W across 3 outlets, atlas 266–737 W across 2, tracking load;
   both read exactly 0 W while powered off, so the metric has a true rest state rather than an
   arithmetic floor. Non-degenerate, and cross-checked against a second instrument.
5. **Existing mechanism.** `ups_snmp_client.py` exists but targets APC PowerNet OIDs on a device
   we no longer talk to; only its `puresnmp` dependency is reused. `IloPoller` is the structural
   pattern and is mirrored rather than duplicated in spirit.
6. **Reversibility.** One optional measurement key, one poller, and env that defaults to
   disabled. Unsetting `PDU_OUTLETS` restores the previous behaviour exactly.

## Schema / bus / API changes

- Added: `pdu_watts` in `measurements` and `FLEET_SUM_KEYS`; `pdu` blob on the pipeline input.
- Removed / renamed: none.
- Behavior changed: `chassis_watts` now has a second possible source, used **only** when the BMC
  produced nothing. No node that already reports `chassis_watts` sees any change — tested.
- Compatibility notes: additive and optional. A node without PDU config is unaffected and
  reports nothing new.

## Env/config changes

- Added keys: `PDU_HOST`, `PDU_OUTLETS`, `PDU_SNMP_COMMUNITY`, `PDU_SNMP_PORT`,
  `PDU_POLL_INTERVAL_SEC`, `PDU_REQUEST_TIMEOUT_SEC`.
- `.env_example` updated: yes, with the outlet map and the write-community warning.
- local `.env` synced: yes, **by hand** — `sync_local_env_from_example.py` resolves the example
  from the primary checkout, so keys added in a worktree are invisible to it and it exits clean.
  athena's `.env` has the keys with `PDU_OUTLETS` empty (it is not on this PDU).
- **circe and atlas need their values set at deploy — see below.**

## Tests run

```text
$ .venv/bin/python -m pytest tests/test_pdu_outlet_power.py tests/test_peak_pressure.py \
    tests/test_io_measurements.py tests/test_fleet_measurements.py \
    tests/test_fleet_roster_coverage.py -q
95 passed

$ .venv/bin/python -m pytest services/orion-biometrics/tests -q
83 passed, 2 failed
```

The 2 failures are the pre-existing `circe expected_offline` pair, unchanged and unrelated.

Fixture values are the device's own admin-panel readout, not numbers copied from the
implementation.

## Evals run

```text
None. services/orion-biometrics has no evals/ harness. This metric is validated against two
independent ground truths -- the device's admin panel and atlas's iLO -- in the live checks
below, which is stronger than an eval could be here.
```

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-biometrics up -d --build       # athena

# poller correctly DISABLED on athena (not on this PDU), iLO path untouched:
$ docker exec orion-athena-biometrics python -c "from app.main import _pdu_poller; ..."
pdu enabled: False   details: {}

# and the container CAN reach the PDU -- proves puresnmp installs and the network path works
# from inside the image, which is what circe and atlas will need:
$ docker exec orion-athena-biometrics python -c "from app.pdu import fetch_pdu_snapshot; ..."
circe outlets from inside the container: watts=419.0 per_outlet={19: 128.0, 25: 127.0, 31: 164.0}

# decode verified against the device's admin panel (OUTLET34): 156 W / 159 VA / 339.4 kWh
# summation check: per-outlet total 678 W vs unit total 680 W
# independent-instrument check: atlas PDU 291 W vs atlas iLO 291 W, same instant
```

**Not yet verified on circe or atlas** — the poller is disabled on athena by design, so this
build exercises the code path but not the per-node config. That happens at their deploy.

## Restart required

**circe** — add to `services/orion-biometrics/.env`, then deploy:

```bash
PDU_HOST=192.168.1.39
PDU_OUTLETS=19,25,31
PDU_SNMP_COMMUNITY=public

scripts/safe_docker_build.sh orion-biometrics up -d --build
curl -fsS http://localhost:8100/snapshot | grep -o '"chassis_watts":[0-9.]*'
```

**atlas** — same, with its own outlets:

```bash
PDU_HOST=192.168.1.39
PDU_OUTLETS=34,35
PDU_SNMP_COMMUNITY=public
```

**athena** — already deployed; keep `PDU_OUTLETS` empty.

If a node cannot reach `192.168.1.39`, its `chassis_watts` simply stays as it is today and
`measurements_missing` names it — no regression, and the telemetry reports the failure directly
rather than guessing.

## Risks / concerns

- Severity: medium. Concern: the outlet map is config, and nothing detects re-cabling. Moving a
  PSU to a different outlet would silently attribute one machine's power to another. The device's
  outlet names would have solved this but are not editable on this firmware. Mitigation: the map
  is documented in `.env_example` and `settings.py` with the date it was traced; a mismatch is
  detectable by comparing `pdu_watts` against `chassis_watts` on atlas, which has both.
- Severity: low. Concern: three nodes polling one PDU controller. Mitigation: 60 s cadence with
  caching, matching the iLO poller's reasoning; at most 5 GETs per node per minute.
- Severity: low. Concern: SNMP v2c sends the community in cleartext on the LAN. The read
  community is scoped to a single manager IP on the device. The **write** community is still the
  factory default `private` on a device that can switch outlets — flagged to Juniper, not
  changed here, and not used by this code.
- Severity: informational. Concern: `ups_snmp_client.py` remains dead code in
  `orion-power-guard`. It should either be deleted or replaced, but it is a different service and
  not this patch's business.
- Severity: informational. Concern: atlas and circe share one bank on this PDU, so one breaker
  covers both. Not a software concern; recorded because the arc's ceiling analysis treats them as
  independent hosts.

## PR link

<to be filled after push>
