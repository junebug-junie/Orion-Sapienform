# B4: real CPU package power, via RAPL

## Summary

- `cpu_watts_total` now measures athena's actual CPU package draw from Intel RAPL energy
  counters — **192.99 W** across two sockets, live. It was the single largest unattributed term
  in the power budget on the node whose entire job is CPU orchestration.
- **No host permission change was needed.** B4 was blocked on "`energy_uj` is mode 400, so we
  need a chmod or a root-run collector." Neither. The `/host_sys:ro` mount shipped in #1667
  plus the container already running as uid 0 makes the file readable as-is.
- Handles the two things that make a cumulative energy counter hard: it **wraps** roughly every
  41 minutes at load, and a gap longer than one wrap period is **ambiguous** and must be
  discarded rather than corrected.
- Documented the containment rule properly: `chassis_watts` already includes both `cpu_watts_total`
  and `gpu_watts_total`, so summing all three over-counts a 425 W machine as 811 W.

## Outcome moved

athena, live through the full chain after this deploy:

```text
  chassis (PSU, iLO)     425.0 W
  cpu packages (RAPL)    193.0 W    45.4%     <- new, and previously invisible
  gpu (nvidia-smi)       193.5 W    45.5%
  unattributed            38.5 W     9.1%     10 disks, RAM, fans, PSU loss, NICs
```

Before this patch the CPU line did not exist and 363 W of a 407 W machine was a single
undifferentiated remainder. This is the direct answer to "we are really missing CPU cost,
particularly with Athena whose main job is CPU warhorse."

The decomposition is also a cross-check on the BMC: two independent instruments (RAPL from the
CPU, iLO from the PSU) now bound the same machine, and the parts do not exceed the whole.

## Current architecture

`_collect_power` read exactly two things: per-GPU `power_draw_watts` from nvidia-smi, and a max
temperature. CPU power was not collected at all — not measured badly, simply absent.

The blocker recorded in the roadmap was access: `/sys/class/powercap/intel-rapl:*/energy_uj` is
mode 400 root-only, which is the CVE-2020-8694 (PLATYPUS) mitigation — RAPL is fine-grained
enough to work as a power side channel against AES. The roadmap assumed the fix was a `chmod`
(reverting that mitigation host-wide, and not surviving reboot) or running the collector as
root (a large hammer for one file).

Both were unnecessary. #1667 mounted host `/sys` read-only for NIC link speed, and this
container already runs as uid 0 with no user-namespace remapping — so root-in-container is root
on the host for file permissions. Verified before writing any code:

```text
$ docker exec orion-athena-biometrics id
uid=0(root) gid=0(root)
$ docker exec orion-athena-biometrics cat /host_sys/class/powercap/intel-rapl:0/energy_uj
91900177637
```

The security posture is unchanged: nothing on the host was relaxed, and the file remains
unreadable to every non-root process exactly as before.

## Metric quality gate

Run in full before wiring anything, per the repo contract.

1. **Provenance.** `/sys/class/powercap/intel-rapl:N/energy_uj`, produced by the kernel's
   `intel_rapl` driver from `MSR_PKG_ENERGY_STATUS`. A cumulative microjoule counter for the
   package domain. Power is Δenergy/Δt — the value itself is not a power reading.
2. **Independence — NO, and that is the point.** `chassis_watts` already contains this. It is a
   *decomposition*, not a new signal, and it is documented and tested as contained. It is not
   independent of chassis power and must never be summed with it.
3. **Theory anchor.** RAPL is the vendor's own instrumented energy counter, and the basis of
   `turbostat`, `powertop` and `scaphandre`. The specific claim — "how much of athena's draw is
   CPU" — is exactly what the package domain measures.
4. **Live-data sanity.** 103.62 W and 106.72 W across the two sockets on the first probe, later
   96.96 and 96.03. Non-degenerate, balanced across sockets as expected on a load-balanced
   80-thread box at load average ~30, and it tracks load. Not flat, not saturated, not null.
   Checked the rest point explicitly: this is a rate derived from a monotonic counter, so it can
   genuinely reach a low idle value rather than having an arithmetic floor.
5. **Existing mechanism.** `rg -in "rapl|energy_uj|powercap"` over the repo: no hits outside the
   roadmap that proposed it. Nothing to reuse and nothing being duplicated.
6. **Reversibility.** One additive optional key in `measurements` plus one entry in
   `FLEET_SUM_KEYS`. Nothing trains on it, no schema migration, no manifest default. Cheap to
   remove.

## Files changed

- `services/orion-biometrics/app/metrics.py`: `_rapl_package_watts()`, `_RAPL_PACKAGE_RE`,
  `_RAPL_MAX_GAP_SEC`, `_prev_rapl` state, and `dt` threaded into `_collect_power`.
- `orion/telemetry/biometrics_pipeline.py`: `cpu_watts_total` in `extract_measurements` and
  `FLEET_SUM_KEYS`; the containment docstring rewritten to cover all three power terms with the
  live decomposition table.
- `services/orion-biometrics/tests/test_rapl_cpu_power.py`: new, 16 tests.
- `tests/test_io_measurements.py`: 4 more tests for the fleet sum and the containment rule.

## The two hard parts

**It wraps.** `max_energy_range_uj` is 262,143,328,850 µJ. At the measured ~105 W per package
that is a wrap roughly every **41 minutes** — about once every 83 ticks at
`TELEMETRY_INTERVAL=30`. A negative delta is the *normal* case several times a day, not an
error. Uncorrected, the test case in the suite would report **−262,143 W**.

**A long gap is ambiguous.** If the collector stalls past one wrap period, the counter may have
wrapped more than once, and one wrap is indistinguishable from three — correcting by a single
range silently under-reports by whole multiples. There is no way to recover it, so a baseline
older than `_RAPL_MAX_GAP_SEC` (300 s, 10× the tick) is discarded and the tick reports nothing.
Absent beats confidently wrong.

Two smaller ones, both tested: `intel-rapl:0:1` subdomains (dram/core) are already inside their
parent package and matching them would roughly double the figure — the same bug class as
summing disk partitions alongside their whole disk, one PR earlier. And a socket appearing
mid-run has no baseline, so the tick is suppressed entirely rather than reporting a sum that
silently excludes half a dual-socket box.

## Schema / bus / API changes

- Added: `power.cpu_package_watts` and `power.cpu_package_watts_by_domain` on the sample;
  `cpu_watts_total` in `measurements` and `FLEET_SUM_KEYS`.
- Removed / renamed: none.
- Behavior changed: none. Nothing existing reads these; no pressure or band changes value.
  `power_pressure` is deliberately untouched — folding CPU power into it would change a shipped
  field signal, which is a separate decision, not a side effect of adding a sensor.
- Compatibility notes: additive and optional throughout. A node without RAPL, or one not yet
  redeployed, is **absent** from `cpu_watts_total` and named in `measurements_missing` — never
  counted as 0 W.

## Env/config changes

- Added / removed / renamed keys: **none.** This rides entirely on `HOST_SYS_PATH` and the
  `/sys:/host_sys:ro` mount that shipped in #1667.
- `.env_example` updated: not needed.
- local `.env` synced: no change required; verified no new keys.

## Tests run

```text
$ .venv/bin/python -m pytest services/orion-biometrics/tests/test_rapl_cpu_power.py -q
16 passed

$ .venv/bin/python -m pytest tests/test_io_measurements.py tests/test_fleet_measurements.py -q
43 passed
```

Wrap arithmetic is hand-computed in the test bodies, not copied from the implementation — e.g.
baseline `MAX_RANGE - 100`, reading `900`, so the raw delta is −262,143,328,750 and the
corrected delta is 1001 µJ. The live-reading test reproduces the real 20 s probe:
2,072,483,098 µJ / 20 s = 103.62 W.

The two pre-existing `circe expected_offline` failures noted in #1667 are unchanged and still
unrelated.

## Evals run

```text
None. services/orion-biometrics has no evals/ harness. This metric has a ground truth
available on the same machine (the iLO chassis reading bounds it from above), so it is
validated by the containment check under Docker below rather than by a harness.
```

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-biometrics up -d --build
  Container orion-athena-biometrics Recreated / Started

# first tick after restart correctly reports nothing -- no baseline:
[1] cpu_package_watts=None
[2] cpu_package_watts=192.989  by_domain={'package-0': 96.958, 'package-1': 96.031}

# reached measurements, and the containment check holds:
  chassis 425.0 W | cpu 193.0 W (45.4%) | gpu 193.5 W (45.5%) | residual 38.5 W (9.1%)

# reached the fleet aggregate (orion-state-service):
  sources: [atlas, circe, athena]
  cpu_watts_total: 192.989      missing: [atlas, circe]
  chassis_watts:   702.0

# reached cognition (cue rendered inside the live cortex-exec container):
  {"status":"fresh","constraint":"NONE","strain":0.16,"homeostasis":0.84,
   "stability":0.98,"fleet_watts":702,"fleet_watts_partial":["circe"],"freshness_s":30}
```

## Review findings fixed

- Finding: a wrapped counter reports a large negative delta ~every 41 minutes at load.
  - Fix: add `max_energy_range_uj + 1` on a negative delta; return absent if still negative.
  - Evidence: `test_a_wrap_is_corrected_not_reported_as_a_power_drop` and
    `test_a_realistic_wrap_at_load_gives_a_realistic_wattage` — the latter straddles a wrap at
    a real 30 s / 105 W tick and asserts 105.0 W out.
- Finding: after a long stall, wrap correction silently under-reports by whole multiples of the
  range with no way to detect it.
  - Fix: `_RAPL_MAX_GAP_SEC` guard discards a stale baseline.
  - Evidence: `test_a_gap_longer_than_the_guard_is_discarded_not_guessed`, plus a companion
    asserting a merely-slow tick still reports.
- Finding: matching `intel-rapl:0:1` alongside `intel-rapl:0` would double-count, since
  subdomain energy is already inside its package.
  - Fix: `_RAPL_PACKAGE_RE` anchors on the top level only.
  - Evidence: `test_subdomains_are_not_double_counted` — 100 W package with 80 W core and 20 W
    dram present must read 100.0, not 200.0.
- Finding: a socket appearing mid-run would make the total silently exclude a package —
  plausible-looking and wrong by ~50% on a dual-socket box.
  - Fix: suppress the whole tick until every domain has a baseline.
  - Evidence: `test_a_socket_appearing_mid_run_suppresses_the_tick`, including recovery.

## Restart required

Already applied on athena.

```bash
# atlas and circe -- they will report cpu_watts_total only if they expose a RAPL tree,
# and are named in measurements_missing otherwise:
scripts/safe_docker_build.sh orion-biometrics up -d --build
curl -fsS 'http://localhost:8100/raw/recent?limit=1' | grep -o '"cpu_package_watts":[0-9.]*'
```

## Risks / concerns

- Severity: medium. Concern: **the containment invariant can invert transiently from sampling
  skew.** iLO chassis power is polled every `ILO_POLL_INTERVAL_SEC=60` while the telemetry tick
  is 30 s, so during a fast GPU ramp `cpu + gpu` can briefly exceed a stale `chassis_watts`.
  Observed adjacent to this: chassis rose only 425 W from 407 W while the GPU rose from 44 W to
  193 W over the same period, which is the lag showing. Nothing consumes the comparison today.
  Mitigation / follow-up: any consumer computing a residual must treat a negative one as
  "instruments disagree, sampling skew" and not as a reading. Worth a `power_sampled_at` per
  source before anyone builds an attribution UI on this.
- Severity: low. Concern: RAPL package domain excludes DRAM on this CPU (no dram subdomain
  exposed), so `cpu_watts_total` is cores + uncore only and the 38.5 W residual includes memory.
  The name could be read as "everything CPU-ish". Mitigation: documented in the containment
  block; rename if a consumer ever needs the distinction.
- Severity: low. Concern: atlas and circe are unverified — I cannot SSH to either, so whether
  they expose a RAPL tree at all is unknown until they are redeployed. Mitigation: absence is
  handled and named; no fleet number silently treats them as 0 W.
- Severity: informational. Concern: PR #1665 (the hub-mode flip) is still **open**, not merged.
  The flip itself is live on athena because it is a local `.env` change, so everything above
  works — but the `.env_example` documentation and the two contract corrections in that PR are
  not on main yet.

## PR link

<to be filled after push>
