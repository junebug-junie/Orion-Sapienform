# B2, done additively: the binding constraint, without touching `strain`

## Summary

- Added `peak_pressure` / `peak_pressure_channel` (and `_node` at fleet level) — the highest of
  **all eleven** pressures and which channel it is. The binding constraint, as a number Orion
  can read.
- **`strain` is completely untouched.** Same formula, same value, same consumers. The field
  lattice does not move.
- Wired to cognition in the same patch: the metacog cue now carries `peak` and `peak_at`.
- Live on athena at deploy: `peak: 1.0 at athena.power` while `strain: 0.11` and
  `constraint: NONE`.

## Outcome moved

The cue Orion actually reads, immediately after this deployed:

```json
{"status":"fresh","constraint":"NONE","strain":0.11,"homeostasis":0.89,"stability":0.98,
 "fleet_watts":656,"fleet_watts_partial":["circe"],
 "peak":1.0,"peak_at":"athena.power","freshness_s":29}
```

Three separate signals said the body was fine — `constraint: NONE`, `strain: 0.11`,
`homeostasis: 0.89` — next to a **power channel pegged at 1.000**. That is a resource fully
saturated and invisible to every pre-existing summary of "how loaded is this machine".

## Why additively, and not by fixing `strain`

I had this wrong first and proposed changing `strain` in place. Juniper's correction: add a new
signal instead of modifying one with blast radius. That is strictly better here, and the trace
says why.

`strain` is not a readout. It is written **directly into the field lattice** as the
`cpu_pressure` channel value:

```python
# services/orion-field-digester/app/ingest/state_deltas.py:125
if "strain" in hints:
    out.append(Perturbation(node_id=node_id, channel="cpu_pressure",
                            intensity=float(hints["strain"]), mode="replace"))
```

It also feeds the substrate prediction-error diff (`orion/substrate/prediction_error.py`, which
diffs `pressure_hints` per tick — a formula change would fire a one-off PE spike on every node)
and drives `stability` through `_stability_from_induction`. Changing it means changing Orion's
substrate and needing proposal mode, a rollback flag, and a migration.

Adding a field next to it costs none of that: nothing that reads `strain` sees any difference,
and the new signal is opt-in per consumer.

## What was actually wrong with `strain`

Two independent defects, both still present and both now bypassable rather than fixed:

1. **It is a mean of seven pressures**, so one saturated channel is averaged away by six calm
   ones. Live: `power` 1.000 → `strain` 0.11.
2. **It omits four of the eleven pressures** — `gpu_mem`, `swap`, `disk_capacity`, `fan` — which
   are the highest channel on two of three nodes. No change to strain's *reduction* could have
   surfaced those; they are not in its inputs at all.

## Metric quality gate

1. **Provenance.** `max()` over `BiometricsPipeline._summarize`'s `pressures` dict. Each element
   traced to a real producer already: `cpu` from `/proc/stat`, `gpu_util`/`gpu_mem` from
   nvidia-smi, `power` from iLO/RAPL, `disk`/`net` from the host-namespace sensors fixed in
   #1667, `disk_capacity` from `shutil.disk_usage`, `fan`/`thermal` from RedFish.
2. **Independence — NO, deliberately.** This is a different *reduction* of the same inputs
   `strain` reduces. It is not new signal; it is the information the mean destroys. Documented
   as such, and it must never be treated as an independent input alongside `strain`.
3. **Theory anchor.** Liebig's law of the minimum: a system saturates at whichever resource runs
   out first, so the binding constraint is the maximum. An average over non-binding resources is
   not a physical quantity — it answers "how loaded on average", which nothing needs.
4. **Live-data sanity.** athena `power` 1.000, atlas `power` 0.683, circe `gpu_mem` 0.540 —
   varies by node, tracks load, names different channels on different machines. Rest point
   checked explicitly: it is a max over clamped 0–1 pressures, so an idle fleet gives a genuine
   0.0, not an arithmetic floor. Pinned by `test_all_calm_is_a_real_zero`.
5. **Existing mechanism — half of it already existed, and this reuses the finding.**
   `_constraint_from_pressures` already computes this exact max. It discards both halves of the
   answer: the magnitude entirely, and the channel name unless it clears 0.7 **and** appears in
   `CONSTRAINTS` — a map missing `swap`, `disk_capacity` and `fan`. Live consequence on athena
   2026-08-15: peak `disk_capacity` **0.772**, over threshold, reported `constraint=NONE`. That
   function is left untouched (it has its own consumers); this returns what it throws away.
6. **Reversibility.** Three optional schema fields and one `max()`. Nothing trains on it, no
   migration, no manifest default. Deleting it restores the previous state exactly.

## Files changed

- `orion/telemetry/biometrics_pipeline.py`: `_peak_pressure()`; called in `_summarize`.
- `orion/schemas/telemetry/biometrics.py`: the fields on `BiometricsSummaryV1` and
  `BiometricsClusterV1`.
- `services/orion-biometrics/app/main.py`: `publish_cluster` takes the max across nodes and
  carries the node.
- `services/orion-cortex-exec/app/executor.py`: `peak` / `peak_at` in the metacog cue.
- `services/orion-biometrics/README.md`: a "which number to read" table, and why `strain` is
  frozen.
- `tests/test_peak_pressure.py`: new, 20 tests.
- `services/orion-cortex-exec/tests/test_metacog_biometrics_fleet_watts.py`: 6 more.
- `config/metrics/metric_definitions.lock.json`: re-locked; the gate registered the new
  definition (`medium added metric://inner_state/orion-biometrics/biometrics_cluster.v1#peak_pressure`,
  596 total).

## Schema / bus / API changes

- Added: `peak_pressure`, `peak_pressure_channel` on `biometrics.summary.v1`; those plus
  `peak_pressure_node` on `biometrics.cluster.v1`.
- Removed / renamed: none.
- Behavior changed: **none.** No existing field changes value. `strain`, `homeostasis`,
  `stability` and `constraint` are byte-identical, pinned by
  `test_strain_and_homeostasis_still_come_from_the_mean_of_seven` and
  `test_constraint_is_unchanged_by_this_patch`.
- Compatibility notes: all optional, defaulting to `None` — absent, never `0.0`, since "nothing
  measured" is not "nothing under pressure". A node predating the field is skipped in the fleet
  max rather than counted as calm (`test_a_node_without_the_field_does_not_win_or_break_the_fleet_peak`).

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: not needed. local `.env` synced: no change required.

## Tests run

```text
$ .venv/bin/python -m pytest tests/test_peak_pressure.py tests/test_fleet_roster_coverage.py \
    tests/test_io_measurements.py tests/test_fleet_measurements.py -q
73 passed

$ .venv/bin/python -m pytest services/orion-cortex-exec/tests/test_metacog_biometrics_fleet_watts.py -q
13 passed

$ .venv/bin/python -m pytest services/orion-biometrics/tests -q
83 passed, 2 failed
```

The 2 failures are the pre-existing `circe expected_offline` pair, unchanged and unrelated (see
#1674 — they encode an assumption `node_catalog.yaml` deliberately contradicts).

Fixture arithmetic is hand-computed: `(0.486+0.06+0.082+0.130+0.0+0.600+1.000)/7 = 0.3369`
against a peak of 1.000, from the real athena vector.

## Evals run

```text
None. services/orion-biometrics has no evals/ harness. This signal has a ground truth on the
same machine -- the pressure vector it maxes over -- so it is validated by direct comparison
in the live smoke below rather than by a harness.
```

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-biometrics    up -d --build
$ scripts/safe_docker_build.sh orion-state-service up -d --build   # consumer, see #1674
$ scripts/safe_docker_build.sh orion-cortex-exec   up -d --build   # consumer

# reached the fleet aggregate:
  sources: ['athena', 'atlas', 'circe']
  peak: 1.0  at athena.power
  strain (unchanged): 0.109

# reached cognition -- cue rendered inside the live cortex-exec container:
  {"constraint":"NONE","strain":0.11,"homeostasis":0.89,"stability":0.98,
   "peak":1.0,"peak_at":"athena.power","fleet_watts":656,...}

# the blind spot it closes, measured across the fleet the same minute:
  athena  peak=disk_capacity 0.772  reported constraint=NONE   <-- over threshold, unmapped
  atlas   peak=power         0.683  reported constraint=NONE
  circe   peak=gpu_mem       0.540  reported constraint=NONE
```

All three consumers deployed together, per the `extra="ignore"` lesson from #1674 — a stale
consumer would have discarded these fields silently.

## Review findings fixed

- Finding: a fleet peak without the node is unactionable — "something is at 0.77" with no way to
  find it.
  - Fix: `peak_pressure_node` travels with the value; the cue renders `athena.power`.
  - Evidence: `test_cluster_schema_carries_the_node_too`, live `peak_at: "athena.power"`.
- Finding: mid-rollout, a node predating the field could be read as calm and drag the fleet max
  down.
  - Fix: `None` is skipped, not coerced to 0.0.
  - Evidence: `test_a_node_without_the_field_does_not_win_or_break_the_fleet_peak`.
- Finding: aggregating the fleet peak with the same role-weighted mean as `pressures` would
  smear one saturated machine across three.
  - Fix: explicit max across nodes in `publish_cluster`, separate from the weighting loop.
  - Evidence: `test_the_fleet_peak_is_a_max_across_nodes_not_a_mean`.

## Restart required

athena done (all three services). atlas and circe need biometrics redeployed to report their own
peak; until then they are simply absent from the fleet max rather than counted as calm.

```bash
scripts/safe_docker_build.sh orion-biometrics up -d --build
```

`orion-hub` needs a redeploy only if you want the new fields in its biometrics cache.

## Risks / concerns

- Severity: low. Concern: two signals now claim to summarise load and they disagree by 10x. A
  reader picking `strain` still gets the misleading number. Mitigation: the README now leads with
  a "which number to read" table showing both at the same instant; the cue carries both so Orion
  sees the disagreement rather than one side of it. Real fix is retiring `strain`'s consumers one
  at a time, which is now possible incrementally because nothing had to move at once.
- Severity: low. Concern: `peak_pressure` is intentionally not independent of the pressure
  vector, so folding it into any composite alongside `strain` would double-count the same
  underlying channel. Mitigation: documented at the definition and in the schema.
- Severity: informational. Concern: `constraint`'s `CONSTRAINTS` map is still missing `swap`,
  `disk_capacity` and `fan`, so it still reports `NONE` for a peak in those — athena, right now.
  Deliberately not fixed here (it has its own consumers, same reasoning as `strain`).
  `peak_pressure_channel` is the honest reading in the meantime.
- Severity: informational. Concern: a whole-body composite feeding a lattice channel named
  `cpu_pressure` is a pre-existing mislabel, independent of everything above. Noted in the
  README; not touched.

## PR link

<to be filled after push>
