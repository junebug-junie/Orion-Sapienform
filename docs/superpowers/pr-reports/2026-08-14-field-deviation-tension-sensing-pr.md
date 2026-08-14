# Field deviation tension sensing — scale-free admission and rank competition

## Summary

- Restores `DeviationGate` (96 lines, deleted 2026-07-30 in `d6a4e892b` as collateral in the
  drives sweep — it was never drive-coupled) and rehomes it under `orion/attention/tension/`.
- Replaces the deleted `signal_drive_map.yaml`'s 69 lines of hand-authored cross-signal
  weights with `config/attention/channel_direction_map.yaml`: **one sign bit per channel, no
  exchange rates at all**.
- Combines admitted deviations in **rank space** via the existing
  `orion/attention/rank_aggregation.py` Borda primitive — channels are scorers, nodes are
  targets — so a thermal spike never has to be priced against a coherence drop.
- Ships `scripts/analysis/measure_field_tension_admission.py`, the read-only instrument that
  months of drive work never had: admission rate and rank discrimination over real history.
- **Live result over 42,048 real ticks (24h): admission rate 36.47% vs the drives-era
  0.064%; top-1 share 53.13% vs the 96% `relational` monoculture; 9 distinct winners.**

## Outcome moved

The measured precondition failure behind the whole drives program. From
`docs/superpowers/specs/2026-07-07-homeostatic-drives-real-tensions-design.md`: **tensions
fired in 0.064% of ticks (284 of 444,943)**. No taxonomy on top of an input that sparse can
work — which is why successive drive taxonomies each failed the same way, and why the fix is
one layer below where the effort was going.

| | Drives era | This patch |
|---|---|---|
| Admission rate | 0.064% | **36.47%** (~570×) |
| Top-1 share | 96% (`relational` monoculture) | **53.13%** |
| Distinct winners | effectively 1 | **9** |
| Cross-signal weights | 69 lines of YAML | **0** |

## Current architecture (before this patch)

- No tension producer of any kind existed post-`d6a4e892b`. `DriveEngine`, `tensions.py`,
  `signal_drive_map.yaml`, `signal_tension.py`, `tension_ratelimit.py`, and
  `deviation_gate.py` were all deleted 2026-07-30; the accepted consequence recorded in
  `orion/sentience_striving_program/README.md` §8 was that Orion lost goal-proposal capability
  with no field-native replacement.
- `orion/attention/rank_aggregation.py` (Borda) existed with two live consumers, explicitly
  built to combine scorers "without ever guessing/calibrating a cross-scorer exchange rate."
- `substrate_field_state` carried a rich live interoceptive field that nothing was reading for
  motivational purposes — the exact gap named in
  `2026-07-17-field-native-motivational-substrate-design.md`.

## Architecture touched

New leaf package `orion/attention/tension/`, read-only. No service, no bus channel, no schema
registration, no consumer. Home is `orion/attention/`, **not** `orion/autonomy/` — being
adjacent to the taxonomy is what made the gate collateral damage last time.

## Files changed

- `orion/attention/tension/deviation_gate.py`: restored from `d6a4e892b^`. `impulse_k`
  removed — a monotonic scaling cannot change a rank, so it was a tunable that provably did
  nothing downstream.
- `orion/attention/tension/direction_map.py` + `config/attention/channel_direction_map.yaml`:
  structural worse-direction lookup; suffix rules cover 28 of 33 live channels. Refuses to
  load an empty map (a silently-inert map would report a clean run).
- `orion/attention/tension/field_observations.py`: `field_json` → observations, subnormal
  coercion, and the geometric-decay artifact detector.
- `orion/attention/tension/competition.py`: admitted deviations → `aggregate_borda`. Returns
  `borda=None` on a quiet tick — "nothing is happening" is representable, not inferred.
- `scripts/analysis/measure_field_tension_admission.py`: the measurement.
- `orion/attention/tension/tests/`: 44 tests.
- `docs/superpowers/specs/2026-08-14-field-deviation-tension-sensing-design.md`: the spec.

## Schema / bus / API changes

- Added: none. No schema registered, no bus channel added, no payload shape changed.
- Removed / renamed / behavior changed: none.
- Compatibility: nothing consumes this package; it cannot affect any running path.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: not applicable — no env keys introduced. The measurement script
  reuses the existing `POSTGRES_URI` convention from `measure_ast_hot_reducer.py`.
- local `.env` synced: not required (no template change).
- Skipped keys requiring operator action: none.
- New config file `config/attention/channel_direction_map.yaml` is checked in and carries no
  secrets.

## Runtime-truth correction found while building

The 2026-07-07 spec targeted `orion:signals:*` at "~55/s". **That lane is dead as of
2026-08-14** — verified against the live bus (`redis://100.92.216.81:6379/0`): zero pubsub
channels matching `orion:signals*` (497 other channels active), no `orion:signals:*` in the
`orion:bus:velocity:*` census, and no `OrionSignalV1`-shaped Postgres table.
`substrate_organ_emissions` is `organ.emission.v1` grammar events from one organ, not signals.

Building against it would have been config truth. Built against `substrate_field_state`
instead — 127,259 rows, live to the minute.

## Tests run

```text
$ pytest orion/attention/tension/tests -q
44 passed in 0.13s

$ pytest orion/attention -q
43 passed in 0.09s          # pre-fix run; tension suite included

$ pytest tests/test_rank_aggregation.py tests/test_attention_candidate_society_of_mind.py -q
30 passed in 0.37s          # existing Borda consumers, no regression
```

Headline test: `test_ranking_is_invariant_under_monotonic_rescaling_of_a_channel` asserts the
scale-freedom property directly — multiplying one channel by 10× on every node leaves the
Borda ranking, winner, and totals unchanged. A weighted-sum combiner fails this test by
construction, which is the whole argument for rank space.

Deviation fixtures are hand-computed in the test docstrings (e.g. `mu=0.10`, `sigma=0.02`,
`z=(0.50-0.10)/0.02=20.0`, `excess=20.0-1.5=18.5`), not read back off the implementation.

## Evals run

```text
$ POSTGRES_URI=...@localhost:55432/conjourney \
    python3 scripts/analysis/measure_field_tension_admission.py --hours 24

ticks                     42048  (2026-08-13 15:55 -> 2026-08-14 15:55 UTC)
ADMISSION RATE            36.4726%     baseline to beat 0.0640%
TOP-1 SHARE               53.13%       (monoculture if -> 1.0)
distinct winners          9
scorer disagreement rate  25.72%
```

`z_threshold` sweep (10,000-tick window) — confirms a single monotonic knob, not a weight
needing calibration:

```text
z=1.5   admission=0.3901  top1=0.609  disagree=0.272
z=2.5   admission=0.1671  top1=0.548  disagree=0.150
z=3.5   admission=0.0853  top1=0.525  disagree=0.082
z=5.0   admission=0.0608  top1=0.497  disagree=0.059
z=8.0   admission=0.0362  top1=0.461  disagree=0.014
```

Even at the most conservative setting, admission is 56× the drives-era rate, and
discrimination *improves* as the threshold rises — no admission-vs-monoculture tradeoff.

## Repo gates run

```text
$ git diff --check                                  # clean
$ python scripts/check_scripts_dir_no_stdlib_shadow.py   # clean
$ python scripts/check_metric_lineage.py                 # exit 0 (same as main)
$ python scripts/check_definition_drift.py               # no definition changes
```

CLAUDE.md §17's named gates (`check_env_template_parity.py`, `check_schema_registry.py`,
`check_bus_channels.py`) do not exist under those filenames in the tree; the real
`scripts/check_*.py` set was run instead. Worth reconciling §17 with reality separately.

## Docker/build/smoke checks

Not applicable — no service, no container, no runtime wiring touched. The measurement runs
host-side against a read-only Postgres session.

## Review findings fixed

- **Finding (self-review, before the review agent reported): the decay probe was structurally
  incapable of firing.** `geometric_decay_ratio()` rejects any series containing a
  non-positive value, and it was being fed the subnormal-coerced observation value (`0.0`) —
  so it could never detect the decayed-to-zero artifact it exists to detect, while reporting
  "decay-suspect series: none" indistinguishably from genuinely clean data. The first spec
  draft recorded that false-clean result as a finding.
  - **Fix**: `Observation` now carries `raw_value` alongside the coerced `value`; the
    measurement feeds the raw series to the probe.
  - **Evidence**: re-run over the identical 42k-tick window immediately surfaced two real
    instances at **ratio = 0.92 exactly** — `node:circe / reasoning_load` and
    `node:rpc_timeout / reliability_pressure` — the `NODE_DECAY_CHANNELS` staleness-decay
    loop. Regression test:
    `test_raw_value_survives_coercion_so_the_decay_probe_can_still_see_it`.

## Restart required

```text
No restart required.
```

## Risks / concerns

- **Severity: low. Concern:** 36% admission is high enough to ask whether `z_threshold=1.5`
  is the right default for a future consumer. **Mitigation:** the sweep above shows a clean
  monotonic response, and nothing consumes this yet — the default can be set by whichever
  consumer lands first, against its own objective.
- **Severity: medium (pre-existing, not introduced here). Concern:** 560,166 subnormal
  coercions in 24h means the decay-to-zero pathology is widespread, not confined to
  `node:circe`. Channels nothing refreshes read as *calm* to any consumer that does not
  check. **Mitigation:** this patch counts and reports them; it does not fix the producers.
  Needs its own patch against `orion-field-digester`'s decay loop.
- **Severity: low. Concern:** 10 channels never admitted across 42k ticks. Some are
  legitimately quiet, but `transport_pressure` / `execution_load` are renamed-lane survivors
  and `staleness` is known-degenerate. **Mitigation:** named explicitly in the spec's findings
  so none of them gets treated as "confirmed calm" without its own metric-gate pass.
- **Severity: low. Concern:** this produces a tension *signal*, not motivation. Nothing acts
  on it. **Mitigation:** stated as an explicit non-goal; wiring it to anything that acts is a
  separate patch requiring proposal sign-off per CLAUDE.md §0A.

## PR link

<to be filled after push>
