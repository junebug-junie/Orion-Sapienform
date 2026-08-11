# Patch A — stop routing `thermal_pressure` into the `resource_pressure` dimension

Date: 2026-08-11
Branch: `fix/thermal-pressure-dimension-unmap`
Design doc: PR #1554 (`docs/proposal-base-priority-deletion`) — **not merged as of this commit**

## Summary

- Removes one entry from `orion/field/pressure.py::CHANNEL_DIMENSION_MAP`. Measured over 28,735 real
  ticks, `thermal_pressure` won the `max()` merge into `resource_pressure` on **91.76% of ticks** —
  a 39-distinct-value quantized reading of one CPU's hottest core overwriting a 1,895-distinct-value
  composite of five independent live capability channels.
- Re-derives `DIMENSION_PRECISION_MIN_VARIANCE["resource_pressure"]` `5e-5` → `2e-3`, mandatory
  because the change moves the distribution the old floor was calibrated against.
- Fixes **seven** stale `self_state_dimension` entries in `config/field/field_channel_glossary.v1.yaml`
  and adds a fail-closed parity gate. Six of the seven were pre-existing, documenting routes killed
  in the 2026-07-22 SelfStateV1 burn; nothing had asserted this file against the map.
- Adds three regression tests, including one that documents a fragility this patch introduces rather
  than claiming it resolved.
- **Retracts a justification** the first version of this patch shipped with. See below.

## Outcome moved

`resource_pressure` measured through the real production path
(`scripts/analysis/measure_proposal_dimension_variance.py`, which parses each historical row into a
real `FieldStateV1` and runs the unmodified `field_pressures()`; 6h / 10,562 ticks, before vs after
on the same window):

| | before | after |
| --- | --- | --- |
| variance | 0.0037973 | **0.0174205** (4.6x) |
| mean | 0.538208 | 0.32551 |
| min | 0.314286 | 0.144535 |
| distinct values (24h window) | 128 | **1,895** (15x) |
| real upward transitions | 270 | **431** (+60%) |
| classification | `NON_DEGENERATE_USABLE` | `NON_DEGENERATE_USABLE`, no decay artifact |

**This patch does not move Work Outcome O1.** Per-tick max across all four scored dimensions goes
from min 0.3035 to min 0.1265, and **0.00% of ticks read below `min_priority` before or after.** The
arena still proposes on every tick. That is Patch B (removing `_pressure_dimension_ids()`'s
fallback), deliberately not bundled here so this patch's post-deploy effect stays attributable.

## Current architecture

`resource_pressure = max(thermal_pressure, pressure)`, merged by
`orion/field/pressure.py::map_channels_to_dimensions` after
`collect_field_channel_pressures` has already taken a `max()` over every node and capability.
`thermal_pressure` is `(T − 50) / (85 − 50)` on the hottest core
(`orion/telemetry/biometrics_pipeline.py::normalize_thermal`, `THERMAL_MIN_C`/`THERMAL_MAX_C`).

## Architecture touched

One routing entry and one calibration constant. No service boundary, contract, or schema moves.

## Files changed

- `orion/field/pressure.py`: removed the `thermal_pressure` → `resource_pressure` entry; tombstone
  recording the measurement, the retracted justification, and the still-open decay defect
- `orion/proposals/scoring.py`: `resource_pressure` precision floor `5e-5` → `2e-3`, with the
  derivation, the EWMA replay verification, and the three known-stale sibling floors recorded inline
- `config/field/field_channel_glossary.v1.yaml`: removed 7 stale `self_state_dimension` entries
- `tests/test_field_channel_glossary.py`: new fail-closed parity gate against `CHANNEL_DIMENSION_MAP`
- `tests/test_feedback_extractors.py`: rewrote the test that asserted the removed mapping; added an
  inverse regression guard and an absent-not-zero contract test

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: `field_pressures()["resource_pressure"]` no longer reflects CPU temperature. Same
  units, same direction, same `[0,1]` range, same key.
- Compatibility notes: no persisted schema changes. Historical `substrate_field_state` rows are
  unaffected; only their interpretation going forward changes.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: **not applicable, no env keys touched**
- local `.env` synced: **not applicable, no env keys touched**
- skipped keys requiring operator action: none

## Tests run

```text
$ PYTHONPATH=. pytest tests/test_feedback_extractors.py tests/test_field_channel_glossary.py \
    tests/test_proposal_scoring.py tests/test_proposal_frame_builder.py \
    tests/test_proposal_policy_loader.py tests/test_proposal_transport_readonly_candidates.py \
    tests/test_execution_dispatch_envelopes.py tests/test_measure_proposal_feedback_correlation.py \
    tests/test_phi_encoder_fit_script.py services/orion-field-digester/tests scripts/analysis/tests
607 passed in 10.11s
```

Pre-existing and NOT caused by this patch: `pytest tests/` as a whole fails collection with 32
errors on clean `main` as well (verified by stashing) — an `app.*` package-name collision where every
service's `app` resolves to `orion-attention-runtime`'s.

## Evals run

```text
$ PYTHONPATH=. POSTGRES_URI=... python scripts/analysis/measure_proposal_dimension_variance.py \
    --window-hours 6
```

Run twice — once on unpatched code, once on patched, same 6h / 10,562-tick window. Results in
"Outcome moved" above. Artifacts: `/tmp/measure-proposal-dimension-variance/report.md`, `ticks.csv`.

EWMA floor verification (replaying the real post-patch series through the production
`orion.bus.ewma.compute_ewma_update`):

```text
floor      p99|z|   max|z|   %|z|>=3   median
5.0e-05     3.131   17.195     1.16     0.741   <- old
2.0e-03     3.116   10.023     1.13     0.741   <- chosen
5.0e-03     3.042    7.493     1.06     0.735
```

`2e-3` bounds the tail without flattening discrimination; p99 stays inside the 1.3–4.3 band the four
original floors were verified against. Raw unfloored variance reaches ~0 during flat holds, so the
floor is load-bearing.

## Docker/build/smoke checks

```text
Not run. No Dockerfile, compose, dependency, port, or health-check change.
```

The deploy hazard is ordering, not building — see "Restart required".

## Review findings fixed

- **Finding:** `config/field/field_channel_glossary.v1.yaml` still declared `thermal_pressure` →
  `self_state_dimension: resource_pressure`, a live contract surface rendered verbatim in Hub's
  glossary panel (`field_channel_glossary_routes.py:128`). No test asserted parity.
  - **Fix:** removed. Investigating it found **six more** stale entries (`availability`,
    `available_capacity`, `confidence`, `expected_offline_suppression`, `field_coherence_warning` →
    `coherence`; `prediction_error` → `uncertainty`) documenting routes deliberately dropped in the
    2026-07-22 burn — `pressure.py`'s own docstring says those "produced values nothing ever read."
    All seven removed, plus a fail-closed exact-equality parity gate.
  - **Evidence:** `tests/test_field_channel_glossary.py::test_self_state_dimension_matches_channel_dimension_map_exactly`; 14 passed.

- **Finding:** the shipped comments cited a design-doc path that does not exist anywhere in the repo,
  and `scoring.py` said three known-stale floors were "tracked as follow-up work in the design doc" —
  i.e. tracked nowhere.
  - **Fix:** cite PR #1554 by number, state explicitly that it is unmerged, and restate every
    load-bearing number inline so no comment depends on the doc resolving. Added why
    `execution_pressure`'s 20.6x-too-low floor matters specifically (highest `dimension_weight` at
    0.30 *and* a `_pressure_dimension_ids()` fallback dimension).
  - **Evidence:** `orion/field/pressure.py:68-77`, `orion/proposals/scoring.py:111-152`.

- **Finding (most serious):** the structural justification was false. The comment disqualified
  `thermal_pressure` for being in `NODE_DECAY_CHANNELS`, but capability `pressure` is produced by
  `apply_diffusion` from `cpu/memory/gpu/disk/stream_backlog_pressure` + `prediction_error` — **all
  also in `NODE_DECAY_CHANNELS`** — and `pressure` itself is in `CAPABILITY_DECAY_CHANNELS`. Removing
  thermal does not remove the decay ambiguity.
  - **Fix:** justification retracted in place, kept visible rather than deleted. The patch now rests
    on the measured resolution recovery alone. The decay defect is recorded as still-open and
    dimension-wide: if `orion-biometrics` goes quiet, every remaining input decays toward 0,
    `resource_pressure` reads calm, and because `feedback_policy.v1.yaml:34` lists
    `resource_pressure: decrease` under `positive_delta_channels`, the in-flight action is **credited
    with a positive outcome for a producer outage.** Needs a liveness guard on the dimension, not a
    different input channel. Not fixed here.
  - **Evidence:** `config/field/orion_field_topology.v1.yaml:57,58,99,106,110,118,119,126,133,140`;
    `services/orion-field-digester/app/digestion/decay.py:28-45,62-69`.

- **Finding:** no restart note shipped, but the score and its confidence are computed in two
  different containers from the same constants.
  - **Fix:** ordered redeploy documented below.
  - **Evidence:** see "Restart required".

- **Finding:** the new regression test put `pressure` in `node_vectors`, but `pressure` is a
  capability-only channel — it exercised the wrong merge branch and would have passed with the
  capability merge entirely broken.
  - **Fix:** rewritten against `capability_vectors`.
  - **Evidence:** `tests/test_feedback_extractors.py::test_thermal_pressure_does_not_route_to_resource_pressure`.

- **Finding:** `resource_pressure` is now wholly dependent on the diffusion layer; if
  `apply_diffusion` produces no `pressure`, the dimension vanishes, `dimension_confidence` → 0.0, and
  `dimension_precision_ewma_n` freezes without self-recovery.
  - **Fix:** documented with an explicit test asserting the absent-not-zero contract. Not claimed
    resolved — the coupling is real and now reviewable.
  - **Evidence:** `tests/test_feedback_extractors.py::test_resource_pressure_absent_when_diffusion_produces_no_pressure`.

Two things the review checked and cleared, recorded so they are not re-litigated: the post-deploy
EWMA baseline transient is only 2–6 digestion ticks; and `test_proposal_scoring.py`'s cold-start
sweep is insensitive to the 40x floor change (worst `|z|` at n=8 is 2.758 either way), so the n=8
guarantee is not weakened.

## Restart required

**Both services must be redeployed together, and the digester first.** `orion-field-digester`
(`app/digestion/precision.py`) computes and persists `dimension_precision_zscore` using
`field_pressures()` and `DIMENSION_PRECISION_MIN_VARIANCE`; `orion-proposal-runtime`
(`orion/proposals/builder.py:209`) independently recomputes `field_pressures()` and *reads* the
digester's persisted z-score. Both `Dockerfile`s `COPY orion /app/orion`, so both bake these
constants. If they skew, every `resource_pressure` confidence in the window scores a new-distribution
value (mean 0.3255) against a baseline built on the old thermal-dominated one (mean 0.538) —
systematically wrong, silently, for as long as the skew lasts.

```bash
scripts/safe_docker_build.sh orion-field-digester up -d --build
scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
```

Expect `dimension_precision_ewma` for `resource_pressure` to re-converge over roughly 2–6 digestion
ticks; the persisted `dimension_precision_ewma_n` counter is not reset, so confidence stays above the
8-sample floor throughout.

## Risks / concerns

- **Severity: medium.** The decay ambiguity is unresolved and is now the dimension's only failure
  mode, with a live feedback consumer that mis-credits actions during a producer outage. Mitigation:
  documented in-code and in the design doc; needs a liveness guard as a follow-up. Not introduced by
  this patch — it was there before and this patch removes the input that partially masked it.
- **Severity: medium.** The other three precision floors are stale by up to 20.6x
  (`execution_pressure` 20x too low, `reasoning_pressure` 6x too high, `reliability_pressure` 2x too
  high), measured on unpatched code. Deliberately not fixed here to keep this patch attributable.
  Mitigation: numbers recorded inline in `scoring.py`; re-derivable by re-running the same script.
- **Severity: low.** `resource_pressure` now depends solely on the diffusion layer. Mitigation:
  contract test above.
- **Severity: informational.** `scripts/analysis/measure_autonomy_gate.py` reports `UNMEASURABLE`
  (0 `self_state` rows since the 2026-07-22 burn) — a dead instrument, unaffected by this patch but
  worth killing or fixing. Separately, the active phi encoder
  (`/mnt/telemetry/models/phi/encoders/active` → `v20260712-seedv4-postfix`) trains on
  `agency_readiness` and `execution_load` and probes `field_intensity` — all dead or renamed. It does
  **not** use `resource_pressure`, so it is unaffected here.

## PR link

_to be filled on open_
