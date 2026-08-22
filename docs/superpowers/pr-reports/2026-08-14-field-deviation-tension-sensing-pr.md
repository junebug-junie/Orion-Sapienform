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
- **Live result over 42,047 real ticks (24h): admission rate 48.30% vs the drives-era
  0.064%; top-1 share 50.56% vs the 96% `relational` monoculture; 10 distinct winners.**
  (Figures are the post-review re-run; see "Review findings fixed" — the HIGH finding
  invalidated the pre-fix numbers.)

## Outcome moved

The measured precondition failure behind the whole drives program. From
`docs/superpowers/specs/2026-07-07-homeostatic-drives-real-tensions-design.md`: **tensions
fired in 0.064% of ticks (284 of 444,943)**. No taxonomy on top of an input that sparse can
work — which is why successive drive taxonomies each failed the same way, and why the fix is
one layer below where the effort was going.

| | Drives era | This patch |
|---|---|---|
| Admission rate | 0.064% | **48.30%** (~755×) |
| Top-1 share | 96% (`relational` monoculture) | **50.56%** |
| Distinct winners | effectively 1 | **10** |
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
- `orion/attention/tension/tests/`: 69 tests.
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

## Blast radius

- **Runtime: zero.** 14 files, 2,189 insertions, **no modifications to any pre-existing
  file**. Nothing in the repo imports `orion.attention.tension`. No schema, no bus channel,
  no service, no consumer.
- **Dependencies are inbound only.** This package reads
  `orion.field.pressure.HIGHER_IS_BETTER_CHANNELS`; the measurement script reads
  `orion.field.channel_glossary.classify_channel_series` and the digester's
  `NODE_DECAY_CHANNELS`. Changing those affects this code, not the reverse -- deliberately,
  so polarity and decay semantics cannot drift into a private second opinion.
- **`config/attention/` already existed** (`field_attention_policy.v1.yaml`); this adds a
  sibling, not a new convention.
- **The real blast radius was epistemic.** `substrate_field_state` is read by
  `orion-proposal-runtime` (-> `ProposalFrameV1`) and by
  `orion.field.pressure.collect_field_channel_pressures` -- the merged, correctly-polarized
  dict every cognition consumer reads. A merged spec claiming designed-decay channels were
  broken could have driven someone to "fix" the decay loop those consumers depend on,
  including re-adding `prediction_error` to a set it was deliberately removed from.

## Assumption corrections (2026-08-14, after Juniper flagged them)

Four claims in the first version of this PR were wrong. All four are corrected in code, not
just prose.

1. **"decay-suspect: NOT calm, unmaintained" was a 100% false-positive rate.** All 9 named
   channels are in `NODE_DECAY_CHANNELS` and are *designed* to decay at 0.92/tick toward
   rest. The detector was also blind by construction to the real shape -- decay on a channel
   *outside* the set, which is the `prediction_error` case CLAUDE.md documents. Inverted; the
   corrected run reports **zero** undesigned decay and **zero** undesigned pinned findings,
   with the by-design half counted separately as explicitly not findings.
2. **"`orion:signals:*` is dead" was unsupported by the method used.** `pubsub channels`
   lists only channels *with subscribers*; `orion-signal-gateway` is in fact running. The
   defensible claim is narrower and still sufficient: no queryable `OrionSignalV1` history
   exists, and a deviation gate needs persisted history to learn baselines from.
3. **The direction map hand-re-derived an existing constant, as a strict subset.**
   `availability`/`delivery_confidence`/`stream_backlog_health` is
   `orion.field.pressure.HIGHER_IS_BETTER_CHANNELS` minus `confidence` and
   `available_capacity`. Now derived from the constant, with the loader raising if the YAML
   contradicts it. Test: `test_higher_is_better_polarity_is_derived_not_hand_listed`.
4. **Channel liveness was inferred from `never_admitted`, and a validated classifier already
   existed.** `classify_channel_series()` (`never_produced`/`dead`/`ratchet_suspect`/`quiet`/
   `live`) already separates "quiet by design" from "telling you nothing". Reused. 7 of the
   10 `never_admitted` entries are simply at their designed resting state.

A fifth issue was caught while fixing the fourth: classifying liveness off the decay probe's
12-sample tail reported `live: 4, dead: 102` -- a short-window artifact. Sampling at a
200-tick stride across the full window gives **`live: 34, dead: 94, quiet: 21`**. The stride
sampling exists specifically so that artifact was not published.

### The real open defect in this area, which this instrument does NOT detect

Already tracked in `orion/field/pressure.py`: when `services/orion-biometrics` goes quiet,
every input to `resource_pressure` decays toward 0, the dimension reads *calm*, and because
`config/feedback/feedback_policy.v1.yaml` lists `resource_pressure: decrease` under
`positive_delta_channels`, **the in-flight action is credited with a positive outcome for a
producer outage.** That is the genuine decay-vs-calm ambiguity; it needs a producer-liveness
guard, not a ratio check. Tracked in PR #1554's design doc; not addressed here.

## Tests run

```text
$ pytest orion/attention tests/test_rank_aggregation.py -q
75 passed in 0.42s

$ pytest tests/test_rank_aggregation.py tests/test_attention_candidate_society_of_mind.py -q
30 passed in 0.37s          # existing Borda consumers, no regression
```

Headline test: `test_ranking_is_invariant_under_monotonic_rescaling_of_a_channel`, now
parametrized over `k in {10, 0.1, 1000, 0.001}`, asserts the scale-freedom property directly
-- multiplying one channel by any constant on every node leaves the Borda ranking, winner,
totals *and the per-channel ballots* unchanged. A weighted-sum combiner fails this by
construction, which is the whole argument for rank space. Guarded by
`test_the_scale_freedom_fixture_produces_a_genuinely_ordered_ranking`, which fails if the
fixture ever degenerates into the Borda tie the review caught in its first version.

Deviation fixtures are hand-computed in the test docstrings (e.g. `mu=0.10`, `var=0.0`,
`sigma = max(0, 0.01*0.10, 1e-12) = 0.001`, `z = 0.40/0.001 = 400.0`,
`excess = 400.0 - 1.5 = 398.5`), not read back off the implementation.

## Evals run

Post-review-fix re-run. The pre-fix figures (36.47% / 53.13% / 9 winners) are superseded:
the HIGH finding meant the absolute sigma floor was silently excluding every small-variance
channel from the denominator. Data-quality lines reflect the by-design-decay correction.

```text
$ POSTGRES_URI=...@localhost:55432/conjourney \
    python3 scripts/analysis/measure_field_tension_admission.py --hours 24

ticks                     42043  (2026-08-13 16:36 -> 2026-08-14 16:36 UTC)
ADMISSION RATE            48.3838%     baseline to beat 0.0640%
TOP-1 SHARE               50.30%       (monoculture if -> 1.0)
distinct winners          10
scorer disagreement rate  28.42%

liveness (211 strided samples, full window)  {'dead': 94, 'live': 34, 'quiet': 21}
by design, NOT findings   decaying=7  pinned=7
FINDING -- undesigned decay   none
FINDING -- undesigned pinned  none
```

`z_threshold` sweep (10,000 most-recent ticks) -- confirms a single monotonic knob, not a
weight needing calibration:

```text
z=1.5   admission=0.4994  top1=0.520  disagree=0.289
z=2.5   admission=0.2403  top1=0.488  disagree=0.155
z=3.5   admission=0.1350  top1=0.462  disagree=0.092
z=5.0   admission=0.1011  top1=0.451  disagree=0.069
z=8.0   admission=0.0649  top1=0.378  disagree=0.032
```

Even at the most conservative setting, admission is 101x the drives-era rate, and
discrimination *improves* as the threshold rises -- no admission-vs-monoculture tradeoff.

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

Code review ran at `high` in a subagent (CLAUDE.md §12) and returned **11 findings — 1 HIGH,
6 MEDIUM, 4 LOW**. All 11 are fixed. Three of them independently invalidated numbers this
report originally published, so the results tables above are the post-fix re-run, not the
originals.

- **Finding (self-review, before the review agent reported): the decay probe was structurally
  incapable of firing.** `geometric_decay_ratio()` rejects any series containing a
  non-positive value, and it was being fed the subnormal-coerced observation value (`0.0`).
  - **Fix**: `Observation` now carries `raw_value` alongside the coerced `value`.
  - **Evidence**: re-run over the identical window surfaced real instances at ratio = 0.92.

- **HIGH — `deviation_gate.py`: the absolute `sigma_floor` broke the headline scale-free
  claim.** `sigma_floor=0.02` was applied to every channel regardless of its natural scale,
  so any channel whose real variation was smaller could never admit, however large its
  *relative* move — then appeared in `never_admitted` indistinguishable from a genuinely calm
  channel.
  - **Fix**: replaced with `relative_sigma_floor` (fraction of the channel's own `|mu|`) plus
    a `1e-12` numerical backstop. Multiplying a channel by any `k>0` now scales `mu`,
    `sqrt(var)` and the floor identically, so `z` and the rank are invariant.
  - **Evidence**: review measured `node:atlas / memory_pressure` (10,528 ticks, range
    0.1109–0.1171, pstdev 8.8e-4) at **0 admissions of 10,528**. Post-fix it admits **539**
    times in the 24h window. Headline admission rate rose 36.47% → **48.30%** and top-1 share
    *improved* 53.13% → **50.56%**. Tests:
    `test_small_variance_channel_can_still_admit` (hand-computed against the review's own
    live numbers) and `test_admission_is_invariant_under_rescaling_in_both_directions`.

- **MEDIUM — `sigma_floor=0.0` raised `ZeroDivisionError` mid-run**; no validation on `alpha`,
  `z_threshold` or `warmup` either, despite all being public fields and CLI flags.
  - **Fix**: `__post_init__` validation raising immediately on construction; `_MIN_SIGMA`
    backstop makes a zero relative floor safe. Test: 7-case parametrized
    `test_invalid_construction_parameters_raise_immediately`, plus
    `test_zero_relative_floor_does_not_divide_by_zero` and
    `test_zero_mean_baseline_does_not_divide_by_zero`.

- **MEDIUM — the decay probe's ratio tolerance was absolute (`1e-6`), making it blind in the
  subnormal range** it exists to police: subnormals carry ~10 significant bits, so a perfect
  0.92 decay at 3e-321 has a ratio spread of ~1.7e-3.
  - **Fix**: relative tolerance (`2e-2`), comfortably above subnormal quantisation and far
    below any real channel's ratio swing.
  - **Evidence**: mid-decay detections went from 2 to **4**.

- **MEDIUM — a bottomed-out series was unreportable.** A series pinned at a constant subnormal
  has successive ratios of exactly 1.0 (probe rejects: `mean >= 1.0`); review found 15 series
  in the `0 < v < 1e-300` band in a 30-minute window, none flagged.
  - **Fix**: new `subnormal_pinned()` detector reported alongside mid-decay.
  - **Evidence**: **7 pinned series** now surfaced, including `staleness` on three nodes —
    which explains a `never_admitted` entry: **`staleness` never admits because it is dead,
    not because it is calm.** Exactly-0.0 series are deliberately *not* flagged (a genuinely
    resting channel reads 0.0 too, and the two are indistinguishable without pre-decay
    history) — documented as a stated limit rather than a false alarm.

- **MEDIUM — the regression test for the first decay fix passed for the wrong reason.** It
  asserted on `1e-300`, a *normal* float, while naming `3e-321` two lines above, concealing
  the tolerance bug.
  - **Fix**: fixture now uses genuine subnormals and asserts
    `any(0.0 < abs(v) < 1e-308 for v in series)` so it cannot silently regress to a normal
    float again.

- **MEDIUM — `fetch_ticks` read as streaming but psycopg2's default client-side cursor
  buffers the whole result set** — ~58 MB RSS per 2,000 rows, i.e. ~1.2 GB for the documented
  24h run and ~8.5 GB at `--hours 168`, on a host with a documented OOM-freeze incident.
  - **Fix**: named (server-side) cursor with `itersize=2000`, with `autocommit` flipped off
    around it since psycopg2 refuses a named cursor outside a transaction.

- **MEDIUM — the headline scale-freedom test was near-vacuous.** Base Borda totals were
  `{node:a: 1.0, node:b: 1.0}` — a tie decided entirely by alphabetical tiebreak, matching for
  almost any implementation. It also only tested 10× *upward*; at 0.1× the assertion was
  actually false under the old absolute floor.
  - **Fix**: fixture rebuilt to 3 nodes × 3 channels producing genuinely distinct totals
    (a=4, b=3, c=2) with deliberate scorer disagreement on top-1; the test is now
    parametrized over `k ∈ {10, 0.1, 1000, 0.001}` and also asserts the *ballots* are
    unchanged, not just the aggregate. A separate guard test,
    `test_the_scale_freedom_fixture_produces_a_genuinely_ordered_ranking`, fails if the
    fixture ever degenerates back to a tie — it caught the old fixture on first run.

- **LOW — `ORDER BY generated_at ASC LIMIT n` returned the OLDEST n ticks**, so the published
  `z_threshold` sensitivity table measured the *start* of the 24h window, not a recent sample.
  - **Fix**: subquery selecting newest-n then re-sorting ascending (chronological order is
    required by the gate, so plain `DESC` is not the fix). Sweep table above re-run.

- **LOW — `subnormal_coercions` was an observation×tick count presented as evidence of
  breadth.** The spec's "not one node — it is widespread" conclusion cited 560,007, which is
  ~19 dead series × tick count and supports no such claim.
  - **Fix**: `subnormal_distinct_series` added and printed with an explicit note. Real figure
    is **19 distinct series** of ~130; the conclusion survives but the cited evidence did not.

- **LOW — `DirectionMapError` promised loud failure but `unmapped:` as a scalar silently
  became a frozenset of characters**, leaving the named channel mapped and voting; a non-dict
  `channels:` raised a bare `AttributeError`.
  - **Fix**: per-section type validation before iteration. Tests:
    `test_unmapped_as_scalar_raises_instead_of_becoming_a_set_of_characters`,
    `test_channels_as_list_raises_direction_map_error_not_attribute_error`,
    `test_suffix_rules_as_list_raises`, `test_non_mapping_top_level_raises`.

- **LOW — `competition.py`'s docstring described abstention behaviour that does not exist.**
  A quiet channel submits no ballot at all (the key is never created), and an actually-empty
  ballot would hand every target the same `(n-1)/2` points — harmless as a constant offset,
  not as abstention.
  - **Fix**: docstring corrected, with the old claim recorded and the real
    empty-ballot behaviour spelled out for future callers.

### Review notes accepted without change

- The gate is a faithful restoration of `d6a4e892b^` minus `impulse_k`; West's incremental
  EWMA variance form is correct.
- **The entire exact-`channels:` half of the YAML is currently inert on live data** —
  `availability`, `delivery_confidence`, `stream_backlog_health` all pinned at exactly 1.0,
  `staleness` subnormal/0.0. Only the suffix rules contribute today. Recorded in the spec's
  findings; worth knowing before treating the map as validated.

## Restart required

```text
No restart required.
```

## Risks / concerns

- **Severity: low. Concern:** 48% admission is high enough to ask whether `z_threshold=1.5`
  is the right default for a future consumer. **Mitigation:** the sweep above shows a clean
  monotonic response, and nothing consumes this yet — the default can be set by whichever
  consumer lands first, against its own objective.
- **Severity: low (corrected from medium). Concern:** 19 distinct series read subnormal, but
  all of them are in `NODE_DECAY_CHANNELS` and at their designed resting state. The original
  framing of this as a widespread pathology was wrong. **Mitigation:** none needed for the
  decay itself; the genuine related defect (producer-outage-reads-as-calm) is named above and
  tracked elsewhere.
- **Severity: low. Concern:** 10 channels never admitted across 42k ticks, and the entire
  exact-`channels:` half of the direction map is currently inert on live data
  (`availability` / `delivery_confidence` / `stream_backlog_health` all pinned at exactly
  1.0). Only the suffix rules contribute today, so the map is not yet validated end to end.
  **Mitigation:** named explicitly in the spec's findings so nothing gets treated as
  "confirmed calm" without its own metric-gate pass; `staleness` is already proven dead
  rather than calm by the pinned-subnormal detector.
- **Severity: low. Concern:** this produces a tension *signal*, not motivation. Nothing acts
  on it. **Mitigation:** stated as an explicit non-goal; wiring it to anything that acts is a
  separate patch requiring proposal sign-off per CLAUDE.md §0A.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1675
