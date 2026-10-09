# EWMA-derived transport thresholds: metric gate (Step 0)

Date: 2026-09-30. Data: `substrate_field_state`, 124,654 rows, 2026-09-27 06:13 to 2026-09-30 07:09 UTC
(the table only holds ~3.4 days), one row per ~2 s tick. Pulled to a CSV and analysed offline.

## Verdict per channel

| Channel (reads) | Verdict | Why |
|---|---|---|
| `bus_synaptic_pressure` (M4 `capability:transport.pressure`) | WIRED | varies, rests low, not saturated, not a decay artifact |
| `contract_pressure` (M4 `contract_pressure`, fed by `catalog_drift_pressure`) | NOT WIRED | flat: 5 distinct values in 3.4 days, 0.0027 for the median tick, max 0.0136. One quantised step (1/367). z-score of a constant is noise. |
| `observer_failure_pressure` (M4 `reliability_pressure`) | NOT WIRED | 96.1% exact zeros, 106 distinct values, longest identical run 45,426 ticks (~25 h). Zero-inflated, event-only spikes (max 0.58); a mean/sd baseline of a spike train is not a usable "normal", and event-triggered stats cannot detect absence. Static thresholds already fire on it (0.34% of ticks >= 0.25). |

## bus_synaptic_pressure, gate steps

1. **Provenance.** `orion/substrate/prediction_error.py::bus_synaptic_prediction_error` = fraction of live bus-synaptic edges with |z| >= 3, written every 30 s by `services/orion-substrate-runtime/app/worker.py::_bus_synaptic_tick` as node `prediction_error`. M4 `pressure` is exactly 0.85 x that (ratio min/median/max 0.85/0.85/0.85 over all rows; topology edge weight 0.85). The producer of the new baseline therefore stores `error * 0.85`, and a test pins 0.85 to the topology yaml.
2. **Independence.** No other model metric is added; the derived threshold is a function of this channel's own history only. Caveat: the channel is already a per-edge z-score aggregate, so a second EWMA on top is a baseline of a baseline. That is acceptable here because the inner z is per edge (mesh-wide fraction), the outer clock measures the chronic level of that fraction.
3. **Theory anchor.** Adaptive control-chart baseline: mean + k*sd of an exponentially weighted history (EWMA chart, Roberts 1959), with a static hard floor so the chart's known failure mode (baseline absorbs a sustained shift) cannot silence the channel. Rest point for a calm normal edge population is P(|Z|>=3) = 0.0027 (the function's own docstring).
4. **Live data.**
   - Not degenerate: 1,383 distinct values, min 0.0, p10 0.007, p50 0.024, p90 0.079, p99 0.199, max 0.389. Exact zero only 1.38% of ticks.
   - Can rest: hourly means sit 0.025-0.076 with no upward trend; signed fast z median -0.34 (skewed metric), p10 -0.71, p90 +1.09. It sits in a low band and returns to it; it is not pinned at a floor.
   - Not a decay-to-zero artifact: ratio of successive distinct values has median 0.95 and p10/p90 0.47/2.1; only 2.1% fall in [0.91, 0.93] (a 0.92 decay would be ~100%). The node is rewritten every tick (see worker comment), so it is refreshed, not decayed.
   - Not mean|z| floor: it is a counting metric (no ~0.8 floor). Median 0.024 is the real rest level, about 9x the 0.0027 normal-theory value (heavy-tailed edges, documented in the function).
   - Absence: the producer tick refreshes every 30 s. The stored state carries `last_ts`; readers treat state older than 600 s as `stale_state` and fall back to static. A dead producer cannot look calm and cannot freeze a learned threshold.
   - Distribution is right-skewed: |z_fast| > 3 on 3-4% of samples, so z alone is a poor trigger. That is why the derived threshold is built from the slow mean and sd, and z_fast is reported, not gated on.
5. **Existing mechanism.** Reuses `orion.bus.ewma.compute_ewma_update` (half-life alpha as in `orion/field/queue_contention.py`). No percentile job, no new stats module. `services/orion-field-digester/.../precision.py` state lives inside FieldStateV1, which the task forbids touching; hence a small Redis hash.
6. **Reversibility.** One env flag (`TRANSPORT_THRESHOLDS_DERIVED_ENABLED=false`) restores pure static yaml in hub and mind; no schema, no yaml change, no persisted default. Redis key can be deleted.

## Replay of the real 3.4 days (30 s samples, 2 d half-life, warm after 24 h and >= 2880 samples)

Run through the shipped `update_state`/`effective_thresholds`. Slow mean 0.037, slow sd ~0.04 (heavy right tail). Fraction of post-warm ticks at or above the watch rung (static 0.25 fires 0.19%):

| k_watch | derived watch | fires |
|---|---|---|
| 3 | 0.15-0.17 | 2.37% |
| 4 | 0.19-0.21 | 0.88% |
| 5 (default) | 0.23-0.25 | 0.32% |

Default k=5 (summarize k=7, propose k=9), tightening capped at 2x below static. Honest read: on this history k=5 tightens the watch rung only slightly (0.23-0.25 against 0.25). k=4 is the visible-but-still-rare setting; that is Juniper's call, it is a knob not a finding. An earlier draft of this replay showed 0.19-0.22 at k=5; code review found the seeded EWMA kept ~71% of its weight on the first sample at 24 h, so the sd was underestimated. Fixed by flooring alpha at 1/n during warm-up (a plain running mean until the exponential window is deeper than the history), and a test pins first-sample independence.

Caveat: only 3.4 days of history exist; a weekly cycle cannot be seen yet. Cold start needs both 2880 samples and 28,800 s of real elapsed time.

## Traps checked

- mean|z| floor: not applicable (proportion metric; documented rest 0.0027, observed 0.024).
- Event-triggered EWMA cannot detect absence: stale-state rule above; the tick writes every 30 s including calm ticks.
- Decayed-to-zero vs calm-at-zero: ratio check above.
- Baseline absorbs a hot channel: test `test_chronic_hot_channel_never_reads_calm` (calm 3 d, then 0.6 for 6 d: z_fast drops under 0.5 while thresholds stay <= static and 0.6 still trips every rung).

## Known pre-existing mismatch (not changed)

The mind resolver renders `latest` from node `prediction_error` (0..1) against rungs defined on the M4 pressure scale (0.85x). Left as is; the thresholds themselves are the same numbers in hub and mind.
