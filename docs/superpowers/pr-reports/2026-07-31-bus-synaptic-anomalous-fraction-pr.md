# bus_synaptic_prediction_error: count the anomalous fraction, retune the consumer

Branch: `fix/bus-synaptic-anomalous-fraction`
Date: 2026-07-31 (original work 2026-07-30, rebased onto 59 commits of `main`)
Status: **DONE**

## Summary

- Replaces `bus_synaptic_prediction_error()`'s magnitude formula with a **counting** one: the
  fraction of live bus-synaptic edges at `|z| >= 3.0`.
- This is the **third** shape for this metric. The two before it both failed CLAUDE.md §0A's metric
  quality gate step 4, in opposite directions — one pinned at 1.0, one pinned at 0.0.
- `_BUS_SYNAPTIC_CALM_FLOOR` (`sqrt(2/pi)`) is **deleted**, not left unused — it corrected a real
  bias in the formula it served, and that formula is gone. "Kill means kill."
- Consumer migrated in the same changeset (§6): equilibrium's threshold `1.0 -> 0.15` across
  `settings.py`, `.env_example`, the compose inline default, README, and the live `.env`.
- Rebase resolution keeps PR #1533's edge-trigger guards alongside the retuned threshold, and
  documents that they compose.
- Fixes **two tests that are red on `origin/main` today**, both belonging to this threshold.

## Outcome moved

`node:substrate.bus_synaptic` stops being a metric whose reading is decided by whichever single edge
currently has the worst tail value. Live measurement on the real `orion_bus_synapse` graph:

```text
median(|z|)                  0.399
p90                          1.123
p99                          7.375
max                       7087.8     <- one 9-hour-stale cortex-orch edge
mean(|z|)     [shape 1]     29.278   -> min(1, 29.3/3.0) = 1.0000  (pinned high)
```

28.6 of that 29.278 came from that one edge. `mean()` over an unbounded heavy tail is a disguised
`max()`.

## The two failed shapes, kept on the record

**Shape 1 — `mean(|z|)`, saturating at 3.0.** Pinned at 1.0, driving continuous false "Bus Anomaly
Detected" alerts through equilibrium's transport gate.

**Shape 2 — clamp per edge, then average, then subtract the calm floor.** Fixed the tail, broke the
floor. Caught in code review before merge and independently reproduced:

```text
live substrate-tick edge set: n=222
  mean|z| raw         13.5025
  mean|z| clamped      0.5575
  calm floor           0.7979     (= sqrt(2/pi), E|Z| for a STANDARD normal)
  headroom            -0.2404     <- negative: pinned at exactly 0.0
```

The real population is narrower than unit normal — its z-scores divide by an EWMA variance the same
outliers inflate — so the theoretical floor over-subtracted. 19 of 222 edges had to hit 3σ
simultaneously for any non-zero reading, and the consumer's 1.0 threshold required all 222.

## Why counting is immune by construction, not by calibration

- **Bounded [0, 1] with no clamp** — it is a proportion.
- **Robust to any tail**: one edge at `|z| = 7087` counts exactly the same as one at `|z| = 3.01`,
  namely `1/N`. Shape 1's bug cannot recur.
- **Theory-anchored rest point** rather than a fitted constant: for a calm standard-normal edge
  population it is `P(|Z| >= 3) = 0.0027`.

### Metric quality gate (§0A), run in order

1. **Provenance.** Per-edge `gap_zscore` is written by
   `services/orion-bus-mirror/app/graph_writer.py::compute_ewma_update` on real inter-service
   publish traffic — a genuine rolling EWMA baseline per edge, not a derived aggregate.
2. **Independence.** No other Active-Inference domain reads bus publish cadence. `execution` reads
   dispatch outcomes, `biometrics` node telemetry, `chat`/`route` grammar events.
3. **Theory anchor.** Named, not vibes: the anomalous-fraction rest point is `P(|Z| >= 3)` for a
   calm normal population.
4. **Live-data sanity.** 60 samples over 10 minutes on the real mesh: median 0.026, mean 0.027,
   p95 0.072, max 0.094, across 24 distinct values — never 0.0, never 1.0. ~10x the normal-theory
   value, consistent with the known heavier-than-normal tail.
   An early **2-minute** sample suggested max 0.043, and "~3.5x margin" had already been written into
   four files on that basis. The 10-minute sample found 0.094, making it 1.6x. Every cited number
   was corrected. Same "window too thin to rank" trap the precision-weighted-attention spec hit once.
5. **Existing mechanism.** `zscore_threshold=3.0` reuses
   `services/orion-hub/scripts/bus_synaptic_graph_routes.py::anomalies()`'s live convention.
6. **Reversibility.** Cheap: one function body, one threshold constant, no schema or manifest
   surface.

## Disclosed structural limit, asserted as a test

This is a **mesh-wide** detector and structurally cannot resolve a single-organ failure:

```text
busiest organ (orion-social-memory, 12 of ~235 edges), all anomalous -> 0.051
three busiest organs together, all anomalous                        -> 0.136
observed baseline max                                                  0.094
```

Baseline-to-few-organ separation is ~1.45x, so **no** threshold on a mesh-wide fraction separates
them. It reliably resolves broad events (>=15-20% of edges). Few-organ detection needs a per-organ
signal — `bus_synaptic_graph_routes.py`'s `/propagate` already walks that blast radius. Asserted as
a test so a future patch has to confront the limit rather than quietly lowering the threshold into
the noise band.

## Rebase resolution (2026-07-31)

`origin/main` moved 59 commits, including PR #1533, which added edge-triggering to the same gate.
Two conflicts, both in equilibrium config, both resolved by keeping **both** changes:

- `.env_example` / `docker-compose.yml`: retuned threshold `0.15` **and** `CLEAR_RATIO=0.8`.
- Documented at the config site that they compose: re-arm now sits at `0.15 * 0.8 = 0.12`, still
  above the measured baseline max of 0.094, so a calm mesh does re-arm. That margin is **1.28x**,
  tighter than the 1.6x on the fire threshold. Called out rather than hidden — if the baseline drifts
  up, the hysteresis band is what gives first.

## Two tests fixed that are red on `origin/main`

Both live in `services/orion-equilibrium-service/tests/test_bus_synaptic_poll_e2e.py`, and both are
this threshold's own tests, so they are in scope rather than drive-by.

1. `test_poll_above_threshold_triggers` fed `error=0.87` against a threshold of `1.0` and asserted a
   trigger fires — the "above threshold" case was never above the threshold. All four mock values
   were written against the old magnitude scale, one of them `1.2`, which the fraction metric cannot
   produce at all. Retuned to `0.05 / 0.15 / 0.40 / 0.60`.
2. `test_trigger_carries_edge_count_and_context` asserted `assertIn("reason", trigger)` — a
   membership test against a pydantic model, not a field check. It had **never run**, because the
   `assertIsNotNone` above it failed first. Now checks the field directly.

Baseline established in a detached scratch worktree at `origin/main`:

```text
origin/main : 2 failed, 4 passed
this branch : 6 passed
```

## Files changed

- `orion/substrate/prediction_error.py`: counting formula; `_BUS_SYNAPTIC_CALM_FLOOR` and the now-unused
  `math` import deleted.
- `orion/substrate/tests/test_prediction_error.py`: fraction semantics, tail-robustness, and the
  disclosed mesh-wide limit.
- `services/orion-equilibrium-service/app/settings.py`, `.env_example`, `docker-compose.yml`,
  `README.md`: threshold `1.0 -> 0.15`, with the measurement behind it recorded at the config site.
- `services/orion-equilibrium-service/app/transport_metacog_gate.py`: docstring migrated to the new
  units; auto-merged cleanly with #1533's `previously_above` / `node_age_sec` params.
- `services/orion-equilibrium-service/tests/test_bus_synaptic_poll_e2e.py`: the two fixes above.
- `services/orion-substrate-runtime/tests/test_worker_bus_synaptic_tick.py`: fraction-scale fixtures.

## Schema / bus / API changes

None. `MetacogTriggerV1` shape unchanged; only the numeric range of `upstream.error` changes meaning,
and its sole consumer (the threshold) is migrated in this same changeset.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- Behavior changed: `EQUILIBRIUM_METACOG_TRANSPORT_BUS_SYNAPTIC_ERROR_THRESHOLD` default `1.0 -> 0.15`.
- `.env_example` updated: yes.
- local `.env` synced: **must be set to `0.15` at deploy time.**
  `scripts/sync_local_env_from_example.py` deliberately does not overwrite an existing override, so
  the live value has to be edited alongside the deploy. Leaving it at `1.0` under the new metric
  makes the detector structurally unreachable (it would require every edge in the mesh anomalous at
  once) — a silent permanent-silence failure, the exact thing shape 2 was rejected for.
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-equilibrium-service$ pytest tests -q
171 passed, 14 warnings in 3.35s          (was 2 failed / 169 passed on origin/main)

$ pytest orion/substrate/tests/test_prediction_error.py -q
61 passed

services/orion-substrate-runtime$ pytest tests/test_worker_bus_synaptic_tick.py -q
9 passed
```

## Evals run

```text
No eval harness exists for services/orion-equilibrium-service or orion/substrate.
```

Flagged, not claimed. The behavior this patch changes is directly observable in the live node value
and in `orion_metacog` row volume, which is the meaningful check.

## Docker/build/smoke checks

Not yet deployed — pending merge and the live `.env` edit above. To deploy:

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
```

`orion-substrate-runtime` needs the rebuild too: it is what computes the metric.

## Review findings fixed

Carried forward from the original review of this branch, plus the rebase pass:

- **Finding (must-fix): shape 2 pins the metric at exactly 0.0 and makes the alert unreachable.**
  - Fix: shape 2 was backed out of PR #1516 entirely; this branch is the replacement.
  - Evidence: clamped mean `0.5575` vs floor `0.7979`, headroom `-0.2404`.
- **Finding (must-fix): consumer threshold must migrate in the same changeset.**
  - Fix: done here across five surfaces plus the live `.env` note above.
- **Finding (should-fix): the "~3.5x margin" claim rested on a 2-minute window.**
  - Fix: re-measured over 10 minutes; every derived number corrected to 1.6x.
- **Finding (rebase): #1533's hysteresis band was not re-checked against the new threshold.**
  - Fix: computed and documented — `0.12` re-arm vs `0.094` baseline max, 1.28x.
  - Evidence: recorded at the `.env_example` config site, not only here.

## Restart required

```bash
# after merge, and after setting the live .env threshold to 0.15
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
```

## Risks / concerns

- **Severity: should-know. The hysteresis re-arm margin is 1.28x, not the 1.6x on the fire
  threshold.** If mesh baseline drifts upward past 0.12, the detector fires once and never re-arms —
  a silent-failure mode. Worth a follow-up that derives `CLEAR_RATIO` from measured baseline rather
  than a fixed 0.8.
- **Severity: should-know. Single-organ failures are invisible to this metric, by construction.**
  Disclosed above and asserted as a test. Do not lower the threshold to chase them.
- **Severity: note. `orion_metacog` still has no consumer.** Reducing false transport triggers makes
  the table less noisy but does not make it read.

## PR link

<pending>
