## Summary

- Three offline reports read saved history and print measurements. Orion's running code, configuration, schemas and stores are unchanged.
- Focus reports completed stretch lengths, daily wall-span shares, rolling 30-minute exposure, and wins outside biometrics/bus targets. The original recorder replay remains available.
- Dreams compare saved pressure with actual attempts, separating legacy and novelty formula families. Missing check samples remain missing.
- Arousal prints hourly proposed labels with evidence, plus a separate provisional lease-event reconstruction when GPU snapshots are absent.
- Focused tests and CI cover boundaries, censoring, missing data, hysteresis and read-only export queries.

## Outcome moved

We can now inspect what these three proposed regulators would be responding to before changing behavior. The evidence does **not** authorize a behavior change yet.

### Findings from live history

All times are UTC. Read-only Postgres exports were taken on 2026-10-09, using a fixed end of **05:00 UTC**. Dream/arousal cover the preceding 14 days; focus covers the preceding 24 hours. Queries use a repeatable-read, read-only transaction, a 60-second statement timeout, and rollback. A failed query exits instead of becoming an empty result. Each script prints its own export SQL with `--print-sql`.

**Focus:** only about **16 minutes** of recorded wall spans intersect this 24-hour window, beginning at 04:44. Eighteen stored completed rows intersect the window; one crosses its end. Biometrics accounts for **63.65%** of recorded spans, bus traffic **31.36%**, and execution **4.98%**. Execution wins 3/18 stretches. Among rows entirely inside the window it wins 22/326 observed ticks (**6.75%**); the boundary-clipped bus stretch is deliberately excluded from that denominator. The longest complete, uncensored stretch inside the window is biometrics at **184.09 seconds**. These are goal-provenance node winners, not overall attention. About **98.89% of the requested day is unaccounted for**, not idle. The observed biometrics/bus concentration is worth following, but this is far short of seven post-change days and cannot justify habituation.

**Dreams:** all **52** stored cycles are completed, legacy-formula cycles. Saved pressures range **12.9–128.968**, always above the saved threshold of **3**. Of 51 intervals with a preceding attempt end, **45** start within one ten-minute check after the six-hour minimum; the median gap is **6.00135 hours**. Six start later (about 10–70 minutes after eligibility); their saved idle times are 47–53 minutes, consistent with waiting for quiet, but the missing intervening checks prevent causal attribution. The largest reading is **73.70%** of the legacy weighted ceiling of 175, so the whole instrument was not pinned at its mathematical ceiling even though individual source caps were hit. No between-cycle pressure samples were persisted. A genuine fall-and-rise curve and a usable replacement threshold are **UNVERIFIED**. `main` already contains novelty pressure, but no saved cycle in this window has `new_counts`; the legacy numbers cannot validate that successor.

**Arousal:** the export contains 107 chat rows (including the pre-window seed), 56 successful outreach decisions, 38,662 cabinet readings, 401 GPU events for leases observed backlogged, and 18 focus rows. No GPU state snapshots exist in this export. Strict replay therefore reads **unknown 98.448%**, **strained 1.552%**, and never proves idle or engaged. A separate, explicitly provisional reconstruction assumes complete lease transitions and a clear GPU backlog at the one-hour warmup boundary: **engaged 4.799%, idle 92.700%, strained 2.062%, unknown 0.439%**. Counting all chat rows instead of Juniper's turns raises provisional engaged time to **17.402%**, an extra **42.34 hours** over 14 days. That comparison includes every non-Juniper chat row, not just outreach. Provisional strain exits peak at two in a UTC day, below the proposal's review threshold of 12. This does not validate the provisional inputs.

Inspect [the hourly labels and evidence](../evidence/2026-10-09-regulation-measurements/arousal-hourly.csv) against remembered days. Label columns name the largest share of each hour; minute columns retain mixed hours. The [machine-readable summary](../evidence/2026-10-09-regulation-measurements/summary.json) includes per-target shares, all dream-cycle pressure/timing rows, totals, assumptions and SHA-256 hashes of the local input exports. Neither artifact contains chat text.

**Next step:** collect enough post-focus-change history and evaluate the already-merged novelty pressure on actual saved observations. Exact arousal validation needs stored GPU state history; adding that producer would be a separately scoped instrumentation change. Do not pick habituation, rest or arousal behavior from these incomplete histories.

## Current architecture

- Attention entry point: `services/orion-attention-runtime/app/worker.py`; `app/dominance_runs.py:advance_run` closes goal-provenance spans into `field_dominance_run`. The existing eval only checked recorder equivalence over legacy ticks.
- Dream entry point: `services/orion-dream/app/cycle.py:run_cycle_once`; `read_pressure` calls `app/replay.py:compute_pressure`. `app/cycle_store.py:persist_cycle` stores pressure inside `dream_cycle.cycle_json` only when a cycle runs. The existing novelty backtest simulates source events; it is not a history of actual check readings.
- Arousal is a proposal in [PR #2369](https://github.com/junebug-junie/Orion-Sapienform/pull/2369), revision `4d66ea3446d004abd2efdc67126470103bc09eda`. No running arousal reducer is added here.
- Source contracts: `FieldDominanceRunV1`, `SleepPressureV1`, `GpuPoolStateV1` and existing SQL columns. `orion:gpu_pool:state` is published by `services/orion-gpu-pool/app/runtime.py:snapshot/publish_state`; its catalog lists Hub as consumer, not SQL writer. `snapshot` counts leases with status `backlogged`, which is the meaning reconstructed provisionally from events.
- Heat reuses `orion/autonomy/cabinet_heat.py:read_cabinet_heat` and the saved athena `orion_biometrics_summary.measurements.cabinet_temp_c`. Missing cabinet readings are not hot. Historical AC-low fallback is unavailable.
- Both services have README, `.env_example`, compose, requirements, tests and evals; settings live in **`app/settings.py`**, not at service root. Those deployment/configuration surfaces are unchanged.

## Architecture touched

Offline analysis only. No bus subscriptions, publishing, HTTP calls, model calls, migrations or database writes. Reports read JSONL exports; `--print-sql` prints SELECT-only export transactions for an operator to run. The arousal script imports the existing pure cabinet verdict. The old focus replay imports the existing pure recorder only when selected.

Metric quality gate: these measurements are **not wired into a model or cognition loop**. Source producers and consumers are named above. Focus is excluded as an arousal vote; outreach is excluded from E1; GPU activity/biometrics strain are not added as duplicate heat/backlog votes. Dream pressure counts weighted new material in the successor; the legacy cycle samples cannot prove a homeostatic rest state. Proposed arousal uses the proposal's global gain rationale and timings as a hypothesis, not a validated detector. Live provenance/coverage fails to establish a complete GPU-state series or pressure reset curve; these inputs must pass that gate before any future wiring. Existing recorder replay, dream backtest, cabinet verdict and stored event contracts are reused. Removing these scripts leaves runtime unchanged.

## Files changed

- `services/orion-attention-runtime/evals/replay_focus_runs.py`: extend existing eval with completed-run report and bounded export SQL.
- `scripts/analysis/measure_dream_pressure_crossings.py`: saved pressure/timing report.
- `scripts/analysis/measure_arousal_replay.py`: hourly strict/provisional replay.
- `scripts/analysis/tests/test_regulation_measurements.py`: deterministic report tests.
- `.github/workflows/regulation-measurements.yml`: focused CI.
- This report and its two evidence files: live findings and reviewable hourly output.

## Schema / bus / API changes

- Added, removed, renamed: none.
- Behavior changed: offline reports only.
- Compatibility: original `replay_focus_runs.py history.jsonl` invocation still runs the legacy recorder eval.

## Env/config changes

- Added/removed/renamed keys: none.
- `.env_example` updated: no.
- Local `.env` sync: not applicable; no templates changed.
- Skipped keys requiring operator action: none.
- No Redis connection is made.

## Tests run

```text
/tmp/orion-focus-runs-venv/bin/python -m pytest -q \
  scripts/analysis/tests/test_regulation_measurements.py \
  services/orion-attention-runtime/evals/test_focus_run_replay.py
18 passed
git diff --check: clean
```

## Evals run

All three scripts ran against fresh live exports, yielding the findings above. This is the live evaluation lane, separate from synthetic unit tests. Export and report files are local under `/tmp/orion-{focus,dream,arousal}-{export.sql,live.jsonl,report.json}`. The report artifacts preserve hashes and non-prose evidence.

Reproduce, using one of the three script paths and its exact window above:

```bash
python SCRIPT --print-sql --start START_ISO --end END_ISO > /tmp/measurement.sql
docker exec -i orion-athena-sql-db psql -XqAt -v ON_ERROR_STOP=1 \
  -U postgres -d conjourney < /tmp/measurement.sql > /tmp/measurement.jsonl
python SCRIPT /tmp/measurement.jsonl --start START_ISO --end END_ISO
```

Add `--runs` for the focus report. All three report modes use the standard library and existing pure arithmetic rules. The legacy focus recorder replay and its schema tests require the repository's existing Pydantic dependency. Report thresholds are explicit assumptions: dream `--interval-hours 6 --check-seconds 600 --legacy-cap 50`, arousal `--idle-minutes 45`. No current setting is silently applied to history. Arousal uses a five-second grid, 5-minute sustained backlog, 10 clear minutes to leave strain, cabinet freshness 90 seconds, and GPU freshness 15 seconds. Missing intervals reset continuity; fresh positive strain can win despite other missing inputs.

## Docker/build/smoke checks

Only `docker exec ... psql` reads were used. Runtime builds/restarts are not applicable. Export queries executed successfully against existing live tables; no source-read failures were hidden.

## Review findings fixed

- Finding: a forced manual cycle below threshold incorrectly invalidated the automatic-cycle timer comparison.
  - Fix: threshold consistency now counts only comparable automatic cycles; manual attempts still appear and reset the refractory clock.
  - Evidence: regression test plus independent reviewer rerun, 18 passed.

The required requesting-code-review skill ran in a separate reviewer agent. Final review verified all 336 hourly rows against the totals and the findings against the evidence; no material findings remain. Its preliminary GPU queued-event concern was withdrawn after checking the actual scheduler transitions.

## Restart required

No restart required.

## Risks / concerns

- Medium: only 16 minutes of focus spans in the requested day. Daily percentages disclose unaccounted time; no habituation recommendation.
- Medium: dream check history and GPU snapshots were not persisted. Exact recovery/crossing and arousal claims remain `UNVERIFIED`; provisional output cannot drive behavior.
- Low: bounded history cannot establish every initial hysteresis state. One-hour warmup, stale-input handling and seed assumptions are explicit; historical configuration revisions are not reconstructed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2561
