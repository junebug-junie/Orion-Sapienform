# Biometrics: offline nodes lose all pressure; standby nodes stop diluting the body signal

## Summary

- An offline node's pressure can now clear completely. Before, the "suppress" step cleared only `strain`. An `availability` pressure stayed forever, and the frozen hints put `strain` back on the next sweep.
- Suppression also resets `pressure_score` to 0. The suppression sweep now also picks up a node whose only leftover is a non-zero score. Live hecate had sat at score 1.0 since 2026-10-15, and nothing could select it.
- A node that has stopped reporting is now left out of the biometrics surprise number (`node:substrate.biometrics` prediction error). The example is hecate in BMC-only standby. "Stopped" means its last report is more than 6x its own report interval behind the newest report from any node. There is no fixed number of seconds and no hand-kept flag.
- The prediction-error definition for `node_biometrics` moves to version 2, so the attention baseline restarts on the new scale.
- Each change has a kill switch, and both are ON by default.

## Outcome moved

- Before: the biometrics surprise number read 26% calmer than the body. It averaged 35 keys, and 9 of them were hecate's frozen values, each adding a 0.0 change. After: it averages the 26 keys that are actually reported, a factor of 35/26 = x1.346.
- Before: an expected-offline node holding `availability` was swept on every tick forever. After: it is suppressed once, then the sweep goes quiet.

## Current architecture

- `orion/substrate/biometrics_loop/pressure_organ.py` turns node state into pressure candidates:
  - Rule A: an expected-offline node that has gone stale gets "suppressed".
  - Rules C/D/E: reinforce, decay or capability, based on `pressure_hints`.
- `pressure_reducer.py` applies those candidates to `substrate_active_node_pressure_projection`.
- `services/orion-substrate-runtime/app/worker.py::_tick` diffs the node-biometrics projection before and after each batch, using `biometrics_prediction_error()` in `orion/substrate/prediction_error.py`. It averages the absolute change of every hint key on every node.

## Live evidence (read-only, 2026-10-20)

- **hecate's standby is invisible to this pipeline except as a frozen entry.**
  - `substrate_node_biometrics_projection.hecate.last_seen_at` is `2026-10-15T06:31:03Z`, with 9 hints. `orion_biometrics` has no hecate row after that time, and `grammar_events` has 0 hecate traces.
  - The only thing that still updates is hecate's PDU wattage. athena polls it by proxy, and it lands in `orion_biometrics_cluster.measurements_by_node.hecate` as `{pdu_watts, chassis_watts}`. Since 06:31Z it reads 95-97 W in 30.8k of about 31.6k rows. It also had a 0.0 stretch on 10-16 from 22:20 to 00:14, and rare dips to 48-80 W.
  - That wattage never reaches the node projection or this mean. It is a powered-down box on standby power and reflects no change in Orion's body, so it stays out. This patch makes no change to it.
- **Report rhythm.** Over 3 days of `grammar_events`, athena's median interval was 31.24 s and circe's was 31.32 s. The longest gap either had was 94 s, which is 3x. A multiple of 6 gives 2x margin over that worst gap and roughly matches the existing 180 s staleness setting.
- **Stuck pressure.** No node currently has a stuck `availability`. hecate is `active_pressures=[]`, `suppressed=["strain"]`, `pressure_score=1.0`, and that score is cleared on the first sweep after deploy. circe and athena are online and normal.

## Before/after on live receipts

I replayed the last 60 minutes of real receipts:
- the per-event node-reducer before/after deltas
- 121 `substrate.node_biometrics` prediction-error ticks

The replay reproduces every stored value exactly (max abs error 0.0). It is not an approximation.

| | mean | median | p90 | max |
|---|---|---|---|---|
| stored / before | 0.0451 | 0.0389 | 0.0791 | 0.1221 |
| after (hecate excluded) | 0.0608 | 0.0524 | 0.1065 | 0.1643 |

The key count goes from 35 to 26 on every tick. The after/before ratio is 1.346 on every tick.

## Interaction with the sustained attention rule (PR #2718)

The scaled numbers make biometrics louder. I modelled the effect with `scripts/analysis/replay_attention_sustained_body.py` over 13,848 frames, 2026-10-13 to 10-20. The replay applies a step to biometrics readings from 2026-10-17 onward. The unscaled control uses the same window with factor 1.0.

| window | rule off: control -> x1.346 | rule on (live): control -> x1.346 |
|---|---|---|
| after step (7,565 frames) | 13.0% -> 19.5% | 0.40% -> 1.77% |
| first 24 h | 13.4% -> 25.2% | 0.35% -> 3.79% |
| first 72 h | 13.6% -> 21.0% | 0.46% -> 2.16% |

These are biometrics win shares. With the sustained rule on, which is live, biometrics goes from almost never winning to about 2% of frames. It is highest (3.8%) in the first day, while the 7-day percentile history is still on the old scale, and then falls back. No continuous win has a dwell of 300 s or less.

Method limits:
- The replay scales by a constant. The real runtime would only scale while hecate is down.
- The definition-version bump restarts the attention precision baseline. The replay does not model that. The percentile history (`substrate_node_prediction_error_history`) is not reset.

## Architecture touched

- `orion-substrate-runtime`: the biometrics tick and the absence sweep.
- Shared `orion/substrate/biometrics_loop/` and `orion/substrate/prediction_error.py`.
- `orion/schemas/prediction_error_definitions.py` (version table, read by orion-attention-runtime).

## Files changed

- `orion/substrate/biometrics_loop/pressure_reducer.py`: suppression clears all kinds and zeroes the score.
- `orion/substrate/biometrics_loop/pressure_organ.py`:
  - frozen-hint guard after Rule A (`offline_frozen_hint_guard`)
  - the sweep also selects nodes with a non-zero score
- `orion/substrate/biometrics_loop/pipeline.py`: passes the guard flag through.
- `orion/substrate/biometrics_loop/report_cadence.py` (new):
  - in-process per-node report interval: median of the last 16, at least 3 needed, fleet-median fallback
  - `silent_nodes()`
- `orion/substrate/prediction_error.py`: `biometrics_prediction_error(..., exclude_nodes=)`; v2 docstring.
- `orion/schemas/prediction_error_definitions.py`: `node_biometrics: 2`.
- `services/orion-substrate-runtime/app/worker.py`:
  - `_silent_biometrics_nodes`
  - stamps v1 when the kill switch is off
  - guard flag on both pipeline calls
- `services/orion-substrate-runtime/app/settings.py`, `.env_example`, `docker-compose.yml`: 3 new keys.
- `.github/workflows/biometrics-loop-tests.yml`: runs the two new root tests.
- Tests:
  - `tests/test_offline_node_stuck_pressure.py`
  - `tests/test_biometrics_silent_node_exclusion.py`
  - `services/orion-substrate-runtime/tests/test_worker_biometrics_silent_nodes.py`

## Schema / bus / API changes

- Added: none. No pydantic model changed. Node-biometrics consumers in hub, sql-writer and organs are unaffected.
- Behavior changed:
  - The `node:substrate.biometrics` prediction error excludes silent nodes. Definition version is now 2.
  - Suppression clears every pressure kind and the score.
- Compatibility notes:
  - orion-attention-runtime folds only receipts whose version stamp matches its own table.
  - Until both runtimes run this code, biometrics receipts are not folded into the precision baseline. This is a pause, not data loss.

## Env/config changes

- Added keys (orion-substrate-runtime):
  - `BIOMETRICS_OFFLINE_FROZEN_HINT_GUARD=true`
  - `BIOMETRICS_SILENT_NODE_EXCLUSION=true`
  - `BIOMETRICS_SILENT_NODE_CADENCE_MULTIPLE=6.0`
- `.env_example` updated: yes.
- Local `.env` synced with `python3 scripts/sync_local_env_from_example.py --all-keys orion-substrate-runtime`: yes, 3 keys added and none diverged.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest tests/test_retired_biometrics_node.py tests/test_biometrics_pipeline.py tests/test_biometrics_pressure_organ.py \
  tests/test_capability_absence_signal.py tests/test_node_biometrics_reducer.py tests/test_node_pressure_reducer.py \
  tests/test_biometrics_grammar_extract.py tests/test_fleet_roster_coverage.py \
  tests/test_offline_node_stuck_pressure.py tests/test_biometrics_silent_node_exclusion.py   -> 102 passed
PYTHONPATH=.:services/orion-substrate-runtime pytest services/orion-substrate-runtime/tests/test_worker_biometrics_silent_nodes.py \
  services/orion-substrate-runtime/tests/test_prediction_error_receipt_not_gated.py          -> 8 passed
services/orion-substrate-runtime/tests (full): 17 failed / 483 passed -- the identical 17 fail on clean main (diffed)
Regression proof: the stuck-pressure tests fail on main's organ/reducer (availability stays ['availability']);
the live-hecate-shape test fails without the sweep change.
Gates: env template parity PASS, metric lineage PASS, definition drift PASS, inner_state_registry OK,
compose relative mounts PASS, sentience instruments static PASS, git diff --check clean.
```

## Evals run

```text
Live receipt replay (121 ticks, exact reproduction) and the sustained-attention replay above.
No service eval harness exists for the biometrics loop; follow-up: turn the receipt replay into one.
```

## Docker/build/smoke checks

```text
Not run (no deploys from this session). UNVERIFIED on the live rail.
```

## Review findings fixed

- Finding (MUST): live hecate (lists empty, score 1.0) is never selected by the sweep, so the score fix could not reach it.
  - Fix: the sweep also selects a node with `pressure_score > 0`.
  - Evidence: `test_live_hecate_shape_score_only_is_cleared` fails without the change and passes with it.
- Finding (SHOULD): after a restart, the backlog replay makes every node look silent by wall-clock age, so the error would read a false 0.0.
  - Fix: age is measured against the newest report in the projection, so the newest reporter is never silent.
  - Evidence: `test_backlog_replay_does_not_silence_every_node` and `test_single_node_is_never_silent`.
- Finding (SHOULD): with the kill switch off, receipts still carried v2.
  - Fix: the worker stamps v1 when the switch is off.
  - Evidence: the parametrized worker test checks the version for both switch states.
- Finding (SHOULD): deploy order across the two runtimes.
  - Fix: documented above, with both restart commands below.
- Finding (NIT): multi-report batches.
  - Fix: documented in the docstring. They only lengthen the learned interval, which is the safe direction.
- Finding (NIT): lazy getattr state.
  - Fix: class-level defaults.
- Finding (NIT): the merge-window path at 1 s sweep spacing was untested.
  - Fix: `test_settles_at_the_runtime_sweep_rate_through_the_merge_window`.
- Finding (NIT): a code comment overstated the within-emission loop.
  - Fix: comment corrected.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-attention-runtime up -d --build
./scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Run both from the clean primary `main` checkout after merge, back to back.

## Risks / concerns

- Severity: low.
  - Concern: waking hecate from standby produces a one-tick spike. Its first report is diffed against values that are days old.
  - Mitigation: this is a real body event. It lasts one tick.
- Severity: low.
  - Concern: in the first ~1-2 minutes after a restart nothing has been learned yet, so nothing is excluded (old behaviour).
  - Mitigation: self-clearing after 3 reports per live node.
- Severity: low.
  - Concern: biometrics wins attention more often (about 0.4% -> 2% of frames with the sustained rule).
  - Mitigation: this is the intended correction. Watch the first 24 h, when it peaks near 3.8%.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2809

🤖 Generated with [Claude Code](https://claude.com/claude-code)
