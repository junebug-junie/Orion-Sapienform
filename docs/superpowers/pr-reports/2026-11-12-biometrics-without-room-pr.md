# Biometrics: room sensors out of the machine-body reading

## Summary

- Orion's machine-body surprise number (`node:substrate.biometrics`) no longer counts the room. athena's cabinet sensor board and ambient microphone write `cabinet_*` hints next to athena's CPU, GPU and memory hints. Those 8 keys read the room (a person near the cabinet, a voice, a door), not the machines. Juniper approved removing them (2026-11-12).
- The biometrics pressure organ (decides "this node is under pressure") also reads machine hints only now.
- One owner for the list: `orion/substrate/biometrics_loop/room_hints.py`. Any `cabinet_` key counts, so a new cabinet sensor cannot slip into the body reading.
- The prediction-error definition for `node_biometrics` moves to version 3. The attention precision baseline restarts on the new scale. This is the same mechanism #2809 used.
- No kill switch. The room reading in the body mean is the thing being retired, so there is nothing to fall back to. Roll back with `git revert`.
- The keys stay on the node projection. Named readers still use them on purpose: the field digester's `cabinet_*` channels, the ambient spike consumer, and the Hub cabinet panels. The cabinet's own attention source is `node:substrate.cabinet`.

## Outcome moved

- Before: a person walking past the cabinet moved Orion's body surprise number. A room-only change could saturate it (`biometrics_prediction_error` returns 1.0 on main for a proximity jump of 0.65 with no machine change; it returns 0.0 on this branch).
- Before: 8 of the 26 averaged keys were room keys. They carried 12.1% of the summed change on average and more than half of it on 4.8% of ticks. On the other ticks they diluted the real machine change.
- After: 18 machine keys (athena 9, circe 9). The number reads about 1.26x higher on the same data, because the room keys no longer dilute it.

## Current architecture

- `orion-biometrics` emits one grammar atom per pressure. The `cabinet_*` pressures come from the cabinet board and the mic on athena (`services/orion-biometrics/app/grammar_emit.py`, `_CABINET_PRESSURE_SPECS` and the staleness atoms).
- `orion/substrate/biometrics_loop/grammar_extract.py` turns atoms into `pressure_hints` on the node projection, room keys included.
- Two generic readers treat every hint as "the body":
  - `orion/substrate/prediction_error.py::biometrics_prediction_error`: mean absolute change over every hint key on every reporting node, / 0.30, capped at 1. Written by `services/orion-substrate-runtime/app/worker.py` as `node:substrate.biometrics`.
  - `orion/substrate/biometrics_loop/pressure_organ.py`: `_has_pressure_hints` (any hint >= 0.55) and `_primary_hint_kind` (largest hint).
- `machine strain` (the `strain` composite) never included the room (`orion/telemetry/biometrics_pipeline.py`, strain inputs are host-only).

## Metric quality gate (changed metric: `node:substrate.biometrics` prediction error, v3)

1. **Provenance.** `biometrics_prediction_error()` in `orion/substrate/prediction_error.py`, called from `worker.py` (`_tick`, biometrics branch) after each node-reducer batch. Inputs are the node projection's `pressure_hints`, written by `grammar_extract.py` from `orion-biometrics` atoms. The room keys come from `_CABINET_PRESSURE_SPECS` + the two staleness atoms (athena only).
2. **Independence.** This patch removes inputs; it adds none. The removed keys are the same sensors that feed the field digester's `cabinet_*` channels and the Hub cabinet panels. `node:substrate.cabinet` is a separate signal (cabinet temperature warming error, `orion/autonomy/cabinet_heat.py`), and the room temperature is already a separate covariate in orion-prediction (commit 17fffa08d). After the patch, the biometrics number and the room readings no longer share an input.
3. **Theory anchor.** Interoception is the sense of the body's own state. The room is exteroception. A surprise number that claims to measure "how much the machines' pressure moved" must use only machine readings. Room motion with no machine change is not a body surprise.
4. **Live-data sanity, including rest.** 7-day replay on real data (below). v3 is not flat and not saturated: min 0.0001, p05 0.0057, p25 0.026, median 0.060, p90 0.153, max 0.37. It goes below 0.01 on 9.9% of ticks, so it can still read calm. It is a mean of absolute changes, not a z-score, so it has no built-in floor: 0.0 when nothing moved (pinned by `test_room_only_change_reads_calm`). The glossary rest band [0, 0.03] is kept; the minimum is far below its top, so the immune `floor_locked` check stays quiet.
5. **Existing mechanism.** No existing room/body split for node hints. The pattern follows #2809 (silent-node exclusion + definition version).
6. **Reversibility.** Cheap. One module, one version number. `git revert` restores v2; the attention baseline restarts again on the next version change.

### Learned models and history that read this metric

| Reader | Effect | Handling |
|---|---|---|
| Attention precision baseline (`substrate_node_prediction_error_baseline`, orion-attention-runtime) | EWMA on the old scale | Definition v3: baseline restarts; v2 receipts are skipped (test `test_v2_biometrics_baseline_restarts_on_v3_room_exclusion`) |
| 7-day percentile history (`substrate_node_prediction_error_history`; world-first attention, self readings, self-signal episodes) | Rank against a week of v2 values | Not reset (same as #2809). Rank-based, so it settles as the week refills. Transient measured below |
| Endogenous curiosity (`min_prediction_error` 0.55, absolute) | Could seed more often | v3 max over 7 days is 0.37; no tick crosses 0.55 under v2 or v3. No change |
| World model features | Excludes this scalar by design | None |
| orion-prediction series | Does not model this node | None |

Nothing here needs a model retrain. Nothing was found that is expensive to handle.

## Before/after on 7 days of real data

`scripts/analysis/replay_biometrics_without_room.py --fetch --days 7` (read-only). It rebuilds each node's hints from `orion_biometrics_summary` and diffs the report before and after each stored tick (the live tick runs every 10-20 s and usually carries one node's report). The v2 replay reproduces the stored value exactly (|diff| < 1e-4) on 94.3% of 5,642 ticks (Pearson r 0.92), 2026-11-05 03:39Z to 2026-11-12 03:38Z.

| series | mean | median | p90 | max | exactly 0 |
|---|---|---|---|---|---|
| stored (live v2) | 0.0610 | 0.0503 | 0.1228 | 0.3314 | 0.0% |
| replay v2 (room in) | 0.0581 | 0.0480 | 0.1182 | 0.2784 | 0.0% |
| replay v3 (room out) | 0.0734 | 0.0598 | 0.1525 | 0.3697 | 0.0% |

- v2 vs v3 Pearson r 0.983. 88.5% of the top-10% ticks are the same under both.
- Daily v3/v2 ratio: 1.25-1.31, stable across all 8 days.

### Attention transient (world-first, sustained-body rule on, as live)

`scripts/analysis/replay_attention_sustained_body.py` over 5,654 stored frames, with v3 values substituted from 2026-11-08 12:00Z and the earlier week left on v2 (control: same window, values unchanged).

| window | control | v3 |
|---|---|---|
| after step (2,778 frames) | 0.32% | 1.33% |
| first 24 h | 0.38% | 1.25% |
| first 72 h | 0.35% | 1.38% |

These are biometrics win shares. Biometrics wins about 1% more of frames while the 7-day history still holds v2 values, then falls back as the history refills. No continuous win has a dwell of 300 s or less. #2809 showed the same shape (0.40% -> 1.77%).

## Architecture touched

- Shared `orion/substrate/biometrics_loop/` and `orion/substrate/prediction_error.py` (run inside orion-substrate-runtime).
- `orion/schemas/prediction_error_definitions.py` (version table, read by orion-attention-runtime).
- Metric glossary meaning + definition lock.

## Files changed

- `orion/substrate/biometrics_loop/room_hints.py` (new): room key list, `is_room_hint`, `machine_body_hints`.
- `orion/substrate/biometrics_loop/grammar_extract.py`: role-to-hint map now comes from `room_hints.py`.
- `orion/substrate/prediction_error.py`: `biometrics_prediction_error` reads machine hints only; v3 docstring.
- `orion/substrate/biometrics_loop/pressure_organ.py`: organ reads machine hints only.
- `orion/schemas/prediction_error_definitions.py`: `node_biometrics: 3`.
- `services/orion-substrate-runtime/app/worker.py`: comment on the version stamp when the silent-node switch is off.
- `config/field/field_channel_glossary.v1.yaml`, `config/metrics/metric_definitions.lock.json`: meaning says the room is out; lock re-generated with `check_definition_drift.py --update`.
- `scripts/analysis/replay_biometrics_without_room.py` (new): the 7-day replay.
- `.github/workflows/biometrics-loop-tests.yml`: runs the new test.
- Tests: `tests/test_biometrics_room_hints_excluded.py` (new); version expectations in `tests/test_attention_runtime_store.py`, `tests/test_biometrics_silent_node_exclusion.py`, `services/orion-substrate-runtime/tests/test_worker_biometrics_silent_nodes.py`, `services/orion-substrate-runtime/tests/test_prediction_error_receipt_not_gated.py`.

## Schema / bus / API changes

- Added: none. No pydantic model or channel changed.
- Removed: none. The `cabinet_*` hints stay on the projection.
- Renamed: none.
- Behavior changed: `node:substrate.biometrics` prediction error and the pressure organ ignore `cabinet_*` hints. Definition version 3.
- Compatibility notes: orion-attention-runtime folds only receipts whose stamp matches its table. Until both runtimes run this code, biometrics receipts are not folded into the baseline. This is a pause, not data loss.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
tests/test_biometrics_room_hints_excluded.py + 11 biometrics-loop / attention-store test files -> 137 passed
orion/substrate/tests (spike consumer, prediction_error, magnitude, trend)                      -> 81 passed
attention worker/store, field glossary, definition drift, spike bus catalog                    -> 113 passed
services/orion-substrate-runtime/tests (full): 14 failed / 567 passed / 10 errors --
  identical set on clean main (diffed)
Regression proof: on main, a room-only proximity jump returns 1.0 from biometrics_prediction_error;
  on this branch 0.0.
Gates: check_definition_drift --gate PASS, check_metric_lineage --gate PASS, --prompt-semantics PASS,
  metric prompt/rest tests 66 passed, inner_state_registry OK, sentience instruments --static-only
  "All claims hold", scripts stdlib-shadow clean, git diff --check clean.
```

## Evals run

```text
scripts/analysis/replay_biometrics_without_room.py: 7-day real-data replay, v2 reproduced exactly on 94.3% of ticks (table above).
scripts/analysis/replay_attention_sustained_body.py: attention win-share transient (table above).
The biometrics loop has no service eval harness; the replay script is the reusable eval for this metric.
```

## Docker/build/smoke checks

```text
Not run. No deploy from this session (as asked). UNVERIFIED on the live rail.
After deploy, check: node_biometrics receipts carry definition_version 3;
substrate_node_prediction_error_baseline row for node:substrate.biometrics shows definition_version 3
and a restarted observation_count.
```

## Review findings fixed

Code review ran in a subagent. No MUST findings.

- Finding (SHOULD): the 7-day percentile readers (`orion/attention/pe_history_cache.py`, `services/orion-equilibrium-service/app/pe_history_reader.py`) have no version column, so v3 readings are ranked against v2 history for about 7 days.
  - Fix: disclosed in Risks with the settle date. The direction is measured, not guessed: the attention replay shows biometrics wins more often (0.32% -> 1.33% of frames). History rows are not deleted (that is a production write). The same seam was open at the v1 -> v2 bump.
  - Evidence: attention transient table above.
- Finding (SHOULD): the replay did not leave out silent nodes, unlike the live tick.
  - Fix: `projection_at` drops a node whose last report is more than 6x its own median interval behind the newest report (the live rule).
  - Evidence: rerun gives identical numbers. hecate never reported in the window, so no frozen node was in the data.
- Finding (NIT): the replay crashed on an empty export.
  - Fix: it exits with a message when fewer than 3 ticks match.
- Finding (NIT): the replay added both staleness keys for any cabinet key.
  - Fix: each staleness key is added only when its own sensor family is present.
- Finding (NIT): no lifecycle test for the deploy transition.
  - Fix: `test_room_held_strain_decays_after_the_change`: strain that only a loud room kept alive now decays.
- Finding (NIT): the stamp 1 with the kill switch off reads like "the v1 formula".
  - Fix: the worker comment says it means "not the live formula, never folded".

## Restart required

```bash
./scripts/safe_docker_build.sh orion-attention-runtime up -d --build
./scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Run both from the clean primary `main` checkout after merge, back to back.

## Risks / concerns

- Severity: low.
  - Concern: biometrics wins attention about 1% more of frames for up to 7 days while the percentile history holds v2 values. Self readings and self-signal episodes rank against the same mixed week.
  - Mitigation: rank-based; settles 7 days after deploy, when the history holds only v3. Same shape #2809 accepted.
- Severity: low.
  - Concern: the room keys stay on the projection, so a future generic reader could pick them up again.
  - Mitigation: `machine_body_hints()` is the one owner; any `cabinet_` key counts as room.

## PR link

TBD
