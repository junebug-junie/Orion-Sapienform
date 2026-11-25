# Attention: band the forecast-surprise nodes on their calm theory, so a sparse source can compete (#3917)

Fixes #3917

## Summary

- Orion's attention could never pick the chat-turn surprise signal. The judge needed 200 readings in 7 days, and the signal writes about 11 a day, so it always read "too little history".
- The five forecast-surprise nodes (`node:prediction.room`, `.body`, `.room_joint`, `.body_joint`, `.chat_turn`) are now banded by how often a calm, well-predicted window would read lower than the current value. That number comes from the producer's math, not from the node's own week, so no history count is needed.
- One table says which rule each node uses: `ATTENTION_BAND_RULE_BY_NODE_ID` in `orion/substrate/calm_band.py`. Unlisted nodes (and, explicitly, the two cameras) keep the own-week percentile and its 200-reading floor.
- The math from PR #3826 (curiosity seed bars) moved out of its tests into `calm_band.py`, and those tests now call it. One copy.
- Orion's self-reading line for these nodes now says "higher than N% of calm, well-predicted stretches" instead of "of my last 7 days", and the truth eval checks that wording against the math.

## Outcome moved

Replay of the field contest over the stored 7 days (2026-11-18 19:35Z .. 11-25, 9,383 one-minute ticks, `services/orion-attention-runtime/evals/replay_sparse_pe_calm_band.py`):

- `node:prediction.chat_turn`: eligible ticks 0 -> 40, won 0 -> 5. Every one of the 8 chat-turn writes that read "high" on its calm theory (calm percentile >= 0.9) got eligible ticks in the 300 s after it (8 of 8, 5 ticks each).
- Dense nodes: no change in win-when-eligible beyond 2 standard errors (hour-block jackknife). Table below.

| source | rule | eligible before | wwe before | eligible after | wwe after | change | 2 SE |
|---|---|---|---|---|---|---|---|
| node:prediction.chat_turn | chat_bets | 0 | n/a | 40 | 0.125 | new | n/a |
| node:prediction.body | floored | 714 | 0.721 | 371 | 0.671 | -0.050 | 0.102 |
| node:prediction.room | floored | 797 | 0.664 | 450 | 0.656 | -0.008 | 0.085 |
| node:prediction.body_joint | gamma | 85 | 0.777 | 287 | 0.770 | -0.006 | 0.351 |
| node:prediction.room_joint | gamma | 25 | 0.240 | 111 | 0.775 | +0.535 | 0.657 |
| node:substrate.strain | empirical | 59 | 0.848 | 59 | 0.898 | +0.051 | 0.101 |
| node:substrate.street | empirical | 3 | 1.000 | 3 | 1.000 | 0 | n/a |
| world:chat | empirical | 782 | 0.930 | 782 | 0.960 | +0.031 | 0.033 |
| world:reading | empirical | 661 | 0.979 | 661 | 0.986 | +0.008 | 0.012 |

What did change: how OFTEN the dense forecast nodes are eligible. Room and body are eligible about half as often (their own week put them in the top 10 % by construction; their calm theory says only ~5 % of their writes are rare under calm). The joint nodes are eligible 3-4x as often: body_joint reads "high" on 20.7 % of its writes against a calm design rate of at most 10 %, so its forecasts really are missing more than calibrated forecasts should, and the own-week percentile was hiding that. The joint nodes also competed for only the last ~17 h of the window under the old rule (they started 2026-11-24 04:41Z and needed 200 readings first); the calm rule lets a new node compete from its first reading.

Why chat_turn wins only 5 of 40 eligible ticks: a chat turn settles while Juniper is talking, so `world:chat` is usually eligible at the same moment and ranks higher (it wins 93-96 % when eligible). That is the contest working, not the floor.

Not modelled in the replay (the same in both arms): `world:play`, camera frame staleness. The cameras had only 48 current-scale rows in the window (PE retire Patch 3 went live 2026-11-24), so they read insufficient history in both arms.

## Current architecture

`compute_prediction_error_magnitude` (`orion/substrate/prediction_error_magnitude.py`) took a node's value and history and returned a percentile against the node's own 7 days, with band `insufficient_history` below `DEFAULT_MIN_READINGS = 200`. Both contests call it: orion-attention-runtime through `NodePeHistoryCache.magnitudes` (also feeds self-readings), and orion-substrate-runtime's broadcast in `worker._prediction_error_magnitudes`. The ranking key was `midrank_percentile` over the same week. History rows are pruned at 168 h, so a node writing ~11 a day can never reach 200.

## Architecture touched

- New pure module `orion/substrate/calm_band.py`: exact calm tails, the per-node rule table, `calm_percentiles(node_id, value)`.
- `compute_prediction_error_magnitude(..., node_id=None)`: a node with a calm rule takes `percentile_now` and `band` from its calm theory, no history floor. `trend` still needs 200 readings (it describes the node's own history).
- `world_first.node_rank_percentile`: the ranking key, calm mid-rank for calm nodes, own-week mid-rank otherwise. Used by both contests' magnitude producers.
- Self-reading wording, the field frame's reason text ("percentile X of calm theory (rule)"), and the self-readings truth eval.

What else was gated by the same 200 floor, checked:

- `self_readings` (prompt): reads `NodePeHistoryCache.magnitudes`, so it is fixed by the same change.
- `self_signal_outcomes` (outcome associations): uses the floor, but only judges `level`/`count`/`trigger`/`binary` nodes; `surprise_excess` nodes are skipped before the floor. Not changed; not a floor problem.
- Metacog episode gate (`services/orion-equilibrium-service/app/self_signal_episode_gate.py`): only observes `node:substrate.*`, so the prediction nodes never reach its floor. Not changed.
- Prime: nothing under `orion/prime` or `services/orion-mind` reads the floor or `n_readings_7d` (grep). Not touched (another agent owns those files).
- `pe_history_cache`: fixed (passes `node_id`, calm rank).
- substrate-runtime `_strain_history_warm`: strain alarm only; empirical. Not changed.
- `run_pe_magnitude_live_sanity.py` eval: its coverage check no longer requires 200 rows for a calm node.

## Files changed

- `orion/substrate/calm_band.py`: new; the math and the one table.
- `orion/substrate/data/chat_turn_calibrated_q_2026-11-24.json`: moved from `orion/substrate/tests/fixtures/` (runtime now reads it; Dockerfiles copy `orion/` whole).
- `orion/substrate/prediction_error_magnitude.py`: `node_id` parameter, calm band without floor.
- `orion/attention/world_first.py`: `node_rank_percentile`; `node_candidate` uses it when given history.
- `orion/attention/pe_history_cache.py`, `services/orion-substrate-runtime/app/worker.py`: pass `node_id`, use `node_rank_percentile`.
- `orion/attention/self_readings.py`: calm wording.
- `orion/attention/field_attention/selectors.py`: reason text names the percentile's basis.
- `services/orion-thought/evals/self_readings_truth.py`: parses `prediction.*` refs and the calm wording, re-derives the calm percentile from the value.
- `orion/substrate/tests/test_curiosity_seed_bar.py`: uses `calm_band` instead of its own copies of the math.
- `orion/substrate/tests/test_calm_band.py`, `tests/test_attention_calm_band.py`: new tests.
- `services/orion-substrate-runtime/tests/test_worker_pe_history.py`: room now gets a band from one reading.
- `services/orion-attention-runtime/evals/replay_sparse_pe_calm_band.py`: new replay eval.
- `services/orion-attention-runtime/evals/replay_world_first.py`, `scripts/analysis/replay_attention_sustained_body.py`, `services/orion-substrate-runtime/evals/run_pe_magnitude_live_sanity.py`: pass `node_id` so band and rank agree.
- `.github/workflows/prediction-surprise-node-tests.yml`: new paths and tests.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: for the five forecast-surprise nodes, `PredictionErrorMagnitudeV1.percentile_now` is the calm-theory percentile (share of calm windows that read lower), not the share of the node's own week. `band` no longer reads `insufficient_history` for them. `n_readings_7d`, `p50_7d`, `p90_7d`, `trend` are unchanged.
- Compatibility notes: no field added or removed; consumers that print the percentile now get the basis named in the reason text and the self-reading line.

## Metric quality gate (CLAUDE.md 0A)

This is not a new metric. It is a new reference distribution for an existing value (the node's stored `clip(z / 3, 0, 1)`).

1. **Provenance.** The value: `orion/substrate/prediction_surprise.py` `reading()` (room, body), `prediction_joint_surprise.py` `reading()` (joint nodes), `prediction_turn_surprise.py` (chat_turn), each `max(0, min(1, z / 3))`. The new percentile: `calm_band.calm_percentiles`, called from `compute_prediction_error_magnitude` and `world_first.node_rank_percentile`.
2. **Independence.** A monotone transform of the stored value, so it adds no signal and no new competitor. It only changes where the "high" line sits for these five nodes.
3. **Theory anchor.** Probability integral transform: a calibrated forecaster's p is uniform, so -ln p is Exp(1); a window sum is Gamma(n) (joint nodes); with p floored at 0.05 each term is min(Exp(1), ln 20) and the sum's tail is exact by inclusion-exclusion (room/body; matches a seeded Monte Carlo within 0.006); calibrated yes/no bets give an exact two-point law per turn (chat_turn); Cantelli's one-sided bound P(z >= k) <= 1 / (1 + k^2) holds for any standardized calm variable. The window size n is not stored, so the tail is the largest over n = 1..12 and the normal limit: it can understate how unusual a reading is, never overstate it.
4. **Live-data sanity.** All `node:prediction.*` history rows (2026-11-18..25): rest is exact 0 on 56-65 % of writes and 0 maps to calm percentile 0.0 (quiet), so the band can return to calm. Share of writes reading "high" or above on calm theory: body 4.6 %, room 5.4 %, room_joint 8.4 %, chat_turn 10.0 %, body_joint 20.7 % (miscalibrated: real surprise). "Unusual" (>= 0.99): room/body 0.6-0.7 %; never for the joint nodes or chat_turn, because the value clips at z = 3 before a 1 % calm tail (stated in the tests; honest, not a bug). Not flat, not saturated.
5. **Existing mechanism.** PR #3826 derived these tails for curiosity seed bars; this patch moves that math into one module both use.
6. **Reversibility.** Cheap: delete a node's entry in `ATTENTION_BAND_RULE_BY_NODE_ID` and it returns to the empirical rule. Nothing persisted depends on it.

Per-node rule table:

| node | rule | why |
|---|---|---|
| node:prediction.room, .body | floored | p floored at 0.05: capped-exponential sum |
| node:prediction.room_joint, .body_joint | gamma | conformal p, Exp(1) per outcome |
| node:prediction.chat_turn | chat_bets | two calibrated bets per turn; live q mix of 71 turns, one- and two-turn windows, larger tail |
| node:substrate.perception, .perception_walkway | empirical | calibrated by construction, but dense, walkway's p99 floor is on its empirical mid-rank, and #3874 relies on insufficient_history after a scale change |
| everything else (alarms, strain, street, ...) | empirical | no calm theory; alarms are triggers on their own path |
| node:prediction.street_rhythm (other branch) | can join as `cantelli` now, or its own tail later | excess surprisal of calibrated stated p; a test shows the table takes it |

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q \
  tests/test_attention_*.py tests/test_pe_*.py tests/test_self_signal_outcomes.py \
  orion/substrate/tests/test_calm_band.py orion/substrate/tests/test_curiosity_seed_bar.py \
  orion/substrate/tests/test_prediction_error_magnitude.py services/orion-attention-runtime/tests
  -> 3 failed, 621 passed. The same 3 fail on main (parity goldens x2, camera atlas label).
Broad run (attention, PE, orion/substrate/tests, attention-runtime tests): branch 5 failed / main 4 failed;
  the extra one (test_rank_is_by_own_percentile_borda_only_breaks_ties) was fixed, now the same 4 as main.
PYTHONPATH=.:services/orion-substrate-runtime pytest test_worker_pe_history.py test_worker_world_first_broadcast.py
  + calm/seed-bar tests -> 117 passed
services/orion-thought: evals/test_self_readings_truth_eval.py tests/test_stance_self_readings.py -> 33 passed, 2 skipped
Static gates: check_definition_drift --gate, check_metric_lineage --gate / --prompt-semantics,
  check_pe_alarm_readers, check_integration_seam, check_scripts_dir_no_stdlib_shadow,
  check_sentience_instruments --static-only, check_env_key_single_source -> all PASS
```

## Evals run

```text
POSTGRES_URI=... python services/orion-attention-runtime/evals/replay_sparse_pe_calm_band.py --days 7 --step-sec 60
  -> table above; chat_turn 8/8 high writes got eligible ticks; no dense source shifts > 2 SE.
services/orion-substrate-runtime/evals -k magnitude -> 3 passed
```

## Docker/build/smoke checks

```text
Not run: no dependency, compose, or Dockerfile change. The moved JSON lives under orion/, which both
Dockerfiles copy whole (COPY orion /app/orion).
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

Rebuild, in this order (both read the new module; order only matters for which contest moves first):

```bash
scripts/safe_docker_build.sh orion-attention-runtime up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Deploy from the clean primary main checkout after merge (the post-pull rebuild does this).

## Risks / concerns

- Severity: medium. Concern: the dense forecast nodes change how often they are eligible (room/body about half as often, joint nodes 3-4x). Win-when-eligible holds within noise, but the mix of what attention attends shifts toward the joint nodes, whose forecasts miss more than calibrated forecasts should. Mitigation: delete their table entries to restore the old behaviour; the replay eval re-measures.
- Severity: low. Concern: the chat q mix is frozen at 2026-11-17..24. If the bets' calibrated q's drift, the chat band drifts with them. Mitigation: the same fixture pins the #3826 curiosity bar and its replay eval re-measures it.
- Severity: low. Concern: live proof is pending. UNVERIFIED until a `won` row for `node:prediction.chat_turn` appears in `substrate_attention_hour_rollup` after a surprising turn.
- Issues opened: VERIFY_ISSUE. Issues closed: #3917 (by this PR).

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
