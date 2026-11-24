## Summary

Curiosity starts an investigation on its own when one of Orion's surprise readings crosses a bar. There was one bar (0.55) for every reading. Now each reading has its own bar, set from the math of how that reading is made, so that on a normal day each one crosses about 5 % of the time or less.

- Four forecast readings (office temperature, cabinet temperature, the office air group, the machines group) get a bar of 0.667. That is the cut where a calm window is surprising with probability 5 % or less. The old 0.55 was a normal-curve cut; these sums are skewed, so 0.55 really meant 6.5-7 %.
- The chat-turn reading gets a bar of 0.80. Its value comes from two yes/no bets, so it jumps in steps. 0.80 gives a 4.6 % average calm crossing rate on the real bets of the last 7 days. The old 0.55 gave 8.5 %.
- The two camera readings keep 0.55. Patch 3 replaces their seed with a scene-change seed.
- A crossing's strength now reads 0.55 at its own bar and 1.0 at the top. So a chat surprise and a temperature surprise compete on one scale. For a 0.55 bar this changes nothing.
- New: a replay eval, tests, and a read grant file so Orion can read the table behind the two group readings. The grant file is NOT applied.

`orion/substrate/curiosity_seed_bar.py` (`CURIOSITY_SEED_BAR_BY_NODE_ID`), used by `orion/substrate/endogenous_curiosity.py::_prediction_error_candidates`.

**The issue's premise is only partly right.** #3821 says the 0.55 bar was set for the retired execution node's scale, so the office air group (room_joint) and chat_turn could never cross. The data says otherwise:

- Every successor reading was built on purpose on the same scale (`clip(z / 3)`), so 0.55 was never a foreign scale for them. It was a slightly loose cut (normal-curve, not exact).
- chat_turn did cross: 8 of 72 node writes in 7 days were at or above 0.55, and it seeded 21 candidates (19 on 11-23). The "0 of 18 in 24 h" in the issue is a quiet day, not a scale problem.
- room_joint has lived for 5 hours (since 2026-11-24 04:35Z). That is about 50 writes, and one surprise window in 20 is expected. A rebuild from its backing table does cross 0.55 once (max 0.626). It is too young to call broken or fine.

So this patch makes curiosity seed **less** often, not more. That is the honest outcome of a theory-anchored bar. If Juniper wants more curiosity from these readings, the lever is a looser stated rate (for example p <= 0.10), not a percentile.

## Outcome moved

Each seedable reading now crosses at a stated, checkable calm rate instead of one borrowed number. Seed ticks over the 7-day replay: body 302 -> 159, room 388 -> 227, chat_turn 76 -> 25, cameras unchanged (171, 125).

## Current architecture

`_prediction_error_candidates` compared each node's staleness-decayed `prediction_error` against one `min_prediction_error = 0.55` and used the decayed value as `signal_strength`. The frontier evaluator invokes only when the top strength is >= 0.5. Retired nodes, alarm nodes, non-measurement nodes and `node:substrate.street` never seed.

## Architecture touched

- `orion/substrate/curiosity_seed_bar.py` (new, stdlib): the per-node bar table and the strength map.
- `orion/substrate/endogenous_curiosity.py`: reads the node's own bar; `EndogenousCuriosityConfig.seed_bar_by_node_id` (defaults to the table); `min_prediction_error` stays as the bar for any unlisted node.
- No schema, bus, env, Docker or API change.

## Files changed

- `orion/substrate/curiosity_seed_bar.py`: bar table, derivation, `seed_bar`, `seed_strength`.
- `orion/substrate/endogenous_curiosity.py`: per-node bar and strength in the PE seed path.
- `orion/substrate/tests/test_curiosity_seed_bar.py`: theory tests (Gamma tail, chat calm tail on live bets), seed / no-seed per node, retired and alarm nodes never seed.
- `orion/substrate/tests/fixtures/chat_turn_calibrated_q_2026-11-24.json`: the calibrated probabilities of the 71 chat turns' bets, 2026-11-17..24 (no ids, no outcomes).
- `services/orion-substrate-runtime/evals/run_curiosity_seed_bar_replay_eval.py` + `test_curiosity_seed_bar_replay_eval.py`: live replay, before vs after.
- `scripts/sql/2026-11-24_grant_orion_readonly_prediction_joint_surprise.sql`: read grant, not applied.
- `.github/workflows/prediction-surprise-node-tests.yml`: runs the new tests.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: curiosity's prediction-error seed bar per node; seed `signal_strength` for listed nodes is mapped so the node's own bar reads 0.55.
- Compatibility notes: unlisted nodes behave exactly as before (bar 0.55, strength = decayed value).

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no env template changed)
- skipped keys requiring operator action: none

Ships on: the table is the default of `EndogenousCuriosityConfig`; no flag.

## Metric quality gate (CLAUDE.md 0A), on the new bars

These are bars on existing readings, not new readings. The gate is run on what each bar claims to measure.

**1. Provenance.** What each value is:

| node | producer | stored value | one outcome's calm shape |
|---|---|---|---|
| `node:prediction.room` / `.body` | `orion/substrate/prediction_surprise.py` | `clip(z/3)`, z = summed excess / sqrt(summed calm var) over 30 min | excess = -ln max(p, 0.05) - 0.95, p uniform |
| `node:prediction.room_joint` / `.body_joint` | `orion/substrate/prediction_joint_surprise.py` | same | excess = -ln p - calm point, conformal p uniform -> Exp(1) - 1 |
| `node:prediction.chat_turn` | `orion/substrate/prediction_turn_surprise.py` <- `services/orion-prediction/app/turn_surprise.py` | same z, per turn = sum of 2 bets | two-point: -(y - q) logit q |
| `node:substrate.perception` / `_walkway` | camera EWMA z (`orion/substrate/prediction_error.py`) | `clip(z/3)` | not a forecast; unchanged |

**2. Independence.** No new signal. Known overlap: the cabinet temperature feeds both `node:prediction.body` and `node:prediction.body_joint`, so one cabinet event can seed both. That was already true at 0.55.

**3. Theory anchor.**
- Forecast nodes: under calibration the window sum is Gamma(n, 1). The exact 95th percentile of a standardized Gamma(n) is 1.996 (n=1), 1.940 (2), 1.903 (3), 1.877 (4), 1.858 (5), 1.842 (6), 1.830 (7), 1.820 (8). So z >= 2.0 is calm p <= 0.05 at every window size. The old z >= 1.645 is calm p 0.071 (n=1) .. 0.065 (n=8). With room/body's p floor at 0.05 (Monte Carlo, 200k): calm tail at z >= 2.0 is 7.2 % (n=1), 3.9 % (n=2), 3.3 % (n=6).
- Chat turn: a two-point variable reaches z >= k with calm probability up to 1/(1+k^2) (Cantelli, tight). No cut below z 4.36 guarantees p <= 0.05 per turn, and the node clips at z 3. So the bar is set from the exact calm tail of the real bets (enumerate each turn's outcomes at its calibrated q): mean calm crossing rate per turn is 8.45 % at 0.55, 5.02 % at 0.75, **4.63 % at 0.80**, 3.61 % at 0.90. It is a mean over turns: one turn with "asks a question" at q 0.084 crosses with calm probability 0.084 at any bar below 0.95.
- Not a percentile of the node's own history: the forecast bar comes from the Gamma distribution, and the chat bar from the bets' stated probabilities, not their outcomes.

**4. Live-data sanity** (7 days to 2026-11-24 08:54Z, `substrate_node_prediction_error_history`; replay below):

| node | writes | exact 0 | writes >= 0.55 | writes >= new bar | max |
|---|---|---|---|---|---|
| body | 1684 | 59.9 % | 4.51 % | 2.67 % | 1.000 |
| body_joint (since 04:35Z) | 51 | 45.1 % | 5.88 % | 3.92 % | 0.738 |
| room | 1893 | 58.0 % | 4.97 % | 3.12 % | 1.000 |
| room_joint (since 04:35Z) | 51 | 58.8 % | 0.00 % | 0.00 % | 0.511 |
| chat_turn | 72 | 65.3 % | 11.11 % | 6.94 % | 1.000 |
| perception | 3033 | 60.6 % | 1.85 % | 1.85 % | 1.000 |
| perception_walkway | 3033 | 57.4 % | 1.38 % | 1.38 % | 1.000 |

Rest: every node sits at exactly 0 about half the time or more. None is flat, always-null or saturated.

Known limit: the blended family A rows (the same p machinery as the two group nodes, 7 days, rebuilt through the real fold) cross the new bar on 7.4 % of writes, above 5 %. Their misses are autocorrelated (consecutive windows share outcomes and the world moves in runs), which widens z past the calm model. That is a producer calm-model question, not a bar question; not changed here.

**5. Existing mechanism.** `ATTENTION_MIN_BAND_BY_NODE_ID` (`orion/schemas/attention_candidate.py`) is the per-node pattern, but it is a percentile band of the node's own 7-day window, which is the thing this issue rules out. The new table follows its shape (a dict in code, keyed by node id), with bars in value units.

**6. Reversibility.** One dict. Delete an entry and that node falls back to 0.55; `seed_bar_by_node_id={}` restores the old behavior everywhere. Nothing persisted.

## Replay: seed ticks per node per day, 0.55 -> new bar

`services/orion-substrate-runtime/evals/run_curiosity_seed_bar_replay_eval.py --days 7`. One tick per 60 s; the newest write, decayed over 1800 s exactly as curiosity reads it. Not modelled: the budget cap, System One admission, the evaluator (so these are upper bounds on seeds).

| day (UTC) | body | room | chat_turn | perception | walkway |
|---|---|---|---|---|---|
| 11-17 | 39 -> 26 | 18 -> 4 | 0 -> 0 | 0 -> 0 | 14 -> 14 |
| 11-18 | 70 -> 48 | 83 -> 58 | 0 -> 0 | 20 -> 20 | 28 -> 28 |
| 11-19 | 12 -> 10 | 67 -> 50 | 0 -> 0 | 9 -> 9 | 6 -> 6 |
| 11-20 | 59 -> 25 | 87 -> 52 | 2 -> 0 | 11 -> 11 | 30 -> 30 |
| 11-21 | 40 -> 19 | 71 -> 31 | 0 -> 0 | 91 -> 91 | 14 -> 14 |
| 11-22 | 12 -> 1 | 15 -> 10 | (no turns) | 32 -> 32 | 4 -> 4 |
| 11-23 | 40 -> 18 | 22 -> 4 | 74 -> 25 | 5 -> 5 | 22 -> 22 |
| 11-24 (part) | 30 -> 12 | 25 -> 18 | 0 -> 0 | 3 -> 3 | 7 -> 7 |
| total | 302 -> 159 | 388 -> 227 | 76 -> 25 | 171 -> 171 | 125 -> 125 |

body_joint (11-24 only): 12 -> 5. room_joint (11-24 only): 0 -> 0 from history; rebuilt from `prediction_joint_surprise` (family_a_room) 3 -> 0, max 0.626. Blended family A proxy, 7 days: 805 -> 547.

Can each node rest? A forecast node writes about 290 times a day, so even at a calibrated 5 % it seeds on most days: zero-seed DAYS is the wrong test for it. The share of fresh HOURS with no seed: body 76.1 % -> 84.3 %, room 75.1 % -> 82.2 %, chat_turn 78.9 % -> 89.5 % (and 6 of 7 days with turns had no chat seed), perception 76.9 %, walkway 79.9 % (unchanged). room_joint / body_joint: 5 hours of life, 100 % / 60 % resting.

## Tests run

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q \
  orion/substrate/tests/test_endogenous_curiosity.py orion/substrate/tests/test_endogenous_curiosity_walkway_fence.py \
  orion/substrate/tests/test_curiosity_seed_bar.py tests/test_retired_pe_step5_readers.py \
  orion/substrate/tests/test_attention_broadcast_world_first.py orion/substrate/tests/test_prediction_surprise.py \
  orion/substrate/tests/test_prediction_turn_surprise.py orion/substrate/tests/test_prediction_joint_surprise.py \
  tests/test_attention_world_first.py tests/test_room_joint_world_source.py tests/test_attention_self_readings.py \
  services/orion-substrate-runtime/evals/test_curiosity_seed_bar_replay_eval.py
220 passed
PYTHONPATH=.:services/orion-substrate-runtime POSTGRES_URI=postgresql://unused/unused ... \
  services/orion-substrate-runtime/tests/test_worker_endogenous_curiosity_tick.py
17 passed
tests/test_blind_experiment_arms_sql.py  4 passed
tests/test_curiosity_urgent_prompt.py tests/test_curiosity_incident_report.py  45 passed
```

## Evals run

```text
POSTGRES_URI=<localhost:55432> python services/orion-substrate-runtime/evals/run_curiosity_seed_bar_replay_eval.py --days 7
(output summarized in the replay section above)
POSTGRES_URI=<localhost:55432> python services/orion-prediction/evals/run_turn_surprise_replay_eval.py --days 14
per turn (last 7 days): 71 turns, z sd 1.309, share >= 0.55 9.9 %, exact 0 71.8 %
```

## Docker/build/smoke checks

```text
None. Pure Python change in orion/substrate; no image, compose or env change.
Live effect after the substrate-runtime rebuild: UNVERIFIED.
Check: substrate_endogenous_curiosity_candidates rows with notes ? 'source:prediction_error', per node per day,
compared against the replay above.
```

## Review findings fixed

Review ran in a sub-agent (read-only, re-derived the Gamma and chat numbers with scipy and a 2M-sample Monte Carlo; all matched).

- Finding: the forecast test used the pure Gamma model, which only describes the joint nodes; room/body floor p at 0.05 and a one-outcome window crosses z 2.0 at 7.3 %.
  - Fix: docstring now says "joint nodes: every n; room/body: n >= 2, 7.3 % at n = 1"; new `test_floored_room_body_cut` (exact n = 1 value, seeded Monte Carlo for n = 2, 3, 6).
  - Evidence: `test_curiosity_seed_bar.py::test_floored_room_body_cut` passes.
- Finding: the chat calm rate is per single turn, but the node's z covers every turn in 30 min.
  - Fix: docstring states it is the one-turn (worst) case; two-turn windows cross at 3.6 % (reviewer's pairing of adjacent live turns).
  - Evidence: `curiosity_seed_bar.py` docstring.
- Finding: the strength map is a ranking change: repair and open-loop seeds keep raw strength, so they now beat forecast seeds they used to lose to.
  - Fix: stated in the module docstring and here; `test_ranking_change_repair_seed_beats_mapped_forecast_seed` pins it (forecast 0.8 -> 0.73 loses to repair 0.74).
  - Evidence: test passes.
- Finding: mutable dict inside a frozen config.
  - Fix: `MappingProxyType`; `test_config_table_is_read_only`.
- Finding: bar looked up before the empty-id / fenced-node filters; table resolved inside the loop.
  - Fix: filters first, table hoisted out of the loop.
- Finding: the eval hard-codes the 1800 s horizon and 60 s tick.
  - Fix: `test_eval_constants_match_live_curiosity` fails if either drifts from `PressureConfig` / `settings.py`.
- Finding: test comment said 0.75 is q ~ 0.15 (it is q ~ 0.165); the single-bet boundary sits at q ~ 0.1479, right on the float 2.4/3.
  - Fix: comment corrected; `test_chat_bar_single_bet_boundary` pins q 0.147 seeds, q 0.149 does not.
- Not changed: `test_retired_and_alarm_nodes_never_seed` would pass without this patch. Kept as a regression guard that the new table cannot reopen them (the task asked for it).

## Restart required

```bash
./scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

(from the clean primary main checkout after merge; the post-pull rebuild does this automatically.)

Grant for Orion's own digging (read-only, not applied by this PR):

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < scripts/sql/2026-11-24_grant_orion_readonly_prediction_joint_surprise.sql
```

## Risks / concerns

- Severity: medium. Concern: curiosity will start fewer runs from forecast surprise (about half the seed ticks). Mitigation: that is the stated 5 % rate; if Juniper wants more, set a looser stated rate in one dict.
- Severity: medium. Concern: room_joint and body_joint have 5 hours of live data; their crossing rate is unmeasured beyond the blended proxy (7.4 % of writes, autocorrelated). Mitigation: re-run the replay eval about 2026-12-01.
- Severity: low. Concern: the chat bar's 4.6 % depends on the mix of stated bet probabilities; it can drift. Mitigation: the fixture test pins today's mix; the replay eval re-measures it.
- Severity: low. Concern: the replay eval reads `substrate_node_prediction_error_history`, which is being retired; once it is gone the eval prints "no writes" for those nodes. It is also UNVERIFIED that each history row's `observed_at` equals the live node's `observed_at`, which the eval's decay mirror assumes. Mitigation: the joint rebuild already reads the backing table; port the rest when the history table is dropped.
- Severity: low. Concern: room/body windows with a single outcome cross z 2.0 at 7.3 % calm, not 5 %. Mitigation: their windows normally hold 2-6 outcomes; stated in the module docstring.
- Issues: Fixes #3821. Opened: (filled below).

Fixes #3821

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
