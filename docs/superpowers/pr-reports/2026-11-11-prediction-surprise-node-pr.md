## Summary

- Orion gets two new body signals: **node:prediction.room** (office temperature) and **node:prediction.body** (cabinet temperature). Each rests at 0 when the temperature does what Orion's own forecast expected, and rises when it does something the forecast did not expect.
- orion-prediction now publishes `prediction.surprise.v1` on `orion:prediction:surprise` for each scored forecast that passes an issue-time gate. A missing reading never publishes.
- orion-substrate-runtime listens, folds the last 30 minutes of outcomes per node into one number, and writes the node. No outcomes = no writes, so the node goes stale (absent), never calm.
- Attention, self-readings and curiosity accept the new measurement kind `surprise_excess`; the PE history, the broadcast contest and the field contest accept the `node:prediction.` prefix.
- Build item 3 of the prediction-error suite rebuild (spec PR #3168). Juniper approved the build on 2026-11-11.

## Outcome moved

The first prediction error that reaches Orion's attention which is a real forecast miss: a forecast stored before its target time, scored when the clock says, measured as surprise in one unit, with a calm point set by theory. Before this patch, all 12 prediction-error nodes were non-forecasts (spec audit, 2026-11-08).

## Proposal-mode record (approved 2026-11-11)

- **Capability change:** two new body signals Orion can feel (attention contest, self-readings line, curiosity seed).
- **Data touched:** only orion-prediction's own forecast scores (`prediction_forecast`, `prediction_model_run`). No personal data.
- **Privacy boundary:** none crossed; office and cabinet temperature only.
- **Trace that proves it works:** `prediction_surprise_published` (orion-prediction log + `/health`), `substrate_prediction_surprise ... verdict=accepted` (substrate-runtime log + `/health` `prediction_surprise`), rows for `node:prediction.*` in `substrate_node_prediction_error_history`.
- **Dangerous failure:** a broken sensor read as huge surprise. Guards: a missing bucket is never scored as surprise (producer gate `no_actual`, schema requires `actual`), and the camera-gap fix #3171 already makes missing readings absent upstream. A dead producer reads stale, never calm.
- **Rollback:** one flag per side, shipped ON: `PREDICTION_SURPRISE_EVENTS_ENABLED` (producer) and `SUBSTRATE_PREDICTION_SURPRISE_NODES_ENABLED` (consumer). Off = no events / no writes; the nodes go stale and drop out of attention.

## Metric quality gate

1. **Provenance.** Per outcome: `services/orion-prediction/app/engine.py::surprise_events` -> `app/surprise.py::conformal_p_value` over the stored quantiles (`prediction_forecast.quantiles`, written at issue) and `actual` (written by `Engine.score` from the 5-minute bucket mean). Node value: `orion/substrate/prediction_surprise.py::PredictionSurpriseState.reading`, written by `services/orion-substrate-runtime/app/worker.py::_process_prediction_surprise`.
2. **Independence.** Shares its input rows with live 80 % coverage (`in_80`) in orion-prediction: coverage is the slow drift reading, surprise the fast spike reading, by design. No causal link to any other PE node (no bus, chat, execution, camera input). The cabinet node and `node:substrate.cabinet` (warming trigger) both read the cabinet temperature: the trigger fires on a rise past a fixed threshold, this node on a value the forecast did not expect (a rise the forecast saw coming reads calm here).
3. **Theory anchor.** Probability integral transform: under a calibrated forecaster the CDF at the outcome is uniform, so `p` is uniform and `E[-ln max(p, 0.05)] = 1 - 0.05 = 0.95`; excess rests at 0. The node uses the same theory for its spread: `Var[-ln max(p, P)] = 2 - 2P(1 + ln 1/P) - (1 - P)^2 = 0.698` (checked by simulation: 0.6979).
4. **Live sanity.** Calm check re-run 2026-11-11 08:40 UTC (below): passes on every adaptive horizon. Node replay over 72 h of real rows: exact 0 on 56-58 % of writes (rest reachable), varies, saturates on real miss runs. Single-outcome excess sd 0.85-0.86 vs 0.835 theory. Not degenerate.
5. **Existing mechanism.** No forecast-surprise signal existed. Reuses the existing node writer, PE history, attention contests, self-readings and curiosity readers; no new reader.
6. **Reversibility.** Cheap: two flags off, then delete the module, the glossary entries and the channel. No table, no migration. History rows age out in 7 days.

## The gate (producer)

A scored row publishes only if all hold (`app/surprise.py::event_gate`, judged on the row alone):

- role is `champion`;
- the model that served this horizon in the run that ISSUED the forecast (`prediction_model_run.assignment ->> horizon`, joined in `store.due_for_scoring`) is not a baseline;
- `prediction_forecast.interval_source = 'adaptive'` (5 and 15 min only: the horizons whose calm check passed);
- the bucket has a real, finite actual.

Because the gate needs no monitor pass, outcomes scored right after a restart are judged like any other (#3170 review carry-over). Event building sits in its own `try` after `write_scores`; a failure is recorded in `/health` and never skips exceedance or shadow scoring (#3170 carry-over, tested). A failed publish drops that event only and keeps going.

Live today: cabinet 5 min is still on backtest ranges (its model changed 2026-11-10 13:08 and the adaptive pool refills), so the cabinet node is fed by 15-min outcomes only until then.

## The aggregation (consumer)

Per node, the outcomes whose target bucket lies in the last 30 minutes, from independent origins only (one origin per horizon-long clock block: every 5-min origin, one in three 15-min origins; same thinning as the calm check), and one outcome per target bucket (the shortest horizon wins: a 5- and a 15-min forecast of the same bucket share its actual). Value = summed excess divided by its calm spread (a z-score), stored as `clip(z / 3, 0, 1)` (3 sigma = 1.0, the scale chat/execution use).

Why a z-score, not the plain mean: the window holds 2 outcomes (cabinet today) or 6 (office). With a plain mean the calm cabinet crossed curiosity's 0.55 bar on 22 % of writes, the office on 9 %. With the z-score the calm crossing rate is ~5 % for any n (unit test by simulation).

Events whose target is already outside the window are dropped, so a backlog scored after an outage never writes the past as now. FalkorDB node only (no field receipt): the field digester's decay/expiry (`NODE_DECAY_CHANNELS`) never touches it, so it cannot decay to a fake 0. Stale = `observed_at` ages; attention, self-readings and curiosity already treat a reading older than 1800 s as stale/absent (`orion/substrate/prediction_error_freshness.py` horizon). The nodes are not on #3201's retired list.

## How attention and curiosity treat the new kind

- `surprise_excess` is added to `MEASUREMENT_VALUE_KINDS` (world_first) on purpose: the calm check passed, so it is a real measurement. Ranked like a level against its own 7-day history (bad-direction percentile); eligible when high/unusual for itself and fresh.
- Not a "continuous" kind (no sustained-dwell rule): the value is already a 30-min mean, written ~every 5 min.
- Not event-written (no 300 s orienting fade): `sparsity: per_tick`.
- Both nodes are internal (body), as approved. Body-sourced goals do not unlock autonomous fetch/recall (#3169 rule unchanged).
- Self-readings: listed via `prompt_sites`; plain words "how much my office's / cabinet's temperature is surprising my own forecast". Pinned in `PROMPT_INVENTORY_URNS`.
- Curiosity: seeds when the stored value is >= 0.55 (its existing bar) and fresh.
- Not widened (deliberately, documented in the substrate-runtime README): equilibrium's self-signal episode gate and outcome rollup (no metacog episodes or "tends to" associations for these nodes yet), Hub's attention-organ history route. The new nodes reach Orion through attention, self-readings and curiosity first; widen these after a week of live rows. `short_signal` in `self_signal_outcomes.py` already strips `node:` so a future association line reads `prediction.room`.

## Current architecture

orion-prediction forecast and scored cabinet/office temperature every 5 min with adaptive ranges at 5/15 min (#3175) and had a read-only surprise measurement (`/series/{id}/surprise`, #3170) with no bus event and no consumer. Every PE reader in attention, self-readings and the history recorder accepted only `node:substrate.*` nodes and the kinds level/count/trigger/binary.

## Architecture touched

- Contract: new schema `PredictionSurpriseV1` (both registries; `resolve()` verified), new channel `orion:prediction:surprise` (producer orion-prediction, consumer orion-substrate-runtime).
- orion-prediction: event gate, event build/publish, `/health` counters, flag.
- orion-substrate-runtime: listener, fold, node write, `/health` `prediction_surprise`, PE history prefix, flag.
- Shared readers: world_first (kind + prefixes), self_readings (kind, words, short name), attention_broadcast and attention-runtime field contest (prefix), metric semantics (kind + prompt pins), glossary (2 entries), sentience instruments (1 entry), metric lock.

## Files changed

- `orion/schemas/prediction.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`: the event contract.
- `services/orion-prediction/app/{surprise,engine,store,main,settings}.py`: gate, events, issuing-model join, health, flag.
- `services/orion-prediction/{.env_example,docker-compose.yml,README.md}`: flag + docs.
- `services/orion-prediction/evals/run_surprise_node_replay_eval.py` (new): gate + fold replay on live rows; `run_surprise_calm_live_eval.py`: docstring records the passing check.
- `orion/substrate/prediction_surprise.py` (new): the fold.
- `services/orion-substrate-runtime/app/{worker,settings,main}.py`, `.env_example`, `docker-compose.yml`, `README.md`: listener, flag, health, history prefix.
- `orion/attention/world_first.py`, `orion/attention/self_readings.py`, `orion/substrate/attention_broadcast.py`, `services/orion-attention-runtime/app/worker.py`: accept the kind and prefix.
- `orion/metrics/semantics.py`, `config/field/field_channel_glossary.v1.yaml`, `config/metrics/metric_definitions.lock.json`: semantics, entries, re-lock.
- `orion/sentience_striving_program/instruments.yaml`: `forecast_surprise_nodes` instrument with the calm-check claim.
- `scripts/sync_local_env_from_example.py`: `SUBSTRATE_PREDICTION_SURPRISE_` prefix (else the key never reached the local `.env`).
- `.github/workflows/prediction-surprise-node-tests.yml` (new), `.github/workflows/orion-prediction-tests.yml` (paths).
- Tests: `orion/substrate/tests/test_prediction_surprise.py`, `services/orion-substrate-runtime/tests/test_prediction_surprise_listener.py` (new); `services/orion-prediction/tests/{test_engine,test_surprise,test_schemas}.py`, `tests/test_attention_{world_first,self_readings,event_decay}.py`, `orion/substrate/tests/test_endogenous_curiosity.py`, `tests/test_field_channel_glossary.py`, `tests/test_metric_rest_range.py`, `tests/fixtures/self_readings_history.py`.

## Schema / bus / API changes

- Added: `PredictionSurpriseV1` / `prediction.surprise.v1` on `orion:prediction:surprise`; `/health` fields `series.<id>.surprise_events` (orion-prediction) and `prediction_surprise` (substrate-runtime); value kind `surprise_excess`; node ids `node:prediction.room`, `node:prediction.body`.
- Removed / Renamed: none.
- Behavior changed: PE history, broadcast contest and field contest now include `node:prediction.*` nodes.
- Compatibility: additive. The consumer only exists in substrate-runtime, which ships in the same PR; an older substrate-runtime ignores the channel.

## Env/config changes

- Added keys: `PREDICTION_SURPRISE_EVENTS_ENABLED=true` (orion-prediction); `SUBSTRATE_PREDICTION_SURPRISE_NODES_ENABLED=true`, `SUBSTRATE_PREDICTION_SURPRISE_CHANNEL=orion:prediction:surprise` (orion-substrate-runtime).
- Removed / renamed keys: none.
- `.env_example` updated: yes (both services), settings + compose fallbacks match.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes; all three keys confirmed in `/mnt/scripts/Orion-Sapienform/services/{orion-prediction,orion-substrate-runtime}/.env`.
- skipped keys requiring operator action: none.

## Tests run

```text
orion-prediction (its image, worktree :ro): pytest tests -> 351 passed
orion/substrate/tests/test_prediction_surprise.py + substrate-runtime test_prediction_surprise_listener.py + attention/self-readings/curiosity/glossary/rest-range/outcome suites -> 229 passed
orion/substrate/tests + services/orion-attention-runtime/tests -> 1234 passed, 4 failed (pre-existing on main: test_felt_state_*)
services/orion-substrate-runtime/tests -> 579 passed, 13 failed (same 13 fail on clean main: cursor_reset_auth, cursor_tail_seed, quarantine_truth, reducer_health, attention_self_model_tick x2, falkor_routed_store, worker_reducer); test_grammar_consumer_integration collection error pre-existing
root tests touching glossary/attention/metrics/registry/channels (92 files) -> 1346 passed; remaining failures pre-existing on main (goal_proposal registry, exec_result catalog, phase4 dynamics x2, blast radius, memory card registry gap, single-consumer channels)
scripts/check_metric_lineage.py --gate -> PASS (785 URNs; bus_channel orphans stay 17)
scripts/check_metric_lineage.py --prompt-semantics -> PASS
scripts/check_definition_drift.py --gate -> PASS after --update (3 added, 0 high)
scripts/check_env_template_parity.py -> PASS; check_env_key_single_source.py -> OK
orion-static-gates.yml commands run locally -> all pass after re-lock
docker compose config (both services, primary .env files) -> new keys resolve "true"
```

## Evals run

```text
evals/run_surprise_calm_live_eval.py --hours 24 --source adaptive (2026-11-11 08:40 UTC, live, read-only):
athena.cabinet_temp_c   5 min  n=  52 (ind  52)  excess +0.004  +-0.230 calm=True  p<=.05 0.038  p<=.2 0.250  cov80 0.850
athena.cabinet_temp_c  15 min  n= 286 (ind  96)  excess -0.072  +-0.167 calm=True  p<=.05 0.056  p<=.2 0.210  cov80 0.790
office.temp_c           5 min  n= 286 (ind 286)  excess -0.032  +-0.099 calm=True  p<=.05 0.049  p<=.2 0.199  cov80 0.801
office.temp_c          15 min  n= 286 (ind  96)  excess -0.050  +-0.179 calm=True  p<=.05 0.080  p<=.2 0.210  cov80 0.790

evals/run_surprise_node_replay_eval.py --hours 72 (live rows, read-only, after the review fix: one outcome per target bucket):
node:prediction.room: writes 398  samples/window median 5  single-outcome excess sd 0.864 (calm theory 0.835)
  z            mean -0.080  sd 1.201  p90 +1.549  max +4.260
  node value   exact 0: 56%  p50 0.000  p90 0.516  p99 1.000  max 1.000  share >= 0.55 (curiosity bar): 9.8%
node:prediction.body: writes 155  samples/window median 2  single-outcome excess sd 0.848 (calm theory 0.835)
  z            mean -0.008  sd 1.035  p90 +1.559  max +2.974
  node value   exact 0: 58%  p50 0.000  p90 0.520  p99 0.878  max 0.991  share >= 0.55 (curiosity bar): 8.4%
(before the fix, counting both horizons of a bucket: room z sd 1.37, crossing 12.6 %)
```

## Docker/build/smoke checks

```text
No image rebuild needed for the tests (code mounted read-only into the existing orion-prediction image).
Live path UNVERIFIED until deploy. After the post-pull rebuild of orion-prediction and orion-substrate-runtime:
curl -fsS http://localhost:8132/health | jq '.series[] | .surprise_events'
docker logs --since 15m orion-athena-prediction 2>&1 | grep prediction_surprise_published
docker logs --since 15m orion-athena-substrate-runtime 2>&1 | grep 'substrate_prediction_surprise '
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "SELECT node_id, count(*), max(observed_at), round(avg(value)::numeric,3) FROM substrate_node_prediction_error_history WHERE node_id LIKE 'node:prediction.%' AND observed_at > now() - interval '1 hour' GROUP BY 1"
```

## Review findings fixed

Review ran in a subagent against `git diff origin/main...HEAD`. No high-severity findings.

- Finding (medium): calm false-alarm rate 2-3x the documented 5 % (5- and 15-min outcomes of one bucket double-counted; misses autocorrelate).
  - Fix: one outcome per target bucket (shortest horizon). Remaining lag-1 correlation documented in the module, glossary, instrument and risks, with a concrete next step.
  - Evidence: replay room crossing 12.6 % -> 9.8 %, z sd 1.37 -> 1.20; `test_one_outcome_per_target_bucket_the_shortest_horizon_wins`.
- Finding (medium): race between `/health` reading the window and the handler thread adding to it (lost sample, "dict changed size").
  - Fix: `threading.RLock` around add/prune/read; `/health` reads a counts snapshot.
  - Evidence: `orion/substrate/prediction_surprise.py` `_lock`; suites pass.
- Finding (medium): readers still filtering on `node:substrate.` without a stated reason.
  - Fix: decision stated (README + this report): episode gate, outcome rollup and Hub route stay `node:substrate.*` for now; `short_signal` strips `node:` so a future line is readable.
  - Evidence: `orion/attention/self_signal_outcomes.py`; self-signal outcome tests pass.
- Finding (low-medium): self-reading unit said "mean excess" for a stored z-score.
  - Fix: unit now "surprise z-score / 3, last 30 min".
- Finding (low): `node_writes_total` counted writes the shared writer skipped.
  - Fix: `_write_prediction_error_node` returns True only on a real upsert; skipped writes count as `node_write_skipped_total`.
  - Evidence: `test_a_write_the_shared_writer_skipped_is_not_counted_as_written`.
- Finding (low): a high value stays current up to 1800 s after the producer stops.
  - Fix: documented in `absent_means` and the module (bounded by the existing staleness horizon).
- Finding (low): origin alignment used exact-second modulo.
  - Fix: whole-minute arithmetic; `test_an_origin_with_seconds_still_counts`.
- Finding (low): new SQL only tested with a fake store.
  - Fix: verified against the live DB by hand (query in the review trail) and by the replay eval, which reads `served_model` through the same join: 0 `no_issuing_model` refusals over 72 h of live rows.

## Restart required

Post-pull rebuild from the clean primary main checkout after merge (both services change code and env):

```bash
scripts/rebuild_services_from_git_diff.sh
```

Services: `orion-prediction`, `orion-substrate-runtime`, `orion-attention-runtime` (field contest prefix). Readers in Hub/thought that import `orion/attention` pick up the new kind on their next rebuild.

## Risks / concerns

- Severity: medium. Concern: consecutive 5-min misses are mildly autocorrelated (lag-1 0.12), so the live calm z spread is 1.0-1.2, not 1.0. The office node crosses curiosity's 0.55 bar on ~10 % of writes on a calm day (theory 5 %; 13 % before the one-outcome-per-bucket fix). Mitigation: attention ranks against the node's own week, so this only affects curiosity's absolute bar. Recommendation: after 7 days of live rows, count curiosity seeds from `node:prediction.*`; if they crowd out other seeds, divide the window variance by the measured autocorrelation (effective-n correction) rather than raising the bar.
- Severity: low. Concern: the cabinet node rests on 2 outcomes per window until its 5-min adaptive range comes back (pool refills ~24 h after the 11-10 13:08 model change). Mitigation: the z-score keeps its calm crossing rate the same for any n.
- Severity: low. Concern: #3201 (item 4) edits the same glossary, metric lock, self_readings, substrate-runtime worker/settings/env. Mitigation: changes here are additive; whichever merges second rebases and re-runs `check_definition_drift.py --update`.
- Severity: low. Concern: is the office "body" or "world"? Shipped as body (internal) as approved. Moving it to world would let it emit world goals (#3169); Juniper's call.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
