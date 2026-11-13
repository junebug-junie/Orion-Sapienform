# Chat-turn surprise replaces the chat prediction error (PE rebuild, retire step 6)

## Summary

- Orion gets a real chat prediction error: **node:prediction.chat_turn**. Before each chat turn, stance already bets privately on how the turn will go. When the turn ends, orion-prediction settles the bets. The new node says how much the last 30 minutes of chat turns surprised those bets. It rests at exactly 0 when the turns went the way the bets leaned.
- Only two bet kinds are used: "will the reply ask a question" and "will it take more steps than usual". They are the only kinds that beat the chat base rate on held-out data. "Fetches outside" and "longer reply than usual" do not, so they are left out.
- The old **node:substrate.chat** is retired end to end (kill means kill). It was a hint-change detector, not a forecast. Its producer, its readers, its glossary entry, its metric URN, its self-model domain and its tests are gone. The field digester prunes the stored node, and every generic reader skips the id.
- New bus event `prediction.turn_surprise.v1` on `orion:prediction:turn_surprise` (producer orion-prediction, consumer orion-substrate-runtime). One flag per side, shipped on.
- Eval `services/orion-prediction/evals/run_turn_surprise_replay_eval.py` replays every settled chat bet (live or the committed fixture) and gates the bet kinds and the rest point in CI.

Spec: `docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md` (PR #3168), "Retire order" step 6 and 7. Pattern: PR #3202 (room/body nodes) and PR #3201 (retire bus_synaptic/route).

## Outcome moved

- The chat prediction error is now a scored forecast: stated before the turn, settled when the turn ends, surprise in one unit (nats), calm point set by theory. The old signal was 65 % exact 0 by clipping and spiked after quiet.
- Replay over every settled chat turn (378 turns, 2026-10-24 .. 11-13): per-turn z mean -0.03, sd 1.00 (theory: 0 and 1). 86 % of turns rest at exactly 0.

## Metric quality gate

1. **Provenance.** Stated p: `thought_decision.self_prediction` (stance, before the motor runs). Outcome: `services/orion-prediction/app/turn_self_bets.py::settle` from `harness_turn_trace.run_artifact` ("?" in the last 300 characters; step count > the caller's 7-day median). Surprise: `app/turn_surprise.py::score_turn` -> `bet_terms` / `turn_surprise`, run in `TurnSelfBets.surprise_events` after each settle pass. Node value: `orion/substrate/prediction_turn_surprise.py::TurnSurpriseState.reading`, written by `services/orion-substrate-runtime/app/worker.py::_process_prediction_turn_surprise`.
2. **Independence.**
   - vs `node:substrate.chat`: different input (stance's stated probabilities and the turn's own record, not chat pressure hints). Both move when there is chat, so they are not unrelated: at the 137 moments both had a reading, Pearson +0.11, Spearman +0.20. The old one is retired in this PR, so there is no double count.
   - vs `node:prediction.room` / `.body`: no shared input (temperature forecasts). No causal chain.
   - vs `node:substrate.execution`: execution reads harness runs' failure/step pressure hints; the "more steps than usual" outcome reads the same run's step count. Short chain on that one bet. Execution is a hint-change z (not a forecast) and is itself scheduled for retirement in step 6; flagged here, not treated as new signal.
3. **Theory anchor.** Surprisal of a stated probability (proper log score). For a calibrated forecast q, the excess `-ln q(y) - H(q) = -(y - q) logit(q)` has mean 0 and variance `q(1-q) logit(q)^2` exactly (tested). Summed over a turn's bets and divided by the summed variance, z is about N(0, 1); calm rests at 0 by math, not tuning. Stated numbers are not calibrated (more_steps: stated 0.17, happened 0.25), so each p goes through a held-out logistic recalibration first (Platt map, ridge toward the identity, fit only on earlier settled chat bets).
4. **Live rest-state check.** Replayed every settled probability bet from Postgres (21 days, 12,766 rows; chat 1,622):

   Held-out skill (log-loss gap per bet over the chat caller's own trailing 7-day base rate; SE = larger of iid and by-day cluster; P(better) = Phi(gap / SE), no alpha line):

   | bet kind (chat) | n | gap (nats/bet) | SE | P(better) | used |
   |---|---|---|---|---|---|
   | asks_question | 378 | +0.155 | 0.035 | 1.000 | yes |
   | more_steps_than_usual | 179 | +0.179 | 0.031 | 1.000 | yes |
   | fetches_outside | 378 | -0.027 | 0.015 | 0.037 | no (chat never fetched: 0 of 378) |
   | longer_reply_than_usual | 179 | -0.416 | 0.347 | 0.115 | no (worse than the base rate) |

   Across all callers pooled (each against its own base rate) no kind beats the base rate (asks -0.027, P 0.30; fetches -0.51; more_steps +0.031, P 0.95; longer_reply +0.007, P 0.57). That is why the node is chat-only. The split is not near any line: the used kinds sit more than 4 SE above 0, the dropped kinds below 0. Caveat: the kinds were chosen on the same 21 days the rest numbers come from; with gaps this far apart the selection effect is small, and the CI eval re-checks the gap.

   Calibration after the held-out map (calm theory: mean excess 0): asks_question +0.002 (0.08 SE, sd 0.60 vs theory 0.58); more_steps -0.074 (-2.4 SE, sd 0.40 vs theory 0.52), so more_steps reads slightly calmer than theory (the map is a little under-confident there).

   Rest state:
   - per turn, all 378: z mean -0.03, sd 1.00, p10 -0.52, p50 -0.31, p90 +0.92, max +7.6. Exact 0: **86.2 %**. Value >= 0.55 (curiosity's bar): 6.9 %. Saturated at 1.0: 2.1 %.
   - per turn, last 7 days (140): z mean -0.06, sd 1.00; exact 0 **82.1 %**; >= 0.55 7.1 %.
   - the node (30-min fold, written at each settle): 378 writes, median 2 turns per window, exact 0 **65.9 %**, p50 0.0, p90 0.44, >= 0.55 6.9 %. Fresh 27 % of the 13.9 days, absent (stale) otherwise.
   - Not a decayed-to-zero artifact: the node is written only by events, never by a decay loop (FalkorDB node only, no field receipt), and an exact 0 comes from a turn whose z <= 0, which the replay recovers from the raw bets.
5. **Existing mechanism.** The bets, settle and grade already existed (PR #2976, #3042/#3044, #3077/#3083: `GET /turn-self-bets/live`). No surprise signal existed on them. Reuses the node writer, PE history (`node:prediction.*` prefix), attention, self-readings and curiosity readers (`surprise_excess`, #3202).
6. **Reversibility.** Cheap: two flags off (`PREDICTION_TURN_SURPRISE_EVENTS_ENABLED`, `SUBSTRATE_PREDICTION_TURN_SURPRISE_NODE_ENABLED`), then delete two modules, one glossary entry and one channel. No table, no migration. The retirement half is a `git revert`.

## What the old node held, and the shift (live, read-only)

| consumer | node:substrate.chat today | after this PR |
|---|---|---|
| Attention frames (7 d) | top target in 3,222 of about 124,000 frames (2.6 %, queried 2026-11-13) | node:prediction.chat_turn competes as an event-gated `surprise_excess` body signal (news for 300 s after each write) |
| Goal emission (attention-runtime log, 11-12 08:59 .. 11-13 00:09) | 1,185 of 7,199 published goals (16 %) | gone; body goals were already denied autonomy (`goal_not_about_world`, #3169), so no fetch/recall/journal permission changes |
| Curiosity seeds (substrate-runtime log since 11-12 21:35) | 1 of 48 seed passes | chat_turn seeds only when >= 0.55 and fresh (~7 % of writes) |
| Self-model confidence (7 d mean) | chat was a domain, fresh 91 % of ticks | domains execution + biometrics; mean confidence 0.917 -> 0.909 on the same rows |
| Self-signal episodes (equilibrium log, ~16 h) | 4 mentions | none (episode gate reads `node:substrate.*` only; not widened, same as #3202) |
| Self-readings prompt line | "surprise in how conversation load and repair are moving" | "how much my recent chat turns surprised my own bets about them" |
| Act ledger workspace binding | `hub.chat_reply` answered `world:chat` or `node:substrate.chat` | `world:chat` only |

Plain version: Orion stops feeling chat as a near-constant body signal (fresh 91 % of the time) and starts feeling it only right after a chat turn, and only when the turn went differently than Orion expected. Attention and goals lose a signal that won about 3 % of frames and 16 % of goals; confidence drops by about 0.01 on average.

## Proposal mode (CLAUDE.md 0A)

- **Capability change:** one self-reading replaced by a real forecast miss; one self-model domain removed.
- **Data touched:** settled turn self-bets (`prediction_turn_self_bet`), read-only. The event carries correlation id, faculty, probabilities and outcomes, no turn text.
- **Privacy boundary:** none crossed (Orion's own bets about its own turns).
- **Trace that proves it works:** `turn_surprise_published` (orion-prediction log, `/health` `turn_self_bets.turn_surprise`), `substrate_prediction_turn_surprise ... verdict=accepted` (substrate-runtime log, `/health` `prediction_turn_surprise`), rows for `node:prediction.chat_turn` in `substrate_node_prediction_error_history`.
- **Dangerous failure:** a stance prompt change that shifts the stated probabilities could read as surprise for up to 7 days until the recalibration window catches up. Mitigation: the map refits on every turn from the trailing 7 days; the eval names the gap per kind; re-run it after any stance prompt change.
- **Rollback:** the two flags above; `git revert` for the retirement.

## Current architecture

`node:substrate.chat` was written by substrate-runtime's `_chat_tick` from `chat_prediction_error()` (EWMA z of the mean change of `conversation_load` and `repair_pressure` over touched turns, negatives clipped) with a field receipt every tick. It was an Active-Inference domain, an attention native target, a self-reading, an act-ledger workspace ref, a mood-arc enrichment channel and a world-model feature slot. orion-prediction settled turn self-bets every 60 s and graded them, but nothing turned them into a signal.

## Architecture touched

- Contract: `PredictionTurnSurpriseV1` / `PredictionTurnSurpriseBetV1` (both registries), channel `orion:prediction:turn_surprise`.
- orion-prediction: scorer module, settle-pass hook, publisher, `/health`, flag, store query.
- orion-substrate-runtime: listener, fold, node write, `/health`, flag; chat tick keeps only the reducer; domain map; world-model slot 1 now padding.
- Shared: retired-id set, self-model domains, selectors native targets, self-readings words, outcome circular pair, act ledger refs, PE definition versions, mood-arc enrichment, glossary, metric semantics + lock, sentience instruments.
- field-digester: `RETIRED_PSEUDO_NODES` prunes `node:substrate.chat`.
- hub / vector-host: domain list and tooltip text.

## Files changed

- `services/orion-prediction/app/turn_surprise.py` (new): scorer, held-out calibration, held-out skill.
- `services/orion-prediction/app/{turn_self_bets,main,store,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`.
- `services/orion-prediction/evals/run_turn_surprise_replay_eval.py` (new) + `evals/fixtures/turn_self_bets_chat_2026-11-13.json` (real chat bets, ids hashed).
- `services/orion-prediction/tests/test_turn_surprise.py` (new).
- `orion/schemas/prediction.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`.
- `orion/substrate/prediction_turn_surprise.py` (new) + `orion/substrate/tests/test_prediction_turn_surprise.py` (new).
- `services/orion-substrate-runtime/app/{worker,settings,main,world_model_features}.py`, `.env_example`, `docker-compose.yml`, `README.md`, tests (new listener test; chat-tick guard; domain map, receipt, node-write, PE history, stale-omit, world-model tests re-based).
- Retirement: `orion/substrate/{prediction_error,prediction_error_freshness,attention_self_model}.py`, `orion/attention/{self_readings,self_signal_outcomes}.py`, `orion/attention/field_attention/selectors.py`, `orion/grammar/act_ledger.py`, `orion/schemas/{prediction_error_definitions,chat_projection}.py`, `orion/mood_arc/corpus_enrichment.py`, `orion/substrate/chat_loop/reducer.py`, `services/orion-field-digester/app/tensor/channels.py`, `services/orion-hub/scripts/{attention_organ_routes,outreach_vocabulary}.py`, `services/orion-hub/static/js/hub-surface.js`, `services/orion-vector-host/app/static/index.html`, `scripts/analysis/measure_ast_hot_reducer.py`.
- Deleted: `scripts/analysis/replay_route_chat_prediction_error_definitions.py` + its test, `services/orion-substrate-runtime/tests/test_worker_chat_tick_baseline_persistence.py`, the chat tests in `orion/substrate/tests/test_prediction_error.py`.
- `config/field/field_channel_glossary.v1.yaml`, `orion/metrics/semantics.py`, `config/metrics/metric_definitions.lock.json`, `orion/sentience_striving_program/instruments.yaml`.
- `scripts/sync_local_env_from_example.py` (prefix `SUBSTRATE_PREDICTION_TURN_SURPRISE_`).
- `.github/workflows/{orion-prediction-tests,prediction-surprise-node-tests,retired-pe-guards}.yml`.

## Schema / bus / API changes

- Added: `PredictionTurnSurpriseV1` / `prediction.turn_surprise.v1` on `orion:prediction:turn_surprise`; node `node:prediction.chat_turn`; metric URNs `metric://bus_channel/orion-prediction/orion:prediction:turn_surprise`, `metric://field_channel/orion-substrate-runtime/node:prediction.chat_turn.prediction_error`; `/health` `turn_self_bets.turn_surprise` (prediction) and `prediction_turn_surprise` (substrate-runtime).
- Removed: `chat_prediction_error()`, `chat_raw_reading()`; metric URN `node:substrate.chat.prediction_error` (HIGH in the drift gate, intended); `chat_session` PE definition version; `ExecutionContextScalars.chat`; "chat" from `ACTIVE_INFERENCE_DOMAINS`.
- Behavior changed: confidence averages 2 domains; world-model execution_context slot 1 is 0.0 padding (dim unchanged).
- Compatibility: `ChatSessionProjectionV1`'s three baseline fields are kept (dead, `extra="forbid"` would reject stored rows otherwise). An older substrate-runtime ignores the new channel.

## Env/config changes

- Added keys: `PREDICTION_TURN_SURPRISE_EVENTS_ENABLED=true` (orion-prediction); `SUBSTRATE_PREDICTION_TURN_SURPRISE_NODE_ENABLED=true`, `SUBSTRATE_PREDICTION_TURN_SURPRISE_CHANNEL=orion:prediction:turn_surprise` (orion-substrate-runtime).
- Removed / renamed keys: none.
- `.env_example` updated: yes; settings and compose fallbacks match.
- local `.env` synced: the three keys were appended to `/mnt/scripts/Orion-Sapienform/services/{orion-prediction,orion-substrate-runtime}/.env` (the sync script resolves `.env` under the worktree it runs from, so it cannot reach the primary checkout's files from a worktree). `check_env_template_parity.py`: PASS.
- skipped keys requiring operator action: none.

## Tests run

TESTS_PLACEHOLDER

## Evals run

```text
evals/run_turn_surprise_replay_eval.py --days 21 (live, read-only, 2026-11-13): numbers in the gate section.
evals/run_turn_surprise_replay_eval.py --fixture evals/fixtures/turn_self_bets_chat_2026-11-13.json --check -> CHECK PASS (now a CI step)
```

## Docker/build/smoke checks

```text
No image rebuild needed for the tests (orion-prediction code mounted read-only into its existing image; substrate-runtime and shared suites on the repo .venv).
Live path UNVERIFIED until deploy. After the post-pull rebuild:
curl -fsS http://localhost:8132/health | jq '.turn_self_bets.turn_surprise'
docker logs --since 1h orion-athena-prediction 2>&1 | grep turn_surprise_published
docker logs --since 1h orion-athena-substrate-runtime 2>&1 | grep 'substrate_prediction_turn_surprise '
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "SELECT count(*), max(observed_at), round(avg((value=0)::int)::numeric,3) zero_share FROM substrate_node_prediction_error_history WHERE node_id='node:prediction.chat_turn' AND observed_at > now()-interval '1 day'"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "SELECT max(observed_at) FROM substrate_node_prediction_error_history WHERE node_id='node:substrate.chat'"   # stays at the pre-deploy time
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

Post-pull rebuild from the clean primary main checkout after merge:

```bash
scripts/rebuild_services_from_git_diff.sh
```

Services: orion-prediction, orion-substrate-runtime, orion-field-digester (prune), orion-attention-runtime and orion-hub (shared readers), orion-thought / orion-equilibrium (shared attention readers on next rebuild).

Post-deploy cleanup (production writes, for Juniper to run, not run here):

```bash
docker exec orion-athena-falkordb redis-cli GRAPH.QUERY orion_substrate "MATCH (n) WHERE n.node_id = 'node:substrate.chat' DETACH DELETE n"
```

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "DELETE FROM substrate_node_prediction_error_baseline WHERE target_id = 'node:substrate.chat'"
```

History rows in `substrate_node_prediction_error_history` age out by the 168 h prune; readers already skip the id.

## Risks / concerns

- Severity: medium. Concern: the node is fresh only ~27 % of the time (only after chat turns), against 91 % for the old one. Attention, goals and the confidence number shift at once. Mitigation: intended (absent is not calm); numbers above; re-measure goal mix and attention share 7 days after deploy.
- Severity: medium. Concern: kind selection and rest numbers come from the same 21 days. Mitigation: gaps > 4 SE apart; the CI eval re-checks; re-run live weekly and after any stance prompt change.
- Severity: low. Concern: "more steps than usual" shares the turn's step count with the execution PE. Mitigation: flagged in the gate; execution is next to retire.
- Severity: low. Concern: a stance prompt change can read as surprise until the 7-day recalibration window catches up. Mitigation: the map refits per turn.
- Severity: low. Concern: equilibrium's episode gate and outcome rollup still read `node:substrate.*` only, so chat surprise opens no metacog episodes yet. Mitigation: same decision as #3202; widen after a week of live rows.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
