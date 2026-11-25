## Summary

PE retire Patch 4: Orion gets one real perception prediction error, `node:prediction.street_rhythm`. It says how surprised the walkway rhythm forecast was by who did or didn't appear. It ships in **shadow**: written, recorded and visible, but attention and curiosity skip it until a 7-day calm check passes.

- orion-prediction reads the walkway rhythm learner's scored forecasts (`vision_percept_expectation`, read-only), scores the **raw** stated probability, and publishes one `prediction.street_rhythm_surprise.v1` per scored outcome.
- Unscorable windows (camera not watching) and people/vehicles not seen for 7+ days are refused, never counted as misses. The refusal rate is reported.
- orion-substrate-runtime folds the last 30 min of outcomes into the node (same z-of-summed-excess shape as `node:prediction.chat_turn`). It writes once per accepted outcome; a republished outcome is a duplicate and writes nothing. No clock rewrites, no private history floor (#3917 owns that).
- The node is a **world** node (`_WORLD_NODE_IDS`, 600 s staleness). Body-only gates and outcome rollups never read it.
- Shadow gate: one constant, `STREET_RHYTHM_SHADOW = True` in `orion/schemas/attention_candidate.py`. Going live is flipping it to `False`.
- Calm eval `services/orion-prediction/evals/run_street_rhythm_calm_eval.py` is the go-live gate (tail-based, per design rev 4). CI runs it on a committed 14-day hashed fixture.

Refs #3827 #3388

## Outcome moved

Before: the only "perception prediction errors" were pixel-change scores (now scene change) and nothing on the prediction-error lane forecast the outside world. After: a real forecast miss about the street exists, is scored against its own stated probability, rests at 0 by theory, and is recorded so its 7-day calm check can run before it is allowed to steer attention.

## Current architecture

- The sql-writer rhythm learner (`services/orion-sql-writer/app/vision_rhythm.py`) writes expectations before a window opens and scores them `met` / `missed` / `unscorable` after. Nothing read those scores into the substrate ("NOT wired to the substrate graph").
- Forecast surprise nodes (`node:prediction.room/.body/.room_joint/.body_joint/.chat_turn`) are produced by orion-prediction events folded in substrate-runtime.
- Patch 3 (#3874) moved the cameras off the prediction-error lane; the lane held 5 forecast nodes.

## Architecture touched

- orion-prediction: new read-only scorer loop + bus publish.
- orion-substrate-runtime: new bus listener + fold + node write.
- Contracts: new schema, new bus channel, glossary entry, metric lock, organ registry channel.
- Attention schema: world node set, shadow set, curiosity fence, staleness map entry.
- Curiosity seed bar for go-live (inactive while shadow).

## Files changed

- `services/orion-prediction/app/street_rhythm.py`: scorer, refusals, live pass (new).
- `services/orion-prediction/app/store.py`: `street_rhythm_outcomes` read-only query in its own block (apart from #3927's Juniper-reachable reads).
- `services/orion-prediction/app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md` (new "Street rhythm surprise" section): loop, publish, `/health.street_rhythm`, two env keys.
- `services/orion-prediction/evals/run_street_rhythm_calm_eval.py` + `evals/fixtures/street_rhythm_2026-11-25.json`: the calm gate and its fixture.
- `services/orion-prediction/tests/test_street_rhythm.py`: producer tests.
- `orion/substrate/prediction_street_rhythm.py` (+ test): the 30-min fold.
- `services/orion-substrate-runtime/app/worker.py`, `main.py`, `settings.py`, `.env_example`, `docker-compose.yml`, `README.md` (+ `tests/test_prediction_street_rhythm_listener.py`): listener, node write, `/health.prediction_street_rhythm` with `shadow`.
- `orion/schemas/prediction.py`, `orion/schemas/registry.py` (both maps, checked with `resolve()`), `orion/bus/channels.yaml` (entry next to joint surprise, away from #3924's edit).
- `orion/schemas/attention_candidate.py`: `STREET_RHYTHM_NODE_ID`, `STREET_RHYTHM_SHADOW`, `SHADOW_NODE_IDS`, world set, curiosity fence, `world_first_fenced`.
- `orion/attention/world_first.py`: 600 s staleness entry only.
- `orion/substrate/curiosity_seed_bar.py`: bar 1.8 / 3 for go-live.
- `orion/attention/self_readings.py`: lane order includes `street_rhythm`.
- `config/field/field_channel_glossary.v1.yaml`, `config/metrics/metric_definitions.lock.json`, `config/integration/organs.yaml`.
- `tests/test_pe_cameras_scene_change.py`: post-cutover prediction-error list now includes street_rhythm.
- `tests/test_pe_street_rhythm_shadow.py`: shadow and world-node tests, including the flip.

## Schema / bus / API changes

- Added: `PredictionStreetRhythmSurpriseV1` (`prediction.street_rhythm_surprise.v1`) on `orion:prediction:street_rhythm_surprise` (orion-prediction -> orion-substrate-runtime). Carries no names (hex id or class label).
- Added: `GET /health` keys `street_rhythm` (prediction) and `prediction_street_rhythm` (substrate-runtime).
- Removed / Renamed: none.
- Behavior changed: the prediction-error lane list (glossary) now holds six forecast nodes; the dated self-readings marker line names street_rhythm.
- Compatibility notes: additive. Old attention-runtime code does not know the id; restart it before substrate-runtime writes the node (order below).

## Env/config changes

- Added keys: `PREDICTION_STREET_RHYTHM_EVENTS_ENABLED=true`, `PREDICTION_STREET_RHYTHM_TICK_SEC=60` (orion-prediction); `SUBSTRATE_PREDICTION_STREET_RHYTHM_NODE_ENABLED=true`, `SUBSTRATE_PREDICTION_STREET_RHYTHM_CHANNEL=orion:prediction:street_rhythm_surprise` (orion-substrate-runtime).
- Removed / Renamed keys: none.
- `.env_example` updated: yes, both services; settings.py and compose match.
- local `.env` synced: the four keys were added to `/mnt/scripts/Orion-Sapienform/services/{orion-prediction,orion-substrate-runtime}/.env`.
- skipped keys requiring operator action: none.
- Shadow is a code constant, not an env key, on purpose: attention-runtime, substrate-runtime (broadcast + curiosity) and the Hub all read it, and per-service env flags could disagree (contests live, curiosity shadow). One constant, one line.

## Metric quality gate (CLAUDE.md 0A)

1. **Provenance.** Stated p = `vision_percept_expectation.confidence`, written by `vision_rhythm.py` before the window opens (fraction of observed days with an occurrence in the window). Outcome = `status` (met / missed / unscorable) from `score_window()`. Scored in `app/street_rhythm.py:outcome_terms`.
2. **Independence.** Shares the walkway camera with `perception_walkway` (scene change) and with the street alarm. Live r (last 7 days): node vs walkway scene change **0.034** (n 242), vs street alarm **-0.034** (n 254); per outcome vs walkway in its window **-0.124** (n 277), vs street alarm **0.001** (n 278). Caveat: walkway rows in this window are mostly scale v1 (old score, r 0.66-0.76 with the new one); v2 rows began 2026-11-25 06:10Z.
3. **Theory.** Log score of a stated probability: under calibration E[-ln p(outcome)] = H(p), so excess has mean 0 and variance p(1-p)logit(p)^2. The node's z is calm N(0, 1) -> exact 0 about half the time.
4. **Live data** (read-only, 2026-11-25 07:29Z):

   | window | closed | scored | refusal rate | mean excess (95 % CI) | p90 | p99 | max | misses at p >= 0.9 | gate |
   |---|---|---|---|---|---|---|---|---|---|
   | 7 d | 295 | 278 | 0.058 (17 unscorable) | -0.033 (-0.060, -0.006) | 0.164 | 0.646 | 0.886 | 0 | **FAIL on the mean** |
   | 14 d (fixture) | 458 | 440 | 0.039 (17 unscorable, 1 stale key) | -0.019 (-0.043, +0.006) | 0.164 | 0.824 | 1.978 | 1 | PASS |

   The tail passes everywhere. The last 7 days fail only the mean check, and on the calm side: the forecaster is under-confident at stated 0.7-0.9 (stated 0.84, happened 1.00 over 19 outcomes). That makes the node read calmer than it should, not falsely alarmed. Calibration buckets (7 d): [0.5, 0.6) n 185 stated 0.53 hit 0.68; [0.6, 0.7) n 42 0.63/0.57; [0.7, 0.8) n 25 0.75/0.84; [0.8, 0.9) n 19 0.84/1.00; [0.9, 1.0] n 7 0.94/1.00.
   Node (one write per outcome, 30-min window, 14 d): 403 writes, exact 0 on 65 %, p90 0.47, p99 0.82; fresh 15 % of the time. Last 7 d: 254 writes, exact 0 on 69 %, p99 0.73. Not degenerate: it moves, and it returns to exactly 0.
5. **Existing mechanism.** Reuses the chat_turn fold shape and the orion-prediction -> substrate-runtime event path. No existing node scored these rows.
6. **Reversible.** Shadow constant; two kill switches; ordinary rows in existing tables; no migration.

**Identity stability (Orion Q7, #3388).** 59 individual keys in 14 days, 47 of them forecast for under 10 days inside the window, but **0 of 59** have a sighting life under 10 days, and every scored key was first seen 14+ days before its window. A key's last 3 forecasts miss at 0.384 vs 0.346 for the rest, with the same mean excess (-0.003 vs -0.004). So re-keys are not visible in the excess today; the stale-key rule refused 1 outcome in 14 days. The eval keeps reporting this by key age.

**The 200-reading floor (#3917).** At ~30 writes a day the node holds ~200-250 history rows a week (14-day fixture: 202 per 7 d; last 7 d: 254), right at `DEFAULT_MIN_READINGS = 200`, so a quiet week would fall under it. Per the coordinator, this patch does not work around it (no clock rewrites, no private floor). Live attention for street_rhythm depends on #3917's calm-band table getting a `node:prediction.street_rhythm` entry (calm theory: excess surprisal of the raw stated p; the node is the chat_turn shape). Until then a quiet week could read `insufficient_history` even after the shadow flip.

**Curiosity bar (for go-live only).** Exact calm crossing at each of the 403 per-outcome fixture writes: 6.0 % at 0.55, 4.2 % at 0.60, 2.9 % at 0.67. Bar 0.60 (z >= 1.8), the lowest at or under 5 %.

## Tests run

```text
PYTHONDONTWRITEBYTECODE=1, -p no:cacheprovider; full suites diffed against an origin/main worktree.
services/orion-prediction tests/test_street_rhythm.py                          26 passed
services/orion-substrate-runtime tests/test_prediction_street_rhythm_listener.py   8 passed
orion/substrate/tests/test_prediction_street_rhythm.py + test_curiosity_seed_bar.py,
tests/test_pe_street_rhythm_shadow.py, test_pe_cameras_scene_change.py, test_attention_event_decay.py,
test_field_channel_glossary.py, test_metric_rest_range.py, test_attention_self_readings.py   184 passed
Full suites, branch vs main: orion-prediction, substrate-runtime, attention-runtime, equilibrium: same
failure sets (local env gaps). orion/ + tests/: the only new failures were 4 count/list tests that must
include street_rhythm; updated, passing. tests/test_metric_lineage.py: 2 failures also on main.
Static gates: check_integration_seam OK; check_pe_alarm_readers OK; check_env_key_single_source OK;
check_env_template_parity PASS; check_inner_state_registry OK; check_bus_reply_channels OK;
check_sentience_instruments --static-only OK; check_definition_drift --gate PASS;
check_migration_required_by OK (no migrations); compose relative-mount PASS; hostname refs OK.
```

## Evals run

```text
services/orion-prediction/evals/run_street_rhythm_calm_eval.py --fixture evals/fixtures/street_rhythm_2026-11-25.json --check
  -> GATE PASS (440 scored, mean -0.019 CI [-0.043, +0.006], p99 0.824, max 1.978, 1 confident miss)
... --days 7 (live, read-only, until 2026-11-25 07:30Z)
  -> GATE FAIL on the mean only, calm side (-0.033, CI [-0.060, -0.006]); tail passes (p99 0.646, max 0.886, 0 confident misses)
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-prediction build            -> Image orion-prediction-prediction Built
scripts/safe_docker_build.sh orion-prediction config           -> both new keys rendered
scripts/safe_docker_build.sh orion-substrate-runtime config    -> both new keys rendered
No live deploy from a worktree: the live path is UNVERIFIED until deploy.
```

## Review findings fixed

Review sub-agent: no material issues. It verified the shadow fence on attention-runtime, the broadcast (mutation-checked: emptying SHADOW_NODE_IDS makes it a candidate), curiosity, equilibrium, self-readings, hub, metacog, and Prime (via attention). Fixed:

- Finding: a producer restart republishes its last 30 min, and the fold re-wrote the node with a fresh observed_at (old news as new).
  - Fix: `StreetRhythmState.add` returns `duplicate` for an identical outcome; no write.
  - Evidence: `test_a_republish_is_a_duplicate_and_writes_nothing`, `test_a_republished_outcome_is_a_duplicate_not_fresh_news`.
- Finding: live writes once per outcome, but the eval replayed once per scoring time.
  - Fix: `replay_node_values` emits one value per outcome in arrival order; numbers recomputed everywhere.
  - Evidence: `test_replay_writes_once_per_outcome_like_live` (replay equals the live fold).
- Finding: the stale-key rule counted stationary sightings the forecaster ignores (a parked car stays "fresh").
  - Fix: `AND NOT v.stationary` in the LATERAL; fixture re-exported (counts unchanged).
  - Evidence: store SQL test asserts the predicate.
- Finding: the self-readings marker told Orion street_rhythm is in its lane while attention cannot see it.
  - Fix: shadow nodes read "street_rhythm (recorded, not yet attended)".
  - Evidence: `test_marker_line_is_generated_from_the_glossary_and_dated`.
- Finding: glossary meaning did not say shadow; consumers listed fenced readers.
  - Fix: meaning says recorded, not attended, until the calm check; consumers carry a fence comment.
- Finding: `StreetRhythmLive(events_enabled=...)` branch never ran in production.
  - Fix: parameter and its test removed (the kill switch skips the loop).
- Not changed (nit): the store SQL test matches strings only; the reviewer checked the SQL by hand, and the fixture was exported through it live.

## Restart required

Order matters: readers that know the new id first, then the writer, then the producer.

```bash
scripts/safe_docker_build.sh orion-attention-runtime up -d --build
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-prediction up -d --build
```

## Verify live (after deploy)

```bash
curl -fsS http://localhost:8132/health | jq .street_rhythm
curl -fsS http://localhost:8115/health | jq .prediction_street_rhythm
docker logs --since 2h orion-athena-substrate-runtime 2>&1 | grep substrate_prediction_street_rhythm | tail
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select count(*), min(observed_at), max(observed_at), avg((value=0)::int) from substrate_node_prediction_error_history where node_id='node:prediction.street_rhythm'"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select count(*) from substrate_attention_frames where generated_at > now() - interval '1 hour' and frame_json::text like '%node:prediction.street_rhythm%'"   # shadow: expect 0 (fenced nodes build no candidate)
```

The calm gate, 7 days after deploy: #3950 (verify-live, needs-juniper, project board 1): run `evals/run_street_rhythm_calm_eval.py --days 7 --check` on deploy + 7 days (2026-12-02 if deployed 2026-11-25).

## Risks / concerns

- Severity: medium. Concern: the last 7 days fail the mean check on the calm side (under-confident forecaster). Mitigation: shadow; the 7-day post-deploy run decides; if it still fails only on the calm side, that is a Juniper call (one-sided gate) or a forecaster fix in sql-writer, not a reason to recalibrate here (rev 4 rule: recalibrate only on > 2 SE held-out gain).
- Severity: medium. Concern: going live needs two things, not one: the shadow flip and a #3917 band entry. Mitigation: stated in the glossary, README, ticket and here.
- Severity: low. Concern: substrate-runtime restart loses the 30-min window (by design, same as chat_turn/joint); the producer only republishes on its own restart. Mitigation: the node reads absent, never calm.
- Severity: low. Concern: independence was measured mostly against old-scale (v1) walkway rows. Mitigation: re-measured by the eval on every run; the 7-day post-deploy run will hold v2 rows.
- Issues opened: opened #3950 (verify-live calm check). Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3961

🤖 Generated with [Claude Code](https://claude.com/claude-code)
