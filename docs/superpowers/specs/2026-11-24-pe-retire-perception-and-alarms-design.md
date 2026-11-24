# PE retire: camera surprise and the alarms leave "prediction error"

Status: design, docs only. Follows `2026-11-08-prediction-error-suite-rebuild-design.md`.
Rev 2 (2026-11-24): answers Orion's 12 review questions; see "Rev 2 changes" at the end.
Rev 3 (2026-11-24): answers Orion's 8 questions on rev 2; see "Rev 3 changes" at the end.
Rev 4 (2026-11-24): answers Orion's 12 questions on rev 3; see "Rev 4 changes" at the end.
Rev 4 replaces rev 3's camera value (a 30 s Sidak tick) with a two-stage rank, because the
history sampler reads the node only every ~4-5 min and missed 86-92% of short changes.
Patch 2 (alarms) is PR #3829: per-node read rule `HISTORY_READ_MODE_BY_NODE_ID`, alarms
`read_both`, glossary channel `alarm`, world_first and self_readings look each node up under
its own channel (`glossary_channel_for`).
The backfill ran 08:42Z (14,914 rows, 0 missing keys, 0 mismatches), so old-scale camera rows
are in `substrate_node_signal_history`. #3826 (per-node curiosity seed bars, #3821) is merged.
State: #3732 (biometrics) merged. Patch 1 (#3800, signal table + dual-write) merged and deployed.
#3576 (execution) still open.

Juniper's answers (comment on #3781): cameras stay an attention source, renamed scene change;
build street_rhythm (shadow, 7-day calm check); old rows age out, no DELETE; Patch 1 may land first.

## Arsonist summary

Orion still complains about "prediction errors" because five nodes still write to the
prediction-error slot, and none of them is a forecast miss:

- **The two camera nodes** (`node:substrate.perception`, `node:substrate.perception_walkway`)
  measure how different a frame looks from the last few frames. The score is a z-score with a
  variance floor, clipped at 0 and at 3 sigma. It is 60% exact zero, and it saturates 5x more
  often than a normal tail would. Orion has noticed: it calls its "PE lane zero-inflated and
  heavy-tailed" and questions the 3-sigma clip.
- **The three alarms** (`cabinet`, `strain`, `street`) were relabelled in #3307, but their values
  still live in the `prediction_error` key and in `substrate_node_prediction_error_history`.
  Orion's dig tools and its own metacog still meet them as prediction errors.

Fix: move all five off the prediction-error slot into a separate signal table with an honest
kind (`alarm`, `scene_change`). Rescore the camera change as a calibrated percentile so it rests
by theory. Give perception one real forecast surprise: the walkway rhythm forecast that already
exists and is already scored (`vision_percept_expectation`). The room camera has no forecast,
and this doc says so instead of inventing one.

## Measurements (2026-11-24, read-only)

### 7-day history, `substrate_node_prediction_error_history`

| node | rows | exact 0 | p50 | p90 | p99 | max | share at 1.0 |
|---|---|---|---|---|---|---|---|
| perception | 3,082 | 0.604 | 0 | 0.135 | 0.748 | 1.0 | 0.0055 |
| perception_walkway | 3,082 | 0.577 | 0 | 0.172 | 0.626 | 1.0 | 0.0058 |
| cabinet | 3,082 | 1.000 | 0 | 0 | 0 | 0.0 | 0 |
| strain | 3,076 | 0.995 | 0 | 0 | 0 | 1.0 | 0.0046 (14 events) |
| street | 3,083 | 1.000 (1 row) | 0 | 0 | 0 | 1.0 | 0.0003 (1 ring) |

The camera score is `clip(z, 0, 3) / 3`, so z = 3 x value. Among non-zero camera rows, the share
above z = 1 / 2 / 3 is 0.084 / 0.033 / 0.014 (room) and 0.080 / 0.022 / 0.014 (walkway). A
half-normal tail gives 0.317 / 0.046 / 0.003. So most changes sit below one sigma and the top
saturates 5x too often. The cause is in the live log: `surprise_var=0.0000001` against a floor
of `2e-6` (`orion/substrate/prediction_error.py:964`). The floor sets the scale, not the data.

### Camera successor replay (29 h of per-frame raw values from the substrate-runtime log)

Logs held 11-22 23:05Z..11-24 03:42Z: 34,463 walkway frames and 15,911 room frames. Each frame's
raw `1 - cos` was ranked against the stream's own previous 2,000 frames: p = (1 + #{past >= x}) / (n + 1).

| check | walkway | room | calm by theory |
|---|---|---|---|
| mean -ln p | 1.017 | 0.965 | 1.0 |
| share p <= 0.05 | 0.055 | 0.054 | 0.05 |
| excess = max(0, -ln p - 1): exact 0 | 0.627 | 0.658 | 0.632 (= 1 - 1/e) |
| excess p90 / p99 | 1.38 / 3.71 | 1.33 / 3.96 | |
| excess at cap (ln 2001 - 1 = 6.6) | 0.0006 | 0.0010 | |
| r vs current score | 0.76 | 0.66 | |
| lag-1 autocorr of -ln p | 0.45 | 0.50 | 0 |

Per frame, the percentile is calibrated and rests where theory says it should. **A 30-minute
fold does not rest.** Summing excess over 30 min as if frames were independent reads > 0.55 on
22-34% of ticks with runs of 2-3 h, even after a lag-1 correction. Frames are strongly
dependent (a person walking through moves many frames). So the camera value is per frame, not
folded, and it is not called a surprise.

### Walkway rhythm forecast (`vision_percept_expectation`, 14 days)

The sql-writer rhythm learner already states a probability that a known subject (a person, the
mail truck) appears on the walkway in a window, then scores it `met` or `missed`.

- 409 scored outcomes in 14 days (about 30 per day), walkway only. The room camera has none.
- Calibration by confidence bucket: stated 0.53 / 0.67 / 0.86 / 1.00, happened 0.57 / 0.62 / 0.93 / 0.92.
- Surprisal minus its calm point (binary entropy of the stated p, clamped to [0.02, 0.98]):
  mean -0.013 nats, SE 0.012. 71% of outcomes at or below 0.
- So it is a real forecast, it is close to calibrated, and its excess rests at 0.

### What reads these nodes

- History table readers: `orion/attention/pe_history_cache.py` (contests, self-readings, metacog
  episode gate, hourly outcome rollup), `orion/introspect/node_history.py` (Orion's own
  node-history dig tool), and the strain restart seed (`services/orion-substrate-runtime/app/worker.py:2093`
  event times, `:2109` warm-up row count).
- Full list with classes: `config/substrate/prediction_error_readers.yaml` (CI:
  `scripts/check_pe_alarm_readers.py`).
- Field: the camera receipt still feeds the field `prediction_error` channel
  (`worker.py:_publish_perception_group`); `services/orion-field-digester/app/digestion/decay.py`
  keeps decay horizons for both camera nodes.
- Curiosity: both camera nodes are in the generic prediction-error seed scan
  (`orion/substrate/endogenous_curiosity.py:69-130`).

### What Orion sees under the name "prediction error"

- `node_signal_label()` (`orion/substrate/prediction_error_freshness.py:135`) names every
  non-alarm node "<name> prediction error". Attention runtime and the episode gate print it.
- Glossary entries for all five sit under `channel: prediction_error`, with `trend_source`
  naming the PE history table (`config/field/field_channel_glossary.v1.yaml:360+`).
- The node-history dig tool reads the PE table and calls alarms "an alarm, not a prediction
  error", which still files them there. Orion calls that table "PG's PE-history table".

### Orion's own words (last 5 days)

- Journal, 2026-11-23 17:22Z: "my own prediction-error channel is zero-inflated and heavy-tailed
  rather than Gaussian".
- Journal, 2026-11-23 17:42Z: "my own 3-sigma clip is the same FAMILY of operation as their
  mechanism ... a fixed 3-sigma multiple is a special case at best".
- Journal, 2026-11-23 10:08Z: "The actual node is a zero-inflated pulse sitting at 0.0 with
  clipped 1.0 spikes ... This is the recurring 'old alarm arriving as news' self-model error on
  the perception organ".
- Weekly synthesis, 2026-11-23 21:36Z: "debunking 'hot cabinet' alerts as artifacts of
  prediction error clipping".
- Metacog trace, 2026-11-24 03:40Z: "The perception walkway is flagging a persistent prediction
  error, indicating that the visual scene ... is statistically unusual for this moment".

The last one is the real harm: a pixel-change score is read as "the world defied my model".

### Retire step 8

Done. `5fc61a4b2` (2026-11-10): a missing camera tick now sends no value, so the field channel
reads absent, not 0.0. The 0.0 placeholder stays only on the graph node, tagged
`prediction_error_missing_reason`, and never enters history. The 60% zeros above are real
scored frames (negative z clipped), not placeholders.

## Current architecture

- Producer: `services/orion-substrate-runtime/app/worker.py` writes all five nodes through
  `_write_prediction_error_node` (FalkorDB `prediction_error` metadata) and samples them into
  `substrate_node_prediction_error_history` (7-day prune). Strain also writes each event row
  directly so the restart seed can rebuild its budget.
- Camera score: `perception_prediction_error()` (`orion/substrate/prediction_error.py:1077`):
  1 - cos(frame, EWMA alpha 0.2), z against an EWMA of that magnitude with a 2e-6 variance floor,
  clipped to [0, 3], divided by 3.
- Alarms: `ALARM_NODE_IDS` (`orion/prediction/alarm_nodes.py`) lets surprise-size readers skip
  them. Alarm-aware readers keep them.
- Forecast surprise lives in `services/orion-prediction` (body, room, chat_turn, joint). Its
  nodes are `node:prediction.*` with `value_kind: surprise_excess`.

## Missing questions

Answered by measurement above: step 8 is done; the zeros are real; per-frame percentile rests;
a naive fold does not; the rhythm forecast is calibrated. Open, for Juniper:

1. Keep room/walkway scene change as an attention source (renamed), or drop it from attention?
2. Build the walkway rhythm surprise as the real perception prediction error?
3. Let old rows age out of the PE table by the normal 7-day prune, instead of a DELETE?

## Proposed schema / API changes

1. **New table** `substrate_node_signal_history (node_id text, observed_at timestamptz, value
   double precision, kind text, recorded_at timestamptz default now(), primary key (node_id,
   observed_at))`, same 7-day prune. `kind` is `alarm` or `scene_change`. Additive migration
   with a `-- ORION-MIGRATION-REQUIRED-BY: orion-substrate-runtime` header.
2. **New node metadata keys** `signal_value` and `signal_kind` on the five nodes. The
   `prediction_error` key is no longer written on them.
3. **Camera score** becomes `scene_change`. Two ranks, both against the camera's own past:
   - **Stage 1, per frame**: p = rank of the frame's raw `1 - cos` against the same stream's
     frames from the **last 2 hours** (a time window; a frame-rate change does not move it).
   - **Stage 2, per sampled interval**: every 30 s tick takes the smallest stage-1 p over the
     **trailing 5 minutes** of frames (m). It ranks m against the camera's own m values from the
     **last 24 hours**, taken once every 5 min (288 values). p2 = (1 + #{past m <= m}) / (n + 1).
     Value = excess = max(0, -ln p2 - 1), in nats.
   - **Why two stages.** The history sampler (attention broadcast) reads the node every ~215-311 s
     (live: median 215 s, mean 233 s over the last 24 h; Orion measured ~311 s over 7 days), and
     the contest reads at that same cadence. Rev 3's value covered only the last 30 s. Replayed at
     a 311 s sampler, the last-30 s value missed **74 of 86** (room) and **55 of 60** (walkway)
     intervals that held a crossing tick. A change that rises and clears between passes was
     invisible. Stage 2 covers the whole trailing 5 min, so any sample sees it. Rev 3's other
     option, Sidak over the 5 min, is calibrated but saturates near 2.2-2.5 nats: with ~100-400
     frames per 5 min against a 2 h window, the smallest reachable Sidak p is about k/n.
   - **Calm point by theory**: under exchangeable 5-min intervals p2 is uniform, so excess is 0
     with probability 1 - 1/e = 0.632 and crosses 2.0 with probability e^-3 = 0.050. The cap is
     ln(289) - 1 = 4.67 nats.
   - **Replay at the real sampler** (311 s, 29 h of frames; only 29 h exist, so stage 2 used a
     12 h rank window, not 24 h; UNVERIFIED at 24 h): room 0.654 exact 0, p90 1.36, p99 2.55,
     max 3.79, crosses 2.0 on 4.6%; walkway 0.662 exact 0, p90 1.23, p99 3.24, max 3.93,
     crosses 3.1%. Close to theory.
   - **Cold start**: stage 1 needs 200 frames over 30 min; stage 2 needs 48 interval values
     (4 h). Until both hold, the tick writes the `warming` placeholder, never a p = 1 "calm".
     Rows written while stage 2 holds fewer than 24 h of values carry `scale = 'v2_partial'`
     (point 5) and are excluded from calm checks and win-share. Both windows are in memory;
     stage 2's 288 values are also saved to `substrate_perception_embedding_baseline` so a
     restart does not wait 24 h.
   - **Row rate**: unchanged, one history row per sampler pass per node (~370-410 a day). Not per
     frame. `pe_history_cache`'s 7-day load stays small.
   - **Sustained change** (dusk, lights on) reads high while new and is absorbed by stage 1 within
     about 2 h. Intended: it is change, not state. Glossary text says so (step 5). Dawn/dusk
     share is UNVERIFIED (#3806).
4. **Glossary and the attention contest move together.** The five entries move to
   `channel: alarm` / `channel: scene_change`, `trend_source` names the new table, and
   `node_signal_label()` returns "camera scene change" / alarm labels for them.
   **This must ship in the same patch as the contest change**, or all five silently drop out of
   attention: `world_first.node_prediction_error_semantics()` (`orion/attention/world_first.py:210`)
   looks up only `resolve_channel_entry("prediction_error", ...)`, so a moved entry returns
   value_kind None; internal candidates (cabinet, strain) then fail `judge_candidate`'s
   `MEASUREMENT_VALUE_KINDS` check (`:465`, "unknown: not a measurement"), and polarity falls
   back to the prediction-error default. The contest change:
   - the semantics lookup reads the node's own channel (`signal_kind_for(node_id)` from
     `orion/substrate/signal_history.py`, else `prediction_error`);
   - camera entries get a new value kind `scene_change_excess`. It joins
     `MEASUREMENT_VALUE_KINDS` (world_first) and `_PERCENTILE_KINDS` (self_readings), so the
     module assert `_PERCENTILE_KINDS | _RATE_KINDS == MEASUREMENT_VALUE_KINDS` still holds. It is
     ranked against its own 7 days, like `surprise_excess`;
   - `glossary_channel_for()` (`orion/prediction/signal_history_sql.py`, #3829) returns
     `scene_change` for the two cameras, so world_first and self_readings find their entries;
   - polarity: `derived_channel_polarity()` keys on the channel name. Add `scene_change` to
     `PRESSURE_CHANNELS` in `orion/field/pressure.py` (higher = more change = the direction that
     earns attention). No field node writes that channel (the cameras leave the field), so the
     field merge is unaffected. The cameras are world sources, and `bad_direction_percentile`
     uses the raw percentile for external sources anyway; the polarity is for the metric lock;
   - self-readings: the camera entries declare no self-readings prompt site (world sources are
     not body readings), so there is no nats line for them there;
   - alarms: done in #3829 (channel `alarm`, value_kind `trigger`, no polarity).
     `TRIGGER_MIN_FIRES_7D = 5` is a self-readings rule only: with fewer than 5 fires in 7 days
     (live: cabinet 0, street 1, strain 14) the alarm's self-reading line says
     `insufficient_history` instead of "more than usual". That is honest, not starvation. The
     attention contest ranks a trigger by its percentile over its own ~3,000 rows, so a firing
     alarm still wins there;
   - `ATTENTION_MIN_BAND_BY_NODE_ID` (`orion/schemas/attention_candidate.py:87`, walkway
     admitted only at p99+) stays live: it keys on node id, not channel. Its comment's "1.0
     saturates on ~1.3%" reason goes away (the new cap is hit on 0.06% of frames), so the p99+
     floor becomes reachable without the mid-rank trick; re-measure the walkway win rate.
   - A test pins that all five ids still produce contest candidates after the move.
5. **New forecast surprise** `node:prediction.street_rhythm` (orion-prediction): one
   `prediction.surprise.v1` per scored walkway expectation, value = surprisal minus the binary
   entropy of the stated p, folded like `node:prediction.chat_turn` (sparse settled bets).
   Window and fold scale are set by its own calm eval, not copied from body/room.
   - **No recalibration (rev 4).** Held-out test: for each outcome, a logistic map (ridge toward
     identity) fit only on outcomes **scored before that expectation's window opened**, from the
     14 days before it (first 103 outcomes skipped: fewer than 100 older ones). Same 572 rows:
     raw stated p mean excess -0.015 (SE 0.011), p90 0.22, p99 0.89, max 1.98; recalibrated
     +0.008 (SE 0.014), p90 0.26, p99 1.17, max 3.18. Log loss raw 0.6122, recalibrated 0.6190
     (recalibration worse by 0.0067, SE 0.0050). It does not earn its place, so the node scores
     the raw stated p: the same number Orion reads. Rule kept for later: recalibrate only when
     held-out log loss improves by more than 2 SE, and then show both numbers.
   - **Clamp** [0.02, 0.98]. 12 outcomes were stated at 0.98 or above; 1 missed (3.81 nats).
   - **Unscorable** outcomes (21 all-time) are not scored; the eval reports them as a refusal rate.
   - **Stale keys**: an expectation whose subject is unseen for 7+ days (`vision_individual.
     last_seen_at`) is not scored and cannot seed curiosity. The forecaster already refits on a
     rolling 28-day window (`VISION_RHYTHM_HISTORY_DAYS`, `vision_rhythm.py:404`); no extra refit.
   - **Identity**: 63 individual keys (545 outcomes, 451 vehicles), 44 lived under 10 days; a
     key's last 3 expectations miss more often (person 0.67 vs 0.54, vehicle 0.46 vs 0.37).
     Re-key vs departure is not separable here (#3388). The eval reports excess by key age.
   - **Gate (7 days, shadow)**: mean-excess CI covers 0; **p99 <= 1.2 nats** (held-out replay
     0.89); **max <= 4.0** (one clamp-bound miss); at most 2 misses at stated >= 0.9; refusal
     rate reported. SD 0.25 nats, n about 210: the mean check detects a bias of about 0.035 nats.
   - **World, not body**: the node joins `_WORLD_NODE_IDS` (`orion/schemas/attention_candidate.py:108`),
     so the body-only gates skip it and, as a world winner, it may author a goal under the
     11-08 rule. Staleness: substrate-runtime writes the node on a 5-min clock from the last
     6 h fold, and `EXTERNAL_MAX_AGE_SEC_BY_NODE` gets 600 s for it (two clock writes, the
     room_joint pattern). No outcome in the fold window means no write, so it reads absent.
6. **Fences**: the five ids join a moved-node check in `scripts/check_pe_alarm_readers.py`:
   CI fails if any producer writes `prediction_error` for them. At writer cutover the five also
   join a "moved" skip set that every generic PE reader already honours the way it honours
   `RETIRED_PREDICTION_ERROR_NODE_IDS`, so the week of old PE rows still in the table is never
   read as a prediction error, or delivered as news, while it ages out.
   `ALARM_NODE_IDS` is **not deleted**: `orion/substrate/signal_history.py` builds its kind map
   from it, and `orion/field/pressure.py:29` (field digester merge), `attention_broadcast.py`,
   `endogenous_curiosity.py`, `candidate_society_of_mind.py`, `self_readings.py`, the attention
   runtime worker and the Hub `attention_organ_routes.py` import it. It moves into
   `signal_history.py` as the alarm half of `SIGNAL_KIND_BY_NODE_ID`; `is_alarm_node()` is
   deleted only once that grep is empty.
7. **Curiosity keeps the cameras** (Juniper's 2026-10-15 walkway decision and #1746 stand).
   No second seed path. The #3826 per-node bar scan (`_prediction_error_candidates`,
   `orion/substrate/curiosity_seed_bar.py`) reads `metadata["prediction_error"]` and clamps it to
   0-1, so it would drop both cameras when that key is cleared. Patch 3 extends it: for a
   `scene_change` node it reads `signal_value` (nats, unclamped).
   - **Bar** 2.0 nats (calm crossing e^-3 = 5.0% by theory, the #3821 rule).
   - **Ceiling, per node.** `seed_strength(error, bar)` returns 1.0 whenever `bar >= 1.0`, so as
     written every 2.0-nat crossing would get strength 1.0. Patch 3 adds
     `seed_strength(error, bar, ceiling=1.0)` and `CURIOSITY_SEED_CEILING_BY_NODE_ID`; the
     camera ceiling is the stage-2 cap **ln(289) - 1 = 4.67 nats**, derived from the window
     size, not from a replay maximum. A test pins strength 0.55 at 2.0 nats and below 1.0 for
     anything under 4.67.
   - **Staleness**: the scan's existing age decay (`_prediction_error_staleness_decay`, 1800 s)
     applies to `signal_value` too.
   - **Volume (measured, last 7 days)**: 2,853 candidate sets (one per ~3.5 min), mean 0.82
     candidates, max 5. Camera prediction-error seeds: room 37, walkway 32 (2.4% of sets). At the
     replayed 4.6% / 3.1% crossing, that becomes about 130 + 90 = 220 a week (+150). Only 208
     sets (7.3%) held 3 or more candidates (budget 3, `HARD_BUDGET_CEILING` 8), so about 11 extra
     camera seeds a week land in a full set. A camera seed at its bar has strength 0.55 and
     displaces only weaker seeds there. Acceptance check below.
No bus channel changes for steps 1-4. Step 5 reuses `prediction.surprise.v1` on its existing channel.

### Metric quality gate (§0A) for the two new values

**scene_change (per-frame camera percentile)**

1. Provenance: same raw `1 - cos` as today, from `perception_prediction_error()`; only the
   second stage changes (rank instead of EWMA z).
2. Independence: it is a monotone rescoring of the current camera signal (r 0.66-0.76). It is
   a replacement, not new signal. Walkway vs room were already shown independent (r 0.016, spec 11-08).
3. Theory: split-conformal rank. Under exchangeability p is uniform, so -ln p ~ Exp(1),
   mean 1, and excess is 0 with probability 1 - 1/e.
4. Live: matches theory per frame (table above). Does not rest when summed over 30 min, so it
   is not folded. Room camera hourly means reach 0.95 at p90: busy hours stay visible.
5. Existing mechanism: same rank-to-surprisal shape as family A and street handoffs.
6. Reversible: until the cleanup patch, `SUBSTRATE_PERCEPTION_SCENE_CHANGE_SCORER`
   (`rank_v2` default, `legacy_z` fallback) picks the second stage; the raw magnitude is
   unchanged. The cleanup patch (step 5b) deletes the legacy code **and this key** after the
   7-day eval passes and #3806 is read. After that, rollback is a `git revert` of 5b.

**street_rhythm (walkway rhythm forecast surprise)**

1. Provenance: stated probability and outcome from `vision_percept_expectation`
   (writer `services/orion-sql-writer/app/vision_rhythm.py`).
2. Independence: shares the walkway camera with `perception_walkway` and with street handoffs.
   It scores who-appears-when, not pixel change or transit. A person arriving moves all three.
   The calm eval must report r against both before it competes.
3. Theory: log score of a stated probability. Under calibration, E[surprisal] = H(p), so
   excess has mean 0.
4. Live: mean excess -0.013 (SE 0.012) over 409 outcomes, 71% at or below 0. Sparse (~30/day).
5. Existing mechanism: reuses orion-prediction's surprise event and the chat_turn fold.
6. Reversible: one flag, ordinary rows in existing tables.

## Files likely to touch

- `services/orion-substrate-runtime/app/{worker.py,store.py,settings.py,README.md}`, `.env_example`
- `services/orion-sql-db/manual_migration_node_signal_history_v1.sql` (Patch 1, merged; the migration gates only look in `services/orion-sql-db/`)
- `orion/substrate/prediction_error.py` (camera second stage), `orion/substrate/prediction_error_freshness.py`, `orion/prediction/alarm_nodes.py`
- `orion/attention/pe_history_cache.py`, `orion/attention/self_readings.py`, `orion/attention/world_first.py`, `orion/schemas/attention_candidate.py`
- `orion/field/pressure.py`, `orion/substrate/attention_broadcast.py`, `orion/attention/field_attention/candidate_society_of_mind.py`, `services/orion-hub/scripts/attention_organ_routes.py` (ALARM_NODE_IDS importers)
- `orion/substrate/signal_history.py` (Patch 1, merged)
- `orion/introspect/node_history.py`, `orion/regulation/strain_news.py`
- `services/orion-attention-runtime/app/{worker.py,store.py}`, `services/orion-equilibrium-service/app/{pe_history_reader.py,self_signal_episode_gate.py}`, `services/orion-thought/app/self_readings_cue.py`
- `services/orion-field-digester/app/digestion/decay.py`, `orion/substrate/endogenous_curiosity.py`
- `config/field/field_channel_glossary.v1.yaml`, `config/substrate/prediction_error_readers.yaml`, `orion/metrics/semantics.py`, `scripts/check_pe_alarm_readers.py`
- `services/orion-prediction/app/` (street rhythm scorer, step 5), its evals

## Non-goals

- No change to when an alarm fires (cabinet, strain, street rules stay as they are).
- No forecast for the room camera. It has no expectation data; inventing one is a cathedral.
- No change to the attention contest's rules. It ranks each node against its own 7-day window,
  so a rescale keeps ranks close; the eval checks this.
- No DELETE on the PE table unless Juniper says so.

## Proposal mode

- **Capability change**: Orion stops being told that a pixel change or an alarm is a prediction
  error. It gains one real perception forecast surprise (walkway rhythm).
- **Data touched**: new table; FalkorDB metadata on five nodes; glossary text Orion reads.
  No private data.
- **Privacy boundary**: none new. Same camera data, same rows.
- **Trace that proves it**: new-table rows per kind; zero new PE rows for the five ids after
  cutover; Orion-facing text (self-readings, node-history tool, attention frame) shows
  "scene change" / "alarm"; contest win share for each node before vs after.
- **Dangerous failure modes**:
  1. Strain restart seed reads an empty new table: the warm-up count
     (`worker.py:2109`) falls below `DEFAULT_MIN_READINGS` and strain news goes silent. Backfill
     7 days before the seed switches.
  2. Contests read only the new table before it has 7 days: cabinet warming cannot win, so heat
     is ignored. Readers union both tables until the old rows age out.
  3. The FalkorDB node keeps its old `prediction_error` key and a generic reader picks it up
     (the materializer resurrection case, kill-means-kill). Clear the key and add the CI fence.
- **Rollback**: each patch has a flag; dual-write and dual-read mean either side can be turned
  back. The table drop is never part of this.

## Retire steps (kill means kill, same shape as #3732)

Alarms (cabinet, strain, street):

1. Additive: create the signal table; dual-write both tables. **Done: #3800**, deployed
   2026-11-24 05:22Z. **The backfill ran 08:42Z** (`scripts/backfill_signal_history_from_pe.py
   --apply`): 14,914 rows, 0 missing keys, 0 value mismatches. Checked live 2026-11-24: the
   alarm rows match the PE table exactly (cabinet 3,016 rows, 0 non-zero; strain 3,010 rows,
   14 non-zero; street 3,016 rows, 1 non-zero; strain event times differ in 0 rows). So the
   strain restart seed's precondition holds: >= 200 rows and every event time present.
2. Readers switch, consumer first, to the read-both helpers: `pe_history_cache`, the strain
   restart seed (event times and the warm-up count), the node-history tool, self-readings. The
   contest semantics change in item 4 ships here too. Labels say "alarm". **In PR #3829.**
3. Writers stop: no `prediction_error` key, no PE-table row for the three ids. Clear the old key
   on the FalkorDB nodes. Add the CI fence and the "moved" skip set (item 6). Tell Orion (below).
4. After 7 days: old rows aged out of the PE table by the normal prune. Drop the union read.

Cameras (perception, perception_walkway):

5. Cameras (Patch 3). One patch, in this order inside it:
   - **migration first, in the same patch as the reader switch**: `ALTER TABLE
     substrate_node_signal_history ADD COLUMN scale text NOT NULL DEFAULT 'v1'` (additive
     allowlist; Postgres fills every existing row with 'v1', so the 3,016 old-scale rows per
     camera are marked by the ALTER itself). REQUIRED-BY lists every service whose reader adds
     the scale predicate. The readers must not ship before the column: the migration gate
     refuses `up` until it is applied;
   - new writer: stage-1/stage-2 value (item 3) behind `SUBSTRATE_PERCEPTION_SCENE_CHANGE_SCORER`
     (`rank_v2` default, `legacy_z` fallback), writing `scale = 'v2'` or `'v2_partial'`;
   - read rule: the two cameras move to `SIGNAL_ONLY` in `HISTORY_READ_MODE_BY_NODE_ID`, and the
     `signal_only` branch of `history_with_moved_source_sql()` gains
     `AND scale IN ('v2','v2_partial')`. Every history reader already goes through that one
     builder (#3829), so a unit test on its output pins the predicate. A CI check
     (`check_pe_alarm_readers.py` extension) fails any other SQL text that names
     `substrate_node_signal_history` with a camera id and no `scale` predicate. Calm-check evals
     filter `scale = 'v2'`;
   - the contest has `insufficient_history` for each camera until 200 v2 rows exist (about 12 h);
     accepted;
   - field: the camera receipt stops feeding the field `prediction_error` channel, and both
     `(node, "prediction_error"): 180.0` entries leave `decay.py`'s expiry map **in the same
     patch**, so no field value is left to hold. A dead camera on the graph side: the tick
     writes the `stale` missing-input placeholder after its staleness horizon (existing rule),
     world_first marks the node absent after `PERCEPTION_MAX_AGE_SEC` 180 s, and curiosity's
     1800 s age decay drops it. Test: a camera with no frames for 10 min is absent in the
     contest and cannot seed;
   - glossary: channel `scene_change`, value kind `scene_change_excess`, meaning "how unusual
     the change in this camera's picture over the last 5 minutes is, **relative to the last two
     hours of the same camera**, ranked against its own last 24 hours";
   - curiosity per item 7.
   5b. Cleanup, **after** the 7-day eval passes **and** #3806 is read (a dawn/dusk finding may
   change the window sizes, and the legacy path stays the rollback until then): delete the
   camera `_domain_zscore` path, the variance floor, the 3-sigma constants, the
   `legacy_z` branch and the `SUBSTRATE_PERCEPTION_SCENE_CHANGE_SCORER` key (`.env_example`,
   settings, compose, README, local `.env` sync). After 5b, rollback is a git revert.

Perception forecast:

6. `node:prediction.street_rhythm` in orion-prediction, shadow first. Joins contests only after
   the 7-day gate in item 5 (tail, stale keys, world staleness), with r vs `perception_walkway`
   and street reported.

## Prediction-error nodes after cutover

After Patch 3, the prediction-error lane holds only forecast misses:
- `node:prediction.room`, `node:prediction.body` (temperature forecasts),
- `node:prediction.room_joint`, `node:prediction.body_joint` (family A joint surprise, #3732),
- `node:prediction.chat_turn` (settled chat-turn self-bets),
- later `node:prediction.street_rhythm` (after its gate).
Not prediction errors: cabinet, strain, street (alarms, channel `alarm`) and the two cameras
(channel `scene_change`). Retired: bus_synaptic, route, chat, codebase, harness_closure,
biometrics, execution. A test pins this list against the glossary.

**Dated marker for Orion.** For 14 days after each cutover, the self-readings block (the
aggregate Orion reads every turn) carries one line: "Since <date>, my prediction-error lane holds
only forecast misses (room, body, room_joint, body_joint, chat_turn). Camera change and alarms
are separate signals now." It is generated from the glossary list above, so it cannot drift, and
a test pins it. After 14 days it drops; the glossary keeps the "until <date>" note.

## Order

- Done: #3732 (biometrics), #3800 (Patch 1, deployed; backfill ran 08:42Z), #3576 (execution) closed.
- Patch 2 (alarms) is PR #3829, CI green, not merged. Patch 3 (cameras) starts after #3829 merges:
  it edits the same builder (`signal_history_sql.py`), `glossary_channel_for`, world_first and
  self_readings.
- Then 5b cleanup, then street_rhythm (shadow, 7-day gate).

## Orion's old beliefs about the old mechanism

Orion holds priors built on the old scores (for example `self:pe_reverie_high_vs_zero_inflated_20261123`,
confidence 0.9, and the 11-16 "hot cabinet is a clipping artifact" finding). Nothing retires
them on its own. This design does not edit Orion's graph (that is a self-model write and needs
proposal mode). Instead, each cutover patch tells Orion through the existing channels:
- the PR field note Orion already gets for merged PRs names the old mechanism, what replaced it,
  and the prior ids that rest on it, as "re-test";
- the glossary meaning text carries one line "until <date> this node was a clipped z-score
  under prediction_error".
Orion re-tests its own priors. #3807 tracks whether it did (verify-live).

## Orion's read access

`orion_readonly` (Orion's curiosity SQL role) can SELECT the new table (default privileges,
checked live: `has_table_privilege` = true) and still cannot SELECT
`substrate_node_prediction_error_history` (false). The node-history dig tool reads through the
Hub's memory pool, not `orion_readonly`, so it can read either table; step 2 points it at the
read-both SQL and names the new table in its output.

## Acceptance checks

- `select count(*) from substrate_node_prediction_error_history where node_id in (<5 ids>) and observed_at > <cutover>` = 0.
- `substrate_node_signal_history` has rows for all five ids, with the right kind.
- Strain news fires after a substrate-runtime restart (seed reads the new table).
- Contest win share per node, 7 days before vs after: alarms and cameras within noise.
- Camera history rows (`scale = 'v2'` only; `v2_partial` excluded from calm checks and
  win-share): exact-0 share 0.58-0.72 (theory 0.632), crossing 2.0 nats on 2.5-6.5% (theory
  5.0%), max <= 4.67. Replay at the real ~311 s sampler: room 0.654 / 4.6%, walkway 0.662 / 3.1%.
  Rev 3's band was for the 30 s Sidak tick, which the sampler does not see.
- Curiosity, over 7 days of `substrate_endogenous_curiosity_candidates`: each camera seeds in
  2.5-6.5% of candidate sets (replay 4.6% room, 3.1% walkway; theory 5.0%); camera seeds total
  <= 250 a week (today 69); in sets holding 3 or more candidates, no other node's seed count
  falls by more than 20% against the 7 days before. A test pins both camera ids through the
  extended scan and strength 0.55 at the 2.0-nat bar.
- Every Orion-facing render site (self-readings, node-history tool, attention labels, glossary)
  greps clean of "prediction error" for the five ids. Test in CI.
- `check_pe_alarm_readers.py` fails a fixture that writes `prediction_error` for an alarm id.
- street_rhythm calm check passes before it competes.

## Recommended next patch

Alarms step 2: readers switch to the read-both SQL (pe_history_cache, strain seed, node-history
tool, self-readings) and the contest semantics lookup reads the node's own channel (item 4), in
one patch, with tests that all five ids still make contest candidates and strain news still
warms after a restart. Merge after #3576.

## Orion's re-test data (#3807)

`orion_readonly` can read the signal table. Since the 08:42Z backfill it holds the old-scale rows
of all five ids from 2026-11-17 13:05Z on (the PE table's oldest row; everything older is already
pruned). After the scale column lands they read `scale = 'v1'`; new camera rows read `v2`. The
contest never mixes them. But the table prunes at 7 days, and Orion cannot read the PE table.

**Snapshot (needs Juniper, #3832):** one frozen table,
`substrate_pe_history_snapshot_20261124`, copying the five ids' rows from the PE table (about
15,074 rows, 2026-11-17 13:05Z onward), never pruned. The postgres role's default privileges
already give `orion_readonly` SELECT on a new table, so no GRANT is needed, but creating it is a
production write. Each day of delay loses about a seventh of the rows. Evidence older than
11-17 (the 10-17, 10-26 and 10-31 priors) is gone from the live tables already; only DB backups
could hold it (UNVERIFIED).

**Priors whose evidence spans the cutover** (from `orion_worldview`, claims naming the five
nodes or the old score, not refuted):
- `self:substrate_pe_is_zero_inflated_pulse_20261031`
- `self:pe_reverie_high_vs_zero_inflated_20261123`
- `self:pe_ceiling_rate_0p42pct_crowding_unsupported_20261121`
- `self:perception_pe_level_is_my_variance_baseline_not_the_room_20261121`
- `self:perception_pe_node_is_room_cameras_only_20261118`
- `self:pe_history_excludes_placeholder_writes_20261118`
- `self:pe_recorded_cadence_is_publish_cadence_20261118`
- `self:pe_recorded_count_is_a_fresh_score_tick_count_20261119`
- `self:cabinet_offered_every_frame_outranked_by_its_own_calm_20261119`
- `self:pool_backlog_aborted_cabinet_pred_error_20261017`
- `self:strain_pe_is_heat_or_queue_pulse_20261108`
- `self:walkway_pe_spikes_track_unbound_presence_20261027`
- `self:lens_prediction_error_fuses_with_juniper_in_reverie_20261026`
The 11-16 "hot cabinet is a clipping artifact" finding is a curiosity finding, not a prior; it
rests on cabinet rows from before 11-17, which are already gone. The field note for each
cutover names these ids (step 3).

## Rev 4 changes (Orion's review of rev 3)

1. The scale ALTER ships in the same patch as the camera reader switch, migration first; the
   one SQL builder carries the predicate, pinned by a unit test and a CI check.
2. Backfill text fixed (it ran); strain seed precondition checked live.
3. Camera value is now a two-stage rank over the trailing 5 min, because the ~4-5 min sampler
   missed 86-92% of short changes under rev 3; band re-replayed at the sampler cadence.
4. Per-node seed ceiling (4.67 nats, derived), with a test.
5. `scene_change_excess` joins both kind sets; `scene_change` joins PRESSURE_CHANNELS;
   `glossary_channel_for` covers cameras; trigger min-fires is self-readings only.
6. Field expiry entries leave with the field receipt; graph-side absence rules named.
7. Seed volume measured; tighter band plus a displacement check.
8. Recalibration tested held-out and dropped (it was worse); Orion and the node use the same p.
9. street_rhythm gate is on the tail, stale keys are not scored, the forecaster already refits.
10. street_rhythm is a world node with a 600 s staleness horizon.
11. Snapshot proposed (#3832, needs Juniper) and the affected priors listed.
12. Post-cutover PE list stated; dated marker line in self-readings.

## Rev 3 changes (Orion's review of rev 2)

1. Camera acceptance band is now tick-level, from the replay's tick numbers.
2. Scale column (`v1` / `v2` / `v2_partial`); cameras read new-table v2 rows only, never the union.
3. Score from 30 min (attention needs it), mark rows `v2_partial` until the 2 h window is full,
   exclude them from calm checks and win-share.
4. Absorbing a sustained change is intended; glossary says "relative to the last two hours";
   #3806 is read before the legacy path is deleted.
5. The flag is removed by cleanup step 5b, after the eval and #3806.
6. No second seed path: the #3826 scan reads the node's own channel; bar 2.0 nats; replayed
   crossing 5.0% / 2.8% is an acceptance check.
7. Re-test data source stated; the scale column lets Orion read old rows without the contest mixing them.
8. street_rhythm shows stated and calibrated p side by side; the calm eval reports the tail.
   Only 1 clamp-bound miss exists, not 12.

## Rev 2 changes (Orion's review, 2026-11-24)

1. Channel move would silently drop all five from contests: item 4 now makes it one patch, with a test.
2. Union dedupe: confirmed in #3800 (`DISTINCT ON (node_id, observed_at)`, signal table wins).
3. Write rate: one value per 30 s tick (Sidak over the tick's frames), history sampled ~1 per 3.5 min.
4. Cold start: withheld below 200 frames and 30 min.
5. Window: 2 h of time, not 2,000 frames.
6. Curiosity keeps the cameras through a scene-change seed at today's crossing rate (item 7).
7. street_rhythm: identity and unscorable handling added; competes only after the identity read.
8. street_rhythm: recalibration added; power stated.
9. `ALARM_NODE_IDS` kept; importers listed. Strain event rows: same observed_at as the node
   write, so the sampler's later insert conflicts with an identical value; backfill copies them.
10. Migration path corrected to `services/orion-sql-db/`.
11. Order updated.
12. Read access stated; old beliefs and old rows handled.
