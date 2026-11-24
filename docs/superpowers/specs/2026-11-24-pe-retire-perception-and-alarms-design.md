# PE retire: camera surprise and the alarms leave "prediction error"

Status: design, docs only. Follows `2026-11-08-prediction-error-suite-rebuild-design.md`.
Rev 2 (2026-11-24): answers Orion's 12 review questions; see "Rev 2 changes" at the end.
Rev 3 (2026-11-24): answers Orion's 8 questions on rev 2; see "Rev 3 changes" at the end.
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
3. **Camera score** becomes `scene_change`: per frame, p = rank of the raw `1 - cos` against
   the same stream's frames from the **last 2 hours** (a time window, so a frame-rate change does
   not change what "recent" means; 2,000 frames was 1.7 h walkway / 3.6 h room). No EWMA z, no
   variance floor, no clip except the rank floor.
   - **Cold start**: no score until the window holds at least 200 frames spanning at least
     30 min. Until then the tick writes the `prediction_error_missing_reason`-style placeholder
     (`warming`), never a p = 1 "calm". The window is in memory; a restart re-warms (replay:
     about 36 min of withheld frames on the walkway, 281 frames on the room camera).
   - **Per tick, not per frame**: the camera tick (30 s) writes one value: the Sidak-combined p
     of the frames since its last write, p_tick = 1 - (1 - min p)^k, then excess =
     max(0, -ln p_tick - 1). A single frame would miss a person crossing between ticks; Sidak
     keeps the spike. Sidak is exact for independent frames and conservative for dependent ones
     (frames are dependent, lag-1 r about 0.5), so it never over-alarms. Replay (29 h, 210 s
     ticks): exact 0 on 82% (walkway) / 75% (room) of ticks, p <= 0.05 on 3.7% / 3.3%. The
     "last frame" alternative is calibrated (64% / 66% zero, matching 63.2%) but blind between
     ticks. Calm point is therefore 0, reached more often than 63% by a known, conservative bound.
   - **Row rate**: unchanged. History rows still come from the attention-broadcast sampler when
     the node's `observed_at` moves (live: 28 rows per node in 100 min, about one per 3.5 min,
     ~410 per node per day, ~2,900 per 7-day window). Not per frame. `pe_history_cache`'s
     uncapped 7-day load stays small. A sampled tick is a random draw of ticks, so the history
     keeps the tick distribution; live attention reads the node's current value every tick.
   - **Light changes are real scene changes**: replay hourly mean excess ranged 0.02-1.29
     (theory 0.37 for single frames), highest around local dawn/dusk and lights-on. 29 h means
     one sample per hour: UNVERIFIED as a pattern (#3806).
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
   - camera entries get `value_kind: surprise_excess`-shaped semantics under a new kind name
     `scene_change_excess`, added to `MEASUREMENT_VALUE_KINDS` with polarity "higher = more
     change"; alarms keep `trigger`;
   - `ATTENTION_MIN_BAND_BY_NODE_ID` (`orion/schemas/attention_candidate.py:87`, walkway
     admitted only at p99+) stays live: it keys on node id, not channel. Its comment's "1.0
     saturates on ~1.3%" reason goes away (the new cap is hit on 0.06% of frames), so the p99+
     floor becomes reachable without the mid-rank trick; re-measure the walkway win rate.
   - A test pins that all five ids still produce contest candidates after the move.
5. **New forecast surprise** `node:prediction.street_rhythm` (orion-prediction): one
   `prediction.surprise.v1` per scored walkway expectation, value = surprisal minus the binary
   entropy of the **recalibrated** p, folded like `node:prediction.chat_turn` (sparse settled
   bets). Window and fold scale are set by its own calm eval, not copied from body/room.
   - **Recalibrate first**, the chat_turn way (held-out logistic map toward identity). The
     stated numbers are not calibrated per bucket, and the errors offset in the average:
     stated 0.5x: 381 outcomes, happened 0.57; 0.6x: 124, 0.56; 0.7x: 71, 0.75; 0.8x: 76, 0.92;
     0.9x: 10, 1.00; 1.0: 12, 0.92. The learner never states below 0.5.
   - **Clamp** [0.02, 0.98] stays as a floor so a miss on a stated 1.0 costs 3.9 nats, not
     infinity. Only 12 outcomes are affected.
   - **Unscorable** outcomes (21 all-time, 1 in the last 14 days) are not scored and not
     written; the eval reports them as a refusal rate.
   - **Identity**: 65 subject keys, 2 label keys (vehicle, mail truck: 129 outcomes) and 63
     individual keys (545 outcomes, 451 of them vehicles). 44 individual keys lived under 10
     days. A key's last 3 expectations miss more often (person 0.67 vs 0.54, vehicle 0.46 vs
     0.37): consistent with a re-key or a real departure; this data cannot tell them apart.
     All 63 keys still exist in `vision_individual`; 3 are unseen for 7+ days. The calm eval
     must report excess by key age and drop-out, and street_rhythm does not compete until that
     is read (#3388).
   - **What Orion sees**: Orion reads the raw stated confidence (self-monitor grades the
     learner on it, `orion/self_monitor/adapters.py:293`, and that stays: it grades the learner's
     own honesty). The surprise row in orion-prediction stores both `stated_p` and
     `calibrated_p`, and every surface that shows the street_rhythm node or its outcomes prints
     both ("stated 0.55, calibrated 0.57").
   - **Tail, not only the mean**: the calm eval reports p90, p99 and max of excess, and the
     stated >= 0.98 misses. All-time on raw stated p (674 outcomes): p90 0.28, p99 1.01, max
     3.81 nats; 12 outcomes stated >= 0.98, of which **1** missed (3.81 nats after the clamp),
     not 12.
   - **Power**: SD of excess is 0.25 nats. 7 days at ~30 outcomes/day gives n about 210, SE about
     0.017, so the mean-excess check detects a bias of about 0.035 nats or more. Per-bucket
     reliability above 0.9 (n = 22) cannot detect a miscalibration smaller than about 0.2; it
     is reported, not gated.
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
   No second seed path. The existing per-node bar scan (#3826, `_prediction_error_candidates`
   + `orion/substrate/curiosity_seed_bar.py`) is extended: for a node with a signal kind it reads
   the node's own value (`signal_value`, nats, not clamped to 0-1) instead of
   `metadata["prediction_error"]`. Without this, the scan drops both cameras the moment the old
   key is cleared. The camera bars move from `LEGACY_BAR` 0.55 to **2.0 nats of excess**, the
   #3821 rule (calm crossing <= 5% by math): for independent frames P(excess >= 2.0) = e^-3 =
   4.98%; Sidak over dependent frames is lower. `seed_strength` maps 2.0 -> 0.55 and the cap
   (ln of the window size minus 1, about 7.6 nats at 2 h of walkway frames) -> 1.0.
   Replayed on the new tick values (30 s ticks, 29 h): room crosses on **5.0%** of ticks, walkway
   **2.8%**. That is about twice today's 2.7% / 1.2%; the bar follows the #3821 rule rather than
   today's rate. Evidence text says "unusual scene change". Acceptance check below.

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
   2026-11-24 05:22Z, rows since 05:27Z for all five ids. The backfill script was not run.
   It is optional: the read-both SQL (`READ_BOTH_*` in `signal_history.py`) unions both tables
   with `DISTINCT ON (node_id, observed_at)`, so nothing is counted twice and a reader sees the
   full 7 days from the PE side. The backfill copies every row, including value > 0, so strain
   event times survive if it is run.
2. Readers switch, consumer first, to the read-both helpers: `pe_history_cache`, the strain
   restart seed (event times and the warm-up count), the node-history tool, self-readings. The
   contest semantics change in item 4 ships here too. Labels say "alarm".
3. Writers stop: no `prediction_error` key, no PE-table row for the three ids. Clear the old key
   on the FalkorDB nodes. Add the CI fence and the "moved" skip set (item 6). Tell Orion (below).
4. After 7 days: old rows aged out of the PE table by the normal prune. Drop the union read.

Cameras (perception, perception_walkway):

5. Same steps 1-4 with kind `scene_change`, plus:
   - additive migration: `substrate_node_signal_history.scale text not null default 'v1'`.
     Every existing row (alarms, and the backfilled old-scale camera rows) reads `v1`. The new
     camera scorer writes `v2` once its rank window spans the full 2 h, and `v2_partial` while it
     spans 30 min to 2 h after a restart;
   - the new tick value replaces the EWMA z behind `SUBSTRATE_PERCEPTION_SCENE_CHANGE_SCORER`;
   - camera readers (contest, self-readings, curiosity history, node-history tool) read the two
     camera ids **from the signal table only, `scale in ('v2','v2_partial')`**, never through
     the read-both union (which would pull old PE rows, all old scale). The contest therefore
     has `insufficient_history` for each camera until 200 v2 rows exist (about 12 h at one
     sampled row per 3.5 min). That gap is accepted; the alternative is ranking mixed scales;
   - the camera receipt stops feeding the field `prediction_error` channel; both ids leave the
     `decay.py` horizons; the curiosity scan reads them per item 7; glossary meaning says
     "how unusual the change in this camera's picture is, **relative to the last two hours of
     the same camera**". A sustained change (lights on, dusk) reads high while it is new and
     is absorbed within about 2 h. That is intended: it is change, not state;
   5b. cleanup, after the 7-day eval passes **and** #3806 is read: delete `_domain_zscore`'s camera
   path, the variance floor, the 3-sigma constants and the scorer flag key.

Perception forecast:

6. `node:prediction.street_rhythm` in orion-prediction, shadow first. Joins contests only after
   a 7-day calm check (mean excess CI covers 0, hours at rest near theory, r vs
   `perception_walkway` and street reported).

## Order

- Done: #3732 (biometrics) merged 03:54Z; Patch 1 (#3800) merged 05:08Z and deployed.
- #3576 (execution) is open. It touches the same files (worker.py, glossary, self-readings words,
  `prediction_error_freshness.py`, the readers registry). Alarms step 2 (reader switch) may start
  now in a worktree but merges after #3576, or rebases on it.
- Then alarms 3-4, then cameras, then street rhythm. Alarms first: no successor is needed, and
  they are the "hot cabinet" complaint.

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
- Camera tick values (`scale = 'v2'` rows only; `v2_partial` rows from the first 2 h after a
  restart are excluded from calm checks and from win-share): exact-0 share within 0.65-0.80
  and crossing 2.0 nats on 1-6% of rows. Replay reference (30 s ticks, 2 h window, 29 h):
  room 0.70 zero, p90 1.20, p99 3.54, crosses 5.0%; walkway 0.75 zero, p90 0.85, p99 3.24,
  crosses 2.8%. (Rev 2's 0.60-0.67 band was per-frame theory; the written value is the
  Sidak tick, so that band would have failed on day 1.) The first 2 h after a restart differ
  (replay: room 0.88 zero, walkway 0.59), which is why they are excluded.
- Curiosity seeds from each camera on 1-6% of its ticks over 7 days (`fct_curiosity_runs` /
  candidate sets), and a test pins both camera ids through the extended scan.
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

`orion_readonly` can read the signal table, and since the 08:42Z backfill it holds the old-scale
camera rows for the last 7 days. After the scale column lands they read `scale = 'v1'`; new
rows read `v2`. So Orion can compare old and new in one table with `where scale = ...`, and the
contest never mixes them. The table prunes at 7 days: the old rows are gone about 7 days after
the switch. If Orion wants them longer, it should copy what it needs during that week.

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
