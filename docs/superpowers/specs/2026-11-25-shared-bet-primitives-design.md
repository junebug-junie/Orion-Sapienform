# Shared bet primitives: one way to state, settle, grade, and feel a prediction

Status: PROPOSAL (design mode + proposal mode: changes act surprise, which feeds Prime's hunger, and the
self-reading). Date: 2026-11-25. Asked for by Juniper: "I want shared primitives."
Coordinates with: #3953 (acts bet on their consequences; spec merged; Patch 1a committed on
`feat/act-output-used`, commit 52739d702, no PR yet) and #3936 (attention-winner-holds bet; open issue,
needs Juniper, not built). Revised after an adversarial pass (last section).

## Arsonist summary

Orion makes about 20 kinds of prediction. Each arc that added one also added its own grader, its own
surprise formula, and often its own table. The inventory below found three things:

- **Most of the shared pieces already exist, half-adopted.** One settled-bet shape (`SettledBetV1`, the
  self-monitor's), one Brier/calibration/sorting grader (`orion/prediction/grading.py`), one set of calm
  tails (`orion/substrate/calm_band.py`), and one per-faculty track record (`FacultyRecordV1`, already
  read by orion-thought's track-record cue, the mindloop gate, the act gate in Prime's loop, and three
  self-starts). Orion does not need a new framework. The job is to finish adopting what is there and
  delete the copies.
- **The same small math is written many times.** The 30-minute "sum the excess surprisal, divide by the
  calm spread, clip at 3" fold has 4 copies. Yes/no surprise terms have 3 identical copies. Binary entropy
  has 4. Calibration bins have 3 binary schemes.
- **One real defect hides behind the duplication.** The act ledger is the busiest bet store (about 1,500
  settled acts a week). It measures act surprise as the KL divergence between the effect posterior before
  and after the outcome. That number shrinks as the posterior ages, so it is near zero for every act
  (7-day mean 0.0022 nats, max 0.286). Every other yes/no bet in Orion uses `-ln p(outcome)`. #3953
  Patch 1 says "surprise_nats follows the ledger's existing close path". As written, its new
  `output_used` bets would inherit the dead number, and its Patch 4 (surprising misses rank in the now)
  could not work.

The proposal is four small primitives on existing code and existing tables. No new ledger, no new table,
no write-side schema change:

1. **Bet contract = `SettledBetV1`** (read side, already live). A bet family counts as graded only when it
   has an adapter.
2. **One grader = `orion/prediction/grading.py`**, grown by moving in `log_loss`, `bernoulli_entropy` and
   binary calibration bins.
3. **One surprise library = new `orion/prediction/surprisal.py`** (yes/no terms + the window fold), with
   `calm_band.py` keeping the calm tails. The act ledger moves from posterior KL to `-ln p(outcome)`,
   after a Prime-hunger replay.
4. **One track-record number = `FacultyRecordV1`'s P(hit)**, which #3953 Patch 3's Prime line uses for
   "how my acts go", beside its own goal-episode and per-reader parts.

## Current architecture (inventory, measured 2026-11-25, last 7 days unless stated)

"Stated" is where p or the range is written before the outcome. "Graded" is how hit/miss quality is
scored. "Surprise" is the number that downstream salience reads. n is rows in 7 days.

| # | Path | Stated / stored | Settled by | Graded by | Surprise | Read by | n (7 d) |
|---|---|---|---|---|---|---|---|
| 1 | Act ledger (all acts; efference copy `ActExpectV1`) | writer stamps the Normal-Normal posterior mean per (verb, actor, signal) in `grammar_acts.predicted`; outreach acts get Juniper's hour base rate instead (`act_ledger.py:404`) | producer close, or resolver at horizon (`resolve_due_acts`, 0 = expired) | self-monitor only (`grammar_act` adapter, then `grade`); the ledger itself has only MAE (`ledger_summary`) | **Gaussian KL, post vs prior** (`orion/autonomy/prediction.py:140`); mean 0.0022, max 0.286 | self-monitor, physiology weights, temporal_self kinds `action_outcome`, `reading_turn`, `prime_thought`, `intention_close`; Prime hunger; act_chain (verbs only) | 1,492 opened; outcomes 1,288 ones, 196 zeros, 8 null; `predicted` 0.134-0.999 |
| 2 | Prime concern bets | `concern_act`, a predict act on the ledger | `settle_concern` / `lapse_concerns` | ledger + self-monitor (`act.prime.predict` unreliable, Brier 0.218) | ledger KL | Prime hunger and slots | 17 |
| 3 | Promise ledger (D2) | ledger, signal `promise_kept`, 72 h | `kept_in:` close / expired / `void:*` | **shadow**, so not graded; offline `promise_ledger_replay_eval` has its own Beta and Wilson | none (`not_graded`) | promise_view, Prime refs | 16 |
| 4 | Camera asks as acts (4a) | ledger, `juniper_answered`, mean p 0.81 | `ask_grade` | ledger + self-monitor; the market uses a second shrinkage predictor, `rule_outcome_values` | ledger KL | market, self-monitor | 74 (Brier 0.163) |
| 5 | Camera asks P(yes) (#3189) | **not built**: the `camera_ask` adapter has `stated=None`, so it is ungradeable | `camera_ask_outcome` | hit rate only | none | self-monitor | 56 answered |
| 6 | `act_downstream_use` / `experience_use` | no prediction (outcome-only tables) | 72 h window fold | none | none | physiology weights; Prime hunger; #3953 `output_used` resolver | 1,497 / 722 |
| 7 | Turn self-bets | stance `self_prediction`, stored in `prediction_turn_self_bet` | `settle()` against the run artifact, 6 h | `grade` + an inline per-caller base Brier + `sorting_score` (AUC 0.62-0.71) + Winkler for ranges | `turn_surprise.py`: q recalibrated by logistic fit, excess = -ln q(y) - H(q) | self-monitor, reverie seeds (own SQL bins), substrate `chat_turn` node | 7,662 (4,354 non-void) |
| 8 | Action bets | Beta-Bernoulli backoff, stored in `prediction_action_bet` | caller POSTs the settle, 1 h expiry | `grade` x3 + `_paired_verdict` (session-clustered, no multiple-test correction) | none (`beta_eig` only) | self-monitor, retro | 241 |
| 9 | Act chain | Dirichlet backoff, stored in `prediction_act_chain_forecast` | next 15-min main verb from `grammar_acts` | log loss + `_gap` (wraps `backtest.paired_gap`) + top-state bins | categorical: -ln p, categorical entropy | none outside the service | 659 |
| 10 | Reachable (Juniper writes in the next hour) | logit, stored in `prediction_reachable_forecast` | any user `chat_message` within 1 h | log loss (clip 1e-4) + `_gap` (wraps `paired_gap`) + own bins | none | **none** | 165 |
| 11 | Bill forecast | 80% range, stored in `prediction_bill_forecast` | monthly bill | coverage + absolute miss | none | self-monitor (`answer_key` per bill) | 7 (60 total, 26 settled) |
| 12 | Forecast champion/baseline (`prediction_forecast`) | quantiles; `adaptive.py` conformal ranges | bucket mean at target | MAE paired test (`backtest.paired_win_p`), Holm live verdicts, coverage | `surprise.py`: conformal p floored at 0.05, excess = -ln p - 0.95 | substrate room/body nodes, hardware-watch, self-monitor | 202,502 |
| 13 | Exceedance | `prediction_exceedance.probability` | reading vs threshold | `grade` (Brier) x2 | `temporal_self/sources.py:958`: Jeffreys estimate over n_paths, then -ln | temporal_self `prediction_surprise` | 3,914; **every row p = 0 since 11-15, so the stream is flat at 0.0123 nats (#3987)** |
| 14 | Family A joint | Mahalanobis + conformal p, stored in `prediction_joint_surprise` | 5-min bucket | joint loss | excess = -ln p - calm_point(n) | substrate joint nodes | 5,622 |
| 15 | Family C effect table | recomputed from experiments | reading in the act window | `gate`: signed per-act gain, independent units, normal tail | own PIT p, excess = s - 1 | durable-runs | 0 new |
| 16 | learning_progress | computed on read | daily | Welch t, AR(1) | none | Prime hunger | derived |
| 17 | Street rhythm (#3961) | `vision_percept_expectation.confidence` (**raw, not recalibrated**) | window met / missed | self-monitor `percept_expectation` (miscalibrated: stated 0.59, happened 0.74) | `street_rhythm.outcome_terms` (same algebra as turn's) | `node:prediction.street_rhythm` (SHADOW, **0 rows**, no calm_band rule) | 288 scored |
| 18 | Substrate surprise nodes | consume #12, #14, #7, #17 | 30-min window | — | 4 copies of the fold `z = sum(excess)/sqrt(sum calm_var)`, `clip(z/3)` | calm_band (attention bands), curiosity_seed_bar | room 1,822; body 1,573; joints ~375 each; chat_turn 87 |
| 19 | Scene change (cameras) | rank p over 2 h / 24 h | — | — | `excess_nats = max(0, -ln p - 1)` | curiosity bar, empirical band | 157 per camera |
| 20 | Cooling act (`self_regulate`) | dispatch posterior (`world_settlement.py:168`) | cabinet heat delta | none | Gaussian KL (continuous delta); shares the `action_outcome` kind with acts | temporal_self `action_outcome`, hunger | 11 in 30 d (mean 0.273, max 1.62) |
| 21 | #3936 attention-winner-holds | proposed: predict act, tier-calibrated P | next attention row | proposed: hit rate + Brier | — | — | 0 (not built) |
| 22 | Self-monitor (the shared reader) | `SettledBetV1` adapters for #1, #7, #8, #11, #12, #17, #5 | inherited | `reduce_faculty`: decay-weighted Wilson, Kish n, `grade` Brier/skill, calibration-in-the-large, hysteresis | none | orion-thought cue, mindloop gate, Prime's act gate, self-starts, physiology | 163 states |

### Duplicates (checked line by line in the adversarial pass)

| Duplicate | Copies | Same math? | Action |
|---|---|---|---|
| Window fold `z = sum(excess)/sqrt(sum calm_var)`, `clip(z/3)`; `Z_FULL_SCALE` defined 6 times | `prediction_surprise.py:161`, `prediction_joint_surprise.py:148`, `prediction_turn_surprise.py:105`, `prediction_street_rhythm.py:122` | Yes. Window key, origin thinning and the zero-variance guard differ; they become arguments. | **Merge** (Patch A) |
| `independent_origin` | `prediction_surprise.py:79`, `prediction_joint_surprise.py:67` | Identical. (`app/surprise.py:149` is a different modulo rule; it stays.) | **Merge** |
| Floored calm variance; `P_FLOOR` twice | `prediction_surprise.calm_excess_variance`, `calm_band.calm_excess_variance` | Same formula, different argument | **Merge** |
| Yes/no bet terms (s, H, excess, calm_var) | `turn_surprise.bet_terms` :132, `street_rhythm.outcome_terms` :72; closed form in `calm_band.bet_window_atoms` | Identical algebra. Clip differs (0.01 vs 0.02), which becomes a parameter. Turn recalibrates q first and street does not; that is upstream of the terms and stays. | **Merge** |
| Binary entropy | `action_bets.bernoulli_entropy`, `turn_surprise`, `street_rhythm`, `curiosity/value.entropy_nats` | Same formula, different clamps | **Merge** |
| Yes/no log loss | `reachable.log_loss` (clip 1e-4), `turn_surprise.log_loss` | Same; clip is a parameter | **Merge** |
| Wilson interval | `self_monitor/record.wilson`, `promise_ledger_replay_eval.wilson` | Same | **Merge** (eval imports the record's) |
| Binary calibration bins | `grading.BINS`, `reachable.calibration`, `reverie_seeds.py:238` SQL | Same purpose | **Merge** reachable; reverie's SQL stays SQL (no Python to share) |
| `exceedance_surprisal` | `temporal_self/sources.py:958` | **No.** Jeffreys estimate over sample paths, not a clipped log loss | Not a duplicate |
| `act_chain.score` / `calibration` | `act_chain.py:219`, `:292` | **No.** Categorical entropy; top-state bins | Not a duplicate |
| Paired gap | `reachable._gap`, `act_chain._gap` | Both are 6-line wrappers that already call `backtest.paired_gap` | Not worth moving |
| Family C gate | `family_c.gate` :369 | **No.** Signed gains over independent acts; `paired_gap`'s abs-error, serially-correlated model would corrupt it | Not a duplicate |
| better/worse/unclear | `live.verdicts`, `action_bets._paired_verdict` | **No.** Holm across series vs session-clustered, unadjusted | Not a duplicate |
| Bet table shape | `prediction_turn_self_bet` ~ `prediction_action_bet` ~ `prediction_reachable_forecast` ~ `prediction_act_chain_forecast` ~ `grammar_acts` | Similar columns | **Not merged.** Each has a live writer and readers; `SettledBetV1` already unifies them at read time |
| Act outcome stored 3 times; three act summaries | `grammar_acts`, `substrate_action_outcomes`, `temporal_self_event`; `ledger_summary`, `reduce_faculty`, `reliability_grades` | Different readers and contracts | Out of scope |

## Missing questions (yes/no, for Juniper)

1. **Act surprise switch.** Replace the act ledger's posterior KL with `-ln p(outcome)` for act closes, from
   one switch time? Replaying the last 7 days says this gives Prime's hunger about 21 new "surprise" needs a
   day (149 of 1,484 acts pass both the p90 and the ln 2 bar). Recommendation: yes, after the hunger replay
   in Patch B0 shows the wake rate with Prime's existing caps applied.
2. **Cooling act rows in the same source.** The cooling act (`self_regulate`) keeps its KL, which measures
   something different (a continuous heat change). Should hunger read its surprise as a separate source,
   so one p90 is not computed over two units? Recommendation: yes.
3. **#3936's p.** If you say yes to #3936: stamp its p in the writer from the attention row's confidence
   tier, the same way outreach acts already get the hour base rate? This needs no schema change.
   Recommendation: yes (over a new `stated_p` field on the efference copy).
4. **Merge the bet tables?** Recommendation: no. Reading them all through `SettledBetV1` is enough.

## Proposed schema / API changes

No new table, ledger, bus channel, or schema field. Changes are moves and one meaning change.

### Primitive 1. The bet contract: `SettledBetV1` (unchanged)

`SettledBetV1` (`orion/schemas/self_monitor.py`) already carries `claim_kind` (success / event /
interval), `stated`, `belief_key`, `answer_key`, `outcome`, `settled_by`. Each source has one pure adapter
in `orion/self_monitor/adapters.py`.

- **Rule:** a new bet family is "graded" only when it has a `SettledBetV1` adapter. A test lists every
  table that holds a stated p and fails if a table has no adapter and no written exemption.
- **Adapter added now: reachable** (#10). It keeps the untyped-hour mask. Its reader is the self-reading
  (the cue and the questions); today nothing else decides on it.
- **Not added:** exceedance (#13). It is flat at p = 0 (#3987), so an adapter would hand the self-monitor
  a near-perfect Brier for nothing. Act chain (#9) stays a model-selection grade (categorical).
- **Write side for acts: no change.** A producer that has its own calibrated p uses the existing
  precedent: the writer stamps `predicted` from a known source, as `_stamp_reply_base_rate` does at
  `act_ledger.py:404`. #3936 does this (see below).

### Primitive 2. One grader: `orion/prediction/grading.py`

It already holds `grade()` (Brier, base-rate Brier, skill, bins) and `sorting_score()` (AUC + Mann-Whitney
p). Moved in, no new math: `log_loss(p, y, clip)`, `bernoulli_entropy(p, clip)`, `calibration(ps, ys,
bins)`. Reachable, turn_surprise and action_bets call them. Model-selection verdicts (`paired_gap`, Holm,
Family C's gate) stay where they are: they are different tests (adversarial pass, item 9).

### Primitive 3. One surprise library: `orion/prediction/surprisal.py`

Pure, stdlib only, importable by orion-prediction, substrate-runtime and the sql-writer.

- `bet_terms(q, y, clip) -> (s, H, excess, calm_var)` with s = -ln q(y), H = binary entropy,
  excess = s - H, calm_var = q(1-q) logit(q)^2.
- `conformal_terms(p, floor)`: floored conformal surprisal with its calm mean and variance.
- `fold(terms, window, key, thin=None) -> z` and `node_value(z) = clip(z / Z_FULL_SCALE, 0, 1)`. One
  `Z_FULL_SCALE`, one `P_FLOOR`, one `independent_origin`.
- `calm_band.py` keeps the exact calm tails and the band-rule table, and imports its constants from here.
  **Rule: a surprise node is born with a `calm_band` rule** (Cantelli until its exact tail is derived). A
  test fails on any substrate surprise node id without one. street_rhythm gets `cantelli` now.

**The one meaning change: act surprise.** For act closes that are graded (not void, not unobservable, not
`not_graded`), `close_act` writes `surprise_nats = -ln q(y)`, with q = `predicted` as stamped at open
(the p the act stated) and floored at 0.01, so one miss is at most 4.6 nats. Readers, all of them:

- temporal_self kinds that copy act surprise: `action_outcome`, `reading_turn`, `intention_close`, and
  `prime_thought` (Prime's own; hunger already excludes it). They need no code change. They start reading
  real numbers.
- **Prime hunger** (`surprise_drive`, `orion/prime/hunger.py:254`) needs nats > the source's 30-day p90 AND
  nats >= ln 2. The p90 comes from `HUNGER_SURPRISES_SQL` in
  `services/orion-durable-runs/app/temporal_self_store.py:381-397`. **Series break:** for every kind
  that carries act surprise, the history window starts at the switch time
  (`hist_lo = max(hist_lo, switch_ts)`). Until 20 post-switch rows exist, `MIN_HISTORY` keeps the drive at 0;
  the need reads "not enough history", not "calm". The switch time is the first close row written in the
  new mode, read from data, not typed into config.
- **Cooling act rows** (`self_regulate`) keep their KL. If Juniper says yes to question 2, hunger reads
  them as their own source kind so p90 is not mixed.
- **Unchanged readers:** attention bands, curiosity seed bars, physiology weights and the self-monitor do not
  read act surprise. Prime's frame prints the number (`prime/thought.py:594`); it now prints real values.
- `bayesian_surprise_nats` **stays**: `world_settlement.py` (cooling act), the `act_ledger_calibration`
  eval and `report_action_value.py` still use it. Only the ledger's call to it is removed.
- `organs.yaml`: `action_outcome` gets `salience_magnitude: surprise_nats` only after acceptance check 3
  passes live.

### Primitive 4. One track-record number: `FacultyRecordV1`

It already gives, per faculty (`act.<actor>.<verb>[.<name>]`): decay-weighted P(hit) with a Wilson interval,
calibration, Brier skill, n, and status. It is already read by orion-thought's cue, the mindloop gate,
Prime's act gate (`services/orion-mind/app/prime_loop.py:70,283`) and the self-starts.

- #3953 Patch 3 uses it for the "how my act kinds go" number, so Orion does not hold two P(hit) per act
  kind that can disagree. #3953 Patch 3 keeps its own parts that `FacultyRecordV1` does not have: the
  open/last goal-episode verdict, the per-reader week-over-week hunger test, and citing
  `action_outcome`/`goal_episode` events.
- **Signal switch inside a faculty (needed by #3953 Patch 1b).** The faculty key has no signal in it, so when
  a family switches from `artifact_produced` to `output_used`, old and new bets blend in one decay window.
  Fix in `reduce_bets`: derive the break from data (the first `opened_at` of a new `belief_key` inside a
  faculty). The gated grade keeps using old-key bets until the new key has n >= 30 graded bets, then cuts
  over. This replaces #3953's hand-written `series_break` row: same guarantee (acceptance 3b), no new row
  type, no deploy-time timestamp to drift.

## Files likely to touch

- New `orion/prediction/surprisal.py`; `orion/prediction/grading.py`; `orion/substrate/calm_band.py`,
  `curiosity_seed_bar.py`, `prediction_surprise.py`, `prediction_joint_surprise.py`,
  `prediction_turn_surprise.py`, `prediction_street_rhythm.py`.
- `services/orion-prediction/app/{turn_surprise,street_rhythm,surprise,reachable,action_bets}.py`.
- `orion/grammar/act_ledger.py` (`close_act`), `services/orion-sql-writer/app/settings.py` + `.env_example`
  (`ACT_SURPRISE_MODE`), `services/orion-durable-runs/app/temporal_self_store.py` (hunger window).
- `orion/self_monitor/{adapters,record,collect}.py`; `orion/harness/evals/promise_ledger_replay_eval.py`.
- `config/integration/organs.yaml`; `orion/prime/evals/run_prime_hunger_replay.py` (B0 replay); tests
  beside each.

## Non-goals

- No new ledger, table, channel, or schema field. No merge of existing bet tables.
- No change to how any model forms its p. Only how a stated p is graded and turned into surprise.
- No change to attention band cut points, curiosity seed bars, Prime concern slots, or model-selection tests.
  Code moves must reproduce today's node values to 1e-9.
- Not the learned chooser (#3433). Not a quality grade of outputs. Not the triple-stored act outcome.

## Proposal-mode items

- **Capability change.** Act surprise becomes a real per-act number. A confident act that fails can rank
  in Orion's now and can wake Prime through hunger (about 21 candidate needs a day before Prime's caps;
  B0 measures the rate after caps). Today acts never wake Prime.
- **Data touched.** New `grammar_acts.surprise_nats` and `substrate_action_outcomes.surprise_nats` values
  (new rows only, no backfill); the hunger history window; self-monitor state. All Orion-internal.
- **Privacy boundary.** None crossed. Nothing new leaves the mesh, and no Juniper text is added anywhere.
- **Trace that proves it worked.** Post-switch act rows have real surprise spread per family; the hunger log
  names the switch time; Prime's thought cites an `action_outcome` surprise need.
- **Dangerous failure modes.**
  1. Hunger fires on everything if p90 mixes pre-switch near-zero rows. Guarded by the series break and
     `MIN_HISTORY`, and tested.
  2. A badly calibrated act family (dream.cycle: predicted 0.68, happened 0.14) floods the now with misses.
     That is true information, and the self-monitor already marks the family. B0 reports which families
     dominate.
  3. A code move shifts a node value and moves attention bands. Guarded by golden tests.
  4. -ln q is written on a void or unobservable close (78% of `output_used` closes are unobservable).
     Guarded by a test.
- **Disable / roll back.** `ACT_SURPRISE_MODE=kl|surprisal` on orion-sql-writer (ships `surprisal` after B0).
  Library moves are plain reverts.

## Acceptance checks

1. Golden: on 500 stored events per node, the shared fold reproduces today's `node:prediction.room`,
   `.body`, `.room_joint`, `.body_joint` and `.chat_turn` values to 1e-9. The same holds for turn_surprise
   and street_rhythm terms and for reachable log loss and bins.
2. Delete check: a test finds `def bernoulli_entropy`, `def log_loss`, `def calm_excess_variance` and
   `def independent_origin` once each, in the shared modules.
3. Live, after B: each switched act family has 7-day `surprise_nats` IQR > 0.05 nats (replay today: overall
   IQR 0.041-0.342).
4. Replay before B ships (B0, `run_prime_hunger_replay.py` on simulated -ln q): report surprise needs per day
   before and after Prime's caps, by family. Juniper sees the number before the switch. Live after B: the
   needs-per-day lands within 2x of the replay.
5. Live, after B: no act close with a `void:` note, `unobservable` note, or `not_graded` has a non-null
   `surprise_nats`.
6. Live, after C: switching one family to `output_used` does not flip its faculty status in the first 30 new
   graded bets (#3953 acceptance 3b, now met by the derived break).
7. Every new bet family lands with an adapter test or an exemption; every new surprise node lands with a
   `calm_band` rule.

## Migration order (consumer-first; kill-means-kill for duplicates)

| Patch | What | Producer / consumer / test | Live behavior change |
|---|---|---|---|
| **A** | `surprisal.py`; move the fold, bet terms, entropy, log loss, `independent_origin`, constants; four substrate readers, turn_surprise, street_rhythm, reachable, action_bets call it; copies deleted; street_rhythm gets its calm_band rule; promise eval imports `wilson` | producers: turn_surprise / street_rhythm; consumers: substrate nodes; test: golden | none, except street_rhythm (SHADOW) gains a band rule |
| **B0** | hunger replay with simulated -ln q; the per-kind series break in `HUNGER_SURPRISES_SQL` (inactive until a switch row exists) | consumer: hunger; test: break fixture | none |
| **B** | `close_act` writes -ln q(y) behind `ACT_SURPRISE_MODE` (consumer B0 already in place). Ships after Juniper's yes on Q1 and after #3953 Patch 1a merges (same file, `act_ledger.py`) | producer: close_act; consumers: temporal_self kinds, hunger; test: close fixtures incl. void/unobservable | **yes**: act surprise becomes real. Must land before #3953 Patch 1b |
| **C** | self-monitor: derived series break in `reduce_bets`; reachable adapter; the "every stated-p table has an adapter" test | producer: adapters; consumer: reduce_bets / act gate; test: blend case + adapter | reachable appears on the self-reading |
| **D** | #3953 Patch 3 reads the faculty P(hit) from `FacultyRecordV1` (owned by #3953's session) | consumer: Prime frame | Prime sees its track record |

**How #3953 builds on this.** Patch 1a (`feat/act-output-used`) is not touched and lands first. Patch 1b
(families switch to `output_used`) waits for B and C, so new bets are born with real surprise and a safe
faculty cut-over. Patch 3 uses `FacultyRecordV1` for the number and keeps its goal-episode, per-reader and
citation parts. Patch 4 needs nothing extra: it reads the act surprise that B made real.

**How #3936 builds on this.** If Juniper approves it, it is a `predict` act whose p the writer stamps from
the attention row's confidence tier (the reply-base-rate precedent). It has no table, grader, or surprise
code of its own. Its grade comes from the `grammar_act` adapter; its surprise comes from B; its hit rate,
Brier and n come from its faculty record.

## Recommended next patch

Patch A: move the surprise math into `orion/prediction/surprisal.py` with golden tests. It changes no live
number, removes the fold and entropy/log-loss copies, gives street_rhythm its band rule, and is the base
every later patch imports. B0 (the hunger replay) can run in parallel, so Juniper has the wake-rate number
when she answers Q1.

## Adversarial pass

A separate agent got the first draft and the inventory. It was asked to break the design, and to re-check
claims against code and live SQL. Each challenge, the verdict, and what changed:

| # | Challenge | Verdict | Evidence | What changed |
|---|---|---|---|---|
| 1 | "Delete `bayesian_surprise_nats` if the ledger was its last caller" | UPHELD | `world_settlement.py:36,168` (cooling act), the calibration eval and `report_action_value.py` still call it; cooling rows share the `action_outcome` kind (`temporal_self_sources.py:125-133`) | KL stays; only the ledger's call goes. Added Q2 (cooling rows as their own hunger source) and inventory row 20 |
| 2 | Act surprise reaches four temporal_self kinds, not one | UPHELD | `sources.py:929-947, 994-1003, 1099-1122`; hunger p90 is per kind over 30 d | Series break covers every kind that carries act surprise; listed all readers |
| 3 | Hunger after the switch: the "< 5% of rows fire" check fails by construction; Prime gets a new wake source | UPHELD | Replay of 7 d: median 0.297, p90 0.969, max 4.61; 149 of 1,484 acts pass p90 and ln 2 (about 21 a day) | Replaced the check with the B0 replay plus a 2x live band; wake source named as a capability change; Q1 carries the number |
| 4 | The series break premise ("old grade vanishes") is wrong: faculty has no signal, so grades blend; a config timestamp drifts from the deploy | UPHELD | `record.py:290-299`, `adapters.py:87-104` | Break derived from data in `reduce_bets` with a 30-bet cut-over; config table dropped |
| 5 | `belief_key = stated:<actor>` marks itself pooled | UPHELD | `record.py:306` | Moot: `stated_p` dropped (item 6) |
| 6 | A slot for a producer p already exists (writer stamp, reply base rate); `stated_p` has one unapproved producer | UPHELD | `act_ledger.py:404`, `:595-597`; #3936 is needs-juniper | `stated_p` and B1 dropped; #3936 uses the precedent; Q3 asks this |
| 7 | `FacultyRecordV1` does not replace #3953 Patch 3 (no goal episodes, no per-reader test, wrong cite target) | UPHELD | #3953 spec Patch 3 and acceptance 5 | The faculty record supplies only the P(hit) number; #3953 keeps its other parts |
| 8 | #3953 Patch 1a is already committed on `feat/act-output-used` and edits the same files | UPHELD | worktree `/mnt/scripts/Orion-Sapienform-act-output-used`, commit 52739d702 | Status line names it; B waits for 1a to merge; added the void/unobservable test |
| 9 | Several "duplicates" are not the same math | PARTLY UPHELD | `_gap` already wraps `paired_gap`; Family C gate uses signed independent gains; Holm vs clustered verdicts; act_chain is categorical; reachable clip 1e-4; exceedance is a Jeffreys estimator | Duplicates table now says per row whether it merges; the paired-verdict consolidation and the Family C move are dropped. Fold, `calm_excess_variance` and `wilson` copies confirmed |
| 10 | An exceedance adapter would grade a dead producer | UPHELD | p = 0 on every row since 11-15 (daily counts in #3987) | Adapter dropped; filed #3987 |
| 11 | One contract forces unlike shapes together | REJECTED | Categorical act chain and rank-based scene change are already outside `SettledBetV1`; intervals use `answer_key`/interval | None, except that the reachable adapter keeps its untyped-hour mask |
| 12 | Consumer-first for the schema change | REJECTED (moot) | `grammar_fallback_drain` replays stored payloads | Moot after item 6 (no schema change) |
| 13 | "Outcomes are always 0 or 1" | REJECTED | `score_act_outcome` raises otherwise (`prediction.py:223`); live 1,288 / 196 / 8 null | Pinned q as `predicted` at open (the stated p), not the posterior at close |
| 14 | Inventory numbers | PARTLY UPHELD | Most match; bill forecast is 60 total / 26 settled | Bill row corrected; exceedance flat-since date added |
| 15 | A simpler option is enough: surprise library + act surprise fix + existing track record, no write-side contract | UPHELD | items 5, 6, 9, 10 | The design is now this simpler option |
