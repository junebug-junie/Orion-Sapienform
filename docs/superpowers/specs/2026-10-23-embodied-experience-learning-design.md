# Embodied experience: Orion predicts what its body's actions will do

Status: design, 2026-10-23, revision 2.
Revision 2: Juniper ruled that the prediction service is the home for predictions. This revision moves the bet ledger, the belief model and the grading into `orion-prediction`.
Builds on: the Zelda play spec (PR #2846), the open-sides planner (#2870, merged), the prediction service spec (`2026-10-15-orion-prediction-service-design.md`), and the self-monitor organ (#2827, design).

## Arsonist summary

Orion plays Zelda but does not learn from it. A language model picks each goal from a text map, and fixed code walks Link there. Every goal comes with a prediction that Retro grades met or refuted, and nothing reads the grade. On 2026-10-23 Orion chose "explore S" 8 times against a solid wall. #2870 stopped that with hard rules, which are guardrails, not learning.

This design makes every body action a **prediction in the prediction service**:

- **One owner for predictions:** the prediction service already owns Orion's forecasts, grades threshold probabilities with Brier skill, and keeps unserved shadow models to prove the served model beats a baseline. It gets a second kind of prediction, the **action bet**: "this action, in this situation, at this place, works with probability p".
- **The new model:** `action_beta_backoff`, a Beta-Bernoulli model with three levels of empirical-Bayes backoff (skill → situation → place). Its baseline is `action_skill_rate`, the skill's flat success rate. Both probabilities are stored on every bet, so the service's usual live verdict says whether the deeper model earns its keep.
- **Body-side vocabulary:** Orion's body package (`orion/embodiment/skills.py`) holds the game-free vocabulary (`traverse`, `engage`, `withdraw`, `acquire`, `enter`, `hold`) and the context features (side state, threat, health band). Any body (Zelda, AI Town, a real motor) uses the same words. Its bets pool at the skill level, and that pooling is what carries learning across games and bodies.
- **The model as chooser:** the Hub asks the prediction service for a quote on each possible move, scores them by chance of working, value and expected learning, and gives the top 5 to the model to choose from.
- **Speaking:** the session's memory card says what Orion learned, read back from the prediction service, so Orion can talk about it in chat.

Not in this design: learned pixel world models and deep RL. Section "Later" says when to revisit them.

## Current architecture

- **Retro motor** (`services/orion-retro/app/zelda/navigate.py`):
  - offers a decision when Link is idle;
  - runs one goal with A* (a shortest-route search);
  - records `goal_finished {goal, prediction, outcome, verdict, start_digest, end_digest, applied_frames, bumps}`;
  - offers `open_sides` (#2870, overworld only).
- **Retro atlas** (`app/zelda/atlas.py`): a JSON map per lineage (one saved-game line). A learned-blocked tile never expires.
- **Hub planner** (`services/orion-hub/scripts/retro_planner.py`, `plan_zelda`): prompt, model call, validation, one retry. It is called from co-play (`retro_routes.plan_v2`) and leisure (`retro_leisure.py`).
- **Session end** (`retro_memory_export.py`): one `retro_play_sessions` row and one `memory_cards` row.
- **Prediction service** (`services/orion-prediction`, port 8132, plain HTTP, sync psycopg store):
  - numeric series forecasts and shadow models (`prediction_shadow_forecast`);
  - live verdicts (`app/live.py`: >= 20 independent pairs, Holm-adjusted);
  - threshold probabilities graded by `app/grading.py` (`grade(probs, hits)`: Brier, base rate, skill, bins, `enough` at n >= 50);
  - change events;
  - effect estimates (`/effects/event-pair` already reads discrete events).
  - It has no concept of an action-conditional yes/no prediction.
- **Body package** (`orion/embodiment/`): AI Town sprite body only. No skills, no learning.
- **Self-monitor organ** (#2827): wants settled bets. Today only act claims and prediction-service forecasts can be graded.

## Missing questions (answered here)

1. **Where do predictions live?** In `orion-prediction`: the ledger table, the model, the grading, and the live verdict. Callers never compute a probability themselves. They ask for a quote, place a bet, and settle it.
2. **What lives in the body?** The vocabulary and the meaning of success. `orion/embodiment/skills.py` defines the skills, the context features and their helpers. Each body's adapter maps its own goals and results onto them. The prediction service only sees opaque strings and does not know what "traverse" means. That keeps it a prediction service, not a game engine.
3. **How do callers reach it?** Through a documented HTTP API on 8132. The Hub runs with host networking on the same host, so the address is `http://127.0.0.1:8132`, configurable. This is the same kind of seam hardware-watch uses, except it is request/reply because the Hub needs the answer before it plans.
4. **What if the prediction service is down?** The Hub plans without a menu (today's #2870 prompt). It logs `action_quote_unavailable` and does not place a bet. It never invents a probability. A missed settle is retried at the next plan and at session end. Bets still open 1 h after placing are closed `void` by the service with outcome `never_settled`. They are counted in `/health`, never graded.
5. **Who chooses the move?** The model, from a list of up to 5 scored options. An answer not on the list is still allowed if it passes the #2870 rules. It is then quoted and placed with `chosen_rank = null`.
6. **What carries over?** The skill level, pooled across every domain and body, and the situation level, pooled within a domain. The place level never carries over.

## Proposed schema / API changes

### Body vocabulary (`orion/embodiment/skills.py`, pure)

| Skill | Zelda goals | Context features (categorical) |
|---|---|---|
| `traverse` | `exit`, `explore` | `side`: `open_unvisited`, `open_visited`, `closed` or `unknown`; `threat`: `none` or `near`; `health`: `full`, `hurt` or `low` |
| `engage` | `fight` | `threat`: `one` or `many`; `health` |
| `withdraw` | `retreat` | `threat`, `health` |
| `acquire` | `collect` | `threat`, `health` |
| `enter` | `enter` | `threat` |
| `hold` | `wait` | `threat`, `health` |

Helpers: `health_band(current, containers)` (`full` when current >= containers; `low` when current <= 1; otherwise `hurt`) and `threat_band(distances_px, near_px)`. Zelda passes 48 px, the motor's `RETREAT_DISTANCE`.

**Settle rules for Zelda** (in the Hub adapter, from `goal_finished` plus the next decision's observation):

| Skill | success | failure | void |
|---|---|---|---|
| `traverse` | `(level, room)` changed | death; `interrupted:*`; `completed` or `budget_expired` in the same room | `stopped:*` |
| `engage` | `outcome == completed` | death; `budget_expired`; `interrupted:stuck` | `stopped:*` |
| `withdraw` | `completed` | death; `interrupted:*` | `stopped:*` |
| `acquire` | an item count went up, or `item_gained` met | anything else except void | `stopped:*` |
| `enter` | `completed` | anything else except void | `stopped:*` |
| `hold` | no heart lost and no death | a heart lost or death | `stopped:*` |

A `stopped:*` goal (human takeover, pause, lost sponsor) never counts as a failure.

### Prediction service: the action bet

**Table** (new migration `services/orion-sql-db/manual_migration_prediction_action_bet_v1.sql`):
- header `-- ORION-MIGRATION-REQUIRED-BY: orion-prediction`;
- only additive DDL, so it fits the post-pull auto-apply allowlist.

```sql
CREATE TABLE IF NOT EXISTS prediction_action_bet (
  bet_id        text PRIMARY KEY,        -- caller-chosen, idempotent: '<domain>:<session>:<decision_id>'
  domain        text NOT NULL,           -- 'retro:zelda'
  body          text NOT NULL,           -- 'retro' now; 'aitown', 'arm' later
  skill         text NOT NULL,
  context       jsonb NOT NULL,          -- categorical features, keys sorted
  place         text,                    -- e.g. 'zelda:0:119:S', or NULL
  action        jsonb NOT NULL,          -- the body's own goal, for inspection only
  model         text NOT NULL,           -- 'action_beta_backoff'
  p             real NOT NULL,           -- served probability, computed by the service at placing
  p_baseline    real NOT NULL,           -- 'action_skill_rate' at the same moment (the shadow)
  level         text NOT NULL,           -- deepest level with data: 'place' | 'situation' | 'skill' | 'prior'
  counts        jsonb NOT NULL,          -- successes/failures per level at placing
  chosen_rank   smallint,                -- 1..5 on the caller's menu, NULL off-menu
  session_id    text,
  placed_at     timestamptz NOT NULL DEFAULT now(),
  settled_at    timestamptz,
  result        text,                    -- 'success' | 'failure' | 'void'; NULL while open
  outcome       text,                    -- the body's own words (Retro outcome, 'never_settled')
  detail        jsonb
);
CREATE INDEX IF NOT EXISTS prediction_action_bet_skill ON prediction_action_bet (skill, settled_at);
CREATE INDEX IF NOT EXISTS prediction_action_bet_domain ON prediction_action_bet (domain, settled_at);
CREATE INDEX IF NOT EXISTS prediction_action_bet_place ON prediction_action_bet (place) WHERE place IS NOT NULL;
```

**Model** (`services/orion-prediction/app/action_bets.py`):

- `action_beta_backoff`, served. Counts come from settled `success` and `failure` rows (void excluded). With strength k = 4:
  - skill: `Beta(1 + s, 1 + f)`, over all domains;
  - situation: `Beta(k·p_skill + s, k·(1 − p_skill) + f)`, within one domain, for the exact context;
  - place: `Beta(k·p_situation + s, k·(1 − p_situation) + f)`, within one domain.
  - `p` is the mean of the deepest level with at least one settled row. Without data the result is `prior` with p = 0.5. The response also carries an 80 % interval from the Beta quantiles (`scipy.stats.beta`; scipy 1.18.1 is already pinned in `requirements.txt`, and digamma comes from `scipy.special`).
- `action_skill_rate`, the baseline: the skill-level mean only. Stored on every bet as `p_baseline`.
- **Expected learning**, returned with each quote: EIG (expected information gain), the mutual information between the next outcome and the rate at the deepest level. That is H(mean) − E[H(θ)], via `scipy.special.digamma`. Hand-checked: 0.193 nats at `Beta(1,1)`, 0.022 at `Beta(11,11)`, 0.020 at `Beta(21,3)`.

**Grading:** settled bets go through the existing `app/grading.py` `grade(p, success)` for the served model, and again for the baseline, with `enough` at n >= 50. The live verdict uses `app/live.py`'s paired test (served vs baseline squared error per bet, independent by session), with the same `better` / `worse` / `unclear` / `pending` words and the same Holm rule. If `action_beta_backoff` is live-`worse`, quotes serve `action_skill_rate` instead and say so (`model` in the response). This is the service's existing gate, applied to the new model.

**Endpoints:**

| Endpoint | Does |
|---|---|
| `POST /action-bets/quote` | `{domain, body, options: [{option_id, skill, context, place?}]}` → per option `{p, interval80, eig_nats, level, counts, model}`. Read-only, ≤ 20 options, one grouped query. |
| `POST /action-bets` | Place one bet. Body: `{bet_id, domain, body, skill, context, place?, action, chosen_rank?, session_id?}`. The service computes `p` and `p_baseline` itself at placing, so a caller cannot backfill a probability after the fact. Idempotent by `bet_id`: a repeat returns the stored bet. |
| `POST /action-bets/{bet_id}/settle` | `{result, outcome, detail?}`. Only from open to settled. A repeat with the same result returns 200; a conflicting result returns 409. |
| `GET /action-bets/live?domain=&body=` | Posteriors per skill and per situation (counts, mean, interval); the top place beliefs; grading of the served model and the baseline; the live verdict; open and never-settled counts; the last 20 bets. |
| `GET /action-bets?session_id=` | That session's bets, for the memory card. |

`/health` gains `action_bets {open, never_settled_24h, served_model}`. Log lines: `action_bet_placed`, `action_bet_settled`, `action_bet_expired`.

Contract models: `ActionBetQuoteRequestV1`, `ActionBetQuoteV1`, `ActionBetV1` and `ActionBetSettleV1` in `orion/schemas/prediction.py`, next to `PredictionForecastV1`. They are HTTP contracts only. There is no bus channel in v1, because nothing needs a push yet. The self-monitor organ can read `/action-bets/live`, or ask for a channel when `SettledBetV1` ships.

### Hub: the Zelda adapter and the menu (`services/orion-hub/scripts/retro_experience.py`)

**Options** come from each decision:

| Option | Built when |
|---|---|
| `exit`/`explore` toward each open side | always (all 4 sides when `open_sides` is null) |
| `fight nearest` | an enemy is present |
| `collect` at each drop tile | a drop is in the observation |
| `enter` at each known cave | the atlas lists a cave |
| `retreat` | an enemy is near |
| `wait` | always |

Each option maps to a skill, context and place through `orion/embodiment/skills.py`.

**Scoring:**
- One `POST /action-bets/quote` returns the probabilities.
- The Hub scores each option: `p · value + 2 · eig_nats`.
- `value` is a fixed v1 table, labelled as hand-set: new room 1.0, visited room 0.3, drop 0.7, fight with a threat present 0.6, withdraw 0.4 when `hurt` or `low` (0.1 otherwise), enter a known cave 0.5, hold 0.05.
- Learning `value` is out of scope for v1.
- Scoring lives in the Hub, not the prediction service, because value is the body's preference, not a prediction.

**Prompt:** the top 5 options are listed as plain lines, for example:

`1. explore W -> room 118 (never visited). Chance it works 0.71 (2 of 2 here; moving overall 9 of 14). Learning value medium.`

The prompt adds: "Choose one of these unless the facts above show it cannot work." `ZeldaChoice` is unchanged, and the choice is matched to an option by goal equality.

**Place and settle:**
- Place after Retro accepts the goal (`POST /goal` 200), in both co-play and leisure, with `bet_id = 'retro:zelda:<retro session>:<decision_id>'`.
- Settle the previous goal's bet at the next plan call, from `decision.previous` plus the new observation.
- Settle any still-open bet of the session at session end, using the last `goal_finished` row Retro shows, or `void` with outcome `session_ended` if none exists.

**Flag:** `HUB_ACTION_BETS_ENABLED=true` (ON per the flags-on rule) and `HUB_PREDICTION_URL=http://127.0.0.1:8132`. Both go in `.env_example`, `settings.py`, the README, and the local `.env` sync. Flag off restores the #2870 behaviour.

### Retro: learned-blocked tiles heal

In `atlas.observe`: when Link is controllable and standing on tile (c, r), remove learned-blocked entries (c, r) and (c+1, r) for that room and `walls_hash`. Standing there is direct evidence that both tiles are passable. This removes the "one false bump is forever" problem. It is map memory, not a prediction, so it stays in Retro.

### Speaking

At session end, `retro_memory_export` reads `GET /action-bets?session_id=` and `GET /action-bets/live?domain=retro:zelda` and adds at most 3 plain fact lines to the card, for example:

- "Room 119's south side is not a way out (0 of 3 tries)."
- "Moving toward rooms I had not visited worked 5 of 6 times."
- "My guesses about my own moves scored 0.18. Always guessing my average would score 0.24."

With fewer than 10 settled bets, the calibration line is not written. If the prediction service is unreachable, the card is written without these lines, and the export logs it.

## Metric quality gate (`p`, EIG, Brier for action bets)

1. **Provenance:**
   - `p`, `p_baseline` and EIG are computed in `orion-prediction/app/action_bets.py` from settled `prediction_action_bet` rows.
   - Each result is set by the Hub adapter from one Retro `goal_finished` row (`navigate.py:_finish`) plus the next `decision_offered` observation.
   - Brier comes from `app/grading.py` over (`p`, result).
2. **Independence:**
   - The place-level `p` overlaps with `open_sides`, because a closed side is never offered as an exit. Its information is what walls cannot show: failed scrolls, enemies in the way, hidden blockers.
   - The three levels pool the same rows: one signal at three depths, not three signals.
   - EIG is a function of the same counts, used only to rank options, never as evidence.
   - `p_baseline` is deliberately nested in `p` (it is the skill level). It is the reference, not an input.
3. **Theory anchor:**
   - Beta-Bernoulli conjugate updating, with empirical-Bayes shrinkage for the backoff (Efron and Morris).
   - EIG as mutual information (Bayesian experimental design; the epistemic term of expected free energy in active inference).
   - Brier skill against a reference forecast, as `grading.py` already does.
4. **Live sanity** (acceptance check 6, on the first real session's rows):
   - a place with 5 or more failures and no success reads below its situation mean, and below 0.35 at any situation mean
     (with k = 4 it is (4·m)/(4 + 5) = 0.44·m; it reads p < 0.2 only once the situation mean is below 0.45);
   - a skill with 20 or more settled bets has EIG < 0.03 nats;
   - Brier is not reported below 10 bets, and `enough` stays false below 50;
   - the share of void and never-settled bets is reported, so a broken settle path cannot hide as calm.
5. **Existing mechanism:**
   - `prediction_exceedance` is the nearest relative (a stored probability, graded later), but it is keyed to series thresholds and forecast windows.
   - The act ledger (`grammar_acts`) has no probability or context.
   - No action-conditional prediction exists anywhere in the repo.
   - Reused: the `grading.py` scoring, the `live.py` verdict and gate, the shadow-baseline pattern, the lazy-migration pattern, and the memory-card path.
6. **Reversibility:**
   - One table, one module, one Hub flag, four endpoints.
   - Flag off restores #2870.
   - The table is read only by these endpoints, so retiring it is one later `-- DESTRUCTIVE` drop, with no consumer outside this design.

## Files likely to touch

| File | Change |
|---|---|
| `orion/embodiment/skills.py`, `orion/embodiment/tests/test_skills.py` | new: vocabulary, context helpers |
| `orion/schemas/prediction.py` | the four action-bet HTTP contracts |
| `services/orion-prediction/app/action_bets.py` | new: model, baseline, EIG, quote, place, settle, expiry, live |
| `services/orion-prediction/app/store.py`, `app/main.py` | table I/O, endpoints, `/health` |
| `services/orion-prediction/tests/test_action_bets.py` | new |
| `services/orion-prediction/README.md` | an "Action bets" section |
| `services/orion-sql-db/manual_migration_prediction_action_bet_v1.sql` | new |
| `services/orion-hub/scripts/retro_experience.py` | new: Zelda adapter, menu, settle rules, client |
| `services/orion-hub/scripts/retro_planner.py`, `retro_routes.py`, `retro_leisure.py` | menu in the prompt; place and settle |
| `services/orion-hub/scripts/retro_memory_export.py` | learned lines |
| `services/orion-hub/scripts/settings.py`, `.env_example`, `README.md` | flag, URL |
| `services/orion-hub/tests/retro/test_experience.py` | new |
| `services/orion-hub/evals/zelda_experience_replay.py` | new: offline learning eval |
| `services/orion-retro/app/zelda/atlas.py`, `tests/test_zelda.py` | heal learned blocks |
| `docs/runbooks/retro-browser.md` | operator notes |

## Non-goals

- No pixel world model, neural training or deep RL. No new GPU use.
- No learned `value` (reward model) in v1.
- No change to the motor's A* or combat reflex.
- No bus channel for action bets in v1. `SettledBetV1` belongs to #2827.
- The prediction service does not learn game meaning. Skills and settle rules stay in the body's adapter.
- No AI Town or real-motor adapter yet.
- No dungeon place beliefs until `open_sides` is verified in dungeons.

## Acceptance checks

1. **Prediction service unit tests:**
   - backoff with no place data equals the situation mean;
   - no data at all gives `prior`, 0.5;
   - EIG values match the hand-checked numbers above;
   - placing computes `p` server-side, and a repeat is idempotent;
   - settle allows only open → settled, and a conflicting settle returns 409;
   - expiry closes `void` / `never_settled` after 1 h;
   - grading uses `grading.grade`;
   - the gate serves the baseline when the served model is live-`worse`.
2. **Body vocabulary:** the helpers' bands are tested at the edges (hearts = containers, 1 heart, 47/48/49 px).
3. **Adapter:** every settle-table row is tested from a real `goal_finished` shape. `stopped:*` is never a failure. With the service unreachable there is no menu, no bet, and a log line.
4. **Offline learning eval** (`zelda_experience_replay.py`, against a temporary prediction store, no GPU):
   - replay the 2026-10-23 session's evidence file through the adapter;
   - room 119's S place must then read p < 0.25, with W and E above it;
   - the menu for decision 15:9 must not rank "explore S" first.
5. **Live planner eval:** `zelda_planner_replay.py` still passes (5/5 goals, 0 dead moves) with the menu on.
6. **Live session**, after one co-play or leisure session of 10 or more decisions:
   - every planned goal has a placed and settled bet, and nothing is open 60 s after the session ends;
   - the gate-item-4 rest states hold;
   - `/action-bets/live` returns counts and grading;
   - the card shows at least one learned line.
   - Otherwise the result is UNVERIFIED.
7. **Generalization (deferred, named now):** a second body's first quote must show skill-level counts that include Zelda's rows. Until then, cross-body transfer is UNVERIFIED.

## Recommended next patch

One PR, in this order, with tests at each step:

1. `orion/embodiment/skills.py` and the contracts in `orion/schemas/prediction.py`.
2. The prediction service: migration, `action_bets.py`, endpoints, `/health`, tests, README.
3. The Hub: adapter, menu in `plan_zelda`, place and settle in co-play and leisure, flag.
4. Retro: atlas healing.
5. The card lines.
6. The offline replay eval and the live planner eval.

Then deploy (the prediction service first, because the Hub tolerates it being absent), play one session, and run acceptance check 6.

## Later (when to revisit world models and RL)

- **A transition model in the prediction service:** when the ledger shows situations the counts cannot separate (the same features with mixed results again and again), add a small model that predicts the next RAM state from the action. It runs on the CPU, trains overnight, and is shadow-scored against `action_beta_backoff` with the same gate. Not on pixels.
- **A learned value:** after a few hundred settled bets, fit `value` from what led to new rooms and items (a contextual bandit over options), in the Hub.
- **Deep RL from pixels:** only with a GPU budget the heat shed does not cancel, and only after the above shows what to reward.
