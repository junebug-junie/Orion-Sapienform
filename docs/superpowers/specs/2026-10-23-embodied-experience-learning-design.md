# Embodied experience: Orion learns from what its body tries

Status: design, 2026-10-23. Juniper asked for this design before the build.
Builds on: the Zelda play spec (PR #2846), the open-sides planner (#2870, merged), and the self-monitor organ (#2827, design).

## Arsonist summary

Orion plays Zelda but does not learn from it. A language model picks each goal from a text map, and fixed code walks Link there. Every goal comes with a prediction that Retro grades met or refuted, and then nothing reads the grade. On 2026-10-23 Orion chose "explore S" 8 times against a solid wall. #2870 stopped that with hard rules, which are guardrails, not learning.

This design adds learning in Orion's body, not inside the game:

- **Action probabilities:** every action Orion's body tries is a bet with a probability, written before the action and settled after it.
- **Three levels of belief:** a probability per abstract skill (pooled over every game and body), per situation, and per place.
- **Scored options:** fixed code scores the possible moves by chance of working, value and expected learning. The model chooses from that list instead of inventing a move.
- **Calibration:** settled bets grade Orion's own guesses with the Brier score the prediction service already uses.
- **Speaking:** the session's memory card says what Orion learned, so Orion can talk about it in chat.

The skill vocabulary is deliberately small and game-free (`traverse`, `engage`, `withdraw`, `acquire`, `enter`, `hold`). A second game, the AI Town body, or a real motor adds only a small adapter: one function that turns its own goals into skills and context, and one that settles the results. Their beliefs start from what Zelda taught at the skill level.

Not in this design: learned pixel world models (DreamerV3-style) and deep RL. They need millions of frames and GPU training time, and the cabinet heat shed blocks that most days. This design is the counted, Bayesian base those methods would need anyway. Section "Later" says when to revisit them.

## Current architecture

- **Retro motor** (`services/orion-retro/app/zelda/navigate.py`):
  - offers a decision when Link is controllable and idle;
  - runs one goal with A* (a shortest-route search);
  - records `goal_finished {goal, prediction, outcome, verdict, start_digest, end_digest, applied_frames, bumps}`;
  - since #2870, offers `open_sides` (overworld only).
- **Retro atlas** (`app/zelda/atlas.py`): a JSON map per lineage (one saved-game line) holding rooms, exits seen, caves, deaths, and learned-blocked tiles. A blocked tile never expires (one false bump closes it for good).
- **Hub planner** (`services/orion-hub/scripts/retro_planner.py`, `plan_zelda`):
  - builds a prompt from the map, the atlas, open sides and the last 5 results;
  - asks `metacog_background`, validates the answer, and retries once on a refusal.
  - It is called from co-play (`retro_routes.plan_v2`) and solo leisure (`retro_leisure.py`).
- **Session end** (`retro_memory_export.py`): writes one `retro_play_sessions` row and one `memory_cards` row (facts and counts only). Co-play reaches it through `app.state.retro_session_end_sink`; leisure reaches it through the listener.
- **Body package** (`orion/embodiment/`): the AI Town sprite body (intents, arbiter, resolver, perception). It has no learning and no experience store.
- **Prediction service** (`services/orion-prediction`): time-series forecasts of telemetry. It grades threshold probabilities with Brier, base-rate Brier and skill, plus calibration bins. It works on 5-minute buckets and fits models every 6 h.
- **Self-monitor organ** (#2827, docs only): wants a `SettledBetV1` feed from many sources. Today only action claims and prediction-service forecasts can be graded.

The gap: Retro grades every prediction, the grade is shown to the model as text, and then it is thrown away.

## Missing questions (answered here, so the build is unblocked)

1. **Where does learning live?** In `orion/embodiment/experience.py` (pure math, no I/O), with one Postgres ledger table. It does not live in Retro, which holds per-save game state only. It does not live in the prediction service either (see question 2). Any service that drives a body imports the same module and writes to the same table.
2. **Should bets go to the prediction service?** No, not as data. Those bets are yes/no, settle in seconds, and are keyed by actions, not series. The prediction service's engine is built for 5-minute numeric buckets and 6-hour model selection. We reuse its grading rule exactly: Brier, base-rate Brier, skill = 1 − Brier/base, and 10 calibration bins. The self-monitor organ can then read both sources the same way. If the organ ships `SettledBetV1`, this ledger becomes one more adapter for it, not a new contract.
3. **Who chooses the move?** Still the model. It chooses from a scored list of up to 5 options built by code. An answer that is not on the list is still accepted if it passes the #2870 rules, and it is recorded with its prior probability. That keeps Orion's own judgment in the loop while stopping it from inventing spatial facts.
4. **What counts as success for each skill?** It is defined once, from Retro's own record (table below). A goal stopped by a human, a pause or a lost sponsor is void, never a failure.
5. **What carries over between games?** The skill level and the situation level do. Situation features are abstract (side state, threat, health band), never Zelda room ids. The place level never carries over.

## Proposed schema / API changes

### Skills and context (game-free)

| Skill | Zelda goals | Context features (all categorical) |
|---|---|---|
| `traverse` | `exit`, `explore` | `side`: `open_unvisited`, `open_visited`, `closed` or `unknown`; `threat`: `none` or `near`; `health`: `full`, `hurt` or `low` |
| `engage` | `fight` | `threat`: `one` or `many`; `health` |
| `withdraw` | `retreat` | `threat`, `health` |
| `acquire` | `collect` | `threat`, `health` |
| `enter` | `enter` | `threat` |
| `hold` | `wait` | `threat`, `health` |

`health`: `full` = current ≥ containers; `low` = current ≤ 1; otherwise `hurt`. `threat`: `near` = any enemy within 48 px (the motor's `RETREAT_DISTANCE`).

**Place key** (domain-specific, optional). Only `traverse` has one in v1: `zelda:<level>:<room>:<side>`. It is the Bayesian belief "this side of this room leads somewhere".

**Success, settled from the finished goal plus the next decision's observation:**

| Skill | success | failure | void |
|---|---|---|---|
| `traverse` | the next observation's `(level, room)` differs from the start | death; `interrupted:*`; `completed` or `budget_expired` in the same room | `stopped:*` |
| `engage` | `outcome == completed` (the target is gone) | death; `budget_expired`; `interrupted:stuck` | `stopped:*` |
| `withdraw` | `completed` | death; `interrupted:*` | `stopped:*` |
| `acquire` | `completed` and `verdict == met` for `item_gained`, or an item count went up | anything else except void | `stopped:*` |
| `enter` | `completed` | anything else except void | `stopped:*` |
| `hold` | no hearts lost and no death | a heart lost or death | `stopped:*` |

### Belief math (`orion/embodiment/experience.py`)

- Each level is a Beta posterior over a success rate. Counts come from settled ledger rows.
- **Empirical-Bayes backoff with strength k = 4:**
  - skill: `Beta(1 + s, 1 + f)`, pooled over every domain;
  - situation: `Beta(k·p_skill + s, k·(1 − p_skill) + f)`, within one domain;
  - place: `Beta(k·p_situation + s, k·(1 − p_situation) + f)`.
  - An option's probability `p` is the mean of the deepest level that applies.
- **Expected learning:** EIG (expected information gain) of one more try = mutual information between the next outcome and the rate. That is H(mean) − E[H(θ)], using the closed form with digamma. It is 0.193 nats for `Beta(1,1)` and falls toward 0 as counts grow. Digamma is implemented in-module (asymptotic series), so no SciPy is needed.
- **Score** = `p · value + λ · EIG`, with λ = 2.
- `value` is a fixed v1 table per option: new room 1.0, visited room 0.3, drop 0.7, fight with a threat present 0.6, withdraw 0.4 when `hurt` or `low` (0.1 otherwise), enter a known cave 0.5, hold 0.05. The utilities are hand-set and named as such. The learned parts are `p` and EIG. Learning `value` (a reward model) is out of scope for v1.
- **Calibration** over settled rows that have `predicted_p`: Brier, base-rate Brier (the mean success rate as a constant guess), skill, and 10 bins. These are the prediction service's definitions.

### Option menu (Hub, Zelda adapter `services/orion-hub/scripts/retro_experience.py`)

The adapter builds options from the decision:

| Option | Built when |
|---|---|
| `exit`/`explore` toward each open side | always (all 4 sides when `open_sides` is null) |
| `fight nearest` | an enemy is present |
| `collect` at each drop tile | a drop is in the observation |
| `enter` at each known cave tile | the atlas lists a cave |
| `retreat` | an enemy is near |
| `wait` | always |

It scores them, sorts them, and puts the top 5 in the prompt as plain lines, for example:

`1. explore W -> room 118 (never visited). Chance it works 0.71 (2 of 2 here; traverse overall 9 of 14). Learning value medium.`

The prompt says "choose one of these unless the facts above show it cannot work". `ZeldaChoice` is unchanged. The chosen goal is matched to an option by goal equality, which gives its `p`.

### Ledger table (new migration, additive)

`services/orion-sql-db/manual_migration_embodied_experience_v1.sql`, header `-- ORION-MIGRATION-REQUIRED-BY: orion-hub`. Only additive DDL (`CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS`), so it fits the post-pull auto-apply allowlist. The Hub also creates it lazily, the same way as `retro_play_sessions`.

```sql
CREATE TABLE IF NOT EXISTS embodied_experience (
  bet_id       text PRIMARY KEY,        -- '<domain>:<retro session>:<decision_id>'
  domain       text NOT NULL,           -- 'retro:zelda'
  body         text NOT NULL,           -- 'retro' now; 'aitown', 'arm' later
  skill        text NOT NULL,
  context      jsonb NOT NULL,          -- the categorical features above
  place        text,                    -- place key or NULL
  goal         jsonb NOT NULL,          -- the domain's own goal, for inspection
  predicted_p  real,                    -- NULL when the goal was not scored (e.g. no menu)
  chosen_rank  smallint,                -- 1..5 on the menu, NULL off-menu
  session_id   text,
  placed_at    timestamptz NOT NULL DEFAULT now(),
  settled_at   timestamptz,
  result       text,                    -- 'success' | 'failure' | 'void'; NULL while open
  outcome      text, verdict text,      -- Retro's own words, copied
  detail       jsonb
);
CREATE INDEX IF NOT EXISTS embodied_experience_skill ON embodied_experience (skill, settled_at);
CREATE INDEX IF NOT EXISTS embodied_experience_place ON embodied_experience (place) WHERE place IS NOT NULL;
```

- **Placed:** after Retro accepts the goal (`POST /goal` 200), one `INSERT ... ON CONFLICT DO NOTHING`.
- **Settled:** at the next plan call. `decision.previous` plus the new observation settle that row (`UPDATE ... WHERE result IS NULL`). Both writes are idempotent by `bet_id`.
- **Read at plan time:** one grouped query (counts per skill, per skill + context, per place) for the domain, plus skill counts over all domains. On a DB error the menu uses `Beta(1,1)` priors, and the prompt says "experience store unavailable". It never claims counts it does not have.

### Retro: learned-blocked tiles can heal

In `atlas.observe`: when Link is controllable and standing on tile (c, r), remove learned-blocked entries (c, r) and (c+1, r) for that room and `walls_hash`. Standing there proves both tiles passable. This removes the "one false bump is forever" problem, with the smallest Bayesian move (direct evidence wins).

### Debug surface

- `GET /api/retro/experience?domain=retro:zelda` (Hub, same auth as the Retro proxy) returns:
  - skill and situation posteriors (counts, mean, 80 % interval);
  - the top place beliefs;
  - calibration (Brier, base, skill, bins);
  - the last 20 bets.
- Log lines: `embodied_bet_placed`, `embodied_bet_settled`, `embodied_store_unavailable`.

### Speaking

`retro_memory_export.build_session_summary` gets an optional `learned` block from the session's settled bets. The card text gains at most 3 plain lines, for example:

- "Room 119's south side is not a way out (0 of 3 tries)."
- "Moving toward rooms I had not visited worked 5 of 6 times."
- "My guesses about my own moves scored 0.18 (always guessing the average scores 0.24)."

They are facts from the ledger only, with no generated prose. The card is recallable in chat like every `retro_play` card.

### Env/config

- `HUB_EMBODIED_EXPERIENCE_ENABLED=true`: the kill switch, ON by default per the flags-on rule. Off means no menu, no writes, and today's #2870 prompt.
- Added to `services/orion-hub/.env_example`, `settings.py`, the README, and the local `.env` sync.

No new bus channel in v1. The ledger table is the event store. The self-monitor organ can read it, or wrap it in `SettledBetV1` when that contract ships.

## Metric quality gate (for `p`, EIG and calibration)

1. **Provenance:**
   - `p` and EIG are computed from the ledger counts by `orion/embodiment/experience.py`;
   - each count comes from one `goal_finished` row (Retro `navigate.py:_finish`) plus the next `decision_offered` observation;
   - calibration comes from `predicted_p` (written at place time) and `result`.
2. **Independence:**
   - `p` at the place level is not independent of `open_sides`: a closed side is never offered as an exit. It is offered only for open sides, where it measures what walls cannot show (a scroll that fails, an enemy in the way).
   - Situation and skill `p` pool the same rows, so they are one signal at three depths, not three signals.
   - EIG is a function of the same counts. It ranks options; it is never a second piece of evidence.
3. **Theory anchor:**
   - Beta-Bernoulli conjugate updating, with empirical-Bayes shrinkage for the backoff (Efron and Morris).
   - EIG as mutual information is the standard Bayesian experimental-design and active-learning criterion. It is also the epistemic term of active inference's expected free energy.
   - Brier skill against the base rate is the prediction service's rule.
4. **Live sanity:** needs data first. Acceptance check 5 pulls the first live session's ledger and checks the rest states:
   - a place with 5 or more failures and no successes must read p < 0.2;
   - EIG for a skill with 20 or more settled bets must be below 0.03 nats;
   - Brier must not be computed from fewer than 10 settled bets (it reads `insufficient` instead).
5. **Existing mechanism:**
   - There is no experience or belief store in `orion/embodiment/`.
   - The act ledger (`grammar_acts`) grades acts as upheld or abandoned, with no probability and no context, and its keys differ from dispatch cells (#2827 audit).
   - The prediction service is time-series only.
   - Reuse: Brier and skill definitions from the prediction service, the lazy-table pattern from `retro_memory_export`, and the card path from the memory export.
6. **Reversibility:**
   - One table, one module, one flag. Flag off restores #2870 behaviour.
   - The table is additive and read only by this design's code, so dropping it later is a manual `-- DESTRUCTIVE` migration that nothing else depends on.

## Files likely to touch

| File | Change |
|---|---|
| `orion/embodiment/experience.py` | new: Beta, digamma, EIG, backoff, scoring, calibration (pure) |
| `orion/embodiment/tests/test_experience.py` | new |
| `services/orion-hub/scripts/retro_experience.py` | new: Zelda adapter (options, skills, context, settle rules) and ledger I/O |
| `services/orion-hub/scripts/retro_planner.py` | menu in the prompt; match the choice to an option; `experience` hook in `plan_zelda` |
| `services/orion-hub/scripts/retro_routes.py`, `scripts/retro_leisure.py` | place the bet after Retro accepts the goal |
| `services/orion-hub/scripts/main.py` | give the experience store the memory pool |
| `services/orion-hub/scripts/retro_memory_export.py` | the `learned` lines on the card |
| `services/orion-hub/scripts/settings.py`, `.env_example`, `README.md` | the flag |
| `services/orion-hub/tests/retro/test_experience.py` | new |
| `services/orion-hub/evals/zelda_experience_replay.py` | new: offline learning eval (below) |
| `services/orion-sql-db/manual_migration_embodied_experience_v1.sql` | new |
| `services/orion-retro/app/zelda/atlas.py`, `tests/test_zelda.py` | heal learned blocks |
| `docs/runbooks/retro-browser.md` | operator notes |

## Non-goals

- No pixel world model, no neural network training, no deep RL, no GPU use beyond today's one planner call.
- No learned `value` (reward model) in v1.
- No change to the motor's A* or combat reflex.
- No new bus channel or schema registry entry. `SettledBetV1` belongs to #2827.
- Bets are not sent to the prediction service.
- No adapter for AI Town or a real motor yet. The skill vocabulary is designed for them, but building them is separate work.
- No dungeon place beliefs until `open_sides` is verified in dungeons (#2870 concern).

## Acceptance checks

1. **Unit:** the Beta, backoff, EIG and Brier values match hand-computed cases:
   - EIG(1,1) = 0.193 nats;
   - EIG falls as counts grow;
   - backoff with no place data equals the situation mean;
   - Brier and skill match the prediction service's definition on a fixed table.
2. **Adapter:** every success/failure/void row in the settle table has a test built from a real `goal_finished` shape. `stopped:*` is never a failure.
3. **Offline learning eval** (`zelda_experience_replay.py`, no GPU):
   - replay the 2026-10-23 session's evidence file through the adapter;
   - after the replay, room 119's S place belief must read p < 0.25 and W/E stay above it;
   - the menu built for decision 15:9 must not rank "explore S" first.
   - The old code has no beliefs, so this check is new.
4. **Live planner eval:** `zelda_planner_replay.py` still passes (5/5 goals, 0 dead moves) with the menu on.
5. **Live session:** after one co-play or leisure session of 10 or more decisions:
   - the ledger holds a placed and settled row for every planned goal (none open longer than 60 s after the session ends);
   - the rest-state checks in gate item 4 hold;
   - the memory card shows at least one `learned` line;
   - `GET /api/retro/experience` returns the counts.
   - If any of these fails, the result is UNVERIFIED.
6. **Generalization proof (deferred, named now):** the first second-game adapter must start with skill-level `p` taken from Zelda's rows, shown in its first menu. Until a second body exists, cross-body transfer is UNVERIFIED and must be reported that way.

## Recommended next patch

One PR, in this order, with tests at each step:

1. `orion/embodiment/experience.py` and its tests.
2. The migration and the Hub ledger I/O.
3. The Zelda adapter, the menu in `plan_zelda`, and bet placing in co-play and leisure.
4. Atlas healing in Retro.
5. The card lines and the debug endpoint.
6. The offline replay eval and the live planner eval.

Then deploy, play one session, and run acceptance check 5.

## Later (when to revisit world models and RL)

- **Learned dynamics model:** once the ledger shows a context the counts cannot separate, meaning the same situation features give mixed results again and again. A small learned model predicting the next RAM state from the action (CPU, trained offline overnight) is the next step. Training on pixels is not.
- **Learned value:** once there are a few hundred settled bets, fit `value` from the outcomes that led to new rooms and items (a contextual bandit over options).
- **Deep RL from pixels:** only with a GPU budget that the heat shed does not cancel, and only after the steps above show what to reward.
