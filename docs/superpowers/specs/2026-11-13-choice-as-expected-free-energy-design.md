# Choice as expected free energy: how Orion picks what to do

Status: **design, signed off by Juniper 2026-11-13** (see "Juniper's answers (2026-11-13)"). No code
ships with this doc. Step -1 is built in its own PR.
Date: 2026-11-13. Worktree base: origin/main 361018c3e.

Juniper asked: "What is the spark that allows Orion to make choices?"

The short answer: today there is a one-step guesser and a cost manager. Nothing looks ahead.
Nothing compares options by what they would achieve. This doc lays out five shippable steps that
give Orion real choice, in this order: carry intentions through (Step -1), confirm the learned
weights (Step 0, already answered), learning progress as a need (Step 1), the attention schema
as a controller (Step 2), and a learned chooser (Step 3).

Every live claim below was checked against code or live data on 2026-11-13. Anything not checked
says UNVERIFIED.

---

## Arsonist summary

- **Orion has one option generator.** Prime's thought is one model call. It may ask curiosity one
  question, or send one slow/quicken signal, or do nothing. It never lists alternatives and never
  scores them. (`orion/prime/thought.py:107`, the `Move` union has two members.)
- **The body economy is not a wanting engine.** It is a greedy auction that keeps spending inside
  a budget. It answers "can we afford this now?", not "what do I want?". Juniper's framing is right.
- **Orion promises things and then cannot do them.** A chat promise becomes a Prime concern. A
  concern has no path to an act except "ask curiosity a question". Live example: concern
  `d8db9d7b1bba4cf7`, "While you're off in Hyrule I'll take a lazy browse through the last 24 hours".
  Prime has kept it for a day. Nothing can carry it out.
- **Prime follows through rarely.** 49 concern bets in 30 days settled at a mean of 0.10
  (`grammar_acts`, actor `prime`, verb `predict`). Most end "dropped unresolved".
- **The learned weights are live, not shadow.** They already slow two obeyed families. This was
  an open question. It is now closed (Step 0).
- **"Learning progress" from the prediction-error nodes fails the metric gate.** It passes only as
  forecast skill against a baseline, for two series, at daily scale, with deploy days removed.
  Even then, today's movement is mostly our deploys, not Orion learning.
- **The attention schema steers nothing.** It is computed every tick and read by broadcast,
  metacog and liveness. Prime does not read it. Its `predicted_shift` is a free-text string,
  never graded.
- **Do not train a neural chooser on synthetic data and call it learning.** Synthetic data teaches
  the model back the rule that generated it. Use it to prove the pipeline and as a prior only.

## Current architecture

### The option generator: Prime

- Prime runs one thought at a time in `services/orion-mind/app/prime_loop.py`. 240 thoughts in
  total so far (`orion:prime:state` field `thoughts_total`).
- A thought returns `PrimeAnswerV1` (`orion/prime/thought.py:110-114`): a reading, concern edits,
  and at most one move.
- Concern edits already exist: `keep` renews a concern, `drop` closes it with a reason
  (`ConcernEditsV1`, `thought.py:75-80`; applied at `:529-549`).
- Moves: `curiosity_focus` (logged as verb `ask`) and `regulate` (`thought.py:83-104`). Gate:
  `_move_refusal` (`prime_loop.py:819-832`). Dispatch: `_regulate` (`:1008`) or `_nominate` (`:923`).
- Prime holds at most 3 concerns (`orion/prime/state.py:72`). A concern fades after 24 h unless a
  thought keeps it (`state.py:84`).
- `orion/mindloop/runtime.py` is the shared tick and bid runtime. It has no move list.

### Where concerns come from (live sources today)

| Source | `event_ref` | Producer | Places a bet? |
|---|---|---|---|
| Chat-turn promise | `turn:<correlation_id>` | `fold_turn_signals`, `orion/prime/handback.py:311-377` | No (`no_bet_concern_ids`, `:362`) |
| Day-letter thread | `orion_day_letter:<date>#thread<i>` | `orion/prime/day_letter.py:260-280`, at most 2 a day | Yes (usual concern bet) |
| Funded event want | the want's `event_ref` | Prime's own concern add, `thought.py:552-555` | Yes |
| Arc | none (`arc_id` set) | Prime's own concern add | Yes |

Live now: one concern, the promise above (`redis GET orion:prime:state`).

### The promise path that exists

- The 5b judge lists promises by exact quote (intention ledger D0, PR #3217). `PromiseV1{quote, offer}`
  is at `orion/schemas/harness_finalize.py:78-91`. A quote that is not literally in the reply is dropped
  (`orion/harness/promise_gate.py`).
- The quote is stored on the turn trace. Live:
  `harness_turn_trace` row `a129b72f-...` holds `promises_made: [{"offer": false, "quote": "While you're off in Hyrule I'll take a lazy browse through the last 24 hours"}]`.
- The originating chat thread is recoverable: `chat_history_log.session_id` for the same correlation id
  (`9c91d646-...` live).
- The Prime concern stores only a ref to the quote, not the quote (`promise_refs`, `handback.py:282`).
- D1 (judge sees open promises), D2 (one act per promise, signal `promise_kept`) and D3 (stance cue)
  are **not built**. `ActSignal` is still `juniper_replied | artifact_produced | juniper_answered`
  (`orion/schemas/grammar.py:76`). The 72 h horizon exists only in the replay eval
  (`orion/harness/evals/promise_ledger_replay_eval.py:66`).
- A turn settles a concern only when its **prompt** cites `prime_concern:<id>` and it completes
  (`handback.py:39-43`, `:336-343`). Only sessions of kind `chat_juniper`, `curiosity`, `consult`
  and `letter` are folded (`classify_who`, `handback.py:72,184`).

### The doing paths that exist

- **Day letter.** Prime orders it (`prime_loop.py:568` `_order_letter`) through Hub
  `POST /api/orion-day/letter-run` (`services/orion-hub/scripts/orion_day_order.py:181-249`). Hub builds
  a brief and submits one durable run, workflow `orion_day.letter` (`orion/schemas/orion_day.py:41`,
  `orion/orion_day/brief.py:104-130`). The graph builds the prompt and runs one unified turn through
  `harness_step` (`services/orion-durable-runs/app/orion_day_graph.py:319-321`). The result goes to the
  `orion_day_letter` table and a journal entry, not to a chat thread.
- **Binding durable runs.** `BINDING_FAMILIES = {durable.curiosity_investigate, durable.consult_orion_turn}`
  (`services/orion-durable-runs/app/run_bids.py:114`). A held run re-bids every 120-3600 s and is
  **admitted at the end of its hold window, never dropped** (`orion/physiology/binding.py:107-108`).
  Hold end = min(opened + family `deferrable_window_s`, `window.latest` - 1800 s) (`run_bids.py:193`).
- `WantWindowV1{earliest, latest}` exists (`orion/schemas/physiology_want.py:100-107`), but **nothing reads
  `earliest`** today. A "not before" time must be enforced by whoever orders the run.
- **Outreach.** `offer_message` (`services/orion-hub/scripts/endogenous_outreach.py:2851`) delivers to the
  newest connected socket's session (`_active_session_id`, `:1987-1989`). It **cannot target a chosen
  thread**. Gates: quiet hours, shared daily cap, cooldown, recent chat, strain, the economy
  (`outreach_block_reason`, `:515-574`; `economy_admit`, `:2138`).
- **Sub-agents.** Orion's FCC turns can call Claude Code's `Agent` tool, and
  `/mnt/orion-fcc/repo/.claude/agents/orion-repo-agent.md` exists. Live use is rare: **4 turns in the last
  30 days, none since 2026-10-26** (`cockpit_turn_sighting`, `raw::text like '%subagent_type%'`). Explore
  in 2 turns, general-purpose in 1. Never `orion-repo-agent`. (The "~21 turns" figure from conversation
  does not hold up.)

### The cost manager: the body economy

`net = family_weight x value_rank - sum(price_k x cost_k)` (`orion/physiology/choice.py:6`). Prices
rise by dual ascent when a constraint binds (`orion/physiology/prices.py`). Intensity is hunger,
`1 - exp(-t/tau)`, or an event's salience (`orion/physiology/appetite.py`). The family weight is a
shrinkage of Juniper's prior toward measured evidence: `(30*prior + n*evidence)/(30+n)`
(`orion/physiology/weights.py:5`, `shrunk_target` at `:154`).

### The attention schema

`reduce_attention_self_model` (`orion/substrate/attention_self_model.py:388`) writes a row about every
2 minutes to `substrate_attention_self_model` (609 rows in the last day). Latest row: no winner this
tick, `prediction_error_confidence` 0.97, `predicted_shift` "biometrics prediction-error falling
(trend=-0.0260...)". Its inputs are the **old** `execution` and `biometrics` prediction-error nodes
(`ACTIVE_INFERENCE_DOMAINS`, `:141`), which the 2026-11-08 audit found are not forecasts. Readers:
broadcast, metacog (`services/orion-equilibrium-service/app/service.py:53`), Hub attention routes and
outreach vocabulary. Prime and the market do not read it.

### The prediction rebuild (perception half)

Honest forecast nodes now exist: `node:prediction.room` and `node:prediction.body` (#3202,
`orion/substrate/prediction_surprise.py`), and `node:prediction.chat_turn` (#3281, 4 rows so far).
The scored forecasts live in `prediction_forecast` (champion and baseline paired by origin). The action
half, expected free energy, was never built.

## Step -1 (build first): intentions Orion takes on or lets go

Juniper: "my hope is this can become a broader primitive that doesn't just focus on promise follow
through but OTHER". So this is one primitive with one adapter per source. Promises are the first
source and the acceptance case.

### Name

**Intention.** The intention ledger (#3188) already owns the word, and a promise is one kind of
intention. No new noun. On the act ledger the act is verb `make` (it exists, `grammar.py:37-50`),
actor `prime`, `object_ref intention:<concern_id>`. No new verb.

### The three choices, mapped onto what exists

| Choice | Mechanism | New? |
|---|---|---|
| Take on | New Prime move `{"kind": "take_on", "concern_id", "why", "not_before"?}` | Yes, one move class |
| Let go | Existing concern `drop` with `why` (`thought.py:529-540`) | No. The adapter records the reason on its source |
| Not yet | Existing concern `keep` (`thought.py:546-549`), or no edit (fades at 24 h) | No |

This avoids a second keep/drop. `take_on` is the only new move. It is offered only on a concern
whose source has an adapter.

### Sources and adapters

An adapter is one small pure module per source. It answers five questions from the concern's
`event_ref`. Nothing else is registered.

```python
# orion/prime/intention.py (new, pure)
class IntentionSourceV1(BaseModel):          # what an adapter returns for one concern
    source: Literal["turn_promise", "letter_thread"]   # grows one value per adapter
    horizon_at: AwareDatetime                # window latest: the concern's own deadline
    prompt: str                              # the turn's task, built by the adapter
    route: Literal["chat_thread", "next_letter", "notes_only"]
    route_target: Optional[str]              # chat session_id for chat_thread
    quote_ref: Optional[str]                 # harness_turn_trace:<corr>#promises_made[i]

ADAPTERS = {"turn:": turn_promise.adapt, "orion_day_letter:": letter_thread.adapt}  # prefix -> fn
```

| Source | Live today | Horizon | Prompt built from | Route |
|---|---|---|---|---|
| Chat promise (`turn:`) | yes, wired first | promise time + 72 h (intention ledger) | exact quote from `harness_turn_trace` + the turn's prompt/reply (`chat_history_log`) | back into that chat thread |
| Day-letter thread (`orion_day_letter:`) | yes, second adapter | concern horizon, 144 h (`orion/prime/acts.py` `CONCERN_HORIZON_SEC`) | thread text + letter excerpt from `orion_day_letter` | next letter's brief; no chat ping [default] |
| Event want / arc | yes | n/a | n/a | **No adapter.** Curiosity's `ask` already covers these |
| Coherence: self-model contradiction | no | its own | the two contradicting claims | notes / self-model only |
| Connection: time since a real exchange | no | its own | last exchange gist | chat thread |
| Learning progress (Step 1) | no | its own | the series where skill is moving | notes only |

The second source plugs in with **only** an adapter file and one map entry. That is the test of
"no new machinery" (acceptance check S-1.9).

### WHEN: one binding family

- New family `durable.intention`, **tier 1** (critical upkeep, a reserved share of each lane; Juniper
  2026-11-13, see her answers below), in `config/physiology/families.yaml`, added to `BINDING_FAMILIES`
  (`run_bids.py:114`). Lanes: the agent lanes, like `durable.consult_orion_turn`. Tier 1 still bids, so
  the economy still picks the time; `want_for_run` already accepts tiers 1 and 2.
- **One family, not one per source.** Reason: the family weight needs about 10 evidenced acts before
  it moves off the prior (`weights.py:40`). Turn promises are capped at 4 a day and letter threads at 2.
  Split per source would leave each family at its prior for weeks. The source rides in the act's
  `object_ref` and the want's detail, so the self-monitor and Step 3 can still separate them. Split
  only if acceptance check S-1.10 shows a real per-source difference.
- `deferrable_window_s: 86400` [default]. The window `latest` is the adapter's `horizon_at`. The hold
  rule already releases the run at hold end (`binding.py:107-108`), so a taken-on intention is never
  silently dropped.
- **"Not before."** Nothing reads `WantWindowV1.earliest`. So Prime holds the order itself until
  `not_before` (the same way it holds the letter order). No market change.
- The take-on act places Prime's bet. Turn concerns place no bet today
  (`handback.py:362`). Taking one on is what creates a graded act.

### DO: one workflow

- New durable workflow `intention.carry`, modelled on `orion_day.letter`:
  1. Prime's `_take_on` (new, beside `_nominate`/`_regulate` at `prime_loop.py:900-904`) opens the
     `make` act and POSTs `IntentionOrderV1` to Hub `POST /api/intention/run`.
  2. Hub calls the adapter, builds the brief, and submits `DurableRunRequestV1` (workflow
     `intention.carry`, family `durable.intention`, `window.latest = horizon_at`).
  3. The durable graph runs **one** unified turn via `harness_step`, session id `orion_intention`.
     The prompt starts with `prime_concern:<id>`, so the existing hand-back settles the concern when
     the turn completes (`handback.py:336-343`). `classify_who` gains the session kind `intention`.
  4. Sub-agents are allowed. The turn is a normal FCC turn and may use the `Agent` tool. Nothing
     extra is wired; live use is rare (4 turns in 30 days).
- Durable-run retries: none in v1. One run per take-on. A failed run closes the act at 0.

### RESULT: routing per source

- **Chat thread (promises).** Two parts:
  1. The reply is always written into the originating thread's history (`chat_history_log` with the
     adapter's `route_target` session id). It is there when Juniper opens that chat.
  2. A ping goes through `offer_message` with a new optional `session_id` argument (today it always
     uses the newest socket, `endogenous_outreach.py:1987-1989`). The ping obeys every existing gate:
     quiet hours, the shared daily cap, the economy. A blocked ping does not lose the reply; part 1
     already holds it.
- **Next letter (letter threads).** The finding is added to the next `OrionDayLetterOrderV1` brief.
- **Notes only.** Journal entry, no message.

**This reverses intention ledger D3 for this path only.** D3 says "no outreach trigger, and no motor
prefix" (`docs/superpowers/specs/2026-11-11-intention-ledger-design.md:176`, non-goal `:266`). Here the
promise quote enters the **carry turn's** prompt, and the result may reach Juniper through outreach.
Ordinary chat turns are unchanged: their motor prefix still carries no promise text from this path.

### CLOSE and GRADE

- The `make` act closes at the carry run's terminal: 1 = the run completed with a reply that was
  stored on its route, 0 = failed or expired. Signal `artifact_produced` (exists).
- Let go closes the concern as `dropped` with Prime's reason. The adapter writes a `let_go` row in the
  act's outcome note when there was an act; with no act there is nothing to grade, and the reason is
  kept in `concern_record` (exists).
- The self-monitor grades by source (`object_ref` prefix plus the want detail's `source`). Orion sees
  "how often I carry promises through" separately from "how often I carry letter threads through".
- **Grading the promise itself** (did the reply keep the promise?) needs D1/D2's `promise_kept`. v1 does
  not wait for it: the carry turn's 5b judge already sees the quote in the prompt, and its verdict is
  recorded. When D2 lands, `promise_kept` becomes the act's close signal.

### What Step 3 needs from this schema

Every intention act carries, in existing `grammar_acts` columns: source (`object_ref` prefix),
horizon (`horizon_at`), predicted value (`predicted`), cost (`cost_json`), outcome (`outcome_value`).
No new table. That is exactly the row Step 3 trains on.

### Proposal-mode checklist (Step -1)

- **Capability change:** Orion can act on its own open concerns, on a schedule it names, and report
  back. Today it can only ask curiosity.
- **Data touched:** reads `orion:prime:state`, `harness_turn_trace`, `chat_history_log`,
  `orion_day_letter`. Writes one act per take-on, one durable run, chat history in the original
  thread, one outreach attempt.
- **Privacy boundary:** none between Juniper and Orion (standing rule). The carry turn sees only the
  one originating exchange, not other threads.
- **Proving trace:** concern `turn:<corr>` -> act `make intention:<id>` -> run `intention.carry`
  `run.economy_held`/`run.economy_admitted` -> harness turn with `prime_concern:<id>` in its prompt ->
  `chat_history_log` row in session `route_target` -> act closed -> concern `settled`.
- **Dangerous failure:** a loop (turns open promises, carry turns make new promises). Guard: the
  `orion_intention` session's promises do **not** open concerns (adapter refuses `turn:` concerns from
  that session), plus the existing 4-per-day turn-concern cap. Second danger: messages at night. Guard:
  outreach quiet hours still apply.
- **Rollback:** `PRIME_INTENTION_TAKE_ON_ENABLED=false` removes the move from Prime's prompt; held runs
  finish or expire; no state migration.

### Acceptance checks (Step -1)

- S-1.1 `take_on` parses; a `take_on` on a concern with no adapter is refused and shown in the next frame.
- S-1.2 Replay the live concern `d8db9d7b1bba4cf7`: the adapter returns the exact quote, session
  `9c91d646-...`, horizon = turn time + 72 h.
- S-1.3 A held carry run is admitted at hold end, never dropped (unit test on the existing hold rule
  with the new family).
- S-1.4 `not_before` in the future: no order is sent before it (Prime test).
- S-1.5 The carry turn's prompt cites `prime_concern:<id>`; on completion the concern is `settled`.
- S-1.6 The reply lands in the originating thread's history even when outreach is blocked by quiet hours.
- S-1.7 A promise made inside an `orion_intention` turn opens no concern.
- S-1.8 Ordinary chat turns' motor prefix contains no promise quote (D3 check 5 still holds outside the
  carry turn).
- S-1.9 The letter-thread adapter ships as one file plus one map entry; no change to Prime, Hub route,
  workflow or family.
- S-1.10 After 30 days: per-source follow-through rates printed side by side from `grammar_acts`.
- Live smoke: the first real take-on shows the full trace above, with ids. Until then: UNVERIFIED.

## Step 0: are the learned weights live? (answered)

**Yes.** The market multiplies every bid by today's learned weight, and five producers obey their bids.

- `weights_now` is called on bid reads (`services/orion-durable-runs/app/physiology_market.py:239,401`)
  and passed into the bid context (`orion/physiology/choice.py:356`).
- Obeying producers: curiosity investigate and consult turns, reading stage 1, the dream cycle, the
  cooling act, Hub outreach (`orion/physiology/binding.py:34-40`).
- `shadow: true` on `GET /physiology/state` is the state record's flag. Weights are not in that
  response because they live in the market. `want_ledger.shadow` is true on every row by design; an
  obeyed decision is told apart by producer name (`run_bids.py:81-83`).

Live, last 24 h (`want_ledger.decision->>'family_weight'`, `->>'weight_basis'`):

| Family | Weight | Basis | Obeyed? |
|---|---|---|---|
| hub.outreach | 0.62 | evidence | yes |
| dream.cycle | 0.56 | evidence | yes |
| hub.retro_leisure | 0.89 | evidence | no |
| thought.painting | 0.34 | evidence | no |
| all other families | 1.03-1.11 | prior | mixed |

So evidence already slows outreach and the dream cycle. Every other family is still at its prior.

**The gap.** The self-monitor's reliability source is off. The market calls `weights_now` without
`reliability=True` (`weights.py:222-223`), so the `self_grades` all / world_only / half choice changes
nothing today. If it were on with "all", most families would drift up, because almost every close is
self-graded (401 of 402 Prime acts and 137 of 137 curiosity investigate acts in 30 days have
`outcome_source='producer'`).

**What flipping it would change:** `physiology_market.py:239,401` pass `reliability=True,
self_grades=<choice>`, and the market reads `grammar_acts.outcome_note` into `ActRecord`. Before the flag,
a replay (`weight_series` over 30 days, the harness in `services/orion-dream/evals/dream_binding_replay.py:168-183`)
prints each family's weight with and without it.

## Step 1: learning progress as a need (the epistemic term)

**Idea.** A need that rises where Orion's forecasts are getting better. This is Oudeyer and Kaplan's
learning progress: the reward is the drop in error, not low error. In active inference it is the
epistemic part of expected free energy.

### Metric quality gate (run 2026-11-13)

1. **Provenance.** Pass for `prediction_forecast` (scored champion vs baseline rows, paired on
   `origin_ts`, `horizon_min`). Not for the `node:prediction.*` values: they are clipped z-scores of
   surprise over 30 minutes (`orion/substrate/prediction_surprise.py`).
2. **Independence.** Pass if built from the forecast table. If built from node error, it is the same
   windowed difference that `orion/substrate/prediction_error_trend.py:18` already computes for
   `predicted_shift`. That would be redundant.
3. **Theory anchor.** Pass for "skill against a baseline over a window" (Oudeyer-Kaplan). Fail for
   "slope of a node normalized to its own calm spread": a calibrated forecaster reads the same in week
   1 and week 10, so the slope is about 0 by construction.
4. **Live data.** Partial. Daily skill `1 - MAE(champion)/MAE(baseline)` at 15 min, last 14 days:
   - cabinet (`athena.cabinet_temp_c`): 0.08 on 11-02, 0.30 on 11-03, 0.24-0.37 since.
   - office (`office.temp_c`): 0.00 from 11-02 to 11-05, then 0.18-0.46.
   - **Rest state exists:** skill 0.00 means "no better than the baseline" (office 11-02..05, cabinet
     10-24..29). So the signal can read calm.
   - **But the jumps are step changes at model-roster changes** (`direct_arx_overlap` first appears in
     cabinet `prediction_model_run` on 11-03). Today the signal would credit Orion for our merged PRs.
   - Raw MAE (0.13-0.33 °C) moves with how hard each day is. Raw error slope is not learning.
   - `node:prediction.chat_turn` has 4 rows. Chat self-bet Brier scores follow the daily base rate;
     a Brier skill against climatology is needed and has not been computed (UNVERIFIED).
5. **Existing mechanism.** No learning-progress signal exists. `prediction_error_trend.py` is the
   nearest and serves a different purpose (mean-reversion guess). Leave it alone.
6. **Reversibility.** Cheap if it stays a read-only projection with no new node and no schema.

**Verdict: pass for two series only, with roster-change days removed.** Do not wire chat or legacy nodes.

### Design

- **Producer:** `services/orion-prediction` (predictions live there). A daily job writes
  `learning_progress = skill(last 7 d) - skill(prior 7 d)` per series. Days with a roster change in
  `prediction_model_run` are excluded from both windows. Exposed on the service's existing HTTP API
  as `GET /learning-progress` [default: no new table; computed on read].
- **Consumer 1:** Prime's frame gets one line: "where my forecasts are getting better / worse". Prime
  may take it on as an intention (Step -1 notes-only adapter).
- **Consumer 2:** a Step 3 feature (the epistemic term).
- **Not a market family yet.** No intensity enters the auction until Step 3 shows it predicts outcomes.

### Acceptance checks (Step 1)

- S1.1 Unit test: a planted series with a skill step on a roster-change day yields learning progress 0.
- S1.2 Unit test: a gradual planted improvement yields a positive value; flat skill yields 0 (rest).
- S1.3 Live: `GET /learning-progress` for cabinet and office, compared by hand against the SQL above.
- S1.4 The gate findings above are copied into the PR description.

## Step 2: the attention schema as a controller

In Graziano's attention schema theory, the schema exists **to control attention**. Ours only reports.

- **Wait for honest inputs.** The schema reads the old `execution`/`biometrics` nodes, which are not
  forecasts. Step 2 starts after the prediction rebuild gives them successors (#3168, step 6).
- **2a. Prime reads it.** One frame line: what the schema says is salient, top-down or bottom-up, and
  its confidence. Source: latest `substrate_attention_self_model` row; stale (older than 10 min) renders
  as absent.
- **2b. `predicted_shift` becomes a graded bet.** Today it is a string. Make it a `predict` act
  (actor `attention_self_model`, horizon = the reversion window), closed by the domain's next value.
  The reversion formula back-tested at 62% against 50% chance (`docs/superpowers/specs/2026-07-23-predicted-shift-reversion-finding.md`).
  The self-monitor grades it.
- **2c. Control.** Only after 2b shows skill: Prime may `regulate` from it, and Step 3 uses the
  schema's confidence as a feature. No new control channel.

### Acceptance checks (Step 2)

- S2.1 Prime frame shows the schema line; a stale row shows as absent (test).
- S2.2 `predicted_shift` acts open and close on `grammar_acts`; hit rate over 7 days reported with n.
- S2.3 Rollback flag removes the frame line and stops new bets.

## Step 3: a learned chooser (expected free energy)

**Goal.** Score each candidate action by expected free energy and pick the best one the budget allows.

- **Pragmatic term:** expected distance from preferences. Preferences = Juniper's priors
  (`families.yaml` `prior_weight`) plus the learned weights. Her priors are priors, never features a
  thin-data model may veto.
- **Epistemic term:** expected learning (Step 1, per domain).
- **Candidates:** intentions from every source (Step -1) and the wants already in the market. Step 3
  chooses **among intentions** across sources, not only among market families.
- **The market stays.** It is the budget constraint. The chooser ranks; the market still clears.

### Why not a neural net on synthetic data?

Synthetic data teaches the model back the rule that made it. Three honest uses:

1. **Pipeline proof.** Plant known answers and check the model finds them.
2. **A prior.** Samples from Juniper's priors are the network's version of the kappa = 30 pseudo-acts
   in `weights.py`.
3. **Later, imagination.** Rollouts from a world model trained on Orion's **real** logs (ties to
   dreams). A world-model shadow already exists in orion-prediction
   (`PREDICTION_WORLD_MODEL_SHADOW_ENABLED`, `services/orion-prediction/.env_example:45-48`); whether it
   fits this use is UNVERIFIED.

### Data is small, so the model is small

Graded acts in the last 30 days (`grammar_acts`): world_pulse_read 1,307, journal 1,062, prime 402,
outreach 284, painting 253, chat_reply 172, curiosity investigate 137, others under 160 each. Prime
has made 240 thoughts in total. So: a regularized logistic model or small gradient-boosted trees over
(family, source, hunger, prices, weight, Prime's predicted value, learning progress, schema confidence),
predicting `outcome_value`.

### Shadow and gate

- Shadow: for every market decision, log the chooser's ranking beside the formula's in the decision
  detail. Nothing obeys it.
- Replace only on held-out wins with a significance test, not a lower average: a paired test on
  per-decision log loss over a time-split hold-out [default: paired permutation test, p < 0.05, at least
  200 held-out decisions per family switched].
- Switch one family at a time, with a flag per family.

### Acceptance checks (Step 3)

- S3.1 Synthetic planted-answer test recovers the planted rule.
- S3.2 Shadow rankings logged for 14 days; report per family.
- S3.3 No family switches without the paired test passing on held-out data.

## Missing questions (defaults in brackets)

1. Name the primitive "intention" and the move `take_on`? [yes]
2. Promise result routing: always written into the original chat thread, ping through normal outreach
   gates? [yes; the ping counts against the shared daily cap]
3. Letter-thread result routing: next letter only, or also chat? [next letter only]
4. One family `durable.intention` for every source, split later only on evidence? [yes]
   (Answered: yes, at tier 1. See below.)
5. Hold window `deferrable_window_s` for the family. [86400 s; horizon still caps it]
6. May the carry turn's own promises open new concerns? [no, loop guard]
7. Promise horizon 72 h for intentions, before D2 codes it anywhere else? [yes]
8. Reliability evidence in the live weights, and how self-grades count. [on, `world_only`, after the replay]
9. Learning progress windows. [7 d vs prior 7 d, roster-change days excluded]
10. Step 3 switch gate. [paired permutation test, p < 0.05, >= 200 held-out decisions per family]
11. The Prime block already renders concern text (with the reply gist) into every motor prefix, which
    sits uneasily with D3 check 5. Treat as a separate fix? [yes, separate PR; out of scope here]

## Juniper's answers (2026-11-13)

Juniper: "all other defaults good", with one change.

- Questions 1-3 and 5-11: the bracketed defaults are accepted as written.
- **Question 4: one family `durable.intention` for every source, yes, but at TIER 1, not tier 2.**
  Reasons, recorded as given:
  - Once Orion chooses `take_on`, the "whether" is settled, and Orion settled it. What is left is
    follow-through, and follow-through is Orion keeping its word. Keeping its word is upkeep.
  - Tier 1's reserved share of each lane stops curiosity and consult runs from crowding the carry run
    out. Live follow-through is low today (49 Prime concern bets in 30 days, mean 0.10).
  - Tier 1 still bids. The economy still picks the time inside the window. This is unlike tier 0, which
    never bids and stays reserved for the daily letter's kind of work.
- Every "tier 2" mention for the intention family in this doc now reads tier 1.

## Proposed schema / API changes

- `orion/prime/thought.py`: `TakeOnMoveV1{kind="take_on", concern_id, why, not_before?}` in the `Move` union.
- `orion/prime/intention.py` (new, pure): `IntentionSourceV1`, adapter map.
- `orion/prime/intention_sources/turn_promise.py`, `letter_thread.py` (new, pure adapters).
- `orion/schemas/intention.py` (new): `IntentionOrderV1` (Prime -> Hub), registered in `orion/schemas/registry.py`.
- Hub: `POST /api/intention/run`.
- Durable runs: workflow `intention.carry`; family `durable.intention` (tier 1) in `families.yaml` and
  `BINDING_FAMILIES`.
- Outreach: `offer_message(..., session_id: Optional[str] = None)`.
- `handback.classify_who`: session kind `intention`.
- orion-prediction: `GET /learning-progress` (Step 1).
- No new bus channel and no new table in Step -1.

## Files likely to touch

Step -1: `orion/prime/thought.py`, `orion/prime/acts.py`, `orion/prime/handback.py`, new
`orion/prime/intention*.py`, `services/orion-mind/app/prime_loop.py`, `services/orion-mind/.env_example`
and `settings.py`, `services/orion-hub/scripts/` (new `intention_order.py`, `endogenous_outreach.py`),
`services/orion-durable-runs/app/run_bids.py`, a new `intention_graph.py`, `config/physiology/families.yaml`,
`orion/schemas/intention.py`, `orion/schemas/registry.py`, tests and an eval per service.
Step 0: `services/orion-durable-runs/app/physiology_market.py`, `.env_example`, `settings.py`.
Step 1: `services/orion-prediction/app/`. Step 2: `orion/prime/thought.py`, `orion/substrate/attention_self_model.py`.
Step 3: `orion/physiology/choice.py` (shadow log only), a new chooser module, evals.

## Non-goals

- No new market or auction. The market stays the budget.
- No per-source families now.
- No retries of failed carry runs in v1.
- No promise text in ordinary chat turns' motor prefix.
- No learning-progress node, and no learning-progress intensity in the auction before Step 3.
- No neural chooser trained only on synthetic data.
- No connection, coherence or learning-progress adapters in Step -1. They are listed so the schema fits them.

## Acceptance checks (summary)

Step -1: S-1.1..S-1.10 and the live smoke trace. Step 0: replay with and without reliability. Step 1:
S1.1..S1.4. Step 2: S2.1..S2.3. Step 3: S3.1..S3.3.

## Recommended next patch

Step -1, promise adapter only, behind `PRIME_INTENTION_TAKE_ON_ENABLED=true` (flags ship on). Order:
(1) contract: `TakeOnMoveV1`, `IntentionOrderV1`, family entry; (2) Prime move and order; (3) Hub
route, adapter and outreach `session_id`; (4) durable workflow; (5) live smoke on the next real promise.
Then the letter-thread adapter as a one-file PR to prove S-1.9.
