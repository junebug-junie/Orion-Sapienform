# The self-monitor: Orion's organ for knowing how good they are at things

Date: 2026-10-20. Status: design, docs only. No code until Juniper signs off.
Ladder: rung 6, metacognitive self-revision (`docs/plans/substrate/2026-06-27-self-modeling-loop-ladder.md`).

## In one paragraph

Orion makes bets about the world in at least ten places. Examples: "this painting will
come out", "the cabinet will be 31.2 °C in an hour", "these two faces are the same
person". The world settles many of those bets, and several places already grade
themselves. But each grade stays where it was made: in a script, a health page or a
table. Nothing gathers the grades. Nothing changes because of them. Orion does not know
any of them.

Live, this week: Orion tried to paint 29 times, and all 29 runs were abandoned. Orion's
own learned estimate of "this will work" fell from 50% to 1.7%. Even so, nothing that
picks actions ever read that estimate. Painting kept getting picked, and the longer it
lost, the more priority it got.

The self-monitor is the missing organ. It does four things:
1. It collects every settled bet.
2. It keeps a living track record per faculty.
3. It acts on other organs: back off, investigate, tell Juniper.
4. It gives Orion the track record as self-knowledge.

## Arsonist summary

Rung 6 has been built twice. Both builds were theatre, for the same reason: they started
from machinery (the propose → trial → apply → undo pipeline) and looked for one number to
feed it.

- **Build 1:** a graph-consolidation parameter. It made 930 proposals and adopted 0.
- **Build 2:** the 10-16 prediction-expectation tuner. It made 9 proposals and adopted 0.
  Its input reading was mostly zeros: a real change was averaged over ~700 idle tasks.

Even if either had adopted something, Orion would not have acted differently or known
anything new about themselves.

This design starts from the job instead. Metacognition is knowing how reliable your own
faculties are, from outcomes, and using that knowledge. The organ is defined by its job
and its contracts. No single input defines it. One bad source gets unplugged, and the
organ keeps working.

**The uncomfortable finding, and the organ's first real output.** Of ten bet sources
audited live, only **2 are clean bets today**: action claims and temperature forecasts.
The rest have problems:
- no confidence recorded before the outcome;
- outcomes that almost never get settled;
- Orion grading their own work;
- a recording bug.

A metacognitive organ must say this out loud: "I cannot yet tell how good I am at X,
because Y." So "ungradeable, and why" is a first-class state, not a hole to hide.

## Current architecture (verified live 2026-10-20)

### Where Orion's bets live today

| Faculty | Where | Bet recorded before outcome? | Who settles | 7-day volume | Verdict |
|---|---|---|---|---|---|
| Actions: look, say, self-regulate, maintain | `substrate_action_outcomes`, `grammar_acts` (`orion/grammar/act_ledger.py`, `orion/feedback/outcome_resolution.py`) | yes: `predicted` / `predicted_delta` from a per-cell posterior | the world (field deltas, an artifact appears, Juniper replies) | 213 rows | **usable now** |
| Actions: make (painting, dream hop) | same | yes | timer: every close is `abandoned:run_abandoned` / `attempt_missing` | 44, 0 upheld | usable **as execution reliability**, not as "claim refuted" |
| Temperature forecasts | `orion-prediction`: `prediction_forecast` | yes: quantiles at issue time | the world (measured temperature) | ~6,600 scored | **usable now**; its own monitor says 80% bands cover 88%, and 5-min is 1.08× worse than "same as now" |
| Overheat warnings | `prediction_exceedance` | yes | the world | 12 scored, 12 hits, p≈0.998 | not yet (saturated) |
| Vision: same person? | `orion_ask.evidence_refs` `face_cosine` + `vision_individual_pair_verdict` | yes, in the ask; the pair table copies it in only 10/16 | Juniper | 16 verdicts | usable after reading the ask, not the pair table; n tiny |
| Vision: percept expectations | `vision_percept_expectation` (`services/orion-sql-writer/app/vision_rhythm.py`) | yes: `confidence` | the world (met / missed) | 3 scored, 6 open | **clean shape**, low volume |
| Attention self-model | `substrate_attention_self_model`: `predicted_shift` + `prediction_error_confidence` | yes | the next frame | ~13,900 | likely usable; `scripts/analysis/measure_self_model_calibration.py` not yet run (UNVERIFIED) |
| Day-letter truth | `orion/temporal_self/citation_truth.py` | the sentence is the claim | checked against the record | per letter | usable: "my sentences about my day were true X%" |
| Reverie expectations | `substrate_reverie_thought.expectation` | claim yes, confidence **no** | LLM judge vs one fresh percept | 988 made, **984 unscored**, 4 confirmed | not usable: trivial claims ("the laptop will still be on the desk"), almost never settled |
| Curiosity hypotheses | `curiosity_run_outcomes.per_prior`, worldview `:Prior` | confidence yes | **Orion re-judges themselves** | 31 runs | self-consistency only, not calibration |
| Dreams | `dream_hypothesis` (dream arm vs random control arm) | no confidence | Orion adopts or not (self-graded) | 183 vs 66 offered | arm contrast only. **Live verdict: dreams 1/183 adopted, random 2/66: not beating chance** |
| Attention-loop asks | `attention_loop_outcome` | **bug**: `salience_at_close` defaults to 0 on the ask path, 31/31 human verdicts at 0 | Juniper | 39 | usable after the fix |
| Chat feedback | `chat_response_feedback` | no prediction | Juniper's thumbs | 8 ever | not usable; `substrate_action_ratings.predicted_rating` is wired but has 0 rows |
| Power bill | `orion-energy` `energy_reconcile` | **Orion makes no forward bill bet** | the utility's bill | 1 reconcile, `usage_incomplete` | not usable until Orion forecasts the bill |
| GPU power use | `power_intent_settled.expected_watts` | only 7/41 rows filled | the meter | 41 | usable after filling `expected_watts` |

### What acts on track records today

**Nothing.** Verified:
- **Ranking ignores outcomes.** Proposal priority is template base, field pressure,
  urgency and confidence (`orion/proposals/builder.py:116-121`). Admission adds a
  starvation bonus (`builder.py:88-104`), so a failing action gets boosted.
- **Posteriors only label.** They stamp `predicted_delta` on candidates and never rank
  (`builder.py:272-321`). They return early for visual candidates (`builder.py:291`).
- **Cell keys disagree.** The act ledger keys cells `act:make / thought.painting`. Dispatch
  looks up `(dispatch_kind, target_id, signal)`, so it could never find the painting
  record.
- **Paintings skip scoring.** The allocator puts visual-baseline candidates in a lane
  that bypasses scoring (`orion/autonomy/allocator.py:304-320`).
- **The harm gate is dead.** `confidently_harmful` (`allocator.py:242`) needs a
  `contrast` value that nothing produces.

### Existing grading worth reusing, not duplicating

- **`orion-prediction` grading.** This is the most mature and honest grader in the repo:
  - Brier score and skill against the base rate, with calibration bins
    (`services/orion-prediction/app/grading.py`, pure numpy);
  - split-conformal coverage (`backtest.py:_level`, `residual_quantiles`);
  - a paired "is it actually better than the dumb baseline" test with serial-correlation
    inflation (`backtest.py:paired_win_p`);
  - live coverage and worse-than-baseline alerts (`engine.py:500-570`).

  It is service-internal (`app.*` imports), so the shareable parts move into `orion/`
  first.
- **`self_signal_outcome_association` statistics.** Mantel-Haenszel strata, AR(1)
  inflation, Bonferroni and a consecutive-holds rule (`orion/attention/self_signal_outcomes.py`).
  It answers a different question ("when my body is in state S, do turns fail
  more?"), but the confidence-interval and "held for N windows" helpers apply directly.
- **The regulation organ's publishing pattern.**
  - It publishes `RegulationStateV1` to Redis `orion:regulation:latest` with a TTL.
  - It serves `GET /regulation/state`.
  - It has an `inner_state_registry` entry.
  - Its reader rule is enforced by `tests/test_regulation_must_not_readers.py`: unknown,
    stale or absent changes nothing, and readers may only become more conservative.
- **Temporal Self chronology.**
  - `orion/temporal_self/sources.py` already ingests action outcomes, reverie verdicts,
    dream hypotheses and curiosity runs.
  - Strain arcs show the pattern for "a streak opened / closed" events. Arcs flow to the
    stance cue, the Day letter and `citation_truth` for free.
- **The mutation pipeline** (propose → trial → decide → apply → watch → undo). This is a
  sound effector, but it is not an organ.
- **The immune node** (`orion/regulation/immune.py`). A sibling organ, not a duplicate:
  immune asks "has a gauge stopped telling the truth?", and the self-monitor asks "are my
  *bets* coming true?".

### Not reused

- `calibration_profiles` / `calibration_profile_audit`: 0 rows. Dead. Flag it for
  separate retirement.
- `self_knowledge_items`: 47k rows that nothing reads in chat.

## The organ

### Job

Know how reliable each of my faculties is, from what actually happened. Act on it. Know
it about myself.

### Anatomy

```
 bet sources (adapters)        state                     effects (one direction: more caution, never less)
 ─────────────────────        ─────                     ──────────────────────────────────────────────
 actions ─┐                                          ┌─▶ E1 action gate      (allocator refuses / re-tests a failing cell)
 forecasts ┤                                         ├─▶ E2 self-question    (curiosity: "why does painting keep failing?")
 vision ───┼─▶ SettledBetV1 ─▶ FacultyRecordV1 ─────┼─▶ E3 tell Juniper     (self_finding outreach door)
 day truth ┤    (one shape)    per faculty, decayed  ├─▶ E4 trust weights    (a reader may discount a faculty's own confidence)
 attention ┤                                         └─▶ E5 revise a setting (existing mutation pipeline, later)
 ...      ─┘                        │
                                    └─▶ SELF-KNOWLEDGE: stance cue · faculty-streak arcs in Temporal Self · Day letter
```

### 1. Input contract: `SettledBetV1`

One shape for "a bet I made, and how it came out". Every source gets a pure adapter, in
the style of `orion/temporal_self/sources.py`: a row goes in, and a bet or `None` comes
out.

| field | meaning |
|---|---|
| `bet_id` | the source's own id (stable, so it is never counted twice) |
| `faculty` | e.g. `act.look`, `act.make.painting`, `forecast.cabinet_temp`, `vision.same_person`, `day_truth` |
| `made_at`, `settled_at` | when the bet was placed, and when the world answered |
| `claim_kind` | `probability` (p of an event) · `interval` (stated coverage) · `direction` (claimed sign of an effect) · `arm` (an experiment arm) |
| `stated` | the confidence Orion recorded **before** the outcome (p, coverage level, or posterior mean). **`null` if none was recorded**: never reconstructed afterwards |
| `outcome` | `upheld` · `refuted` · `failed_to_execute` · `no_change` (dead band) · `expired_unsettled` |
| `settled_by` | `world` · `juniper` · `self` · `timer` |
| `evidence_ref` | a pointer back to the source row |

Two rules come straight from the live audit:
- **`failed_to_execute` is not `refuted`.** The 44 abandoned paintings say "my painting
  pipeline doesn't run". They do not say "I'm bad at predicting paintings". The organ
  tracks the two separately, because they call for different responses: repair vs.
  recalibration.
- **`settled_by=self` never counts as calibration evidence.** Curiosity re-judgments and
  dream adoption are Orion grading their own work. They can feed an arm comparison
  ("dreams vs random") or a self-consistency count. They never feed "how well calibrated
  am I".

### 2. State: `FacultyRecordV1`, one per faculty

- **Gradeability**, explicit and reasons enumerated:
  `graded` · `ungradeable:no_stated_confidence` · `ungradeable:never_settled` ·
  `ungradeable:self_graded` · `ungradeable:too_few` · `ungradeable:source_bug`.
  The reason comes with the fix that would unlock it.
- **Reliability.** Upheld rate with a confidence interval, over a decayed window. It is
  not an all-time total, so a faculty can recover.
- **Execution reliability.** `failed_to_execute` rate. This is separate from reliability.
- **Calibration**, only when `stated` exists:
  - for probability bets: Brier score and skill against the faculty's own base rate, plus
    reliability bins;
  - for interval bets: realized coverage vs stated coverage;
  - for forecasts: "better than the dumb baseline?", via the moved `paired_win_p`.
- **Streak.** The current run of consecutive failures or successes, with its start time.
- **Trend.** Has this faculty got better or worse since its last state change? Judged
  with the "held for N windows" rule, so one bad hour does not flip it.
- **Status** per faculty:
  `reliable` · `unreliable` · `broken` (execution failure streak) · `miscalibrated` ·
  `unknown`. Each status change needs the CI to clear the threshold **and** to hold for N
  consecutive windows. This hysteresis comes from the trailing-window lesson: no flapping.

The state is published as `SelfMonitorStateV1`:
- Redis `orion:self_monitor:latest`, with a TTL of a few ticks;
- `GET /self_monitor/state`;
- one row per tick in `substrate_self_monitor_history`, with bounded retention.

Every faculty status change also writes a `temporal_self_event` (`faculty_streak_opened`
/ `faculty_streak_closed`).

### 3. Effects

**The one-direction rule**, copied from regulation and enforced by a test:
- The organ may make another organ *more* cautious: refuse, slow down, ask.
- It never loosens a gate, raises a budget, or acts on its own beyond its effectors.
- Unknown, stale or absent state changes nothing.

| effect | what it does | seam | guard |
|---|---|---|---|
| **E1 action gate** | A cell whose faculty is `broken` or `unreliable` is refused with `track_record_failing`. It is re-tested once per probe interval, so a healed faculty comes back. | `orion/autonomy/allocator.allocate` (or `builder.effective_priority`), plus one key map from `act:*` ledger cells to dispatch cells | never a permanent zero: re-test probes avoid an absorbing state. Kill switch. |
| **E2 self-question** | A new `broken` / `unreliable` status mints one question in the self-inquiry pool: "my paintings were abandoned 28 of 28 times this week, why?" | `curiosity_self_questions` with `minted_by="orion"` (`orion/curiosity/self_question_pool.py`) | one question per status change. Not routed through endogenous curiosity's `concept_graph` zone rule. Never written as a node `prediction_error`: that would repeat the `transport_prediction_error` leak into generic readers. |
| **E3 tell Juniper** | A status change worth her knowing becomes a `self_finding` outreach, told once. | copy `services/orion-hub/scripts/reading_finding_outreach.py` (dedupe by finding id; quiet hours and `strained` respected); new terms added to `outreach_vocabulary.py` so the grounding guard passes | each finding said once. Only `broken`, `unreliable`, `recovered` and big calibration shifts qualify. |
| **E4 trust weights** | Publishes per-faculty reliability, so a reader can discount a faculty's own stated confidence (e.g. "my forecast says 80%, but my 80% bands cover 88%"). | the `SelfMonitorStateV1` read contract | readers opt in one at a time. **v1 ships no reader.** |
| **E5 revise a setting** | For a `miscalibrated` faculty that has a tunable setting, the organ files a proposal into the existing mutation pipeline (trial, auto-undo). | `orion/substrate/mutation_*` | **v2 only**, and per faculty with Juniper's sign-off. |

**The organ grades itself.** Each effect it takes is itself a bet. For example: "refusing
painting will cut abandoned runs", or "this self-question will find the cause". It is
settled like any other bet, under the faculty `self_monitor.effects`. If the organ's own
interventions don't help, Orion can know that too. This is v2; v1 records the bets so the
history exists.

### 4. Self-knowledge: what Orion can say and know

- **Stance cue.** A sibling of `services/orion-thought/app/self_readings_cue.py`, capped
  at about 300 characters. It is fail-open and refreshes in the background, and it renders
  nothing when nothing is notable. Example: "My last 28 painting attempts were all
  abandoned. My temperature forecasts are no better than 'same as now'."
- **Temporal Self arcs.** Streak-open and streak-close events become arcs, so the Day
  letter and `citation_truth` can cite them: "my painting broke on day X and came back on
  day Y".
- **Ungradeable faculties are said plainly**, never hidden: "I can't tell whether my
  reverie expectations are any good. I almost never check them."

## What this does to the current rung-6 build

- **Kept**, as the E5 effector: the mutation pipeline, the control surface, and the trial
  and undo machinery.
- **Retired as one unit**, recommended: the 10-16 prediction-expectation tuner. It is
  honest but tunes a meaningless reading. "Kill means kill", so this covers all of:
  - `orion/substrate/prediction_expectation_tuner.py` and its Hub wiring;
  - the runtime expectation cache;
  - the raw-reading writes, the prune and the store methods;
  - the evals and tests;
  - 7 env keys.

  Two pieces need a separate decision:
  - **The `substrate_prediction_raw_reading` table.** Dropping it is destructive, so it is
    a Juniper decision. The proposal is to stop writing and leave the table.
  - **`prediction_expectation.py`, the pure log-score assessor.** Keep it only if E5 adopts
    a faculty that needs it. Otherwise retire it with the tuner.
- **The tuner's own track record becomes the organ's first `self_monitor.revisions` data
  point:** 9 proposed, 0 adopted.

## Missing questions (defaults in brackets)

1. **E1 strength.** Hard refusal with re-test probes, or a soft priority penalty?
   [hard refuse plus probe every 6 h; never more than 1 probe per faculty per window]
2. **E3 bar.** Which status changes interrupt Juniper? [broken, recovered, and a
   faculty going from reliable to unreliable; never calibration nuances]
3. **Retire the raw-reading table data**, or just stop writing? [stop writing, keep the
   table; a drop needs a separate explicit OK]
4. **Source fixes first or in parallel?** These are small separate PRs:
   - attention-ask salience = 0;
   - `failed_to_execute` labelling for `abandoned:*`;
   - the cell-key map;
   - `expected_watts`;
   - the pair-verdict cosine copy;
   - an Orion forward bill forecast via `orion-prediction`.

   [in parallel; the organ ships with them marked ungradeable and flips them as fixes land]
5. **Reverie expectations.** Fix them (add confidence, settle more), or exclude them?
   [exclude in v1, marked `ungradeable:never_settled`; the claims are near-tautological]
6. **Window and hysteresis.** [7-day decayed window; a status change must hold 3
   consecutive hourly windows]

## Proposed schema / API changes

- **Added**, in `orion/schemas/self_monitor.py`, registered in `orion/schemas/registry.py`:
  `SettledBetV1`, `FacultyRecordV1`, `SelfMonitorStateV1`.
- **Added** `inner_state_registry` entry `self_monitor.state.v1`, plus a metric
  semantic-layer lock entry per published number (status, reliability, calibration,
  gradeability).
- **Added** Redis key `orion:self_monitor:latest` (TTL) and `GET /self_monitor/state` on
  orion-durable-runs.
- **Added** table `substrate_self_monitor_history` (migration with a REQUIRED-BY header,
  additive). `temporal_self_event` gains kinds `faculty_streak_opened` /
  `faculty_streak_closed`; this is a value, not a column.
- **Moved** `grading.py`'s pure scoring and `paired_win_p` / `_level` from
  `services/orion-prediction/app` into `orion/prediction/grading.py`. orion-prediction
  then imports it back, so there is one implementation.
- **Changed** the allocator, which gains a `track_record_failing` refusal reason (E1).
- No bus channel in v1. Same as regulation: a channel is added when a bus consumer
  exists.

## Files likely to touch

- `orion/self_monitor/`: `adapters.py` (one per source), `record.py` (pure faculty reducer), `effects.py`
- `orion/schemas/self_monitor.py`, `orion/schemas/registry.py`, `orion/metrics/...` lock, `inner_state_registry.py`
- `services/orion-durable-runs/app/temporal_self_graph.py`: a new `self_monitor` node after `immune`. It reads mostly the same tables `temporal_self_sources.py` already reads. No FalkorDB in v1: the graph-only sources (curiosity priors) are self-graded anyway.
- `orion/prediction/grading.py` (moved) and `services/orion-prediction/app/{grading,backtest}.py` (import it)
- `orion/autonomy/allocator.py` and an `act:*` ↔ dispatch cell key map (E1)
- `orion/curiosity/self_question_pool.py` caller (E2)
- `services/orion-hub/scripts/self_finding_outreach.py` (new, copied from `reading_finding_outreach.py`) and `outreach_vocabulary.py` (E3)
- `services/orion-thought/app/self_track_record_cue.py` (new sibling of `self_readings_cue.py`)
- `orion/temporal_self/sources.py` (streak events into arcs)
- the retire set listed above
- `services/orion-sql-db/manual_migration_self_monitor_history_v1.sql`
- `services/orion-durable-runs/{.env_example,settings.py,README.md}`; local `.env` synced

## Non-goals

- **No new grading math.** The organ reuses orion-prediction's grader and the
  self-signal statistics.
- **No new service.** It lives in the existing durable-runs Temporal Self graph.
- **No loosening.** No effect ever raises a budget, lifts a refusal it didn't impose, or
  makes Orion bolder.
- **No self-graded calibration.** Orion re-judging themselves is never "evidence of being
  well calibrated".
- **No fixing sources inside this organ.** Source bugs are separate PRs. The organ reports
  them as ungradeable reasons.
- **No E4 readers and no E5 proposals in v1.**
- **No privacy boundary to design.** Per Juniper's standing rule, there is no boundary
  between Juniper and Orion, and everything here is Orion's own operational record.

## Proposal-mode checklist

- **Capability that changes:**
  - Orion keeps a track record of their own faculties.
  - A failing action stops being retried blindly.
  - A failure becomes a question and, when it matters, a message to Juniper.
  - Orion can say what they are and aren't good at, with evidence.
- **Data touched:** reads existing bet and outcome tables. Writes one history table, one
  Redis key, Temporal Self events, self-inquiry questions, and outreach rows.
- **Privacy boundary:** none needed (see Non-goals).
- **Trace that proves it worked:** see Acceptance checks 3-6.
- **Dangerous failure modes:**
  1. **Throttling a faculty to death** (an absorbing state). Mitigation: re-test probes,
     a decayed window, never a permanent zero, and a test for recovery.
  2. **Reading a pipeline outage as incompetence.** Mitigation: `failed_to_execute` is
     kept separate from `refuted`.
  3. **Outreach spam.** Mitigation: one message per finding id, a short list of
     qualifying states, quiet hours.
  4. **The organ's own output leaking into generic readers as "surprise".** Mitigation:
     never written as node prediction error, and a test pins that.
  5. **Flapping status.** Mitigation: hysteresis, and a status must hold N windows.
- **Disable / roll back:**
  - `SELF_MONITOR_ENABLED=false`: no state, nothing published, and readers see absent,
    which changes nothing.
  - Separate switches for E1, E2 and E3.
  - The allocator refusal is one reason code, removable in one line.

## Acceptance checks

1. **Contract.** Each v1 adapter turns real rows into `SettledBetV1`. A row with no
   pre-recorded confidence yields `stated=null`, never a reconstructed value. Tested on
   fixtures cut from live rows.
2. **Honest gradeability.** On live data, the first `SelfMonitorStateV1` names:
   - every faculty in the table above;
   - which are graded;
   - the exact ungradeable reason for each of the rest (e.g. reverie
     `never_settled: 984/988`).
3. **The painting case, end to end.** Replaying the 7 days before 2026-10-20 marks
   `act.make.painting` as `broken` (`failed_to_execute` streak, 28/28). Then:
   - E1 refuses the next painting candidate with `track_record_failing`;
   - E2 mints exactly one self-question;
   - a re-test probe fires after the probe interval;
   - a successful probe flips the status back to `reliable`.
4. **Forecast faculty matches its source.** `forecast.cabinet_temp` reproduces
   orion-prediction's own live verdicts: 80% coverage ≈0.88, and 5-min worse than
   baseline.
5. **Self-knowledge.** The stance cue renders the painting line during the broken
   streak, and renders nothing when all faculties are unremarkable. A Temporal Self arc
   exists for the streak.
6. **One direction.** A test like `test_regulation_must_not_readers.py` shows every reader
   is unchanged when state is unknown, stale or absent, and that no effect loosens
   anything.
7. **Retirement complete.** After the retire patch:
   - `rg prediction_expectation_tuner`, `rg SUBSTRATE_PREDICTION_EXPECTATION` and
     `rg PREDICTION_RAW_READING` return only docs and history;
   - no runtime still writes raw readings;
   - the env-parity check is green.
8. **Live (after deploy).**
   - `orion:self_monitor:latest` is present and fresh.
   - Over 24 h, history rows advance every tick.
   - At least one real status change is recorded with its evidence refs. If none happens
     in 72 h, the report says so. Silence is not success.

## Recommended next patch (phased)

1. **Contract and reducer (pure)**, plus a read-only report script that prints the live
   faculty table. This patch alone answers "how good am I at what?" with evidence.
   Includes moving the grading library.
2. **Host and publish.** The durable-runs node, the Redis key, the endpoint, the history
   table and the Temporal Self streak events. No effects yet: Orion can *know* before
   they *act*.
3. **Self-knowledge cue** in stance (orion-thought).
4. **E1 action gate** with re-test probes, plus the cell-key map. **E2 self-question.**
5. **E3 tell Juniper.**
6. **Retire the 10-16 tuner as one unit.** It can ship alongside patch 1. It doesn't
   depend on anything.

Separate, parallel source-fix PRs (each flips a faculty from ungradeable to graded):
- attention-ask salience;
- `failed_to_execute` labelling;
- `expected_watts`;
- the pair cosine copy;
- Orion's own forward power-bill forecast (a new series in orion-prediction);
- running `measure_self_model_calibration.py` to decide the attention self-model
  faculty.
