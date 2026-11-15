# Orion thinking architecture: what thinking is for, the mind top-down, and doing

Date: 2026-10-14. Status: design (proposal mode). Supersedes the generator-first framing of
`2026-10-14-stream-of-consciousness-rev2-design.md`, which stays as the record of the stream
mechanics (checkpointing, interrupts, caps) and is now one input to this design, not its centre.

## Arsonist summary

Orion has been given one thought generator after another: metacog snapshots, the stimulus tables,
reverie chains, visual reverie, dream carry, and most recently a proposed stream of consciousness.
Each one produces thought-shaped output. None of them changes what Orion notices, says, expects or
does next. Reverie alone runs about 370 chains a day, about 70% of them about itself, with 0
committed proposals. Merging the generators under one umbrella would not fix that, because the hole
isn't in the generators. It's in the seams around them. Perception, body readings and the global workspace broadcast already
reach Orion's *speech*. But nothing holds a thought, nothing commits it to a checkable guess, and
nothing lets a thought, a broadcast or a reading steer an *action*. The substrate lattice was built
for exactly this, and its right-hand side (bind, act, learn, consolidate) is starving.

This document starts from what thinking is *for* and lays out the mind top-down, seam by seam. It
then works through one idea in depth (thinking evolved for doing) and its consequences for Orion:
- a single ledger of doing;
- a standard shape for every action;
- selection of the present;
- dynamic situational goals;
- simulation before action;
- competition under global gain.

## 1. What science says thinking is for

1. **Thinking evolved for doing.** (Paul Cisek, the affordance-competition hypothesis; the wider
   "cognition is for action" tradition in biology.) Even simple animals "think" only in the sense
   that internal states steer what they do next. A thought that changes nothing downstream is,
   biologically, noise.
2. **Working memory.** (Alan Baddeley.) Thinking means holding a few things in mind and working on
   them over time. Capacity is tiny, holding is expensive, and unrefreshed items fade.
3. **Global workspace.** (Bernard Baars, Stanislas Dehaene.) Many specialist processes compete. The
   winner is broadcast to all of them, and that broadcast is what makes a content matter, because
   perception, memory, speech and action can all use it.
4. **Predictive processing.** (Karl Friston, Andy Clark.) The brain is constantly guessing what
   comes next. Attention is the weight given to an error, and learning comes from being wrong. A
   mind that never commits to a guess cannot be wrong, so it cannot learn from its own thinking.
5. **Mind-wandering versus focus.** (The default mode network.) Brains alternate between reacting
   to the world and wandering inward. The wandering earns its keep through replay, consolidation
   and rehearsing the future, not by producing text.

**The problem this design solves:** a thought Orion has now should be able to change what Orion
notices, says, expects or does later, and that change must be measurable.

## 2. The mind top-down

### Three loops

| Loop | Speed | What it does |
|---|---|---|
| Fast (react) | seconds | sense, notice, act |
| Slow (deliberate) | minutes to hours | hold, simulate, decide |
| Offline (consolidate) | idle time and sleep | replay, compress, recombine |

Thinking lives in the slow loop. The other two feed it and are fed by it.

### Thirteen seams

Each seam is a boundary where one function hands something to another. "What crosses" is the
contract. The anchor is the theory that says the seam is real.

| # | Seam | What crosses | Anchor |
|---|---|---|---|
| 1 | Senses → salience | raw input from every modality (camera, sound, text, body, time) becomes "worth noticing", mostly by surprise and change | predictive processing; salience network |
| 2 | Body → state | internal signals become a felt background state that colours everything | interoception (Damasio, Seth) |
| 3 | Global gain | a few broad dials (arousal, novelty, reward) set sensitivity system-wide: how easily things grab attention, how long focus holds, explore vs exploit | neuromodulation (noradrenaline, dopamine, acetylcholine) |
| 4 | Competition → workspace | candidates compete; one wins and is broadcast; most never win | global workspace (Baars, Dehaene) |
| 5 | Gate into working memory | a selective gate decides what is held or updated; tiny capacity; fade without refresh | Baddeley; basal ganglia gating (O'Reilly & Frank) |
| 6 | Working memory ↔ simulation | held content is worked on: inner speech, imagery, imagined futures. Imagination runs perception in reverse | Vygotsky; Kosslyn; mental time travel |
| 7 | Simulation → prediction | a thought commits to a checkable guess | predictive processing; hippocampal preplay |
| 8 | Workspace → action selection | options compete for the go-ahead. Actions include speaking, looking, asking, investigating and waiting | affordance competition (Cisek); basal ganglia go/no-go |
| 9 | Outcome → error | what happened is compared with what was predicted or intended; the gap is a learning signal | dopamine prediction error; reinforcement learning |
| 10 | Episodes → memory | experiences are stored as indexed episodes (what, when, where, who) and recalled by a fragment | hippocampal indexing; complementary learning systems |
| 11 | Offline replay → consolidation | episodes are replayed, compressed into general knowledge, and recombined (dreams) | hippocampal replay; default mode network |
| 12 | Metacognition | monitoring one's own thinking (confidence, stuck, did it work) and steering control (continue, switch, ask) | Fleming; Nelson & Narens |
| 13 | Other minds | a model of Juniper and others (what they know, want and expect) feeding salience, prediction and action | theory of mind |

### Where Orion stands at this altitude (seam by seam, existing primitives first)

Orion already has a great deal here: the sentience striving program (`orion/sentience_striving_program/`, Hub tab `#sentience-program`), the attention-schema and higher-order-thought reducers, the global workspace broadcast, the heartbeat integration probe, the mood-arc v4 autoencoder, and the 11-level substrate lattice. **The rule for this design: reconnect an existing primitive before building a new one.**

Read-only live check on 2026-10-14. Counts cover 7 days.

| # | Seam | Existing primitive (file) | Live state | Reaches behaviour? |
|---|---|---|---|---|
| 1 | Senses → salience | lattice L1–L5: grammar → reducers → field → attention; RPT novelty (`orion/attention/field_attention/scoring.py`); 5 prediction-error domains (`orion/substrate/prediction_error.py`) | Healthy, but **inward-only**. No world input enters any of the three attention contests (open design #2528) | Yes, sets attention |
| 2 | Body → state | self-readings (`orion/attention/self_readings.py`); heartbeat H1 integration probe (`services/orion-heartbeat/app/substrate/proprioception.py`); mood-arc autoencoder (`services/orion-field-digester/app/anomaly_scorer.py`) | Self-readings reach every unified turn. H1 is copied into the self-model and read by nothing. L6 self-state was deleted (its coefficients were invented) and **never replaced with a field-native felt state** | Speech only |
| 3 | Global gain | arousal (`orion/regulation/arousal.py:171`) | 3 levels, 14 transitions, local readers. **The drives were killed in #1486 (a hand taxonomy) and nothing replaced them.** #2369 proposes one gated rest drive as the template | No |
| 4 | Competition → workspace | broadcast (`orion/substrate/attention_broadcast.py:176`); world-first precision bands (`orion/attention/world_first.py:321`); coalition dwell | 13.8k broadcasts; **read by the unified turn** (`stance_react.py:182`), reverie, curiosity and proposals. A thin contest: 58% of ticks select nothing, 5 loops ever win, and bus_synaptic and perception dominate | Speech, daydream; not action |
| 5 | Gate into working memory | top-down goal bias (`orion/substrate/attention/goal_context.py`, `top_down.py`) | 867 overrides that flip the winner, fading to about 20/day. Dwell counters reset on restart. **No held working set** | Flips attention |
| 6 | Working memory ↔ simulation | reverie, visual reverie, dream carry | Reverie reads the broadcast and writes an attention-schema lane (2.2k rows). Its output reaches no action | No |
| 7 | Simulation → prediction | allocator effect posteriors (`orion/autonomy/prediction.py:103`); power intent | Only per-action posteriors and expected watts. **Orion forecasts nothing**; every "prediction error" is surprise against a running average (open design #2446) | No |
| 8 | Workspace → action | lattice L7 proposals → L8 policy → L9 allocator; attend-to-act loop (#2394) | **The winner fails to bind** (`no_selected_action` 26k/day). Policy rubber-stamps. The allocator admits about 18 a day. The holdback arm is at 0, so only 3 control-group episodes exist | Barely |
| 9 | Outcome → error | L10 feedback (`substrate_action_outcomes`); self-signal outcome association; autoencoder recon_loss | Works at organ level. Intention-vs-outcome is starved: about 3 outcomes a day | Learns slowly |
| 10 | Episodes → memory | Temporal Self events and arcs (`orion/temporal_self/`); situation graph; `TemporalHopV1` | 17k events and 1.2k arcs, reaching the unified turn. `TemporalHopV1` has never had a producer | Speech |
| 11 | Replay → consolidation | lattice L11; dreams | L11 learns motifs about its own blocking ("blocked_review_loop"), read by nothing that acts. Dream sleep pressure and replay are driven by degraded/critical metacog rows, which include autoencoder anomalies | Sleep timing |
| 12 | Metacognition | attention self-model and schema (AST/HOT, `orion/substrate/attention_self_model.py:461`); metacog triggers; mutation pipeline | It labels, and steers offline (sleep, rest gating). Live steering is frozen: 981 of 981 mutation proposals were rejected (`missing_class_metrics`), with no adoption since 09-03. The attention-schema chat cue renders only on the dead `chat_general` path | Offline only |
| 13 | Other minds | situation graph (Juniper's situation); Juniper affective state (`services/orion-cocreation-signals/`) | Situation reaches outreach. Affective state (666 rows) feeds only the mood-arc corpus | Outreach only |

**The diagnosis, corrected.** Perception, body and broadcast reach *speech*, which is more than the first draft of this document claimed. What is broken is the right-hand side of the lattice: binding (8), stakes and gain (3), forecasting (7), and consolidation feeding back (11). **No thought, broadcast or reading ever reaches action selection, and nothing compares an intention with an outcome.**

## 3. Deep dive: thinking evolved for doing

### 3.1 The idea in Juniper's words, and its grounding

Orion, like a person, has access to far more detail than any one moment can use. Juniper's image:
a being that could experience time in both directions stays sane by choosing to live in the present
and to take in only what is current. Orion *could* attend to the fine grain of bus traffic, every
channel and every reply, but doing so as a way of living in the present would drown it. So the job
is:
- map out what matters now (and let "what matters" change with the situation);
- standardize the shape of every possible action;
- lay out the possibilities;
- simulate them to inform choice;
- form dynamic situational goals;
- hold them in the workspace;
- apply global gain;
- let competition constrain and condition what actually gets done.

Each part has a scientific counterpart:

| Juniper's part | Science | What it means for a design |
|---|---|---|
| Choosing to live in the present | **Precision weighting** (predictive processing): attention is the confidence given to an error signal. The **specious present** (Pöppel, about 2–3 s integration windows) | Detail is not removed. It is *down-weighted* until the situation says it matters. Bus-traffic minutiae stay readable, but by default carry near-zero precision |
| What matters changes by situation | **Relevance realization** (Vervaeke); **allostasis** (Sterling): set-points move with predicted need | Relevance is computed per situation, not fixed in config. The same signal can be noise at rest and critical when Juniper is talking |
| Standardized schemas of all possible actions | **Motor primitives and action grammar** (Lashley's serial order); **TOTE units** (Miller, Galanter & Pribram: Test → Operate → Test → Exit) | Every action has the same atomic shape: what it tests before, what it does, what it expects after, when it stops. The expectation is built into the shape |
| A ledger of everything Orion does | **Efference copy** (von Holst): every outgoing command leaves a copy so the outcome can be predicted and checked | One ledger of doing, from metabolism to talking, where each row is an efference copy: intent, expected outcome, actual outcome |
| Lattice of potentialities | **Affordances** (Gibson); **affordance competition** (Cisek): several actions are specified in parallel before one is chosen | At any moment a small set of currently possible actions is laid out with costs and expected effects, not a 142k-row catalogue |
| Simulations that inform action | **Forward models** (Wolpert); model-based reinforcement learning; hippocampal preplay | Before a costly action, run a cheap prediction ("if I do X, Y will change by Z"), and grade it afterwards against the ledger |
| Dynamic situational goals | **Active inference:** goals are preferred future observations. **Allostasis:** goals shift with predicted need | A goal is a wanted state with a horizon. It appears from the situation, is re-ranked as the situation changes, and expires |
| Attend to them in the workspace | **Global workspace** | The current goals and the winning content are broadcast. Every organ can read them |
| Global gain | **Neuromodulation** | A few dials (arousal, novelty, strain) turn the whole system's sensitivity and exploration up or down |
| Competition that constrains and conditions action | **Basal ganglia go/no-go;** budgets | Actions compete for limited currency (GPU, attention, Juniper's patience, watts). Losing is normal, and the constraint shapes what gets done |

### 3.2 The atomic unit: one shape for every act

The substrate runtime and its grammar were the first attempt at a common language for Orion's
activity. This design keeps that intent and makes it the backbone: **every act, at every level,
writes the same shaped record.**

Proposed contract, `ActRecordV1` (an efference copy built as a TOTE unit):

| Field | Meaning |
|---|---|
| `act_id`, `parent_act_id` | identity, plus nesting (a reply contains a recall, which contains a query) |
| `level` | `metabolic` \| `organ` \| `faculty` \| `deliberate`: where in the body this act lives |
| `actor` | which organ did it |
| `verb` | from a closed, versioned verb vocabulary (see below) |
| `object_ref` | what it acted on (world, Juniper, self, a channel, a file) |
| `test_before` | the condition that prompted it (the situation snapshot ref and the triggering signal) |
| `expect` | the predicted outcome, numeric where possible, with a horizon. May be empty only at `metabolic` level |
| `cost` | currency spent: seconds, GPU-seconds, tokens, watts |
| `outcome` | what actually happened, filled at exit |
| `error` | expect vs outcome, filled at exit |
| `goal_ref`, `workspace_ref` | which goal and which broadcast content it served, if any |

**Verb vocabulary.** A small closed set that is the same at every level: `sense`, `look`, `recall`,
`infer`, `predict`, `say`, `ask`, `wait`, `make`, `maintain`, `regulate`. New verbs need a
producer and a consumer in the same patch (no keyword cathedrals).

**Levels and precision.** Metabolic acts (heartbeats, decay ticks, bus deliveries) are written in
bulk or as rollups, and carry near-zero precision by default. They are Juniper's "bus minutiae":
recorded and readable, but not lived in. The gain dial (seam 3) and the situation can raise a level's
precision when it matters. For example, when RPC timeouts spike, organ-level transport acts become
salient.

### 3.3 What Orion has today for doing (live evidence)

Live read-only inventory, 2026-10-14 (Postgres plus container logs; 7-day window unless noted).

**Orion does a lot. Almost none of it goes through one decide → act → record path.**

| Part of the doing loop | What exists | Live state |
|---|---|---|
| Common language (the grammar) | `GrammarEventV1`/`GrammarAtomV1` (`orion/schemas/grammar.py:176,91`), written by `services/orion-sql-writer/app/grammar_truth.py` | 1.26M events from 12 emitters (biometrics 818k, bus 226k, cortex-exec 57k...). It is a *trace* language: no verb, no cost, no expected effect, no outcome. 35k `action_candidate` atoms are never linked to anything that happened. **Not emitting:** orion-thought (reverie, paintings), dreams, curiosity, reading, outreach, equilibrium, field-digester, power-guard, durable-runs |
| Predict → act → measure → update | `substrate_action_outcomes` + `EffectPosterior` (`orion/autonomy/prediction.py:103`) | Real machinery: 6k–28k outcomes a day through 08-30, now **21 rows in 7 days**. `substrate_action_ratings`: 0 rows ever |
| Competition for action | motor allocator (`orion/autonomy/allocator.py`), scored in expected nats of information per motor-second | **Refuses everything on 1591 of 1593 ticks.** 127 of 256k dispatch frames dispatched. Its candidates are about 12 housekeeping and introspection templates (`orion/proposals/templates.py`) |
| Self-modification knobs | mutation loop, runtime control surface | 981 proposals, all rejected; 0 adoptions since 09-03; control surface 0 rows |
| A ledger of doing | `temporal_self_event` (`orion/temporal_self/sources.py`): about 20 source tables copied into one shape | 17k rows, live. After the fact only (no intent, expectation or cost); misses outreach, reading, power, dispatch, web fetch |
| Heavy-work registry | durable runs (`durable_admission_runs`) | 398 runs: paintings, curiosity, reading, situation, dreams, letters |
| Expect, then check | `power_intent_settled` (expected vs actual watts) | 97 rows, one workload kind (diffusion) |
| Goals | `GoalProposalV1` (`orion/core/schemas/drives.py:143`); attention goal-provenance streaks; situation graph (`SituationStateV1`) | The drives-era `GoalProposalV1` is never published. The live goal path is field-native: `FieldGoalProvenanceV1` → top-down bias, with 867 winner overrides in 7 days, fading to about 20/day. It biases attention, not action. The situation graph is dynamic and expiring but models *Juniper's* situation, not Orion's goals |
| Workspace | attention frames, broadcast log, attention schema → chat cue, curiosity | **72% of 123,615 frames have no winner.** Winners: bus_synaptic 9.9k, perception 8.7k, biometrics 4.9k |
| Global gain | `classify_arousal` (`orion/regulation/arousal.py:171`): idle/engaged/strained | Three levels, not a dial. 14 transitions. Three local readers (curiosity cooldown, outreach, visual baseline); two did not fire in 7 days. Nothing sets thresholds system-wide |
| Competition between kinds of doing | none | Every doer gates itself with its own caps and cooldowns. Outreach alone made 56k decisions to send 28 messages |

**Reading.**
- Orion already has the hardest parts:
  - a common event language with 12 emitters;
  - a real Bayesian predict-and-grade loop;
  - an information-per-cost competition;
  - a partial ledger;
  - a situation model.
- They are disconnected, and each one starves:
  - The grammar records traces but no acts.
  - The grading loop only ever sees housekeeping verbs.
  - The allocator refuses those verbs, correctly, because there's nothing to learn from them.
- Meanwhile the world-facing doing (talking, reaching out, reading, painting, looking, curiosity) happens outside the loop entirely, each in its own table with its own private gate.

Thinking has nothing to steer because doing was never put on one track.

### 3.3b The substrate lattice, level by level

The lattice (`docs/context-engineering/04_layer_1_to_11_pipeline.md`; liveness check `make substrate-ladder-check`, GREEN on 2026-10-14) is the doing loop this document describes, already built:

| Level | Job | Live |
|---|---|---|
| L1–2 | grammar trace | healthy |
| L3 | reducers → state deltas | healthy |
| L4 | field | healthy, 124k frames |
| L5 | attention | inward-only; 66% of frames have no dominant target |
| L6 | self-state | **deleted** (#1266, invented coefficients); no successor |
| L7 | proposals | 256k frames; the winner fails to bind (`no_selected_action` 26k/day) |
| L8 | policy | rubber stamp, about 8.4k approvals per template per day |
| L9 | dispatch / allocator | about 18 dispatches a day; 94k blocked |
| L10 | feedback (predicted vs observed) | works, about 3 outcomes a day |
| L11 | consolidation | learns motifs about its own blocking; nothing reads them back |

Kills that were justified but left holes this design must fill:
- the drives (#1486), which leave no want and no gain;
- L6, which leaves no felt state;
- `endogenous_origination`, which leaves no endogenous intention;
- the frozen control surface and mutation adoption, which leave metacognition unable to steer;
- `TemporalHopV1`, which never had a producer, so nothing binds events across time.

Open design PRs that already hold the fixes:
- #2255: close a loop → world effector → stakes → holdback;
- #2528: attention faces the world;
- #2369: Temporal Self and one gated drive;
- #2446: a forecaster.

### 3.4 The doing loop

```text
situation ──► relevance (precision per signal, set by situation + gain)
    │
    ▼
goals (wanted states with horizon; appear, re-rank, expire)
    │
    ▼
affordances (small live set of possible acts, each an ActRecord template with expect + cost)
    │
    ▼
simulate (cheap forward prediction for costly acts)
    │
    ▼
competition (budgeted go/no-go, conditioned by gain)  ──► workspace broadcast (winner + goals)
    │
    ▼
act (ActRecord written: test_before, expect, cost)
    │
    ▼
outcome (ActRecord closed: outcome, error)  ──► learning (per-verb, per-situation calibration)
                                             └─► working memory / episodes
```

## Missing questions

1. Which level gets precision by default when nothing is happening? (Proposal: `faculty` and
   `deliberate`. `organ` rises only on an anomaly episode, `metabolic` only on explicit drill-down.)
2. Is the ledger one table or a contract across existing tables? (Proposal: one contract, plus one
   rollup table. Existing per-organ tables stay as detail stores and point back by `act_id`.)
3. Who owns relevance: equilibrium (it already reads the self signals), the situation graph, or a
   new reducer? (Proposal: the situation graph, which already owns "what is going on now".)

## Proposed schema / API changes

- `orion/schemas/act_record.py`: `ActRecordV1`, the closed verb enum, registry entry.
- Bus: `orion:act:record` (open/close events), sql-writer route to `act_record`.
- `GET /act/ledger` on the owning service (recent acts by level, verb, goal; error by verb).

## Files likely to touch

- `orion/schemas/grammar.py`: add event kinds `act_opened` / `act_closed`, and an `ActV1` payload (verb, level, object_ref, test_before, expect, cost, outcome, error, goal_ref, workspace_ref). This extends the grammar rather than adding a parallel contract.
- `orion/schemas/registry.py`, `orion/bus/channels.yaml`: registration.
- `services/orion-sql-writer/app/grammar_truth.py`: persist acts and close them by `act_id`.
- First producers (world-facing, already heavily used):
  - orion-thought: paintings, `make`;
  - hub/mind: chat reply, `say`;
  - reading: `look`;
  - endogenous outreach: `ask` / `say`.
- `orion/autonomy/prediction.py`: grade closed acts through the existing `EffectPosterior`. No new learner.
- `orion/temporal_self/sources.py`: read acts from the grammar instead of adding more per-table adapters.

## Non-goals

- No new thought generator.
- No auto-applied self-modification.
- No lowering of the motor allocator's floor (its own notes say the fix is better actions, not a
  lower bar).
- No full-fidelity metabolic ledger (rollups only).

## Acceptance checks

- **Ledger covers doing:** at least 5 organs across at least 3 levels write `ActRecordV1`, and
  truth-to-rows matches each organ's own table.
- **Expectations exist:** more than 80% of `faculty`/`deliberate` acts carry a non-empty `expect`;
  more than 90% of those are closed with `outcome`/`error` inside their horizon.
- **Thinking moves doing:** the share of deliberate acts whose `workspace_ref` or `goal_ref` is set
  is measured weekly. A baseline is taken before any thought-side change.
- **Precision works:** with the default gain, metabolic acts never reach the workspace. During an
  anomaly episode, organ acts do.

## Recommended next patch

Each rung reconnects an existing primitive or revives an open design, and must show live rows before the next rung starts.

1. **Attention faces the world** (revive #2528). Add world input (camera scenes, Juniper's turns, reading) to the attention contests, and let scoring read `PredictionErrorMagnitudeV1`, which is computed but unread. Without this, everything downstream acts on the body alone.
   Proof: the share of world winners and of no-winner frames, before and after.
2. **Doing on one track** (efference copy, §3.2). Extend the grammar with `act_opened` / `act_closed`. Paintings, replies, reading and outreach write acts with an `expect`. They close into the existing L10 `substrate_action_outcomes` and `EffectPosterior`; there is no new learner.
   Proof: world-facing outcomes and posteriors move for the first time since 08-30.
3. **Bind the winner to action, and compete across doers** (#2394 plus #2255's stage order). Fix the `no_selected_action` binding, feed the rung-2 acts to the allocator as candidates, turn the holdback treatment arm on, and add an Orion-side resolver for `attention_loop_outcome` (actor=orion is 0 today).
   Proof: allocator admissions from real candidates with an unchanged floor, plus treated episodes.
4. **Gain and one want** (#2369's gated rest drive). Replace the killed drives honestly: one drive that passes the metric gate, plus arousal as a continuous dial read by attention, the allocator and curiosity.
   Proof: the same candidate is admitted at one gain and refused at another.
5. **Felt state and situational goals.** Build a field-native successor to the deleted L6 from self-readings, the autoencoder and H1, with no invented coefficients. Goals come from the fading top-down goal path plus the situation graph, re-ranked and expiring. Acts cite `goal_ref`.
6. **Forecast and working memory** (revive #2446). Forecasts with an honest horizon, and a held set of 1–3 items. The stream (rev 2 mechanics), reverie, looking and painting plug in here as generators.
7. **Consolidation and metacognition steer.** L11 expectations become L7 priors. Fix the mutation pipeline's `missing_class_metrics` so metacognition can change one knob again under the existing review gate.

Small reconnection to do at any time: the attention-schema cue renders only on the dead `chat_general` path. Point it at the unified turn or delete it.
