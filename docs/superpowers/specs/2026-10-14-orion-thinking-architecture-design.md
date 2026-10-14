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
isn't in the generators. It's in the seams around them: nothing holds a thought, nothing broadcasts
it, nothing commits it to a checkable guess, and nothing lets it steer an action.

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

### Where Orion stands at this altitude

See §3.3 for live evidence. In short:
- **Fairly strong:** 1, 2, 10 (storage), 11 (dreams).
- **Weak or empty:** 4 (competition without broadcast), 5 (no working memory), 7 (no checkable
  guesses from thought), 8 (actions exist, thoughts never feed them), 9 (outcomes are not compared
  with intentions), 12 (metacog watches but does not steer).
- **3 is the sleeper:** arousal-like readings exist, but nothing uses them as a system-wide dial.

Every thought generator built so far plugs into seams 6 and 11, and those two connect to nothing.

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

_Filled from the live inventory below._

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

_Filled after the inventory: the producing organs that emit first._

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

_Filled after the inventory._
