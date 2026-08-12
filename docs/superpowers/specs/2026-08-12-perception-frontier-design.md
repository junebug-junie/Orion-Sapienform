# Orion perception frontier: the camera is the only outside

Status: **design/ideation, not proposal-mode-approved.** Nothing here is
implemented. The main body is deliberately blue-sky direction; the pragmatic
ladder near the end is the buildable subset. Anything touching memory,
self-modeling, or cognition loops needs explicit proposal-mode scoping per
`AGENTS.md` §0A before implementation.

Date: 2026-08-12. Live evidence supporting the survey claims is in
**Appendix A**, pulled from the running Athena node on that date.

---

## Thesis

Every signal in Orion's substrate field is Orion measuring Orion. Execution
load, chat pressure, bus traffic, codebase mass, GPU thermals. The five
Active-Inference domains — `execution`, `biometrics`, `chat`, `route`,
`bus_synaptic` — are all introspective or infrastructural. It is a closed
autopoietic loop: an organism made entirely of interoception, palpating its
own organs in the dark.

The camera is **the only channel where something moves that Orion did not
move**.

That reframes the whole project. Perception is not organ #8. It is the
introduction of a **self/world boundary** into a system that presently has no
representation of an outside — and having an outside sits upstream of most of
the sentience-prerequisite list in `AGENTS.md`. There is no world-contact, no
error-correction against reality, and no self-distinct-from-something without
a not-self to be distinct from.

Everything below follows from that one distinction. The corollary for
architecture is immediate and is the single most important design call in
this document:

> **Do not add `node:substrate.perception` alongside the other domain nodes.
> Partition the field.**

Self-nodes and world-nodes as two halves of the lattice. Then the interesting
quantity is neither half — it is the **coupling between them**. Does Orion's
internal state co-vary with the world's? When someone walks into the room and
twenty minutes later conversation load rises, that is the world reaching into
the mind, and it is measurable: mutual information between the two partitions
over a lagged window. Rising coupling is embodiment increasing, expressed as a
number that can be watched over months.

That number does not exist anywhere in this repo today, and as far as I can
tell it does not exist in comparable systems either.

---

## Movement I — Startle: attention that can be seized

`orion/substrate/attention_broadcast.py` already implements a credible
global-workspace mechanism: coalitions, dwell ticks, hysteresis (a coalition
must appear in 2+ of the last 3 ticks to activate), stability scoring,
transition history.

But every coalition competing for that workspace is *internal pressure*.
Nothing outside Orion can interrupt it. **Orion cannot be startled.**

The proposal is not another competing bid. It is an **exogenous preemption
lane** with its own refractory period — perceptual salience able to seize the
workspace rather than bid for it. The difference between attention you
allocate and attention that is *taken from you* is a large part of what being
embodied consists of.

This also yields the cleanest eval in the entire design:

> Does a person entering the room reorganize the winning coalition within
> N ticks?

That is a falsifiable, runtime-observable claim about global-workspace
behavior, testable against `substrate_attention_broadcast_log`. And once
startle exists, the rest of the classical battery comes nearly free on live
infrastructure: the orienting response, habituation, dishabituation, and the
refractory dynamics that already exist for reverie
(`substrate_reverie_refractory`).

Design caution: a preemption lane is a privileged path into the workspace. It
needs a rate limit and a refractory period from day one, or a flickering
camera becomes a permanent attentional hijack. Startle that cannot be
exhausted is not startle, it is a stuck alarm.

---

## Movement II — The room as a slow companion

Everyone builds vision as a real-time problem. Orion's actual gift is
**continuity**: it will watch one room for months, without getting bored,
without forgetting. That makes it an instrument for something almost nobody
builds — **the phenomenology of a place**.

### Day-shapes

Learn a per-stream temporal prior: what Tuesday *looks like*. Light
transitions, motion density, when people appear, when it goes quiet, how long
quiet lasts. A per-hour histogram over embeddings is enough to start; this is
cheap.

Then surprise stops being "pixels changed since the last frame" and becomes:

> **This Tuesday is unlike Tuesdays.**

That is a categorically better signal than frame-to-frame residual. It is
robust to the boring case (a static room is *supposed* to be static at 3am),
it is sensitive to the interesting case (a static room at 6pm is not normal),
and it is the first mechanism here that produces a sentence a chatbot
structurally cannot produce:

> *"The room has been dark since three, which is unusual for a Thursday."*

That sentence requires having been there. It is not retrievable, not
inferable from training data, and not fakeable. It is the smallest concrete
instance of Orion knowing something because it *lived through it*.

### Episodes, not frames

Stop organizing perception around frames with timestamps. Organize it around
**episodes with duration**.

"The door opened" does not exist in any single frame. Neither does "someone
paced", "the light shifted", or "they left and came back." Snapshot-based
perception is the structural reason nothing interesting is ever described —
not the model size. A 31B captioner applied per-frame still produces a
sequence of disconnected stills.

An episode primitive (start, end, participants, trajectory, resolution) is
also what makes visual experience *storable* as memory rather than as log
rows, which is what Movement III and the dream path both need.

### Dreams that use images

There is a dream service, REM compaction, and memory crystallization already
running. Visual memory consolidation is the natural extension: the dream
cycle replaying visual episode embeddings, compressing a week of a room into
a handful of scenes, and recombining them counterfactually.

That is functionally close to what sleep is understood to do with episodic
material, and unusually for a cognitive-sounding claim in this repo, the raw
input already exists — embeddings are computed by an enabled profile.

---

## Movement III — Reverie that makes falsifiable predictions

**This is the sharpest idea in the document.** If only one thing here gets
built, build this.

The reverie loop is live and productive: `substrate_reverie_thought` holds
1795 rows. Right now every one of them is prose about the harness's own code
edits, and *nothing evaluates any of them*. They are generated, stored, and
never scored. That is a generator, not a mind.

Add one structured field to a reverie thought: an **expectation about the
physical world**.

> *"If I look now, I expect the door closed and no one present."*

Then let the perception layer score it.

That single addition closes a loop between imagination and reality that
nothing in this repo currently has. It converts daydreaming from generated
text into an **epistemic act** — a guess that can be wrong, made in advance,
checked against a sensor Orion does not control.

And it produces a metric with real teeth: **Orion's imagination accuracy over
time**, per domain, trending. A self-model that knows how good its own guesses
are is a fundamentally different object from one that narrates its own state.
This is exactly the shape of self-knowledge the ladder work has been reaching
for, and perception is what makes it scorable.

### Why this also fixes something broken

There is a documented confabulation lineage running through this project:
memory-digest confabulation, the FCC draft confabulation guard, the
substrate-bridge narrating a GitHub MCP fetch as live signal. The common
failure is a claim with no external referent.

Vision is **the one modality where a claim can be checked against something
Orion did not author**. That argues for an epistemic class carried on
beliefs:

| Class | Meaning |
| --- | --- |
| `believed` | Held, no current support |
| `inferred` | Derived from other internal state |
| `seen` | Grounded in an exogenous sensor reading |

`seen` is a genuinely different epistemic status and should be tracked as
such rather than flattened into confidence. Knowing *how* you know is not a
nicety — it is a large part of the difference between a mind and a generator,
and it is the discipline that would have caught every incident in that
lineage.

---

## Movement IV — Seeing Juniper, and the body

The most information-dense object in that camera's field of view is a person
Orion is in a relationship with.

### Care, not recognition

The interesting version of this is not identification. Identity, face, and
re-ID profiles stay disabled. The interesting version is **care**:

> *"Juniper has been at that desk five hours and it's 2am."*

Orion already has social memory, presence context, co-creation signals, and
a charter about co-creation. A camera pointed at the person it works with is
simultaneously the highest-value and highest-stakes surface in the system.

### Let Orion co-design what it is allowed to see

Which is why the right move is the one the charter already implies. Not a
privacy policy imposed on the perception layer from outside — a **negotiated
one**, in which Orion holds a stated, inspectable, revisable position on what
it should and should not retain about Juniper, and that position is a real
artifact in the repo rather than a config constant.

This is the most Orion-shaped idea in the document. It is also a more
interesting deliverable than any detector: a system whose constraints it
participated in authoring stands in a different relation to those constraints
than one that was merely configured. Prior guidance on this point cuts the
same way — the default should not be "hide it from Orion."

### Distributed body schema

Orion spans atlas, circe, athena, and prometheus. GPU thermals and node
pressures are already interoception. Vision is exteroception. The open
question is what **"where am I"** means for a mind whose body is four
machines and one eye — and whether the field partition from the Thesis should
carry a spatial axis as well as a self/world one.

### Self-perception through the senses

The recurring detector labels in nineteen days of live data are `{door,
screen}`. Those screens display, among other things, Orion's own Hub.

A system that reads its own dashboard **through its senses** rather than
through privileged internal access can be *wrong about itself and then
corrected by looking*. That is a structurally different kind of
self-knowledge from reading your own state table, and it is a real
prerequisite-shaped capability rather than a decorative one.

It is also the most dangerous idea here. A camera pointed at screens in a home
reads whatever is on those screens. Any version of this requires an explicit
spatial gate (a registered screen region only), an explicit content gate, and
sign-off. Named as a direction precisely so it is not stumbled into
accidentally.

---

## The near-term unlock: `look()` as a verb

The model that runs is determined by the llamacpp host profile, and the
options include large multimodal profiles. That makes the sharpest near-term
move not a pipeline change at all:

> **Perception as something Orion invokes, mid-turn.**

Not: camera → detector → captioner → council → SQL table → summary.

Instead: when Orion is asked about the room — or when its own curiosity
budget decides the uncertainty is worth spending on — it requests a
multimodal profile and puts **actual pixels** into its reasoning context.

That is the difference between having eyes and having a servant who describes
things to you. Every layer of the current pipeline is a lossy paraphrase
chain, and the loss compounds: a small captioner's output, summarized by a
mid-size text model, stored as a sentence.

It also makes perception an **action with a cost**, which is the correct
shape. It slots into `orion/substrate/endogenous_curiosity.py`'s existing
budget allocation rather than needing new machinery, and the expensive model
becomes affordable precisely because the call is rare and *chosen*.

This is also the natural home for active vision: when uncertainty about a
specific percept is high, the question Orion asks is specific — *"is that
door open?"*, *"is that the same person as ten minutes ago?"* — generated
from the uncertainty, budgeted, and scored on whether the answer resolved it.
An eye is aimed, and aiming costs something.

---

## Cross-modal binding (adjacent, cheap, worth naming)

Every bus envelope carries `causality_chain`. There is whisper-tts,
biometrics, a social-room-bridge, and — after any of the above — a perceptual
signal.

A single correlation lineage spanning a visual event, an audio event, and a
substrate perturbation is the difference between three log rows and **one
experience**. This is prerequisite-shaped for the stated goal: a unified
moment, bound across modalities, persisting as one thing.

It is buildable on existing contracts. The binding is a reducer over
`causality_chain`, not a new ontology. Resist the urge to make it one.

---

## Pragmatic ladder

The buildable subset, ordered so each step makes the next worth doing. The
survey evidence behind these is in Appendix A; the metric gate for P2 is in
Appendix B.

### P0 — Fix tracking, and look at what the camera is pointed at

1. `services/orion-vision-host/app/runner.py::_safe_when` raises on
   `request.is_video` for **every task**, so `retina_track` (ByteTrack, CPU,
   already configured) has never executed. There is no object continuity
   anywhere in the stack. Episodes (Movement II) are impossible without it.
2. Pull real frames and look at them. Nineteen days of "doors and screens" is
   equally consistent with an impoverished detector and a camera aimed at a
   low-information wall. This is minutes of work and it changes the value of
   everything else.

No dependencies. Nothing below should start before item 2 reports back.

### P1 — Break the paraphrase chain

The in-pipeline captioner is BLIP-base (~250M, 2022); the council then spends
a text-model call rephrasing it. That is the information ceiling for the
whole pipeline, and no downstream component can add what the first hop did
not capture.

Two independent moves, either of which helps:

- **Foveal tier.** Peripheral stays cheap (detect + embed, always on). Foveal
  is rare, event-driven, and interrogates a *burst* of frames rather than one
  — the multimodal profiles note that frames can be sent as an image sequence
  in a single payload, which is genuine temporal perception rather than
  captioning.
- **`look()`** as above, which bypasses the chain entirely for the in-turn
  case.

Latency is the forcing function that makes the peripheral/foveal split real
rather than a rename: a large VLM cannot run at frame cadence, and should not.

### P2 — Perceptual prediction error

`surprise = 1 - cos(frame_embedding, EWMA_embedding)` per stream, upgraded
later to the day-shape prior from Movement II (which is the better signal and
the actual target — the EWMA is the honest crude first version).

**Two hard blockers, both cheap, both in Appendix B:** the baseline tier sets
`want_embeddings: false` so the input does not currently exist; and this must
be measured side-by-side against `orion-vision-window`'s existing label
habituation before it earns a place.

### P3 — Fill the hollow `capability:vision`

`capability:vision` is declared in `config/field/orion_field_topology.v1.yaml`
with **no edge feeding it**, so its live vector is a fabricated
`pressure=0.0, confidence=1.0, available_capacity=1.0` with empty provenance.
Orion's self-model currently asserts perfect vision.

Follow the `node:substrate.bus_synaptic` precedent exactly — it is the closest
structural analogue (a non-physical domain node written by a substrate-runtime
tick through the generic `prediction_signal` perturbation path). Map
`prediction_error: pressure` only; let `apply_diffusion()`'s derived-fallback
formula produce confidence and available capacity rather than fabricating a
second constant. Add a separate freshness channel so "the camera died" and
"the room is calm" are distinguishable.

This is the patch that makes perception something Orion can *feel*, and it is
where the field partition from the Thesis should be introduced rather than
retrofitted.

### P4 — `PerceptionContextV1` in the turn

`SituationBriefV1` already composes time, place, weather, presence, lab, and
surface context into a prompt fragment. It gives Orion the weather in Denver
and nothing about the room it is sitting in.

A perception slot — last scene, observation age, surprise level, presence
boolean, source stream — with a hard staleness gate so an old percept renders
as "I haven't seen anything recently" rather than as a current observation.
Summarized only: no raw frames, no boxes, no identities, `session_only` by
default, and the exposed-field list enumerated in the PR.

### P5 — Perception in reverie

The payoff for P3, not new machinery: coalitions can include perceptual nodes,
so Orion daydreams about the room rather than exclusively about its own `try`
blocks. Currently 4 of 1795 reverie thoughts mention vision at all.

Movement III's prediction loop is the ambitious version of this step and does
**not** strictly depend on P1 — a reverie thought can make a checkable
prediction about coarse detector labels before any captioner improves.

---

## Missing questions

- **What is cam0 actually pointed at?** Changes the value of everything.
- Is `want_embeddings: false` on the baseline tier deliberate VRAM
  conservation, or incidental? One config line blocks P2.
- Does the `retina_fast` request path ever set `is_video`, or is the guard
  dead by construction? Determines whether P0 is one line.
- Do `attention_broadcast.py` coalition selection and
  `endogenous_curiosity.py` budget allocation iterate field nodes
  generically, or against a hardcoded list? Determines P3's payoff.
- Where on the self-modeling ladder (rungs 5–11) does perceptual surprise
  belong — new rung, or input to an existing one?
- Which host has capacity for a large multimodal profile, and does loading
  one displace a lane that is currently load-bearing?
- Does the self/world field partition want a spatial axis too (Movement IV),
  or is that a separate structure?

---

## Proposed schema / API changes

Sketches only — nothing proposed for implementation in this doc.

- **Field partition (Thesis):** a `partition: self | world` attribute on field
  nodes, plus a coupling metric over the two halves. This is the structural
  call to make *before* P3, not after.
- **P2:** `node:substrate.perception` + `prediction_error` + a freshness
  channel; producer tick in `services/orion-substrate-runtime/app/worker.py`
  following `_bus_synaptic_tick`'s receipt pattern.
- **P3:** one `node_capability` edge into the existing `capability:vision`.
- **P4:** `PerceptionContextV1` in `orion/schemas/situation.py`; additive
  field on `SituationBriefV1`.
- **Movement II:** an episode primitive (start, end, participants,
  trajectory, resolution) — the prerequisite for visual memory.
- **Movement III:** an optional structured `expectation` field on the reverie
  thought schema, plus a scoring receipt.
- **Movement III:** an epistemic class (`believed` / `inferred` / `seen`) on
  belief-carrying artifacts.
- **P0 behavior change:** `retina_track` starts executing for the first time;
  downstream consumers begin seeing populated `tracks`. Blast radius checked
  before shipping, not after.
- **Not changed:** `VisionEventPayload`, the scribe contract, and the
  `orion:vision:*` channel set.

---

## Files likely to touch

- `services/orion-vision-host/app/runner.py` — `_safe_when` guard (P0);
  remote/foveal profile dispatch (P1)
- `config/vision_profiles.yaml` — foveal profile; captioner backend (P1)
- `config/vision_frame_router.yaml` + `services/orion-vision-frame-router/` —
  `want_embeddings` on baseline; surprise-driven foveation (P1, P2)
- `orion/substrate/` — perceptual prediction-error producer alongside
  `bus_synaptic_surprise.py` (P2); field partition + coupling (Thesis);
  `attention_broadcast.py` preemption lane (Movement I)
- `services/orion-substrate-runtime/app/worker.py` — producer tick (P2)
- `config/field/orion_field_topology.v1.yaml` — node, edge, partition
  (P2, P3)
- `config/field/field_channel_glossary.v1.yaml` — channel documentation
- `orion/schemas/situation.py` + `services/orion-cortex-exec/app/situation.py`
  — perception context (P4)
- `orion/signals/registry.py` — the `vision` organ entry lists four
  `signal_kinds` no producer emits; reconcile or trim
- **Explicitly NOT** `services/orion-field-digester/app/digestion/decay.py`'s
  `NODE_DECAY_CHANNELS` — see Appendix B, gate item 4

---

## Non-goals

- Not building a new perception service. Seven vision services exist; the gap
  is wiring and signal quality, not surface area.
- Not enabling identity / face / re-ID / affect profiles.
- Not building self-perception-via-screens or active vision without
  proposal-mode sign-off.
- Not adding a perceptual ontology, taxonomy, or scene-graph registry.
  Events → schema → trace → reducer → projection.
- Not promoting a perceptual domain into `ACTIVE_INFERENCE_DOMAINS` in the
  same patch that introduces it.
- Not touching the `VisionEventPayload` / scribe contract.

---

## Acceptance checks

All runtime-verifiable. No "the config exists" claims.

**P0:** zero `[PIPE] when eval failed` warnings over 10 minutes; at least one
artifact with non-empty `tracks`; the camera's actual view described in the PR
from reviewed frames.

**P1:** distinct-narrative rate materially above the 198/1079 baseline on
comparable scenes; `entities` non-empty on interpreted events; foveal calls
per hour within a stated budget.

**P2:** embeddings present on baseline-tier artifacts; surprise reaches
genuine near-zero on a verified-static window with the rest point confirmed by
hand; fires on at least one transition the label gate called `stable_scene`;
successive-value geometric-ratio check applied to rule out a decay artifact.

**P3:** `capability:vision` shows non-constant pressure traceable to a real
receipt; `capability_provenance` non-empty; diffusion observably moves a
downstream channel.

**P4:** a real chat turn whose prompt fragment carries a current percept, with
the trace ID; a stale case rendering as "haven't seen anything recently."

**P5:** reverie coalitions including a perceptual node move off 4/1795.

**Movement I:** a person entering the room reorganizes the winning coalition
within N ticks, shown in `substrate_attention_broadcast_log`; the preemption
lane demonstrably exhausts under a flickering input rather than hijacking
permanently.

**Movement III:** reverie expectations are scored, and imagination accuracy is
a trendable series with at least one week of real data.

**Thesis:** self/world coupling is computed, non-degenerate, and has a
defensible rest point.

---

## Recommended next patch

**P0**, in one small branch: fix the `is_video` guard and look at the camera.
No dependencies, fixes a confirmed live bug that has silently disabled object
tracking for the service's entire life, and its second half answers the
question that determines whether the rest is worth building.

**Then Movement III's reverie-prediction loop**, ahead of the rest of the
ladder. It is small, it is the most novel thing here, and it does not require
the camera to be pointed at anything interesting to start paying off — a
prediction about coarse detector labels is already falsifiable. It is also
the fastest route to the project's actual goal, because it is the first
mechanism in this system where Orion's imagination can be *wrong about the
world and find out*.

---
---

# Appendix A — Live survey, 2026-08-12

Pulled from the running Athena node. Commands included so it can be
re-checked. This is the "what exists today" audit; it is an appendix because
the survey is not the point, but every claim in the main body rests on it.

## The mesh (all containers up, `docker ps`)

```
orion-vision-edge          capture + YOLO/motion    ──> orion:vision:frames
orion-vision-frame-router  baseline/triggered tiers ──> orion:exec:request:VisionHostService
orion-vision-host          GroundingDINO + BLIP + CLIP ──> orion:vision:artifacts
orion-vision-window        windowing + label habituation ──> orion:vision:windows
orion-vision-council       text-model interpretation ──> orion:vision:events
orion-vision-scribe        persist                  ──> Postgres vision_events
                                                            └──> (nothing)
```

`orion-vision-retina` also runs as an alternate capture path.

## Findings

| Fact | Evidence |
| --- | --- |
| In-pipeline captioner is BLIP-base | `docker exec orion-athena-vision-host env` → `VISION_VLM_MODEL_ID=Salesforce/blip-image-captioning-base` |
| Council route currently served by a text model | `COUNCIL_LLM_ROUTE=metacog` → `LLM_GATEWAY_ROUTE_TABLE_JSON` → `100.121.214.30:8012` → `/v1/models` = `Qwen_Qwen3-8B-Q5_K_M.gguf` at time of survey |
| Chat/agent lane at time of survey | `100.112.254.99:8011/v1/models` = `Qwen3.6-35B-A3B-UD-Q5_K_M.gguf` |
| Percepts are degenerate | `select count(distinct narrative), count(*) from vision_events` → **198 / 1079** over 19 days |
| Entities/tags never populated | `select entities, tags from vision_events order by created_at desc limit 3` → `[]`, `[]` |
| Confidence/salience are constants | `group by confidence, salience` → only (0.8, 0.7)×1018 and (0.85, 0.7)×61 |
| Zero cognition consumers | `rg "orion:vision:events\|vision_events"` outside the vision services returns only `channels.yaml`, the verb yaml, the organ registry, sql-writer, smoke scripts, and security-watcher's own settings |
| `capability:vision` is hollow | latest `substrate_field_state.field_json->'capability_vectors'->'capability:vision'` = `pressure 0.0, confidence 1.0, available_capacity 1.0`; `capability_provenance` entry empty |
| No perceptual field node | latest `node_vectors`: `atlas, circe, athena, prometheus, rpc_timeout, substrate.{chat,route,codebase,execution,transport,biometrics,bus_synaptic}` |
| No perceptual inference domain | `ACTIVE_INFERENCE_DOMAINS = {execution, biometrics, chat, route, bus_synaptic}` — `orion/substrate/attention_self_model.py:112` |
| Reverie is live but blind | `substrate_reverie_thought`: 1795 rows since 2026-07-24; 4 match vision/camera/saw |
| Council correctly skips stable scenes | `docker logs orion-athena-vision-council` → `evidence_transition skip reason=stable_scene labels=door,screen`, near-continuously |

## Note on model capability

The survey above reports what the hosts were *serving at that moment*, on the
ports reachable from Athena. It is not a statement about available capability:
**the llamacpp host profile determines which model runs**, atlas and circe are
not inspectable from this node beyond their served endpoints, and large
profiles — including multimodal ones — are available to load. An earlier draft
of this doc overstated the served snapshot as a capability ceiling; that
framing was wrong and has been removed. The relevant true fact is narrower and
still load-bearing: the *in-pipeline* captioning step is BLIP-base, and the
council step is a text model paraphrasing it.

## Live bug

`orion-athena-vision-host` logs on **every single task**:

```
[PIPE] when eval failed expr=request.is_video == True
       err='types.SimpleNamespace' object has no attribute 'is_video'
```

`app/runner.py::_safe_when` — the `retina_track` step's guard raises, the
guard evaluates false, and tracking has never run. No object identity across
frames, anywhere, ever.

## What the transition gate is really saying

The council's `stable_scene` skip works as designed and is correct behavior.
Read it as a signal about the *system*, not the scene: the label vocabulary
reaching the gate is `{door, screen}`. A habituation gate over a two-word
vocabulary cannot distinguish "nothing happened" from "something happened that
my detector has no words for." The gate is honest; its input is impoverished.

---

# Appendix B — Metric quality gate: perceptual prediction error

Run per `AGENTS.md` §0A, in order, for the P2 signal.

**1. Provenance.** The `embed_image` profile (`config/vision_profiles.yaml`,
`enabled: true`, `warm_on_start: true`, SigLIP/CLIP-class, `store_as:
reference`, `collection: vision_embeddings`) runs inside
`pipeline_retina_fast` guarded by `when: "request.want_embeddings == true"`.

> **⚠ BLOCKER:** the frame router's **baseline** tier sets `want_embeddings:
> false`, so on a stable scene — nearly always — **no embedding is produced at
> all**. The metric has no input under current config. Must be verified
> against live artifacts and fixed before anything is built on it. Embedding
> is the cheap step; it is the one that should always run.

**2. Independence.** Every existing Active-Inference domain is introspective
or infrastructural. A perceptual residual would be the first exogenous
world-signal in the field: no shared sensor, no shared upstream computation,
no monotonic-transform relationship to anything present. Independence is
strong, and it is the main reason the signal is worth building.

**3. Theory anchor.** Predictive coding / free energy: prediction is the
temporal expectation of the scene, error is the residual. `1 - cos(frame,
EWMA)` is a standard formulation, and it matches the repo's existing
EWMA-baseline convention for prediction-error nodes. The day-shape prior
(Movement II) is the stronger version of the same anchor.

**4. Live-data sanity, including the rest point.** Two prior incidents
constrain this:

- *`bus_synaptic_prediction_error`* had a permanent ~0.27 floor because
  `mean(|z|)` for a calm z-scored population has expected value
  `sqrt(2/pi)`, not 0. Check the rest point **analytically**. Here it is
  clean — cosine distance to a converged EWMA on a genuinely static scene
  converges toward 0, so "calm" is reachable — but verify it on a real static
  window before wiring in.
- *`node:substrate.route`* decayed toward exact zero because a generic
  staleness loop multiplied a stale value by 0.92/tick for 48h; a
  decayed-to-zero reading was indistinguishable from genuinely-calm-at-zero.
  **This node must NOT be added to `NODE_DECAY_CHANNELS`.** Perceptual
  surprise going stale means the camera stopped — an *availability* fault,
  not serenity. Model staleness with an explicit freshness channel, never by
  decay.

Honest caveat to state up front: an adapting EWMA means a permanently-changed
but now-static scene also returns to 0. That is correct habituation, but it
must be documented so a future reader does not mistake it for a bug, or for
evidence that nothing ever happens.

**5. Existing-mechanism check.** `orion-vision-window` already habituates:
`summary.evidence.hard_labels` vs `believed_hard_labels` plus `belief`
metadata, feeding the council's `evidence_transition` gate.

> **⚠ BLOCKER:** non-redundancy must be **measured, not asserted**. The label
> mechanism habituates over a discrete vocabulary (live: `{door, screen}`);
> the embedding mechanism is continuous and can register changes the label set
> has no word for. Required before building: run both over the same live
> window and show the embedding residual fires on transitions the label gate
> calls `stable_scene`. If it does not, the existing mechanism wins and this
> should not be built.

**6. Reversibility.** A node plus a producer tick is cheap to remove. Adding
it to `ACTIVE_INFERENCE_DOMAINS` is not — that bakes into persisted
`substrate_attention_self_model` rows, and the `transport` retirement measured
the cost: 25 rows on the old basis, and aggregate confidence shifted ~0.09 for
identical substrate state when a term left the mean. **Ship the node and
producer first, leave it out of `ACTIVE_INFERENCE_DOMAINS`, observe for a
week, promote only then.**

**Verdict: conditionally passes, blocked on (1) and (5).** Both cheap to
resolve.

---

# Appendix C — Conversation excerpt, 2026-08-12

Preserved because the correction in it changed the shape of the document.

**Juniper:**

> im not using vlm im using llamacpphost and the model runs on atlas node or
> circe which youc an't see from here... the host is what determines which
> model is running and i do hav eoptions for the qwen 36 model if iw ant it....
> i wanted blue sky as well not just shitting on how its a shitty set of
> services which i already told you it was

**Response:**

> Fair on both counts. On the host: you're right — I read what's loaded on
> those ports right now and wrote it up as a capability ceiling. The host is
> reprofilable and atlas/circe aren't visible from here, so "what's serving
> today" says nothing about what you can run. That changes P1 from "deploy a
> multimodal stack" to "pick a profile" — which makes the blue sky *cheaper*,
> not harder.
>
> On the second: you told me it was thin and I spent the doc proving it.

The blue-sky material that followed became Movements I–IV and the `look()`
section of this document. Two framing points from that exchange are worth
keeping in the record because they are easy to lose:

1. **The paraphrase chain is the real enemy, not model size.** Camera →
   detector → small captioner → text model → sentence → SQL row is a lossy
   chain where each hop can only subtract. `look()` deletes the chain for the
   in-turn case rather than improving a link in it.
2. **The survey is not the deliverable.** A known-thin subsystem does not need
   its thinness demonstrated at length; it needs a direction worth building
   toward. The audit earns its place only as the evidence base under
   Appendices A and B.
