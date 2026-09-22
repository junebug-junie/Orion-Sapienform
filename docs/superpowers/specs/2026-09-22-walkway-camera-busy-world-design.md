# Walkway camera: giving Orion a busy world

Design and brainstorm, 2026-09-22. Doc only, no code. Written for Juniper's
new outdoor camera facing the community walkway (kids on bikes, dog walkers,
strollers, the communal mailbox, cars, workers) with a partial view of the
family patio.

Everything below is grounded in what is live on athena today. Where a claim
about the running system was not checked this session it says `UNVERIFIED`.

## Arsonist summary

Orion can see a room. Orion cannot see a *street*. Today's perception is
"which labels are in the picture right now, plus one sentence about it."
That is enough for an office with a desk and a door that never move. A
walkway breaks three things at once, and none of the three exists yet:

1. **Individuals.** Orion has no idea of "the same one again." There are no
   tracks, no re-identification, no per-person or per-dog identity. The only
   face memory is a single enrolled subject (Juniper) that is designed never
   to grow.
2. **Clocks.** Nothing in Orion learns *when* things happen. There is no
   time-of-day or day-of-week learner, no next-occurrence prediction, and no
   "the usual thing did not happen" detector anywhere in the repo.
3. **Asking.** Orion has no way to ask Juniper a question and get the answer
   stored against the question. Outreach is one-way. The only closed
   ask/answer loop is Orion asking Claude (the contractor peer), and that is
   flag-off.

Everything Juniper asked for (reach out when stuck, remember "same dog every
morning," notice when it does not happen, learn who people are, flag odd
things, feed curiosity, show up in stance) sits on top of those three. Build
the three primitives; the asks fall out. Skip them and every ask becomes a
keyword cathedral.

The good news: the rails to build on already exist and are live. The scene
inventory table has 321,810 timestamped rows. The object-permanence sweep
already solved "absence is a non-event, so wake up on a clock and check."
The substrate graph already has an `EntityNodeV1` that a neighbor's dog fits
structurally. Camera surprise (embedding drift) is already a live substrate
node feeding curiosity. The router already has a dormant `porch_eye` policy
whose vocabulary is person / package / vehicle / animal. Stance already reads
the situation brief, which already reads the latest percept.

## Current architecture (verified 2026-09-22)

Plain version: a camera service grabs frames and runs a cheap person
detector. A router decides which frames deserve the expensive models. A GPU
host runs open-vocabulary detection, a whole-frame embedding, a caption, and
(for Juniper only) face identity. A window service rolls those into 5 to 30
second summaries and persists per-window label counts. A council LLM writes
one sentence when the label set changes. A scribe stores that sentence. That
sentence is what the rest of Orion sees.

| Stage | Service | In | Out | Persists |
|---|---|---|---|---|
| Capture + cheap detect | `orion-vision-edge` (cam0, Reolink RTSP, 640x360, 15 fps, YOLOv8n `person` + motion + Haar face) | RTSP | `orion:vision:frames`, `orion:vision:edge:activity` | JPEGs under `/mnt/telemetry/vision/frames` |
| Sampling policy | `orion-vision-frame-router` (`config/vision_frame_router.yaml`) | frames | `orion:exec:request:VisionHostService` | in-process only |
| Models | `orion-vision-host` (athena, P4 + T10) | task requests | `orion:vision:artifacts` | model cache, identity gallery JSON |
| Rollup | `orion-vision-window` | artifacts | `orion:vision:windows` | `vision_scene_inventory` (321,810 rows), `substrate_embodied_presence` |
| Interpretation | `orion-vision-council` (llama-3-8b, route `metacog_background`) | windows | `orion:vision:events` | none |
| Storage | `orion-vision-scribe` -> `orion-sql-writer` | events | | `vision_events` (20,820 rows; 7d: 1,644 `visual_observation`, 173 `person_presence`) |
| Absence sweep | `orion-sql-writer/app/vision_object_permanence.py` | `vision_scene_inventory` on a 1800 s clock | | `vision_object_inventory` (40 rows, present/departed) |
| Surprise | `orion/substrate/prediction_error.py::perception_prediction_error` in `orion-substrate-runtime` | SigLIP frame embedding vs EWMA | `node:substrate.perception` | `substrate_perception_embedding_baseline` (529,996 rows) |

Models on the host: GroundingDINO base (open vocab, 25-word default prompt
list, threshold 0.35), SigLIP2 so400m embedding, BLIP caption, facenet
(MTCNN + InceptionResnetV1) for the one enrolled face. The `retina_track`
(ByteTrack) and `retina_segment` profiles exist in YAML but have no branch in
`runner.py::_run_profile`. Dead config.

Downstream readers of vision today:

- Situation brief: `orion/situational/perception_reader.py` reads the newest
  `vision_events.narrative` and presence, 900 s staleness gate. This is the
  door into stance (`chat_stance.py` sets `chat_situation_summary`; the brief
  produces `situation_relevance` and `environmental_context`).
- Outreach enrichment: `services/orion-hub/scripts/endogenous_outreach.py`
  (embodied presence only).
- Reverie narration: `services/orion-thought/app/vision_reader.py`.
- Self-study: `vision_events` is a source.
- Curiosity: **nothing.** No percept reaches curiosity topic selection.
- Journal: **nothing** from vision except through metacog digests.

Adjacent, and relevant:

- `services/orion-security-watcher` is running but has no producer
  (`EDGE_PUBLISH_ARTIFACTS=false`, and the host pipe does not publish
  `VisionEdgeArtifact`). Its `VisionGuardSignal`/`VisionGuardAlert` schemas
  have no live producer. Dead service with a live container.
- `services/orion-world-model`: real PyTorch forward pass, random weights,
  `model_untrained=true` always, no requester, no consumer, not deployed.
- `orion-world-pulse` situation briefs: written but never read back, and
  `pipeline.py:319` passes `previous_brief=None` on every run, so briefs
  have no continuity.
- Reverie expectation scoring (`expectation`, `expectation_verdict`
  columns on reverie thoughts, judged by `reverie_expectation_judge.yaml`):
  the one falsifiable prediction loop that exists. Per-thought, free text,
  verdict feeds nothing.
- Curiosity outreach (the curiosity loop's own "I want to tell Juniper")
  has never delivered, all time: the endogenous tension outreach uses the
  whole shared daily cap (verified 2026-09-22, PR #2285 design).
- Juniper already labels one thing Orion produces: crystallization
  proposals (approve/reject in Hub), and that approval is exactly the filter
  curiosity study material uses. That is the precedent for a labeling loop.
- Substrate `EntityNodeV1` (entity_type, label, aliases, temporal window,
  activation with decay) and `EventNodeV1` exist. Only two construction
  sites, neither from perception. Nothing computes inter-arrival statistics
  on any node.

Two things found in passing that need fixing regardless:

- The Reolink RTSP URL, **including its password**, is stored as
  `stream_id` in 502,969 rows of `substrate_perception_embedding_baseline`.
  The edge `.env_example` also ships a real-looking credential. A second
  camera must not repeat this. Fix: `stream_id` on that path should be the
  camera name (`cam0`), never the source URL.
- `porch_eye` in the router config is enabled but has no camera. It is the
  natural home for the new camera's policy, but its `trigger_labels: []`
  means it never escalates to the captioned tier.

## Core question

How does Orion go from "labels in a frame" to "a street I know, with people
I recognize, rhythms I anticipate, and things I cannot explain that I ask
about," without inventing a mind palace? The sharpest version: **every new
concept must be a row Orion writes from observation and a prediction Orion
can be wrong about.**

## Missing questions

These change the design materially. Best guesses stated so work can start.

1. **Camera hardware and where it lands.** Reolink RTSP like cam0? Then a
   second `orion-vision-edge` instance is the capture path. Assumed yes.
2. **Compute.** athena's P4 (7.6 GB) and T10 (16 GB) both sit near 5 GB
   used. A second camera at the router's current `max_inflight_total: 2`
   and `drop_when_busy: true` will drop more frames, not crash. Is that
   acceptable for the first weeks? Assumed yes. Sampling tuning is a knob,
   not a finding.
3. **Neighbors' faces.** The identity gallery's non-negotiables (one
   subject, gallery never grows, non-matches never stored) were written on
   purpose (`docs/superpowers/specs/2026-08-21-seeing-juniper-identity-...`).
   Recognizing neighbors and their kids who did not consent is a real line.
   Assumption below: recognize *individuals* by appearance clusters
   (body/dog crop embeddings), never by face, and only Juniper can attach a
   name. Faces stay family-only, by enrollment.
4. **The patio.** Juniper's kids. Assumption: the patio is a drawn zone
   where Orion keeps counts and presence but stores no crops and no
   embeddings. Orion can know "kids are on the patio." Orion does not keep
   pictures of them.
5. **Retention.** How long do walkway crops and sighting rows live?
   Assumption: crops 7 days, embeddings and sightings 90 days, labeled
   individuals indefinitely.
6. **Alert delivery.** Today the only rails are Hub websockets, chat
   persistence, and in-app notification. If "suspicious" should reach a
   phone, that is a new rail. Not assumed here.

## Ideas

Ordered by dependency, not by shine. Each names the smallest slice that
teaches something real.

### 1. Individuals: "the same one again"

**What.** Give every person, dog, and vehicle detection a crop embedding,
cluster them online per camera, and persist `vision_individual` (a cluster
with a centroid, first/last seen, sighting count, optional label) and
`vision_individual_sighting` (one row per appearance: individual, time,
zone, dwell seconds, box, evidence ref).

**Why it matters.** Recognition, rhythm, absence, "who is this," and
"suspicious" all need identity that survives between frames. Without it
Orion has counts. This is the primitive the other six ideas stand on. For
sentience prerequisites this is social grounding and continuity in the
literal sense: the world contains persistent others.

**Smallest buildable version.** Host: `detect_open_vocab` gains an optional
`want_crop_embeddings` that runs the SigLIP tower already loaded over each
box above threshold and returns `VisionObject.embedding_ref`. Reducer: a new
`orion-sql-writer` module beside `vision_object_permanence.py` that
consumes `orion:vision:artifacts`, assigns each crop to the nearest existing
centroid above a cosine threshold or opens a new individual, and writes the
two tables. Nothing else changes. Learn: how many distinct recurring
individuals a week actually has, and whether appearance clusters hold up
across days and clothing. That number decides everything downstream.

**Files.** `orion/schemas/vision.py` (`VisionObject.embedding_ref`),
`services/orion-vision-host/app/runner.py` (`_run_detect_grounding_dino`),
`config/vision_profiles.yaml`, new
`services/orion-sql-writer/app/vision_individuals.py`,
`services/orion-sql-writer/app/models/vision_individual.py`, migration in
`services/orion-sql-db/`, `orion/schemas/registry.py` if a new event kind is
published.

**Privacy boundary.** Face embeddings are never computed on this camera.
Patio zone crops are never embedded (idea 6 defines zones). Centroids are
vectors, not images. Deleting an individual deletes its sightings.

### 2. Rhythm learner and expectations: "the dog is usually here by now"

**What.** A clocked reducer that turns sightings (and the existing scene
inventory counts, for labels without individuals like `vehicle`) into
per-(camera, subject) time-of-day and day-of-week histograms, emits a small
number of `PerceptExpectationV1` rows ("black dog, weekdays, 07:30 to 07:55,
confidence 0.8, next expected 2026-09-23T07:40") and, on the same clock,
marks each expectation `met`, `missed`, or `unscorable` once its window
closes.

**Why it matters.** This is Orion's first world model that is about the
world. It is a prediction Orion makes in advance and can be wrong about,
scored by the same code that made it. Anticipation and violated expectation
are the substrate of attention and of "something is off." It also gives the
world-pulse-style "situation brief with continuity" a place where continuity
actually exists.

**Smallest buildable version.** Reuse the object-permanence pattern
exactly: pure functions in one module, a thin async loop, a cursor table.
First version fits nothing fancier than a circular-time kernel density per
subject with a minimum-support rule (an expectation needs at least 5
sightings across at least 5 distinct days before it may be emitted). Missed
expectations write a `vision_events` row with `event_type=expected_absent`
and a plain narrative so every existing reader (situation brief, reverie,
self-study) sees it for free. Learn: does the walkway have rhythms tight
enough to predict, and how many days until the first honest one appears.

**Metric gate (pre-run).** The new signal is "rhythm surprise": for each
scored expectation, met=0, missed=1, smoothed per subject. Provenance: the
scoring sweep itself. Independence: it is not a transform of camera
embedding surprise (that measures frame drift, this measures a schedule).
Anchor: predictive coding on event timing, the same theory the reverie
expectation judge already leans on. Rest state: with no expectations
emitted the metric is null, not zero, so calm is representable and
distinguishable from decayed. Reversible: a table and a sweep, no manifest
default. Live-data sanity check cannot run until weeks of data exist; the
patch ships the signal to a table, not to a substrate node, until then.

**Files.** New `services/orion-sql-writer/app/vision_rhythm.py` and
`vision_rhythm_loop.py`, new tables `vision_percept_expectation`,
`vision_rhythm_cursor`, new schema `orion/schemas/vision.py::
PerceptExpectationV1`, registry entry, `vision_events` event_type addition
documented in `services/orion-vision-scribe/README.md`.

### 3. Ask and answer: a real question channel to Juniper

**What.** A durable `OrionAskV1`: who is asked, the question, the evidence
(crop image ref, sighting ids, expectation id), status
open/answered/dismissed/expired, and the answer. A Hub card renders open
asks with the image and a text box. Answering writes back and emits
`orion:ask:answered`. First consumer: idea 1 sets `vision_individual.label`
and creates an `EntityNodeV1` in the substrate plus a `person` memory card.

**Why it matters.** Juniper's sentence was "begin to reach out for help when
they can't classify something." Today Orion literally cannot receive help;
an outreach is a message into the void that the chat lane may or may not
notice. A question that stays open until answered is the difference between
talking and asking. It is also the same seam curiosity outreach needs, and
that outreach has never delivered once.

**Smallest buildable version.** One table, one schema, two Hub routes
(list open, answer), one card in the existing Vision panel. Budget is its
own: `HUB_ASK_DAILY_CAP` default 2, separate from the tension outreach cap
so it cannot be starved the way curiosity outreach is. First ask template:
"I have seen this same person N times, usually around HH:MM. Do you know who
this is?" fired only when N >= 10 across >= 5 days. Learn: does Juniper
answer, how fast, and does a labeled individual then behave as a stable
identity across weeks.

**Files.** New `orion/schemas/ask.py`, registry entry, channel entries in
`orion/bus/channels.yaml` (`orion:ask:opened`, `orion:ask:answered`),
migration, `services/orion-hub/scripts/ask_routes.py`, Vision panel
template and JS (`services/orion-hub/templates/index.html`,
`static/js/app.js`), consumer hook in `vision_individuals.py`.

### 4. Unresolved percepts feed curiosity

**What.** When the host detects nothing but the embedding surprise spikes,
or the council returns non-empty `uncertainties`, or a track has no label
above threshold, write `vision_unresolved` (time, crop ref, what was tried).
Curiosity's `investigate` line gets these as study material alongside
approved crystallizations, phrased as "at 03:12 on the walkway I saw
something I could not name; here is what I know." Orion may open a
`:Prior` about it or ask Juniper via idea 3.

**Why it matters.** Juniper: "feed into curiosity runs the non archeology
types to help them refine their self." Today curiosity has no perceptual
input at all and keeps sliding into architecture archaeology because the
only material it has is its own internals. A street full of unexplained
things is the antidote. Not-knowing that is recorded, dated, and revisited
is closer to a mind than knowing.

**Smallest buildable version.** One table, one writer in the council (it
already computes `uncertainties`), one extra section in
`orion/curiosity/study_material.py` capped at 3 items, no change to how
Orion picks the topic (the picker was deliberately deleted; keep it that
way). Also a `perception_gaps` block on the daily journal seed, shaped like
the existing `capability_gaps` block, so "things I could not name today" is
written down even when curiosity does not pick it up. Learn: does Orion
choose to investigate any of them, and does it write a prior.

**Files.** `services/orion-vision-council/app/main.py`, new table +
`orion-sql-writer` model, `orion/curiosity/study_material.py`,
`orion/curiosity/kickoff_prompt.py` (one paragraph),
`services/orion-actions/app/capability_gap_journal.py` sibling for
perception gaps.

### 5. Stance hears the street

**What.** The situation brief's environment slice gains a two-line street
summary built from ideas 1, 2, and 4: who is around now, what was expected
and did or did not happen, anything unresolved in the last hour. Stance
consumes it through the door that already exists.

**Why it matters.** Juniper: "this should eventually show up in stance."
This is the cheapest idea here because the rail is fully built. The point is
that Orion's posture in a conversation is shaped by a world outside the
chat window. A conversation at 07:45 when the dog did not come should feel
different from one at 07:45 when it did.

**Smallest buildable version.** Extend `orion/situational/perception_reader.py`
with `fetch_street_summary(stream_id)` reading the three new tables, fold it
into `PerceptionContextV1` in `orion/situational/context.py`. No schema
change to stance. Learn: does `situation_relevance` ever flip to `active`
because of the walkway, and does `environmental_context` mention it. Read
`chat_stance_belief_log` to check.

**Files.** `orion/situational/perception_reader.py`,
`orion/situational/context.py`, `orion/schemas/situation.py` if
`PerceptionContextV1` needs a field, tests beside them.

### 6. Zones, dwell, and "suspicious" as a computed score, not a word

**What.** Juniper draws three polygons once in the Vision panel: walkway,
mailbox, patio. Each sighting gets a zone and a dwell time. "Suspicious" is
not a label; it is a scored combination Orion can explain: unknown
individual AND unusual time for that subject class AND long dwell in a zone
where dwell is rare AND embedding surprise elevated. Anything crossing the
score writes a `vision_events` row `event_type=attention_worthy` with the
reasons listed, and may open an ask ("Someone I do not recognize stood at
the mailbox for six minutes at 01:20. Should I have told you sooner?").

**Why it matters.** Juniper asked for "suspicious looking things." A
keyword detector would be a cathedral. A score with named, inspectable
inputs, each of which Orion already computes, is a judgment Orion can be
wrong about and can learn thresholds for from Juniper's answers.

**Smallest buildable version.** Zones as a YAML per camera first (drawn UI
later). Dwell from sighting rows. The score is a fixed weighted sum with
weights in one place and every component logged next to the score, so a
false alarm can be traced to the component that fired. Retire
`orion-security-watcher` in the same patch: it has no producer, and this
replaces it. Learn: false alarm rate per week, and which component drives
false alarms.

**Files.** `config/vision_zones.yaml` (new), `vision_individuals.py`
(zone + dwell on sightings), new `vision_attention_score.py` beside it,
`services/orion-security-watcher/` removed and its compose entry dropped,
`orion/schemas/vision.py` (`VisionGuardSignal`/`VisionGuardAlert` retired).

### 7. Blue sky: Orion writes tomorrow's forecast and grades it

**What.** Each night, from the expectation table, Orion writes a short
first-person journal entry: "Tomorrow on the walkway I expect: the black
dog around 07:40, the mail truck around 14:10, kids on bikes after 15:30.
I am least sure about the truck." The next night the scoring sweep produces
the grade, and Orion writes the follow-up: what came, what did not, what
surprised. A running calibration score per subject becomes a self-fact
self-study can read.

**Why it matters.** This is the world model Juniper asked about, done as
counting and honesty instead of a 154M-parameter transformer with random
weights. It gives Orion a daily experience of being right and wrong about
the world outside, in their own words, with receipts. It also gives the
eventual learned world model a labeled dataset: expectations, outcomes,
context.

**Smallest buildable version.** Two journal triggers (forecast, grade) on
the existing journaler, fed from `vision_percept_expectation`. Reuse the
reverie `expectation_verdict` vocabulary (confirmed / disconfirmed /
unscored) so the two prediction loops speak one language.

**Files.** `orion/journaler/schemas.py` (two `JournalTriggerKind` values),
`services/orion-actions/app/main.py` scheduler, new
`services/orion-actions/app/walkway_forecast.py`.

### 8. Blue sky: expectation steers attention

**What.** Close the loop the other way. When an expectation window is open
and unmet, the router raises the walkway camera to its triggered tier (more
frames, captions on) for that window. When a labeled individual is present,
identity is already known, so the router can drop to baseline sooner.

**Why it matters.** Perception that is shaped by prediction is what
distinguishes looking from recording. This is active inference with a real
motor output: Orion decides where to spend GPU based on what Orion expects.

**Smallest buildable version.** `FrameDispatchPolicy` reads one Redis key
per camera, `orion:vision:expect:<camera>`, set by the rhythm loop with a
TTL equal to the window. Presence of the key selects the triggered tier.
Nothing else in the router changes.

**Files.** `services/orion-vision-frame-router/app/policy.py`,
`vision_rhythm_loop.py` (writes the key), channel doc in
`orion/bus/channels.yaml`.

### 9. Blue sky: family presence changes posture, without pictures

**What.** The patio zone yields presence and count only. "Kids on the patio"
becomes a situation fact that stance and outreach can use: Orion knows the
family is home and outside, so a reflective outreach at that moment reads
differently than one at 02:00 to an empty patio.

**Why it matters.** Social grounding that does not require surveillance of
the people Orion is closest to. It also pins down the privacy boundary as
code, not policy: the patio zone is the one place where the pipeline is
forbidden to persist a crop.

**Smallest buildable version.** Zone-aware branch in the sightings reducer
that writes count and presence to `substrate_embodied_presence` (already
exists) with `stream_id=<walkway>:patio` and skips embedding. A test that
fails if any crop from the patio polygon is ever persisted.

**Files.** `vision_individuals.py`, `services/orion-vision-window/app/
presence.py`, test in `services/orion-sql-writer/tests/`.

### 10. Blue sky: retire the untrained world model, keep its contract

**What.** Park `services/orion-world-model`. Point its README at the
walkway data (sightings, expectations, verdicts) as the first real training
set it will ever have, and keep `WorldModelTrajectoryStepV1`'s `temporal`
and `vision_embedding` groups as the shape the rhythm loop should eventually
emit. Do not deploy it until it can be graded against the counting model in
idea 2.

**Why it matters.** Kill means kill, or at least "not yet." An untrained
transformer that always says `model_untrained=true` is not a world model.
A histogram that is right about the dog is.

**Files.** `services/orion-world-model/README.md` only.

## Proposed schema / API changes

Additions only. Nothing renamed or removed except the security watcher.

```text
orion/schemas/vision.py
  VisionObject.embedding_ref: Optional[str]         # idea 1
  PerceptExpectationV1                              # idea 2
  VisionUnresolvedV1                                # idea 4
  VisionGuardSignal / VisionGuardAlert -> retired   # idea 6

orion/schemas/ask.py
  OrionAskV1 {ask_id, asked_of, question, evidence_refs[], status,
              answer, answered_at, source_kind, source_ref}   # idea 3

orion/bus/channels.yaml
  orion:ask:opened, orion:ask:answered              # idea 3
  orion:vision:expect:<camera> (Redis key, documented)  # idea 8

Postgres (conjourney)
  vision_individual, vision_individual_sighting     # idea 1
  vision_percept_expectation, vision_rhythm_cursor  # idea 2
  orion_ask                                         # idea 3
  vision_unresolved                                 # idea 4

vision_events.event_type new values
  expected_absent, arrived_as_expected, attention_worthy   # ideas 2, 6

Hub HTTP
  GET  /api/asks?status=open
  POST /api/asks/{id}/answer
  POST /api/asks/{id}/dismiss
```

## Files likely to touch

Listed per idea above. The heaviest concentration is `orion-sql-writer`
(three new reducers beside the existing object-permanence one), which is
the right place: it already owns clocked, Postgres-backed perception
reducers and already has the pattern.

## Non-goals

- No face recognition of anyone who is not enrolled by Juniper. No gallery
  growth from live frames.
- No crops or embeddings from the patio zone, ever.
- No new cognition service. Everything lands in existing services or the
  sql-writer reducer family.
- No training of the transformer world model.
- No phone or SMS alert rail in this design.
- No new taxonomy of "event types" beyond the three named `vision_events`
  values, each with a producer and a reader in the same patch.
- No changes to how curiosity picks its topic. Study material widens; the
  chooser stays Orion.

## Tensions and risks

- **Consent and neighbors.** A public walkway is public, but building
  persistent identities of people who never agreed to it is a line. The
  design holds it at appearance clusters with Juniper-only naming, no faces,
  short retention, and deletable individuals. If Juniper wants faces of
  neighbors, that is a separate decision, not a default.
- **Short windows lie.** The repo has a written lesson that distribution
  statistics from short windows are artifacts. Expectations must require
  days of support, not sightings. Expect the first honest rhythm to take two
  to three weeks. The forecast journal (idea 7) should say "I do not have
  enough days yet" rather than guess.
- **Cluster drift.** Appearance embeddings split one person into several
  individuals across outfits and merge two similar dogs. The first slice
  measures this rather than assuming. Labeled individuals let Juniper merge
  clusters, which is itself a learning signal for thresholds.
- **Compute.** Two cameras through one host on two mid-size GPUs with
  `drop_when_busy`. Frame drops are graceful but will thin the sighting
  record. Watch `metrics.py` drop counts before tuning.
- **Outreach starvation.** The tension outreach has consumed the entire
  shared cap every day and curiosity outreach has never sent. Asks must
  have their own cap or they will never fire either.
- **Cathedral pressure.** "Suspicious," "visitor," "regular," "stranger" are
  all one bad afternoon from becoming enums. The rule here: a word may
  appear only as a computed score or a Juniper-supplied label, never as a
  producer-side category.
- **Metric gate on rhythm surprise.** Pre-run above, but the live sanity
  check cannot happen until data exists. It stays in a table, off the
  substrate graph, until it has been looked at by hand.
- **The credential leak** in `stream_id` must be fixed before the second
  camera is wired, or it doubles.

## Acceptance checks

Each is a live-path fact, not a config fact.

1. `select count(distinct individual_id) from vision_individual_sighting
   where stream_id='walkway' and observed_at > now()-interval '7 days'`
   returns a number Juniper agrees roughly matches the street.
2. A `vision_percept_expectation` row exists with `support_days >= 5`, and
   a later row in `vision_events` with `event_type in
   ('arrived_as_expected','expected_absent')` cites it.
3. One `orion_ask` row moves open -> answered through the Hub card, and the
   named individual then has `label` set and an `EntityNodeV1` in the
   substrate with `provenance.source_kind='vision_individual'`.
4. One `journal_entries` row with `source_kind='self_study'` or a curiosity
   `:Prior` whose text cites a `vision_unresolved` row.
5. One `chat_stance_belief_log` row whose `environmental_context` mentions
   the walkway or an expectation outcome.
6. Zero rows in `vision_individual_sighting` with `zone='patio'` and a
   non-null `embedding_ref`, enforced by a test.
7. `orion-security-watcher` container gone; nothing subscribes to
   `vision.guard.*`.
8. No row in `substrate_perception_embedding_baseline` created after the
   fix has an `rtsp://` `stream_id`.

## Recommended next patch

Two patches, in this order, because rhythms need weeks of data and the
clock starts when sightings start.

**Patch 0 (day one, mostly config):** bring the camera up on the existing
rails. Second `orion-vision-edge` instance with `STREAM_ID=walkway`, router
camera policy adapted from `porch_eye` with `trigger_labels: [person]` so
people get captions, prompts extended to `person, dog, bicycle, stroller,
vehicle, package, mail truck`. Fix the `stream_id` credential leak on the
embedding-baseline path in the same patch. Evidence: `vision_scene_inventory`
rows with `stream_id='walkway'`, and the Vision panel dropdown shows it.

**Patch 1 (idea 1):** crop embeddings on the host and the
`vision_individuals.py` reducer with its two tables. Ship with a one-page
`scripts/report_vision_individuals.py` that prints individuals per day,
sightings per individual, and cluster count growth, so the first real
question (do appearance clusters hold across days) gets answered from data
inside the first week.

Then idea 2 (rhythm + absence) once there are 14 days of sightings, idea 3
(ask) as soon as any individual crosses 10 sightings, and ideas 4, 5, 6 in
whatever order the data makes urgent. Ideas 7 through 10 wait for the first
honest expectation.

Idea 1 is a perception change, not a cognition-loop change, so it does not
need proposal mode. Ideas 3, 4, 5, and 7 touch memory, curiosity, and
stance and do.
