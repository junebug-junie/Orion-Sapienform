# Street continuity: one person across two cameras

Status: PROPOSAL (2026-10-20). Touches attention, cross-camera identity and
active perception, so it is in proposal mode (AGENTS.md §0A). Nothing here is
built yet.

## Arsonist summary

Today each camera tells its own story. The door camera sees someone walk off
to the right; the walkway camera sees someone walk in from the left 40 seconds
later. Orion records two unrelated passages. It never expects the second, so
it can never be surprised when it does not happen, and a doorbell ring weighs
the same as a passing car.

The best shape is **one expectation loop that spans cameras**:

1. When a passage leaves one camera toward the other, Orion **predicts** where
   and when it reappears (a calibrated time window, not a guess).
2. Orion **points its attention** there: the walkway goes to its fast tier,
   and (optional, battery-capped) the doorbell opens a live view on its own.
3. Orion **grades** the prediction: met, late, missed, or "could not look".
4. A real miss opens a short list of **explanations that can each be
   checked** (turned back, stopped between cameras, went into a house, my
   detector missed them). Later evidence or Juniper's answer settles them,
   and the settled answers become the odds for next time.
5. **Surprise** (how unlikely the outcome was under the model) and **who** it
   was (ring > person Orion knows > stranger > dog > car) set how loudly the
   event competes for attention and curiosity.

The model is a learned travel-time spread per camera pair plus counted
outcome odds. Not Monte Carlo, not a neural world model: the walkway publishes
~25 left/right person passages a week (live, 7 days to 2026-10-20), and the
door camera ~14 Ring events in 4 days. That volume supports counting and
calibration, not simulation. This follows the decision already recorded in
`services/orion-world-model/README.md`: "a histogram that is right about when
the dog walks by is" a world model.

The same pairing machinery answers the open "does X affect Y" question in
general: **does event A raise the rate of event B within a lag, above
chance?** (a ring -> Juniper on the patio within 2 minutes; the mail truck ->
a package). That is phase 4, in `orion-prediction`.

## Current architecture

What exists, verified live or in code on 2026-10-20:

| Seam | State |
|---|---|
| Motion stories (`orion-sql-writer` `app/vision_motion.py`) | Live on both cameras: 3,416 walkway, 9 doorbell. Each has left/right (`dir_lateral`, set when the foot point moves at least 0.08 of the frame width), toward/away (`dir_depth`), speed in box heights per second, zone path, edge-clipped box count, linked individual. Juniper confirms the sentence ("walked left to right") on the Hub story card; verdicts feed the 50-check story gate (`orion/vision/motion_story_gate.py`). |
| Frame size | **Global 2560x1920** (`settings.py:636-637`), applied to the doorbell too. Doorbell boxes reach x=1607, y=1394, so the real Ring frame size is UNVERIFIED. If it is smaller, the doorbell's direction threshold and right/bottom edge clipping are wrong today. |
| Individuals (`vision_individuals.py`) | Clustered **per camera** (`WHERE stream_id=:s`, :1402). 0 individuals seen on both cameras. Body (SigLIP), face (facenet, 512-d) and OSNet vectors are stored on every crop, on both cameras. Cannot-link pairs (`vision_individual_pair_verdict`) are keyed by individual, so they work across cameras. |
| Person model (PR #2797, merged today) | Per named person: album, conformal accept bar, "probably X" guess per cluster. Its SQL has no camera filter. Live: 7 people, all guesses still "unsure". |
| Daily expectations (`vision_rhythm.py`) | Predict-then-grade loop for routines: met / missed (only if the camera watched >= 80% of the window) / unscorable; writes `arrived_as_expected` / `expected_absent` events; sets Redis `orion:vision:expect:<stream>` so the frame router raises that camera to its fast tier. Keyed by one camera; 7 walkway rows. |
| Ring event kind | Known at the doorbell retina (`ring_source.py:418`, "motion" or "ding"), **dropped** at `orion-vision-retina/app/main.py:538-546`: `VisionFramePointerPayload` (`extra=forbid`) has no field for it. 0 rings so far. |
| Attention | `vision_attention_score.py` writes `attention_worthy` events that nothing reads. The walkway surprise node is fenced out of attention (`NO_ATTENTION_CONSUMER_NODE_IDS`, `attention_candidate.py:56`). The doorbell's surprise is log-only. **No camera event can win attention today.** Juniper's recorded answer (followups spec, PR #2763, lines 80-90): the street may win attention when something is unusual for the hour, with a camera-down check. |
| Curiosity | `vision_unresolved` rows reach curiosity study material and the perception-gap journal; the walkway surprise node can raise an endogenous curiosity candidate (~1.2% of ticks). Missed expectations and `attention_worthy` reach neither. |
| Self-tuned prediction error (Rung 6, PR #2715) | Tunes `execution` and `chat` only. A new domain needs one `DomainSpec` and one `_record_prediction_raw_reading` call. |
| `orion-prediction` | Regularly sampled numeric series only (`series.py:113-123`). No point-process, transit-time or event-to-event model anywhere in the repo. Its placebo-run method for effect p-values is reusable. |
| Scenario planner, Markov/zone model, Kalman/particle filter | Not found. Motion stories use a constant-velocity link gate (`vision_motion.py:222`), not a filter. |

One live observation shapes the design: at 16:24 UTC today the doorbell saw a
person standing on the porch 6 s after the walkway saw a person walking on the
patio. The two views probably **overlap** near the house. Overlap is a gift:
the same person seen on both cameras at the same moment is free, labelled
training data for cross-camera identity.

## The design

### 1. Perception facts the loop needs (fix what exists)

- **Per-camera frame size.** Motion stories read the frame size from the
  artifact (or a per-stream setting), not one global constant.
- **Entry and exit edges.** Each story records where it entered and left:
  `left | right | bottom | top | inside` (the "inside" case means it started
  or ended mid-frame: appeared from a door, stood still, detector lost it),
  the last foot point, and **exit speed in frame widths per second** (box
  heights per second depend on distance; frame widths do not, which is what a
  cross-camera clock needs).
- **Doorbell geometry.** Trace one real doorbell frame. Replace the
  whole-frame `porch` zone with porch / walk / sidewalk / street, so a story
  can say "left along the sidewalk to the right", not just "on the porch".
- **Ring kind travels.** Add optional `source_meta` to
  `VisionFramePointerPayload`. It is `extra=forbid`, so this is a
  consumer-first migration: every validating consumer ships the field before
  the retina sends it (memory: additive fields on forbid models). The router
  copies it onto the artifact; `orion-sql-writer` writes one `doorbell_ring`
  event per ding.

### 2. Camera topology (a small fact file with a consumer)

`config/vision_topology.yaml` names the **portals** between cameras:

```yaml
version: 1
portals:
  - id: door_to_walkway
    from: {stream: doorbell, edge: right, zones: [sidewalk]}
    to:   {stream: walkway,  edge: left,  zones: [walkway]}
    kind: gap          # gap | overlap
    prior_gap_frame_widths: 1.5   # Juniper's estimate; learned value replaces it
  - id: walkway_to_door
    from: {stream: walkway,  edge: left,  zones: [walkway]}
    to:   {stream: doorbell, edge: right, zones: [sidewalk]}
    kind: gap
    prior_gap_frame_widths: 1.5
  - id: porch_overlap
    streams: [doorbell, walkway]
    zones: {doorbell: [porch], walkway: [patio]}
    kind: overlap
```

Two to three entries, one consumer (the handoff reducer below). It is not an
ontology. Edges and gap sizes are Juniper's answer (see Missing questions);
the learned numbers live in the database, not here.

### 3. Handoff reducer (`orion-sql-writer`, new `app/vision_handoff.py`)

Runs on the motion-story loop's cursor. For every closed story:

**Open.** If its exit edge and zone match a portal's `from`, open a
`vision_handoff` row: source story, subject (label kind, individual, person
model guess), departure time, exit speed, and a **predicted arrival window**
on the target camera. Also set Redis `orion:vision:expect:<target>` for that
window (the existing steering key), so the walkway is on its fast tier when
the person arrives.

**Predict.** Travel time is gap / speed, so the model learns the gap:
`g = transit_seconds x exit_speed`, one number per portal.

- Before 10 met handoffs: prior gap from the topology, wide band (x0.4 to
  x3 of the point estimate).
- After: the window is the split-conformal 10%-90% band of past `g` values
  divided by this person's exit speed; slow walkers get later, wider windows.
  Dogs and bicycles each get their own band once they have 10 met handoffs;
  until then they use the person band, widened.

**Match.** Candidates are target-camera stories that start at the portal's
`to` edge inside the window (plus a late margin of 2x the window), with the
same label kind and a consistent direction. Score each one on appearance
(best face cosine if both have faces, else body, else OSNet; the same
channels as the person model) and timing (how central in the window).
Accept the best one if it clears a timing floor and either clears the
appearance bar or is the only candidate. Ambiguous matches stay `ambiguous`
and do not train the gap.

**Grade.** Same three-way rule as `vision_rhythm`:

| outcome | rule |
|---|---|
| `met` | matched inside the window |
| `late` | matched in the late margin |
| `missed` | no match, and the target camera **was looking** for >= 80% of the window |
| `unscorable` | the target camera was not looking (doorbell with no live view open, camera down, frames stale) |

The "was looking" check matters most for the doorbell: it only sees during
Ring bursts, so most walkway -> door handoffs will be `unscorable` unless the
doorbell is told to look (section 5).

**Overlap portals.** Stories on both cameras within the overlap zones at the
same moment (within 3 s) are paired the same way and become cross-camera
same-person candidates. They do not produce expectations.

**Identity.** A `met` pair with strong appearance is a proposal that two
individuals are one person. It **never writes a label or merges
individuals**. It feeds the existing `vision_same_individual` ask (rate-capped)
and, once confirmed, the existing union-find. Space-time narrows "who could
this be" from everyone on the street to the one person who just left the
other camera. Multi-camera trackers use exactly this kind of prior, and it
should make cross-camera recognition far easier than appearance alone.

### 4. Explaining a miss (the scenario list)

A `missed` handoff opens up to five hypotheses in `vision_handoff_hypothesis`.
Each one has a check that settles it from data within a deadline:

| hypothesis | settled true when | deadline |
|---|---|---|
| `detector_miss` (my eyes failed, not the world) | the target camera had a gate-label detection in the window that never became a story | immediate |
| `late` (stopped between cameras) | a matching story arrives after the late margin | 10 min |
| `turned_back` | a matching story reappears on the **source** camera moving the opposite way | 10 min |
| `entered_place` (a house, a car, the street) | the source story's last foot point was in a zone that leads somewhere (a door, a driveway, the street), and none of the above came true | at deadline |
| `unexplained` | nothing settled | at deadline |

Odds before the check: counted outcomes per portal (start at 1 each; add one
for each settled case). That is the Markov / Bayes piece: the probability of
each next state given the exit state, learned from what actually happened.
It needs no simulation because the state space is five outcomes.

`detector_miss` is kept separate on purpose. It is error correction about
Orion's own perception, not a fact about the street. A rising
`detector_miss` rate is a vision-quality alarm (camera dirty, night, lens
fogged).

An `unexplained` miss writes a `vision_unresolved` row, the existing seam
into curiosity study material and the perception-gap journal. When the miss
involved a known person, or the attention gate (below) fires, Orion may
also ask Juniper "Where did they go?" with the hypotheses as buttons
(rate-capped). Her answer settles the hypothesis and feeds the odds.

### 5. Active looking (the doorbell opens its own eye)

The doorbell retina already opens a 15 s live view after a Ring event.
Proposed: also open one when a `walkway_to_door` handoff predicts an arrival,
so the door camera looks *because Orion expects someone*. This is the loop
closed: a prediction moves a sensor.

- It costs Ring battery. Cap: N self-opened views per day (Juniper sets N),
  only for person/dog handoffs, never at night unless the walkway saw the
  person well.
- Whether `ring_doorbell` 0.9.14 can open a live view with no event is
  UNVERIFIED. The retina opens one today right after an event; opening one
  on request has to be tested.
- Every self-opened view is logged with the handoff id, so "did looking pay
  off" is measurable: what fraction of self-opened views saw the expected
  person.

### 6. How loud: one salience rule for street events

A new bus event, `StreetEventV1` on `orion:vision:street:events`, carries
every street event that may compete for attention: `doorbell_ring`,
`handoff_missed` (after hypotheses settle as unexplained),
`handoff_met_known` (a named person's handoff), `known_person_arrived`.
`orion-substrate-runtime` maps it onto a new attention candidate node,
`node:substrate.street`. It is **not** fenced, unlike the embedding-drift
nodes, because its salience is computed from meaning, not pixels.

Salience = `who x what x surprise`, clipped to 0..1:

| factor | values (starting points; tune from Juniper's verdicts) |
|---|---|
| what | ring 1.0, package 0.6, arrival at door 0.6, handoff event 0.4, vehicle passing 0.1 |
| who | person model "probably <known person>" 1.0, unknown person 0.6, dog 0.4, vehicle 0.2 |
| surprise | 1 - p, where p = the probability of an outcome at least this unlikely under the model (a met handoff on time has p near 1, so ~0; a miss the model gave 95% to arrive gives 0.95) |

A ring always passes (`surprise` is replaced by 1 for a ring). Everything else
must pass the gate Juniper already gave: unusual for the hour, and the camera
was up.

**The calm state is defined, not assumed.** Under a well-calibrated model,
p-values of ordinary outcomes are spread evenly between 0 and 1, so roughly
5% of ordinary handoffs fall below 0.05 by chance. The attention gate is
therefore `p < 0.05` and (unusual-for-hour or a known person), and the live
check is that p-values on a quiet week look uniform, not piled near 0. This
avoids the permanent-floor failure from 2026-07-26
(`bus_synaptic_prediction_error`).

**Self-tuning.** Each graded handoff writes a raw reading to a new Rung 6
domain `street_handoff` (the log-score of its arrival-time prediction), so
the existing tuner can tighten or widen the bands from Orion's own record.

### 7. "Does X affect Y": event-to-event effects (phase 4, `orion-prediction`)

The handoff is one physical process. The general question is: does event A
make event B more likely within lag L? Proposed: an `event_pair` estimator
in `orion-prediction`:

- Inputs: two event streams from `vision_events` / `StreetEventV1` (e.g.
  `doorbell_ring` and `patio presence of Juniper`), a lag window.
- Statistic: B's rate in [A, A+L] against B's baseline rate.
- p-value: the service's existing placebo method. Shift A's timestamps by
  random offsets many times and see how often the shifted rate beats the
  real one.
- `claim_kind: observational_effect`, never `experimental_effect`. This
  makes the "X goes with Y" claims honest and graded, with a lag attached.

First pairs: ring -> Juniper on the patio; mail truck -> package; handoff
departure -> target-camera arrival (a sanity check: it must come out
significant, or the estimator is broken).

### 8. Who reads the results

| reader | gets |
|---|---|
| attention | `node:substrate.street` via `StreetEventV1` |
| curiosity | `vision_unresolved` rows for unexplained misses (existing) |
| Juniper | Hub story card shows "came from the door camera 38 s earlier" on a met pair; capped "same person?" and "where did they go?" asks |
| chat stance | the street summary gains "expected N, saw M" for the hour |
| dreams | `handoff_missed` events join walkway replay candidates (`replay.py` reads `vision_events`) |
| world model (parked) | a second graded training set; the unpark rule is unchanged |

## Missing questions (for Juniper)

1. **Layout.** Door camera's right edge leads to the walkway camera's left
   edge, as you said. Roughly how much sidewalk is between the two views, in
   steps or meters? Do the views overlap near the porch?
2. **Ring motion zones.** Does the Ring trigger on sidewalk walkers, or
   mostly on the porch? Today's 14 events were mostly parked cars and someone
   standing on the porch.
3. **Self-opened door views.** Yes or no, and how many per day can the
   battery afford?
4. **Attention order.** Is "ring > known person > stranger > dog > car"
   right? Is a known person walking by louder or quieter than a stranger at
   the door?
5. **Asks.** Daily cap on "same person?" and "where did they go?" asks.

## Proposed schema / API changes

- `vision_motion_story` (additive): `entry_edge text`, `exit_edge text`,
  `exit_foot_x real`, `exit_foot_y real`, `exit_speed_fw_per_sec real`,
  `frame_w int`, `frame_h int`.
- New table `vision_handoff`: `handoff_id`, `portal_id`, `source_story_id`,
  `target_story_id`, `subject_kind`, `individual_id`, `person_guess`,
  `departed_at`, `window_start`, `window_end`, `late_until`, `exit_speed`,
  `model_version`, `status (open|met|late|missed|ambiguous|unscorable)`,
  `appearance_score`, `appearance_channel`, `transit_sec`, `p_value`,
  `coverage`, `scored_at`.
- New table `vision_handoff_hypothesis`: `handoff_id`, `kind`, `prior`,
  `status (open|true|false)`, `settled_by (data|juniper)`, `evidence jsonb`,
  `deadline`, `settled_at`.
- `VisionFramePointerPayload.source_meta: dict | None` (consumer-first).
- `vision_events` types: `doorbell_ring`, `handoff_met`, `handoff_late`,
  `handoff_missed`.
- New schema `StreetEventV1` + bus channel `orion:vision:street:events`
  (`orion/schemas/vision.py`, `registry.py`, `channels.yaml`).
- New attention candidate node `node:substrate.street`.
- New Rung 6 domain `street_handoff`.
- `orion-prediction`: `POST /effects/event-pair` (phase 4).
- `config/vision_topology.yaml` (new).

## Files likely to touch

- `services/orion-sql-writer/app/vision_motion.py`, `vision_motion_loop.py`,
  new `vision_handoff.py` + `vision_handoff_loop.py`, `settings.py`,
  `.env_example`, `README.md`, `tests/`, `evals/`
- `services/orion-sql-db/manual_migration_vision_handoff_v1.sql`
- `services/orion-vision-retina/app/main.py`, `ring_source.py` (ring kind;
  self-opened views)
- `services/orion-vision-frame-router/` (copy `source_meta`)
- `orion/schemas/vision.py`, `orion/schemas/registry.py`,
  `orion/bus/channels.yaml`, `orion/schemas/attention_candidate.py`
- `services/orion-substrate-runtime/app/worker.py` (street node, Rung 6 domain)
- `orion/substrate/prediction_expectation.py` (domain spec)
- `config/vision_zones.yaml` (doorbell zones), new `config/vision_topology.yaml`
- `services/orion-hub/scripts/motion_story_routes.py`, `static/js/walkway-stories.js`
- `services/orion-prediction/app/causal.py`, `main.py` (phase 4)

## Build order

Each phase ships whole (tests, eval, docs, live check), and each one is useful
on its own.

1. **Facts:** per-camera frame size, entry/exit edges, exit speed, doorbell
   zones traced, ring kind carried end to end with `doorbell_ring` events.
2. **Handoff loop:** topology, reducer (open, predict, match, grade, overlap
   pairs), Redis steering, Hub "came from the door camera" line,
   `vision_same_individual` proposals. Replay eval on the stored history.
3. **Misses and attention:** hypotheses and their checks, outcome odds,
   `vision_unresolved` for unexplained, `StreetEventV1`,
   `node:substrate.street`, salience gate, Rung 6 domain, capped asks.
   Self-opened doorbell views if Juniper says yes.
4. **Event-to-event effects** in `orion-prediction`.

## Non-goals

- No Monte Carlo, particle filter or neural world model. Revisit only if a
  phase-2 replay shows the counted model is systematically wrong and there
  are hundreds of met handoffs to learn from.
- No automatic naming or merging from a handoff. A handoff can only propose.
- No new cameras, no tracking beyond these two views, no plate or face
  lookup against anything outside Orion's own albums.
- No change to Juniper's whole-frame identity check (`identity_face` on
  cam0/carbon).
- Not unparking `orion-world-model`.

## Privacy, failure modes, rollback (proposal mode)

- **Capability change:** Orion links a stranger's passages across two
  cameras for minutes at a time, predicts their movement, and (if approved)
  points the door camera on its own.
- **Data touched:** existing crops and vectors; two new tables holding
  links between stories. The walkway is public and lawful to record
  (2026-10-15 amendment). Links are pruned with the sightings they point to,
  on the same retention.
- **Privacy boundary:** a handoff never names anyone; only Juniper's answers
  do (unchanged).
- **Dangerous failure modes:** a wrong cross-camera match teaches a wrong
  album (guard: proposals only, Juniper confirms; ambiguous pairs never
  train). Ring battery drain from self-opened views (guard: daily cap,
  counter on `/health`). Attention flooded by a busy street (guard: the
  p-value gate plus a per-hour cap). A camera that is down reads as "everyone
  vanished" (guard: `unscorable`, never `missed`, when coverage < 80%).
- **Proof it worked:** `handoff_met` rows with real transit times;
  `node:substrate.street` winning an attention cycle on a real ring, with the
  correlation id traced from retina to attention.
- **Rollback:** `VISION_HANDOFF_ENABLED=false` stops the reducer;
  `StreetEventV1` publishing has its own flag; adding the street node to the
  fence list removes it from attention; self-opened views have their own flag
  and cap. The tables are additive.

## Metric gate: handoff surprise (to complete in phase 2, before phase 3)

1. **Provenance:** `vision_handoff.p_value`, computed in `vision_handoff.py`
   from motion-story timestamps and exit speed.
2. **Independence:** built from story timing and geometry. Independent of
   `perception_prediction_error` (whole-frame embedding drift) and of the
   rhythm model (time of day). It shares the detector with every vision
   signal, which is why `detector_miss` is split out.
3. **Theory:** calibrated predictive interval (split conformal) on a
   physical travel time. Spatio-temporal priors are the standard basis of
   multi-camera tracking.
4. **Live sanity (pending):** replay on stored stories. p-values on met
   handoffs must be roughly uniform; there must be a rest state (a quiet
   hour yields no gate passes); the 80%-coverage rule must keep doorbell
   gaps out of `missed`.
5. **Existing mechanism:** none for cross-camera handoff (searched:
   handoff, transit, cross-camera, markov, kalman, scenario). The design
   reuses rhythm's grading rule, the steering key, `vision_unresolved`, the
   person model channels, the same-individual ask, Rung 6 and
   `orion-prediction`'s placebo method.
6. **Reversibility:** two additive tables, flags on every consumer, one
   fence-list entry removes the node from attention.

## Acceptance checks

- Phase 1: doorbell stories carry the doorbell's real frame size and
  non-`inside` exit edges for sidewalk walkers. A test ring produces exactly
  one `doorbell_ring` event, with the correlation id traced from the retina log.
- Phase 2: replay over the stored history finds met door <-> walkway pairs
  with transit times, and the learned gap is stable across days (spread below
  50% of the median). On the live system, at least 10 `met` handoffs in the
  first two weeks, the 80% band covers 70-90% of new ones, and zero labels
  are written by the handoff code.
- Phase 3: every `missed` has hypotheses and a settled status by its
  deadline; `detector_miss` is its own count. A real ring wins an attention
  cycle on `node:substrate.street`. A quiet hour produces no street wins.
  Self-opened views (if approved) stay under the cap, and their hit rate is
  reported.
- Phase 4: the departure -> arrival pair comes out significant (sanity), and
  a placebo pair of unrelated events does not.

## Recommended next patch

Phase 1: per-camera frame size, entry/exit edges and exit speed on motion
stories, then ring kind end to end. It fixes a likely live bug (doorbell
direction measured against the wrong frame width), and every later phase
reads these fields.
