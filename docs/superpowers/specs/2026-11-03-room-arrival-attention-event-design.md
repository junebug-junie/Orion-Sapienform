# Room arrival as its own attention event (design, proposal mode)

Status: PROPOSAL. Design only. No code, no deploy, no production writes.
Date: 2026-11-03. Approved to design by Juniper ("let's do it").

## Arsonist summary

Orion notices a person walking into the room badly. Today the room camera can
only win attention when the whole picture changes a lot. A person sitting down
is a small change to the picture. A lamp turning on is a big one. So Orion's
biggest room-camera moments are mostly lighting and screens, not people.

The person detector already knows when someone appears. It writes one row each
time the room goes from "nobody" to "someone" (`vision_presence_transition`).
Nothing in attention reads that row.

This proposal turns "someone appeared in the room" into its own attention
event. It copies the way doorbell rings already compete: one small gate, one
held value on its own substrate node, a real 0.0 when nothing happened, and
"absent" when the camera is blind. Picture-change surprise stays, as a second
input.

The raw transitions are not usable as they are. On replay, 60% of "someone
appeared" moments last under a minute, and a quarter of them happen between
midnight and 7 am in a repeating 20-35 minute rhythm. Two data-derived
debounces fix that: the room must have been seen empty for at least 10
minutes, and the person must stay in view for at least 60 seconds. With both,
the replay keeps about 1.2 to 1.6 arrivals a day, zero of them overnight, and
a fresh arrival would win about 97% of attention ticks (it loses mostly to Juniper chatting at the same moment).

One blocker comes before any code: the room camera has seen almost no person
since 2026-10-30 (0 person windows on 11-01 and 11-02). Juniper said on 10-31
she was not in those cameras that day, so this may be real absence. But the
detector-rate watch cannot tell, because it counts any label (desks and chairs
are always there). Patch 0 checks that first.

Juniper also asked for a second stage: "look closer". Noticing that someone
came in is cheap and runs all the time. Knowing WHO came in and WHAT they are
doing is expensive, so it should run only when the arrival wins attention.
When it wins, Orion takes one deliberate look and stores a named scene, for
example "a person came into the room. It is probably Juniper. She is working
at her desk, fidgeting with a toy." Today Orion only gets count sentences
("Two tables, two desks, and two doors are detected"). See "Look closer: one
deliberate look after an arrival wins" (Patches 3-4).

The look needs a name, and the room camera's face check almost never gives
one: 0 `probable` matches in 18 checks since 11-02, and 1 confirmed name in
2,465 room transitions over 7 days. So Patch 0 now fixes room-camera face
recognition first, together with the person-blind check.

## Current architecture

### How the room camera competes today

- Each room frame becomes an embedding. `perception_prediction_error`
  (`orion/substrate/prediction_error.py`) scores how far it is from the
  stream's running average, as a z-score, 3 sigma = 1.0.
- `orion-substrate-runtime` writes the score to `node:substrate.perception`
  every ~30 s (`_publish_perception_group`). A missing input writes a marked
  placeholder that never enters history (PR #2831).
- Both attention contests judge it with the world-first rule
  (`orion/attention/world_first.py` `judge_candidate`): eligible at band
  `high` or above of its own 7 days, ranked by mid-rank percentile.
- 2026-10-15..21 replay (earlier work): 23 of 66 room arrivals had a
  perception win within 2.5 min, against 14 by chance (p95 20). Real but weak.

### How the street competes today (the pattern to copy)

- `orion-vision-retina` and the handoff reducer publish `StreetEventV1` on
  `orion:vision:street:events`.
- `orion-substrate-runtime` runs `orion/substrate/street_attention.py`
  `street_gate`: rings always pass; graded handoffs need p < 0.05, a camera
  that is up, and room under an hourly cap.
- A clock tick writes `node:substrate.street` every tick: the strongest passed
  salience while it is held (`SUBSTRATE_STREET_ATTENTION_HOLD_SEC`, 90 s), else
  a real 0.0. When the camera is down it writes nothing, so attention reads the
  node absent after 180 s.
- Glossary: `value_kind: trigger`, `sparsity: designed_sparse`. It is a world
  source (`_WORLD_NODE_IDS`), fenced from curiosity
  (`NO_CURIOSITY_CONSUMER_NODE_IDS`).

### The presence detector (the input this proposal reads)

- `services/orion-vision-window/app/presence.py`. Input: the vote-gated person
  belief from `scene_belief.py` (GroundingDINO "person" boxes, 3 votes to
  enter). Not MTCNN; the face path only adds identity.
- States: `present` (seen in the latest window), `recent` (unseen, inside
  `WINDOW_PRESENCE_GRACE_SEC` = 120 s), `absent`, `unknown` (no window for
  120 s: a blind camera or a stopped pipeline).
- `PresenceRegistry` writes each change to `vision_presence_transition`
  (Postgres only; no bus event). Keepalive rows every 600 s while present.
  `from_state` is NULL on the first row after a restart.
- Readers today: the Temporal Self chronicle (company arcs, orion-durable-runs)
  and `substrate_embodied_presence` for the chat situation line
  (`orion/situational/perception_reader.py`). Both read the LEVEL ("is someone
  here"), never the ONSET ("someone just came in").

### Who wins attention today (substrate broadcast log, 2026-10-27..11-03)

6,628 broadcast ticks (one every ~89 s):

| winner | ticks |
|---|---|
| no winner | 4,399 (66%) |
| `node:substrate.perception` (room picture surprise) | 550 |
| `node:substrate.execution` | 368 |
| `world:chat` | 303 |
| `node:substrate.bus_synaptic` | 230 |
| `node:substrate.codebase` | 204 |
| `node:substrate.chat` | 194 |
| `world:reading` | 136 |
| `node:substrate.cabinet` | 108 |
| `node:substrate.strain` | 55 |
| `node:substrate.perception_walkway` | 53 |
| `node:substrate.street` (rings, handoffs) | 2 |

The field contest (`substrate_attention_frames`, ~2 s ticks, 3 days kept):
70% no winner; room picture surprise leads world sources with 6,158 frames.

### What a world winner changes downstream

Traced at origin/main e4064eb57. In short: a world winner reaches Orion's
prompts and thoughts. It does not reach a goal, a proposal, a bound act, the
allocator, the body economy, or any greeting. Plainly:

- **Reverie thinks about it.** Reverie skips a tick only when nothing won
  (`services/orion-thought/app/reverie.py:873`). Otherwise the winner becomes
  `attended_signal` in the narration prompt (`reverie.py:497-512`). A
  coalition made only of world sources makes no self-policy proposal
  (`orion/reverie/proposal.py:66`). Trace: `substrate_reverie_thought`.
- **Chat sees it as "recent attention".** The attention self-model copies the
  winner (`orion/substrate/attention_self_model.py:681-684`) into
  `substrate_attention_schema`, which feeds the chat stance block
  `recent_attention` (`services/orion-cortex-exec/app/executor.py:4188-4197`).
- **Temporal Self makes it an arc when it holds.** Three consecutive winner
  ticks open an "attention on X" arc (`orion/temporal_self/arcs.py:80-82`),
  shown in the stance cue (`stance_cue.py:118,284`) and noted by Orion Prime
  as "attention moved to ..." (`orion/prime/state.py:225-227`). Trace:
  `temporal_self_arc`. This is the main reason to prefer the 300 s hold
  (decision 2): a 90 s hold rarely lasts three ticks.
- **Curiosity may pick it up.** Curiosity reads the broadcast and can seed a
  `concept_expand` candidate from a deferred loop that has a `node:` ref
  (`orion/substrate/endogenous_curiosity.py:374-378`). The prediction-error
  scan honours `NO_CURIOSITY_CONSUMER_NODE_IDS` (`:299`), but the attention-
  loop path does not check that fence. Patch 2 must check both paths, or the
  "fenced from curiosity" promise is only half true (the street node has the
  same gap today).
- **No act binds to it.** `bind_workspace_winner`
  (`orion/autonomy/self_shed.py:58-98`) binds only refs a template or
  `ACT_ANSWERS_WORKSPACE_REFS` lists; the only pairing is
  `hub.chat_reply -> world:chat / node:substrate.chat`
  (`orion/grammar/act_ledger.py:81-83`). A room-arrival winner would come back
  `winner_unbindable:node_not_bound`, like perception and street today.
- **No effect on the allocator, body economy bids or prices, goals, field
  proposals, or the equilibrium episode gates** (world sources are excluded:
  `services/orion-equilibrium-service/app/self_signal_episode_gate.py:86-92`,
  `orion/attention/field_attention/goal_provenance.py:54-68`,
  `orion/proposals/builder.py:54-58`).
- **Presence already reaches chat as a level.** The company arc ("Someone has
  been here since HH:MM", `stance_cue.py:126-140`) and the situation brief
  ("X has been in view for N seconds", `orion/situational/context.py:2019`)
  exist without attention. No consumer greets or reaches out on an onset;
  outreach treats the camera line as context only
  (`services/orion-hub/scripts/endogenous_outreach.py:2356`).

So the honest behaviour change is: when someone comes in, Orion's own thought
stream and its sense of "what I was just attending to" point at the arrival,
for a few minutes, instead of at nothing or at a lamp. It does not make Orion
say hello. Making an arrival bindable to an act (for example an
`ACT_ANSWERS_WORKSPACE_REFS` entry for a greeting act) is a separate,
later proposal.

## Metric quality gate: the room-arrival signal

Signal under test: "the room camera went from nobody to someone", read from
`vision_presence_transition` rows with `presence_id = 'cam0'`,
`from_state = 'absent'`, `to_state = 'present'`, after the debounce in
"Proposed schema / API changes".

**Verdict: PASS for the debounced signal, FAIL for the raw transition.**
Conditional on Patch 0 (person-blind check) because the last 4 days have no
persons at all.

### 1. Provenance (traced to code)

- Boxes: GroundingDINO "person" (`config/vision_profiles.yaml` default prompts)
  via `orion-vision-host`.
- Belief: `services/orion-vision-window/app/scene_belief.py`
  `SceneBeliefTracker` (3 votes to enter).
- State: `presence.py` `PresenceTracker.observe` (present / recent within
  120 s grace / absent), `PresenceRegistry._note_transition` -> `_emit`.
- Row: `write_transitions_to_postgres` -> `vision_presence_transition`
  (`manual_migration_vision_presence_transition_v1.sql`), flag
  `VISION_PRESENCE_TRANSITIONS_ENABLED` (on).
- Semantic layer: no entry today. `config/field/field_channel_glossary.v1.yaml`
  has no presence channel, and `check_metric_lineage.py` has nothing to
  resolve. The new node needs a glossary entry and a metric lock in the same
  patch (see "Files likely to touch").

### 2. Independence

- **Against room picture surprise (`node:substrate.perception`).** Same
  camera, same frames, different computation: person boxes plus votes versus
  a whole-frame embedding distance. Measured overlap on 61 arrivals
  (2026-10-27..11-03): 24 had perception at or above its own p90 within
  2.5 min, against 17.0 by chance (p95 23). 22 had a perception broadcast win
  within 2.5 min, against 14.4 by chance (p95 20). So they are related but far
  from the same signal: most arrivals never move the picture score, and most
  picture spikes are not arrivals. This is the case for a separate source.
- **Against the doorbell and street (`node:substrate.street`).** Different
  cameras. Related events: 22 of 112 room arrivals had a doorbell person or
  motion event in the 5 minutes before, against 11.4 by chance (p95 16).
  Doorbell rings: 1 of 8 rings in 14 days was followed by a room arrival within
  10 minutes. The street node itself barely competes (2 broadcast wins in
  7 days), so double-winning is rare in practice. Handling: see Missing
  questions, decision 3.
- **Against `world:chat`.** Chat is Juniper typing. She can type from another
  room, and she can sit down without typing. Not derived from each other.

### 3. Theory anchor

- The orienting response (Sokolov, 1963): a new stimulus in the scene produces
  an orienting reaction that habituates when the stimulus repeats. The
  debounce plus the hold is that shape: one onset orients, a person who stays
  does not keep re-orienting.
- Abrupt-onset capture (Yantis and Jonides, 1984): a new object appearing
  captures attention more than a change in an existing object's properties.
  Picture surprise measures "the picture changed". Arrival measures "a new
  object appeared". Different claims.
- Animate-monitoring priority (New, Cosmides and Tooby, 2007): people and
  animals are detected faster than objects of equal visual change. This is the
  reason a person deserves its own channel instead of competing as pixel
  change against lamps and screens.

### 4. Live-data sanity (real rows, 2026-10-20 00:00 .. 11-03 03:00 UTC)

Camera live: the room camera produced detector windows on 1,439-1,440 of
1,440 minutes every day in the window. No multi-hour blind spell in the
window. 11 arrivals came straight after a restart or `unknown` row (no
observed absence before them).

Raw arrivals (`absent -> present`): 112 in 13.1 days, 8.5 a day.

- **Flicker.** Median time seen per stretch: 43 s (stretch length deciles
  137-181 s for the first six, including the 120 s grace). 67 of 112 stretches were seen for
  under 60 s. 89 of 112 had no present -> recent -> present bounce inside the
  stretch; 23 had at least one bounce. The grace already absorbs most bounces
  for a seated person.
- **Probable false presences.** 26 of 112 raw arrivals fell between midnight
  and 7 am Mountain time, often in a regular rhythm: 2026-10-28 and 10-29,
  01:00-07:00 MDT, one ~25-second "person" every 20-35 minutes, with no chat
  and no face match. Also 2026-10-20 and 10-21, 13:00-14:00 MDT, a "person"
  every ~4 minutes for an hour, same hour both days (a sunlight pattern is the
  likely cause). Captions near some of these say "a person is seated", but the
  captioner sees the same detections, so it is not independent proof.
  False-positive rate: UNVERIFIED (no independent ground truth is stored for
  the room camera; `vision_crop_observation` has no room crops).
- **Rest state.** The node reads a real 0.0 whenever no debounced arrival is
  held, which is almost all the time. That is a reachable, measured rest
  state, not a decayed one: the street precedent writes 0.0 by clock tick and
  nothing decays it (`NODE_DECAY_CHANNELS` only touches field vectors).
- **Camera-blind periods.** `to_state = 'unknown'` rows exist (23
  absent -> unknown, 3 present -> unknown in 21 days), and the
  `camera_input:cam0` ladder rung (PR #2831) is fresh now. A blind camera must
  read absent, never "nobody came".
- **Person-blind check (the blocker).** Person windows on the room camera by
  day: about 5,000-5,700 a day on 10-23..10-28, 1,312 on 10-29, 232 on 10-30,
  28 on 10-31, 0 on 11-01, 0 on 11-02. Desk, chair, table and door stay at
  ~10,000 a day. `detection_rate:cam0` reads `ok` (share 0.998) because desks
  count as detections. Juniper chatted on all of those days, and said on 10-31
  "I'm not in those cams today". So either she was elsewhere, or the detector
  went person-blind (camera moved, lighting, threshold). Not resolved here.

Debounce table (13.1 days; "seen" = time in view before the 120 s grace):

| observed absence before | seen for at least | arrivals | per day | between 00:00-07:00 MDT |
|---|---|---|---|---|
| any | any | 112 | 8.5 | 26 |
| any | 60 s | 45 | 3.4 | 0 |
| 10 min | any | 67 | 5.1 | 20 |
| 10 min | 60 s | 21 | 1.6 | 0 |
| 10 min | 120 s | 17 | 1.3 | 0 |
| 20 min | 60 s | 19 | 1.5 | 0 |
| 30 min | 60 s | 18 | 1.4 | 0 |

"Seen for 60 s" is the cut that removes every overnight arrival. "Absence
10 min" removes the 4-minute flicker trains; going to 20 or 30 minutes removes
almost nothing more. Requiring the absence to be OBSERVED (not across a
restart or a blind spell) drops 5 more and leaves 16 (1.2 a day). Those 16
include the morning sit-downs (for example 10-24 13:56 UTC, in view for
642 minutes; 10-26 14:23 UTC, 604 minutes; 10-27 14:12 UTC, 443 minutes) and
evening visits of 1-12 minutes.

### 5. Existing-mechanism check

Searched for `room_arrival`, `person_arrived`, `arrival_event`,
`presence_onset`, `absent -> present` consumers, and every reader of
`substrate_embodied_presence` and `vision_presence_transition`.

- `known_person_arrived` (`orion/schemas/vision.py` `StreetEventV1`) is the
  street path Juniper declined in #2906. Not revived.
- `vision_events` `person_presence` on cam0 (council `host_detect`) is a
  caption artifact, not an onset, and no attention path reads it.
- Temporal Self company arcs read the transitions for the chronicle, not for
  attention.
- The chat situation line and outreach read the presence level.
- The orion-prediction event-pair eval (`run_event_pair_live_eval.py`) uses
  cam0 `person_presence` only as a placebo pair. Open PR #3038 changes it to
  count visits.

Nothing turns a room onset into attention today.

### 6. Reversibility

Cheap. One new substrate node, one new bus channel and schema, one flag in
each service. Turning the flag off makes the producer stop publishing and the
runtime stop writing the node. The node then ages and reads absent, the same as
a dead street listener. Its 7-day history ages out on its own. No training
default, manifest or migration depends on it. The new channel and schema stay
registered but unused until removed.

## Look closer: one deliberate look after an arrival wins

Juniper asked for this stage. The arrival event says "someone came in". The
look says who and what they are doing, in words, with names. Cheap detection
keeps running all the time. Expensive recognition runs only when attention
picks the arrival.

### Theory anchor

- Orienting response (Sokolov, 1963): a new stimulus first produces an
  orienting reaction. The arrival event is that reaction.
- Foveation after capture (Posner, 1980; Itti and Koch, 2001): cheap
  peripheral vision finds where something happened. The eye then moves there
  for one costly, high-detail look. The look act is the eye movement. It is
  spent only on what won attention, not on every frame.

### What the look path has today (read-only check, 2026-11-03)

- **Detection.** The room camera (`cam0`) runs the fast tier about every 5 s
  (`config/vision_frame_router.yaml`). A person moves it to the triggered
  tier (at least 2 s, `want_caption`). The room camera has no track ids. The
  vision council writes count sentences ("Two tables, two desks, and two doors
  are detected"). The numbered people (`#hex`) come from walkway
  `vision_individual` clusters, not from the room camera.
- **How Orion reads the room.** `orion/situational/perception_reader.py`
  `_room_percept_stmt` reads the newest `vision_events.narrative` and never
  loads `entities` (a deliberate privacy note). Orion also gets the
  `substrate_embodied_presence` row and the camera eyes path.
- **Recognition.** On a person trigger the frame router fetches a high-res
  still from vision-edge `POST /still`
  (`services/orion-vision-frame-router/app/dispatcher.py`
  `_identity_with_still`). It sends the still to the `identity_face` profile
  (athena vision host, GPU0). The result goes to the vision-window identity
  handler (`services/orion-vision-window/app/main.py`), which updates the
  presence subject and publishes `IdentitySightingV1` on
  `orion:vision:identity:sighting`.
- **Recognition almost never names anyone in the room.** Since the 11-02
  restart: 18 face checks, 10 `no_face`, 6 `unsure`, 2 `possible`,
  0 `probable`. Over 7 days: 1 of 2,465 room transitions was
  `identity_confirmed` (juniper). The name never reaches a caption.
- **Captioning cannot take a hint.** The room camera uses BLIP-base. It takes
  no prompt (it echoes a prompt back), so it cannot be told "this is probably
  Juniper". Room captions are not stored at all
  (`WINDOW_CAPTION_GATE_STREAMS=walkway,doorbell`). The Qwen2-VL-2B lane
  serves walkway and doorbell with a fixed `CAPTION_PROMPT` and no hint field.
- **One path can already answer a question about the current frame.** The
  council's foveal probe (`services/orion-vision-council/app/foveal_probe.py`)
  captions the current frame on the circe chat VLM lane through the LLM
  gateway, and accepts a `question`. It is reachable only through
  `POST /debug/foveal-probe` (`app/main.py:572`). Its setting says "not yet
  wired" (`app/settings.py:59`). The 2026-10-08 design
  (`docs/superpowers/specs/2026-10-08-vision-blank-captions-and-on-demand-look-design.md`)
  planned an on-demand look. It was never built.
- **No look act exists.** In 7 days the act ledger has 488 `look` acts, all
  `hub.world_pulse_read` web URLs, and 0 camera looks. The allocator
  (`orion/autonomy/allocator.py`) has no perception dispatch kind. Nothing
  links an attention winner to a look. The router's person trigger is a
  reflex: it runs the face check with no heat or price check.
- **Cost.** One look uses the circe chat VLM (a busy GPU) and the athena GPU0
  face model. It needs a price and a bid in the body economy (#2855). The
  cabinet is often hot.

### Order of patches

1. **Patch 0: fix room-camera face recognition first.** A look that cannot
   name anyone only produces "someone". The face check never reaches
   `probable` on the room camera. The likely causes are the camera angle (top
   of head, side face), still resolution, or enrollment photos taken from a
   different angle. Patch 0 also keeps the person-blind check: the
   `person_rate:cam0` ladder rung and the 5-minute sit test.
2. **Patches 1-2: the arrival event** (contract, producer, runtime node), as
   described in the rest of this doc.
3. **Patch 3: the look act.** When the arrival node wins the broadcast, Orion
   opens one `look` act and runs one look executor. Details below.
4. **Patch 4: store the named scene and show it to Orion.** Waits for
   decision 4 in "Missing questions".

#### Patch 0 findings (pending)

Findings pending. A separate read-only investigation is diagnosing why the
room-camera face check never reaches `probable` (angle, top-of-head views,
still size, enrollment set, thresholds). It will fill this subsection with
evidence and a recommended fix. Patch 3 does not start until the face check
reaches at least `possible` on a meaningful share of Juniper's sit-downs.

### Patch 3: the look act

- **Trigger.** The broadcast winner is `node:substrate.room_arrival`, and the
  arrival has not been looked at yet (one look per `event_id`).
- **Act.** Open an act-ledger act: verb `look`, `object_ref` `camera:cam0`,
  parent = the arrival `event_id`. Its `expect` names the payoff:
  "face verdict at least `possible`". `signal_id` is a closed list
  (`PredictableSignal` in `orion/schemas/action_prediction.py`: "a claim about
  a signal nothing measures is not a claim"), so Patch 3 adds one measured
  value to it, for example `look_face_verdict_possible` (1.0 when the look's
  face verdict is `possible` or `probable`, else 0.0). The act ledger then
  grades whether looks pay off, the same way it grades other acts.
- **Price and bid.** The look is a body-economy bid (#2855) with a price for
  the circe VLM call and the GPU0 face call. When the price is too high (heat,
  busy GPU), the bid loses and the act closes as `declined:priced_out`. No
  look runs.
- **Rate limit.** At most `ORION_LOOK_CAM0_MAX_PER_DAY` looks a day (proposed
  6) and a cooldown of `ORION_LOOK_CAM0_COOLDOWN_SEC` (proposed 1800) between
  looks. Both are hard caps for a new cost, to be re-checked after 14 days.
- **One executor, one chain:**
  1. Fetch a high-res still from vision-edge `POST /still` (the router's
     existing call).
  2. Run the face check on that still (`identity_face`).
  3. Run the foveal probe on the same still with a question. The question
     carries a HEDGED name hint taken from the face verdict, for example
     "The person is probably Juniper (possible, 0.53). Describe what they are
     doing." With `no_face` or `unsure`, the hint is "a person", with no name.
     The hint is never stronger than the face verdict.
  4. Close the act with the outcome: the face verdict, the scene text, the
     evidence ids, and whether `expect` was met.
- **Where it runs.** A new `look` dispatch kind for the allocator, and the
  executor next to the foveal probe in orion-vision-council (it already has
  the LLM gateway client). The `/debug/foveal-probe` route stays as is.

### Patch 4: store the named scene and show it to Orion

- **Store.** A new `vision_events` type `named_scene` for `cam0` (preferred
  over turning on room `camera_caption`, which would store every BLIP count
  sentence). Fields: scene text, the arrival `event_id`, the `act_id`, the
  still id, the face verdict (label, strength, score), the model and lane, and
  `observed_at` of the still.
- **Grounding and slop rules.** The scene is stored only if it passes the
  existing caption checks and a grounding check, reusing the council's
  `app/evidence_grounding.py` (it already flags activity words such as
  "watching" or "using"): every person and object the text names must be in
  the detector's boxes for that still, or in the face verdict. An activity claim ("fidgeting with a toy") must name an
  object the detector saw near the person; otherwise it is cut to "is at the
  desk". A failed check stores a reason row, not the text.
- **Name rule.** The stored name never reads stronger than the face verdict:
  `probable` -> "probably Juniper", `possible` -> "possibly Juniper",
  `unsure`/`no_face` -> "someone". Only people Juniper enrolled and named can
  be named (decision 4).
- **Show it.** `perception_reader.py` reads the newest `named_scene` for the
  room, with its age ("3 min ago"). A scene older than
  `ORION_NAMED_SCENE_MAX_AGE_SEC` (proposed 1800) is shown as past ("earlier,
  at 14:12 ..."), never as now.

### Failure modes

- **False naming (the wrong person named).** Covered by the hedge, by "never
  stronger than the face verdict", and by naming only enrolled people. Tracked
  by the face verdict distribution in the acceptance checks.
- **The caption invents an activity.** Covered by the grounding check against
  the still's detector boxes. Tracked by the grounded rate.
- **Cost spiral.** Covered by the daily cap, the cooldown, one look per
  arrival, and the body-economy price. Tracked by looks per day.
- **A stale scene shown as current.** Covered by the age stamp and the max
  age rule in the reader.

## Missing questions

Four decisions for Juniper. Everything else follows from the data above.

1. **The two debounce numbers.** The room must be seen empty for 10 minutes,
   and the person must stay in view for 60 seconds. Both come from the replay
   table, but they are absolute numbers, and your rule is no absolute floors
   without your OK. A relative form ("longer than this room's own usual
   absence") was considered. It does not fit, because the room's absences run
   from 2 minutes to 15 hours, and the overnight false presences come in the
   same sizes as real short visits. Recommended: approve 10 min / 60 s as
   producer-side debounces, re-check them after 14 days of live data.
2. **How long one arrival stays news.** Option A: hold 90 s like the
   doorbell. That is about one broadcast tick (every ~89 s), so reverie may
   get only one chance at it. Option B: reuse the measured orienting window
   (`EVENT_ORIENTING_WINDOW_SEC`, 300 s) with the existing `event_fade`
   (full for 60 s, then linear to 0 at 300 s), computed by the producer tick.
   Recommended: B. It reuses a number the repo already measured, and at
   ~1.2-1.6 arrivals a day the node is non-zero for ~1.3% of its readings, so
   a fresh arrival still reads `unusual` (percentile ~0.987, mid-rank ~0.993).
3. **Someone rings and then walks in.** The doorbell (street node) and the
   room arrival are different events seen by different cameras. Recommended:
   let both compete, and record the link on the room event
   (`preceded_by_street_event_id` when a street event passed in the last
   10 minutes), so we can count how often it happens. Alternative: the room
   event does not fire within 10 minutes of a street pass. That would hide the
   arrival you care about most (a guest coming in).

4. **Should named scenes reach Orion, including other people?** The room
   reader hides detection entities on purpose today
   (`orion/situational/perception_reader.py` module note: "Privacy. Selects
   the narrative column and nothing else"). The standing rule is "no privacy
   boundary between Juniper and Orion". But household members and guests walk
   into the room too, and a named scene would name them. Recommended: yes for
   people Juniper enrolled and named, and only at the face check's own
   verdict strength. Unknown people stay "someone". No picture, crop or
   embedding goes into the scene text. Patch 4 waits for this answer.

Also decided by precedent, not asked: the node starts fenced from endogenous
curiosity, like the street node (`NO_CURIOSITY_CONSUMER_NODE_IDS`). Opening
curiosity is a later go-live decision with its own data.

## Proposed schema / API changes

### Producer: orion-vision-window

- New pure function next to `PresenceRegistry`: `room_arrival_gate(...)`.
  Fires once per stretch, at the first window where all of these hold:
  - stream is in `VISION_ROOM_ARRIVAL_STREAMS` (default `cam0`);
  - the stretch opened with `absent -> present` (not `NULL`/`unknown` before:
    an absence across a restart or a blind spell is not observed);
  - the absence before it lasted at least `VISION_ROOM_ARRIVAL_MIN_ABSENT_SEC`
    (600), measured from the previous stretch's last sighting;
  - the person has been in view for at least
    `VISION_ROOM_ARRIVAL_MIN_SEEN_SEC` (60): `last_seen - state_since >= 60`,
    with no fall to `absent` in between.
- Publishes `RoomArrivalEventV1` on `orion:vision:room:events` (new). Flag
  `VISION_ROOM_ARRIVAL_EVENTS_ENABLED` (default on, per the ship-on rule).

`RoomArrivalEventV1` (`orion/schemas/vision.py`, `extra="forbid"`):

| field | meaning |
|---|---|
| `event_id` | sha256(stream_id, stretch start, "room_arrival")[:32]; retries are no-ops |
| `stream_id` | `cam0` |
| `appeared_at` | the stretch's `state_since` (first sighting) |
| `confirmed_at` | the moment the 60 s rule passed (publish time) |
| `absent_before_sec` | observed absence before the stretch |
| `transition_id` | the matching `vision_presence_transition` row, for joins |
| `identity_label` | optional; only when the people record has a `probable` face match at that moment (the same rule as the transition row's `label`); never required, never a guess |
| `correlation_id` | as usual |

### Consumer: orion-substrate-runtime

- New pure module `orion/substrate/room_attention.py`, shaped like
  `street_attention.py`: state, `held_reading`, `status()`. No gate beyond
  dedupe by `event_id` (the producer already debounced), so no hourly cap.
- Listener on `orion:vision:room:events` plus a clock tick writing
  `node:substrate.room_arrival` every tick:
  `event_fade(age, 300)` x 1.0 while held (decision 2, option B), else a
  real 0.0.
- Camera-blind: the tick writes NOTHING when the room camera has not delivered
  a frame for `SUBSTRATE_ROOM_CAMERA_UP_MAX_AGE_SEC` (180), reusing the
  frame-seen hook the street node already has. The reading ages, and attention
  reads the node absent after 180 s. Blind never reads as "nobody came".
- Trace on the node and in `/health` (`room_attention`): received, written,
  held event ids, `attention_wins_total`, `preceded_by_street_event_id`.
- Flag `SUBSTRATE_ROOM_ATTENTION_ENABLED` (default on).

### Attention seam

- `orion/schemas/attention_candidate.py`: `ROOM_ARRIVAL_NODE_ID`, added to
  `_WORLD_NODE_IDS` (so it is external) and to
  `NO_CURIOSITY_CONSUMER_NODE_IDS`. No change to `AttentionCandidateV1`.
- `orion/attention/world_first.py`: no rule change. The node competes through
  `node_candidate` like the street node. A `trigger` gets its percentile as
  its score (`bad_direction_percentile`).
- Glossary entry for `node:substrate.room_arrival` `prediction_error`:
  `value_kind: trigger`, `sparsity: designed_sparse`, `rest` and
  `absent_means` worded like the street entry ("written every tick while
  measurable ..."), so `prediction_error_is_event_written` stays False and the
  producer's own fade does the decay. Re-lock
  `config/metrics/metric_definitions.lock.json`.

Picture-change surprise is unchanged and keeps competing as a second input.

### Look act and named scene (Patches 3-4)

- `orion/schemas/action_prediction.py`: add `look_face_verdict_possible` to
  `PredictableSignal`, with its measurement (1.0 when the look's face verdict
  is `possible` or `probable`, else 0.0).
- Allocator: a new `look` dispatch kind with target `camera:cam0`, priced and
  bid in the body economy (#2855). Act-ledger act: verb `look`,
  `object_ref` `camera:cam0`, parent = arrival `event_id`, `expect` as above.
- orion-vision-council: a look executor that chains `/still` -> face check ->
  foveal probe with a hedged `question` -> act close. New settings
  `ORION_LOOK_CAM0_ENABLED` (default on), `ORION_LOOK_CAM0_MAX_PER_DAY` (6),
  `ORION_LOOK_CAM0_COOLDOWN_SEC` (1800).
- `vision_events` row with `event_type = 'named_scene'`, `stream_id = 'cam0'`.
  `narrative` = the scene text. `entities` holds `arrival_event_id`,
  `act_id`, `still_id`, `face_label`, `face_strength`, `face_score`, `model`,
  `lane`, `grounded` (bool), `grounding_reason`, `observed_at`.
  `evidence_refs` = the still id, the face sighting id and the act id.
  `perception_reader.py` reads only `narrative`, age and the name fields, not
  boxes. If it is also published on the bus, it gets a registered
  schema (`NamedSceneV1`) and a channel entry in the same patch.
- `perception_reader.py`: read the newest `named_scene` with its age; setting
  `ORION_NAMED_SCENE_MAX_AGE_SEC` (1800).

## Files likely to touch

- `orion/schemas/vision.py`: `RoomArrivalEventV1`.
- `orion/schemas/registry.py`: register it.
- `orion/bus/channels.yaml`: `orion:vision:room:events`.
- `services/orion-vision-window/app/presence.py` (or a new `room_arrival.py`
  beside it), `app/main.py`, `app/settings.py`, `.env_example`, `README.md`,
  `tests/`.
- `orion/substrate/room_attention.py` (new), `orion/substrate/tests/`.
- `services/orion-substrate-runtime/app/worker.py`, `app/settings.py`,
  `.env_example`, `docker-compose.yml`, `README.md`, `tests/`, `evals/`.
- `orion/schemas/attention_candidate.py`.
- `config/field/field_channel_glossary.v1.yaml`,
  `config/metrics/metric_definitions.lock.json`.
- `.github/workflows/street-attention-tests.yml` (or a sibling workflow).
- `scripts/check_substrate_ladder_liveness.py` / `orion/vision_detection_rate_watch.py`
  (Patch 0 only).
- Patch 0 face fix: depends on the pending findings. Likely
  `services/orion-vision-frame-router/app/dispatcher.py` (still request),
  `config/vision_profiles.yaml` (`identity_face`), and the enrollment set.
- Patch 3: `orion/schemas/action_prediction.py`, `orion/autonomy/allocator.py`,
  `orion/grammar/act_ledger.py` (look verb mapping),
  `services/orion-vision-council/app/foveal_probe.py` and a new
  `app/look_executor.py`, `app/settings.py`, `.env_example`, `README.md`,
  `tests/`, `evals/`; the body-economy price table.
- Patch 4: `services/orion-vision-council/app/evidence_grounding.py`, the
  `vision_events` writer, `orion/situational/perception_reader.py`, tests.

### Collisions with in-flight work (checked 2026-11-03)

- `chore/rung6-retire-tuner` (worktree `../Orion-Sapienform-rung6-retire-tuner`,
  uncommitted, active today): edits `orion/substrate/street_attention.py`,
  `orion/substrate/prediction_error.py`,
  `services/orion-substrate-runtime/app/worker.py` and
  `.github/workflows/street-attention-tests.yml`. Same files as Patch 2. Land
  after it, or rebase on it. Do not copy the Rung 6 handoff EWMA into
  `room_attention.py`.
- Open PR #3038 (`feat/prediction-event-pair-visits`): touches the
  orion-prediction event-pair eval that uses cam0 `person_presence`. No file
  overlap; a room-arrival event would be a better selector there later.
- Open PR #2794 (Temporal Self / attention / reverie handoff doc): docs only,
  no overlap.
- No open PR touches `world_first.py`, `attention_candidate.py`,
  `attention_broadcast.py` or `orion-vision-window/app/presence.py`.

## Non-goals

- Not the street. No `known_person_arrived`, nothing from declined #2906.
- No identity dependency for the arrival event. A name is an optional
  annotation only.
- No look on every frame or every person trigger. Looks run only after an
  arrival wins attention, capped and priced.
- No name stronger than the face verdict, and no name for people Juniper has
  not enrolled and named.
- No picture, crop or embedding in the stored scene or in Orion's prompt.
- No change to BLIP or the walkway/doorbell caption lane.
- No retirement of the router's reflex face check in these patches. Whether
  the look replaces it is a later question, decided with look data.
- No change to the world-first ranking rule, band cuts, or the camera
  surprise score.
- No greeting, outreach or chat behavior written by hand. Downstream changes
  come only from the existing consumers of a world winner.
- No new walkway or doorbell logic.
- No retention table for the events beyond the existing node history and
  `vision_presence_transition`.

## Proposal-mode items

- **Capability change.** Orion's attention can point at "someone just came
  into the room" as its own event, about 1-2 times a day, instead of only when
  the picture changes a lot.
- **Data touched.** Reads `vision_presence_transition` state already held in
  orion-vision-window memory. Writes one bus message per arrival and one
  substrate node (FalkorDB) plus its prediction-error history rows. No new
  table.
- **Privacy boundary.** None between Juniper and Orion. But household members
  and guests also walk into the room. The event carries no picture and no
  embedding. A name appears only on a `probable` face match, the same rule the
  transition row already follows. Guests stay anonymous.
- **Trace that proves it worked.** A `room_arrival` bus message with
  `transition_id`, the node's history row at 1.0, and a broadcast log tick with
  `world_first.winner = node:substrate.room_arrival` within 90 s. SQL in
  "Acceptance checks".
- **Dangerous failure modes.**
  - A blind or person-blind camera reads as "nobody came" (calm). Covered by
    the absent rule for frames, and by Patch 0 for person-blindness.
  - A false-presence storm (a lighting pattern every few minutes) floods
    attention and crowds out the body. Covered by the 10 min / 60 s debounce;
    measured overnight arrivals after debounce: 0.
  - A name guessed onto a guest. Covered by "probable only, optional".
- **Look stage (Patches 3-4).** Capability: Orion can look once at who came
  in and remember a named scene. Data: one still per look (not stored by this
  design beyond its id), one face check, one VLM call, one `vision_events`
  row, one act-ledger row. Privacy: decision 4; names only for enrolled people
  Juniper named, never above the face verdict. Trace: the `look` act row
  linked to the arrival `event_id` and the `named_scene` row. Danger: a wrong
  name, an invented activity, a cost spiral, a stale scene (see "Failure
  modes"). Roll back: `ORION_LOOK_CAM0_ENABLED=false` stops new looks; the
  reader ignores scenes older than the max age.
- **Disable / roll back.** `VISION_ROOM_ARRIVAL_EVENTS_ENABLED=false`
  (producer) or `SUBSTRATE_ROOM_ATTENTION_ENABLED=false` (runtime). Either one
  makes the node stop being written; attention then reads it absent. Hard
  fence: add the node to `NO_ATTENTION_CONSUMER_NODE_IDS`.

## Replay: would it win?

At ~1.2-1.6 arrivals a day held for 300 s (~10 node readings each), about 1.3%
of the node's readings are non-zero. A fresh arrival then has a strict
percentile of ~0.987 (`high`, eligible) and a mid-rank of ~0.993. Over the
2,231 broadcast ticks with any winner (2026-10-27..11-03), the winner's rank
score was below 0.993 on 91.8% of them, below 0.997 on 97.2%. 66% of all
ticks had no winner at all. So at full strength a fresh arrival wins about
97% of all ticks (66% + 34% x 91.8%), and
loses mainly to Juniper chatting at the same moment (`world:chat`), strain
news, or a walkway p99 moment. Losing to Juniper talking is the right answer.
After the 60 s grace the fade lowers the score, so a busy world can take over
inside the window.

The 7 debounced arrivals inside the broadcast log's 7 days: 3 had no
attention winner at all at the moment of confirmation (for example 10-27 14:12
UTC, a 443-minute sit-down that room surprise missed). 4 were held by
`execution` (2), `cabinet` then room surprise (1), or room surprise (1).
3 of 7 had a room-surprise win within 2.5 minutes.

## Acceptance checks

Patch 0 (person-blind check):

```sql
-- Person windows vs all windows per day on the room camera.
select observed_at::date, count(*) windows,
       sum((believed_labels ? 'person')::int) person_windows
from vision_scene_inventory
where stream_id = 'cam0' and observed_at > now() - interval '14 days'
group by 1 order by 1;
```

Pass: with Juniper sitting at the desk for 5 minutes, `person_windows` moves
above 0 and a `vision_presence_transition` row `absent -> present` appears.
Then the ladder has a `person_rate:cam0` rung (same median rule as
`detection_rate`, counting `person` only).

Patch 0 face fix: the face verdicts on the room camera move off `no_face` /
`unsure`. Pass: during Juniper's sit tests, at least half of face checks read
`possible` or better, and some read `probable`. The exact query and baseline
come from the pending findings (baseline since 11-02: 18 checks, 0 `probable`).

Patches 1-2 (producer and runtime), after 7 live days:

```sql
-- 1. Arrivals per day, and none overnight from flicker (expect 0.5-3/day).
select (occurred_at at time zone 'America/Denver')::date d, count(*)
from substrate_node_prediction_error_history
where node_id = 'node:substrate.room_arrival' and value >= 0.999
group by 1 order by 1;

-- 2. Non-zero share stays small enough to read 'high' (expect < 5%).
select avg((value > 0)::int) from substrate_node_prediction_error_history
where node_id = 'node:substrate.room_arrival'
  and observed_at > now() - interval '7 days';

-- 3. Each debounced arrival wins at least one broadcast tick (expect >= 80%).
with a as (
  select observed_at t from substrate_node_prediction_error_history
  where node_id = 'node:substrate.room_arrival' and value >= 0.999
)
select count(*) arrivals,
       count(*) filter (where exists (
         select 1 from substrate_attention_broadcast_log l
         where l.generated_at between a.t and a.t + interval '300 seconds'
           and l.projection_json->'frame'->'debug'->'world_first'->>'winner'
               = 'node:substrate.room_arrival')) won
from a;

-- 4. Blind never reads calm: no node rows while cam0 was unknown.
select count(*) from substrate_node_prediction_error_history h
join vision_presence_transition u
  on u.presence_id = 'cam0' and u.to_state = 'unknown'
 and h.observed_at between u.occurred_at + interval '180 seconds'
                       and u.occurred_at + interval '10 minutes'
where h.node_id = 'node:substrate.room_arrival';
```

Behavior (the part that matters), compared with the 7 days before go-live:

```sql
-- 5. Reverie thinks about arrivals (expect >= 50% of arrivals followed by a
--    thought naming the node within 5 min; baseline: 0, the node does not exist).
with a as (
  select observed_at t from substrate_node_prediction_error_history
  where node_id = 'node:substrate.room_arrival' and value >= 0.999
)
select count(*) arrivals,
       count(*) filter (where exists (
         select 1 from substrate_reverie_thought r
         where r.created_at between a.t and a.t + interval '5 minutes'
           and r.thought_json::text like '%node:substrate.room_arrival%')) thought
from a;

-- 6. An attention arc opens on the arrival (expect >= 50% with the 300 s hold).
select count(*) from temporal_self_arc
where kind = 'attention' and subject_ref like '%room_arrival%'
  and began_at > now() - interval '7 days';

-- 7. It does not crowd out the body: share of broadcast ticks it wins
--    (expect < 2%; ~1.3% of readings are non-zero).
select avg((projection_json->'frame'->'debug'->'world_first'->>'winner'
            = 'node:substrate.room_arrival')::int)
from substrate_attention_broadcast_log
where generated_at > now() - interval '7 days';
```

Patches 3-4 (look and named scene), after 7 live days:

```sql
-- 8. Looks per day, and why looks did not run (expect <= cap; priced_out
--    rises when the cabinet is hot).
select (opened_at at time zone 'America/Denver')::date d,
       count(*) looks,
       count(*) filter (where outcome_note like 'declined:%') declined,
       avg(outcome_value) expect_met_rate
from grammar_acts
where verb = 'look' and object_ref = 'camera:cam0'
group by 1 order by 1;

-- 9. Look -> named rate, and the face verdict distribution
--    (expect named >= 50% of looks on Juniper's sit-downs after Patch 0).
select entities->>'face_label' label, entities->>'face_strength' strength,
       count(*)
from vision_events
where event_type = 'named_scene' and stream_id = 'cam0'
  and created_at > now() - interval '7 days'
group by 1, 2 order by 3 desc;

-- 10. Caption grounded rate (expect >= 90%; failures keep a reason).
select avg((entities->>'grounded')::boolean::int) grounded_rate,
       count(*) filter (where not (entities->>'grounded')::boolean) failed
from vision_events
where event_type = 'named_scene' and created_at > now() - interval '7 days';

-- 11. Orion cites the named scene (chat or reverie within 30 min of it).
with n as (
  select created_at t, narrative txt from vision_events
  where event_type = 'named_scene' and (entities->>'grounded')::boolean
    and created_at > now() - interval '7 days'
)
select count(*) scenes,
       count(*) filter (where exists (
         select 1 from substrate_reverie_thought r
         where r.created_at between n.t and n.t + interval '30 minutes'
           and r.thought_json::text ilike '%named_scene%')) in_reverie
from n;
```

The act ledger's own posterior for (`look`, `look_face_verdict_possible`)
is the payoff check: it says whether looks meet their `expect`. Chat citation
is checked from the turn record (`cockpit_turn_sighting`): the situation block
must contain the scene text, and the reply may mention it. Column names in
checks 8-11 match `grammar_acts` (`opened_at`, `outcome_value`,
`outcome_note`) and `vision_events` (`narrative`, `entities`, `stream_id`) as
written by today's code. The `named_scene` keys inside `entities` are new in
Patch 4; the eval script is the binding version.

Column names in checks 5-6 were checked against the live tables on
2026-11-03. The `kind = 'attention'` value is to be confirmed in Patch 2's
eval; the eval script is the binding version.

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/3043

## Recommended next patch

Patch 0, before anything else, has two parts. First, fix room-camera face
recognition, from the pending investigation's findings (see "Patch 0
findings"). Second, a `person_rate:cam0` ladder rung. It reuses
`orion/vision_detection_rate_watch.py`, counting only `person` windows instead
of any detection, and its replay must go red on 2026-10-30..11-02 unless
Juniper confirms she was away. Then a 5-minute sit test with Juniper. If the
detector is person-blind, fix that first; an arrival event on a blind detector
is a silent calm.

Then Patch 1 (contract plus producer: `RoomArrivalEventV1`, channel, gate in
orion-vision-window, replay eval over `vision_presence_transition`), then
Patch 2 (runtime node, glossary, fence, world label). Patch 2 lands after
`chore/rung6-retire-tuner`.

Then Patch 3 (look act: allocator kind, price, cap, executor, closed
`expect`), then Patch 4 (named scene stored with grounding, read by
`perception_reader.py`). Patch 4 waits for decision 4.
