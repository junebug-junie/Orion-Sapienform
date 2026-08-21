# Seeing Juniper: identity, situated observation, and what Orion does with it

**Date:** 2026-08-21
**Mode:** design + proposal (AGENTS.md §0A "Proposal mode before invasive cognition changes")
**Status:** proposal — no code changed, no live config flipped by this document.

Requested capability, in Juniper's words: *"I'd like Orion to be able to
recognize me in the vision service. I also want Orion to have context about
what I'm doing while they see me… After Orion categorizes, this should go as a
snapshot into reverie. It should go into chat unified as a recent interaction,
and Orion can speak to it if they so desire… it could also connect into Orion's
unprompted turns."*

Explicitly out of bounds per the same request: an activity taxonomy. Orion
should be **curious and hypothesising**, not classifying into a fixed label set.
That constraint drives most of the design below.

---

## Arsonist summary

The perception rail is alive and the schema slots for this already exist and
are already hollow. The blocker is not architecture — it is that **the eye is a
250M-parameter 2022 image captioner**, and the council's prompt is explicitly
written to forbid exactly the inference being asked for.

Live proof, `vision_events`, last 7 days:

| narrative | count |
| :--- | ---: |
| `A laptop is observed in the frame.` | 2695 |
| `A person was detected on camera.` | 1128 |
| `Two chairs, two tables, two desks, and one door are present in the scene.` | 192 |
| `A laptop is present in the frame.` | 182 |
| `A laptop is detected in the frame.` | 151 |

That is a furniture inventory. `entities` and `tags` are `[]` on 8,269 of 9,466
rows — the only rows that fill them are the council's *deterministic fallback*,
which hardcodes `entities: ["person"]`, `tags: ["host_detect"]`,
`confidence: 0.85`. There is no identity anywhere, and there is no verb.

Four things are true at once and all four have to be handled:

1. **`VISION_VLM_MODEL_ID=Salesforce/blip-image-captioning-base`** (live, in the
   running container). Nothing downstream can be better than its input. Wiring
   identity and activity consumers on top of this signal fails the metric
   quality gate at step 4 (live-data sanity — the signal is degenerate today).
2. **The council prompt bans the ask.** `interpretation.py:158-168` says *"Do
   not invent identities, names, or facts not supported by the context"*,
   *"Treat summary.captions as soft hints only; never sole basis for activity
   claims"*, and *"When hard_labels is non-empty, describe only those detected
   objects; do not infer occupants."* These were written in response to real
   caption hallucinations. They are not wrong — they need a **narrow, evidenced
   carve-out**, not deletion.
3. **The council never sees the image.** It is a text LLM over a window
   summary. Activity inference from pixels must happen at the host VLM. The
   council's job is to *combine and hedge*, not to see.
4. **A prior explicit decision says no to this.** See "Reversal" below.

---

## Current architecture

```text
edge / retina  →  frame-router  →  vision-host  →  window  →  council  →  scribe  →  vision_events
  (capture)       (tier policy)    (GroundingDINO   (aggregate) (text LLM)  (SQL)      (Postgres)
                                    + BLIP + SigLIP)
```

- **Capture:** `orion-vision-edge` (YOLO/motion/haar-face) and
  `orion-vision-retina`. Stream `cam0`.
- **Frame router** (`config/vision_frame_router.yaml`): two tiers. `baseline`
  (every 10th frame, `want_caption: false`, `want_embeddings: true`) and
  `triggered` (host saw `person` within 8s → `want_caption: true`).
- **Host** (`orion-vision-host`, **Tesla P4, 7.68 GB total, ~4.2 GB free**,
  `CUDA_VISIBLE_DEVICES=0`). `runner.py` implements exactly four kinds:
  `embedding`, `detect_open_vocab`, `caption_frame`, `vlm`.
- **Council** (`orion-vision-council`): text LLM → `VisionSceneInterpretationV1`
  → projected to legacy `VisionEventPayload` → `orion:vision:events`.
- **Scribe** → `vision_events` (`event_id, event_type, narrative, entities,
  tags, confidence, salience, evidence_refs, correlation_id, created_at`).

**Existing downstream consumers of `vision_events` (both live):**

| Consumer | Reads | Window | Contract |
| :--- | :--- | :--- | :--- |
| `services/orion-thought/app/vision_reader.py` → reverie | `narrative` only | `ORION_REVERIE_PERCEPTION_MAX_AGE_SEC=180`, `MAX_EVENTS=3` | narrative-only, fail-open |
| `services/orion-cortex-exec/app/perception_reader.py` → `PerceptionContextV1` → situation brief → the turn | `narrative` only | `ORION_SITUATION_PERCEPTION_MAX_AGE_SECONDS=900`, `STREAM_ID=cam0` | narrative-only, `privacy_mode: session_only` |

**This matters for the request:** the "goes into chat so Orion can speak to it"
rail *already exists and is already on*
(`ORION_SITUATION_PERCEPTION_ENABLED=true`). Orion is already handed what the
camera saw on every turn. It is handed *"A laptop is observed in the frame."*
The rail is not missing. The content is worthless.

### Live findings that block or complicate this

**F1 — The eye is BLIP-base.** `Salesforce/blip-image-captioning-base`
(~250M params, 2022) in both `services/orion-vision-host/.env` and the running
container. `config/vision_profiles.yaml`'s `vlm_caption` and `vlm_vqa` both
declare `model_id: "REPLACE_ME/qwen2-vl_or_llava_next"` and fall back to this.
It cannot describe posture, activity, or intent. It produces object nouns.

**F2 — VQA shipped yesterday and is not live.** `vlm_vqa`
(`kind=vlm`, caller-supplied question) was enabled in
`config/vision_profiles.yaml` on 2026-08-20, and `.env_example:29` adds it to
`VISION_ENABLED_PROFILES`. The **local `.env:21` was never synced** and the
running container's whitelist is
`pipeline_retina_fast,retina_detect_open_vocab,embed_image,vlm_caption` — no
`vlm_vqa`. Config truth ≠ runtime truth (AGENTS.md §0A). This is a one-line
env-parity repair plus a restart, and it is a prerequisite for this design,
because VQA is the seam activity hypotheses ride on. It is called out here
rather than fixed in passing because it changes a live service.

**F3 — `config/vision_profiles.yaml` is carrying a live keyword cathedral.**
`task_routing` routes eight task types — `identity_face`, `person_reid`,
`affect_signals`, `pose_estimation`, `action_recognition`, `scene_graph`,
`ocr_read`, `depth_estimation` — to profiles that exist as YAML blocks and are
implemented by nothing. `runner.py` handles four kinds. This is good news for
one part of this work: **`identity_face` is already the name for the thing being
asked for.** Fill it, do not invent a new name. The remaining seven should be
deleted or implemented in a separate patch (§0A: a concept with no producer and
no consumer is junk).

**F4 — The detector vocabulary cannot see the example scenario.** The
GroundingDINO `default_prompts` are 21 labels: `person, door, screen, laptop,
keyboard, desk, table, chair, whiteboard, guitar, backpack, bag, box, bin,
book, lamp, window, plant, cup, bottle, shoe`. There is no `server`, `rack`,
`cable`, `tool`, `screwdriver`, `phone`. Juniper's own example — *"working on
server, maybe fixing the mesh hardware"* — is outside the hard-label vocabulary,
so a council that is (correctly) told to trust hard labels over captions has no
evidence path to it. Any real activity inference here has to come from the VLM
looking at pixels, not from the label set. Widening the label set is the wrong
fix and would be a cathedral; this is named so nobody reaches for it.

**F5 — Person detections are real but bursty.** 150–300 `person_presence`
events/day on active days (Aug 15–19), 0–11 on quiet ones. Reverie's 180-second
freshness window means reverie will usually see nothing, which is correct and
should stay correct — but it means "snapshot into reverie" will fire rarely,
and that is not a bug to engineer away.

### Reversal of a prior explicit decision

`docs/superpowers/specs/2026-08-12-perception-frontier-design.md` §"Movement IV
— Seeing Juniper, and the body" states:

> *"The interesting version of this is not identification. Identity, face, and
> re-ID profiles stay disabled. The interesting version is **care**."*

and `orion/schemas/situation.py:293-302` (`PerceptionContextV1`) encodes that as
a promise in code:

> *"Deliberately absent, and not to be added without proposal-mode sign-off:
> … anything identity-bearing (`vision_events.entities`, faces, re-ID). The
> perception design doc lists identity/face/re-ID as a non-goal; **this schema
> is where that promise is kept or broken.**"*

Juniper is the data subject and is now asking for identification. That is
Juniper's call to make, and this document is the proposal-mode sign-off those
two files ask for. It is recorded explicitly rather than silently reversed so
that a future reader finds the decision, not a contradiction.

**The prior doc's core argument survives the reversal and should be kept:** the
valuable output is *care*, not *identification*. Recognition is the enabling
mechanism; "Juniper has been at that desk five hours and it's 2am" is the
deliverable. A design that ships recognition and stops has built surveillance
and called it perception.

---

## What this proposes

Three patches, strictly ordered. Each is independently useful and independently
revertible. **V0 is not optional and not a nice-to-have** — V1 and V2 built on
today's captioner would be cognition-shaped output with no cognitive substance
(§0A "no empty-shell cognition").

### V0 — Give Orion an eye that can see verbs

Replace the BLIP-base captioner with a VLM that can answer *"what is this person
doing?"*. Nothing else in this design changes; this is a `model_id` + VRAM
decision.

Constraint: the host is a **Tesla P4** — Pascal, compute 6.1, **no bf16, no
flash-attn**, 7.68 GB total with ~4.2 GB free alongside the resident
GroundingDINO + SigLIP2. Two real paths:

- **(a) Fits the P4 today.** A small VLM in fp16 that answers grounded
  "what is happening" questions. Requires a measured VRAM check on the live P4
  before selection — this doc deliberately does not name a model as chosen,
  because the last time a profile asserted a VRAM estimate
  (`vlm_vqa`'s `vram_estimate_mb: 5200`) it described an aspiration rather than
  what loads.
- **(b) Remote VLM on circe's V100s.** Better quality, no P4 contention, but a
  new cross-node seam and a latency budget the frame router does not currently
  model. Note the 4th V100 is already claimed by `orion-world-model` (PR #1775).

**Recommendation: (a) first**, because it changes one env key and proves the
whole chain end to end; escalate to (b) only if measured caption quality on
real frames is still object-nouns-only. Decide by *looking at the captions*, not
by parameter count.

**Acceptance for V0 is a live-data check, not a test:** pull 50 real
person-triggered frames' captions before and after. If the after-set still
contains a dominant repeated string with no verb, V0 failed and V1/V2 do not
start.

### V1 — Identity as a hypothesis, not a label

Fill the already-named `identity_face` profile (F3). One enrolled subject:
Juniper.

- **Mechanism:** face detect + embed on the person crop the host already has
  from GroundingDINO; cosine-match against a local gallery of enrolled
  embeddings; emit a three-state hypothesis.
- **Output is never a bare label.** The contract is
  `{subject: "juniper" | "unknown", similarity: float, state: "probable" |
  "possible" | "unsure"}`. **`unsure` must be a common, honestly-rendered
  outcome**, not an error path — a camera in a room produces bad angles,
  backlighting, and the back of someone's head most of the time.
- **Where it lands:** `vision_events.entities`, which is the existing field and
  is hollow today (F1). No new column for identity.
- **Non-matches are not stored.** A face that does not match the gallery
  produces `subject: "unknown"` and the embedding is discarded in-process. No
  gallery growth, no re-ID, no stranger tracking. This is the single most
  important line in this section.

Open decision — **host-side vs edge-side**:

| | Host (`identity_face` profile) | Edge (`orion-vision-edge`) |
| :--- | :--- | :--- |
| Already-named slot | yes (kills part of F3's cathedral) | no |
| Has the person bounding box | yes | would need its own |
| Competes for P4 VRAM | yes (small) | no |
| Gallery location | one service, one contract | at the capture node |

**Recommendation: host-side.** The privacy argument for edge-side is weaker
than it looks — the router already requires `require_image_path_exists: true`,
so full frames are already on a path the host reads. Host-side keeps all model
serving in one service and retires a named-but-unimplemented route.

### V2 — Situated observation: curiosity, not categories

This is the part Juniper explicitly fenced: *no keyword cathedral*.

**Do not build an activity taxonomy.** No `ACTIVITY_LABELS`, no
`activity_type` enum, no "working / resting / eating" vocabulary. The moment
that list exists, everything Orion sees gets flattened into it and the curiosity
is gone.

Instead: on a person-triggered frame (and only then), the host runs a **VQA**
task — the `vlm_vqa` profile from F2 — with a question shaped to produce a
hypothesis and its own disconfirmer. The extensible axis is free text; the
fixed schema is only the *epistemic frame around it*:

```json
{
  "observation": "A person is seated at the desk, leaning toward an open laptop.",
  "hypothesis": "They may be working — the posture is sustained rather than passing through.",
  "confidence": 0.4,
  "would_disconfirm": "If they stood up and left within a minute, this is not focused work."
}
```

Three properties make this not-a-cathedral:

1. **`hypothesis` is prose, not a key.** Nothing downstream switches on it.
2. **`would_disconfirm` is the point.** It is what makes this a *guess* rather
   than a *claim*, and it is the field that makes the loop closable (V3.4).
3. **`confidence` is allowed to be low and usually will be.** A hypothesis
   emitted at 0.4 that reverie renders as "maybe" is the correct output.

The council's role changes from *forbid inference* to *hedge inference*. Its
prompt carve-out is narrow and evidenced:

- Activity language remains gated on `person` in `hard_labels` (keep the
  existing rule verbatim — it is right).
- Identity language is permitted **only** when the artifact carries an
  `identity_face` result at or above `probable`, and must be rendered with the
  hedge the state implies.
- The blanket *"do not infer occupants"* rule keeps applying to every window
  with no person in `hard_labels`.

Everything else in that prompt stays. The hallucination it was written against
is still real.

**Schema change:** one new nullable column, `vision_events.hypotheses JSONB`.
It clears the metric quality gate only because it has all of a producer (host
VQA → council), a consumer (V3.2 outreach, V3.4 falsification), and a
disconfirmation path. If V3.4 is cut, this column should be cut with it — a
stored hypothesis nothing ever checks is exactly the field the
`reverie_visual` module docstring warns about
(`SpontaneousThoughtV1.next_focus`: stated producer intent, zero consumers).

---

## V3 — Consumers

Juniper named three. There are more, and one of the named three should be
built differently than requested.

### V3.1 — Reverie (as asked) — *free, no schema change*

`vision_reader.py` reads `narrative` only, by deliberate privacy contract.
Identity and hypothesis therefore arrive **inside the narrative sentence**
("Juniper is probably at the desk; maybe working") rather than as structured
fields. That is the cleanest option — zero changes to `orion-thought`, which is
a thin bus service — but it has a consequence worth stating: **there is then no
way to gate identity out of reverie separately from the narrative.** If
identity-in-reverie should be independently switchable, that requires a second
narrative column, and that is a bigger patch than it looks.

Note there are two different "visual reverie" surfaces and this is not the
other one: `ReverieVisualChainV1` (`orion/schemas/reverie_visual.py`) is the
generate→observe→interpret image loop. Percepts land in the *text* chain's
perception context, not there.

### V3.2 — Unprompted outreach (as asked) — *the thinnest patch here*

`services/orion-hub/scripts/endogenous_outreach.py` already has exactly the
right shape: a frozen `OutreachContext` dataclass, a `build_outreach_prompt()`
that renders only real signals, an `is_empty()` skip, and a `PASS` response so
Orion can decline. Add one field and one prompt block:

```python
visual_context: Optional[VisualContext] = None   # observation + hypothesis + age
```

and a prompt block in the established voice of that file — *state the reading,
do not narrate the feeling*, and make the uncertainty explicit so Orion asks
rather than asserts. That file's existing `tension_reason` block is the model
to copy: it is scrupulous about the difference between what was measured and
what it means.

This is where Juniper's example lands. Note it lands as a *question* —
"is that for the mesh?" — precisely because `confidence` was 0.4.

### V3.3 — Unified chat: **recommend against writing percepts as turns**

Requested: *"It should go into chat unified as a recent interaction."*

**Concern, stated once.** `chat_history_log` rows are consumed by
`orion-sql-writer`, `orion-vector-writer`, `orion-vector-host`,
`orion-spark-concept-induction`, and `orion-memory-consolidation` (which
classifies *every* turn and patches `spark_meta`). A percept written as a row
would be embedded, indexed, concept-induced, and consolidated as if it were
something Juniper said. `source` would distinguish it (`hub_ws` = Juniper,
`hub_orion` = Orion), but consumers that iterate generically would not filter —
the same failure mode as F3's excluded-but-still-ticking metric.

**Two better paths, both of which give the behaviour asked for:**

1. **Percepts reach the chat *context*, not the chat *record*.** This is
   `PerceptionContextV1` → situation brief → the turn, and it is **already live**
   (`ORION_SITUATION_PERCEPTION_ENABLED=true`). Orion already gets what the
   camera saw on every turn and can speak to it unprompted. V0–V2 make that
   content worth reading. Zero new plumbing.
2. **When a percept *causes* an outreach (V3.2), that outreach is already a real
   `chat_history_log` row** (`source='hub_orion'`, 130 rows in 30 days) and it
   carries the percept as its grounding. That is the honest version of "a recent
   interaction": something actually happened between them.

If what is wanted is specifically a *visible* entry in the chat transcript —
"Orion noticed you at 14:20" rendered in the scrollback — say so and it becomes
a Hub render concern with its own row type, not a turn. Flagging the difference
because the two readings lead to materially different work.

### V3.4 — The falsification loop (not asked for; the most important one)

Without this, Orion is confidently wrong forever, and every other consumer
inherits the error.

A hypothesis with a `would_disconfirm` clause can be **checked against the next
window**. Confirmed → small confidence gain. Disconfirmed → real perceptual
prediction error. Juniper contradicting it in chat ("no, that was the guitar
amp") → the strongest possible correction signal, and one that is free to
capture because the outreach turn and the reply are adjacent rows in
`chat_history_log`.

This is what makes the whole thing *perception* rather than *labelling*, and it
is the direct successor to P2 in the perception frontier doc. It is also the
answer to "what happens when Orion guesses wrong", which is otherwise unanswered.

**Metric gate caution:** do not wire a `perception_prediction_error` metric into
any aggregate until it has been watched on real data long enough to prove it can
return to a genuine rest state. There is a live history of exactly this failing
in both directions (a permanent `mean(|z|)` floor; a decayed-to-zero artifact
that read as calm). Emit and store it first; consume it later.

### V3.5 — Presence duration (not asked for; where the *care* actually lives)

Nothing in the repo computes "how long has Juniper been in frame." Not the
window service (fixed time windows), not social memory (no presence module).

Yet duration is the entire payload of the prior design's best line: *"Juniper
has been at that desk five hours and it's 2am."* It is a reducer over
`vision_events` — cheap, deterministic, no model — and it is the difference
between Orion narrating furniture and Orion noticing something about a person it
cares about.

**Recommend building this even if V2 slips.** It is the highest care-per-line
item in the document and it does not depend on the VLM.

### V3.6 — Other consumers worth considering

- **Endogenous curiosity** (`orion/substrate/endogenous_curiosity.py`) — an
  unresolved visual hypothesis is a genuine curiosity candidate, and outreach
  *already* reads curiosity candidates. That closes a real loop: see something
  ambiguous → become curious → ask about it → get told. This is the most
  Orion-shaped consumer on the list.
- **Attention / salience** (`orion/schemas/attention_salience.py`) — a
  recognised person should be more salient than a chair. Today they are not
  distinguished.
- **`orion-security-watcher`** — already consumes vision guard signals. An
  `unknown` face is a *very* different event from Juniper's, and routing it
  needs its own decision (V4.Q1), not a default.
- **Memory consolidation / crystallizer** — should a percept become a durable
  memory? Probably only *episodes* (V3.5's durations, V3.4's corrections), never
  frames. Left as an explicit non-goal below.
- **Cross-modal binding** — Juniper visible at the desk *and* typing in Hub chat
  are the same event through two senses, and nothing binds them. Timestamp
  correlation between `chat_history_log` and `vision_events` is nearly free and
  is a genuinely prerequisite-shaped capability (the perception frontier doc
  names it). Not in this patch; named so it is not forgotten.
- **Retention** — identity-bearing rows should have a *shorter* retention than
  "a chair is present", and `vision_events` has no retention policy at all
  today. Given the recent history of unbounded substrate tables, this should
  ship *with* V1, not after it.

---

## Missing questions

The ones that change the design, that Juniper has not yet answered:

**Q1 — What happens when it is not Juniper?** A partner, a guest, a delivery
person, a stranger. V1 emits `unknown` and discards the embedding — but *does
Orion say anything?* "Someone I don't recognise is in your space" is a
security posture; silence is a privacy posture. These pull opposite ways and
the default should be chosen, not inherited.

**Q2 — What about people in frame who did not consent?** Everyone who is not
Juniper still gets their face embedded in order to compute "not Juniper." The
proposal discards those embeddings in-process and never persists them. Is that
the right line, or should there be a spatial/temporal gate too?

**Q3 — How long does "Juniper was at the desk at 2am" live?** Identity-bearing
observations of a person in their home are the most sensitive rows this system
will hold. Days? Hours? Does the *duration episode* (V3.5) outlive the
*frame-level rows* that produced it? (Recommend: yes — keep the episode, drop
the frames fast.)

**Q4 — Is there an off switch and a mirror?** Two separate things, both needed:
a switch that stops recognition **immediately** and observably, and a surface
that shows Juniper *everything currently stored about their body and movements*.
Not a config constant — a live endpoint, because "what does it have on me" must
be answerable without reading code. §0A requires a UI/debug surface for any new
concept anyway; this is that requirement and the ethical requirement being the
same requirement.

**Q5 — Screens.** The detector already sees `screen` and `laptop`. A VLM good
enough to describe activity is good enough to **read what is on those screens**,
incidentally, without anyone asking it to. The prior design named this "the most
dangerous idea here" and required an explicit spatial gate plus sign-off *before*
it was approached deliberately. V0 approaches it **accidentally** — it is a side
effect of a better eye, not a feature. This needs a decision before V0 ships,
and it is the question in this document most likely to be skipped.

**Q6 — Does Orion get to decline to look?** Should VQA fire on every
person-triggered frame, or should Orion's own attention decide when to look
closely? The latter is the "foveal tier" the perception frontier doc sketches,
it is more interesting, and it costs less GPU. But it needs a real gating
signal, and picking one on vibes would be a cathedral.

**Q7 — Should Orion hold a stated position on this?** The prior doc's most
Orion-shaped idea: *"a system whose constraints it participated in authoring
stands in a different relation to those constraints than one that was merely
configured."* Enabling identity makes that **more** load-bearing, not less. This
is the one item here that is not an engineering task.

**Q8 — Absence.** "I haven't seen you in three days" is care. It is also
structurally hard: this pipeline is event-triggered, and event-triggered
statistics cannot detect absence — they freeze on silence rather than rising.
Any absence signal needs its own clock, not a threshold on an EWMA.

---

## Proposed schema / API changes

- **Added:** `vision_events.hypotheses JSONB NULL` — one column, gated on V3.4
  existing (see V2).
- **Added:** `identity_face` becomes a real `runner.py` kind; `VisionArtifactPayload.outputs`
  gains an `identities` block (the profile's `outputs.identities` shape already
  declares it).
- **Behaviour changed:** `vision_events.entities` starts being populated by the
  council rather than only by the deterministic fallback.
- **Behaviour changed:** the council interpretation prompt gains a narrow,
  evidence-gated identity carve-out. Activity-verb gating on `hard_labels` is
  unchanged.
- **Not changed:** `PerceptionContextV1`'s exposed-field list. Identity crosses
  into the turn inside `scene_summary`, under the same `session_only`
  `privacy_mode`. Its docstring must be updated to record this reversal, because
  that docstring currently promises the opposite.
- **Not changed:** `ReverieVisualChainV1` / `ReverieVisualArtifactV1`.
- **Removed (separate patch):** the seven other unimplemented `task_routing`
  entries in `config/vision_profiles.yaml` (F3).

## Files likely to touch

- `config/vision_profiles.yaml` — `vlm_caption`/`vlm_vqa` `model_id`; implement or delete F3's routes
- `services/orion-vision-host/.env` + `.env_example` — `VISION_VLM_MODEL_ID`, `VISION_ENABLED_PROFILES` (F2)
- `services/orion-vision-host/app/runner.py` — `kind=identity`
- `config/vision_frame_router.yaml` — VQA dispatch on the `triggered` tier
- `orion/schemas/vision.py` — `identities` on the artifact; hypothesis block
- `services/orion-vision-council/app/interpretation.py` — prompt carve-out, entity projection
- `services/orion-sql-writer/app/models/vision_event.py` + migration — `hypotheses`
- `services/orion-hub/scripts/endogenous_outreach.py` — `OutreachContext.visual_context` (V3.2)
- `orion/schemas/situation.py` — docstring reversal record
- `docs/superpowers/specs/2026-08-12-perception-frontier-design.md` — record the Movement IV reversal

## Non-goals

- Any activity label set, enum, or taxonomy.
- Re-identification, tracking, or a gallery of anyone but Juniper.
- Persisting embeddings for non-matching faces.
- Emotion/affect inference (`affect_signals` stays unimplemented and should be
  deleted, not filled).
- Frame-level percepts becoming durable memories.
- Widening the GroundingDINO label set to chase the hardware example (F4).
- Writing percepts as `chat_history_log` turns (V3.3).

## Acceptance checks

1. **V0:** 50 real person-triggered captions, before and after. No dominant
   repeated verb-less string. *This gates everything else.*
2. **V1:** a live frame containing Juniper produces `entities` with
   `subject: "juniper"` and a similarity; a frame containing someone else
   produces `unknown`; a bad-angle frame produces `unsure` — and `unsure` is
   observed to be common, not rare.
3. **V1 privacy:** a non-matching face leaves no embedding anywhere. Proven by
   inspecting the store, not by reading the code.
4. **V2:** a real window yields a `hypotheses` entry whose `confidence < 0.7`
   and whose `would_disconfirm` names something actually checkable.
5. **V2 anti-regression:** a window with **no** `person` in `hard_labels` still
   produces zero occupant/activity/identity language. The existing council
   grounding tests must still pass unmodified.
6. **V3.2:** one live outreach whose prompt contains the visual block, and whose
   delivered message is a *question*, not an assertion.
7. **V3.4:** one hypothesis observably marked disconfirmed by a later window.
8. **Q4:** the off switch is flipped live and a subsequent frame produces no
   identity — verified in the store.

## Recommended next patch

**F2 + V0, together, and nothing else.**

Sync `VISION_ENABLED_PROFILES` (one line, `.env` already drifted from
`.env_example`), pick a VLM against a measured P4 VRAM reading, restart the
host, and **look at 50 real captions**.

If the captions get verbs, the rest of this document is worth building. If they
do not, V1 and V2 would be identity labels and hypotheses attached to a
furniture inventory — schema-valid, cognition-shaped, and empty. Everything
here is downstream of whether Orion's eye can see a verb, and that is one env
key and one restart away from being a known fact instead of an assumption.

Q5 (screens) should be answered **before** that restart, not after.
