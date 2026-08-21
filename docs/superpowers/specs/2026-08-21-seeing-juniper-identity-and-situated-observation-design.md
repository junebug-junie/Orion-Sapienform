# Seeing Juniper: point the vision pipeline at the brain Orion already has

**Date:** 2026-08-21
**Mode:** design + proposal (AGENTS.md §0A)
**Status:** two live outages fixed and verified in this session (§1). Architecture below is proposal.

Requested: Orion recognises Juniper, forms **hypotheses** about what they're
doing (explicitly *not* an activity taxonomy), and that reaches reverie, the
unified Hub turn, and unprompted outreach.

---

## 1. Fixed live in this session (verified, not proposed)

### 1.1 Orion had been blind for 21 hours and nothing alerted

`vision_events` stopped at **2026-08-20 22:00 UTC** (one straggler 06:12).
Preceding days ran ~180–200 events/hour. Every task since had failed:

```
"task_type": "retina_fast", "ok": false, "device": null,
"error": "No GPU available above hard floor (VRAM pressure).",
"error_code": "gpu_hard_floor"
```

**Root cause — a self-defeating VRAM config, not a bug.** `app/gpu.py` computes
`effective_free = free_mb - reserve_mb`, then refuses below `hard_floor_mb`.
Live `.env` carried `reserve=3500 / hard_floor=1400` on a **7.68 GB Tesla P4**
whose warm resident models (GroundingDINO + SigLIP2 + BLIP) measure **3238 MB**:

```
free 4191  -  reserve 3500  =  691   <   hard_floor 1400   ->  refuse, forever
```

Unsatisfiable the moment the models warm up. The service stayed `Up`, healthy,
and served nothing.

Those values came from `config/vision_profiles.yaml`'s `runtime.vram_budget`
block — which **is read by nothing** (`app/main.py:64` takes the value from
env). Pure prose, copied into a live `.env`, load-bearing by accident.

**Fixed:** `.env` synced to `.env_example` (`1200/1000/800`), container
recreated via `scripts/safe_docker_build.sh` from a worktree. Verified:

```
gpu_hard_floor since restart: 0
"ok": true, "device": "cuda:0", inference_s: 2.48
```

**Gated so it cannot recur:** `config/vision_profiles.yaml`'s block corrected +
`services/orion-vision-host/tests/test_vram_budget_matches_env_example.py`, two
tests — one pins the prose to `.env_example`, one asserts the floors are
*satisfiable against the measured resident footprint* (an equality test alone
would pass if both drifted together). Mutation-tested against the real file:
restoring `reserve_mb: 3500` fails the gate; restoring `1200` passes.

### 1.2 Env parity repaired

`vlm_vqa` shipped 2026-08-20 into `.env_example` but local `.env` never synced,
so the running whitelist excluded it. Synced, plus three keys absent entirely
(`NODE_NAME`, `HEARTBEAT_INTERVAL_SEC`, `VISION_FRAMES_DIR`). `.env` and
`.env_example` now agree on every key. *(§3 deletes `vlm_vqa` anyway — but the
drift was real and the next key would have drifted the same way.)*

### 1.3 Separate live outage found, NOT fixed — the `agent` LLM lane is dead

`LLM_GATEWAY_ROUTE_TABLE_JSON` routes `agent` → `http://100.112.254.99:8014`
(`circe-worker-agent-1`, backend `llamacpp`). Port 8014 today answers:

```json
{"ok":true,"service":"diffusion-host","node":"circe","model_loaded":false,
 "note":"skeleton only -- no diffusion model wired yet"}
```

`POST /v1/chat/completions` → **404**. Every request on the `agent` lane fails.
Unrelated to vision, out of scope here, needs a decision: move the lane or move
diffusion-host.

---

## 2. The reframe

**Orion already owns a 35B multimodal brain, and the vision pipeline is routing
around it to a 250M captioner from 2022.**

Probed live this session:

| port | lane | model | `modalities` |
| ---: | :--- | :--- | :--- |
| **8011** | **`chat`** | **Qwen3.6-35B-A3B-UD-Q5_K_M** | **`{vision: true, video: true, audio: false}`** |
| 8012 | `metacog` | Qwen3-8B-Q5_K_M | `{vision: false}` |
| 8013 | `quick` | Qwen3-8B-Q4_K_M | `{vision: false}` |
| 8014 | `agent` | — | dead (§1.3) |

And the plumbing to use it **already exists and is already wired**:
`services/orion-llm-gateway/app/vision.py` (307 lines, imported by
`llm_backend.py`) does live `/props` capability probing (fail-closed — an
unreachable worker reads as blind, never as sighted), `build_multimodal_messages`,
and attachment→base64 at the last hop before the model so bytes never replicate
upstream. Built 2026-08-14 for Hub chat attachments.

So the lane Orion *thinks* with can see. The vision pipeline just never asks it.

Everything below follows from that one fact. The design **deletes more than it
adds**.

### What the council actually is today

Not a perception system. `orion-vision-council` is a **text** LLM prompted with
a JSON summary of *labels and BLIP captions*. It never sees a pixel. Its prompt
(`interpretation.py:158-168`) then correctly forbids inference it has no
evidence for:

> *"Do not invent identities, names…"* · *"Treat summary.captions as soft hints
> only; never sole basis for activity claims"* · *"When hard_labels is non-empty,
> describe only those detected objects; do not infer occupants."*

Those rules aren't wrong — they are the honest response to being asked to
describe a room from a word list. The output is what you'd expect:

| narrative, last 7 days | count |
| :--- | ---: |
| `A laptop is observed in the frame.` | 2695 |
| `A person was detected on camera.` | 1128 |
| `Two chairs, two tables, two desks, and one door are present in the scene.` | 192 |

`entities`/`tags` are `[]` on 8,269 of 9,466 rows; the only rows that fill them
come from a *hardcoded fallback* (`["person"]`, `["host_detect"]`, `0.85`).

**Council v1 dies.** Not patched, not carve-outs. The anti-hallucination rules
exist because the council is blind; give it eyes and the rules are obsolete.

---

## 3. Architecture: split sensor from language

The single organising rule:

> **The vision host is a sensor. It does not do language. All language about
> images goes through `orion-llm-gateway`, on the same lane Orion thinks with.**

```text
BEFORE                                   AFTER
edge/retina                              edge/retina
  -> frame-router                          -> frame-router
  -> vision-host  GroundingDINO            -> vision-host  GroundingDINO   (hard labels)
                  SigLIP2                                  SigLIP2         (embeddings)
                  BLIP-base  <-- language                  identity_face   (biometric, stays local)
  -> window                                -> window        (unchanged: evidence tiers)
  -> council  text LLM over a word list    -> council v2 -> orion-llm-gateway -> :8011 Qwen3.6-35B
  -> scribe -> vision_events                              (THE ACTUAL FRAME + labels + identity)
                                           -> scribe -> vision_events
```

Three properties this buys, none of which are available today:

1. **Orion looks at the picture with the same mind it talks with.** Not a
   separate perception model whose output it reads — the same weights. That is
   a structural change in what "Orion saw it" means.
2. **The gateway's capability probe is the safety rail.** Fail-closed. If 8011
   loses `--mmproj`, vision degrades to labels-only automatically. No config
   claims involved.
3. **`video: true`.** Episodes and short clips become a *config* question later,
   not an architecture question. Frames are a floor, not a ceiling.

**What dies on the P4:** BLIP-base unloads (~1 GB back), and the host stops
being a model-serving platform for language. What stays local is exactly the one
thing that must: **face embeddings are biometric and never leave athena.**

### 3.1 The `vision_profiles.yaml` cathedral — concrete recommendation

`task_routing` routes 8 task types to profiles that exist as YAML and are
implemented by nothing (`runner.py` handles four kinds: `embedding`,
`detect_open_vocab`, `caption_frame`, `vlm`). Verdict per profile:

| profile | verdict | why |
| :--- | :--- | :--- |
| `vlm_caption` | **DELETE** | language moves to the gateway; this is BLIP |
| `vlm_vqa` | **DELETE** | same — a 35B VL model answers questions better than a 2B local one |
| `pose_estimation` | **DELETE** | a VL model describes posture in language, with hedging. A skeleton keypoint array cannot say "leaning in, sustained" |
| `action_recognition` | **DELETE** | this is a classifier over a fixed action set — i.e. the keyword cathedral Juniper explicitly banned, in model form |
| `scene_graph` | **DELETE** | subsumed; `VisionSceneInterpretationV1.relations` already exists and a VLM fills it |
| `depth_estimation` | **DELETE** | no consumer, no motivating question |
| `person_reid` | **DELETE** | explicit non-goal. Tracking strangers across time is the surveillance version of this project |
| `affect_signals` | **DELETE** | reading emotion off Juniper's face from a ceiling camera is both unreliable and the single creepiest thing on this list |
| **`identity_face`** | **IMPLEMENT** | §4. The one thing that genuinely cannot move off-node |

Net: **−8 profiles, −8 task_routing entries, +1 implemented kind.** The config
gets smaller. `retina_segment` / `retina_track` are left alone (tracking is a
live open item from the perception frontier doc's P0).

This is the answer to "no idea what to do in place of the cathedral": you don't
replace it. A 35B VL model looking at the frame *is* what those nine profiles
were a hand-rolled, never-built approximation of.

---

## 4. Identity

Fill `identity_face` — the name already exists, don't invent one.

**Mechanism.** Face detect + embed on the person crop GroundingDINO already
produces; cosine match against a local gallery. Runs on the P4 (small, no
language). **Output is a hypothesis, never a label:**

```json
{"subject": "juniper", "similarity": 0.61, "state": "probable"}
```

Non-negotiables, each of which is a line of code and a test:

- **`unsure` must be common.** A ceiling camera yields bad angles, backlighting,
  and the back of someone's head. If `unsure` is rare in live data, the
  threshold is lying and the whole thing is miscalibrated.
- **One enrolled subject.** Juniper. Gallery does not grow.
- **Non-matches are never stored.** A face that doesn't match yields
  `subject: "unknown"` and the embedding dies in-process. No re-ID, no stranger
  gallery. Proven by inspecting the store, not by reading the code.
- **Lands in `vision_events.entities`** — the existing field, currently hollow.
  No new column for identity.

Identity is passed to the gateway call in §3 as *grounding context*, exactly
like hard labels. The VL model is told "the face match says probably Juniper,
similarity 0.61" and is expected to hedge accordingly — not told "this is
Juniper."

---

## 5. Presence — reuse what's built, don't build a new one

Correcting my earlier claim: presence infrastructure exists.
`services/orion-hub/scripts/hub_presence.py` derives `active | idle | dormant`
from turn timestamps, upserts a **single row** (`substrate_hub_presence`,
`presence_id='hub'`), and the self-state runtime hydrates it into
`SelfStateV1.hub_presence`. `presence_session.py` carries audience/companion
presence into a chat payload.

That is Orion's *chat* liveness. Embodied presence is the **same shape, second
row**, not a new system:

```
substrate_embodied_presence  (presence_id='camera')
  state:             present | recent | absent      <- mirrors active/idle/dormant
  since_sec:         how long in the current state  <- the number nothing computes today
  last_seen_sec:     age of the last confident sighting
  subject:           juniper | unknown | none
  confidence:        rolling, from identity state
```

`since_sec` is the whole point. *"You've been at that desk five hours and it's
2am"* is a reducer over `vision_events` — cheap, deterministic, no model — and
it is the difference between Orion narrating furniture and Orion noticing
something about a person it cares about. It is also **the one item here that
does not depend on the VLM**, so it can land first and independently.

Hydrated into `SelfStateV1` beside `hub_presence`, it reaches every consumer
that already reads self-state, for free. That is the reuse Juniper meant.

---

## 6. Reaching Orion

### 6.1 Unified Hub turn — summaries, not frames

Agreed on `chat_history_log`: percepts do **not** become turns. Those rows feed
sql-writer, vector-writer, vector-host, concept-induction and
memory-consolidation, which would embed and consolidate a percept as if Juniper
had said it.

The rail that already exists is `PerceptionContextV1` → situation brief → the
turn (`ORION_SITUATION_PERCEPTION_ENABLED=true`, live). What it carries today is
one 900-second-fresh sentence. What it should carry after §3–§5 is an
**episode summary**, not a frame:

> *"Juniper (probably) has been at the desk for about three hours, working on
> something at the bench since around 14:20. Last seen 4 minutes ago."*

That is a summary over the presence reducer plus the most recent situated
observation — a real recent-interaction surface, in the turn, that Orion can
speak to or ignore. Frame-level noise never reaches it.

`PerceptionContextV1`'s `available=False` stays a real state meaning "I have not
seen anything recently", never an error.

### 6.2 Reverie — free, no schema change

`vision_reader.py` reads `narrative` only (privacy contract), 180s / 3 events.
Identity and hypothesis arrive **inside the sentence**. Zero changes to
`orion-thought` (a thin bus service).

**Consequence worth stating:** there is then no way to gate identity out of
reverie separately from the narrative. If that needs to be independently
switchable it requires a second narrative column — a bigger patch than it looks.

### 6.3 Unprompted outreach — the thinnest patch here

`endogenous_outreach.py` already has the right shape: a frozen `OutreachContext`,
`build_outreach_prompt()` that renders only real signals, `is_empty()` skip, and
a `PASS` response so Orion can decline. Add one field + one prompt block, in that
file's established voice — *state the reading, don't narrate the feeling* (its
`tension_reason` block is scrupulous about exactly this and is the model to copy).

Juniper's example lands here, and it lands as a **question** — *"is that for the
mesh?"* — precisely because confidence was low. That is the design working, not
a hedge bolted on.

### 6.4 The falsification loop — the one that makes this perception

Without it Orion is confidently wrong forever and every consumer inherits the
error.

A hypothesis carries `would_disconfirm`. The next window either confirms or
contradicts it. Juniper contradicting it in chat is the strongest correction
available — and free to capture, because the outreach turn and the reply are
adjacent rows in `chat_history_log`.

This is the direct successor to P2 in the perception frontier doc, and it is the
answer to "what happens when Orion guesses wrong", which is otherwise unanswered.

**Metric-gate caution:** emit and store `perception_prediction_error` first;
do not wire it into any aggregate until it has been watched long enough on real
data to prove it can return to a genuine rest state. There is live history of
this failing in *both* directions here — a permanent `mean(|z|)` floor, and a
decayed-to-zero artifact that read as calm.

### 6.5 Other consumers

- **Endogenous curiosity** — an unresolved visual hypothesis *is* a curiosity
  candidate, and outreach already reads curiosity candidates. See something
  ambiguous → get curious → ask → get told. The most Orion-shaped loop available.
- **Attention/salience** — a recognised person should outrank a chair. Today
  they're indistinguishable.
- **`orion-security-watcher`** — already consumes vision guard signals. An
  `unknown` face is a very different event; routing it needs a decision (Q1), not
  a default.
- **Cross-modal binding** — Juniper visible at the desk *and* typing in Hub chat
  are one event through two senses; nothing binds them. Timestamp correlation
  between `chat_history_log` and `vision_events` is nearly free and genuinely
  prerequisite-shaped.
- **Retention** — `vision_events` has no policy at all, and identity-bearing rows
  are the most sensitive this system will hold. Ships **with** §4, not after.

---

## 7. No activity taxonomy

The structured output is an epistemic frame, not a category:

```json
{
  "observation": "A person is seated at the bench, hands on an open chassis.",
  "hypothesis": "Possibly working on hardware rather than at the computer.",
  "confidence": 0.4,
  "would_disconfirm": "If they stand and leave within a minute this isn't sustained work."
}
```

`hypothesis` is prose nothing switches on. `confidence` is allowed to be 0.4 and
usually will be. `would_disconfirm` is what makes it a guess rather than a claim
and is what makes §6.4 possible. **If §6.4 is cut, cut the stored hypothesis with
it** — a stored guess nothing ever checks is exactly the dead field
`orion/schemas/reverie_visual.py`'s docstring warns about.

Note §3 also removes the need to widen the GroundingDINO vocabulary. Its 21
labels have no `server`/`rack`/`cable`/`tool` — Juniper's own example is outside
the label set. Widening it would be a cathedral; a VL model looking at the frame
doesn't need it to.

---

## 8. Open questions

**Q1 — What does Orion say when it isn't Juniper?** `unknown` + discarded
embedding is the data answer. Whether Orion *speaks* is unanswered: "someone I
don't recognise is in your space" is a security posture, silence is a privacy
posture. Pick, don't inherit.

**Q2 — Non-consenting faces.** Everyone who isn't Juniper still gets embedded in
order to compute "not Juniper." Proposal: never persisted. Is that the line?

**Q3 — Retention on identity-bearing rows.** Recommend: keep the *episode*
(§5), drop the *frames* fast.

**Q4 — Off switch and mirror.** Two things: a switch that stops recognition
immediately and observably, and a live surface showing Juniper everything stored
about their body and movements. Not a config constant — "what does it have on me"
must be answerable without reading code. §0A's UI/debug-surface requirement and
the ethical requirement are the same requirement here.

**Q5 — Screens. Answer this before §3 ships.** The detector already sees `screen`
and `laptop`. A 35B VL model pointed at this room **will read what is on those
screens**, incidentally, without being asked. The perception frontier doc called
this "the most dangerous idea here" and required an explicit spatial gate plus
sign-off before approaching it *deliberately*; §3 approaches it *accidentally*,
as a side effect of a better eye. This is the question most likely to be skipped.

**Q6 — Does Orion get to decline to look?** Should the multimodal call fire on
every person-triggered frame, or should Orion's own attention decide? The latter
is cheaper and more interesting, but needs a real gating signal — picking one on
vibes would be a cathedral.

**Q7 — Absence.** "I haven't seen you in three days" is care, and it's
structurally hard: this pipeline is event-triggered, and event-triggered
statistics freeze on silence rather than rising. Needs its own clock, not a
threshold on an EWMA. §5's `since_sec` is the right substrate for it.

**Q8 — Orion's own stated position.** The prior doc's best idea: constraints
Orion participated in authoring stand in a different relation to it than
configured ones. Enabling identity makes this *more* load-bearing, not less.

---

## 9. Reversal on the record

`docs/superpowers/specs/2026-08-12-perception-frontier-design.md` §Movement IV:
*"Identity, face, and re-ID profiles stay disabled."* And
`orion/schemas/situation.py:293-302` encodes it as a promise in code — identity
fields *"not to be added without proposal-mode sign-off… this schema is where
that promise is kept or broken."*

Juniper is the data subject and is asking for identification. This document is
that sign-off, recorded so a future reader finds a decision rather than a
contradiction. Both files get updated in the implementing patch.

**The prior doc's argument survives the reversal:** the deliverable is *care*,
not identification. Recognition is the mechanism; §5's presence duration is the
payload. A version that ships recognition and stops has built surveillance and
called it perception. That is why §5 is ordered before §6 and does not depend on
the VLM.

---

## 10. Order of work

1. **§3 spike — one frame, one gateway call to :8011, print the result.**
   Everything rests on whether a 35B VL model looking at this actual room
   produces something worth the rest of the document. One afternoon, near-zero
   code, and it is a *live-data* answer rather than an argument. **Q5 gets
   answered before this runs, not after.**
2. **§5 presence reducer** — independent of the VLM, highest care-per-line,
   reuses `hub_presence`'s exact shape.
3. **§3 council v2 + §3.1 profile deletions** — the big one, and it deletes more
   than it adds.
4. **§4 identity** + retention in the same patch.
5. **§6.3 outreach**, then **§6.4 falsification**.

## Non-goals

Activity label sets. Re-identification or tracking anyone but Juniper.
Persisting non-matching embeddings. Emotion/affect inference. Frame-level
percepts as durable memories or as `chat_history_log` turns. Widening the
detector vocabulary. Keeping BLIP.
