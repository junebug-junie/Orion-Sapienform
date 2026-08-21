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

**Root cause — the config never drifted, the hardware moved.** `app/gpu.py`
computes `effective_free = free_mb - reserve_mb`, then refuses below
`hard_floor_mb`. The budget in force was `reserve=3500 / hard_floor=1400`.

Those are the **originals from the first vision-host commit** (`626c103ee`),
written when athena carried a **P100 16GB** — and correct there:

```
P100:  16384 - 3238 resident = 13146 free  -  3500 = 9646  >  1400   OK
```

The P100 later moved to **circe**. Athena was left with a **Tesla P4, 7.68 GB**.
Nobody re-derived the budget:

```
P4:     7680 - 3238 resident =  4191 free  -  3500 =  691  <  1400   refuse, forever
```

Arithmetically unsatisfiable the moment the models warm. The container stayed
`Up` and healthy and served nothing.

**Independently corroborated.** `orion_biometrics` shows circe's GPUs 3/4/5
first appearing at `2026-08-21T06:02:25`, GPU 3 being a `Tesla P100-PCIE-16GB`.
athena's vision died `2026-08-20 22:00` with a final straggler at `06:12`. The
card was pulled and reinstalled that morning; two independent sources agree to
the hour. See §3's inventory table.

**This budget is a function of the card, not of the service.** That is the
durable lesson, and it is why the gate below reads real hardware instead of
baking in a constant — a constant pinned to "the P4" would rot identically on
the next card move.

**Fixed:** `.env` synced to `.env_example` (`1200/1000/800`), container
recreated via `scripts/safe_docker_build.sh` from a worktree. Verified:

```
gpu_hard_floor since restart: 0
"ok": true, "device": "cuda:0", inference_s: 2.48
```

**Gated so it cannot recur:** `config/vision_profiles.yaml`'s block corrected +
`services/orion-vision-host/tests/test_vram_budget_matches_env_example.py`, two
tests. One pins the doc block to `.env_example`. The other **reads the real GPU
via `nvidia-smi`** and asserts the budget is satisfiable on the smallest card
actually present — so it follows the hardware rather than rotting on the next
swap. Mutation-tested against the real file: the P100-era `reserve_mb: 3500`
fails on this P4 host; `1200` passes.

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

> **The vision host is a sensor. It does not do language.** Language about
> images goes through `orion-llm-gateway` — but on the **small** lane, not the
> chat lane.

### Which lane does the labelling — and on which card

The council **already routes to `metacog`** (`COUNCIL_LLM_ROUTE=metacog`,
`services/orion-vision-council/.env:13` → :8012). Per-window interpretation must
not contend with Orion's conversation for the 35B on :8011 — that part is right.
But metacog is already carrying a lot, and it should not absorb vision either.

**There is no need to squeeze anything.** circe's real GPU inventory, read from
`orion_biometrics` (24h, 1,144–1,690 samples per card — not a snapshot):

| idx | card | total | peak used 24h | avg used | peak util | first seen |
| ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| 0 | V100-PCIE-32GB | 32768 | 8532 | 8509 | 83% | 2026-07-24 |
| 1 | V100-SXM2-32GB | 32768 | 23213 | 23183 | 82% | 2026-07-24 |
| 2 | V100-PCIE-32GB | 32768 | 21320 | **4440** | 93% | 2026-07-24 |
| 3 | **P100**-PCIE-16GB | 16384 | 7979 | 7979 | **100%** | **2026-08-21 06:02** |
| 4 | V100-PCIE-16GB | 16384 | 7340 | 7340 | **100%** | **2026-08-21 06:02** |
| 5 | V100-PCIE-16GB | 16384 | **0** | **0** | **5%** | **2026-08-21 06:02** |

Six cards, 147 GB. **GPU 5 has been literally 0 MB used across 1,143 samples
since it was installed.** GPU 2 is bursty — idle at any given glance but peaking
to 21 GB, so a single-sample read would have called it free and been wrong.

**This table also independently dates §1.1's root cause.** GPUs 3/4/5 first
appear at `2026-08-21T06:02:25`, and GPU 3 is the P100. athena's vision died
`2026-08-20 22:00` with a final straggler at `06:12`. The card was physically
pulled from athena and installed in circe that morning; the P100-era VRAM budget
became unsatisfiable on the P4 the moment it left. Two independent sources, same
hour.

### Why "add an mmproj to the affective worker" cannot work

llama.cpp serves **one model per server process**, and an `--mmproj` is the
vision projector belonging to *one specific VL model* — not a generic capability
that can be attached to an unrelated model. An affective model and a VL model
therefore cannot share a llama-server on :8014 no matter how much VRAM the card
has. It is two processes either way.

Once it is two processes, the only question is which cards they sit on — and
there is an untouched V100 for exactly this.

| lane | port | card | why |
| :--- | :--- | :--- | :--- |
| affective | 8014 | **GPU 2** (32 GB, avg 4.4 GB) | episodic — `JuniperAffectiveStateV1` ticks on a 900 s window, so it tolerates the bursty neighbour |
| **perception** | **8015 (new)** | **GPU 5** (16 GB, never used) | continuous — every window. Needs a card that is actually free |
| metacog | 8012 | GPU 0 | **untouched.** Stops absorbing vision work it was never sized for |
| chat | 8011 | GPU 1 | deliberate looking only; already `vision:true`, `video:true` |

A 7–8B VL model (Q5 weights ~5.5 GB + mmproj ~1.4 GB) leaves ~9 GB on GPU 5 for
KV and slots. **That headroom is the webcam answer:** metacog runs
`total_slots: 4` on an 8B today, so multi-camera becomes a slot-count and
frame-router policy question rather than a "does it fit" question. Adding
cameras does not require another card.

Worth noting the V100 is Volta (sm_70) with real fp16 tensor cores — a
materially better VLM host than either the P4 (Pascal, crippled fp16) or the
P100 that used to do this work.

**What dies on the P4:** BLIP-base unloads (~1 GB back). athena keeps
GroundingDINO (hard labels), SigLIP2 (embeddings), and gains `identity_face` —
biometric, must stay on-node. That sensor set is roughly all the P4 can carry,
and it does not need to carry more.

**Also worth checking on circe:** GPU 3 (the ex-athena P100) sits at 100% peak
utilisation with a flat 7,979 MB — the busiest card in the fleet, arrived today,
and nothing in this repo's config accounts for it.

### 3.1 The `vision_profiles.yaml` cathedral — with blast radius checked

I recommended deleting nine profiles without checking callers. Corrected — I
grepped every `task_type` across `*.py`/`*.yaml`/`*.json`, excluding the profile
file itself and docs:

| profile | live refs | verdict | why |
| :--- | ---: | :--- | :--- |
| `vlm_caption` / `caption_frame` | **11** | **KEEP the contract, swap the backend** | implemented in `runner.py:141,306`; a **cognition verb** (`orion/cognition/verbs/perceive_caption_frame.yaml`); in `orion/schemas/vision.py:58`'s task_type contract; in the frame-router's `ALLOWED_TASK_TYPES` (`app/policy.py:16`). Deleting it breaks `perceive_caption_frame`. |
| `vlm_vqa` / `vqa` | **19** | **KEEP the contract, swap the backend** | real code path `runner.py:683 _run_vlm_vqa`, referenced by `orion/vision/caption_echo.py` |
| `pose_estimation` | **0** | DELETE | |
| `action_recognition` | **0** | DELETE | a fixed-action classifier is the keyword cathedral in model form |
| `scene_graph` | **0** | DELETE | `VisionSceneInterpretationV1.relations` already exists |
| `depth_estimation` | **0** | DELETE | no consumer, no motivating question |
| `person_reid` | **0** | DELETE | tracking strangers is the surveillance version of this |
| `affect_signals` | **0** | DELETE — **but see §3.2** | there is a real affect system already, and it isn't a face classifier |
| `ocr_read` | **0** | DELETE | and see Q5 |
| **`identity_face`** | **0** | **IMPLEMENT** | the one thing that cannot leave the node |

**Corrected: delete 7 (all with literally zero references anywhere), keep 2,
implement 1.** The two survivors keep their `task_type` names so every caller
and the cognition verb keep working — what changes is that `runner.py` delegates
to the gateway instead of loading BLIP locally. Same contract, better backend,
zero blast radius, and BLIP still unloads.

### 3.2 Affective state — the synergy is real, and it is not a face classifier

`orion:substrate:juniper_affective_state` already exists:
`orion/schemas/affective_state.py` (`JuniperAffectiveStateV1`), produced by
`services/orion-cocreation-signals/app/producers/affective_state.py` scanning
Juniper's real Claude Code transcripts on a 900s tiling window.

This matters here for three reasons:

1. **It is already a signal about Juniper specifically** — the same subject
   vision is about to start observing. Presence duration (§5) and
   `swear_frequency` over the same window are **two senses on one person**, both
   already time-window keyed. That is cross-modal binding with a real second
   modality, not a hypothetical one. *"Three hours at the desk and the language
   is getting shorter"* is a far better care signal than either alone.
2. **It sets the privacy precedent to copy exactly.** Aggregate scalar only,
   never the underlying text, never which words were flagged. Vision should hold
   the same line: an episode summary, never the frame.
3. **It already killed a metric at this gate, and that is the answer on
   `affect_signals`.** `typo_rate` was computed, tested, and then *not wired* —
   it never reached a genuine rest state across 111 real sessions. Facial-affect
   inference from a ceiling camera is a far weaker instrument than typo rate and
   would fail the same check harder. So: **do not build facial affect. Fuse
   vision presence with the affect signal that already exists and already
   passed.** `affect_signals` stays deleted, and this is the reason, not squeamishness.

Also worth copying: `JuniperAffectiveStateV1`'s `cold_start` flag exists because
overlapping windows silently inflated a SUM by 34.7% on the first two rows ever
persisted. §5's presence reducer tiles windows the same way and will hit the
same trap; take the flag with it.

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

Juniper has said plainly they don't mind Orion's unified turn seeing them, so
there is no consent question left here — only a plumbing one.

Percepts still don't become `chat_history_log` **turns**, for a purely technical
reason: those rows feed sql-writer, vector-writer, vector-host,
concept-induction and memory-consolidation, which would embed and consolidate a
percept as if Juniper had typed it. Nothing about that is a privacy hedge; it's
that a percept isn't an utterance and the consumers can't tell.

The rail that exists is `PerceptionContextV1` → situation brief → the turn
(`ORION_SITUATION_PERCEPTION_ENABLED=true`, live). Today it carries one
900-second-fresh sentence. After §3–§5 it should carry an **episode summary**:

> *"Juniper (probably) has been at the desk about three hours, at the bench
> since around 14:20. Last seen 4 minutes ago. Language in chat has been
> shorter than their baseline for the last hour."*

— presence duration (§5) plus the latest situated observation plus the affect
signal (§3.2), which is exactly the fusion that makes it worth reading. Frame
noise never reaches it. `available=False` stays a real state meaning "I have not
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

1. **Give `metacog` eyes.** Load a VL model + `--mmproj` on :8012. The gateway's
   live `/props` probe means nothing else has to change to find out — the
   council starts sending frames the moment the worker reports `vision: true`,
   and degrades automatically if it doesn't. **Q5 gets answered before this
   runs, not after.**
2. **Point `caption_frame`/`vqa` at the gateway** instead of local BLIP. Same
   task_type contract, so `perceive_caption_frame` and the frame-router policy
   are untouched. BLIP unloads, ~1 GB back on the P4.
3. **§5 presence reducer** — independent of all of the above, highest
   care-per-line, reuses `hub_presence`'s exact shape, takes
   `JuniperAffectiveStateV1`'s `cold_start` flag with it.
4. **Delete the 7 zero-reference profiles.**
5. **§4 identity** + retention in the same patch.
6. **§3.2 fusion** into the episode summary, then **§6.3 outreach**, then
   **§6.4 falsification**.

Council v1's prompt rules come out in step 2, not before — they are correct for
as long as the interpreter is blind.

## Non-goals

Activity label sets. Facial-affect/emotion inference (§3.2 — fuse with the
existing signal instead). Re-identification or tracking anyone but Juniper.
Persisting non-matching embeddings. Emotion/affect inference. Frame-level
percepts as durable memories or as `chat_history_log` turns. Widening the
detector vocabulary. Keeping BLIP. Deleting `caption_frame`/`vqa` — they have real callers (§3.1).
