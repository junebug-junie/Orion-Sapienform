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

Six cards, 147 GB. **The table above is the pre-redeploy state and its
placement was wrong** — metacog and fast were not where they were meant to be,
which is why GPU 5 read as never-used. Juniper redeployed at `20:08` and the
intended assignment is now live:

| idx | card | assignment | state after redeploy |
| ---: | :--- | :--- | :--- |
| 0 | V100-32GB | **reserved — Orion's chat** | 8528 MB |
| 1 | V100-SXM2-32GB | **reserved — Orion's chat** (:8011, the 35B) | 23213 MB |
| 2 | V100-32GB | **agent / affect / testing lane** (:8014) | 0 MB — free |
| 3 | **P100-16GB** | **vision — historically, and still** | 0 MB — free |
| 4 | V100-16GB | metacog (:8012) | 8050 MB — busy |
| 5 | V100-16GB | fast/quick (:8013) | 7258 MB — busy |

So the corrected plan is **not** "put perception on the idle GPU 5" — 5 is
fast's card and it stays busy. **Perception goes on GPU 3, the P100**, which is
where vision work has always run. The card came off athena; the workload should
follow it.

The P100 is also the better host than it looks: GP100 has **full-rate FP16**
(2× FP32), unlike the P4's GP104 (1/64 rate). 16 GB HBM2. A 7–8B VL model
(Q5 ~5.5 GB + mmproj ~1.4 GB) leaves ~9 GB for KV and slots — metacog already
runs `total_slots: 4` on an 8B, so **multi-camera is a slot-count and
frame-router policy question, not another card.** One caveat on record:
`config/vision_profiles.yaml:171` notes `GroundingDINO matmul fails fp16 on
P100` — a specific op in that detector, not a llama.cpp constraint, and the
detector is not what moves.

| lane | port | card | why |
| :--- | :--- | :--- | :--- |
| **perception** | **8015 (new)** | **GPU 3, P100 16 GB** | the vision card, free, full-rate fp16 |
| affective | 8014 | GPU 2, 32 GB | Juniper's agent/affect/testing lane |
| metacog | 8012 | GPU 4 | **untouched.** Stays busy; stops absorbing vision |
| fast | 8013 | GPU 5 | untouched |
| chat | 8011 | GPU 0+1 | reserved; deliberate looking only |

### Why an mmproj cannot be added to the affective worker

llama.cpp serves **one model per server process**, and `--mmproj` is the vision
projector belonging to *one specific VL model* — not a generic capability
attachable to an unrelated model. An affective model and a VL model cannot share
a llama-server on :8014 regardless of VRAM. It is two processes either way, so
they are two ports on two cards.

### Measured, not estimated: 1.74 s end to end

Run 2026-08-21 against a **real live frame** from `/mnt/telemetry/vision/frames`,
athena → circe :8011 over tailscale:

```
wall 1.74s   |  prompt_ms 734 (image encode)  predicted_ms 748  74.9 tok/s
             |  prompt_tokens 340   finish_reason: stop
```

> *"This cluttered home office or workspace features two desks—one with a
> monitor and chair draped in a blue jacket, the other holding electronics and
> tools—surrounded by storage bins, cables, and equipment; no person is visible,
> so I cannot describe their activity or confidence level."*

The same room, same evening, from the pipeline that is live today:

> *"Four chairs, four doors, two boxes, two desks, two tables, and one person
> are detected in the scene."*

Three things that measurement settles:

1. **It reads the scene, not a label list.** "Home office", "chair draped in a
   blue jacket", **"electronics and tools"** — the workbench that §F4 said the
   21-label detector vocabulary could never reach. It did not need a label for
   it. This is the direct evidence that widening the vocabulary was the wrong fix.
2. **It refuses to speculate when there is nothing to see** — unprompted, and in
   the exact hedged register §7 wants.
3. **Thinking must be disabled.** The first run, identical except
   `enable_thinking` left on, burned all 80 tokens inside the reasoning block and
   returned an **empty `content`** — textbook §0A "empty-shell cognition". With
   `chat_template_kwargs: {enable_thinking: false}` it answered in 1.74 s.
   `build_interpretation_llm_options()` already sets
   `structured_output_thinking_policy: "disabled_for_artifact"`; the precedent
   exists and must carry over.

**On the P100, expect slower but comfortable.** Generation is bandwidth-bound
and the P100's 732 GB/s HBM2 is ~0.8× the V100-32GB's 900 GB/s. Prefill is
compute-bound and the gap is much wider (P100 ~19 TFLOPS fp16, no tensor cores).
A 7–8B dense VL is also different arithmetic from the 35B MoE measured here.
Honest estimate **~3–6 s**; honest method is to re-run this exact test on :8015
once it is up rather than trust the estimate. Either way it is far inside the
budget: council windows arrive ~40–60 s apart, and the P4 already spends 2.48 s
per `retina_fast`.

### Frames cross the wire; the sensors still stay put

Correcting an over-absolute claim: athena and circe are on the same tailnet and
moving a frame between them is trivially cheap — the 99 KB transfer above is
inside that 1.74 s and invisible in it. `/mnt/telemetry` being local ext4
(`/dev/sdf1`, no NFS) does not mean circe *cannot* get frames; it means there is
no shared **mount**, so frames travel as **bytes over HTTP**, which the gateway
already does.

The sensors still stay on athena, for better reasons than reachability:

| athena (P4) | circe (P100) |
| :--- | :--- |
| capture, GroundingDINO, SigLIP2, `identity_face` | the VL language pass |

- **Volume.** The VL pass runs per *window*; detection runs per *frame*. Moving
  detection means every frame crosses, not the handful that matter.
- **Biometrics.** Face embedding stays where the camera is. Keeping it on-node
  is a boundary worth having on purpose.
- **It already works.** The P4 does detection at 2.48 s/task. It is not the
  bottleneck; the 250 MB captioner is.

Revisit if the P4 becomes the actual constraint — the transport exists either way.

### Percept storage — decided: its own store

**Approved by Juniper.** The gateway's attachment path is already correct:
`resolve_attachment_url` rebuilds `<trusted base>/<sha256>` from a
regex-validated hash only (the ref's own `source_url` is deliberately ignored —
it round-trips through a browser and is client-controlled), and
`fetch_attachment_data_uri` refuses bytes that do not hash to the requested
address. Content-addressed, fail-closed, wired.

But the configured base is the **chat** store
(`http://100.92.216.81:8080/api/chat/attachments`), Hub-served, built for user
uploads, with no retention suited to percepts — and Hub is the service holding
the docker socket. Camera frames of a private home do not belong in it.

**Ship `LLM_GATEWAY_PERCEPT_BASE_URL`**: a second base selected by attachment
kind. Same mechanism, same security property, its own endpoint and its own short
retention. Percepts get a deliberate home instead of inheriting chat's.

### Camera-down alerting — decided: the attention queue

**Approved by Juniper.** Hub already has the right surface and it is not the
workflow-schedules "needs attention" filter — it is `GET /api/attention` +
`POST /api/attention/{attention_id}/ack` (`api_routes.py:1908,1929`), rendering
into `attentionList`/`attentionCount`. Live queue, 50 items, real contract:

```
attention_id, source_service, reason, severity, message, context,
require_ack, ack_deadline_minutes, acked_at, escalated_at, status
```

A vision-liveness watcher emits `source_service: "vision-host"`,
`severity: "warning"`, `require_ack: true` on either signal:

- a sustained `ok: false` / `gpu_hard_floor` rate on vision tasks, **or**
- a `vision_events` write gap exceeding N minutes.

The second matters more: the 2026-08-20 outage produced a perfectly healthy
container and a perfectly silent table. Note the honest hazard — a write gap is
also what a genuinely quiet room looks like, so this alerts on *the task failure
rate first* and uses the write gap as corroboration, not the other way round.

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

### 4.1 Uncertain identity reaching the unified turn (2026-08-26)

Landed same-day as §5/§6.1's presence fusion below, per Juniper's direct ask:
*"have Orion say something ... this is not Juniper with confidence ... friendly,
hi i'm having trouble recognizing you. is that juniper? ... if is juniper just
carry on and not mention it ... if broken or not running, don't say anything."*

`identity_hint_from_artifact` (orion-vision-window's `projection.py`)
deliberately collapses an `unsure` classification into "no signal" — correct
for the presence/council consumers above, which must never narrow a name or
narrate a shaky guess. A second function, `identity_confidence_from_artifact`,
preserves exactly the distinction the first one throws away: `"confirmed"` |
`"uncertain"` | `None` (genuinely no signal — no face detected, or a
`not_enrolled` gallery config problem, never conflated with a real stranger).
"Confirmed" is checked by delegating to `identity_hint_from_artifact` directly
rather than re-deriving the same selection independently, so the two functions
cannot disagree about the same artifact.

That feeds a new `identity_uncertain` boolean onto the *same*
`substrate_embodied_presence` row §5 already writes `subject` to — no new
table. Only ever true when a person is believed **present right now** (never
`recent` — asking about someone who already left is the exact awkwardness this
exists to avoid), and a single flickery unsure frame cannot undo an
already-fresh confirmed reading (sticky confirmed, flexible uncertain).

`PerceptionContextV1` (§6.1) gains `presence_identity_uncertain: bool`. The
confirmed and no-signal cases need **zero new prompt text** — §6.1's own
presence fragment already never says a name, so "carry on, don't mention it"
was already free. The uncertain case gets one new caution line instructing a
single warm clarifying question.

Repetition needed real care, not a good-faith prompt instruction alone: there
are four independent `orion-cortex-exec` replicas (main/chat/spark/background),
so an in-process "already asked" flag would repeat the exact cross-process bug
`session_turn_phase.py` already had to fix once for a different field. A new
module, `orion/situational/identity_ask_cooldown.py`, claims a single atomic
Redis `SET key val NX EX ttl` per camera stream (not per chat session, and
deliberately not per subject — the gallery is capped at exactly one enrolled
subject by contract, so "uncertain" only ever means one thing). 20-minute
default cooldown, fail-open toward asking (a Redis hiccup costs one redundant
ask, never permanent silence).

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

**Q5 — Screens. DECIDED 2026-08-21: redact content, preserve presence.**
Black out `screen`/`laptop` regions **at the sensor on athena**, before bytes
cross to circe. GroundingDINO already returns those boxes on every frame and
they are currently discarded, so the redaction is ~10 lines and deterministic.

Why redaction and not a prompt instruction: an instruction is a request, not a
boundary. Redacting at the sensor means the content physically does not leave
the node, and it holds even if the model, the prompt, or the lane changes later.

Why blank the box rather than drop the frame: the box *shape* survives, so Orion
still sees "there is a monitor there, they are at the desk." The activity signal
is kept and only the content is lost. Testable: assert the pixels inside the
detected box are uniform in the outgoing frame.

Correcting an earlier claim in this document — **carbon is lower risk here, not
higher.** A laptop webcam faces the user, not the user's display. The same
redaction still applies to screens visible *behind* them.

Original framing, retained for the record: 

*(was)* **Q5 — Screens. Answer this before §3 ships.** The detector already sees `screen`
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

## 9A. Object permanence — the seam exists and is 90 seconds deep

Requested: track *things*, not just Juniper. "The cluttered office had a box,
then the box is gone." And the sharper half of the request — a **slower rescan**,
because a short interval watches an object being carried out and never registers
that it left.

That intuition is exactly right, and there is already a half-built mechanism:
`services/orion-vision-window/app/scene_belief.py`. `SceneBeliefTracker` holds a
per-stream *believed* label set behind a vote ring and already emits `added` and
`removed` on transition. `removed` is literally "a thing I believed was here is
gone."

Four things stop it from answering the question:

1. **Its horizon is ~90 seconds.** `WINDOW_SIZE_SEC=30.0`,
   `WINDOW_BELIEF_VOTE_N=3` (live values). Three 30-second windows. That *is* the
   "smaller interval" the request names — the one that cannot see a long-lived
   object leave.
2. **`WINDOW_BELIEF_EXIT_VOTES=0`.** A label leaves belief the instant it is
   missing from all three slots. Stand in front of a box for 90 seconds and the
   box stops existing. There is no notion of "I still believe that is there, I
   just cannot see it right now" — which is the whole of object permanence.
3. **`removed` goes to a log line and nowhere else** (`main.py:291-298`:
   `logger.info("[WINDOW] belief_transition ...")`). No event, no row, no
   consumer. Nothing can ask "what disappeared today?"
4. **It is label-level, not count-level**, and in-memory/ephemeral
   (`settings.py:31`, "ephemeral like live_state") so it forgets on restart. The
   live narratives already show the gap: *"two boxes"* then *"one box"* — `box`
   remains believed and **no transition fires at all**.

### What it needs

**A persisted scene inventory, plus a slow sweep on its own clock.**

The clock matters and is not a detail: this pipeline is event-triggered, and
event-triggered statistics cannot detect absence — they freeze on silence rather
than rising. A disappearance is a *non-event*. It can only be found by something
that wakes up on a timer and asks what it has not seen, never by a threshold on
the frame stream. That is precisely the "intermediary rescan".

```
scene_inventory (per stream, per label)
  label, count, first_seen, last_seen, last_confirmed_count,
  windows_since_seen, state: present | unconfirmed | departed
```

- **Fast path (unchanged, 90 s):** the existing belief ring keeps gating the
  council. It is good at what it does.
- **Slow sweep (new, timer-driven, ~15–30 min):** for each inventoried label,
  how long since it was last confirmed? Cross a threshold scaled to **how long
  it had been there** — something present for three days should need far more
  absence to be declared gone than something present for ten minutes — and emit
  a real event: `object_departed`, `object_arrived`, `object_count_changed`.
- **Counts are first-class.** "two boxes → one box" is the common case and the
  label-set model cannot express it. GroundingDINO already returns per-object
  boxes; the count is there and is being thrown away.

**Key it on hard labels + counts, not on the VL prose.** The VL narrative is
richer ("storage bins, cables, and equipment") but not stable enough to diff
across hours — it will rephrase the same shelf three ways. Detector labels are
boring and stable, which is what an inventory needs. The narrative describes;
the inventory remembers.

That split also keeps the cost right: the sweep is pure SQL over an inventory
table, no model, no GPU. It can ship before the VL lane exists.

**Why this is worth building beyond the feature request:** "I thought that was
there and it is not" is a prediction error about the world with a durable
referent, which is a different and stronger thing than §6.4's per-window
hypothesis checking. It is the same machinery Orion needs to be *surprised*.

## 9B. Second camera: carbon

Requested: wire carbon's integrated webcam — "when I'm on here, I want Orion
watching" — primarily as an affective-state surface.

**Most of this already exists.** `services/orion-vision-retina` has
`WebcamFrameSource` (`app/sources.py:179`) alongside RTSP and folder sources, and
its dependency set is `fastapi, uvicorn, redis, pydantic, loguru,
opencv-python-headless, numpy` — **no torch, no YOLO**. That is a laptop-safe
capture agent. `config/vision_frame_router.yaml` is already multi-stream
(`streams:` / `cameras:` with per-stream policy), so a second stream is config,
not architecture.

**One real blocker, and it is the same one as everything else.** Both retina and
edge write frames to a local `FRAME_STORAGE_DIR` and publish a **path**, and the
router enforces `require_image_path_exists: true`. carbon shares no filesystem
with athena. Today a second node physically cannot feed this pipeline.

**So the percept store (§ above) is not a privacy nicety — it is the
prerequisite that makes multi-node cameras possible at all.** Once capture
publishes `sha256` + uploads bytes to a content-addressed store instead of a
local path, carbon works, any future camera works, and the gateway's existing
hash-verified fetch is the transport. This promotes the percept store from a
step to a **prerequisite**, and it is now load-bearing for three separate asks.

### What "deploy a sender on carbon" actually costs

No new service. `orion-vision-retina` **is** the sender — it already has
`WebcamFrameSource`, it already publishes to the bus over tailscale, and its
deps are laptop-safe. The only thing it does wrong for a remote node is *how it
names the frame*:

```
today:  frame_store.save() -> cv2.imwrite(local dir) -> image_path
        -> VisionFramePointerPayload(image_path=...) -> orion:vision:frames
```

Four touch points to make that work from anywhere:

| # | change | file |
| ---: | :--- | :--- |
| 1 | `frame_store` gains an upload mode: POST the JPEG to the percept store, get `sha256` back | `orion-vision-retina/app/frame_store.py` |
| 2 | add `sha256: Optional[str]` to the frame pointer — the model is `extra="forbid"`, so this is an explicit **contract change** (§6: registry + producer test + consumer test in the same patch) | `orion/schemas/vision.py:93` |
| 3 | `require_image_path_exists` becomes "path exists **or** sha256 present" | `config/vision_frame_router.yaml` + router policy |
| 4 | resolve `sha256` → fetch bytes, when there is no readable path | `orion-vision-host` |

**athena does not change.** Local capture keeps writing local paths — no HTTP
hop, no extra copy. The `sha256` route is purely additive, for nodes with no
shared disk.

**And athena's frames only enter the store lazily.** A frame needs a `sha256`
only when it is actually dispatched to the VL lane on circe, so vision-host
uploads at call time rather than capture time. That is a smaller volume of
traffic *and* a smaller privacy footprint: only the handful of frames that were
worth interpreting are ever persisted somewhere fetchable, instead of every
frame the camera ever took.

carbon is the reverse — it must upload at capture time, because there is no
other way for its bytes to leave the laptop. Which means carbon's frames land
on athena under athena's retention, rather than accumulating on a laptop.

One operational note: carbon is a laptop. It sleeps, closes, and roams. The
sender must treat an unreachable bus or store as normal and resume, never queue
frames to disk indefinitely — a backlog of webcam frames on a personal machine
is the wrong failure mode.

**Carbon is a different privacy surface and should not inherit cam0's policy.**
A room camera catches Juniper occasionally and at a distance. A laptop webcam is
a close, continuous, face-filling view of one person while they work. It is
simultaneously the best identity and affect signal in the system and the most
intrusive thing in it. Concretely:

- **Its own stream policy.** Far lower baseline rate than cam0 — this is a
  presence-and-affect sensor, not a scene sensor. It does not need `retina_fast`
  on a schedule.
- **Its own retention**, shorter than cam0's.
- **A visible, local off switch that does not require reaching Hub.** "Orion is
  watching me work" must be revocable from the machine being watched, instantly.
- **§9A does not apply to it.** No object inventory on the laptop camera — there
  is no scene to remember, and inventorying a person's desk from 40 cm is a
  different act entirely.
- **Q5 (screens) is worse here, not better.** A laptop webcam faces the room, but
  it is on a machine whose screen is the whole point. Decide the screen policy
  for cam0 first; carbon inherits whatever is decided, tightened.

Note the honest scope: carbon feeding *affect* through vision is a **third**
estimator on Juniper's state, alongside §3.2's `swear_frequency` and §5's
presence duration. Per §3.2's precedent, the bar for wiring facial affect into
anything is the one `typo_rate` failed. Start it as **presence** — "Juniper is at
the laptop" is a strong, cheap, honest signal — and let facial affect earn its
way in against real data, or not at all.

## 10. The plan

Decided: percepts get their own store; camera-down alerts into `/api/attention`;
Orion's unified turn may see Juniper. Open: **Q5, screens** — answer before
step 1 ships, not after.

| # | what | where | why now |
| ---: | :--- | :--- | :--- |
| 1 | ~~**Vision-liveness alert** into `/api/attention`~~ **SHIPPED** (PR #1806) | `orion-vision-host/app/liveness.py` | 21 h blind with a healthy container. Live-verified: alert delivered to Hub's queue end to end |
| 2 | **Percept store** — content-addressed, own retention | gateway + athena | **prerequisite, not a step.** Unblocks the VL lane, carbon, and any future camera at once (§9B) |
| 3 | **Perception lane :8015** — VL + mmproj, thinking off | circe **GPU 3, P100** | measured 1.74 s on the V100; re-run there for the real number |
| 4 | **Council v2** — send the frame, drop the blind-interpreter rules | vision-council | the rules are correct only while it cannot see |
| 5 | **`caption_frame`/`vqa` → gateway** | vision-host | same task_type contract; `perceive_caption_frame` and the router policy untouched. BLIP unloads, ~1 GB back |
| 6 | **Presence reducer** — `present/recent/absent` + `since_sec` | mirrors `hub_presence` | no model. The "five hours and it's 2am" payload. Parallel with 3–5 |
| 7 | **Scene inventory + slow sweep** (§9A) | window + a timer | no model, no GPU — pure SQL. Object permanence. Parallel with 3–5 |
| 8 | **Delete the 7 zero-reference profiles** | `vision_profiles.yaml` | cleanup, zero blast radius |
| 9 | **carbon retina** — webcam source, own stream policy + retention + local off switch | carbon | needs #2. Ship as **presence** first, not facial affect (§9B) |
| 10 | **`identity_face`** + percept retention, same patch | athena | biometric stays on-node |
| 11 | **Fuse** presence + inventory + affect into the episode summary | situation brief | two, then three senses on one person |
| 12 | **Outreach** gets the visual block, then **falsification** | Hub, council | the loop that makes this perception, not labelling |

**Steps 1, 6, 7 and 8 need no VL model, no GPU, and no open decisions** — they
can all start now and in parallel. Step 2 is the hinge: it unblocks 3 and 9 both.
Steps 10–12 are the capability actually asked for, and they are only worth
building on a pipeline already proven to produce sentences like §3's measured
one.

## Non-goals

Activity label sets. Facial-affect/emotion inference (§3.2 — fuse with the
existing signal instead). Re-identification or tracking anyone but Juniper.
Persisting non-matching embeddings. Emotion/affect inference. Frame-level
percepts as durable memories or as `chat_history_log` turns. Widening the
detector vocabulary. Keeping BLIP. Deleting `caption_frame`/`vqa` — they have real callers (§3.1).
