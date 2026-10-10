# Dream carry-through: text → image → text → image → text → image

Status: approved 2026-10-10 (Juniper: "surprise me, do it" — recommended answers: replace the story, end on a picture, yield to the cabinet). Implementing.
Date: 2026-10-10
Follows: PR #2565 (every sleep ends in a story dream; live since 10-10 01:02 UTC)

## Arsonist summary

A sleep now ends in one story dream: one model call, one paragraph. Juniper wants the
dream to keep going, like a game of telephone between words and pictures:

```text
T0 dream text  →  I1 painted + seen  →  T2 dream text  →  I3 painted + seen  →  T4 dream text  →  I5 painted + seen
```

Each arrow is a **hop**. A text hop writes the next stretch of the dream and a short
picture prompt. An image hop paints that prompt and then *looks at the painting* (the
vision caption), and that caption is what the next text hop dreams from. The story
drifts because of what the pictures actually turned out to show, not what was asked for.

It runs as one **durable run** (`dream.carry`), checkpointed after every hop. A restart,
a hot cabinet, or a busy GPU pauses it, never loses it. A carry that cannot finish
still keeps every hop it made and says where it stopped.

## Current architecture (verified 2026-10-10)

- **Story dream (#2565):** a completed sleep publishes `dream.trigger` with a sleep
  digest. cortex-orch runs the `dream_cycle` verb (recall + one LLM call, `dream_cycle.j2`).
  The result `dream.result.v1` goes to the `dreams` table. Live: dream 21, 13 s after
  sleep `dc-1b31a0ce6658`.
- **Painting (`reverie.visual`, durable):** `prepare → resource_request → resource_wait →
  generate (diffusion hold) → caption → finish`, driven by orion-durable-runs and executed
  by orion-thought step handlers (`app/visual_steps.py`).
  - Each run is tied to a waking painting dispatch (`dispatch_id`). It is gated by the
    baseline schedule (one per 5400 s) and continues the waking chain (`prior_description`,
    slot rotation).
  - By design its brief carries **no text in**: thought picks the prompt itself.
- **Painting cost, live (3 days):** diffusion takes about 50 s per image (49–55 s).
  - 15 runs produced an image. 26 ended `failed / retry_window_expired`, after repeated
    `thermal_refused` deferrals (the cabinet was too hot).
  - Heat, not code, is the main reason paintings don't happen.
- **Seeing:** the caption comes from vision-host and is literal ("The image shows a porch
  with a white railing…").
- **Hub:** `GET /api/reverie/visual/image/{sha256}` already serves paintings by hash.

## Design

### As built (changes from the first draft)

- **Image hops are child `reverie.visual` runs.** No second copy of the painter's hold, heat
  retry, abandon and fence machinery. The carry submits one child per image hop
  (dispatch `dream-carry:<run>:<hop>`, inside durable-runs) and waits for its terminal
  detail without holding anything.
- **The dream flag is on the painting run, not the request.** `ReverieVisualRunBriefV1.dream_hop`
  and `ReverieVisualStepRequestV1.dream_hop` (`DreamHopImageV1`) carry it; `VisualRunRequestV1`
  is unchanged, so waking receipts and every reader of them are untouched.
- **No new table or migration.** `dreams.fragments` is stored as sent (sql-writer copies
  `DreamResultV1.fragments`), so each hop is a fragment. The Hub image route authorizes a dream
  picture's sha by finding it in a dream's fragments.
- **Text hops run under the run's own LLM hold** (`llm.route.metacog_background`); orion-dream
  passes it as `options.gpu_lease`, the same way journal.compose attaches to its hold.
- **Spacing comes free.** thought allows one open painting attempt at a time and a 600 s
  cooldown after each (`claim_visual_attempt`), so dream pictures interleave with waking ones
  and never stack up. That is the "yield to the cabinet" answer, enforced by existing code.

### The run: `dream.carry` (orion-durable-runs, `app/dream_carry_graph.py`)

```text
next_hop ─┬─ even hops done → resource_request → resource_wait → text_hop (LLM hold) → next_hop
          ├─ odd hops done  → image_submit (child reverie.visual) → image_wait (polls, holds nothing) → next_hop
          └─ all done / past deadline → finish_dream → finish | failed
   retry_wait: text retries (no attempt spent), replacement children, finish retries within grace
```

- **Hops:** `DreamCarryBriefV1.hops` (default 6, even, ends on an image).
- **Text hop:** an RPC to orion-dream under the run's `llm.route.metacog_background` hold, which is
  released as soon as the hop is checkpointed. Retries never spend attempts.
- **Image hop:** one child `reverie.visual` run per attempt, with dispatch
  `dream-carry:<run>:<hop>[:r<n>]` and `brief.dream_hop` set.
  - **Painting:** thought paints the prompt verbatim and captions it. It skips baseline,
    continuity, slot rotation and interpret, and writes no chain row, artifact row or receipt.
  - **Heat-type misses** (window expired, deferred thermal/busy/resource/unknown) get a
    replacement child, bounded by `DREAM_CARRY_CHILD_MAX_ATTEMPTS` (3) and
    `DREAM_CARRY_CHILD_MIN_WINDOW_SEC` (900).
- **Deadline:** 4 h (`DREAM_CARRY_DEADLINE_SEC`). Past it the carry finishes partial with
  `stopped_reason`; finish keeps retrying for `DREAM_CARRY_FINISH_GRACE_SEC` (1800).
- **Zero hops:** finish still runs, and orion-dream falls back to the one-paragraph story for that
  sleep.

### Contract changes (as built)

- `orion/schemas/dream_carry.py`: brief, hop, step request/result, run/dispatch id helpers,
  prompt clip (45 words).
- `orion/schemas/reverie_visual_run.py`: `DreamHopImageV1` on `ReverieVisualRunBriefV1.dream_hop`
  and `ReverieVisualStepRequestV1.dream_hop`; `ReverieVisualStepResultV1.caption`.
  `VisualRunRequestV1` is unchanged.
- `orion/schemas/durable_run.py`: `"dream.carry"` workflow and brief.
- Channels: `orion:dream:carry:step:request` and `orion:dream:carry:step:reply:*`. orion-dream is
  added as a producer of `orion:dream:log` and `orion:cortex:request`, and as a consumer of
  `orion:cortex:result*`.
- **No table or migration:** each hop is a fragment of the carry's `dreams` row (profile
  `dream.carry`).

### Trigger

- A completed, saved sleep submits `dream.carry` through cortex-orch's durable ingress when
  `DREAM_CARRY_ENABLED=true` (shipped on).
- If the submit fails, the sleep falls back to the one-paragraph story (#2565 path, kept intact).
- `POST /dreams/carry/run` starts a hand-started carry.

### Hub

The Dream tab's "Carried dreams" section shows each carry as passage → picture → "Orion saw:" → …,
with the sleep it came from and any early stop. `/api/dream/carry/image/{sha}` serves only
pictures a carried dream names.

## Missing questions (for Juniper)

1. **Replace the one-paragraph story, or run alongside it?** I recommend replacing it, since
   T0 is the story and two dreams per sleep double the noise. Alongside is a one-flag change.
2. **End on a picture or a sentence?** You specified six hops ending on an image, so the last
   caption closes the dream. I'd keep that. A seventh text hop would give it a written ending.
3. **Heat budget.** Each carry adds 3 paintings (~150 s of diffusion) on top of ~5 waking
   paintings a day. That is about +3–6 images a day on the GPU the cabinet already refuses
   most often. I recommend carries yield to the cabinet (retry with no rush, 4 h window,
   keep partials) rather than pushing priority.

## Non-goals

- Changing when Orion sleeps or what the sleep replays.
- Feeding carry images back into the next sleep's replay. That is the second half of the
  "visual reveries both ways" idea, and needs its own echo damping.
- Moving diffusion off the thermal gate or raising its priority.
- Video or animation between hops.

## Acceptance checks

- **Graph tests (durable-runs):**
  - A crash after `generate(3)` resumes at `see(3)` with no second diffusion call.
  - Thermal retries don't spend attempts.
  - The deadline finishes **partial**, with every hop made, never empty.
- **Text-hop tests (orion-dream):**
  - T2's prompt contains I1's caption and T0's passage.
  - An empty or refused reply retries.
  - `image_prompt` over 60 words is clipped.
- **Thought tests:**
  - `dream_hop` prepare uses the given prompt verbatim.
  - It never touches baseline or continuity state.
  - A waking run after a dream hop sees an unchanged `prior_description`.
- **Contract:** schema registry, channels, `DurableWorkflowV1`, and the definition-drift
  re-lock.
- **Eval:** a carry run end to end against fixtures (fake LLM + fake diffusion with real
  captions from 3 live paintings). It asserts that each text hop names something from the
  previous caption, so the drift is grounded in what was seen.
- **Live proof after deploy:**
  - A `dream.carry` row in `substrate_durable_run_state` with 6 hops.
  - Three images whose painting attempts record `stage_json.dream_hop`, with no waking chain rows.
  - A `dreams` row whose fragments list them.
  - The strip in Hub.
  - Waking painting count and continuity unchanged.

## Proposal-mode checklist

- **Capability change:** the dream after each sleep becomes a 6-hop word/picture chain
  instead of one paragraph.
- **Data touched:**
  - Dream painting attempts (`reverie_visual_attempt`) and image files; no new table.
  - One `dreams` row per carry.
  - Durable run state.
- **Privacy boundary:** hop text (derived from the sleep's material, which can include chat
  memories) travels in step requests and durable checkpoints, unlike the waking painting
  brief. No new reader outside orion-dream, orion-thought and durable-runs. No Juniper↔Orion
  boundary applies (standing decision).
- **Trace that proves it worked:** the live proof list above.
- **Dangerous failure modes:**
  - A carry hogging the diffusion GPU. Bounded by one hold per image hop, release after
    generate, the 4 h deadline, and yielding to the thermal gate.
  - Dream images leaking into waking continuity. Covered by a test.
- **Disable / roll back:** `DREAM_CARRY_ENABLED=false` returns to the one-paragraph story
  (#2565 path kept intact).

## Files likely to touch

- `orion/schemas/dream_carry.py` (new), `orion/schemas/durable_run.py`,
  `orion/schemas/reverie_visual.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`
- `services/orion-durable-runs/app/dream_carry_graph.py` (new), plus registration in
  `admission_runtime.py` / `main.py`
- `services/orion-dream/app/carry.py` (new: text hop, finish), `main.py`, `story.py`,
  `settings.py`, `.env_example`, `docker-compose.yml`
- `services/orion-thought/app/visual_steps.py`, `visual_chain.py` (`dream_hop` mode)
- `services/orion-hub/static/js/dream-tab.js`, `scripts/dream_routes.py`
- Tests in each service, plus `services/orion-dream/evals/test_dream_carry_eval.py`

## Recommended next patch

Answer the three questions, then build it in one branch, contract first. Deploy order:
migration → sql-writer → durable-runs → orion-thought → orion-dream → hub. Smoke it with
`POST /dreams/carry/run` once by hand before the first sleep uses it.
