# Dream carry-through: text → image → text → image → text → image

Status: proposal (cognition loop change; needs Juniper's yes before implementation)
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

### The run: `dream.carry` (orion-durable-runs)

```text
text_hop(0) → image_prepare(1) → resource_request → resource_wait → generate(1) → see(1)
  → text_hop(2) → image_prepare(3) → … → see(5) → finish
        ↑ any image stage: retry_wait (thermal / busy / transport) → same stage
```

- **Hop count:** `DREAM_CARRY_HOPS=6` (T, I, T, I, T, I). It ends on an image, so that
  image's caption, what Orion *saw*, is the dream's last word.
- **Checkpoints:** after every hop. A replayed hop returns its recorded result: no second
  LLM call, no second painting.
- **Diffusion hold:** taken per image hop and released right after `generate`, the same as
  `reverie.visual`. Text hops and `see` never carry it.
- **Retries:** heat, busy and transport problems retry with backoff and never spend the
  run's attempt budget (the same rule as `reverie.visual`).
- **Deadline:** `DREAM_CARRY_DEADLINE_SEC=14400` (4 h). When it passes, the run finishes
  **partial**, keeping every hop it made, with `stopped_reason` (e.g.
  `thermal_refused at hop 3`). It is never `failed` with nothing to show.

### Who does each hop

| Hop | Service | What it does |
|---|---|---|
| text | **orion-dream** (new step handler, `orion:dream:carry:step:request`) | One gateway call (background lane, like the sleep's own calls). Returns `{passage, image_prompt}`. |
| image prepare / generate / see | **orion-thought** (existing visual step handlers, new `mode="dream_hop"`) | Paint the given prompt, then caption it. |
| finish | **orion-dream** | Publishes one `dream.result.v1` so the carry lands in `dreams` like any other dream. |

orion-dream owns dream meaning; orion-thought owns pixels. Neither reads the other's tables.

### Text hop prompt (orion-dream)

- **T0** is today's story prompt, with the same sleep material (replay plus both
  hypothesis arms, shuffled; #2565). It returns a passage plus an image prompt.
- **T2 and T4** get:
  - the previous passage;
  - "the dream turned into a picture; looking at it you see: <caption>";
  - instructions to continue the dream from what was *seen*, letting the picture change
    the story.
- **Output contract:** JSON `{"passage": str, "image_prompt": str}`.
  - `image_prompt` is ≤ 60 words, because the diffusion model's CLIP encoder silently
    drops everything past 77 tokens (`visual_chain.select_context_slot`).
  - An empty, unparseable, or refused reply is a hop **retry**, never a blank hop. It
    reuses `llm.GatewayRefused` from #2549.

### Image hop (orion-thought, `mode="dream_hop"`)

- **prepare:**
  - Takes the hop's `image_prompt` verbatim.
  - Skips the baseline schedule, continuity, slot rotation and interpret.
  - Freezes the prompt on its own attempt row.
- **generate / see:** the same code as waking paintings: thermal gate, diffusion,
  content-addressed file, vision caption.
- **Isolation:** dream images never count toward the waking painting allowance, and never
  advance waking continuity (`prior_description`, rotation). Artifacts are stored with
  `source="dream"` and the carry's `run_id`.

### Contract changes

- `orion/schemas/durable_run.py`: `DurableWorkflowV1` adds `"dream.carry"` (additive
  Literal; consumer-first deploy).
- New `orion/schemas/dream_carry.py`:
  - `DreamCarryBriefV1` (trigger: sleep `cycle_id` + digest, or manual; hops; deadline).
  - `DreamCarryHopV1` (index, kind `text|image`, passage, image_prompt, sha256, caption,
    elapsed_sec, deferrals).
  - `DreamCarryStepRequestV1` / `ResultV1` (status `done|retry|terminal`, mirroring
    reverie.visual).
- `VisualRunRequestV1` gains optional `mode: "waking" | "dream_hop"` and
  `dream_prompt: str | None`. `dream_prompt` is required when `mode="dream_hop"` and
  forbidden otherwise.
- Channels: `orion:dream:carry:step:request` and the reply prefix, registered.
- Storage:
  - New table `dream_carry_hop` (orion-dream writes; migration file).
  - The final `dreams` row has `narrative` = the passages in order, and `fragments` = one
    entry per hop (`kind: text|image`, `sha256`, `caption`).
  - `metrics._dream_audit.trigger` keeps the sleep link.

### Trigger

- `story_trigger` (#2565) starts a `dream.carry` run instead of the one-shot
  `dream_cycle` verb when `DREAM_CARRY_ENABLED=true` (shipped **on**). T0 *is* the story,
  so it is still one dream per sleep, now carried through.
- `POST /dreams/carry/run` (orion-dream) starts one by hand, for trying it out.

### Hub

The Dream tab shows each carry as a strip: passage, picture (by sha), what Orion saw,
passage… and so on. Unfinished carries show where they stopped and why.

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
  - Three images with `source="dream"`.
  - A `dreams` row whose fragments list them.
  - The strip in Hub.
  - Waking painting count and continuity unchanged.

## Proposal-mode checklist

- **Capability change:** the dream after each sleep becomes a 6-hop word/picture chain
  instead of one paragraph.
- **Data touched:**
  - New `dream_carry_hop` rows and new dream-sourced image files.
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
- `services/orion-sql-db/manual_migration_dream_carry_hop.sql`
- `services/orion-hub/static/js/dream-tab.js`, `scripts/dream_routes.py`
- Tests in each service, plus `services/orion-dream/evals/test_dream_carry_eval.py`

## Recommended next patch

Answer the three questions, then build it in one branch, contract first. Deploy order:
migration → sql-writer → durable-runs → orion-thought → orion-dream → hub. Smoke it with
`POST /dreams/carry/run` once by hand before the first sleep uses it.
