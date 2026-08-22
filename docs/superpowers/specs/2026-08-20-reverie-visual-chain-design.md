# Reverie Visual Chain — design + Patch 1

Date: 2026-08-20
Status: **proposal + implementation** (Juniper asked for implementation directly, which §0A
allows; this document exists because the visual reverie chain touches memory/self-modeling/
cognition-loop territory and the proposal content is owed regardless of whether it blocks).

---

## 1. What this is for

The existing text reverie chain (`substrate_reverie_chain`/`substrate_reverie_thought`, owned by
`orion-thought`) is live and writing continuously — confirmed 2026-08-19, 12,003+ rows, latest
write seconds before check. It is narration-only: attention coalition → LLM interpretation →
Postgres row. Nothing renders it, nothing looks at it and re-describes what it sees.

This adds a second, parallel chain: on a slow, capacity-gated cadence, gather salient context
(recent activity, chats, dreams, and whatever else is salient at trigger time), generate an image
about that context via a diffusion model, run the image back through vision captioning to get a
description, and feed that description into the next reverie's context — a real generate →
observe → interpret loop, not narration alone.

## 2. Why not reuse the text chain's continuity mechanism

The text chain's `SpontaneousThoughtV1` schema carries `next_focus`/`drift` fields, commented as
"the LLM's forward pointer for the next step." **Nothing reads them.** `parse_reverie_payload()`
(`services/orion-thought/app/reverie.py`) never extracts them from the LLM's JSON response, and no
other code in `services/orion-thought/` or `orion/` reads them either (confirmed by repo-wide
grep, 2026-08-19). Every text-chain step independently re-reads the live attention-coalition
snapshot; the only things actually threaded across steps are an EMA salience float and an
accumulated `thought_ids` list. This is a live instance of the keyword-cathedral pattern §0A
warns about — a schema slot with stated producer intent and zero consumer.

The visual chain needs real step-to-step continuity (image N's description seeding image N+1's
prompt), so this gets built as an enforced column the context-builder actually reads
(`prior_description`), not a JSON field left for someone to wire up later.

## 3. Architecture

```
trigger (slow cadence, capacity-gated)
  -> orion-thought: visual_chain.py (new, alongside chain.py)
       -> gather context: recent activity / chats / dreams / prior_description
       -> orion-diffusion-host (new service): generate image
       -> orion-vision-host (existing service, new task_type): describe the generated image
       -> reverie_visual_chain / reverie_visual_artifact (new Postgres tables)
       -> prior_description persisted for the next trigger to read
```

Ownership and reuse, as decided:

- **Naming**: `reverie_visual_chain` / `reverie_visual_artifact` — no `substrate_` prefix. That
  namespace is scoped to the live attention-coalition rung; this chain pulls from broader context
  (chats, dreams), so it doesn't belong there.
- **Orchestration**: lives inside `orion-thought`, next to `chain.py` — not a new service. Same
  team, same DB ownership as the table it's named after.
- **Image generation**: new service, `orion-diffusion-host`, modeled on `orion-vision-host`'s shape
  (FastAPI + `diffusers`/torch on a raw CUDA base image, not llama.cpp — vision-host is the right
  analog since it's the one existing non-llama.cpp GPU-resident inference service in this repo).
- **Re-observation**: reuses `orion-vision-host` — a new `task_type` on its existing
  `orion:exec:request:VisionHostService` / `orion:vision:reply:*` channel pair, not a second vision
  worker. `orion-vision-host` already does BLIP2 captioning with real VRAM-aware scheduling
  (`VISION_DEVICE_STRATEGY`, per-GPU inflight caps, soft/hard VRAM floors) — existing-mechanism
  check passed, no reason to duplicate it.
- **GPU**: originally scoped as a new, 4th physical V100 32GB on Circe, decoupled from Patch 1.
  Superseded 2026-08-20 (per Juniper): instead of waiting on a new card, this service claims
  Circe's existing `orion-llamacpp-host` "agent lane" slot — port 8014, `CUDA_VISIBLE_DEVICES=2` —
  vacated by stopping the `atlas-agent` llama.cpp worker there. See
  `services/orion-diffusion-host/README.md` for the exact assignment and stop/start sequence.
  What that GPU will eventually share with (a "world model" component, still undefined — no such
  service exists in this repo today) is out of scope for this document and doesn't block anything
  below.

## 4. Trigger and capacity gate

Slower cadence than the text chain (which runs roughly every ~90s at metacog-lane-adjacent scale).
**No backlog, ever**: a trigger fires on its interval; if a visual-reverie run is currently in
flight, that trigger is dropped — not queued, not retried. The next trigger after the current run
finishes is what starts the next reverie. Single-flight, check-and-set on a `running_since`
marker: busy → no-op, free → start.

Consequence worth stating plainly: real cadence is `max(trigger_interval, actual_run_duration)`,
not the nominal interval — image generation + vision re-observation latency stretches the real
gap between reveries whenever a run outlasts one trigger period. This is intentional (per
Juniper: "once a trigger hits and there's capacity, it should kick off the next reverie"), not a
defect to fix later.

## 5. Schema

```sql
create table if not exists reverie_visual_chain (
    chain_id text primary key,
    created_at timestamptz not null,
    theme_key text,
    terminal_reason text not null,
    ema_salience double precision not null default 0.0,
    prior_description text,
    chain_json jsonb not null,
    stored_at timestamptz not null default now()
);
create index if not exists idx_reverie_visual_chain_created_at
    on reverie_visual_chain (created_at desc);

create table if not exists reverie_visual_artifact (
    sha256 text primary key,
    chain_id text not null references reverie_visual_chain(chain_id),
    step_index integer not null,
    mime text not null,
    bytes bigint not null,
    width integer,
    height integer,
    path text not null,
    description text,
    created_at timestamptz not null default now()
);
create index if not exists idx_reverie_visual_artifact_chain_id
    on reverie_visual_artifact (chain_id);
```

`prior_description` is the enforced continuity column (§2) — the context-builder for step N+1
reads it directly, not via `chain_json`.

## 6. Storage

Images land on `/mnt/storage-lukewarm` — confirmed live 2026-08-20 (`df -h`: ext4, 1.7TB, 1.5TB
free, `rw,noatime,discard`; write-tested successfully). Content-addressed by SHA-256, following
`services/orion-hub/scripts/chat_attachments.py`'s discipline (the one real precedent in this repo
for "generated binary + durable pointer," motivated by a real incident: per-event blobs into
Postgres already took the host down once, TOAST OOM crash loop, 2026-07-23):

- Path: `/mnt/storage-lukewarm/orion/reverie-visual/{sha256}.{ext}`
- MIME sniffed from magic bytes, never trusted from the generator's claimed type
- Write-then-atomic-rename (`tmp.replace(target)`) — no torn files under concurrent readers
- Image bytes never go in `chain_json`/`jsonb` — only the pointer row does

## 7. Privacy

Context feeding this chain includes recent chat content. A generated image is a lossy rendering
of that private material. §0A requires the same privacy boundary here as anywhere else: no
debug/UI surface may expose these images without the same gating private chat content already
requires. Not building any such surface in Patch 1 — flagging so it isn't skipped when one is
eventually built.

## 8. Non-goals (this document / Patch 1)

- Context-seeding logic (which specific recent-activity/chat/dream sources feed a step, and how)
  — Patch 3.
- Chain orchestration and the `prior_description` continuity wiring — Patch 2.
- A `node_catalog.yaml` entry tracking this specific GPU-lane reassignment (that file only tracks
  coarse per-node capability flags, not per-service device/port bindings — no existing convention
  in this repo to extend for that granularity) — and whatever "world model" turns out to mean —
  separate, later, undefined.
- Any UI/debug surface for viewing generated images.

## 9. Acceptance checks

- A visual-chain step writes: one file under `/mnt/storage-lukewarm/orion/reverie-visual/`, one
  `reverie_visual_artifact` row pointing at it by hash, a non-empty `description`, **and** the
  *next* step's outbound context demonstrably contains that description — same-run evidence, not
  schema presence. This directly guards against repeating the dead `next_focus`/`drift` pattern
  (§2).
- No orphaned files: every artifact row's `path` resolves to a real file; no file on disk lacks a
  row.
- Single-flight gate holds under a forced-overlap test: firing two triggers while one run is
  in-flight results in exactly one run, not two, and no dropped-trigger silently becomes a stuck
  chain (a dropped trigger must not prevent the *next* trigger from starting once free).

## 10. Patch 1 scope (this changeset)

- `reverie_visual_chain` / `reverie_visual_artifact` migrations
- `orion/schemas/reverie_visual.py` (Pydantic models) + `orion/schemas/registry.py` registration
- `orion-diffusion-host` service skeleton: README, `.env_example`, `docker-compose.yml`,
  `requirements.txt`, `settings.py`, `app/main.py` (health check only — no real generation wired
  yet), `Dockerfile`, `tests/`
- Storage helper module (content-addressed write, following `chat_attachments.py`'s pattern)
- No chain orchestration logic yet (Patch 2), no context-seeding (Patch 3)
