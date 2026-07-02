# Vision grounded pipeline — edge triggers, selective VLM, council evidence — design

**Date:** 2026-07-02  
**Status:** Approved (brainstorming)  
**Scope:** `orion-vision-edge`, `orion-vision-frame-router`, `orion-vision-host`, `orion-vision-window`, `orion-vision-council`, shared schemas/channels/config

---

## 1. Purpose

Fix the split-brain vision path observed on Eye-Ball-1 (`stream_id=cam0`):

- **Edge YOLO** detected a person at walk-by time (~20:43 UTC) but results did not reach Window/Council (`EDGE_PUBLISH_ARTIFACTS=false`).
- **Host** ran `retina_fast` with `want_caption: true` on a fixed schedule, producing weak BLIP captions (`"describe this image. youtube"`).
- **Council** narrativized caption slop into fabricated activity (*"person watching YouTube"*) despite prompt rules against inventing facts.

**Goal:** Keep fast edge detectors, wire them into the downstream pipeline, run VLM caption **only when triggered**, improve caption quality, and **ground Council events on structured evidence** with deterministic fallbacks.

**Success criteria (all required):**

| ID | Criterion |
|----|-----------|
| A | Reliable **person-presence** events when someone walks through `cam0` (no identity) |
| B | **No activity hallucinations** unsupported by hard detection evidence |
| C | Presence + grounded narratives in one pipeline |
| D | **Richer description when someone is present** — selective VLM on trigger, not every baseline frame |

---

## 2. Chosen approach

**Approach B — trigger-gated host + grounded council** (not edge-only artifacts, not VLM-replaces-YOLO).

| Approach | Verdict |
|----------|---------|
| A — edge artifacts only | Rejected: does not stop always-on caption |
| B — trigger-gated + grounding | **Selected** |
| C — host-only VLM on motion | Rejected: loses sub-second edge person signal that already worked |

---

## 3. Architecture

```text
Eye-Ball-1 (stream_id=cam0, camera_id=RTSP SOURCE URL)
  │
  ├─ orion-vision-edge (YOLO / motion / optional face / presence)
  │     ├─► orion:vision:frames                    (unchanged)
  │     ├─► orion:vision:artifacts                 (slim edge_detection artifacts)
  │     └─► orion:vision:edge:activity             (NEW — compact trigger signals)
  │
  ├─ orion-vision-frame-router
  │     subscribes: frames + edge:activity
  │     baseline:  retina_fast { want_caption:false, want_embeddings:false }
  │     triggered:  retina_fast { want_caption:true,  want_embeddings:true  }
  │
  ├─ orion-vision-host (GroundingDINO + conditional VLM)
  ├─ orion-vision-window (evidence-aware summary)
  ├─ orion-vision-council (LLM + enforce_evidence_grounding)
  └─ orion-vision-scribe (unchanged contract)
```

---

## 4. Identifier conventions

**Do not change edge `camera_id`.** Today `camera_id` is the RTSP `SOURCE` URL; that is a valid stable identifier for this deployment.

| Field | Eye-Ball-1 | Role |
|-------|------------|------|
| `camera_id` | RTSP URL (`SOURCE`) | Router inflight/rate state key |
| `stream_id` | `cam0` | Window grouping, operator-facing stream key |

**Policy lookup change (frame router):** resolve camera policy by `camera_id` first, then fall back to `stream_id`. Per-stream overrides in YAML use **`streams.cam0`**, not the RTSP URL.

---

## 5. Item 1 — Keep fast edge detectors

**No YOLO removal.** Keep `DETECTORS=motion,yolo` (face/presence remain optional).

### 5.1 Slim edge artifacts

- Set `EDGE_PUBLISH_ARTIFACTS=true` (`.env_example` + local sync).
- Publish `VisionArtifactPayload` with:
  - `task_type=edge_detection`
  - `outputs.objects` only (boxes + labels + scores)
  - No caption, no embedding
- Same channel as host: `orion:vision:artifacts`

### 5.2 Edge activity signals (NEW)

New schema `VisionEdgeActivityPayload` on channel `orion:vision:edge:activity`:

```json
{
  "stream_id": "cam0",
  "camera_id": "rtsp://…",
  "labels": ["person", "motion"],
  "max_score": 0.82,
  "frame_ts": 1783025641.853,
  "image_path": "/mnt/telemetry/vision/frames/frame_….jpg",
  "artifact_id": "uuid"
}
```

- Envelope kind: `vision.edge.activity.v1`
- Rate limit: max **1 publish per second per (stream_id, label)** to avoid bus flood
- Emit when YOLO `person` or motion detector fires above configured thresholds

---

## 6. Item 5 + 2 — Wire edge in; selective VLM on host

### 6.1 Frame router

Subscribe to `orion:vision:edge:activity`. Maintain per-stream state:

- `last_trigger_ts` per label
- `active_labels` within TTL

**Two dispatch tiers** (`config/vision_frame_router.yaml`):

```yaml
defaults:
  baseline:
    task_type: retina_fast
    every_n_frames: 10
    min_seconds_between_tasks_per_camera: 5
    request:
      want_caption: false
      want_embeddings: false
  triggered:
    task_type: retina_fast
    trigger_labels: [person, motion]
    trigger_ttl_seconds: 8
    min_seconds_between_tasks_per_camera: 2
    request:
      want_caption: true
      want_embeddings: true

streams:
  cam0:
    enabled: true
```

Rules:

- **Baseline** keeps GroundingDINO fresh for static scene geometry (door, screen) without VLM every 5s.
- **Triggered** dispatch when `person` or `motion` appears in activity TTL; may use shorter rate limit.
- Triggered request **preempts** baseline sampling when inflight budget allows.
- Existing `cameras:` block preserved for backward compatibility; **`streams:`** is the preferred key for `stream_id` overrides.

Policy resolution order: `cameras[camera_id]` → `streams[stream_id]` → `defaults`.

### 6.2 Host

No pipeline rewrite. `pipeline_retina_fast` already gates caption/embed steps on `request.want_caption` / `request.want_embeddings`.

---

## 7. Item 3 — Caption quality

### 7.1 Model default

Upgrade default `VISION_VLM_MODEL_ID` from `Salesforce/blip-image-captioning-base` to `Salesforce/blip2-opt-2.7b` (env override remains; operator may choose a heavier VLM if VRAM allows).

### 7.2 Prompt

Replace bare `"Describe this image."` with:

> List visible objects and people. State only what is directly visible. No guesses about activity.

### 7.3 Caption sanitizer (host post-process)

After decode, reject caption if any:

- Echoes prompt prefix (`/describe this image/i`)
- Length `< 12` characters
- `> 40%` tokens match activity stoplist: `youtube`, `google`, `video`, `watching`, etc.

On reject: set `caption: null`, add warning in artifact/runner meta — **do not persist garbage text**.

### 7.4 Temperature

Set `VISION_VLM_TEMPERATURE=0.2` (from `0.4`) for factual caption mode.

---

## 8. Item 4 — Ground Council on structured evidence

### 8.1 Window summary extension

Extend `summarize_items()` in `orion-vision-window/app/projection.py`:

```python
"evidence": {
  "hard_labels": ["person", "door"],   # edge YOLO + host DINO, score >= threshold
  "soft_labels": [...],                 # caption-token derived only
  "edge_person_hits": 3,
  "host_person_hits": 1,
  "caption_count": 1,
}
```

Classification rules:

| Source | Tier |
|--------|------|
| Edge YOLO / host DINO object with score ≥ threshold | **hard** |
| Caption text tokens | **soft** only |
| Motion-only (no person box) | hard label `motion`, not `person` |

### 8.2 Council prompt (`build_interpretation_prompt`)

Add explicit rules to the existing prompt:

- `summary.evidence.hard_labels` are admissible for factual events.
- `summary.captions` are soft hints; never sole basis for activity claims.
- Activity verbs (watching, reading, using, talking) require `person` in `hard_labels`.
- If evidence is weak, populate `uncertainties` instead of inventing narrative.

### 8.3 Python choke point — `enforce_evidence_grounding()`

**File:** `services/orion-vision-council/app/evidence_grounding.py`  
**Call site:** `CouncilService._finalize_interpretation()` in `main.py` — after LLM parse/salvage, before `project_interpretation_to_events()` (intake and RPC paths).

| Rule | Action |
|------|--------|
| Event mentions person/activity; `person` ∉ hard_labels | Drop or downgrade to uncertainty |
| Activity claim without hard `person` | Remove or rewrite to observational language |
| Event confidence from caption-only | Cap at `0.4`, tag `caption_inferred` |
| Parse failure + `edge_person_hits > 0` | Deterministic fallback event (below) |

**Deterministic fallback** (walk-by case):

```json
{
  "event_type": "person_presence",
  "narrative": "A person was detected on camera.",
  "confidence": 0.85,
  "tags": ["edge_yolo"],
  "evidence_refs": ["<artifact_ids>"]
}
```

---

## 9. Components touched

| Component | Change |
|-----------|--------|
| `orion/schemas/vision.py` | Add `VisionEdgeActivityPayload` |
| `orion/schemas/registry.py` | Register schema + kind |
| `orion/bus/channels.yaml` | Register `orion:vision:edge:activity` |
| `orion-vision-edge` | Publish artifacts + activity; rate limits |
| `orion-vision-frame-router` | Activity subscription; baseline/triggered policy; `streams` lookup |
| `orion-vision-host` | Caption prompt, sanitizer, BLIP2 default |
| `orion-vision-window` | Evidence block in summary |
| `orion-vision-council` | Prompt rules + `enforce_evidence_grounding()` + tests |
| `config/vision_frame_router.yaml` | Baseline/triggered + `streams.cam0` |
| Service `.env_example` files | New channel keys, `EDGE_PUBLISH_ARTIFACTS`, sync script |

---

## 10. Non-goals

- Face identity / person naming
- Grammar projection / `orion:grammar:event`
- Vector persistence or memory cards
- Scribe SQL/RDF schema changes
- Hub UI changes
- Replacing GroundingDINO with VLM-only detection
- Changing edge `camera_id` away from RTSP URL

---

## 11. Testing & acceptance

### 11.1 Unit / fixture tests

| Test | Pass |
|------|------|
| Edge activity publish | `person` label on mock detection → envelope on `edge:activity` |
| Router policy | `streams.cam0` resolves when `camera_id` is RTSP URL |
| Router gating | Baseline → `want_caption=false`; trigger → `want_caption=true` |
| Caption sanitizer | `"describe this image. youtube"` → null caption + warning |
| Council grounding | Window: edge person + garbage caption → `person_presence`, no YouTube |
| Council grounding | Window: door/screen only → no person activity events |
| Regression | Existing council V2 interpretation tests pass |

### 11.2 Live acceptance (Eye-Ball-1)

Walk past camera → within one window cycle:

- SQL `vision_events` row with `person_presence` **or** grounded narrative citing hard labels
- **No** fabricated activity (YouTube, watching, etc.) unless `person` ∈ hard_labels **and** caption corroborates

---

## 12. Rollout order

1. Schemas + channel catalog + edge activity publisher (artifacts optional off)
2. Frame router activity subscription + policy (`streams.cam0`)
3. Window evidence summary
4. Host caption fixes
5. Council grounding + fallback
6. Enable `EDGE_PUBLISH_ARTIFACTS=true` on live edge; restart affected containers

---

## 13. Risks

| Risk | Mitigation |
|------|------------|
| BLIP2 VRAM vs llama-cpp coexistence | Env override to keep BLIP-base; caption only on trigger reduces load |
| Activity channel flood | Per-label rate limit |
| Router misses trigger if edge down | Baseline DINO still runs; graceful degradation |
| Over-aggressive grounding drops valid events | Tune thresholds; log dropped events at INFO |

---

## 14. References

- `docs/vision_services.md` — pipeline overview
- `docs/superpowers/specs/2026-05-02-orion-vision-window-projection-design.md` — window projection contract
- Live incident: Eye-Ball-1 walk-by 2026-07-02 ~20:43 UTC (edge YOLO person; council YouTube hallucination)
