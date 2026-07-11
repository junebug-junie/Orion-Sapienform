# Vision host pipe — decouple from edge

**Date:** 2026-07-03  
**Status:** Approved (brainstorming)  
**Supersedes (partially):** `docs/plans/vision/2026-07-02-vision-grounded-pipeline-design.md` — edge-as-control-plane sections only  
**Scope:** `orion-vision-frame-router`, `orion-vision-window`, `orion-vision-council`, config/docs; edge unchanged except README/env defaults

---

## Problem

The July 2026 grounded pipeline **coupled** the host vision pipe to edge in three places:

| Coupling | Effect |
|----------|--------|
| Frame router subscribes to `orion:vision:edge:activity` | Edge YOLO/motion gates baseline vs triggered (`want_caption`) |
| Window ingests `task_type=edge_detection` artifacts | Edge boxes populate `hard_labels` / `edge_person_hits` |
| Council fallback uses `edge_person_hits > 0` | Person presence requires edge authority; host DINO ignored for fallback |

The **May 2026 frame router spec** defined a decoupled bridge: `frames → router → host` with no edge dependency. That contract was broken.

Observed failure (2026-07-03 office visit): edge stopped detecting person; router stayed baseline; council could not name person despite host pipe running. User experience: "not working at all."

**Principle (non-negotiable):** Edge has its own purpose (local capture, YOLO, UI, security). The host pipe (`frames → router → host → window → council → scribe`) is **self-contained** and never gates on or requires edge.

---

## Chosen approach

**Approach 1 — Reply-fed host triggers**

- Remove router subscription to `orion:vision:edge:activity`.
- On each **host task reply**, extract detection labels from `result.artifact.outputs.objects` (GroundingDINO) and refresh per-stream trigger TTL (reuse existing `RouterState.record_activity` / `active_labels` machinery).
- Window ignores `edge_detection` artifacts for evidence.
- Council person fallback uses `host_person_hits` (and existing `person` ∈ `hard_labels` from host-only evidence).

Rejected alternatives:

| Approach | Verdict |
|----------|---------|
| Artifact-fed triggers (subscribe `orion:vision:artifacts`) | Rejected: extra hot-channel consumer, duplicates reply path |
| Flat policy (always caption on schedule) | Rejected: hallucination + GPU cost regression |

---

## Architecture

```text
PATH A — Edge (independent, unchanged behavior)
  capture → YOLO/motion
    ├─► orion:vision:frames          (frame pointers; router also consumes)
    ├─► orion:vision:edge:raw        (UI / local)
    ├─► orion:vision:edge:activity   (edge-local consumers only)
    └─► orion:vision:artifacts       (optional edge_detection; pipeline ignores)

PATH B — Host pipe (authoritative for Orion cognition)
  orion:vision:frames
    → orion-vision-frame-router
        (trigger TTL fed by host replies, NOT edge activity)
    → orion-vision-host  (retina_fast: embed + GroundingDINO + conditional VLM)
    → orion:vision:artifacts
    → orion-vision-window  (host evidence only)
    → orion-vision-council (host-grounded + enforce_evidence_grounding)
    → orion-vision-scribe
```

No service in Path B subscribes to `orion:vision:edge:activity` or treats edge artifacts as authoritative.

---

## §1 Frame router

### Remove

- `_activity_loop()` and its task in `main.py` lifespan
- `CHANNEL_EDGE_ACTIVITY_IN` from settings, `.env_example`, README (or mark deprecated unused)
- `app/activity.py` edge envelope handler usage from router (file may remain for tests until deleted)
- Import/subscription of `handle_activity_envelope` from edge path

### Add — host reply trigger feed

**Choke point:** `FrameDispatcher._handle_reply_envelope_inner` in `services/orion-vision-frame-router/app/dispatcher.py`

After validating `VisionTaskResultPayload` and clearing pending:

1. If `not result.ok` or `result.artifact` is None → return (no trigger update).
2. Parse `stream_id` from `result.artifact.inputs` (fallback: pending task meta / `camera_id`).
3. Extract labels from `result.artifact.outputs.objects` where `score >= TRIGGER_SCORE_THRESHOLD` (default `0.25`, match window `HARD_SCORE_THRESHOLD`).
4. Normalize labels: lowercase; map to configured `trigger_labels` intersection (default `[person]`).
5. Call `state.record_activity(stream_id, labels, now=time.time())`.

**Latency note:** First frame after presence is baseline (DINO only); once reply arrives with `person`, subsequent frames within TTL use triggered tier (`want_caption=True`). Acceptable trade for decoupling.

### Policy config (`config/vision_frame_router.yaml`)

- Keep `baseline` / `triggered` tier structure and `trigger_ttl_seconds: 8`.
- Change default `trigger_labels` from `[person, motion]` to **`[person]`** — host DINO emits object labels, not motion events.
- Rename in docs only if helpful; YAML key stays `trigger_labels`.

### Metrics / observability

- Log when reply updates trigger state: `[ROUTER] host_trigger stream=… labels=…`
- Optional metric: `host_trigger_updates_total`

---

## §2 Window

**Choke point:** `_build_evidence()` in `services/orion-vision-window/app/projection.py`

| Rule | Action |
|------|--------|
| `art.task_type == "edge_detection"` | Skip entirely for evidence (do not count objects, captions, or hits) |
| Host artifacts (`retina_fast`, etc.) | Unchanged — populate `hard_labels`, `host_person_hits` |
| `edge_person_hits` field | Keep in schema for compat; **always 0** (or omit increment path) |

Window continues subscribing to `orion:vision:artifacts`. Edge may still publish there; evidence builder filters them out.

---

## §3 Council

**Choke points:**

- `enforce_evidence_grounding()` — unchanged rules; `hard_labels` now host-only
- `_finalize_interpretation()` in `main.py` — change fallback gate
- `build_person_presence_fallback()` in `evidence_grounding.py` — retag

| Today | After |
|-------|-------|
| `edge_person_hits(window) > 0` | `host_person_hits(window) > 0` |
| `parse_mode="edge_fallback"` | `parse_mode="host_fallback"` |
| `tags=["edge_yolo"]` | `tags=["host_detect"]` |

Add helper `host_person_hits(window)` mirroring existing `edge_person_hits()` (or generalize to `person_hits(window, source="host")`).

**Keep** anti-hallucination grounding: person/activity claims still require `person` ∈ `hard_labels`. Authority source changes from edge YOLO to host DINO.

---

## §4 Edge (no required code changes)

Edge continues capture, detection, and publishing on edge-specific channels.

**Documentation only:**

- README: `orion:vision:edge:activity` is for edge-local consumers, **not** the host pipe.
- Optional default: `EDGE_PUBLISH_ARTIFACTS=false` in `.env_example` to reduce shared-channel noise (edge-local paths unaffected).

**Do not** remove `VisionEdgeActivityPayload`, edge activity publisher, or edge detection — other services may consume them later.

---

## §5 Bus catalog / docs

Update:

- `docs/vision_services.md` — remove router as consumer of `edge:activity`
- `services/orion-vision-frame-router/README.md` — host reply trigger diagram
- `services/orion-vision-council/README.md` — `host_fallback` / `host_detect`
- `orion/bus/channels.yaml` — frame-router consumer removed from `orion:vision:edge:activity` (edge remains producer)

---

## §6 Files likely touched

| Path | Change |
|------|--------|
| `services/orion-vision-frame-router/app/dispatcher.py` | Host reply → trigger state |
| `services/orion-vision-frame-router/app/main.py` | Remove activity loop |
| `services/orion-vision-frame-router/app/settings.py` | Remove `CHANNEL_EDGE_ACTIVITY_IN` |
| `services/orion-vision-frame-router/.env_example` | Same |
| `services/orion-vision-frame-router/tests/` | Reply trigger tests; remove/update activity tests |
| `services/orion-vision-window/app/projection.py` | Skip edge_detection in evidence |
| `services/orion-vision-window/tests/test_projection_evidence.py` | Host-only evidence cases |
| `services/orion-vision-council/app/evidence_grounding.py` | `host_person_hits`, fallback tags |
| `services/orion-vision-council/app/main.py` | Fallback gate |
| `services/orion-vision-council/tests/` | Update fixtures |
| `config/vision_frame_router.yaml` | `trigger_labels: [person]` |
| `docs/vision_services.md`, service READMEs | Decouple narrative |
| `orion/bus/channels.yaml` | Consumer list |

---

## §7 Testing & acceptance

| ID | Check |
|----|-------|
| A | Router has **no** subscription to `orion:vision:edge:activity` (code + runtime: `PUBSUB NUMSUB` shows 0 router consumers) |
| B | Host reply with `person` object (score ≥ 0.25) → router logs `host_trigger` → next dispatch `tier=triggered want_caption=True` within TTL |
| C | Window evidence for mixed host+edge artifacts: `hard_labels` reflects host only; `edge_person_hits==0` |
| D | Council emits `person_presence` on parse failure when `host_person_hits > 0`; tag `host_detect`; `parse_mode=host_fallback` |
| E | Council **drops** person narrative when host has no person in `hard_labels` (grounding unchanged) |
| F | Edge restart or edge detection gap does **not** change router tier behavior (host replies drive triggers) |
| G | Walk-through smoke: person in office → `vision_events` row mentioning person within 2 window cycles, with edge activity channel unused by router |

**Verification commands (post-impl):**

```bash
# Unit
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-vision-frame-router/tests \
  services/orion-vision-window/tests/test_projection_evidence.py \
  services/orion-vision-council/tests/test_evidence_grounding.py \
  services/orion-vision-council/tests/test_main_grounding_wiring.py -q

# Runtime: confirm router not on edge activity
redis-cli -u "$ORION_BUS_URL" PUBSUB NUMSUB orion:vision:edge:activity
```

---

## §8 Non-goals

- Edge detector tuning, camera FOV, YOLO confidence changes
- Scribe SQL/RDF schema changes
- Hub UI
- Removing edge activity schema or edge publish paths
- Replacing GroundingDINO with VLM-only detection
- Cross-service refactor beyond the three pipe services + config/docs

---

## §9 Migration / rollout

1. Deploy router + window + council together (partial deploy leaves inconsistent fallback authority).
2. Restart `orion-vision-frame-router` — confirm no activity loop in logs.
3. Edge can stay running unchanged.
4. Monitor: router `tier=triggered` should correlate with host DINO person boxes, not edge logs.

---

## Spec self-review (2026-07-03)

- [x] No TBD / placeholder sections
- [x] Architecture consistent with Approach 1 (reply-fed triggers only)
- [x] Scope limited to decouple; edge behavior preserved
- [x] Acceptance tests distinguish host authority from edge independence
- [x] Supersedes edge-control sections of July grounded design without removing anti-hallucination grounding
