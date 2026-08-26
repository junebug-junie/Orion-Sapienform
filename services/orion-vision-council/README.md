# Orion Vision Council

Consumes visual window summaries from the bus, calls the LLM gateway for scene interpretation, enforces evidence grounding, and publishes structured vision events.

Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s), independent of `CouncilService`'s own bus connection.

## V2 pipeline

```
VisionWindowPayload → evidence_transition (host label delta) → VisionSceneInterpretationV1 → enforce_evidence_grounding → VisionEventPayload
```

1. **Intake** — `VisionWindowPayload` arrives on the council intake channel (or via RPC request).
2. **Evidence transition gate** — `evidence_transition.py` compares **believed** host labels (`summary.evidence.believed_hard_labels` when present) and person presence per stream. Council runs metacog only on `first_window`, `person_entered`, `person_exited`, `salient_labels_changed`, or `refresh_ttl` (fires every `COUNCIL_TRANSITION_REFRESH_SEC`, default `600`/10min — `0` disables it and means "never force"). Stable scenes otherwise skip LLM and publish nothing.

   **Incident, 2026-08-23/25:** this defaulted to `0` and a home-office scene's coarse `hard_labels` (chair/clothing/desk/door/person/table) legitimately never changed turn to turn, so the gate correctly and deterministically read `stable_scene` on every single window for 44+ continuous hours — `orion:vision:events` went completely silent with no error anywhere in the pipeline (vision-host kept running inference the whole time; council kept ticking the whole time). The gate was working exactly as coded; it just had no ceiling. `600` bounds the worst case to comfortably inside the 900s staleness cutoff `orion/situational/context.py` uses to decide "stale" vs "live", well below the ~5.5min average cadence of genuine scene changes.

   **Known, accepted gap:** `orion-thought` (reverie)'s own percept-freshness gate is tighter (`reverie_perception_max_age_sec=180s`) than this 600s ceiling, so on a genuinely static scene reverie still sees "no fresh percept" ~70% of the time (420 of every 600s) — a real improvement over ~100%-blind, but not "always fresh." Not lowered to 180s on purpose: that would force an LLM reconfirmation every 3 minutes even when nothing is happening, more often than genuine changes occur naturally. See `app/settings.py`'s comment on `COUNCIL_TRANSITION_REFRESH_SEC` for the full reasoning.
3. **Interpretation** — `build_interpretation_prompt` shapes the LLM prompt (includes `summary.evidence` rules); the response is parsed into `VisionSceneInterpretationV1` via strict validation, salvage/coercion, then legacy fallback.
4. **Grounding** — `CouncilService._finalize_interpretation()` calls `enforce_evidence_grounding()` on **both** intake and RPC paths:
   - Drop person/activity claims when `person` ∉ `summary.evidence.hard_labels`
   - Cap activity confidence when only captions support the claim
   - On parse failure + `host_person_hits > 0`: deterministic `person_presence` fallback (no YouTube hallucination)
5. **Projection** — `project_interpretation_to_events` maps grounded `event_candidates` to `VisionEventBundleItem` entries.
6. **Publish** — the event bundle is published on `orion:vision:events` (and returned on RPC reply when applicable).

Bus intake honors the transition gate; **RPC requests always run interpretation** (on-demand callers must not hang or get silent no-ops). Concurrent windows on the same stream coalesce via `interpret_in_flight` so atlas metacog is not double-called on the same transition. Host pipe only — edge is out of scope.

## Evidence grounding rules

| Condition | Action |
|-----------|--------|
| Narrative mentions person/activity; `person` not in `hard_labels` | Drop event candidate |
| Activity verb without hard `person` | Drop event candidate |
| Activity claim with hard `person` but caption-only support | Cap confidence at 0.4, tag `caption_inferred` |
| LLM parse fails; `host_person_hits > 0` | Emit `person_presence` fallback (`parse_mode=host_fallback`, tag `host_detect`) |

Choke point: `services/orion-vision-council/app/evidence_grounding.py`, wired via `_finalize_interpretation()` in `main.py`.

Evidence transition choke point: `services/orion-vision-council/app/evidence_transition.py`, wired in `_generate_interpretation()` / `_process_window()` in `main.py`.

## Debug endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness check (`{"ok": true}`) |
| `GET /debug/last-interpretation` | Most recent `VisionSceneInterpretationV1` (in-memory ring buffer) |
| `GET /debug/recent-interpretations?limit=10` | Last N interpretations (max 20); each record includes Council-local `parse_mode` and optional `salvage_warnings` |
| `POST /debug/foveal-probe?question=...` | Manual trigger for the Foveal tier — see below |

Interpretations are retained in an in-memory ring buffer (max 20 items) for local debugging only; they are not persisted. Debug endpoints are unauthenticated — restrict network exposure in production.

## Foveal probe

`docs/superpowers/specs/2026-08-12-perception-frontier-design.md`'s Foveal
tier: a manually-triggered, event-driven call for a real caption (or, with
`?question=`, a real VQA answer) on the current frame — distinct from the
always-on peripheral pipeline. Not on any automatic cadence yet
(surprise-driven foveation is P2, blocked on `want_embeddings`); this exists
to prove the lane works end to end.

```
POST /debug/foveal-probe
POST /debug/foveal-probe?question=is+the+door+open%3F
```

Three real hops (`app/foveal_probe.py`): read the newest local frame
(`FOVEAL_FRAMES_DIR`, read-only mount) → upload it to `orion-percept-store`
(`FOVEAL_PERCEPT_STORE_URL`) → ask `orion-llm-gateway`'s vision-capable
`FOVEAL_LLM_ROUTE` (default `chat`) route and return its real reply.

**2026-08-25 architecture change — no longer a dedicated vision-host RPC.**
This originally targeted a second, isolated `orion-vision-host` instance
(`CHANNEL_FOVEAL_HOST_REQUEST` → e.g. `orion:exec:request:VisionHostService:
circe-vl`, deliberately never the shared channel the frame-router's
continuous pipeline uses — a second host instance racing the shared channel
had already killed 2m13s of live `host_trigger` updates, PR #1859). Live-
tested end to end after real config landed for that path: it worked
mechanically, but every call returned an **empty** caption/answer, rejected
by `sanitize_caption`/`sanitize_answer` as too-short. Root cause:
`config/vision_profiles.yaml`'s `vlm_caption`/`vlm_vqa` profiles still carry
the unfilled placeholder `model_id: "REPLACE_ME/qwen2-vl_or_llava_next"`,
which falls back to `orion-vision-host`'s own `VISION_VLM_MODEL_ID`
default — itself another BLIP-family model (`blip2-opt-2.7b`), not a real
VLM. The "richer-than-BLIP" tier had nothing richer behind it.

Replaced with a call through `orion-llm-gateway` instead of standing up a
second vision-host: the gateway's `chat` route already serves a real
vision-capable model (`modalities.vision=true`, confirmed live via
`/props`), and `orion-llm-gateway/app/vision.py` already implements a
`kind="percept"` attachment path built specifically for camera frames.
`orion-vision-council` was already a registered producer on
`orion:exec:request:LLMGatewayService` (`CHANNEL_LLM_REQUEST` above — used
for the council's own metacog interpretation calls), and the gateway's
`LLM_GATEWAY_PERCEPT_BASE_URL`/`LLM_GATEWAY_ATTACHMENT_ALLOWED_HOSTS` already
pointed at `orion-athena-percept-store`. So this needed zero new bus
contract, zero new channel, and zero new gateway-side config — only a
different envelope shape at the RPC hop. `CHANNEL_FOVEAL_HOST_REQUEST` and
`CHANNEL_FOVEAL_HOST_REPLY_PREFIX` are retired; the probe now shares
`CHANNEL_LLM_REQUEST`/`CHANNEL_LLM_REPLY_PREFIX` with the council's own
metacog calls.

No quality eval exists for this endpoint yet (unit tests cover the plumbing
— upload/RPC/error-path correctness — not caption/answer quality).

**Resolved, 2026-08-25:** the two live calls made while building this
endpoint returned real inference but an empty caption/answer, rejected by
`sanitize_caption`/`sanitize_answer` as too-short — BLIP-base's documented
quality ceiling (see the design doc's P1 rationale). circe's foveal host
now runs `Qwen/Qwen2-VL-2B-Instruct` instead (`services/orion-vision-host`'s
`app/vlm_family.py`/`app/model_manager.py`) — live-verified same day with
real, detailed, non-rejected output on both modes:

```
POST /debug/foveal-probe
"The image shows a room with a desk, chairs, a computer monitor, and a
door leading to another room. There is a blue towel hanging on the door..."

POST /debug/foveal-probe?question=What+is+in+this+image%3F
"This image appears to be a surveillance camera view of a room. The room
contains a desk with a laptop, a monitor, and some other office
equipment..."
```

## Tests

From repo root:

```bash
PYTHONPATH=services/orion-vision-council:. pytest services/orion-vision-council/tests -q
```

Key suites: `test_evidence_grounding.py`, `test_main_grounding_wiring.py` (intake/RPC parity via `_finalize_interpretation`).

Design spec: `docs/plans/vision/2026-07-02-vision-grounded-pipeline-design.md`
