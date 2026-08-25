# Orion Vision Council

Consumes visual window summaries from the bus, calls the LLM gateway for scene interpretation, enforces evidence grounding, and publishes structured vision events.

Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s), independent of `CouncilService`'s own bus connection.

## V2 pipeline

```
VisionWindowPayload → evidence_transition (host label delta) → VisionSceneInterpretationV1 → enforce_evidence_grounding → VisionEventPayload
```

1. **Intake** — `VisionWindowPayload` arrives on the council intake channel (or via RPC request).
2. **Evidence transition gate** — `evidence_transition.py` compares **believed** host labels (`summary.evidence.believed_hard_labels` when present) and person presence per stream. Council runs metacog only on `first_window`, `person_entered`, `person_exited`, `salient_labels_changed`, or `refresh_ttl` when explicitly enabled (`COUNCIL_TRANSITION_REFRESH_SEC > 0`). Stable scenes skip LLM and publish nothing.
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
tier: a manually-triggered, event-driven call to a dedicated VLM host for a
real caption (or, with `?question=`, a real VQA answer) on the current
frame — distinct from the always-on peripheral pipeline. Not on any
automatic cadence yet (surprise-driven foveation is P2, blocked on
`want_embeddings`); this exists to prove the lane works end to end.

```
POST /debug/foveal-probe
POST /debug/foveal-probe?question=is+the+door+open%3F
```

Three real hops (`app/foveal_probe.py`): read the newest local frame
(`FOVEAL_FRAMES_DIR`, read-only mount) → upload it to `orion-percept-store`
(`FOVEAL_PERCEPT_STORE_URL`) → RPC the foveal host on
`CHANNEL_FOVEAL_HOST_REQUEST` and return its real reply.

**`CHANNEL_FOVEAL_HOST_REQUEST` must be an ISOLATED channel, never the
shared `orion:exec:request:VisionHostService` the frame-router's continuous
pipeline uses.** A second vision-host instance subscribed to the bare shared
channel raced its fast local-path rejections against the real, slower
replies and silently killed 2m13s of live `host_trigger` updates on
2026-08-25 (PR #1859). `orion/bus/channels.yaml` registers
`orion:exec:request:VisionHostService:*` (wildcard) specifically so every
dedicated/foveal host gets its own suffixed channel (e.g. `...:circe-vl`)
without a new catalog entry each time — `scripts/check_single_consumer_channels.py`
resolves that glob against the live bus and checks each realized channel's
subscriber count, so a second consumer accidentally reusing a suffix still
gets caught.

No quality eval exists for this endpoint yet (unit tests cover the plumbing
— upload/RPC/error-path correctness — not caption/answer quality). Both live
calls made while building this returned real inference but an empty
caption/answer, rejected by `sanitize_caption`/`sanitize_answer` as
too-short — BLIP-base's documented quality ceiling (see the design doc's P1
rationale), the concrete case for swapping the foveal host's captioner for a
real VLM next.

## Tests

From repo root:

```bash
PYTHONPATH=services/orion-vision-council:. pytest services/orion-vision-council/tests -q
```

Key suites: `test_evidence_grounding.py`, `test_main_grounding_wiring.py` (intake/RPC parity via `_finalize_interpretation`).

Design spec: `docs/plans/vision/2026-07-02-vision-grounded-pipeline-design.md`
