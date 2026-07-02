# Orion Vision Council

Consumes visual window summaries from the bus, calls the LLM gateway for scene interpretation, and publishes structured vision events.

## V2 pipeline

```
VisionWindowPayload → VisionSceneInterpretationV1 → VisionEventPayload
```

1. **Intake** — `VisionWindowPayload` arrives on the council intake channel (or via RPC request).
2. **Interpretation** — `build_interpretation_prompt` shapes the LLM prompt; the response is parsed into `VisionSceneInterpretationV1` (with legacy flat event-list fallback when needed).
3. **Projection** — `project_interpretation_to_events` maps `event_candidates` to `VisionEventBundleItem` entries in a `VisionEventPayload`.
4. **Publish** — the event bundle is published on `orion:vision:events` (and returned on RPC reply when applicable).

## Debug endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness check (`{"ok": true}`) |
| `GET /debug/last-interpretation` | Most recent `VisionSceneInterpretationV1` (in-memory ring buffer) |
| `GET /debug/recent-interpretations?limit=10` | Last N interpretations (max 20) |

Interpretations are retained in an in-memory ring buffer (max 20 items) for local debugging only; they are not persisted. Debug endpoints are unauthenticated — restrict network exposure in production.
