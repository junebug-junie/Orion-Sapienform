# AutonomyStateV2 reducer (operator notes)

## What this is

An optional, **env-gated** deterministic reducer that combines graph-loaded autonomy (`AutonomyStateV1`), turn-level evidence (user message, infra availability, reasoning fallback, social-bridge hazards), and optional action outcomes into **`AutonomyStateV2`** plus a **`AutonomyStateDeltaV1`** for one chat turn. Exported compact previews flow through cortex router metadata and Hub `extract_autonomy_payload` for observability.

## What this is **not**

- **Not** sentience, consciousness, or moral status.
- **Not** durable persistence: V2 exists in **request context only**; nothing is written back to the graph as V2.
- **Phi / spark / proxy telemetry** are **non-canonical**: they may appear as `proxy_telemetry` evidence and trigger inhibition and summary hazards; they must **not** be treated as ground-truth inner state.

## Causal loop (ASCII)

```
Graph (V1)                    Turn evidence (ctx)
    \                               /
     ---- upgrade V1 -> V2 ---------+
                  \
                   --> reduce_autonomy_state --> AutonomyStateV2 + Delta
                                  |
                    chat stance inputs["autonomy"]["state_v2"] / ["delta"]
                                  |
                    router metadata (preview + delta)
                                  |
                    Hub autonomy_payload whitelist
```

## Environment flag

Default **off**. Enable:

```bash
AUTONOMY_STATE_V2_REDUCER_ENABLED=true
```

When unset or not `true`, cortex skips the reducer entirely (no new `ctx` keys).

## Known limitations

1. **Polarity-blind text heuristics** — substring checks can fire on negated phrases (e.g. “no contradiction” still matches “contradiction”). Confidence caps mitigate but do not eliminate false signal.
2. **No durable state** — each turn rebuilds prior state from graph V1; delta `changed_fields` is relative to the **upgrade baseline at turn start**, not a stored prior-turn V2 snapshot.
3. **Mixed evidence** — user turns and infra health are excluded from **drive pressure** deltas but still occupy evidence slots and can influence confidence and summaries.
4. **`entity_id` / bindings** — subjects resolve via `SUBJECT_BINDINGS`; unknown subjects fall back to a safe default.
