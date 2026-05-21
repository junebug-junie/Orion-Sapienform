# Chat stance signal adapter contract

**Milestone B3** — maps cortex-exec stance artifacts to `OrionSignalV1` (`organ_id=chat_stance`, `signal_kind=chat_stance`).

## Bus channels (verified against `orion/bus/channels.yaml`)

| Channel | Kind | Source |
|---------|------|--------|
| `orion:cognition:trace` | `cognition.trace` | `orion-cortex-exec` PlanRunner |
| `orion:cortex:exec:request` | `cortex.exec.request` | Hub / gateway dispatch |
| `orion:chat:history:turn` | `chat.history.turn` | Hub `publish_chat_turn` |

Registry reference: `orion/signals/registry.py` → `chat_stance.bus_channels`.

## Payload field paths

### Primary: `ChatStanceBrief` (no PII in signals)

Extract from any of:

- `payload.chat_stance_brief` (exec ctx / step merge)
- `payload.ChatStanceBrief`
- `payload.metadata.chat_stance_brief`
- `payload.chat_stance_debug.final_prompt_contract.chat_stance_brief`

Schema: `orion/schemas/chat_stance.py` → `ChatStanceBrief`.

**Mapped dimensions (ordinal encodings, no brief text):**

| Dimension | Source field | Encoding |
|-----------|--------------|----------|
| `coherence` | `task_mode` | direct_response=0.85, triage=0.55, reflective/identity=0.7, mixed=0.5 |
| `valence` | `identity_salience` | low=0.35, medium=0.55, high=0.75 |
| `confidence` | parse success | 0.9 validated brief; 0.45 partial; 0.25 missing |

**Forbidden in `summary` / `dimensions`:** `stance_summary`, `user_intent`, `juniper_relevance`, snippet text.

### Secondary: cognition trace metadata

When only `metadata.chat_stance_debug_present` is true (no brief dict):

- Emit `signal_kind=chat_stance` with `confidence=0.4` and note `stance_debug_only_no_brief`.

## Causal parents

Registry order: `recall`, `autonomy`, `equilibrium`, `social_memory`, `spark_introspector` — resolve from gateway `prior_signals` window.

## Degradation

| Condition | Behavior |
|-----------|----------|
| Missing brief | `confidence≤0.25`, note `partial chat_stance payload` |
| Invalid brief | skip validation; use frame/task_mode strings if present |
| No `correlation_id` | synthetic `source_event_id` from verb+timestamp; note `synthetic_correlation_id` |

## Tests

- `orion/signals/adapters/tests/test_chat_stance_adapter.py`
- Fixture brief must not appear in `signal.summary` or dimension string values.
