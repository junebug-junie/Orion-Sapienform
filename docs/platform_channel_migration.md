# Platform channel migration report

## Disambiguation rules

- **Channel / topic**: Redis PubSub topic used for publish/subscribe. Must start with `orion:` and use `:` segments (e.g., `orion:system:health`).
- **Envelope kind**: `BaseEnvelope.kind` describes the message type/version. It remains dotted/versioned (e.g., `system.health.v1`, `recall.query.v1`). Never derive kinds from env channels.
- **schema_id**: Payload schema identifier used for validation. It maps to a Pydantic payload model via the schema registry (`orion/schemas/registry.py`).

## Wildcard semantics

- Channel catalog entries may include a trailing `*` suffix only.
- Matching logic: exact match OR `channel.startswith(prefix_before_star)`.
- Any other wildcard placement is invalid.

## Rename map (topics only)

- **exec**: `orion-exec:*` → `orion:exec:*`
- **conversation**: `orion-conversation:*` → `orion:conversation:*`
- **cortex**: `orion-cortex:*` → `orion:cortex:*`
- **state**: `orion-state:*` → `orion:state:*`
- **spark**: `orion.spark.*` → `orion:spark:*`
- **vision**: `vision.*` → `orion:vision:*`
- **system**: `system.*` → `orion:system:*`
- **recall telemetry**: `recall.decision.v1` (topic) → `orion:recall:telemetry`
- **recall exec channels**:
  - `recall.query.v1` → `orion:exec:request:RecallService`
  - `recall.reply.v1` → `orion:exec:result:RecallService`

## Env vars removed

- None in this migration.
