# Active Verbs

Orion now supports a simple activation manifest at `orion/cognition/verbs/active.yaml` to reflect which verbs are actually runnable on each node.

## Manifest format

```yaml
default:
  allow: [ ... ]
  deny:  [ ... ]
nodes:
  athena:
    allow: [ ... ]
    deny:  [ ... ]
  atlas:
    allow: [ ... ]
    deny:  [ ... ]
```

## Rules

1. Verb discovery scans `orion/cognition/verbs/*.yaml` and excludes `active.yaml`.
2. If `allow` is empty/missing, all discovered verbs are allowed unless denied.
3. `deny` always blocks a verb.
4. Node rules overlay default rules (`allow` union + `deny` union).

## Hub behavior

- Hub exposes `GET /api/verbs?include_inactive=0|1`.
- The verb list in Hub shows active flags and supports an **Active only** toggle (default ON).
- Packs behavior is unchanged.
- Selecting exactly one verb is treated as an explicit verb override.

## Request enforcement behavior

- Brain mode with missing verb still defaults to `chat_general`.
- Agent/Council mode no longer gets gateway-overwritten to `chat_general` when verb is absent.
- If an explicit override requests an inactive verb, Hub/Orch returns a clear deterministic error.
- In Exec supervisor loops, if an agent/council chosen verb is inactive, execution fails deterministically (no silent `chat_general` fallback).
