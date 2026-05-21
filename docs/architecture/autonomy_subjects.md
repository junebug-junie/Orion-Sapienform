# Autonomy Subject Routing Contract

## Purpose

Autonomy goals, drives, and tensions are keyed by **subject**. Operators and Hub UI must understand which events materialize under which subject so compact cards and graph reads stay aligned with chat stance queries.

This document is the canonical routing contract for the three primary subjects: **orion**, **relationship**, and **juniper**.

## Routing table

| Subject | Entity ID | Model layer | Materialized when | Used for |
|---------|-----------|-------------|-------------------|----------|
| orion | `self:orion` | self-model | metacog ticks, assistant turns, telemetry | Orion self-model drives and goals |
| relationship | `relationship:orion\|juniper` | relationship-model | dyadic chat turns (Hub WS/HTTP, GPT log, social chat) | dyadic drives and goals for the orion↔juniper pair |
| juniper | `user:juniper` | user-model | explicit user-model events (future; not chat) | user-model layer; **not** chat materialization |

## Dyadic chat → relationship (by design)

Hub and GPT chat turns carry both participants but represent a **dyad**, not a solo user profile. Concept induction therefore routes dyadic chat to `relationship`, not `user:juniper`.

Source: `orion/spark/concept_induction/identity.py` — `resolve_subject_identity()`:

```73:78:orion/spark/concept_induction/identity.py
    # Dyadic chat turns (Hub WS/HTTP, GPT log, etc.) are relationship-scoped: drives/tensions must
    # land on `relationship` so Graph autonomy + Hub cards match what chat_stance queries.
    ch = (intake_channel or "").lower()
    if any(tok in ch for tok in ("chat:history", "chat:gpt", "chat:social")):
        if isinstance(payload.get("prompt"), str) and isinstance(payload.get("response"), str):
            return RELATIONSHIP_SUBJECT
```

**Implication:** an empty juniper slot on the Hub compact card is expected when only chat turns have run. Juniper goals appear only after explicit user-model materialization (see below).

## Graph read bindings

The autonomy repository maps subjects to entity IDs for SPARQL lookups:

| Subject key | Entity ID | Model layer |
|-------------|-----------|-------------|
| `orion` | `self:orion` | self-model |
| `relationship` | `relationship:orion\|juniper` | relationship-model |
| `juniper` | `user:juniper` | user-model |

Source: `orion/autonomy/repository.py` — `SUBJECT_BINDINGS`.

## What not to do

- **Do not** route dyadic chat turns to `juniper`. That breaks alignment between concept induction materialization and `chat_stance` graph queries.
- **Do not** treat an empty juniper row as a fetch failure when relationship goals exist from chat.

## Future: juniper user-model path (stretch)

If operators want the juniper subject populated independently of chat:

- Trigger on `source_kind in {user_profile, biometrics, journal_write}` with explicit `subject=juniper`.
- Keep chat on `relationship` (preserve dyadic scope).
- Gate behind env: `CONCEPT_MATERIALIZE_JUNIPER_USER_MODEL=false` (default off).

## Hub display modes

| Mode | Env | Behavior |
|------|-----|----------|
| two (recommended default) | `HUB_AUTONOMY_SUBJECT_DISPLAY=two` | Availability shows `orion + relationship (orion↔juniper)` — 2/2, not 2/3; no juniper debug row on compact card |
| three (legacy) | `HUB_AUTONOMY_SUBJECT_DISPLAY=three` | Juniper row shows `n/a — dyadic scope uses relationship` when empty |

Raw debug payloads may still include juniper for engineers regardless of display mode.

## Related files

| Area | Path |
|------|------|
| Subject resolution | `orion/spark/concept_induction/identity.py` |
| Graph subject bindings | `orion/autonomy/repository.py` |
| Chat stance goal fetch | `services/orion-cortex-exec/app/chat_stance.py` |
| Hub compact card | `services/orion-hub/static/js/app.js` |
| Goal materialization | `orion/spark/concept_induction/goals.py` |

## Spec reference

Autonomy Goals v2 design spec §8.1 (Phase 2 — Subject model clarity): `docs/superpowers/specs/2026-05-21-autonomy-goals-v2-design.md`.
