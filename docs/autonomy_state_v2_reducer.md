# AutonomyStateV2 reducer (operator notes) — RETIRED 2026-07-16

**This reducer is no longer called from anywhere.** `chat_stance.py`'s call site
(`_run_autonomy_reducer`) was deleted outright, not flag-gated off —
`AUTONOMY_STATE_V2_REDUCER_ENABLED` no longer exists in any `.env_example` or settings file.
`DriveEngine`'s `drive_state` (with real `tension_kinds` now pulled through) is the sole live
drive/tension signal for chat stance and Mind, with no fallback. See
`orion/autonomy/drives_and_autonomy_retrospective.md` §10 for the full story. The rest of
this doc is kept as historical operator notes for the module, which still exists
(`orion/autonomy/reducer.py` and friends) and still has its own passing tests, but has zero
live callers.

## What this is (historical)

An optional, **env-gated** deterministic reducer that combines graph-loaded autonomy (`AutonomyStateV1`), turn-level **typed** evidence (user message, infra availability, reasoning quality when upstream artifacts exist, social hazards from stance locals), and optional action outcomes into **`AutonomyStateV2`** plus a **`AutonomyStateDeltaV1`** for one chat turn.

Pressure math uses the shared `signal_drive_map` via `chat_evidence_to_tension` (same family as endogenous `failure_to_tension`). Keyword substring matching is **removed**.

## What this is **not**

- **Not** sentience, consciousness, or moral status.
- **Not** durable persistence: V2 exists in **request context only**.
- Empty reasoning repositories do **not** emit `reasoning_quality` theater.

~~Previously: not an input to phi features, `build_self_state`, or homeostatic `DriveEngine`.~~
Retired 2026-07-16: `DriveEngine`'s `drive_state` now feeds chat stance and the
`orion-cortex-orch`-triggered Mind path directly, with no fallback to this reducer at all (see
`orion/autonomy/drives_and_autonomy_retrospective.md` §10). **Still not covered**:
`orion-thought`'s independent "light Mind" path
(`services/orion-thought/app/mind_enrichment.py`) builds its own `MindRunRequestV1` and
does not include this facet -- known gap, unrelated to this reducer's retirement.

## Evidence contract (omit-when-empty)

| Kind | Emit when | Moves pressures? |
|------|-----------|------------------|
| `user_turn` | non-empty user message | No |
| `infra_health` | availability ∈ {available, degraded, empty, unavailable} | No |
| `reasoning_quality` | upstream repo/artifacts non-empty **and** `fallback_recommended` | Yes (`chat_reasoning_quality`/`fallback`) |
| `relational_signal` | hazards on social/social_bridge locals | Yes only for mapped exact keys |

Mapped social dimensions (v1): `cooldown_active`, `duplicate_message`, `self_message_loop`.  
Unmapped hazards still carry typed `signal_kind` / `dimension` / `value`; `SignalDriveMap.match` is the sole pressure gate (miss → no tension).

Confidence values on evidence are **kind-literal constants (uncalibrated)** in v1.

## Environment flag — removed

`AUTONOMY_STATE_V2_REDUCER_ENABLED` no longer exists (removed from
`services/orion-cortex-exec/.env_example` 2026-07-16). The reducer is unreachable regardless
of any env value now that its call site is deleted.

## Debug keys (when flag on)

- `ctx["chat_autonomy_evidence_debug"]` — emitted/omitted kinds + reasons
- `ctx["chat_autonomy_tension_debug"]` — minted tensions
- `ctx["chat_autonomy_movement_debug"]` — pressures / dominant_drive before vs after

## Known limitations

1. **No durable state** — each turn rebuilds prior state from graph V1; delta is relative to the upgrade baseline at turn start.
2. **Sparse map** — unmapped hazards still emit typed evidence but do not move pressures (preferred over fake motion).
3. ~~**Dual pipelines** — chat reducer and endogenous tick both use `signal_drive_map` helpers but chat does not feed `DriveEngine`.~~ Resolved 2026-07-16: `autonomy_slice.py` and `mind_runtime.py` are repointed at `DriveEngine`'s `drive_state` exclusively, no fallback. This reducer's typed evidence-compiler pattern (`evidence_compiler.py`/`signal_tension.py`) is unused in production but kept in the module for potential reuse as a future async mapping layer feeding `DriveEngine` via the bus -- not built, not scheduled.
