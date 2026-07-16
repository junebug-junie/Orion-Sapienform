# AutonomyStateV2 reducer (operator notes)

## What this is

An optional, **env-gated** deterministic reducer that combines graph-loaded autonomy (`AutonomyStateV1`), turn-level **typed** evidence (user message, infra availability, reasoning quality when upstream artifacts exist, social hazards from stance locals), and optional action outcomes into **`AutonomyStateV2`** plus a **`AutonomyStateDeltaV1`** for one chat turn.

Pressure math uses the shared `signal_drive_map` via `chat_evidence_to_tension` (same family as endogenous `failure_to_tension`). Keyword substring matching is **removed**.

## What this is **not**

- **Not** sentience, consciousness, or moral status.
- **Not** durable persistence: V2 exists in **request context only**.
- Empty reasoning repositories do **not** emit `reasoning_quality` theater.

~~Previously: not an input to phi features, `build_self_state`, or homeostatic `DriveEngine`.~~
Superseded 2026-07-16: `DriveEngine`'s `drive_state` now feeds chat stance and the
`orion-cortex-orch`-triggered Mind path directly (see
`orion/autonomy/drives_and_autonomy_retrospective.md` §8). This reducer's own pressure
output is being retired from those consumers in favor of it. **Not yet covered**:
`orion-thought`'s independent "light Mind" path
(`services/orion-thought/app/mind_enrichment.py`) builds its own `MindRunRequestV1` and
does not include this facet -- known gap, not a claim this doc makes.

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

## Environment flag

Default **off**. Enable only after the movement eval is green:

```bash
python orion/autonomy/evals/run_autonomy_v2_movement_eval.py
# exit 0 required before considering:
AUTONOMY_STATE_V2_REDUCER_ENABLED=true
```

When unset or not `true`, cortex skips the reducer entirely.

## Debug keys (when flag on)

- `ctx["chat_autonomy_evidence_debug"]` — emitted/omitted kinds + reasons
- `ctx["chat_autonomy_tension_debug"]` — minted tensions
- `ctx["chat_autonomy_movement_debug"]` — pressures / dominant_drive before vs after

## Known limitations

1. **No durable state** — each turn rebuilds prior state from graph V1; delta is relative to the upgrade baseline at turn start.
2. **Sparse map** — unmapped hazards still emit typed evidence but do not move pressures (preferred over fake motion).
3. ~~**Dual pipelines** — chat reducer and endogenous tick both use `signal_drive_map` helpers but chat does not feed `DriveEngine`.~~ Being resolved 2026-07-16: `autonomy_slice.py` and `mind_runtime.py` are being repointed at `DriveEngine`'s `drive_state`; this reducer's typed evidence-compiler pattern is kept as a mapping layer feeding `DriveEngine` asynchronously via the bus instead.
