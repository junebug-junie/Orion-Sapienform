# PR: Reasoning telemetry adapter (phase 1 of φ truthful corpus)

**Status:** IMPLEMENTED + reviewed, producer default-OFF. Makes Orion's reasoning
observable to φ. Ships behind `PUBLISH_REASONING_TELEMETRY=false`; off = zero
emissions, byte-identical current behavior. First of three specs; unblocks seed-v4.

## Summary

- **Contract** (`orion/schemas/telemetry/reasoning.py`): `ReasoningCallV1` (per-call
  metadata — booleans + token counts, **never trace text**) and `ReasoningActivityV1`
  (rolling-window aggregate). Registered in `_REGISTRY` + `SCHEMA_REGISTRY`; new
  channel `orion:cognition:reasoning_call`.
- **Producer** (cortex-exec): emits `ReasoningCallV1` at the end of `run_plan` when
  the flag is on, sourced from the reasoning diagnostics `_extract_final_text`
  already computes (`provider_has_reasoning_content` / `provider_reasoning_available`
  / `think_tags_detected` / `provider_completion_tokens`) — previously log-only.
  Double-guarded (flag + try/except); never breaks `run_plan`.
- **Assembler** (orion-thought): capped rolling-window consumer → `ReasoningActivityV1`,
  exposed at `GET /projections/reasoning_activity` for φ (spark-introspector) to poll,
  mirroring how it already polls substrate's `execution_trajectory`.

## Outcome moved

Reasoning was invisible to φ (`reasoning_present` True in 1/29,165 runs — detected
only on the harness FCC lane). This lights the wide-radius signal at cortex ingress
(where all metacog/reverie/journal calls resolve). Once enabled + accruing, φ can
finally read `reasoning_present` + completion-token throughput per window — the
input seed-v4 needs.

## Current architecture (before)

cortex-exec computed reasoning diagnostics + completion tokens per call at
`router.py` final-text assembly and **only logged them**. orion-thought is a thin
bus service. spark-introspector already HTTP-polls substrate projections.

## Architecture touched

- New bus channel `orion:cognition:reasoning_call` (cortex-exec → orion-thought).
- New HTTP projection `GET /projections/reasoning_activity` on orion-thought.
- No change to any existing channel/endpoint; all additive.

## Files changed

- `orion/schemas/telemetry/reasoning.py` (new) — both models (metadata-only).
- `orion/schemas/registry.py`, `orion/bus/channels.yaml` — register schema + channel.
- `services/orion-cortex-exec/app/reasoning_emit.py` (new) — pure build + publish, never-raise.
- `services/orion-cortex-exec/app/router.py` — one guarded emit at end of `run_plan`.
- `services/orion-cortex-exec/app/settings.py` + `.env_example` — flag + channel.
- `services/orion-thought/app/reasoning_activity.py` (new) — capped store, worker, decode.
- `services/orion-thought/app/main.py` — worker in lifespan + endpoint.
- `services/orion-thought/app/settings.py` + `.env_example` — window/cap/channel.
- Tests: `test_reasoning_emit.py` (15), `test_reasoning_activity.py` (11).

## Schema / bus / API changes

- Added: `ReasoningCallV1` (kind `cognition.reasoning_call.v1`), `ReasoningActivityV1`
  (kind `cognition.reasoning_activity.v1`); channel `orion:cognition:reasoning_call`.
- Reserved (unwired) fields, documented as such in the schema: `thinking_enabled`,
  `prompt_tokens`, `thinking_tokens` (+ derived `thinking_call_count`,
  `thinking_tokens_sum`) — no provider exposes a separate thinking-token count and
  `enable_thinking` isn't threaded into `run_plan` yet. A follow-on wire.
- Compatibility: fully additive; no existing shape changed.

## Env/config changes

- Added keys: `PUBLISH_REASONING_TELEMETRY=false`, `CHANNEL_REASONING_CALL`
  (cortex-exec); `CHANNEL_REASONING_CALL`, `REASONING_ACTIVITY_WINDOW_SEC=120`,
  `REASONING_ACTIVITY_MAX_CALLS=2000` (orion-thought).
- `.env_example` updated for both services. **Operator action:** run
  `python scripts/sync_local_env_from_example.py` in the main tree on merge — the
  worktree has no local `.env` so the sync was a no-op here. All keys default-safe
  (flag off), so nothing breaks unsynced.

## Tests run

```text
pytest services/orion-cortex-exec/tests/test_reasoning_emit.py -q        → 15 passed
pytest services/orion-cortex-exec/tests/test_router_final_text_assembly.py
       test_cognition_trace_metadata.py -q                                → 44 passed (incl. above; regression)
pytest services/orion-thought/tests/test_reasoning_activity.py -q         → 11 passed
pytest services/orion-thought/tests -q                                    → 127 passed (after merge, no regression)
```

## Merge with main (2026-07-09, post-review)

`origin/main` moved ~50 commits ahead while this branch was in review (through
PR #918). Merged `origin/main` into `feat/reasoning-telemetry-adapter`
(commit `a0fe2ddb`). One real conflict, in
`services/orion-thought/app/main.py` lifespan shutdown: main had added a
graceful `bus_task` drain (`asyncio.wait_for(..., timeout=125.0)` then cancel
if still running) and dropped `bus_task` from the plain cancel loop; this
branch had added `reasoning_stop_event.set()` + `reasoning_task` to that same
loop. Resolved by keeping both — main's graceful `bus_task` drain, plus
`reasoning_stop_event.set()` and `reasoning_task` folded into the remaining
`(reverie_task, reverie_chain_task, reasoning_task)` cancel loop. Everything
else (`.env_example`, `settings.py`, etc.) auto-merged clean.

Re-verified after merge:
```text
pytest services/orion-thought/tests -q                                    → 127 passed
pytest services/orion-cortex-exec/tests/test_reasoning_emit.py -q         → 15 passed
```
The 12 collection errors / ~58 failures seen when running the full
`orion-cortex-exec/tests` directory in one pytest invocation are a
pre-existing cross-service `app`-module double-import issue (verb registry
re-registration) — reproduced identically on `origin/main` pre-merge, not
introduced by this branch or the merge.

## Review findings fixed (fresh-eyes subagent)

- **#1 MINOR (kind drift)**: envelope `kind` was `orion.event`; set to
  `cognition.reasoning_call.v1` to match the declared contract. Evidence: runtime
  assert `REASONING_CALL_KIND == 'cognition.reasoning_call.v1'`.
- **#3 MINOR (latent silent-drop)**: `BaseEnvelope.correlation_id` is a UUID; a
  non-UUID id would fail validation → swallowed → no emission. Added local
  `_coerce_correlation_uuid` (UUID-form passthrough, else uuid5). Evidence: assert
  non-UUID → hashed UUID, not dropped.
- **#4 NIT (negative window 500s endpoint)**: clamped `window_sec = max(0.0, ...)`
  at store init (the ge=0.0 fallback path would have raised inside the handler).
- **#2/#6 (empty-shell edge)**: thinking_*/prompt_tokens can't be populated yet;
  marked explicitly **reserved/unwired** in the schema so the projection doesn't
  advertise a signal that can't fire.
- Confirmed clean: never-raise (all paths), flag-off byte-identical, capped buffer,
  metadata-only privacy, producer↔consumer round-trip, variable scope at emit site,
  aggregation math, thin-import boundary. Verdict: SHIP-WITH-FIXES → fixes applied.

## Docker/build/smoke checks

Not run (worktree, no live bus in this env). Contract validated by import +
round-trip; both service suites green. Live smoke deferred to enable-time.

## Restart required

Only when enabling the producer flag (default off = no behavior change):
```bash
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml up -d --build
```
orion-thought consumer is always-on (harmless when idle); rebuild it to pick up
the new worker + endpoint whenever convenient.

## Known limitations / follow-ons

- Emits only on `run_plan`'s terminal success path (early-return/error paths emit
  nothing) — acceptable for a v1 health readout.
- `thinking_*` fields reserved until `enable_thinking` is threaded into ctx.
- spark-introspector consumer (`fetch_reasoning_activity`) is **deferred to seed-v4**
  (spec 2) by design — building it here would leave a function nothing calls.

## Risks / concerns

- Low while flag off. When enabled, the only new work on the hot path is one guarded
  bus publish at run end.

## PR link

Branch pushed: `feat/reasoning-telemetry-adapter`.
Compare: https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/reasoning-telemetry-adapter
