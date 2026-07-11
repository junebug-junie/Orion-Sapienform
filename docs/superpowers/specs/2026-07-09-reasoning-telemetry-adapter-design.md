# Reasoning telemetry adapter — cortex-exec → orion-thought → φ

**Mode:** Proposal (cognition-touching: gives φ a new perception — Orion's deliberation). No build until sign-off.

## Arsonist summary

φ is blind to reasoning (`reasoning_present` True in 1/29,165 runs) because reasoning-presence is only computed on the harness FCC lane. But **cortex-exec already computes reasoning diagnostics and completion tokens for every call** at `router.py:1389-1424` — `provider_has_reasoning_content`, `provider_has_reasoning_trace`, `provider_reasoning_available`, `think_tags_detected`, `provider_completion_tokens` — and **only logs them.** The thin seam: emit those as a per-call telemetry event, let orion-thought (the mesh's cognitive-assembly organ, and the correct home per Juniper) aggregate them into a rolling **reasoning-activity projection**, and let spark-introspector read that the same way it already reads substrate's `execution_trajectory`. One real observation point (cortex ingress, where all metacog/reverie/journal calls feed down), not eight scattered clients.

## Current architecture

- **cortex-exec** `router.py`: extracts `usage.completion_tokens`/`total_tokens` (`:189-198`); at final-text assembly (`:1389`) computes reasoning flags (`provider_has_reasoning_content`, `provider_has_reasoning_trace`, `provider_reasoning_available`, `inline_think_extracted`, `think_tags_detected`, `think_full_block_detected`) + `provider_completion_tokens` — **log-only today.** `enable_thinking` toggle at `pre_turn_appraisal.py:52`.
- **orion-thought**: thin bus service (`bus_listener.py`, `store.py`, `main.py`, `broadcast_reader.py`). Per [[project_orion_thought_thin_bus_no_substrate]] it must NOT import `orion.substrate.*`; it reads via bus + direct DSN. Good home for a bus-fed reducer + HTTP projection.
- **spark-introspector**: already HTTP-polls substrate `GET /projections/execution_trajectory` (`worker.py:583`, `substrate_reads.py:66`) and maps runs → cognitive features (`inner_state.py:140`). Symmetric consumer for a reasoning projection.

## Proposed schema / API changes

**New per-call event** `ReasoningCallV1` (metadata only — NO raw reasoning text; privacy-preserving):
```
correlation_id: str
turn_id: str | None
verb: str
mode: str                    # brain | reverie | metacog | agent | ...
node_id: str
reasoning_present: bool       # provider_has_reasoning_content OR reasoning_available OR think_tags_detected
thinking_enabled: bool        # from enable_thinking
completion_tokens: int | None
prompt_tokens: int | None
reasoning_trace_present: bool  # provider_has_reasoning_trace (bool only, never the trace)
emitted_at: datetime
```
- Channel: `orion:cognition:reasoning_call` (producer: cortex-exec). Register in `orion/bus/channels.yaml` + `orion/schemas/registry.py`.

**New windowed projection** `ReasoningActivityV1` (orion-thought reducer output, exposed via `GET /projections/reasoning_activity`):
```
generated_at: datetime
window_sec: float
call_count: int
reasoning_call_count: int             # reasoning_present True
thinking_call_count: int              # thinking_enabled True
reasoning_present_rate: float         # reasoning_call_count / call_count
completion_tokens_sum: int
completion_tokens_p50: float
by_mode: dict[str, int]               # capped small
```
- Producer: orion-thought new module `app/reasoning_activity.py` (bus consumer + rolling window store, cap N calls) + endpoint in `main.py`. Mirrors substrate's projection endpoint shape (`{"ok": true, "projection": {...}}`).

**Consumer:** spark-introspector adds `fetch_reasoning_activity()` in `substrate_reads.py` (own base URL to orion-thought) + wires into inner_state cognitive features (see seed-v4 spec).

## Files likely to touch

- `services/orion-cortex-exec/app/router.py` — emit `ReasoningCallV1` at final-text assembly (flag-gated).
- `services/orion-cortex-exec/app/settings.py` + `.env_example` — `PUBLISH_REASONING_TELEMETRY=false` default.
- `services/orion-thought/app/reasoning_activity.py` (new), `bus_listener.py` (subscribe), `main.py` (endpoint), `settings.py` + `.env_example` (window/cap keys), `store.py` if persistence wanted.
- `orion/schemas/telemetry/reasoning.py` (new — both models), `orion/schemas/registry.py`, `orion/bus/channels.yaml`.
- Tests + evals per service.

## Missing questions (resolve during build)

1. **Do providers expose a separate *thinking* token count**, or only `completion_tokens`? If no separate count, `reasoning_load` (seed-v4) is derived from `reasoning_present_rate` × token throughput, not a pure thinking-token measure. Verify against the actual `usage`/provider_meta payloads before finalizing the reasoning_load formula.
2. **Window length** for the projection (align with φ tick cadence; substrate uses 120s active window — match it unless data says otherwise).
3. Cap size for the rolling call buffer (avoid the 29k-run bloat we just diagnosed — cap from day one).

## Data / privacy

- Emits **only metadata** (booleans, token counts, verb/mode). The reasoning trace text and thinking content are NEVER emitted — `reasoning_trace_present` is a bool. Preserves the privacy-boundary mandate.

## Trace that proves it worked

- `orion:cognition:reasoning_call` events observable on the bus with non-zero `reasoning_present` for brain-mode introspect/journal runs.
- `GET :.../projections/reasoning_activity` returns `reasoning_present_rate` strictly between 0 and 1 over a live window.
- spark-introspector inner_state rows show `reasoning_present` with real variance (not 0.003%).

## Dangerous failure mode

- Mis-mapping so `reasoning_present` is universally True → re-freezes at 1.0 (same failure, other rail). Mitigation: acceptance check asserts **partial** rate (0 < rate < 1) on real traffic.
- Emitting reasoning trace text by accident → privacy breach. Mitigation: schema has no text field; test asserts payload contains no trace string.

## Rollback / disable

- `PUBLISH_REASONING_TELEMETRY=false` (default) → cortex-exec emits nothing, current behavior. orion-thought reducer idle. spark-introspector fetch fails closed to the existing `.none`/0 path (fail-open like `fetch_execution_trajectory`).

## Acceptance checks

1. cortex-exec unit: given a payload with reasoning content + tokens, emits one `ReasoningCallV1` with correct flags; with flag off, emits nothing.
2. orion-thought unit: N calls → correct windowed aggregate; buffer capped.
3. Contract: `check_schema_registry` + `check_bus_channels` pass with the new schema/channel.
4. Live smoke: reasoning_present_rate ∈ (0,1) on real traffic; no trace text in any payload.
5. Code review subagent clean.

## Recommended next patch

Contract first (`reasoning.py` models + registry + channel), then cortex-exec producer, then orion-thought reducer+endpoint, then spark-introspector consumer — split by contract, each with its own test.
