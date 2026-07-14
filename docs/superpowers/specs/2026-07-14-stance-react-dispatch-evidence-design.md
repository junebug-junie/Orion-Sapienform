# Design spec: surface recent dispatch-action evidence into stance_react (pre-motor, unified-turn)

## Arsonist summary

`chat_recent_dispatch_actions` (real Layer-9 dispatch outcomes — proven live today: a real `inspect` action produced a real observation, a real `action_outcomes` row, and a real drive-pressure relief) is built once per turn in `chat_stance.py`'s shared context builder and rendered into `chat_general.j2`. But `ORION_UNIFIED_TURN_ENABLED=true` is live right now — **unified-turn, not `chat_general`/Brain mode, is the actual active chat path** — and `chat_recent_dispatch_actions` never reaches it. `chat_general` is itself on a sunset track (`docs/superpowers/checklists/2026-07-05-unified-turn-sunset.md`). Real self-directed-action evidence currently only surfaces in the path being phased out, not the live default one.

There is already a proven, live, end-to-end pipe carrying a conceptually adjacent payload — `AutonomySliceV1` — from this exact same context builder into the unified-turn's pre-motor stance prompt (`stance_react.j2`), which explicitly shapes `imperative`/`tone` before the FCC motor ever runs. This spec proposes reusing that pipe rather than building a parallel one.

## Current architecture

`build_chat_stance_inputs` (`services/orion-cortex-exec/app/chat_stance.py`) is a single shared context builder invoked for **both** the `chat_general` verb (Brain mode) and the `stance_react` verb (unified-turn's pre-motor cognition pass) — not two separate builders. Confirmed via the function's own inline comment at chat_stance.py:2588-2591, which states `autonomy_slice` is built there specifically "so it's present BEFORE the stance_react LLM step renders its prompt."

Two projections get built in this same function, back to back:

- `ctx["autonomy_slice"]` (chat_stance.py:2593, via `build_autonomy_slice(ctx)` in `services/orion-cortex-exec/app/autonomy_slice.py`) — a compact, bounded, fail-open projection of `AutonomyStateV2` reducer output (`dominant_drive`, up to 3 `active_tensions`, `pressure_trend`, `confidence`).
- `ctx["chat_recent_dispatch_actions"]` (chat_stance.py:2604, via `_project_recent_dispatch_actions(ctx)`) — up to `_MAX_RECENT_DISPATCH_ACTIONS = 3` real dispatch outcomes, newest-first by `observed_at`, queried directly via `load_action_outcomes(subject="orion")` (independent of `ctx`, fail-open to `[]`). Each entry is `{kind, summary, success, observed_at}` only — no internal correlation/audit fields.

Only `autonomy_slice` reaches unified-turn. Its full live path, confirmed by reading every hop:

1. `chat_stance.py:2593-2595` → `ctx["autonomy_slice"] = autonomy_slice.model_dump(mode="json")`
2. `services/orion-cortex-exec/app/router.py:1355-1371` — gated on `plan.verb_name == "stance_react"` and `step.step_name == "llm_stance_react"`, this block also assembles `grounding_capsule`. Separately, `router.py:1563-1564` copies `ctx["autonomy_slice"]` into that step's result `metadata["autonomy_slice"]`.
3. `services/orion-thought/app/bus_listener.py:_extract_autonomy_slice` (line ~212) reads `exec_result["metadata"]["autonomy_slice"]` and validates it as `AutonomySliceV1`.
4. `orion/thought/stance_react.py:parse_stance_react_payload` (line ~225-229) explicitly strips any LLM-authored `autonomy_slice`/`grounding_capsule` from the raw JSON first — these are documented as "assembled deterministically in cortex-exec and mapped on from result metadata — never authored by the stance LLM" — then the extracted (trusted) value from step 3 is attached onto the final `ThoughtEventV1.autonomy_slice`.
5. `orion/cognition/prompts/stance_react.j2` (lines 51-60) renders `autonomy_slice.dominant_drive` / `active_tensions` / `pressure_trend` directly into the stance LLM's own prompt, inside a `PRIOR SELF-SIGNAL (advisory)` block that explicitly instructs the model to "use it to color imperative/tone."
6. The resulting `ThoughtEventV1` — its `imperative`/`tone` already shaped by step 5 — is what `orion.hub.turn_orchestrator.execute_unified_turn` (line ~407) hands to `HarnessRunRequestV1(thought_event=thought, ...)`, which `HarnessGovernorClient(harness_bus).run(harness_req)` sends to the FCC motor.

Everything from step 1 through step 6 happens before the motor runs.

## Missing questions (resolved during this investigation, recorded for the record)

- **Does `stance_react.j2` see `autonomy_slice` at all today?** Yes, confirmed by reading the template directly (lines 51-60) — this is live, not aspirational.
- **Is unified-turn actually the active path, or a dark-shipped future one?** Confirmed live: `docker exec orion-athena-hub env` shows `ORION_UNIFIED_TURN_ENABLED=true` right now.
- **Does `chat_recent_dispatch_actions` depend on `chat_general`-specific ctx state, or is it verb-agnostic?** Verb-agnostic — `_project_recent_dispatch_actions` ignores `ctx` entirely and queries `load_action_outcomes` directly (see its own docstring), so it is already computed unconditionally regardless of which verb is running.

## Proposed schema / API changes

Add one bounded field to `AutonomySliceV1` (`orion/schemas/thought.py`):

```python
class AutonomySliceV1(BaseModel):
    """Compact, post-hoc-attached projection of Orion's own current
    drive/tension state (autonomy V2 reducer output) and recent
    self-directed action outcomes (Layer 9 dispatch), not authored by
    the LLM."""

    schema_version: Literal["autonomy.slice.v1"] = "autonomy.slice.v1"
    dominant_drive: str | None = None
    active_tensions: list[str] = Field(default_factory=list)
    pressure_trend: str | None = None
    confidence: float | None = None
    # New: compact, one-line-per-entry summaries of recent successful
    # Layer-9 dispatch outcomes. Same cap discipline as active_tensions
    # (at most _MAX_RECENT_DISPATCH_ACTIONS entries) -- not enforced at
    # this layer, callers are responsible for the bound, matching the
    # existing active_tensions comment's own convention.
    recent_actions: list[str] = Field(default_factory=list)
```

`schema_version` stays `"autonomy.slice.v1"` — this is an additive, optional field with a safe empty default, not a version bump. Existing producers/consumers that never populate `recent_actions` are unaffected; `model_validate` on old payloads works unchanged since the field defaults to `[]`.

No other schema changes. `ThoughtEventV1.autonomy_slice`, `HarnessRunRequestV1`, the bus contract for `orion:exec:request:LLMGatewayService`/stance_react's result metadata — none of these need touching. They already carry `AutonomySliceV1` as a typed sub-object; a new optional field flows through `model_dump`/`model_validate` on both ends automatically.

### Where the new field gets populated

In `chat_stance.py`, right after (or merged into) the existing autonomy_slice build block (~line 2593), format `ctx["chat_recent_dispatch_actions"]`'s `{kind, summary, success, observed_at}` dicts into compact one-line strings and pass them into the `AutonomySliceV1` construction. Concretely:

- Move/duplicate the `_project_recent_dispatch_actions(ctx)` call (or its result) so it's available before `build_autonomy_slice` returns, OR keep `build_autonomy_slice` reducer-only-scoped (matching its current single responsibility) and merge `recent_actions` into the slice as a small separate step in `chat_stance.py` right after both projections exist — mirroring the existing precedent at chat_stance.py:2562 ("drive_state — a sibling of `inputs["autonomy"]`, never merged into it... independently-computed signals"), except here the two things *should* merge into one slice object because they're both destined for the same downstream schema and the same prompt block.
- Format: `"{kind}: {summary}"`, truncated to a short character budget (match `ACTION_OUTCOME_SUMMARY_MAX_CHARS`-style discipline already used in the execution-dispatch-runtime emit path — reuse that constant's value or pick an equivalently small one, e.g. 160 chars, rather than inventing a new number without a home). Only include entries where `success is True` (failed/empty outcomes are not "what Orion did," they're noise for this advisory block — mirrors `extract_tensions_from_action_outcome`'s existing success-only convention from today's P3 work).
- Cap at 3 (reuse `_MAX_RECENT_DISPATCH_ACTIONS`, already imported/available in the same file — do not hardcode a second copy of `3`).
- Fail-open: empty `chat_recent_dispatch_actions` → empty `recent_actions` → the existing `if dominant_drive is None and not active_tensions and pressure_trend is None: return None` omit-check in `build_autonomy_slice` needs to also consider `recent_actions` in that condition (a turn with *only* recent actions and no drive/tension signal should still emit a slice, not be silently omitted).

### Where the new field gets rendered

`orion/cognition/prompts/stance_react.j2`, inside the existing `{% if autonomy_slice %}` block (lines 51-60), add one more conditional line following the exact same pattern as `active_tensions`:

```jinja2
{% if autonomy_slice.recent_actions %}- recent_actions: {{ autonomy_slice.recent_actions }}
{% endif %}
```

Also update the block's advisory framing comment if the "recent self-directed action" framing needs a one-line mention alongside "drive/tension state" (currently line 52 says only "Oríon's own current drive/tension state").

## Files likely to touch

- `orion/schemas/thought.py` — add `recent_actions` field + updated docstring on `AutonomySliceV1`.
- `services/orion-cortex-exec/app/autonomy_slice.py` — extend `build_autonomy_slice` (or add a small sibling merge step in chat_stance.py, per the design tradeoff above — implementer's call, document whichever is chosen) to populate `recent_actions`; update the omit-check condition.
- `services/orion-cortex-exec/app/chat_stance.py` — reorder/wire the merge (`_project_recent_dispatch_actions` result must be available where `AutonomySliceV1` gets constructed).
- `orion/cognition/prompts/stance_react.j2` — one new `{% if %}` line.
- Tests: `services/orion-cortex-exec/tests/` — extend whatever existing test file covers `build_autonomy_slice` (find via `test_autonomy_slice.py` or similar) with cases: recent_actions populated + capped at 3, success-only filter, omit-check considers recent_actions, empty-list fail-open. Also a schema round-trip test for `AutonomySliceV1.model_validate` with `recent_actions` present, proving the existing router.py/bus_listener.py plumbing needs zero changes (this is the acceptance check for "reuses the existing pipe" — the test's whole point is showing the field survives dict → JSON → dict → model without any of the intermediate hops needing awareness of it).

## Non-goals

- Not touching `chat_general.j2` — it already renders this evidence correctly; no change needed there.
- Not building a new parallel schema/plumbing path (a dedicated field on `ThoughtEventV1`, a new router.py map-on block, a new `bus_listener.py` extractor). `AutonomySliceV1`'s existing wiring already reaches the unified-turn path end-to-end; adding a second, structurally identical pipe for a 3-item list would be exactly the kind of ornamental layer AGENTS.md's "thin seams, not ornamental layers" rule warns against.
- Not touching `EXECUTION_DISPATCH_MODE` / `ORION_DISPATCH_MAX_PER_DAY` live settings — those are a separate, already-in-flight decision from today's earlier work, unrelated to this patch.
- Not addressing the earlier finding that `substrate.inspect`'s LLM call routes through the gateway's default `quick` lane rather than a dedicated background/metacog lane — a real observation from today's investigation, but a separate concern from this patch.
- Not touching P7 (`ORION_ENDOGENOUS_ORIGINATION_ENABLED`).
- Not adding a kill-switch/flag for this specific field — it inherits the exact same fail-open/omit-on-empty discipline `AutonomySliceV1` already has for its other fields, and the whole slice is already implicitly gated by `settings.orion_unified_grounding_enabled` upstream (the same flag gating `grounding_capsule` assembly) plus `AUTONOMY_STATE_V2_REDUCER_ENABLED` (which gates whether `autonomy_slice` gets built at all, per chat_stance.py:2571). A new dedicated flag would be config-surface bloat for an additive, fail-open, already-gated field.

## Acceptance checks

```bash
pytest services/orion-cortex-exec/tests/ -k "autonomy_slice" -q
pytest services/orion-cortex-exec/tests/ -k "chat_stance" -q
```

- `build_autonomy_slice` (or its sibling merge step) includes `recent_actions` when `chat_recent_dispatch_actions` has success=True entries, capped at 3, formatted as `"{kind}: {summary}"` truncated to the chosen char budget.
- Failed/empty (`success is False`) dispatch outcomes are excluded from `recent_actions`.
- The omit-check (`return None` when nothing meaningful) considers `recent_actions` — a turn with only recent-action signal (no drive/tension) still emits a slice.
- `AutonomySliceV1.model_validate(payload_with_recent_actions)` round-trips correctly — proves router.py/bus_listener.py need no changes.
- `stance_react.j2` renders `recent_actions` only when non-empty (Jinja conditional present and correctly gated).
- No regression in existing `active_tensions`/`dominant_drive`/`pressure_trend` behavior (full existing `autonomy_slice` test file still green).
- Live/manual verification (documented in the PR report, not necessarily automatable here): after deploying, trigger a real turn through the unified-turn path shortly after a real Layer-9 dispatch succeeds, and confirm `ThoughtEventV1.autonomy_slice.recent_actions` is non-empty for that turn — mirrors the exact live-verification method used earlier today for the P3 satisfaction-tension pipeline (temporary diagnostic log, live watch, then removed/cleaned up before commit).

## Recommended next patch

Implement via `/superpowers:subagent-driven-development`, worktree per repo convention (`scripts/new_worktree.sh fix stance-react-dispatch-evidence` or equivalent), single combined patch (schema + builder + template + tests) since the pieces are small and tightly coupled — not worth splitting into parallel tracks. Code review, README note if `services/orion-cortex-exec/README.md` documents `AutonomySliceV1`'s shape anywhere, commit, push, PR report.
