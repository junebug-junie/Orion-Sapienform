# AutonomyStateV2 closed-loop wiring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `AutonomyStateV2` non-amnesiac (persist turn-to-turn) and give it two real consumers on the orion-unified (`stance_react`) path only: the stance-decision LLM call itself, and every FCC/Claude harness system prompt. Relocate the never-enabled curiosity/attention-frame injection to the same two consumers. `chat_general`/Lane A is untouched.

**Architecture:** New Postgres-backed `AutonomyStateV2` store (subject-keyed) closes the reducer's own fold loop. A new `autonomy_slice.py` builder (mirrors the existing `grounding_capsule.py` exactly) produces a compact `AutonomySliceV1`. It reaches the stance-decision LLM call via `ctx`/Jinja render (mirrors `mind_coloring`'s prompt-injection path) **and** reaches `ThoughtEventV1` as a real field via the `metadata` → `orion-thought` post-hoc-attach path (mirrors `grounding_capsule`'s path exactly). `compile_harness_prefix()` renders the field into every harness turn.

**Design source:** `docs/superpowers/specs/2026-07-11-autonomy-v2-closed-loop-wiring-design.md` (gitignored, local-only — see that file for full architecture rationale; this plan is self-contained for implementation purposes)

**Builds on:** autonomy-v2-evidence-signal-tension work (PR #939/#941) — that shipped the reducer math; this plan makes its output persist and actually matter.

---

## Context for the implementer (read before starting)

Work in a clean branch/worktree from repo root `/mnt/scripts/Orion-Sapienform`.

Verified facts (confirmed live this session, not assumed):

- Reducer entry point: `services/orion-cortex-exec/app/chat_stance.py:_run_autonomy_reducer()` (~line 2179), called from `build_chat_stance_inputs()` when `AUTONOMY_STATE_V2_REDUCER_ENABLED=true`.
- `previous_state` currently comes only from `_load_autonomy_state`/`_project_autonomy_from_beliefs` (V1/graph baseline) — no store for V2's own output exists anywhere in the repo.
- **Two existing, proven patterns to mirror, not invent new ones:**
  - `grounding_capsule` pattern (`services/orion-cortex-exec/app/grounding_capsule.py` → `router.py:1358-1367,1557-1558` sets `ctx["grounding_capsule"]` and `metadata["grounding_capsule"]` → `orion/thought/stance_react.py:183` defensively pops any LLM-invented copy from raw JSON → `services/orion-thought/app/bus_listener.py:193-203` `_extract_grounding_capsule(exec_result)` + `bus_listener.py:278-280` `model_copy(update={"grounding_capsule": capsule})`). Deterministic data, never rendered into the prompt, attached to `ThoughtEventV1` post-hoc.
  - `mind_coloring` pattern (`stance_react.j2:22-35`): rendered **into** the prompt as advisory context ("does not add output keys"), colors the LLM's own imperative/tone synthesis, never becomes a `ThoughtEventV1` field.
  - `autonomy_slice` needs **both halves**: prompt-injection (so the stance decision itself reacts to it) + post-hoc structured attachment (so `compile_harness_prefix` gets a reliable value, not a hoped-for LLM echo).
- Prompt template lookup is generic via verb registry: `orion/cognition/verbs/stance_react.yaml` → `prompt_template: stance_react.j2`. No hardcoded render call site to touch — adding keys to `ctx` is sufficient, same mechanism already used for `mind_coloring`/`chat_attention_frame` elsewhere.
- `orion/substrate/attention_frame.py` (`attention_frame_enabled()` reads `ORION_CURIOSITY_FRAME_ENABLED`, default false) is already lane-agnostic — computed inside `build_chat_stance_inputs` regardless of verb, already lands in `ctx["chat_attention_frame"]` today. Relocating it to Lane B is template-only (`stance_react.j2` + `compile_harness_prefix`) — **no code changes to the detector**.
- Harness prompt injection point: `orion/harness/prefix.py:compile_harness_prefix()`, which currently calls `_format_stance_slice()` and `_format_grounding_self_block()`. New formatters go here, called the same way.
- **No** changes to `orion/bus/channels.yaml` or `orion/schemas/registry.py` — this is a schema field + a new Postgres table, not a new bus event.
- **No** goal-pipeline integration, **no** graph-drives unification, **no** `chat_general`/`chat_stance_brief.j2`/`chat_general.j2` changes — deliberately excluded from this patch.

### Test commands

```bash
pytest orion/autonomy/tests -q
pytest orion/schemas/tests -k thought -q
pytest services/orion-cortex-exec/tests -k autonomy_slice -q
pytest services/orion-cortex-exec/tests -k grounding_capsule -q  # non-regression
pytest services/orion-thought/tests -k stance_react -q
pytest orion/harness/tests -k prefix -q
```

Full gate before PR:

```bash
pytest orion/autonomy/tests -q
pytest orion/schemas/tests -q
pytest services/orion-cortex-exec/tests -q
pytest services/orion-thought/tests -q
pytest orion/harness/tests -q
```

---

## File structure

| File | Responsibility |
|------|----------------|
| `orion/autonomy/state_store.py` | New: Postgres read/write for latest `AutonomyStateV2` by subject |
| `orion/autonomy/tests/test_state_store.py` | Store round-trip tests (mocked asyncpg) |
| SQL migration (new, under `services/orion-sql-db/` per existing manual-migration convention) | `autonomy_state_v2` table DDL |
| `services/orion-cortex-exec/app/chat_stance.py` | `_run_autonomy_reducer()` reads/writes via the store |
| `orion/schemas/thought.py` | New `AutonomySliceV1` model + `ThoughtEventV1.autonomy_slice` field |
| `services/orion-cortex-exec/app/autonomy_slice.py` | New: `build_autonomy_slice(ctx)` — mirrors `grounding_capsule.py` |
| `services/orion-cortex-exec/app/router.py` | Set `ctx["autonomy_slice"]` + `metadata["autonomy_slice"]`, mirroring lines 1358-1367 / 1557-1558 |
| `orion/cognition/prompts/stance_react.j2` | New `{% if autonomy_slice %}` block + new `{% if chat_attention_frame %}` block |
| `orion/thought/stance_react.py` | Defensive `raw.pop("autonomy_slice", None)`, mirroring line 183 |
| `services/orion-thought/app/bus_listener.py` | `_extract_autonomy_slice(exec_result)` + `model_copy` attach, mirroring `_extract_grounding_capsule` |
| `orion/harness/prefix.py` | `_format_autonomy_slice()`, `_format_attention_frame()`, wired into `compile_harness_prefix()` |
| `.env_example` (whichever service owns the new DSN) | New env key if a dedicated connection is used |
| tests across the above | Round-trip, render, non-regression |

Build order: store → schema field → cortex-exec builder + wiring → template blocks → orion-thought extraction → harness formatters → attention-frame relocation → full gate.

---

### Task 1: AutonomyStateV2 Postgres store

**Files:**
- Create: `orion/autonomy/state_store.py`
- Create: `orion/autonomy/tests/test_state_store.py`
- Create: SQL migration for `autonomy_state_v2` table (columns: `subject text primary key`, `state jsonb`, `updated_at timestamptz`)

- [ ] **Step 1:** Decide the DSN home — check whether `RECALL_PG_DSN` or the existing action-outcomes shared Postgres connection is reusable before provisioning a new one. Reuse if the table can live there without schema collision risk; otherwise add a dedicated env key.
- [ ] **Step 2:** Write failing tests in `test_state_store.py` for `load_autonomy_state_v2(subject) -> AutonomyStateV2 | None` and `save_autonomy_state_v2(subject, state: AutonomyStateV2) -> None`, mocked asyncpg connection, asserting the round-trip shape.
- [ ] **Step 3:** Run tests, confirm `ModuleNotFoundError`.
- [ ] **Step 4:** Implement `state_store.py` (asyncpg, upsert-on-write, same connection-pool convention as `orion/autonomy/action_outcomes.py`'s SQL path).
- [ ] **Step 5:** Write the SQL migration, following `services/orion-sql-db/manual_migration_memory_consolidation_v1.sql`'s naming/structure convention.
- [ ] **Step 6:** Run tests, confirm PASS.
- [ ] **Step 7:** Commit.

---

### Task 2: Wire persistence into the reducer

**Files:**
- Modify: `services/orion-cortex-exec/app/chat_stance.py` (`_run_autonomy_reducer`, ~line 2179)

- [ ] **Step 1:** Before calling `reduce_autonomy_state`, attempt `load_autonomy_state_v2(subject)`; if present, this becomes `previous_state` for the reducer input **instead of** the V1/graph-loaded state (the V1 baseline stays the fallback for the first-ever turn for a subject, where no persisted V2 state exists yet).
- [ ] **Step 2:** After `reduce_autonomy_state` returns, call `save_autonomy_state_v2(subject, result.state)`. Fail-open: log and continue on write failure, never block the turn (matches the existing `autonomy_reducer_v2_failed` try/except pattern already wrapping this call).
- [ ] **Step 3:** Write a test asserting: two sequential calls to `_run_autonomy_reducer` with different evidence, same subject, produce a second call whose `previous_state` matches the first call's `result.state` (not the V1 baseline). This is the core "non-amnesiac" acceptance check.
- [ ] **Step 4:** Run `pytest services/orion-cortex-exec/tests -k autonomy_reducer -q`, confirm PASS.
- [ ] **Step 5:** Commit.

---

### Task 3: `AutonomySliceV1` schema

**Files:**
- Modify: `orion/schemas/thought.py`
- Modify/create: `orion/schemas/tests/test_thought.py`

- [ ] **Step 1:** Add:

```python
class AutonomySliceV1(BaseModel):
    schema_version: Literal["autonomy.slice.v1"] = "autonomy.slice.v1"
    dominant_drive: str | None = None
    active_tensions: list[str] = Field(default_factory=list)
    pressure_trend: str | None = None
    confidence: float | None = None
```

  Add `autonomy_slice: AutonomySliceV1 | None = None` to `ThoughtEventV1`, placed near `grounding_capsule` (same "attached post-hoc, not LLM-required" treatment).
- [ ] **Step 2:** Test: `ThoughtEventV1` still validates with `autonomy_slice` omitted (back-compat with existing fixtures/tests that construct `ThoughtEventV1` without it).
- [ ] **Step 3:** Run `pytest orion/schemas/tests -k thought -q`, confirm PASS.
- [ ] **Step 4:** Commit.

---

### Task 4: `autonomy_slice.py` builder + ctx/metadata wiring

**Files:**
- Create: `services/orion-cortex-exec/app/autonomy_slice.py`
- Modify: `services/orion-cortex-exec/app/router.py`

- [ ] **Step 1:** In `autonomy_slice.py`, mirror `grounding_capsule.py`'s shape exactly:

```python
def build_autonomy_slice(ctx: Dict[str, Any]) -> AutonomySliceV1 | None:
    """Assemble the compact slice from the V2 reducer output already in ctx.

    Returns None (omit, not empty) when the reducer produced no meaningful
    signal — never fabricates a dominant_drive or tension."""
```

  Read from `ctx["chat_autonomy_state_v2"]` / `ctx["chat_autonomy_state_delta"]` (already populated by `_run_autonomy_reducer` per the existing wiring at `chat_stance.py:2307-2308`). Omit-when-empty.
- [ ] **Step 2:** In `router.py`, mirror lines 1358-1367 / 1557-1558: after the autonomy reducer runs, set `ctx["autonomy_slice"] = slice.model_dump(mode="json")` if not None, and mirror it into `metadata["autonomy_slice"]` at the same point `grounding_capsule` is copied into metadata.
- [ ] **Step 3:** Test: given a `ctx` with populated `chat_autonomy_state_v2`, `build_autonomy_slice` returns a slice with the right dominant_drive/tensions; given empty/absent state, returns `None`.
- [ ] **Step 4:** Run tests, confirm PASS.
- [ ] **Step 5:** Commit.

---

### Task 5: `stance_react.j2` — autonomy_slice block

**Files:**
- Modify: `orion/cognition/prompts/stance_react.j2`

- [ ] **Step 1:** Add a block styled exactly like the existing `{% if mind_coloring %}` block (lines 22-35): named sub-fields, "advisory — reconcile, do not obey," explicit instruction on how it should bias `imperative`/`tone`/`response_priorities`, and the same "does not add output keys" guard.

```jinja
{% if autonomy_slice %}
PRIOR SELF-SIGNAL (advisory — Oríon's own current drive/tension state)
- This reflects Oríon's own persisted autonomy state, computed before this turn.
- Use it to color imperative/tone; it is NOT task instruction and never overrides
  user_message, association, grounding, or the actual task.
- It does not add output keys — still emit only valid ThoughtEventV1.
- dominant_drive: {{ autonomy_slice.dominant_drive }}
- active_tensions: {{ autonomy_slice.active_tensions }}
- pressure_trend: {{ autonomy_slice.pressure_trend }}
{% endif %}
```
- [ ] **Step 2:** Manual/inspectable check: trigger a live turn with V2 signal present, confirm the rendered prompt (via existing grammar-step logging) contains the block with real values.
- [ ] **Step 3:** Commit.

---

### Task 6: `orion-thought` extraction + attach

**Files:**
- Modify: `orion/thought/stance_react.py`
- Modify: `services/orion-thought/app/bus_listener.py`

- [ ] **Step 1:** In `orion/thought/stance_react.py`, add `raw.pop("autonomy_slice", None)` alongside the existing `raw.pop("grounding_capsule", None)` (line 183) — defensive, discards anything the LLM invents under that key.
- [ ] **Step 2:** In `bus_listener.py`, add `_extract_autonomy_slice(exec_result)`, mirroring `_extract_grounding_capsule` exactly (reads `metadata["autonomy_slice"]`, validates against `AutonomySliceV1`, warns+returns `None` on parse failure).
- [ ] **Step 3:** In `run_stance_react`, after the existing `capsule = _extract_grounding_capsule(exec_result)` / `model_copy` block, add the same for autonomy: `slice_ = _extract_autonomy_slice(exec_result); if slice_ is not None: enriched = enriched.model_copy(update={"autonomy_slice": slice_})`.
- [ ] **Step 4:** Test: `run_stance_react` with a mocked `exec_result` containing `metadata.autonomy_slice` produces a `ThoughtEventV1` with that field populated; missing/malformed metadata produces `None` without raising.
- [ ] **Step 5:** Run `pytest services/orion-thought/tests -k stance_react -q`, confirm PASS.
- [ ] **Step 6:** Commit.

---

### Task 7: Harness prefix — autonomy_slice

**Files:**
- Modify: `orion/harness/prefix.py`

- [ ] **Step 1:** Add `_format_autonomy_slice(sl: AutonomySliceV1) -> list[str]`, mirroring `_format_stance_slice`'s shape (dominant_drive / active_tensions / pressure_trend lines, only emitted when non-empty).
- [ ] **Step 2:** In `compile_harness_prefix()`, after `parts.extend(_format_stance_slice(...))`, add: `if thought.autonomy_slice is not None: parts.extend(_format_autonomy_slice(thought.autonomy_slice))`.
- [ ] **Step 3:** Test: `compile_harness_prefix` output contains the autonomy lines when `thought.autonomy_slice` is set, and is unchanged (byte-identical) when it's `None` (non-regression for existing harness-prefix tests).
- [ ] **Step 4:** Run `pytest orion/harness/tests -k prefix -q`, confirm PASS.
- [ ] **Step 5:** Commit.

---

### Task 8: Relocate attention-frame to Lane B (template-only)

**Files:**
- Modify: `orion/cognition/prompts/stance_react.j2`
- Modify: `orion/harness/prefix.py`

- [ ] **Step 1:** Add `{% if chat_attention_frame %}` to `stance_react.j2`, adapted from `chat_stance_brief.j2:23-24,62-65`'s content/instructions but in `stance_react.j2`'s house style (matching the `mind_coloring`/`autonomy_slice` block format). No code changes needed — `ctx["chat_attention_frame"]` is already populated by `build_chat_stance_inputs` today when `ORION_CURIOSITY_FRAME_ENABLED=true`.
- [ ] **Step 2:** Add `_format_attention_frame()` to `orion/harness/prefix.py`, adapted from `chat_general.j2:26-27,47,53`'s instructions (advisory only, ask only when `selected_action.action_type == "ask"`, obey suppressions), wired into `compile_harness_prefix()` the same way as Task 7.
- [ ] **Step 3:** Explicitly confirm `chat_stance_brief.j2` and `chat_general.j2` are untouched (diff check) — this is a relocation of the *unused* wiring's Lane-B equivalent, not a removal from Lane A.
- [ ] **Step 4:** Commit.

---

### Task 9: Full gate + PR

- [ ] **Step 1:** Run full test suite (see "Test commands" above).
- [ ] **Step 2:** `python scripts/sync_local_env_from_example.py` if any `.env_example` changed (new DSN key).
- [ ] **Step 3:** Live smoke: trigger one Hub unified-turn chat message, confirm via logs — reducer runs, state persists (second turn's previous_state check), `stance_react.j2` render contains the autonomy block, and (if a harness/FCC turn is also triggered) the harness prefix contains it too.
- [ ] **Step 4:** Run code-review skill in a subagent; fix material findings.
- [ ] **Step 5:** Commit, push, PR description per AGENTS.md §18 template.
