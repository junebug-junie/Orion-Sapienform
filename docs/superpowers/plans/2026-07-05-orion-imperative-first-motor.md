# Imperative-first motor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unified-turn motor acts from felt imperative and coalition salience; 5b/5c use grammar traces for accountability — no pre-motor AnswerContract gating or task_mode taxonomy routing.

**Architecture:** Single unified operator brief in harness prefix; Hub passes empty `AnswerContract()`; stance_react prompt gains IMPERATIVE DISCIPLINE; finalize chain passes `grammar_receipts` into 5b/5c cortex contexts. No grammar substrate changes.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, Jinja2 prompts, existing `ThoughtEventV1` / `GrammarReceiptV1` / `HarnessRunRequestV1` schemas.

**Spec:** `docs/superpowers/specs/2026-07-05-orion-imperative-first-motor-design.md`

**Branch:** `fix/unified-turn-harness-runtime` (or fresh worktree from `main` per AGENTS.md)

---

## File map

| File | Responsibility |
|------|----------------|
| `orion/harness/operator_brief.py` | Add `HARNESS_UNIFIED_OPERATOR_BRIEF` constant |
| `orion/harness/prefix.py` | Imperative-first prefix; remove contract/task_mode gating |
| `orion/harness/tests/test_harness_prefix.py` | Prefix + motor instruction gate tests |
| `orion/hub/turn_orchestrator.py` | Drop `heuristic_answer_contract`; pass empty contract |
| `services/orion-hub/tests/test_turn_orchestrator_ws_frames.py` | Assert harness gets conceptual contract |
| `orion/cognition/prompts/stance_react.j2` | IMPERATIVE DISCIPLINE block |
| `orion/thought/tests/test_stance_react_prompt_imperative_discipline.py` | Prompt gate test (new) |
| `orion/harness/finalize.py` | Thread `grammar_receipts` through 5b/5c context builders |
| `orion/harness/tests/test_finalize_grammar_receipts_context.py` | Context builder tests (new) |
| `orion/cognition/prompts/harness_finalize_reflect.j2` | grammar_receipts input + integrative rule |
| `orion/cognition/prompts/orion_voice_finalize.j2` | Trace-based preserve-code rule |
| `orion/cognition/prompts/tests/test_imperative_first_prompts.py` | Voice/reflect prompt gate tests (new) |

**Do not modify:** `orion/substrate/appraisal/`, grammar reducers, `answer_contract_normalize.py`, Brain/context-exec legacy paths.

---

### Task 1: Unified operator brief constant

**Files:**
- Modify: `orion/harness/operator_brief.py`

- [ ] **Step 1: Add constant** (append after existing briefs — keep repo/runtime strings for legacy callers)

```python
HARNESS_UNIFIED_OPERATOR_BRIEF = """\
Orion harness motor.
Tools are available from the start. Your imperative states what this turn requires.
When the imperative calls for facts from the codebase or live runtime, use tools before
answering. Record each meaningful step. Do not guess repo structure or service state from memory.
Before Read on any file: prefer rg/Grep with a path or pattern. For live failures, inspect logs,
docker, and bus traces before diagnosing.
"""
```

- [ ] **Step 2: Verify import**

Run: `python -c "from orion.harness.operator_brief import HARNESS_UNIFIED_OPERATOR_BRIEF; assert 'Tools are available' in HARNESS_UNIFIED_OPERATOR_BRIEF"`

- [ ] **Step 3: Commit**

```bash
git add orion/harness/operator_brief.py
git commit -m "feat(harness): add unified operator brief for imperative-first motor"
```

---

### Task 2: Imperative-first harness prefix (TDD)

**Files:**
- Modify: `orion/harness/prefix.py`
- Modify: `orion/harness/tests/test_harness_prefix.py`

- [ ] **Step 1: Write failing tests** — replace `test_compile_harness_prefix_adds_repo_operator_brief_for_repo_contract` and `test_harness_motor_instruction_repo_turn` with:

```python
def test_compile_harness_prefix_imperative_first_unified_brief() -> None:
    thought = make_thought(
        imperative="Search orion/thought for stance_react; cite file paths.",
        tone="direct",
    )
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="implement a function",
    )
    assert "Tools are available from the start" in prompt
    assert "Search orion/thought for stance_react" in prompt
    assert "Imperative: Search orion/thought" in prompt
    assert "Requires repo grounding" not in prompt
    assert "Answer contract:" not in prompt
    assert "Orion harness motor — repo/technical turn" not in prompt  # old gated brief
    assert "prefer rg/Grep" in prompt  # unified brief includes tool guidance


def test_compile_harness_prefix_ignores_answer_contract() -> None:
    thought = make_thought()
    contract = AnswerContract(
        request_kind="repo_technical",
        requires_repo_grounding=True,
    )
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        answer_contract=contract,
    )
    assert "Requires repo grounding" not in prompt
    assert "Answer contract:" not in prompt


def test_harness_motor_instruction_imperative_forward() -> None:
    thought = make_thought(imperative="Inspect docker logs for orion-hub.")
    instruction = harness_motor_instruction(thought=thought, answer_contract=None)
    assert instruction == (
        "Execute your imperative. Use tools when the turn requires verified facts "
        "from the repo or runtime."
    )
```

Keep `test_compile_harness_prefix_includes_stance_slice` unchanged.

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest orion/harness/tests/test_harness_prefix.py -v`  
Expected: FAIL on new assertions (`Requires repo grounding` still present, etc.)

- [ ] **Step 3: Rewrite `orion/harness/prefix.py`**

Replace entire file content with:

```python
from __future__ import annotations

from orion.harness.operator_brief import HARNESS_UNIFIED_OPERATOR_BRIEF
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.harness_finalize import HarnessRepairOverlayV1
from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1


def _format_stance_slice(sl: StanceHarnessSliceV1) -> list[str]:
    lines = [
        f"Task mode: {sl.task_mode}",
        f"Conversation frame: {sl.conversation_frame}",
        f"Answer strategy: {sl.answer_strategy}",
    ]
    if sl.interaction_regime:
        lines.append(f"Interaction regime: {sl.interaction_regime}")
    if sl.response_priorities:
        lines.append(f"Response priorities: {', '.join(sl.response_priorities)}")
    if sl.response_hazards:
        lines.append(f"Response hazards: {', '.join(sl.response_hazards)}")
    return lines


def harness_motor_instruction(
    *,
    thought: ThoughtEventV1,
    answer_contract: AnswerContract | None,
) -> str:
    _ = thought
    _ = answer_contract  # deprecated on unified motor path; kept for signature compat
    return (
        "Execute your imperative. Use tools when the turn requires verified facts "
        "from the repo or runtime."
    )


def compile_harness_prefix(
    thought: ThoughtEventV1,
    *,
    repair_overlay: HarnessRepairOverlayV1,
    user_message: str = "",
    answer_contract: AnswerContract | None = None,
) -> str:
    """Deterministic fcc system prefix from stance thought + repair overlay."""
    _ = answer_contract  # deprecated on unified motor path; kept for signature compat
    parts: list[str] = [HARNESS_UNIFIED_OPERATOR_BRIEF.strip()]

    parts.extend(
        [
            f"Imperative: {thought.imperative}",
            f"Tone: {thought.tone}",
        ]
    )
    parts.extend(_format_stance_slice(thought.stance_harness_slice))

    if thought.strain_refs:
        parts.append(f"Strain refs: {', '.join(thought.strain_refs)}")

    if user_message.strip():
        parts.append(f"User message: {user_message.strip()}")

    if repair_overlay.mode != "default":
        parts.append(f"Repair mode: {repair_overlay.mode}")

    if repair_overlay.prefix_overlay:
        parts.append(repair_overlay.prefix_overlay)

    if repair_overlay.rule_lines:
        parts.append("Rules: " + "; ".join(repair_overlay.rule_lines))

    return "\n".join(parts)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest orion/harness/tests/test_harness_prefix.py orion/harness/tests/test_repair_overlay_changes_harness_prefix.py orion/harness/tests/test_harness_runner.py::test_harness_runner_uses_compile_harness_prefix -v`

Note: `test_harness_runner_uses_compile_harness_prefix` should still pass — it checks imperative appears in captured prompt.

- [ ] **Step 5: Commit**

```bash
git add orion/harness/prefix.py orion/harness/tests/test_harness_prefix.py
git commit -m "feat(harness): imperative-first prefix, drop contract gating"
```

---

### Task 3: Hub — remove heuristic contract on unified turn

**Files:**
- Modify: `orion/hub/turn_orchestrator.py`
- Modify: `services/orion-hub/tests/test_turn_orchestrator_ws_frames.py`

- [ ] **Step 1: Write failing test** — append to `test_turn_orchestrator_ws_frames.py`:

```python
@pytest.mark.asyncio
async def test_turn_orchestrator_passes_empty_answer_contract_not_heuristic() -> None:
    harness_run = HarnessRunV1(
        correlation_id=_CORR_ID,
        final_text="hello",
        finalize_ran=True,
        compliance_verdict="completed",
        grounding_status="grounded",
    )
    bus = MagicMock()
    harness_client_run = AsyncMock(return_value=harness_run)
    patches = _hub_client_patches(thought=_thought(), harness_run=harness_client_run)
    with patches[0], patches[1], patches[2]:
        await execute_unified_turn(
            bus=bus,
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="docker compose logs show traceback",
            emit_observation_fn=lambda **_kwargs: None,
        )

    req = harness_client_run.await_args.args[0]
    assert req.answer_contract.request_kind == "conceptual"
    assert req.answer_contract.requires_repo_grounding is False
    assert req.answer_contract.requires_runtime_grounding is False
```

Also add source gate test:

```python
def test_turn_orchestrator_source_has_no_heuristic_answer_contract() -> None:
    source = (REPO_ROOT / "orion/hub/turn_orchestrator.py").read_text(encoding="utf-8")
    assert "heuristic_answer_contract" not in source
```

- [ ] **Step 2: Run source gate test — expect FAIL**

Run: `pytest services/orion-hub/tests/test_turn_orchestrator_ws_frames.py::test_turn_orchestrator_source_has_no_heuristic_answer_contract -v`

- [ ] **Step 3: Edit `orion/hub/turn_orchestrator.py`**

Remove line 8:
```python
from orion.cognition.answer_contract_normalize import heuristic_answer_contract
```

Add import:
```python
from orion.schemas.cognition.answer_contract import AnswerContract
```

Replace line 240:
```python
        answer_contract=AnswerContract(),
```

- [ ] **Step 4: Run hub tests — expect PASS**

Run: `pytest services/orion-hub/tests/test_turn_orchestrator_ws_frames.py -v`

- [ ] **Step 5: Commit**

```bash
git add orion/hub/turn_orchestrator.py services/orion-hub/tests/test_turn_orchestrator_ws_frames.py
git commit -m "feat(hub): drop heuristic answer contract on unified turn"
```

---

### Task 4: Stance react — IMPERATIVE DISCIPLINE prompt

**Files:**
- Modify: `orion/cognition/prompts/stance_react.j2`
- Create: `orion/thought/tests/test_stance_react_prompt_imperative_discipline.py`

- [ ] **Step 1: Write failing prompt gate test**

```python
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_stance_react_prompt_imperative_discipline() -> None:
    text = (REPO_ROOT / "orion/cognition/prompts/stance_react.j2").read_text(encoding="utf-8")
    assert "IMPERATIVE DISCIPLINE" in text
    assert "efference copy" in text.lower() or "what Orion must DO" in text
    assert "world-contact" in text.lower() or "world contact" in text.lower()
    assert "if user says" not in text.lower()
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest orion/thought/tests/test_stance_react_prompt_imperative_discipline.py -v`

- [ ] **Step 3: Insert block in `stance_react.j2`** after line 22 (`Represent posture...`), before `REQUIRED JSON FIELDS`:

```jinja
IMPERATIVE DISCIPLINE
- imperative is the efference copy: what Orion must DO this turn, not a summary of the user's message.
- When the turn requires verified facts from code, config, or live services, imperative MUST command
  world-contact in plain language (search repo, read files, inspect logs/traces) before answering.
- When the turn is relational presence, imperative commands companionship — not repo search.
- Bad: "Answer the user's question." / "Explain Python functions."
- Good: "Search services/orion-thought for stance_react wiring; cite paths and symbols."
- Good: "Stay present; one situated question — no task tracking."
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add orion/cognition/prompts/stance_react.j2 orion/thought/tests/test_stance_react_prompt_imperative_discipline.py
git commit -m "feat(stance): imperative discipline in stance_react prompt"
```

---

### Task 5: Grammar receipts in 5b finalize reflect context

**Files:**
- Modify: `orion/harness/finalize.py`
- Modify: `orion/cognition/prompts/harness_finalize_reflect.j2`
- Create: `orion/harness/tests/test_finalize_grammar_receipts_context.py`

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

from orion.harness.finalize import build_finalize_reflect_context
from orion.harness.tests.fixtures import make_appraisal, make_repair_overlay, make_thought
from orion.schemas.harness_finalize import GrammarReceiptV1


def test_build_finalize_reflect_context_includes_grammar_receipts() -> None:
    thought = make_thought()
    receipts = [
        GrammarReceiptV1(step_index=0, tool_name="Read", summary="read coalition.py"),
    ]
    ctx = build_finalize_reflect_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=thought,
        substrate_appraisal=make_appraisal(),
        repair_overlay=make_repair_overlay(),
        user_message="hello",
        grammar_receipts=receipts,
    )
    assert ctx["grammar_receipts"] == [
        {"step": "0", "tool": "Read", "summary": "read coalition.py"},
    ]
```

- [ ] **Step 2: Run test — expect FAIL** (unexpected keyword or KeyError)

Run: `pytest orion/harness/tests/test_finalize_grammar_receipts_context.py -v`

- [ ] **Step 3: Add helper and wire through `orion/harness/finalize.py`**

Add near top of file (after imports):

```python
def grammar_receipt_summaries(receipts: list[GrammarReceiptV1] | None) -> list[dict[str, str]]:
    return [
        {
            "step": str(receipt.step_index),
            "tool": receipt.tool_name or "",
            "summary": receipt.summary,
        }
        for receipt in (receipts or [])
    ]
```

Update `build_finalize_reflect_context` signature to add `grammar_receipts: list[GrammarReceiptV1] | None = None` and include in return dict:

```python
        "grammar_receipts": grammar_receipt_summaries(grammar_receipts),
```

Update `build_finalize_reflect_plan_request` — add same param, pass through.

Update `run_finalize_reflection` — add `grammar_receipts: list[GrammarReceiptV1] | None = None`, pass to `build_finalize_reflect_plan_request`.

Update `run_harness_finalize_chain` call to `run_finalize_reflection`:

```python
    reflection, quick_lane_skipped_5b, cortex_trace_id = await run_finalize_reflection(
        ...
        grammar_receipts=grammar_receipts,
    )
```

- [ ] **Step 4: Update `harness_finalize_reflect.j2`**

After line 11 (`substrate_appraisal`), add:

```jinja
- grammar_receipts: {{ grammar_receipts }}
```

After line 19, add:

```text
- When imperative required world-contact but grammar_receipts is empty, lean misaligned or uncertain.
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `pytest orion/harness/tests/test_finalize_grammar_receipts_context.py orion/harness/tests/test_harness_finalize_chain.py -v`

- [ ] **Step 6: Commit**

```bash
git add orion/harness/finalize.py orion/cognition/prompts/harness_finalize_reflect.j2 orion/harness/tests/test_finalize_grammar_receipts_context.py
git commit -m "feat(harness): pass grammar receipts into 5b reflect context"
```

---

### Task 6: Grammar receipts in 5c voice finalize

**Files:**
- Modify: `orion/harness/finalize.py` (voice context builders)
- Modify: `orion/cognition/prompts/orion_voice_finalize.j2`
- Create: `orion/cognition/prompts/tests/test_imperative_first_prompts.py`

- [ ] **Step 1: Write failing tests**

In `orion/harness/tests/test_finalize_grammar_receipts_context.py` add:

```python
from orion.harness.finalize import build_voice_finalize_context
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.harness.tests.fixtures import make_reflection


def test_build_voice_finalize_context_includes_grammar_receipts() -> None:
    thought = make_thought()
    receipts = [GrammarReceiptV1(step_index=1, tool_name="Grep", summary="grep coalition")]
    ctx = build_voice_finalize_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=thought,
        substrate_appraisal=make_appraisal(),
        reflection=make_reflection(),
        stance_harness_slice=thought.stance_harness_slice,
        voice_contract=AnswerContract(),
        repair_overlay=make_repair_overlay(),
        user_message="hello",
        grammar_receipts=receipts,
    )
    assert ctx["grammar_receipts"][0]["tool"] == "Grep"
```

Create `orion/cognition/prompts/tests/test_imperative_first_prompts.py`:

```python
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_voice_finalize_uses_grammar_not_contract_flags() -> None:
    text = (REPO_ROOT / "orion/cognition/prompts/orion_voice_finalize.j2").read_text(encoding="utf-8")
    assert "grammar_receipts" in text
    assert "requires_repo_grounding" not in text


def test_reflect_prompt_includes_grammar_receipts() -> None:
    text = (REPO_ROOT / "orion/cognition/prompts/harness_finalize_reflect.j2").read_text(encoding="utf-8")
    assert "grammar_receipts" in text
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest orion/harness/tests/test_finalize_grammar_receipts_context.py orion/cognition/prompts/tests/test_imperative_first_prompts.py -v`

- [ ] **Step 3: Update voice builders in `finalize.py`**

Add `grammar_receipts: list[GrammarReceiptV1] | None = None` to:
- `build_voice_finalize_context`
- `build_voice_finalize_plan_request`
- `run_orion_voice_finalize`

Include `"grammar_receipts": grammar_receipt_summaries(grammar_receipts)` in voice context dict.

Update `run_harness_finalize_chain` call to `run_orion_voice_finalize`:

```python
    final_text, voice_meta = await run_orion_voice_finalize(
        ...
        grammar_receipts=grammar_receipts,
    )
```

- [ ] **Step 4: Update `orion_voice_finalize.j2`**

After `STANCE HARNESS SLICE` block (before `VOICE CONTRACT`), add:

```jinja
GRAMMAR RECEIPTS (motor action trace)
{{ grammar_receipts }}
```

Replace line 50:

```jinja
- When grammar_receipts show repo or runtime tool use, preserve code blocks, file paths, commands, and step structure from the draft.
- When alignment_verdict is misaligned, revise — do not ship confabulated specifics.
```

(Remove the old `requires_repo_grounding or requires_runtime_grounding` line entirely.)

- [ ] **Step 5: Run tests — expect PASS**

Run: `pytest orion/harness/tests/test_finalize_grammar_receipts_context.py orion/cognition/prompts/tests/test_imperative_first_prompts.py orion/harness/tests/test_harness_finalize_chain.py -v`

- [ ] **Step 6: Commit**

```bash
git add orion/harness/finalize.py orion/cognition/prompts/orion_voice_finalize.j2 orion/cognition/prompts/tests/test_imperative_first_prompts.py orion/harness/tests/test_finalize_grammar_receipts_context.py
git commit -m "feat(harness): trace-backed 5c voice rules via grammar receipts"
```

---

### Task 7: Final gate + review

- [ ] **Step 1: Grep gate**

Run:
```bash
rg 'heuristic_answer_contract|_TECH_TASK_MODES|_needs_repo_operator_brief|requires_repo_grounding' \
  orion/hub/turn_orchestrator.py orion/harness/prefix.py
```
Expected: no matches.

Run:
```bash
rg 'requires_repo_grounding' orion/cognition/prompts/orion_voice_finalize.j2
```
Expected: no matches.

- [ ] **Step 2: Full test sweep**

Run:
```bash
pytest orion/harness/tests/ orion/thought/tests/test_stance_react_prompt_imperative_discipline.py \
  orion/cognition/prompts/tests/test_imperative_first_prompts.py \
  services/orion-hub/tests/test_turn_orchestrator_ws_frames.py -q
```
Expected: all passed.

- [ ] **Step 3: Code review subagent** — run code-reviewer on branch; fix material findings.

- [ ] **Step 4: Commit any review fixes**

---

## Spec coverage self-review

| Spec requirement | Task |
|------------------|------|
| Drop `heuristic_answer_contract` on unified turn | Task 3 |
| Remove contract/task_mode motor gating | Task 2 |
| Unified operator brief | Tasks 1–2 |
| Imperative-forward motor instruction | Task 2 |
| IMPERATIVE DISCIPLINE in stance_react.j2 | Task 4 |
| grammar_receipts in 5b context | Task 5 |
| grammar_receipts in 5c; no requires_repo_grounding in voice | Task 6 |
| No grammar substrate changes | Explicit non-touch |
| Legacy Brain heuristics frozen | Not in plan |
| Gate tests from spec acceptance table | Tasks 2–7 |

No placeholders remain. Type names consistent: `grammar_receipt_summaries`, `GrammarReceiptV1`, `build_finalize_reflect_context`.

---

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml up -d --build
```

Adjust compose service names to match local layout if needed.

---

Plan complete and saved to `docs/superpowers/plans/2026-07-05-orion-imperative-first-motor.md`.

**Two execution options:**

1. **Subagent-driven (recommended)** — fresh subagent per task, review between tasks, fast iteration  
2. **Inline execution** — implement all 7 tasks in this session with checkpoints

Which approach?
