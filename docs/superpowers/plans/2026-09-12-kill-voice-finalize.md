# Kill Voice Finalize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kill mandatory `orion_voice_finalize`; return exact motor drafts when 5b accepts them; run a minimal `orion_response_repair` LLM only when 5b rejects the draft.

**Architecture:** Deterministic `needs_repair` gate after 5b; motor gets `response_policy_summary`; delete `finalize_overlay` as a second-pass concept; `surprise_resolved` follows reflection not string mutation; old verb removed from the cognition registry with reader-only historical compatibility.

**Tech Stack:** Python 3, Pydantic schemas in `orion/schemas/harness_finalize.py`, harness finalize chain in `orion/harness/finalize.py`, FCC prefix in `orion/harness/prefix.py`, cognition verbs/prompts under `orion/cognition/`, cortex-exec route tables, pytest.

**Spec:** `docs/superpowers/specs/2026-09-12-kill-voice-finalize-design.md`

**Worktree:** `/mnt/scripts/Orion-Sapienform-kill-voice-finalize` — rename branch `docs/kill-voice-finalize` → `fix/kill-voice-finalize` at Task 1’s first code commit.

## Global Constraints

- Repair gate is **only** `alignment_verdict in {"misaligned","uncertain"} or strain_unresolved` — overlay must not force repair.
- Aligned + strain resolved → exact draft passthrough (no repair LLM).
- `orion_response_repair` prompt is minimal repair only — no style teaching, no identity dump, no “refine voice and rhythm”.
- Delete `orion_voice_finalize` from verbs/routes; no callable alias.
- Historical readers may map phase/status containing `orion_voice_finalize` → response_repair failure.
- Fail-closed: if repair is required and fails, do not publish the known-bad draft.
- `finalize_ran=true` means 5a/5b completed; add `response_repair_ran` / `response_repair_reason`.
- `surprise_resolved` must not depend on `finalize_changed` or surprise epsilon.
- Same PR ships motor `response_policy_summary` with the skip gate.
- No keyword detectors on user text; no feature flag that restores mandatory voice.
- Follow TDD: failing test → minimal impl → pass → commit per task.

## File map

| File | Responsibility |
|------|----------------|
| `orion/schemas/harness_finalize.py` | `response_repair_*` on run + outcome; drop `finalize_overlay`; doc `finalize_ran` |
| `orion/harness/finalize.py` | Gate, passthrough, surprise_resolved, repair invoke/rename |
| `orion/harness/prefix.py` | Motor response policy block |
| `orion/harness/repair.py` | Stop producing `finalize_overlay` |
| `orion/cognition/verbs/orion_response_repair.yaml` | New verb |
| `orion/cognition/prompts/orion_response_repair.j2` | Minimal repair prompt |
| delete voice verb + prompt | Semantic deletion |
| `services/orion-cortex-exec/app/executor.py` + `grammar_emit.py` | Route/lane for new verb only |
| `services/orion-harness-governor/app/bus_listener.py` | Plumb new fields onto `HarnessRunV1` |
| `orion/hub/turn_orchestrator.py` | Legacy status-string reader |
| harness/cortex tests + READMEs | Acceptance + purge |

---

### Task 1: Schema fields + surprise_resolved (no mutation)

**Files:**
- Modify: `orion/schemas/harness_finalize.py`
- Modify: `orion/harness/finalize.py` (`emit_turn_outcome_molecule` only)
- Create: `orion/harness/tests/test_surprise_resolved_from_reflection.py`
- Modify: `services/orion-harness-governor/app/bus_listener.py` (HarnessRunV1 construction — fields default-safe)

**Interfaces:**
- Produces: `HarnessRunV1.response_repair_ran: bool = False`, `HarnessRunV1.response_repair_reason: str | None = None`
- Produces: same two fields on `HarnessTurnOutcomeMoleculeV1`
- Produces: `surprise_resolved` formula without `finalize_changed` / epsilon
- Consumes: existing `emit_turn_outcome_molecule(...)` kwargs

- [ ] **Step 1: Rename branch for code work**

```bash
cd /mnt/scripts/Orion-Sapienform-kill-voice-finalize
git branch -m fix/kill-voice-finalize
```

- [ ] **Step 2: Write the failing surprise_resolved test**

Create `orion/harness/tests/test_surprise_resolved_from_reflection.py`:

```python
from __future__ import annotations

import pytest

from orion.harness.finalize import emit_turn_outcome_molecule, emit_verdict_molecule
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_thought


@pytest.mark.asyncio
async def test_aligned_passthrough_high_surprise_is_resolved() -> None:
    thought = make_thought()
    appraisal = make_appraisal(surprise_level=0.9)
    reflection = make_reflection(alignment_verdict="aligned", strain_unresolved=False)
    verdict = await emit_verdict_molecule(correlation_id="c-1", reflection=reflection)
    outcome = await emit_turn_outcome_molecule(
        correlation_id="c-1",
        thought=thought,
        substrate_appraisal=appraisal,
        reflection=reflection,
        verdict_molecule=verdict,
        draft_text="hey. i'm here. what's on your mind?",
        final_text="hey. i'm here. what's on your mind?",
        finalize_changed=False,
    )
    assert outcome.surprise_resolved is True


@pytest.mark.asyncio
async def test_misaligned_is_not_surprise_resolved() -> None:
    thought = make_thought()
    appraisal = make_appraisal(surprise_level=0.01)
    reflection = make_reflection(alignment_verdict="misaligned")
    verdict = await emit_verdict_molecule(correlation_id="c-1", reflection=reflection)
    outcome = await emit_turn_outcome_molecule(
        correlation_id="c-1",
        thought=thought,
        substrate_appraisal=appraisal,
        reflection=reflection,
        verdict_molecule=verdict,
        draft_text="draft",
        final_text="draft",
        finalize_changed=False,
        finalize_failed=False,
    )
    assert outcome.surprise_resolved is False
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /mnt/scripts/Orion-Sapienform-kill-voice-finalize
pytest orion/harness/tests/test_surprise_resolved_from_reflection.py::test_aligned_passthrough_high_surprise_is_resolved -v
```

Expected: FAIL — `surprise_resolved` is False (old `finalize_changed or epsilon` clause).

- [ ] **Step 4: Add schema fields**

In `orion/schemas/harness_finalize.py`:

On `HarnessTurnOutcomeMoleculeV1`, after `finalize_changed`:

```python
    response_repair_ran: bool = False
    response_repair_reason: str | None = None
```

On `HarnessRunV1`, after `finalize_ran` / `finalize_changed`, update the `finalize_ran` comment and add:

```python
    # finalize_ran=True means 5a/5b finalization completed (appraisal + reflection),
    # not that a voice/repair LLM rewrote the draft.
    response_repair_ran: bool = False
    response_repair_reason: str | None = None
```

Remove `finalize_overlay` from `HarnessRepairOverlayV1` (delete the field entirely).

- [ ] **Step 5: Fix surprise_resolved producer**

In `orion/harness/finalize.py::emit_turn_outcome_molecule`, replace the formula with:

```python
    surprise_resolved = (
        not finalize_failed
        and reflection.alignment_verdict == "aligned"
        and not reflection.strain_unresolved
    )
```

Add kwargs `response_repair_ran: bool = False`, `response_repair_reason: str | None = None` and pass them into `HarnessTurnOutcomeMoleculeV1(...)`.

Update every `emit_turn_outcome_molecule` / `HarnessRepairOverlayV1(finalize_overlay=...)` call site that breaks typecheck — `map_repair_pressure_contract` in `orion/harness/repair.py` must stop setting `finalize_overlay` (set only `prefix_overlay` + `rule_lines`; delete `_compile_finalize_overlay` if unused).

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest orion/harness/tests/test_surprise_resolved_from_reflection.py -v
pytest orion/harness/tests/test_repair_overlay_changes_harness_prefix.py -v
```

Expected: PASS (fix any overlay tests that asserted `finalize_overlay`).

- [ ] **Step 7: Commit**

```bash
git add orion/schemas/harness_finalize.py orion/harness/finalize.py orion/harness/repair.py \
  orion/harness/tests/test_surprise_resolved_from_reflection.py \
  orion/harness/tests/test_repair_overlay_changes_harness_prefix.py
git commit -m "$(cat <<'EOF'
fix(harness): surprise_resolved follows reflection, not rewrite

Add response_repair_* schema fields; drop finalize_overlay.
EOF
)"
```

---

### Task 2: `needs_repair` gate + aligned passthrough

**Files:**
- Modify: `orion/harness/finalize.py`
- Create: `orion/harness/tests/test_response_repair_gate.py`
- Modify: `orion/harness/tests/test_harness_finalize_chain.py`

**Interfaces:**
- Produces:
  ```python
  def needs_response_repair(reflection: FinalizeReflectionV1) -> bool: ...
  def response_repair_reason_for(reflection: FinalizeReflectionV1) -> str | None: ...
  ```
- Produces: `HarnessFinalizeChainResult.response_repair_ran: bool`, `.response_repair_reason: str | None`
- Consumes: `run_orion_voice_finalize` still present until Task 4 (call only when `needs_response_repair`)

- [ ] **Step 1: Write failing gate + passthrough tests**

Create `orion/harness/tests/test_response_repair_gate.py`:

```python
from __future__ import annotations

import pytest

from orion.harness.finalize import needs_response_repair, response_repair_reason_for, run_harness_finalize_chain
from orion.harness.runner import build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_reflection,
    make_repair_overlay,
    make_thought,
)


def test_needs_repair_false_when_aligned() -> None:
    assert needs_response_repair(make_reflection(alignment_verdict="aligned")) is False
    assert response_repair_reason_for(make_reflection()) is None


def test_needs_repair_true_for_misaligned_uncertain_strain() -> None:
    assert needs_response_repair(make_reflection(alignment_verdict="misaligned")) is True
    assert response_repair_reason_for(make_reflection(alignment_verdict="misaligned")) == "misaligned"
    assert needs_response_repair(make_reflection(alignment_verdict="uncertain")) is True
    assert response_repair_reason_for(make_reflection(alignment_verdict="uncertain")) == "uncertain"
    assert needs_response_repair(
        make_reflection(alignment_verdict="aligned", strain_unresolved=True)
    ) is True
    assert response_repair_reason_for(
        make_reflection(alignment_verdict="aligned", strain_unresolved=True)
    ) == "strain_unresolved"


@pytest.mark.asyncio
async def test_aligned_chain_passthrough_skips_repair_llm() -> None:
    thought = make_thought()
    draft_text = "hey. i'm here. what's on your mind?"
    molecule = build_draft_molecule(
        correlation_id="c-yo",
        thought=thought,
        draft_text=draft_text,
        grammar_receipts=[],
        coalition_snapshot=build_coalition_snapshot(thought),
        repair_overlay=make_repair_overlay(),
    )
    reflection = make_reflection(alignment_verdict="aligned", strain_unresolved=False)
    cortex_calls: list[object] = []

    async def substrate_client(_mol: object):
        return make_appraisal(surprise_level=0.25)

    async def cortex_client(req: object):
        cortex_calls.append(req)
        return {"final_text": reflection.model_dump(mode="json"), "trace_id": "t-5b"}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "orion.harness.finalize.extract_finalize_reflection_payload",
            lambda _result: reflection.model_dump(mode="json"),
        )
        # If repair is wrongly called, force a distinct bad rewrite so we fail loudly.
        mp.setattr(
            "orion.harness.finalize.extract_voice_finalize_text",
            lambda _result: "assumption: checking in. next concrete move: story or quiet.",
        )
        chain = await run_harness_finalize_chain(
            correlation_id="c-yo",
            draft_text=draft_text,
            draft_molecule=molecule,
            thought=thought,
            grammar_receipts=[],
            repair_overlay=make_repair_overlay(mode="concrete_bias", rule_lines=["show assumptions"]),
            user_message="yo",
            voice_contract=None,
            cortex_client=cortex_client,
            substrate_client=substrate_client,
        )

    assert chain.final_text == draft_text
    assert chain.finalize_changed is False
    assert chain.response_repair_ran is False
    assert chain.response_repair_reason is None
    # Overlay must not force a second LLM: only 5b (at most), never repair.
    assert all(
        getattr(getattr(c, "plan", None), "verb_name", None) != "orion_voice_finalize"
        and "voice_finalize" not in str(getattr(getattr(c, "plan", None), "name", ""))
        for c in cortex_calls
    ) or len(cortex_calls) <= 1
```

Also add `test_misaligned_chain_invokes_repair` (expect repair extract called / final differs) in the same file using `alignment_verdict="misaligned"` and stubbing repair extract to return `"repaired text"`.

- [ ] **Step 2: Run tests — expect fail**

```bash
pytest orion/harness/tests/test_response_repair_gate.py -v
```

Expected: FAIL — `needs_response_repair` missing / chain still rewrites.

- [ ] **Step 3: Implement gate helpers**

Near the top of the repair/voice section in `orion/harness/finalize.py`:

```python
def needs_response_repair(reflection: FinalizeReflectionV1) -> bool:
    return (
        reflection.alignment_verdict in {"misaligned", "uncertain"}
        or reflection.strain_unresolved
    )


def response_repair_reason_for(reflection: FinalizeReflectionV1) -> str | None:
    if not needs_response_repair(reflection):
        return None
    if reflection.alignment_verdict == "misaligned":
        return "misaligned"
    if reflection.alignment_verdict == "uncertain":
        return "uncertain"
    return "strain_unresolved"
```

- [ ] **Step 4: Wire passthrough in `run_harness_finalize_chain`**

Replace the `else: run_orion_voice_finalize(...)` branch with:

```python
        elif needs_response_repair(reflection):
            reason = response_repair_reason_for(reflection)
            final_text, voice_meta = await run_orion_voice_finalize(  # renamed in Task 4
                correlation_id=correlation_id,
                draft_text=draft_text,
                thought=thought,
                substrate_appraisal=substrate_appraisal,
                reflection=reflection,
                voice_contract=voice_contract,
                repair_overlay=repair_overlay,
                user_message=user_message,
                grammar_receipts=grammar_receipts,
                cortex_client=cortex_client,
            )
            voice_meta = {
                **voice_meta,
                "response_repair_ran": True,
                "response_repair_reason": reason,
            }
        else:
            logger.info(
                "response_repair_skipped corr=%s reason=aligned",
                correlation_id,
            )
            final_text = draft_text
            voice_meta = {
                "finalize_changed": False,
                "response_repair_ran": False,
                "response_repair_reason": None,
            }
```

Extend `HarnessFinalizeChainResult` with `response_repair_ran` and `response_repair_reason`; populate from `voice_meta` when returning.

Pass `response_repair_*` into `emit_turn_outcome_molecule`.

In `bus_listener.py` when building `HarnessRunV1`:

```python
        response_repair_ran=chain.response_repair_ran,
        response_repair_reason=chain.response_repair_reason,
```

- [ ] **Step 5: Update `test_harness_finalize_chain.py`**

- `test_ordinary_finalize_remains_voice_finalized_for_backward_compatibility`: **delete or rewrite** to `test_ordinary_aligned_finalize_passthrough` expecting `final_text == "motor draft"` and `response_repair_ran is False`.
- `test_run_harness_finalize_chain_orchestrates_5a_through_6b`: use `make_reflection(alignment_verdict="misaligned")` so the second cortex call (repair) still runs; assert `response_repair_ran is True`.
- Keep structured-output test as-is (still skips prose repair).

- [ ] **Step 6: Run focused tests**

```bash
pytest orion/harness/tests/test_response_repair_gate.py orion/harness/tests/test_harness_finalize_chain.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add orion/harness/finalize.py orion/harness/tests/test_response_repair_gate.py \
  orion/harness/tests/test_harness_finalize_chain.py \
  services/orion-harness-governor/app/bus_listener.py
git commit -m "$(cat <<'EOF'
fix(harness): skip response repair when 5b accepts the draft

Aligned/strain-resolved turns return the motor draft unchanged.
EOF
)"
```

---

### Task 3: Motor gets `response_policy_summary`

**Files:**
- Modify: `orion/harness/prefix.py` (`_format_grounding_self_block`)
- Modify: `orion/harness/tests/test_grounding_capsule_consumers.py`
- Modify: `orion/harness/evals/test_unified_turn_grounding_eval.py`
- Modify: `orion/harness/tests/test_harness_prefix.py` if it asserts policy absence

**Interfaces:**
- Produces: motor prefix includes `RESPONSE POLICY` lines from `capsule.response_policy_summary`
- Consumes: `GroundingCapsuleV1.response_policy_summary`

- [ ] **Step 1: Flip the failing assertion**

In `test_grounding_capsule_consumers.py::test_prefix_renders_compact_self_block_when_capsule_present`, replace the “reserved for voice” block with:

```python
    assert "RESPONSE POLICY" in prompt
    assert "no generic-assistant framing" in prompt
```

- [ ] **Step 2: Run — expect fail**

```bash
pytest orion/harness/tests/test_grounding_capsule_consumers.py::test_prefix_renders_compact_self_block_when_capsule_present -v
```

Expected: FAIL.

- [ ] **Step 3: Implement**

In `_format_grounding_self_block`, after identity/relationship/memory, before provenance:

```python
    if capsule.response_policy_summary:
        lines.append("RESPONSE POLICY")
        lines.extend(f"- {item}" for item in capsule.response_policy_summary)
```

Update the docstring: policy is **included** for the motor (no longer reserved for voice finalize).

- [ ] **Step 4: Update grounding eval**

In `test_unified_turn_grounding_eval.py`, assert the banned phrase appears in **prefix** (motor), not only in voice context. Remove or shrink dependence on `build_voice_finalize_context` for policy (voice context goes away in Task 4).

- [ ] **Step 5: Run tests**

```bash
pytest orion/harness/tests/test_grounding_capsule_consumers.py orion/harness/tests/test_harness_prefix.py \
  orion/harness/evals/test_unified_turn_grounding_eval.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add orion/harness/prefix.py orion/harness/tests/test_grounding_capsule_consumers.py \
  orion/harness/evals/test_unified_turn_grounding_eval.py orion/harness/tests/test_harness_prefix.py
git commit -m "$(cat <<'EOF'
feat(harness): feed response_policy_summary into the FCC motor

Speech policy belongs on the draft writer, not a dead voice pass.
EOF
)"
```

---

### Task 4: Semantic deletion — `orion_response_repair` verb + minimal prompt

**Files:**
- Create: `orion/cognition/verbs/orion_response_repair.yaml`
- Create: `orion/cognition/prompts/orion_response_repair.j2`
- Delete: `orion/cognition/verbs/orion_voice_finalize.yaml`
- Delete: `orion/cognition/prompts/orion_voice_finalize.j2`
- Modify: `orion/harness/finalize.py` — rename `run_orion_voice_finalize` → `run_orion_response_repair`, `build_voice_finalize_*` → `build_response_repair_*`, `extract_voice_finalize_text` → `extract_response_repair_text`; default system-error `phase="orion_response_repair"`
- Modify: all imports/tests that name the old symbols
- Create: `orion/cognition/prompts/tests/test_response_repair_prompt.py`
- Modify: `orion/cognition/prompts/tests/test_imperative_first_prompts.py` (drop voice template asserts)

**Interfaces:**
- Produces: `build_plan_for_verb("orion_response_repair", mode="brain")`
- Produces: repair context keys only: `user_message`, `draft_text`, `reflection` (verdict/notes/strain), `grammar_receipts`, `tool_execution`, `llm_route`/`llm_lane` agent
- Does **not** produce: stance slice dump, grounding capsule, voice_contract, finalize_overlay, style curriculum

- [ ] **Step 1: Write prompt contract test (fails until files exist)**

```python
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_response_repair_prompt_is_minimal() -> None:
    text = (REPO / "orion/cognition/prompts/orion_response_repair.j2").read_text(encoding="utf-8")
    assert "smallest necessary correction" in text.lower() or "smallest necessary" in text
    for banned in (
        "refine voice and rhythm",
        "STYLE RULES",
        "WHO YOU ARE",
        "VOICE CONTRACT",
        "STANCE HARNESS",
        "companion presence",
    ):
        assert banned not in text
    assert "orion_voice_finalize" not in text
    assert not (REPO / "orion/cognition/prompts/orion_voice_finalize.j2").exists()
    assert not (REPO / "orion/cognition/verbs/orion_voice_finalize.yaml").exists()
    assert (REPO / "orion/cognition/verbs/orion_response_repair.yaml").exists()
```

- [ ] **Step 2: Run — expect fail**

```bash
pytest orion/cognition/prompts/tests/test_response_repair_prompt.py -v
```

- [ ] **Step 3: Add verb YAML**

`orion/cognition/verbs/orion_response_repair.yaml`:

```yaml
name: orion_response_repair
label: Orion Response Repair
description: >
  Conditional post-reflection repair for unified turns. Invoked only when 5b
  marks the motor draft misaligned, uncertain, or strain_unresolved. Makes the
  smallest necessary correction — never a style polish pass.

category: Generative
priority: high

requires_gpu: true
requires_memory: false
timeout_ms: 300000

services:
  - LLMGatewayService

prompt_template: orion_response_repair.j2

steps:
  - name: llm_orion_response_repair
    order: 0
    services: [LLMGatewayService]
    prompt_template: orion_response_repair.j2
    requires_gpu: true
```

- [ ] **Step 4: Add minimal prompt**

`orion/cognition/prompts/orion_response_repair.j2`:

```jinja2
You are repairing Orion's draft reply to Juniper after integrative reflection rejected it.

Do not polish style. Do not rewrite an already-acceptable answer. Make the smallest necessary correction so the reply matches the reflection verdict and notes. Preserve tool-grounded facts, code blocks, paths, and commands from the draft and receipts. Do not output JSON, meta-planning, or harness narration.

ORIGINAL USER MESSAGE
{{ user_message }}

REJECTED DRAFT
{{ draft_text }}

REFLECTION
- alignment_verdict: {{ reflection.alignment_verdict }}
- alignment_notes: {{ reflection.alignment_notes }}
- strain_unresolved: {{ reflection.strain_unresolved }}
- imperative: {{ reflection.imperative }}
- tone: {{ reflection.tone }}

GRAMMAR RECEIPTS
{{ grammar_receipts }}

TOOL EXECUTION THIS TURN (deterministic)
{{ tool_execution }}

OUTPUT
Write only the repaired user-facing reply below.
```

- [ ] **Step 5: Rename finalize helpers to call the new verb**

- `build_plan_for_verb("orion_response_repair", ...)`
- `build_response_repair_context`: include only the fields the new prompt uses (drop grounding_capsule, stance_harness_slice, voice_contract, finalize_overlay, substrate_appraisal if unused by prompt — keep reflection + receipts + draft + user_message). Keep `llm_route`/`llm_lane` agent + `allow_chat_fallback=False`.
- Delete old voice prompt/verb files.
- Update `test_voice_changes_on_misaligned_verdict.py` → test `run_orion_response_repair` only for misaligned (aligned path is chain-level skip, not this function’s job). Or delete the aligned half of that test.
- Rewrite `test_grounding_capsule_consumers.py` voice-template tests: delete voice Jinja tests; keep motor prefix tests.

- [ ] **Step 6: Registry / load check**

```bash
python -c "from orion.cognition.plan_loader import load_verb_yaml; load_verb_yaml('orion_response_repair'); 
import pytest
try:
  load_verb_yaml('orion_voice_finalize')
  raise SystemExit('old verb still loadable')
except Exception as e:
  print('old verb correctly missing:', type(e).__name__)"
pytest orion/cognition/prompts/tests/test_response_repair_prompt.py orion/harness/tests/test_response_repair_gate.py -v
```

Expected: old verb missing; tests PASS.

- [ ] **Step 7: Commit**

```bash
git add -A orion/cognition/verbs orion/cognition/prompts orion/harness/finalize.py orion/harness/tests \
  orion/harness/evals
git commit -m "$(cat <<'EOF'
feat(harness): replace orion_voice_finalize with minimal response repair

Semantic deletion of the voice pass; repair prompt has no style curriculum.
EOF
)"
```

---

### Task 5: Cortex-exec routes — new verb only

**Files:**
- Modify: `services/orion-cortex-exec/app/executor.py` (replace `orion_voice_finalize` with `orion_response_repair` in agent-lane / max-tokens / logprobs sets)
- Modify: `services/orion-cortex-exec/app/grammar_emit.py`
- Modify: cortex-exec tests listed by grep for `orion_voice_finalize`

**Interfaces:**
- Produces: `_default_llm_route_for_step(verb_name="orion_response_repair", ...) == "agent"`
- Produces: `trace_lane_for_verb("orion_response_repair")`
- Consumes: none from Task 4 beyond verb name string

- [ ] **Step 1: Update tests to the new verb name** (search-replace in the test files under `services/orion-cortex-exec/tests/`)

- [ ] **Step 2: Run — expect fail if executor still lists old name only**

```bash
pytest services/orion-cortex-exec/tests/test_default_llm_route_for_step.py \
  services/orion-cortex-exec/tests/test_harness_finalize_route.py \
  services/orion-cortex-exec/tests/test_chat_reply_logprobs_gate.py -v
```

- [ ] **Step 3: Replace verb strings in executor.py and grammar_emit.py**

Every set membership / comment that named `orion_voice_finalize` becomes `orion_response_repair`. Do not leave both.

- [ ] **Step 4: Re-run tests — PASS, then commit**

```bash
git add services/orion-cortex-exec
git commit -m "$(cat <<'EOF'
fix(cortex-exec): route orion_response_repair on the agent lane

Drop all callable wiring for orion_voice_finalize.
EOF
)"
```

---

### Task 6: Hub historical readers + fail-closed phase string

**Files:**
- Modify: `orion/hub/turn_orchestrator.py` (`_finalize_phase_error` and any `orion_voice_finalize` status checks)
- Modify: `services/orion-hub/tests/test_turn_orchestrator_ws_frames.py`
- Modify: `orion/harness/finalize.py` failure helpers default `phase="orion_response_repair"`
- Modify: `scripts/trace_unified_turn.py` hop labels if they hardcode voice finalize
- Modify: `services/orion-harness-governor/README.md` — document `finalize_ran` + repair fields

**Interfaces:**
- Produces: `_finalize_phase_error` true when `grounding_status` contains `orion_response_repair` **or** legacy `orion_voice_finalize`
- Consumes: `HarnessRunV1.grounding_status`

- [ ] **Step 1: Write/adjust hub test**

```python
def test_finalize_phase_error_recognizes_legacy_voice_and_repair_status() -> None:
    from orion.hub.turn_orchestrator import _finalize_phase_error
    from orion.schemas.harness_finalize import HarnessRunV1

    legacy = HarnessRunV1(
        correlation_id="c",
        final_text=None,
        draft_text="draft",
        finalize_ran=False,
        step_count=1,
        compliance_verdict="failed",
        grounding_status="orion_voice_finalize exec failed: timeout",
    )
    modern = legacy.model_copy(
        update={"grounding_status": "orion_response_repair exec failed: timeout"}
    )
    assert _finalize_phase_error(legacy) is True
    assert _finalize_phase_error(modern) is True
```

(Adapt constructors to whatever required fields `HarnessRunV1` needs — mirror existing hub tests.)

- [ ] **Step 2: Implement reader**

```python
    status = run.grounding_status or ""
    return "orion_response_repair" in status or "orion_voice_finalize" in status
```

- [ ] **Step 3: README note** in harness-governor: mandatory voice finalize is gone; repair is conditional; `finalize_ran` ≠ repair ran.

- [ ] **Step 4: Tests + commit**

```bash
pytest services/orion-hub/tests/test_turn_orchestrator_ws_frames.py -v -k finalize
git add orion/hub/turn_orchestrator.py services/orion-hub/tests \
  services/orion-harness-governor/README.md scripts/trace_unified_turn.py orion/harness/finalize.py
git commit -m "$(cat <<'EOF'
fix(hub): treat legacy voice_finalize status as repair failure

Document finalize_ran vs response_repair_ran for operators.
EOF
)"
```

---

### Task 7: Purge remaining references + fail-closed repair test

**Files:**
- Grep-driven cleanup across repo (tests, docs comments, field-digester notes, smoke scripts)
- Modify: `orion/harness/tests/test_finalize_failure_closure.py` / add misaligned repair failure case
- Modify: `orion/harness/evals/test_layer_attribution.py` — stop treating “5c always rewrites” as quality; assert gate behavior instead

- [ ] **Step 1: Inventory**

```bash
cd /mnt/scripts/Orion-Sapienform-kill-voice-finalize
rg -n "orion_voice_finalize|voice_finalize|run_orion_voice_finalize|build_voice_finalize|refine voice and rhythm|finalize_overlay" \
  --glob '!docs/superpowers/specs/2026-09-12-kill-voice-finalize-design.md' \
  --glob '!docs/superpowers/plans/2026-09-12-kill-voice-finalize.md'
```

- [ ] **Step 2: Fix every **runtime** hit** (code, tests, active READMEs). Historical design docs under `docs/superpowers/specs/2026-07-*` may keep the old name as history; add a one-line “superseded by 2026-09-12-kill-voice-finalize” only if you touch them — do not mass-rewrite archaeology.

- [ ] **Step 3: Fail-closed test** — misaligned reflection + cortex repair raises → `HarnessFinalizeFailedError`, no successful `final_text` publish of the bad draft (mirror existing failure-closure patterns).

- [ ] **Step 4: Full focused suite**

```bash
pytest orion/harness/tests orion/harness/evals \
  services/orion-cortex-exec/tests/test_default_llm_route_for_step.py \
  services/orion-cortex-exec/tests/test_harness_finalize_route.py \
  services/orion-hub/tests/test_turn_orchestrator_ws_frames.py -q
python scripts/check_schema_registry.py
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore(harness): purge voice_finalize call sites and lock fail-closed repair

Aligned drafts never enter a second writer; rejected drafts never ship on repair failure.
EOF
)"
```

---

### Task 8: Spec status + PR report + push gate

**Files:**
- Modify: `docs/superpowers/specs/2026-09-12-kill-voice-finalize-design.md` — Status → implemented (when code lands)
- Create: `docs/superpowers/pr-reports/2026-09-12-kill-voice-finalize-pr.md` (AGENTS §18 template)

- [ ] **Step 1: Run agent-check style gates**

```bash
git diff --check
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
pytest orion/harness/tests -q
```

- [ ] **Step 2: Write PR report** with tests run, restart commands:

```bash
# After merge — rebuild consumers of harness finalize / cortex-exec:
# (print for Juniper; do not sudo)
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
# hub if it vendors turn_orchestrator status parsing from image
scripts/safe_docker_build.sh orion-hub up -d --build
```

- [ ] **Step 3: Commit docs, push branch, open PR** (only when Juniper asks to push/PR, or if standing implement-mode includes it — this task waits for explicit push/PR request if unclear).

```bash
git add docs/superpowers/specs/2026-09-12-kill-voice-finalize-design.md \
  docs/superpowers/pr-reports/2026-09-12-kill-voice-finalize-pr.md
git commit -m "docs: kill-voice-finalize PR report + spec status"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| `needs_repair` gate (no overlay force) | Task 2 |
| Exact draft passthrough when aligned | Task 2 |
| `surprise_resolved` from reflection | Task 1 |
| Motor `response_policy_summary` | Task 3 |
| Delete `finalize_overlay` | Task 1 (+ repair.py) |
| Minimal `orion_response_repair` prompt | Task 4 |
| Delete old verb from registry/routes | Tasks 4–5 |
| Historical reader only | Task 6 |
| `response_repair_ran` / reason | Tasks 1–2, bus_listener |
| Fail-closed on repair failure | Task 7 |
| Structured output unchanged | Task 2 (existing test kept) |
| No mandatory voice restore flag | Global constraints |

## Placeholder / consistency self-check

- No TBD steps; verb name is consistently `orion_response_repair`.
- Helpers renamed in Task 4; Task 2 may still call `run_orion_voice_finalize` until Task 4 renames — implementers must not leave the old public name after Task 4.
- `response_repair_reason` priority: misaligned → uncertain → strain_unresolved.
