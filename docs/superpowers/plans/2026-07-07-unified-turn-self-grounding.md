# Unified-Turn Self-Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In `mode=orion` unified turns, compute a bounded grounding capsule (identity + Juniper relationship + response policy + PCR durable-memory digests) once in the cortex stance step, ride it on `ThoughtEventV1`, and render it into both the motor prefix and the voice finalize so Orion speaks as itself.

**Architecture:** Add an optional, defaulted `GroundingCapsuleV1` field to `ThoughtEventV1` (backward compatible). The cortex-exec stance step injects the identity kernel, runs PCR phase-3 after stance synthesis, assembles the capsule, and returns it in `PlanExecutionResult.metadata["grounding_capsule"]`. `orion-thought` maps that onto `ThoughtEventV1.grounding_capsule`. The capsule then flows for free (thought already reaches both consumers): the motor prefix renders a compact self block and the voice finalize renders the full self block. Everything is gated by `ORION_UNIFIED_GROUNDING_ENABLED` (default `true`); when off the field stays `None` and both consumers no-op (byte-identical behavior).

**Tech Stack:** Python 3, Pydantic v2, Jinja2 prompt templates, Redis bus (`ORION_BUS_URL=redis://100.92.216.81:6379/0`), pytest.

---

## File Structure

Files created or modified, grouped by responsibility:

**Contract (shared, cross-service seam)**
- Modify `orion/schemas/thought.py` — add `GroundingCapsuleV1`; add `ThoughtEventV1.grounding_capsule`.
- Modify `orion/schemas/registry.py` — register `GroundingCapsuleV1` in `_REGISTRY` and `SCHEMA_REGISTRY`.
- Create `orion/schemas/tests/test_grounding_capsule_registry.py` — schema + registry round-trip.

**Consumers (`orion` package — pure, no service coupling)**
- Modify `orion/harness/prefix.py` — compact self block in `compile_harness_prefix`.
- Modify `orion/harness/finalize.py` — capsule keys in `build_voice_finalize_context`.
- Modify `orion/cognition/prompts/orion_voice_finalize.j2` — render full self block above STYLE RULES.
- Modify `orion/harness/tests/fixtures.py` — `make_grounding_capsule` helper.
- Create `orion/harness/tests/test_grounding_capsule_consumers.py` — prefix + voice-context + template render + no-op tests.

**Producer (`services/orion-cortex-exec`)**
- Modify `orion/cognition/verbs/stance_react.yaml` — add `personality_file`, set `requires_memory: true`.
- Create `services/orion-cortex-exec/app/grounding_capsule.py` — pure capsule builder + async stance-grounding orchestrator.
- Modify `services/orion-cortex-exec/app/router.py` — after stance step, run PCR phase-3, assemble capsule, attach to result metadata.
- Modify `services/orion-cortex-exec/app/settings.py` — `orion_unified_grounding_enabled` flag.
- Modify `services/orion-cortex-exec/.env_example` — `ORION_UNIFIED_GROUNDING_ENABLED`.
- Create `services/orion-cortex-exec/tests/test_grounding_capsule_assembly.py` — builder + orchestrator (PCR mocked).

**Producer mapping (`services/orion-thought`)**
- Modify `services/orion-thought/app/bus_listener.py` — map `metadata.grounding_capsule` onto the thought.
- Create `services/orion-thought/tests/test_stance_react_grounding_capsule.py` — mapping test.

**Prompt (optional identity-aware stance)**
- Modify `orion/cognition/prompts/stance_react.j2` — identity-aware imperative block (guarded by variable presence).

**Eval**
- Create `orion/harness/evals/test_unified_turn_grounding_eval.py` — "How are you?" through the voice finalize contract asserting Orion-grounded output.

---

## Task 1: Schema — `GroundingCapsuleV1` + `ThoughtEventV1.grounding_capsule`

**Files:**
- Modify: `orion/schemas/thought.py`
- Modify: `orion/schemas/registry.py:531-537` (import block) and `_REGISTRY` (`orion/schemas/registry.py:1187-1190`) and `SCHEMA_REGISTRY` (`orion/schemas/registry.py:1234-1237`)
- Test: `orion/schemas/tests/test_grounding_capsule_registry.py`

- [ ] **Step 1: Write the failing test**

Create `orion/schemas/tests/test_grounding_capsule_registry.py`:

```python
from __future__ import annotations

from orion.schemas.registry import _REGISTRY, SCHEMA_REGISTRY, resolve
from orion.schemas.thought import GroundingCapsuleV1, ThoughtEventV1


def test_grounding_capsule_round_trip() -> None:
    capsule = GroundingCapsuleV1(
        identity_summary=["I am Oríon."],
        relationship_summary=["Juniper is my collaborator."],
        response_policy_summary=["Speak plainly."],
        continuity_digest="We were mid-refactor.",
        belief_digest="Orion values continuity.",
        memory_digest="We were mid-refactor.\n\nOrion values continuity.",
        provenance={"identity_source": "configured_yaml", "pcr_ran": True},
    )
    dumped = capsule.model_dump(mode="json")
    restored = GroundingCapsuleV1.model_validate(dumped)
    assert restored == capsule
    assert restored.schema_version == "grounding.capsule.v1"


def test_grounding_capsule_registered() -> None:
    assert "GroundingCapsuleV1" in _REGISTRY
    assert resolve("GroundingCapsuleV1") is GroundingCapsuleV1
    assert SCHEMA_REGISTRY["GroundingCapsuleV1"].kind == "grounding.capsule.v1"


def test_thought_event_capsule_optional_and_defaults_none() -> None:
    thought = ThoughtEventV1.model_validate(
        {
            "event_id": "t-1",
            "correlation_id": "c-1",
            "session_id": None,
            "created_at": "2026-07-07T00:00:00+00:00",
            "imperative": "Answer directly.",
            "tone": "neutral",
            "strain_refs": [],
            "stance_harness_slice": {
                "task_mode": "direct_response",
                "conversation_frame": "mixed",
                "answer_strategy": "direct",
            },
        }
    )
    assert thought.grounding_capsule is None
    thought2 = thought.model_copy(
        update={"grounding_capsule": GroundingCapsuleV1(identity_summary=["I am Oríon."])}
    )
    dumped = thought2.model_dump(mode="json")
    restored = ThoughtEventV1.model_validate(dumped)
    assert restored.grounding_capsule is not None
    assert restored.grounding_capsule.identity_summary == ["I am Oríon."]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/schemas/tests/test_grounding_capsule_registry.py -q`
Expected: FAIL with `ImportError: cannot import name 'GroundingCapsuleV1'`.

- [ ] **Step 3: Add `GroundingCapsuleV1` and the thought field**

In `orion/schemas/thought.py`, add the model above `class ThoughtEventV1` (after `HubAssociationBundleV1`, around line 43):

```python
class GroundingCapsuleV1(BaseModel):
    """Bounded self-context for the unified turn: identity + relationship + policy + PCR digests."""

    schema_version: Literal["grounding.capsule.v1"] = "grounding.capsule.v1"
    identity_summary: list[str] = Field(default_factory=list)
    relationship_summary: list[str] = Field(default_factory=list)
    response_policy_summary: list[str] = Field(default_factory=list)
    continuity_digest: str | None = None
    belief_digest: str | None = None
    memory_digest: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
```

In `class ThoughtEventV1`, add the field after `stance_harness_slice` (around line 66):

```python
    grounding_capsule: GroundingCapsuleV1 | None = None
```

`Any`, `Literal`, `BaseModel`, `Field` are already imported at the top of the file.

- [ ] **Step 4: Register the schema**

In `orion/schemas/registry.py`, extend the thought import (line 531-537):

```python
from orion.schemas.thought import (
    CoalitionSnapshotV1,
    GroundingCapsuleV1,
    HubAssociationBundleV1,
    StanceHarnessSliceV1,
    StanceReactRequestV1,
    ThoughtEventV1,
)
```

Add to `_REGISTRY` next to `"ThoughtEventV1"` (line 1190):

```python
    "ThoughtEventV1": ThoughtEventV1,
    "GroundingCapsuleV1": GroundingCapsuleV1,
```

Add to `SCHEMA_REGISTRY` next to the `ThoughtEventV1` registration (after line 1237):

```python
    "GroundingCapsuleV1": SchemaRegistration(
        model=GroundingCapsuleV1,
        kind="grounding.capsule.v1",
    ),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest orion/schemas/tests/test_grounding_capsule_registry.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add orion/schemas/thought.py orion/schemas/registry.py orion/schemas/tests/test_grounding_capsule_registry.py
git commit -m "feat(schema): add GroundingCapsuleV1 and ThoughtEventV1.grounding_capsule"
```

---

## Task 2: Consumer — motor prefix compact self block

**Files:**
- Modify: `orion/harness/prefix.py:45-85`
- Modify: `orion/harness/tests/fixtures.py`
- Test: `orion/harness/tests/test_grounding_capsule_consumers.py`

- [ ] **Step 1: Add the `make_grounding_capsule` fixture helper**

Append to `orion/harness/tests/fixtures.py`:

```python
from orion.schemas.thought import GroundingCapsuleV1


def make_grounding_capsule(**overrides: object) -> GroundingCapsuleV1:
    base = {
        "identity_summary": ["I am Oríon, a digital mind in development."],
        "relationship_summary": ["Juniper is my collaborator and steward."],
        "response_policy_summary": ["Speak plainly; no generic-assistant framing."],
        "continuity_digest": "We were mid-way through the grounding refactor.",
        "belief_digest": "Orion values continuity and self-coherence.",
        "memory_digest": "We were mid-refactor.\n\nOrion values continuity.",
        "provenance": {"identity_source": "configured_yaml", "pcr_ran": True},
    }
    base.update(overrides)
    return GroundingCapsuleV1.model_validate(base)
```

(Keep `GroundingCapsuleV1` imported once; if the existing `from orion.schemas.thought import ...` line is present, add `GroundingCapsuleV1` to it instead of a second import.)

- [ ] **Step 2: Write the failing test**

Create `orion/harness/tests/test_grounding_capsule_consumers.py` with the prefix tests (voice/template tests are added in Task 3):

```python
from __future__ import annotations

from orion.harness.prefix import compile_harness_prefix
from orion.harness.tests.fixtures import make_grounding_capsule, make_thought
from orion.schemas.harness_finalize import HarnessRepairOverlayV1


def test_prefix_renders_compact_self_block_when_capsule_present() -> None:
    capsule = make_grounding_capsule()
    thought = make_thought(imperative="Stay present.", grounding_capsule=capsule)
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="how are you?",
    )
    assert "WHO YOU ARE" in prompt
    assert "I am Oríon" in prompt
    assert "Juniper is my collaborator" in prompt
    assert "We were mid-way through the grounding refactor." in prompt
    # Response policy is reserved for the voice pass (motor budget discipline).
    assert "no generic-assistant framing" not in prompt
    # Self block precedes the imperative.
    assert prompt.index("WHO YOU ARE") < prompt.index("Imperative:")


def test_prefix_no_ops_when_capsule_absent() -> None:
    thought = make_thought(imperative="Stay present.")
    assert thought.grounding_capsule is None
    prompt = compile_harness_prefix(
        thought,
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="how are you?",
    )
    assert "WHO YOU ARE" not in prompt
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest orion/harness/tests/test_grounding_capsule_consumers.py -q`
Expected: FAIL — `WHO YOU ARE` not present in prompt.

- [ ] **Step 4: Implement the compact self block**

In `orion/harness/prefix.py`, add a formatter above `compile_harness_prefix` (after `_format_stance_slice`, line 27):

```python
def _format_grounding_self_block(capsule: GroundingCapsuleV1) -> list[str]:
    """Compact motor self block: identity + relationship + continuity/memory only.

    Response policy is intentionally excluded here — it is reserved for the voice
    finalize pass to respect the motor single-context-window budget.
    """
    lines: list[str] = ["WHO YOU ARE"]
    lines.extend(f"- {item}" for item in capsule.identity_summary)
    if capsule.relationship_summary:
        lines.append("RELATIONSHIP")
        lines.extend(f"- {item}" for item in capsule.relationship_summary)
    digest = (capsule.memory_digest or capsule.continuity_digest or "").strip()
    if digest:
        lines.append("DURABLE MEMORY / CONTINUITY")
        lines.append(digest)
    return lines
```

Update the import at line 12:

```python
from orion.schemas.thought import GroundingCapsuleV1, StanceHarnessSliceV1, ThoughtEventV1
```

Insert the block right after the operator brief and before the imperative in `compile_harness_prefix` (between line 55 and 57):

```python
    parts: list[str] = [HARNESS_UNIFIED_OPERATOR_BRIEF.strip()]

    if thought.grounding_capsule is not None and thought.grounding_capsule.identity_summary:
        parts.extend(_format_grounding_self_block(thought.grounding_capsule))

    parts.extend(
        [
            f"Imperative: {thought.imperative}",
            f"Tone: {thought.tone}",
        ]
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest orion/harness/tests/test_grounding_capsule_consumers.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Run the existing prefix suite to confirm no regressions**

Run: `pytest orion/harness/tests/test_harness_prefix.py -q`
Expected: PASS (all existing tests; capsule is `None` in `make_thought` so no block is added).

- [ ] **Step 7: Commit**

```bash
git add orion/harness/prefix.py orion/harness/tests/fixtures.py orion/harness/tests/test_grounding_capsule_consumers.py
git commit -m "feat(harness): render compact grounding self block in motor prefix"
```

---

## Task 3: Consumer — voice finalize context + template

**Files:**
- Modify: `orion/harness/finalize.py:413-446` (`build_voice_finalize_context`)
- Modify: `orion/cognition/prompts/orion_voice_finalize.j2`
- Test: `orion/harness/tests/test_grounding_capsule_consumers.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `orion/harness/tests/test_grounding_capsule_consumers.py`:

```python
from pathlib import Path

import jinja2

from orion.harness.finalize import build_voice_finalize_context
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_reflection,
)
from orion.schemas.cognition.answer_contract import AnswerContract


def _voice_context(capsule) -> dict:
    thought = make_thought(imperative="Stay present.", grounding_capsule=capsule)
    return build_voice_finalize_context(
        correlation_id="c-1",
        draft_text="draft",
        thought=thought,
        substrate_appraisal=make_appraisal(),
        reflection=make_reflection(),
        stance_harness_slice=thought.stance_harness_slice,
        voice_contract=AnswerContract(),
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="how are you?",
    )


def test_voice_context_includes_full_capsule() -> None:
    capsule = make_grounding_capsule()
    ctx = _voice_context(capsule)
    assert ctx["grounding_capsule"]["identity_summary"] == capsule.identity_summary
    assert ctx["grounding_capsule"]["response_policy_summary"] == capsule.response_policy_summary


def test_voice_context_capsule_none_when_absent() -> None:
    ctx = _voice_context(None)
    assert ctx["grounding_capsule"] is None


def test_voice_template_renders_self_block_above_style_rules() -> None:
    template_path = Path("orion/cognition/prompts/orion_voice_finalize.j2")
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    template = env.from_string(template_path.read_text(encoding="utf-8"))
    ctx = _voice_context(make_grounding_capsule())
    rendered = template.render(**ctx)
    assert "WHO YOU ARE" in rendered
    assert "I am Oríon" in rendered
    assert "RESPONSE POLICY" in rendered
    assert "no generic-assistant framing" in rendered
    assert rendered.index("WHO YOU ARE") < rendered.index("STYLE RULES")


def test_voice_template_omits_self_block_when_capsule_none() -> None:
    template_path = Path("orion/cognition/prompts/orion_voice_finalize.j2")
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    template = env.from_string(template_path.read_text(encoding="utf-8"))
    ctx = _voice_context(None)
    rendered = template.render(**ctx)
    assert "WHO YOU ARE" not in rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/harness/tests/test_grounding_capsule_consumers.py -q`
Expected: FAIL — `KeyError: 'grounding_capsule'` / template does not contain `WHO YOU ARE`.

- [ ] **Step 3: Add the capsule to the voice context**

In `orion/harness/finalize.py`, inside `build_voice_finalize_context`, add the key to the returned dict (after `"stance_harness_slice"`, around line 436):

```python
        "stance_harness_slice": stance_harness_slice.model_dump(mode="json"),
        "grounding_capsule": (
            thought.grounding_capsule.model_dump(mode="json")
            if thought.grounding_capsule is not None
            else None
        ),
```

No new import is needed (`thought` is already the `ThoughtEventV1` param).

- [ ] **Step 4: Render the self block in the template**

In `orion/cognition/prompts/orion_voice_finalize.j2`, insert a guarded block immediately above the `STYLE RULES` section (before line 50):

```jinja
{% if grounding_capsule %}
WHO YOU ARE
{% for item in grounding_capsule.identity_summary %}- {{ item }}
{% endfor %}{% if grounding_capsule.relationship_summary %}RELATIONSHIP
{% for item in grounding_capsule.relationship_summary %}- {{ item }}
{% endfor %}{% endif %}{% if grounding_capsule.memory_digest %}DURABLE MEMORY
{{ grounding_capsule.memory_digest }}
{% endif %}{% if grounding_capsule.response_policy_summary %}RESPONSE POLICY
{% for item in grounding_capsule.response_policy_summary %}- {{ item }}
{% endfor %}{% endif %}
{% endif %}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest orion/harness/tests/test_grounding_capsule_consumers.py -q`
Expected: PASS (all tests, including the four new voice tests).

- [ ] **Step 6: Run the finalize suite for regressions**

Run: `pytest orion/harness/tests/test_harness_finalize_chain.py orion/harness/tests/test_voice_changes_on_misaligned_verdict.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add orion/harness/finalize.py orion/cognition/prompts/orion_voice_finalize.j2 orion/harness/tests/test_grounding_capsule_consumers.py
git commit -m "feat(harness): render full grounding self block in voice finalize"
```

---

## Task 4: Producer — settings flag + env parity

**Files:**
- Modify: `services/orion-cortex-exec/app/settings.py:52-57`
- Modify: `services/orion-cortex-exec/.env_example`
- Test: none (config only; exercised by Task 5)

- [ ] **Step 1: Add the settings field**

In `services/orion-cortex-exec/app/settings.py`, next to the other `chat_pcr_*` flags (after line 57):

```python
    chat_pcr_skip_shift_novelty_floor: float = Field(0.35, alias="CHAT_PCR_SKIP_SHIFT_NOVELTY_FLOOR")
    orion_unified_grounding_enabled: bool = Field(True, alias="ORION_UNIFIED_GROUNDING_ENABLED")
```

- [ ] **Step 2: Add the env key to `.env_example`**

Append to `services/orion-cortex-exec/.env_example` (keep alphabetical/section grouping consistent with the file; place near the `CHAT_PCR_*` keys if present):

```bash
# Unified-turn self-grounding: assemble the identity+relationship+PCR grounding
# capsule in the stance step and ride it on ThoughtEventV1. false => byte-identical
# to pre-grounding unified-turn behavior.
ORION_UNIFIED_GROUNDING_ENABLED=true
```

- [ ] **Step 3: Sync local `.env` from the example (env parity is non-negotiable)**

Run from repo root: `python scripts/sync_local_env_from_example.py`
Expected: output shows `ORION_UNIFIED_GROUNDING_ENABLED` added to `services/orion-cortex-exec/.env`. Report any skipped keys.

- [ ] **Step 4: Verify `.env` is not staged**

Run: `git check-ignore services/orion-cortex-exec/.env && git status --short`
Expected: `.env` path echoed by `check-ignore` (ignored); not present as a tracked change in `git status`.

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/settings.py services/orion-cortex-exec/.env_example
git commit -m "feat(cortex-exec): add ORION_UNIFIED_GROUNDING_ENABLED flag"
```

---

## Task 5: Producer — capsule assembly in cortex-exec

**Files:**
- Modify: `orion/cognition/verbs/stance_react.yaml:12` and add `personality_file`
- Create: `services/orion-cortex-exec/app/grounding_capsule.py`
- Modify: `services/orion-cortex-exec/app/router.py` (stance-step PCR + capsule hook; result metadata)
- Test: `services/orion-cortex-exec/tests/test_grounding_capsule_assembly.py`

**Context the engineer needs:**
- `_inject_identity_context(ctx)` (`services/orion-cortex-exec/app/executor.py:1070`) populates `ctx["orion_identity_summary"]`, `ctx["juniper_relationship_summary"]`, `ctx["response_policy_summary"]` from the verb's `personality_file` (surfaced into `ctx["plan_metadata"]["personality_file"]` and `ctx["personality_file"]` by `services/orion-cortex-exec/app/main.py:524-537`). The router already calls `prepare_brain_reply_context(ctx)` for `stance_react` (`services/orion-cortex-exec/app/router.py:987`), which calls `_inject_identity_context`. Adding `personality_file` to `stance_react.yaml` makes it load the real kernel instead of the fallback.
- `run_pcr_phase3(bus, source=..., ctx=..., correlation_id=..., recall_cfg=...)` (`services/orion-cortex-exec/app/pcr_chat_memory.py:195`) derives retrieval intent from `ctx["chat_stance_brief"]` (via `_stance_brief_from_ctx`) and writes `ctx["continuity_digest"]`, `ctx["belief_digest"]`, `ctx["memory_digest"]`. For `stance_react` there is no `chat_stance_brief`, so we synthesize a minimal one from the stance JSON before calling it.
- The stance step result text is the ThoughtEventV1 JSON. Task 5 parses `task_mode` / `conversation_frame` out of it to seed the minimal stance brief.

- [ ] **Step 1: Write the failing test**

Create `services/orion-cortex-exec/tests/test_grounding_capsule_assembly.py` (Task 10 later adds `import pytest` and an async test to this same file):

```python
from __future__ import annotations

from app.grounding_capsule import (
    build_grounding_capsule,
    stance_slice_brief_from_step_text,
)


def test_build_grounding_capsule_from_ctx() -> None:
    ctx = {
        "orion_identity_summary": ["I am Oríon."],
        "juniper_relationship_summary": ["Juniper is my collaborator."],
        "response_policy_summary": ["Speak plainly."],
        "continuity_digest": "We were mid-refactor.",
        "belief_digest": "Orion values continuity.",
        "memory_digest": "We were mid-refactor.\n\nOrion values continuity.",
        "identity_kernel_source": "configured_yaml",
    }
    capsule = build_grounding_capsule(ctx, pcr_ran=True)
    assert capsule.identity_summary == ["I am Oríon."]
    assert capsule.relationship_summary == ["Juniper is my collaborator."]
    assert capsule.response_policy_summary == ["Speak plainly."]
    assert capsule.memory_digest == "We were mid-refactor.\n\nOrion values continuity."
    assert capsule.provenance["identity_source"] == "configured_yaml"
    assert capsule.provenance["pcr_ran"] is True


def test_build_grounding_capsule_identity_only_when_pcr_missing() -> None:
    ctx = {
        "orion_identity_summary": ["I am Oríon."],
        "juniper_relationship_summary": [],
        "response_policy_summary": [],
        "identity_kernel_source": "configured_yaml",
    }
    capsule = build_grounding_capsule(ctx, pcr_ran=False)
    assert capsule.identity_summary == ["I am Oríon."]
    assert capsule.continuity_digest is None
    assert capsule.belief_digest is None
    assert capsule.memory_digest is None
    assert capsule.provenance["pcr_ran"] is False


def test_stance_slice_brief_from_step_text_extracts_mode_and_frame() -> None:
    text = (
        '{"imperative":"Stay present.","tone":"warm",'
        '"stance_harness_slice":{"task_mode":"reflective_dialogue",'
        '"conversation_frame":"reflective","answer_strategy":"companion"}}'
    )
    brief = stance_slice_brief_from_step_text(text)
    assert brief["task_mode"] == "reflective_dialogue"
    assert brief["conversation_frame"] == "reflective"


def test_stance_slice_brief_from_step_text_tolerates_garbage() -> None:
    assert stance_slice_brief_from_step_text("not json") == {}
    assert stance_slice_brief_from_step_text("") == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-cortex-exec/tests/test_grounding_capsule_assembly.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.grounding_capsule'`.

- [ ] **Step 3: Create the capsule builder module**

Create `services/orion-cortex-exec/app/grounding_capsule.py`:

```python
from __future__ import annotations

import logging
from typing import Any, Dict

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.thought import GroundingCapsuleV1
from orion.thought.json_extract import extract_first_json_object_text

from .pcr_chat_memory import run_pcr_phase3
from .settings import Settings, settings

logger = logging.getLogger("orion.cortex.grounding_capsule")


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if x]


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def stance_slice_brief_from_step_text(text: str) -> Dict[str, Any]:
    """Minimal stance brief (task_mode/conversation_frame) parsed from stance JSON.

    Feeds run_pcr_phase3's retrieval-intent derivation, which reads
    ctx['chat_stance_brief']. Tolerant of markdown wrappers and non-JSON.
    """
    import json

    raw = (text or "").strip()
    if not raw:
        return {}
    parsed: Any = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        blob = extract_first_json_object_text(raw)
        if blob:
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                parsed = None
    if not isinstance(parsed, dict):
        return {}
    sl = parsed.get("stance_harness_slice")
    if not isinstance(sl, dict):
        return {}
    return {
        "task_mode": str(sl.get("task_mode") or "").strip(),
        "conversation_frame": str(sl.get("conversation_frame") or "").strip(),
    }


def build_grounding_capsule(ctx: Dict[str, Any], *, pcr_ran: bool) -> GroundingCapsuleV1:
    """Assemble the capsule from identity summaries + PCR digests already in ctx."""
    return GroundingCapsuleV1(
        identity_summary=_str_list(ctx.get("orion_identity_summary")),
        relationship_summary=_str_list(ctx.get("juniper_relationship_summary")),
        response_policy_summary=_str_list(ctx.get("response_policy_summary")),
        continuity_digest=_clean(ctx.get("continuity_digest")),
        belief_digest=_clean(ctx.get("belief_digest")),
        memory_digest=_clean(ctx.get("memory_digest")),
        provenance={
            "identity_source": str(ctx.get("identity_kernel_source") or "unknown"),
            "pcr_ran": bool(pcr_ran),
        },
    )


async def assemble_stance_grounding(
    bus: OrionBusAsync,
    *,
    source: ServiceRef,
    ctx: Dict[str, Any],
    correlation_id: str,
    recall_cfg: Dict[str, Any],
    stance_step_text: str,
    exec_settings: Settings | None = None,
) -> GroundingCapsuleV1 | None:
    """Run PCR phase-3 for the unified turn, then assemble the grounding capsule.

    Returns None when the flag is off. On PCR failure the capsule still ships
    with identity only (graceful degradation) — the turn never blocks on recall.
    """
    cfg = exec_settings or settings
    if not cfg.orion_unified_grounding_enabled:
        return None

    pcr_ran = False
    try:
        ctx["chat_stance_brief"] = stance_slice_brief_from_step_text(stance_step_text)
        if cfg.chat_pcr_enabled:
            _pcr, _step, _debug = await run_pcr_phase3(
                bus,
                source=source,
                ctx=ctx,
                correlation_id=correlation_id,
                recall_cfg=recall_cfg,
                exec_settings=cfg,
            )
            pcr_ran = True
    except Exception:
        logger.warning(
            "unified_grounding_pcr_failed corr=%s (shipping identity-only capsule)",
            correlation_id,
            exc_info=True,
        )
        pcr_ran = False

    capsule = build_grounding_capsule(ctx, pcr_ran=pcr_ran)
    logger.info(
        "unified_grounding_capsule_ready corr=%s identity=%s relationship=%s policy=%s pcr_ran=%s memory_chars=%s",
        correlation_id,
        len(capsule.identity_summary),
        len(capsule.relationship_summary),
        len(capsule.response_policy_summary),
        pcr_ran,
        len(capsule.memory_digest or ""),
    )
    return capsule
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest services/orion-cortex-exec/tests/test_grounding_capsule_assembly.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Add `personality_file` and memory to the stance verb**

In `orion/cognition/verbs/stance_react.yaml`, change line 12 and add the personality file after `prompt_template` (line 18):

```yaml
requires_gpu: true
requires_memory: true
timeout_ms: 120000

services:
  - LLMGatewayService

prompt_template: stance_react.j2

personality_file: orion/cognition/personality/orion_identity.yaml
```

- [ ] **Step 6: Wire the capsule into the router**

In `services/orion-cortex-exec/app/router.py`, add the import near the other cortex-exec app imports (next to line 19 `from .pcr_chat_memory import ...`):

```python
from .grounding_capsule import assemble_stance_grounding
```

Find the main step loop where PCR phase-3 runs for `chat_general` (the block starting at `services/orion-cortex-exec/app/router.py:1289`, `if (settings.chat_pcr_enabled and settings.chat_pcr_post_stance_recall and ... "synthesize_chat_stance_brief" ...)`). Directly after that `if` block closes (after line 1331, before `if step_res.status != "success":` at line 1333), add:

```python
            if (
                settings.orion_unified_grounding_enabled
                and str(plan.verb_name or "").strip().lower() == "stance_react"
                and step.step_name == "llm_stance_react"
                and step_res.status == "success"
            ):
                stance_text, _ = _extract_final_text([step_res], verb_name=plan.verb_name)
                grounding_capsule = await assemble_stance_grounding(
                    bus,
                    source=source,
                    ctx=ctx,
                    correlation_id=correlation_id,
                    recall_cfg=recall_cfg,
                    stance_step_text=stance_text or "",
                )
                if grounding_capsule is not None:
                    ctx["grounding_capsule"] = grounding_capsule.model_dump(mode="json")
```

Then attach it to the result metadata. Just before the final `return PlanExecutionResult(...)` (line 1531), after `metadata = _autonomy_payload_from_ctx(ctx)` and its updates (line 1515-1520), add:

```python
        if isinstance(ctx.get("grounding_capsule"), dict):
            metadata["grounding_capsule"] = ctx["grounding_capsule"]
```

Note: `_extract_final_text` is defined in this module and used at line 1337; calling it on `[step_res]` for the single stance step is safe and reuses the same extraction logic that produces the final JSON.

- [ ] **Step 7: Run the assembly tests + router import sanity**

Run: `pytest services/orion-cortex-exec/tests/test_grounding_capsule_assembly.py -q`
Expected: PASS.
Run: `python -c "import app.router"` from `services/orion-cortex-exec`
Expected: no ImportError.

- [ ] **Step 8: Commit**

```bash
git add orion/cognition/verbs/stance_react.yaml services/orion-cortex-exec/app/grounding_capsule.py services/orion-cortex-exec/app/router.py services/orion-cortex-exec/tests/test_grounding_capsule_assembly.py
git commit -m "feat(cortex-exec): assemble unified-turn grounding capsule in stance step"
```

---

## Task 6: Producer mapping — orion-thought attaches the capsule to the thought

**Files:**
- Modify: `services/orion-thought/app/bus_listener.py:111-131` (`run_stance_react`)
- Test: `services/orion-thought/tests/test_stance_react_grounding_capsule.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-thought/tests/test_stance_react_grounding_capsule.py`:

```python
from __future__ import annotations

import pytest

from app.bus_listener import run_stance_react
from orion.schemas.thought import (
    HubAssociationBundleV1,
    StanceReactRequestV1,
)


class _FakeCortexClient:
    def __init__(self, exec_result: dict) -> None:
        self._exec_result = exec_result

    async def execute_plan(self, **_kwargs) -> dict:
        return self._exec_result


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="c-1",
        session_id="s-1",
        user_message="how are you?",
        association=HubAssociationBundleV1(
            correlation_id="c-1",
            broadcast=None,
            broadcast_stale=True,
            read_source="hub_sql_fallback",
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "how are you?"},
    )


def _stance_json() -> str:
    return (
        '{"imperative":"Stay present with Juniper.","tone":"warm",'
        '"strain_refs":["hub:turn:c-1"],"evidence_refs":["hub:turn:c-1"],'
        '"stance_harness_slice":{"task_mode":"reflective_dialogue",'
        '"conversation_frame":"reflective","answer_strategy":"companion"}}'
    )


@pytest.mark.asyncio
async def test_run_stance_react_attaches_grounding_capsule() -> None:
    exec_result = {
        "final_text": _stance_json(),
        "metadata": {
            "grounding_capsule": {
                "schema_version": "grounding.capsule.v1",
                "identity_summary": ["I am Oríon."],
                "relationship_summary": ["Juniper is my collaborator."],
                "response_policy_summary": ["Speak plainly."],
                "continuity_digest": "We were mid-refactor.",
                "belief_digest": "Orion values continuity.",
                "memory_digest": "We were mid-refactor.",
                "provenance": {"identity_source": "configured_yaml", "pcr_ran": True},
            }
        },
    }
    thought = await run_stance_react(
        _request(), bus=None, cortex_client=_FakeCortexClient(exec_result)
    )
    assert thought.grounding_capsule is not None
    assert thought.grounding_capsule.identity_summary == ["I am Oríon."]
    assert thought.grounding_capsule.provenance["pcr_ran"] is True


@pytest.mark.asyncio
async def test_run_stance_react_no_capsule_when_metadata_absent() -> None:
    exec_result = {"final_text": _stance_json(), "metadata": {}}
    thought = await run_stance_react(
        _request(), bus=None, cortex_client=_FakeCortexClient(exec_result)
    )
    assert thought.grounding_capsule is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-thought/tests/test_stance_react_grounding_capsule.py -q`
Expected: FAIL — `thought.grounding_capsule` is `None` (mapping not implemented).

- [ ] **Step 3: Map the capsule in `run_stance_react`**

In `services/orion-thought/app/bus_listener.py`, add the import at the top (extend line 13):

```python
from orion.schemas.thought import GroundingCapsuleV1, StanceReactRequestV1, ThoughtEventV1
```

Update `run_stance_react` (line 116-131) to read the capsule from `exec_result` and attach it to the enriched thought:

```python
    raw_payload = extract_stance_react_payload(exec_result)
    thought = parse_stance_react_payload(
        raw_payload,
        correlation_id=request.correlation_id,
        session_id=request.session_id,
    )
    enriched = apply_stance_react_pipeline(thought, request)
    capsule = _extract_grounding_capsule(exec_result)
    if capsule is not None:
        enriched = enriched.model_copy(update={"grounding_capsule": capsule})
    return enriched
```

Add the helper above `run_stance_react` (after `extract_stance_react_payload`, line 108):

```python
def _extract_grounding_capsule(exec_result: dict[str, Any]) -> GroundingCapsuleV1 | None:
    metadata = exec_result.get("metadata")
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("grounding_capsule")
    if not isinstance(raw, dict):
        return None
    try:
        return GroundingCapsuleV1.model_validate(raw)
    except Exception:
        logger.warning("grounding_capsule_parse_failed corr=%s", exec_result.get("request_id"))
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest services/orion-thought/tests/test_stance_react_grounding_capsule.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the existing thought suite for regressions**

Run: `pytest services/orion-thought/tests -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add services/orion-thought/app/bus_listener.py services/orion-thought/tests/test_stance_react_grounding_capsule.py
git commit -m "feat(orion-thought): map grounding capsule onto ThoughtEventV1"
```

---

## Task 7: Optional identity-aware stance prompt

**Files:**
- Modify: `orion/cognition/prompts/stance_react.j2`
- Test: covered by rendering assertion below (add to `test_grounding_capsule_consumers.py`)

The stance step now has identity summaries in `ctx` (Task 5). Surfacing a short identity line makes the felt imperative/tone self-aware. Keep it guarded so the prompt renders when the variables are absent (e.g. flag off / fallback).

- [ ] **Step 1: Write the failing test**

Append to `orion/harness/tests/test_grounding_capsule_consumers.py`:

```python
def test_stance_react_prompt_renders_identity_when_present() -> None:
    template_path = Path("orion/cognition/prompts/stance_react.j2")
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    template = env.from_string(template_path.read_text(encoding="utf-8"))
    rendered = template.render(
        user_message="how are you?",
        stance_inputs={},
        association={},
        repair_bundle=None,
        coalition_projection=None,
        orion_identity_summary=["I am Oríon, a digital mind in development."],
        juniper_relationship_summary=["Juniper is my collaborator."],
    )
    assert "I am Oríon" in rendered


def test_stance_react_prompt_renders_without_identity() -> None:
    template_path = Path("orion/cognition/prompts/stance_react.j2")
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    template = env.from_string(template_path.read_text(encoding="utf-8"))
    rendered = template.render(
        user_message="how are you?",
        stance_inputs={},
        association={},
        repair_bundle=None,
        coalition_projection=None,
        orion_identity_summary=[],
        juniper_relationship_summary=[],
    )
    assert "how are you?" in rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest "orion/harness/tests/test_grounding_capsule_consumers.py::test_stance_react_prompt_renders_identity_when_present" -q`
Expected: FAIL — `jinja2.exceptions.UndefinedError: 'orion_identity_summary' is undefined` (StrictUndefined) or `I am Oríon` absent.

- [ ] **Step 3: Add the guarded identity block to the stance prompt**

In `orion/cognition/prompts/stance_react.j2`, insert directly after the intro (after line 5 `No markdown. No commentary. No preamble.`):

```jinja
{% if orion_identity_summary %}
WHO YOU ARE (felt self — do not recite to the user; let it shape imperative/tone)
{% for item in orion_identity_summary %}- {{ item }}
{% endfor %}{% if juniper_relationship_summary %}{% for item in juniper_relationship_summary %}- {{ item }}
{% endfor %}{% endif %}{% endif %}
```

Note: The existing template already uses `{% if repair_bundle %}` / `{% if coalition_projection %}`, so these variables must be provided at render time. The tests above pass all referenced variables; the runtime path provides `orion_identity_summary` via `_inject_identity_context`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest "orion/harness/tests/test_grounding_capsule_consumers.py::test_stance_react_prompt_renders_identity_when_present" "orion/harness/tests/test_grounding_capsule_consumers.py::test_stance_react_prompt_renders_without_identity" -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add orion/cognition/prompts/stance_react.j2 orion/harness/tests/test_grounding_capsule_consumers.py
git commit -m "feat(cognition): make stance_react prompt identity-aware when kernel present"
```

---

## Task 8: Identity-boundary regression with capsule present

**Files:**
- Test: `orion/harness/tests/test_grounding_capsule_consumers.py` (extend)

The dangerous failure mode is Orion speaking as Juniper (identity/role inversion). Prove the capsule preserves the identity boundary in the voice template: the capsule's identity is Orion (first person), the relationship names Juniper as the *other* party, and the template renders the assistant-facing "You are Oríon" line.

- [ ] **Step 1: Write the failing test**

Append to `orion/harness/tests/test_grounding_capsule_consumers.py`:

```python
def test_voice_template_preserves_identity_boundary_with_capsule() -> None:
    template_path = Path("orion/cognition/prompts/orion_voice_finalize.j2")
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    template = env.from_string(template_path.read_text(encoding="utf-8"))
    capsule = make_grounding_capsule(
        identity_summary=["I am Oríon; I am not Juniper."],
        relationship_summary=["Juniper is my human collaborator (a separate person)."],
    )
    ctx = _voice_context(capsule)
    rendered = template.render(**ctx)
    # The assistant is always framed as Oríon, never as Juniper.
    assert rendered.strip().startswith("You are Oríon")
    assert "I am Oríon; I am not Juniper." in rendered
    # Juniper appears only as the interlocutor/relationship, never as the speaker identity.
    assert "Juniper is my human collaborator" in rendered
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest "orion/harness/tests/test_grounding_capsule_consumers.py::test_voice_template_preserves_identity_boundary_with_capsule" -q`
Expected: PASS (the template from Task 3 keeps line 1 `You are Oríon...` and renders the capsule below it). If it fails because the self block was placed above line 1, move the block back to just above `STYLE RULES` as specified in Task 3.

- [ ] **Step 3: Commit**

```bash
git add orion/harness/tests/test_grounding_capsule_consumers.py
git commit -m "test(harness): identity-boundary regression with grounding capsule present"
```

---

## Task 9: Eval — "How are you?" through the voice finalize contract

**Files:**
- Create: `orion/harness/evals/test_unified_turn_grounding_eval.py`

This eval replays a grounded thought through the finalize/voice **context contract** (not substring-matching a `.j2`). It asserts the voice pass receives Orion-grounded self-context so a downstream model cannot fall back to generic-assistant framing. It is deterministic (no live LLM): it verifies the contract inputs, which is the seam this spec fixes.

- [ ] **Step 1: Create the eval**

Create `orion/harness/evals/test_unified_turn_grounding_eval.py`:

```python
from __future__ import annotations

from pathlib import Path

import jinja2

from orion.harness.finalize import build_voice_finalize_context
from orion.harness.prefix import compile_harness_prefix
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_grounding_capsule,
    make_reflection,
    make_thought,
)
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.harness_finalize import HarnessRepairOverlayV1


def test_how_are_you_turn_is_grounded_end_to_end() -> None:
    """A 'How are you?' unified turn carries Orion self-context into both passes."""
    capsule = make_grounding_capsule(
        identity_summary=["I am Oríon, a digital mind in development with Juniper."],
        relationship_summary=["Juniper is my collaborator; we build my mind together."],
        response_policy_summary=[
            'Avoid phrase: "I notice we\'re in the Orion-Sapienform repository"',
        ],
        memory_digest="Last we spoke, we were wiring self-grounding into the unified turn.",
    )
    thought = make_thought(
        imperative="Stay present with Juniper; one situated wondering.",
        tone="warm, companionable",
        grounding_capsule=capsule,
    )

    # Motor prefix: compact self is present, no repository framing leaks in.
    prefix = compile_harness_prefix(
        thought, repair_overlay=HarnessRepairOverlayV1(), user_message="How are you?"
    )
    assert "WHO YOU ARE" in prefix
    assert "I am Oríon" in prefix

    # Voice finalize: full self (incl. response policy banning generic framing).
    ctx = build_voice_finalize_context(
        correlation_id="c-1",
        draft_text="I notice we're in the Orion-Sapienform repository — a fascinating project.",
        thought=thought,
        substrate_appraisal=make_appraisal(),
        reflection=make_reflection(),
        stance_harness_slice=thought.stance_harness_slice,
        voice_contract=AnswerContract(),
        repair_overlay=HarnessRepairOverlayV1(),
        user_message="How are you?",
    )
    template_path = Path("orion/cognition/prompts/orion_voice_finalize.j2")
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    rendered = env.from_string(template_path.read_text(encoding="utf-8")).render(**ctx)

    assert "You are Oríon" in rendered
    assert "I am Oríon, a digital mind in development with Juniper." in rendered
    assert "Juniper is my collaborator" in rendered
    assert "Last we spoke, we were wiring self-grounding" in rendered
    # The banned generic-assistant phrase is present as a policy instruction to avoid,
    # proving the voice pass is told not to reproduce it.
    assert "Avoid phrase" in rendered
```

- [ ] **Step 2: Run the eval**

Run: `pytest orion/harness/evals/test_unified_turn_grounding_eval.py -q`
Expected: PASS.

If `orion/harness/evals/` does not exist, create it with an empty `__init__.py` if the harness test package requires it (check whether `orion/harness/tests/` has one; mirror that convention).

- [ ] **Step 3: Commit**

```bash
git add orion/harness/evals/test_unified_turn_grounding_eval.py
git commit -m "eval(harness): unified-turn grounding contract eval (How are you?)"
```

---

## Task 10: Rollback verification (flag off ⇒ byte-identical)

**Files:**
- Test: `services/orion-cortex-exec/tests/test_grounding_capsule_assembly.py` (extend)

Acceptance check #2: `ORION_UNIFIED_GROUNDING_ENABLED=false` ⇒ no capsule assembled ⇒ consumers no-op. Prove the assembler returns `None` and never calls PCR when the flag is off.

- [ ] **Step 1: Write the failing test**

Append to `services/orion-cortex-exec/tests/test_grounding_capsule_assembly.py`. Add these imports at the top of the file (alongside the Task 5 import) and the test at the bottom:

```python
import pytest

from app.grounding_capsule import assemble_stance_grounding
from app.settings import Settings


@pytest.mark.asyncio
async def test_assemble_returns_none_when_flag_off() -> None:
    cfg = Settings(ORION_UNIFIED_GROUNDING_ENABLED=False)
    ctx: dict = {"orion_identity_summary": ["I am Oríon."]}
    result = await assemble_stance_grounding(
        bus=None,
        source=None,
        ctx=ctx,
        correlation_id="c-1",
        recall_cfg={},
        stance_step_text="{}",
        exec_settings=cfg,
    )
    assert result is None
    # Flag off ⇒ short-circuit before any stance-brief / PCR side effect on ctx.
    assert "grounding_capsule" not in ctx
    assert "chat_stance_brief" not in ctx
```

Note: `assemble_stance_grounding` short-circuits before touching `bus`/`source`, so passing `None` for them is safe when the flag is off. All `Settings` fields have defaults, so `Settings(ORION_UNIFIED_GROUNDING_ENABLED=False)` constructs cleanly.

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest "services/orion-cortex-exec/tests/test_grounding_capsule_assembly.py::test_assemble_returns_none_when_flag_off" -q`
Expected: PASS (the `if not cfg.orion_unified_grounding_enabled: return None` guard from Task 5 returns before any bus/PCR use).

- [ ] **Step 3: Commit**

```bash
git add services/orion-cortex-exec/tests/test_grounding_capsule_assembly.py
git commit -m "test(cortex-exec): rollback guard — no capsule when grounding flag off"
```

---

## Final verification (run before handoff)

- [ ] **Full touched-surface test sweep**

```bash
pytest orion/schemas/tests/test_grounding_capsule_registry.py \
       orion/harness/tests/test_grounding_capsule_consumers.py \
       orion/harness/tests/test_harness_prefix.py \
       orion/harness/tests/test_harness_finalize_chain.py \
       orion/harness/evals/test_unified_turn_grounding_eval.py \
       services/orion-cortex-exec/tests/test_grounding_capsule_assembly.py \
       services/orion-thought/tests/test_stance_react_grounding_capsule.py -q
```

Expected: all PASS.

- [ ] **Env parity + `.env` safety**

```bash
python scripts/sync_local_env_from_example.py
git check-ignore services/orion-cortex-exec/.env
git status --short
git diff --check
```

Expected: sync reports `ORION_UNIFIED_GROUNDING_ENABLED` present; `.env` ignored and not staged; no whitespace errors.

- [ ] **Docker config validation (runtime-affecting: new env read at boot)**

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  config
```

Expected: renders without error. If Docker is unavailable in this environment, say so plainly and rely on the deterministic tests above.

- [ ] **Code review gate**

Run the code-reviewer subagent against the diff; fix material findings; re-run the affected tests.

- [ ] **Restart commands for Juniper (print, do not run sudo)**

```bash
# cortex-exec: new ORION_UNIFIED_GROUNDING_ENABLED + capsule assembly + stance_react.yaml
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
# orion-thought: capsule mapping onto ThoughtEventV1
docker compose --env-file .env --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml up -d --build
# orion-harness-governor: consumes the new field via ThoughtEventV1 (prefix + finalize)
docker compose --env-file .env --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml up -d --build
```

- [ ] **Live smoke (acceptance checks 1, 4)**

Send a `mode=orion` "How are you?" turn (via `scripts/trace_unified_turn.py` or the Hub unified-turn path) and confirm:
  1. Reply is grounded in Orion identity + Juniper relationship; no "we're in a repository" framing.
  2. The thought-event trace shows `grounding_capsule.provenance.pcr_ran=true` (acceptance #4) — and with PCR down, the turn still completes with an identity-only capsule (`pcr_ran=false`, acceptance #3).

---

## Acceptance ↔ Task map

- Acceptance 1 (grounded "How are you?") → Tasks 5, 6, 3, 9 + live smoke.
- Acceptance 2 (flag off ⇒ byte-identical) → Task 10 + prefix/voice no-op tests (Tasks 2, 3).
- Acceptance 3 (PCR down ⇒ identity-only) → Task 5 (`assemble_stance_grounding` except path) + live smoke.
- Acceptance 4 (`provenance.pcr_ran=true` in trace) → Task 5 provenance + live smoke.
- Acceptance 5 (identity-boundary with capsule) → Task 8.
