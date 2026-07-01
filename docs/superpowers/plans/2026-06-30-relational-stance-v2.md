# Relational Stance v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix relational chat by injecting a deterministic speech contract near the generation point and persisting regime across follow-up turns via a session cache.

**Architecture:** Add `interaction_regime` + `companion_closing_move` to `ChatStanceBrief`; compile a flat natural-language `speech_contract` string after enforce; inject it into `chat_general.j2` immediately before `TASK`; seed the stance synthesizer with the prior turn's regime via an ephemeral in-process session cache keyed by `session_id`.

**Tech Stack:** Python 3.12, Pydantic v2, Jinja2, pytest-asyncio; all changes in `orion-cortex-exec` and shared `orion/` schemas/prompts.

---

## File map

| File | Role |
|---|---|
| `orion/schemas/chat_stance.py` | Add `interaction_regime`, `companion_closing_move` fields |
| `services/orion-cortex-exec/app/chat_stance.py` | Add `compile_speech_contract`, `_inject_prior_stance_to_inputs` |
| `services/orion-cortex-exec/app/executor.py` | Session cache helpers; pre-render prior-load hook; post-enforce store + compile call; update import |
| `orion/cognition/prompts/chat_stance_brief.j2` | Add `prior_stance` input, carryforward rule, new output fields |
| `orion/cognition/prompts/chat_general.j2` | Add `TURN CONTRACT` block before `TASK` |
| `services/orion-cortex-exec/tests/test_chat_relational_stance.py` | New unit tests for schema + compiler + cache + prior wiring |
| `services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py` | TURN CONTRACT position regression |

---

## Task 1 — Schema fields

**Files:**
- Modify: `orion/schemas/chat_stance.py`
- Test: `services/orion-cortex-exec/tests/test_chat_relational_stance.py`

- [ ] **Step 1.1: Write failing tests**

Add to `test_chat_relational_stance.py` (after the last test):

```python
def test_chat_stance_brief_new_fields_default_none() -> None:
    brief = _relational_brief()
    assert brief.interaction_regime is None
    assert brief.companion_closing_move is None


def test_chat_stance_brief_new_fields_roundtrip() -> None:
    brief = _relational_brief(
        interaction_regime="relational",
        companion_closing_move="end_with_a_wondering",
    )
    d = brief.model_dump(mode="json")
    assert d["interaction_regime"] == "relational"
    assert d["companion_closing_move"] == "end_with_a_wondering"
    restored = ChatStanceBrief(**d)
    assert restored.interaction_regime == "relational"
    assert restored.companion_closing_move == "end_with_a_wondering"
```

- [ ] **Step 1.2: Run to verify failure**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  -k "test_chat_stance_brief_new_fields" -q --tb=short
```

Expected: FAIL — `unexpected keyword argument 'interaction_regime'`

- [ ] **Step 1.3: Add fields to schema**

In `orion/schemas/chat_stance.py`, after the `stance_summary` field (line 52):

```python
    interaction_regime: Literal["instrumental", "relational", "minimal"] | None = Field(default=None)
    companion_closing_move: str | None = Field(default=None)
```

The top of the file already has `from typing import Literal` — no new imports needed.

- [ ] **Step 1.4: Run to verify pass**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  -k "test_chat_stance_brief_new_fields" -q --tb=short
```

Expected: 2 passed

- [ ] **Step 1.5: Confirm no regressions in full relational stance suite**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py -q --tb=short
```

Expected: all pass (10 + 2 = 12 passed)

- [ ] **Step 1.6: Commit**

```bash
git add orion/schemas/chat_stance.py \
        services/orion-cortex-exec/tests/test_chat_relational_stance.py
git commit -m "feat(schema): add interaction_regime and companion_closing_move to ChatStanceBrief"
```

---

## Task 2 — `compile_speech_contract`

**Files:**
- Modify: `services/orion-cortex-exec/app/chat_stance.py`
- Test: `services/orion-cortex-exec/tests/test_chat_relational_stance.py`

- [ ] **Step 2.1: Write failing tests**

**2.1a** — Update the import block at the **top** of `test_chat_relational_stance.py` (replace the existing `from app.chat_stance import` line):

```python
from app.chat_stance import (
    compile_speech_contract,
    enforce_chat_stance_quality,
    strip_identity_recital_leadin,
    strip_transactional_closers,
)
```

**2.1b** — Add `_instrumental_brief` immediately after the existing `_relational_brief` helper (not at end of file):

```python
def _instrumental_brief(**overrides) -> ChatStanceBrief:
    base = {
        "conversation_frame": "mixed",
        "task_mode": "direct_response",
        "identity_salience": "low",
        "user_intent": "direct question",
        "self_relevance": "answer",
        "juniper_relevance": "practical",
        "answer_strategy": "direct",
        "stance_summary": "short",
    }
    base.update(overrides)
    return ChatStanceBrief(**base)
```

**2.1c** — Append the following tests at the end of the file:

```python
def test_compile_speech_contract_relational_with_closing_move() -> None:
    brief = _relational_brief(
        interaction_regime="relational",
        companion_closing_move="end_with_a_wondering",
    )
    contract = compile_speech_contract(brief)
    assert "companion turn" in contract
    assert "wondering" in contract
    assert "let me know" not in contract.lower()
    assert "if you need" not in contract.lower()


def test_compile_speech_contract_relational_default_no_move() -> None:
    brief = _relational_brief(
        interaction_regime="relational",
        response_priorities=["companion_presence", "hold_space"],
    )
    contract = compile_speech_contract(brief)
    assert "companion turn" in contract
    assert "next steps" in contract or "support closers" in contract


def test_compile_speech_contract_relational_situated_curiosity() -> None:
    brief = _relational_brief(
        interaction_regime="relational",
        response_priorities=["companion_presence", "situated_curiosity"],
    )
    contract = compile_speech_contract(brief)
    assert "grounded question" in contract
    assert "generic reversal" in contract


def test_compile_speech_contract_minimal() -> None:
    brief = _instrumental_brief(interaction_regime="minimal")
    contract = compile_speech_contract(brief)
    assert "short" in contract
    assert "replying" in contract


def test_compile_speech_contract_instrumental_direct() -> None:
    brief = _instrumental_brief(interaction_regime="instrumental")
    contract = compile_speech_contract(brief)
    assert "directly" in contract
    assert "blocker" not in contract


def test_compile_speech_contract_instrumental_triage() -> None:
    brief = _instrumental_brief(interaction_regime="instrumental", task_mode="triage")
    contract = compile_speech_contract(brief)
    assert "blocker" in contract


def test_compile_speech_contract_derives_regime_from_task_mode() -> None:
    brief = _relational_brief(interaction_regime=None)
    contract = compile_speech_contract(brief)
    assert "companion turn" in contract


def test_compile_speech_contract_all_closing_moves() -> None:
    moves = {
        "end_with_a_wondering": "wondering",
        "leave_space_without_offer": "offer",
        "ground_observation": "observation",
        "be_with_silence": "silence",
    }
    for move, expected_word in moves.items():
        brief = _relational_brief(interaction_regime="relational", companion_closing_move=move)
        contract = compile_speech_contract(brief)
        assert expected_word in contract, f"move={move!r}: expected {expected_word!r} in {contract!r}"
```

- [ ] **Step 2.2: Run to verify failure**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  -k "compile_speech_contract" -q --tb=short
```

Expected: FAIL — `cannot import name 'compile_speech_contract'`

- [ ] **Step 2.3: Add `_COMPANION_CLOSING_MOVE_MAP` and `compile_speech_contract` to `chat_stance.py`**

Find the line that contains `def strip_identity_recital_leadin` (line ~1041). Insert the following block **immediately before** that function:

```python
_COMPANION_CLOSING_MOVE_MAP: dict[str, str] = {
    "end_with_a_wondering": "End with a wondering, not an offer.",
    "leave_space_without_offer": "Leave space. Do not close with an offer to help.",
    "ground_observation": "End with a grounded observation from the thread.",
    "be_with_silence": "Hold the silence. No closing move required.",
}


def compile_speech_contract(brief: "ChatStanceBrief") -> str:
    """Deterministic regime-specific contract injected near TASK in chat_general.j2.

    Pure Python — no LLM, no I/O. Called after enforce_chat_stance_quality.
    """
    regime = brief.interaction_regime

    if regime is None:
        if brief.task_mode in _RELATIONAL_TASK_MODES or brief.conversation_frame in _RELATIONAL_CONVERSATION_FRAMES:
            regime = "relational"
        else:
            regime = "instrumental"

    if regime == "minimal":
        return (
            "Keep this reply very short. Do not ask questions. "
            "Release Juniper from replying — offer voice, a pause, or continuation later."
        )

    if regime == "relational":
        parts = ["This is a companion turn."]
        move = brief.companion_closing_move
        if move and move in _COMPANION_CLOSING_MOVE_MAP:
            parts.append(_COMPANION_CLOSING_MOVE_MAP[move])
        else:
            parts.append("Stay present; do not offer next steps, trackers, or support closers.")
        if "situated_curiosity" in list(brief.response_priorities or []):
            parts.append("Ask one grounded question from this thread — not a generic reversal.")
        return " ".join(parts)

    # instrumental (default)
    parts = ["Answer directly."]
    if brief.task_mode == "triage":
        parts.append("Lead with the operational blocker.")
    return " ".join(parts)
```

Note: `ChatStanceBrief` is already imported at the top of the file; the string annotation avoids any forward-reference issue.

- [ ] **Step 2.4: Run to verify pass**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  -k "compile_speech_contract or new_fields" -q --tb=short
```

Expected: all new tests pass

- [ ] **Step 2.5: Full suite non-regression**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py \
  -q --tb=short
```

Expected: all pass

- [ ] **Step 2.6: Commit**

```bash
git add services/orion-cortex-exec/app/chat_stance.py \
        services/orion-cortex-exec/tests/test_chat_relational_stance.py
git commit -m "feat(stance): add compile_speech_contract — deterministic regime contract compiler"
```

---

## Task 3 — Prior stance session cache

**Files:**
- Modify: `services/orion-cortex-exec/app/executor.py`
- Test: `services/orion-cortex-exec/tests/test_chat_relational_stance.py`

- [ ] **Step 3.1: Write failing tests**

**3.1a** — Add to the import block at the **top** of `test_chat_relational_stance.py`:

```python
from app.executor import _prior_stance_cache_get, _prior_stance_cache_set
```

**3.1b** — Append the following tests at the end of the file:


def test_prior_stance_cache_set_and_get() -> None:
    _prior_stance_cache_set("sess-abc", {"interaction_regime": "relational", "task_mode": "reflective_dialogue"})
    result = _prior_stance_cache_get("sess-abc")
    assert result is not None
    assert result["interaction_regime"] == "relational"


def test_prior_stance_cache_returns_none_for_missing_key() -> None:
    result = _prior_stance_cache_get("sess-does-not-exist-xyz")
    assert result is None


def test_prior_stance_cache_evicts_expired_entry(monkeypatch) -> None:
    import app.executor as _exec
    # Override TTL to 0 to force immediate expiry
    monkeypatch.setattr(_exec, "_PRIOR_STANCE_TTL_SECONDS", 0)
    _prior_stance_cache_set("sess-ttl-test", {"interaction_regime": "relational"})
    import time as _t; _t.sleep(0.01)
    result = _prior_stance_cache_get("sess-ttl-test")
    assert result is None
```

- [ ] **Step 3.2: Run to verify failure**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  -k "prior_stance_cache" -q --tb=short
```

Expected: FAIL — `cannot import name '_prior_stance_cache_get'`

- [ ] **Step 3.3: Add session cache to executor.py**

In `executor.py`, after the `logger = logging.getLogger(...)` line (~line 85) and before `_SYSTEM_TELEMETRY_KEYS`, add:

```python
import threading as _threading

_PRIOR_STANCE_CACHE: dict[str, tuple[float, dict]] = {}
_PRIOR_STANCE_LOCK = _threading.Lock()
_PRIOR_STANCE_TTL_SECONDS: float = 1800.0  # 30 minutes; mutable for tests


def _prior_stance_cache_set(session_id: str, summary: dict) -> None:
    """Store compact brief summary keyed by session_id for next-turn carryforward."""
    with _PRIOR_STANCE_LOCK:
        _PRIOR_STANCE_CACHE[session_id] = (time.monotonic(), dict(summary))


def _prior_stance_cache_get(session_id: str) -> dict | None:
    """Return stored brief summary if present and not expired; evict on expiry."""
    with _PRIOR_STANCE_LOCK:
        entry = _PRIOR_STANCE_CACHE.get(session_id)
        if entry is None:
            return None
        ts, summary = entry
        if time.monotonic() - ts > _PRIOR_STANCE_TTL_SECONDS:
            del _PRIOR_STANCE_CACHE[session_id]
            return None
        return dict(summary)
```

(`time` is already imported at line 14 of executor.py.)

- [ ] **Step 3.4: Run to verify pass**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  -k "prior_stance_cache" -q --tb=short
```

Expected: 3 passed

- [ ] **Step 3.5: Commit**

```bash
git add services/orion-cortex-exec/app/executor.py \
        services/orion-cortex-exec/tests/test_chat_relational_stance.py
git commit -m "feat(exec): add prior stance session cache for inter-turn regime persistence"
```

---

## Task 4 — `_inject_prior_stance_to_inputs` + `build_chat_stance_inputs` wiring

**Files:**
- Modify: `services/orion-cortex-exec/app/chat_stance.py`
- Test: `services/orion-cortex-exec/tests/test_chat_relational_stance.py`

- [ ] **Step 4.1: Write failing test**

**4.1a** — Add to the import block at the **top** of `test_chat_relational_stance.py`:

```python
from app.chat_stance import _inject_prior_stance_to_inputs
```

**4.1b** — Append the following tests at the end of the file:


def test_inject_prior_stance_to_inputs_when_present() -> None:
    ctx = {"prior_chat_stance_brief": {"interaction_regime": "relational", "task_mode": "reflective_dialogue"}}
    inputs: dict = {}
    _inject_prior_stance_to_inputs(ctx, inputs)
    assert inputs.get("prior_stance") == ctx["prior_chat_stance_brief"]


def test_inject_prior_stance_to_inputs_noop_when_absent() -> None:
    ctx: dict = {}
    inputs: dict = {}
    _inject_prior_stance_to_inputs(ctx, inputs)
    assert "prior_stance" not in inputs


def test_inject_prior_stance_to_inputs_noop_for_empty_dict() -> None:
    ctx = {"prior_chat_stance_brief": {}}
    inputs: dict = {}
    _inject_prior_stance_to_inputs(ctx, inputs)
    assert "prior_stance" not in inputs
```

- [ ] **Step 4.2: Run to verify failure**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  -k "inject_prior_stance" -q --tb=short
```

Expected: FAIL — `cannot import name '_inject_prior_stance_to_inputs'`

- [ ] **Step 4.3: Add helper to `chat_stance.py`**

Find the line `def _project_concept_from_beliefs` (~line 1070 after Task 2's insertion). Insert the following function **immediately before** `build_chat_stance_inputs` (the function definition at line ~2107 after insertions):

```python
def _inject_prior_stance_to_inputs(ctx: Dict[str, Any], inputs: Dict[str, Any]) -> None:
    """Copy prior brief summary from ctx into stance inputs when present and non-empty."""
    prior = ctx.get("prior_chat_stance_brief")
    if isinstance(prior, dict) and prior:
        inputs["prior_stance"] = prior
```

- [ ] **Step 4.4: Wire into `build_chat_stance_inputs`**

Find the line `ctx["chat_stance_inputs"] = inputs` (line ~2197). Insert the call **immediately before** it:

```python
    _inject_prior_stance_to_inputs(ctx, inputs)
    ctx["chat_stance_inputs"] = inputs
```

- [ ] **Step 4.5: Run to verify pass**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  -k "inject_prior_stance" -q --tb=short
```

Expected: 3 passed

- [ ] **Step 4.6: Full suite**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/ -q --tb=short
```

Expected: all pass

- [ ] **Step 4.7: Commit**

```bash
git add services/orion-cortex-exec/app/chat_stance.py \
        services/orion-cortex-exec/tests/test_chat_relational_stance.py
git commit -m "feat(stance): wire prior_stance into build_chat_stance_inputs for regime carryforward"
```

---

## Task 5 — Prompt updates

**Files:**
- Modify: `orion/cognition/prompts/chat_stance_brief.j2`
- Modify: `orion/cognition/prompts/chat_general.j2`
- Test: `services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py`
- Test: `services/orion-cortex-exec/tests/test_chat_relational_stance.py`

- [ ] **Step 5.1: Write failing plumbing tests**

Add to `test_chat_general_stance_plumbing.py`:

```python
def test_turn_contract_block_appears_before_task() -> None:
    prompt = Path("orion/cognition/prompts/chat_general.j2").read_text(encoding="utf-8")
    assert "{% if speech_contract %}" in prompt
    assert "TURN CONTRACT" in prompt
    contract_pos = prompt.find("TURN CONTRACT")
    task_pos = prompt.find("\nTASK\n")
    assert task_pos > 0, "TASK section not found"
    assert contract_pos < task_pos, "TURN CONTRACT must appear before TASK"


def test_turn_contract_block_absent_when_speech_contract_falsy() -> None:
    """Jinja block must be guarded — no TURN CONTRACT emitted when speech_contract is not set."""
    from jinja2 import Template
    tmpl = Template(Path("orion/cognition/prompts/chat_general.j2").read_text(encoding="utf-8"))
    rendered = tmpl.render(
        user_message="test",
        message_history="",
        memory_digest="",
        orion_identity_summary=[],
        juniper_relationship_summary=[],
        response_policy_summary=[],
        chat_stance_brief="{}",
        chat_attention_frame=None,
        situation_prompt_fragment=None,
        world_context_capsule=None,
        menu_topic_selection=None,
        speech_contract=None,
    )
    assert "TURN CONTRACT" not in rendered
```

Add to `test_chat_relational_stance.py`:

```python
def test_stance_brief_prompt_has_prior_stance_and_regime_fields() -> None:
    prompt = Path("orion/cognition/prompts/chat_stance_brief.j2").read_text(encoding="utf-8")
    assert "prior_stance" in prompt
    assert "interaction_regime" in prompt
    assert "companion_closing_move" in prompt
    assert "carry" in prompt.lower() or "carryforward" in prompt.lower() or "carry forward" in prompt.lower()
```

- [ ] **Step 5.2: Run to verify failure**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py \
  -k "turn_contract" -q --tb=short
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  -k "prior_stance_and_regime" -q --tb=short
```

Expected: FAIL — strings not found

- [ ] **Step 5.3: Update `chat_stance_brief.j2`**

**Change 1:** In the SOURCES block (after the `{% if chat_situation_summary %}` block, around line 25–27), add:

```jinja
{% if prior_stance %}
- prior_stance: {{ prior_stance }}
{% endif %}
```

**Change 2:** In SCHEMA LITERALS (after the `- List-shaped fields...` line, around line 33), add:

```
- interaction_regime must be one of: instrumental, relational, minimal, or null (when unclear)
- companion_closing_move must be one of: end_with_a_wondering, leave_space_without_offer, ground_observation, be_with_silence, or null; set when connection_seek is high
```

**Change 3:** Under INTERACTION POSTURE ASSESSMENT, after the `Do not infer avoid_open_ended_questions` line (~line 89), add:

```
{% if prior_stance %}
- If prior_stance.interaction_regime is "relational" and this turn continues the same emotional thread (venting, presence-seeking, recovery, companionship) without a clear pivot to task or technical work, carry interaction_regime="relational" forward. Do not re-infer as instrumental just because the explicit companion invite is absent from this message. A task pivot (explicit technical question, deploy/debug/restart request) overrides.
{% endif %}
- When connection_seek is high, set interaction_regime="relational" and companion_closing_move to the most fitting value.
- When interface_cost is high and connection_seek is low, set interaction_regime="minimal".
- Otherwise set interaction_regime="instrumental".
```

**Change 4:** In the `Return JSON with exactly these keys` block (around line 101–124), add `interaction_regime` and `companion_closing_move` to the list. Insert after `stance_summary`:

```
interaction_regime
companion_closing_move
```

- [ ] **Step 5.4: Update `chat_general.j2`**

Find the line `TASK` (the section header, not inside a code block). Insert the following block **immediately before** the blank line that precedes `TASK`:

```jinja
{% if speech_contract %}
TURN CONTRACT
{{ speech_contract }}

{% endif %}
```

The file currently has the `TASK` section near line 111. After insertion it should read:

```
{% if speech_contract %}
TURN CONTRACT
{{ speech_contract }}

{% endif %}
TASK
Produce exactly one user-facing reply.
```

- [ ] **Step 5.5: Run to verify pass**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py \
  services/orion-cortex-exec/tests/test_chat_relational_stance.py \
  -q --tb=short
```

Expected: all pass

- [ ] **Step 5.6: Commit**

```bash
git add orion/cognition/prompts/chat_stance_brief.j2 \
        orion/cognition/prompts/chat_general.j2 \
        services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py \
        services/orion-cortex-exec/tests/test_chat_relational_stance.py
git commit -m "feat(prompts): add prior_stance input + regime fields to stance brief; TURN CONTRACT to speech prompt"
```

---

## Task 6 — Executor wiring

**Files:**
- Modify: `services/orion-cortex-exec/app/executor.py`

This task has no new tests — it wires together the pieces from Tasks 2–5 in the executor's existing step-processing loop.

- [ ] **Step 6.1: Update the import block in executor.py**

Find the `from .chat_stance import (` block (~lines 73–81). Add `compile_speech_contract` to it:

```python
from .chat_stance import (
    build_chat_stance_debug_payload,
    build_chat_stance_inputs,
    compile_speech_contract,
    enforce_chat_stance_quality,
    fallback_chat_stance_brief,
    identity_kernel_with_fallbacks,
    parse_chat_stance_brief_with_debug,
    suppress_chat_general_speech_identity_priming,
)
```

- [ ] **Step 6.2: Add pre-render hook to load prior stance**

Find the block at ~line 2547–2550:

```python
            ctx["message_history"] = _format_message_history_for_chat_prompt(ctx.get("messages"))
            if step.verb_name == "chat_general" and step.step_name == "llm_chat_general":
                if suppress_chat_general_speech_identity_priming(ctx):
                    logs.append("info <- suppressed identity kernel priming for ordinary chat_general speech turn")
```

Insert **after** `ctx["message_history"] = ...` and **before** the `if step.verb_name == "chat_general" and step.step_name == "llm_chat_general":` block:

```python
            if step.verb_name == "chat_general" and step.step_name == "synthesize_chat_stance_brief":
                _session_id = str(ctx.get("session_id") or "")
                if _session_id:
                    _prior = _prior_stance_cache_get(_session_id)
                    if _prior:
                        ctx["prior_chat_stance_brief"] = _prior
```

- [ ] **Step 6.3: Add post-enforce store + compile call**

Find the block at ~line 3806 (immediately after `ctx["chat_stance_brief"] = parsed_brief.model_dump(mode="json")`):

```python
                    ctx["chat_stance_brief"] = parsed_brief.model_dump(mode="json")
                    quality_modified = synthesized_brief != ctx["chat_stance_brief"]
```

Insert between these two lines:

```python
                    ctx["chat_stance_brief"] = parsed_brief.model_dump(mode="json")
                    _session_id = str(ctx.get("session_id") or "")
                    if _session_id:
                        _prior_stance_cache_set(_session_id, {
                            "interaction_regime": parsed_brief.interaction_regime,
                            "task_mode": parsed_brief.task_mode,
                            "response_priorities": list(parsed_brief.response_priorities[:4]),
                            "response_hazards": list(parsed_brief.response_hazards[:4]),
                        })
                    ctx["speech_contract"] = compile_speech_contract(parsed_brief)
                    quality_modified = synthesized_brief != ctx["chat_stance_brief"]
```

- [ ] **Step 6.4: Compile-check executor.py**

```bash
cd /mnt/scripts/Orion-Sapienform
./orion_dev/bin/python -c "import services.orion-cortex-exec.app.executor" 2>&1 || \
  PYTHONPATH=. ./orion_dev/bin/python -m compileall \
  services/orion-cortex-exec/app/executor.py -q
```

Expected: no syntax errors

- [ ] **Step 6.5: Full test suite**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/ -q --tb=short
```

Expected: all pass

- [ ] **Step 6.6: Commit**

```bash
git add services/orion-cortex-exec/app/executor.py
git commit -m "feat(exec): wire prior stance load/store and compile_speech_contract into chat_general path"
```

---

## Task 7 — Final verification

- [ ] **Step 7.1: Run full cortex-exec test suite**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/ -q --tb=short
```

Expected: all pass, no regressions

- [ ] **Step 7.2: Verify prompt rendering end-to-end**

```bash
cd /mnt/scripts/Orion-Sapienform
python3 - <<'EOF'
from jinja2 import Template
import json
from pathlib import Path

brief = {
    "conversation_frame": "reflective",
    "task_mode": "reflective_dialogue",
    "identity_salience": "low",
    "user_intent": "Companion presence during recovery.",
    "self_relevance": "Hold space.",
    "juniper_relevance": "Relational continuity matters.",
    "active_relationship_facets": ["companionship"],
    "active_identity_facets": [],
    "response_priorities": ["companion_presence", "situated_curiosity", "hold_space"],
    "response_hazards": ["avoid_transactional_closers", "avoid_next_steps"],
    "answer_strategy": "Stay present.",
    "stance_summary": "Companion mode.",
    "interaction_regime": "relational",
    "companion_closing_move": "end_with_a_wondering",
}
tmpl = Template(Path("orion/cognition/prompts/chat_general.j2").read_text())
rendered = tmpl.render(
    user_message="thanks just hard",
    message_history="[user]: shoulder to talk\n[orion]: I am here.",
    memory_digest="",
    orion_identity_summary=[],
    juniper_relationship_summary=[],
    response_policy_summary=[],
    chat_stance_brief=json.dumps(brief),
    chat_attention_frame=None,
    situation_prompt_fragment=None,
    world_context_capsule=None,
    menu_topic_selection=None,
    speech_contract="This is a companion turn. End with a wondering, not an offer. Ask one grounded question from this thread — not a generic reversal.",
)
contract_pos = rendered.find("TURN CONTRACT")
task_pos = rendered.find("\nTASK\n")
total = len(rendered)
print(f"Total chars: {total} (~{total//4} tokens)")
print(f"TURN CONTRACT at char {contract_pos} (~token {contract_pos//4})")
print(f"TASK at char {task_pos} (~token {task_pos//4})")
print(f"Gap before generation: ~{(total - contract_pos)//4} tokens")
assert contract_pos > 0 and task_pos > 0 and contract_pos < task_pos
print("OK: TURN CONTRACT appears before TASK")
EOF
```

Expected output includes `TURN CONTRACT at char ~1900+` and `TURN CONTRACT appears before TASK`.

- [ ] **Step 7.3: Push branch and open PR**

```bash
cd /mnt/scripts/Orion-Sapienform
git push origin main
```

Then rebuild `orion-cortex-exec` (prompts are baked into the image):

```bash
docker compose \
  --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  build --no-cache && \
docker compose \
  --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  up -d
```

- [ ] **Step 7.4: Live acceptance check**

Send two chat turns:
1. "just looking for a shoulder to talk. Can you help keep my mind off things?"
2. "thanks, it's just hard… we have to be out by 5:30am and nurses are in and out all night."

**Pass criteria:**
- Turn 2 reply contains no "let me know if you need anything" or similar support closer
- Turn 2 reply ends with a grounded wondering or observation, not an offer
- Executor logs show `chat_stance_brief_quality_guard` with `frame=reflective` on turn 2
- `ctx["speech_contract"]` in logs (if debug logging enabled) shows "This is a companion turn."
