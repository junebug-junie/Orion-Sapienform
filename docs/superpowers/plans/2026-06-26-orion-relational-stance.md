# Orion Relational Stance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `chat_general` from crushing relational turns into transactional "all business" replies by fixing the stance post-processor and aligning prompt contracts — without keyword ontologies.

**Architecture:** The stance LLM (`chat_stance_brief.j2`) already infers turn meaning semantically. Python `enforce_chat_stance_quality` currently overwrites relational output. v1 exempts relational `task_mode`/`conversation_frame` from business compression, expands stance-brief vocabulary for `interface_cost` vs `connection_seek`, and makes the speech pass follow stance curiosity on relational turns. Tests replay synthetic briefs through enforce.

**Tech Stack:** Python 3, Pydantic (`ChatStanceBrief`), Jinja2 prompts, pytest via `./scripts/test_service.sh orion-cortex-exec`

**Spec:** `docs/superpowers/specs/2026-06-26-orion-relational-stance-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| `services/orion-cortex-exec/app/chat_stance.py` | Add relational detection helper; skip business compressor on relational briefs |
| `services/orion-cortex-exec/tests/test_chat_relational_stance.py` | TDD fixtures for enforce + prompt contracts |
| `services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py` | Update low-bandwidth prompt asserts for renamed section |
| `orion/cognition/prompts/chat_stance_brief.j2` | Replace monolithic LOW-BANDWIDTH block with interface_cost / connection_seek dimensions |
| `orion/cognition/prompts/chat_general.j2` | Relational turns: stance wins curiosity; transactional closer hazards |
| `orion/cognition/personality/orion_identity.yaml` | Scope clarifying-questions rule; add transactional banned phrases |

Already done (no tasks): `.cursor/rules/conversational-behavior-anti-slop.mdc`, `AGENTS.md` guardrail section.

---

### Task 1: Failing tests for relational enforce exemption

**Files:**
- Create: `services/orion-cortex-exec/tests/test_chat_relational_stance.py`

- [ ] **Step 1: Write the failing test file**

```python
from __future__ import annotations

from pathlib import Path

from app.chat_stance import enforce_chat_stance_quality
from orion.schemas.chat_stance import ChatStanceBrief


def _relational_brief(**overrides) -> ChatStanceBrief:
    base = {
        "conversation_frame": "reflective",
        "task_mode": "reflective_dialogue",
        "identity_salience": "low",
        "user_intent": "Companion presence; mind off recovery.",
        "self_relevance": "Hold space; be curious.",
        "juniper_relevance": "Relational continuity matters this turn.",
        "active_relationship_facets": ["shared_history", "companionship"],
        "active_identity_facets": [],
        "active_growth_axes": [],
        "social_posture": ["presence"],
        "reflective_themes": ["recovery"],
        "active_tensions": ["existential"],
        "dream_motifs": [],
        "response_priorities": [
            "companion_presence",
            "situated_curiosity",
            "hold_space",
            "no_solutioning",
        ],
        "response_hazards": [
            "avoid_task_tracking",
            "avoid_next_steps",
            "avoid_transactional_closers",
        ],
        "answer_strategy": "RelationalHoldSpace",
        "stance_summary": "Be present; ask one situated question; do not solution.",
    }
    base.update(overrides)
    return ChatStanceBrief.model_validate(base)


def test_enforce_preserves_relational_brief_on_non_identity_turn() -> None:
    brief = _relational_brief()
    ctx = {"user_message": "just be curious about it and take my mind off recovery"}

    enriched, _ = enforce_chat_stance_quality(brief, ctx)

    assert enriched.task_mode == "reflective_dialogue"
    assert enriched.conversation_frame == "reflective"
    assert "shared_history" in enriched.active_relationship_facets
    assert enriched.juniper_relevance == "Relational continuity matters this turn."
    assert "companion_presence" in enriched.response_priorities
    assert "avoid_identity_recital" not in enriched.response_priorities
    assert "identity_recital_on_ordinary_turn" not in enriched.response_hazards


def test_enforce_still_compresses_instrumental_direct_response() -> None:
    brief = ChatStanceBrief.model_validate(
        {
            "conversation_frame": "mixed",
            "task_mode": "direct_response",
            "identity_salience": "low",
            "user_intent": "Quick ack.",
            "self_relevance": "x",
            "juniper_relevance": "y",
            "active_relationship_facets": ["juniper_builder"],
            "response_priorities": ["answer_directly_first"],
            "response_hazards": [],
            "answer_strategy": "DirectAnswer",
            "stance_summary": "short",
        }
    )
    ctx = {"user_message": "hey"}

    enriched, _ = enforce_chat_stance_quality(brief, ctx)

    assert enriched.juniper_relevance == "Prioritize practical usefulness over relationship labels."
    assert enriched.active_relationship_facets == []
    assert "avoid_identity_recital" in enriched.response_priorities


def test_enforce_preserves_playful_exchange_frame() -> None:
    brief = _relational_brief(
        conversation_frame="playful_relational",
        task_mode="playful_exchange",
    )
    ctx = {"user_message": "someone to talk. im lonely"}

    enriched, _ = enforce_chat_stance_quality(brief, ctx)

    assert enriched.task_mode == "playful_exchange"
    assert enriched.active_relationship_facets


def test_stance_brief_prompt_has_connection_seek_vocabulary() -> None:
    prompt = Path("orion/cognition/prompts/chat_stance_brief.j2").read_text(encoding="utf-8")
    assert "interface_cost" in prompt
    assert "connection_seek" in prompt
    assert "Do not use keyword matching" in prompt
    assert "companion_presence" in prompt
    assert "LOW-BANDWIDTH / EMBODIED INTERACTION ASSESSMENT" not in prompt


def test_speech_prompt_relational_curiosity_overrides_attention_frame() -> None:
    prompt = Path("orion/cognition/prompts/chat_general.j2").read_text(encoding="utf-8")
    assert "reflective_dialogue" in prompt
    assert "playful_exchange" in prompt
    assert "advisory" in prompt.lower()
    assert "avoid_transactional_closers" in prompt or "avoid_next_steps" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run from repo root:

```bash
./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_chat_relational_stance.py -v
```

Expected: FAIL — `test_enforce_preserves_relational_brief_on_non_identity_turn` (relationship facets stripped, juniper_relevance overwritten). Prompt contract tests may also FAIL until Tasks 3–5.

- [ ] **Step 3: Commit failing tests**

```bash
git add services/orion-cortex-exec/tests/test_chat_relational_stance.py
git commit -m "test: add relational stance enforce regression fixtures"
```

---

### Task 2: Exempt relational modes in enforce_chat_stance_quality

**Files:**
- Modify: `services/orion-cortex-exec/app/chat_stance.py` (near `_ALLOWED_TASK_MODES` ~523 and `enforce_chat_stance_quality` ~2470)

- [ ] **Step 1: Add helper and constants after `_ALLOWED_TASK_MODES`**

```python
_RELATIONAL_TASK_MODES = frozenset({"reflective_dialogue", "playful_exchange"})
_RELATIONAL_CONVERSATION_FRAMES = frozenset({"reflective", "playful_relational"})


def _is_relational_stance_brief(brief: ChatStanceBrief) -> bool:
    return (
        brief.task_mode in _RELATIONAL_TASK_MODES
        or brief.conversation_frame in _RELATIONAL_CONVERSATION_FRAMES
    )
```

- [ ] **Step 2: Wrap the business compressor block in `enforce_chat_stance_quality`**

Replace the block starting at `if not identity_turn:` (line ~2470) with:

```python
    if not identity_turn and not _is_relational_stance_brief(merged):
        if merged.task_mode != "identity_dialogue":
            merged.identity_salience = "low"
        _fallback_identity_boilerplate = frozenset(
            {"continuity", "juniper_builder", "known_person", "avoid_generic_assistant"}
        )

        def _is_identity_boilerplate_facet(facet: str) -> bool:
            normalized = _normalize_brief_phrase(facet)
            if normalized in _fallback_identity_boilerplate:
                return True
            lowered = normalized.lower()
            return any(token in lowered for token in ("orion", "oríon", "juniper"))

        merged.active_identity_facets = [
            f for f in merged.active_identity_facets if not _is_identity_boilerplate_facet(f)
        ]
        merged.active_relationship_facets = [
            f for f in merged.active_relationship_facets if not _is_identity_boilerplate_facet(f)
        ]
        if merged.identity_salience == "low":
            merged.active_identity_facets = []
            merged.active_relationship_facets = []
        merged.self_relevance = "Answer the latest message directly without identity preamble."
        merged.juniper_relevance = "Prioritize practical usefulness over relationship labels."
        merged.response_priorities = _unique(
            list(merged.response_priorities)
            + ["avoid_identity_recital", "preserve_continuity_without_labels"],
            limit=8,
        )
        merged.response_hazards = _unique(
            list(merged.response_hazards) + ["identity_recital_on_ordinary_turn"],
            limit=8,
        )
```

Relational briefs skip this entire block; triage handling above (lines ~2458–2468) still runs.

- [ ] **Step 3: Run enforce tests**

```bash
./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_chat_relational_stance.py::test_enforce_preserves_relational_brief_on_non_identity_turn -v
./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_chat_relational_stance.py::test_enforce_still_compresses_instrumental_direct_response -v
./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_chat_relational_stance.py::test_enforce_preserves_playful_exchange_frame -v
```

Expected: PASS for all three.

- [ ] **Step 4: Run existing stance plumbing tests (no regressions)**

```bash
./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py -v
```

Expected: PASS (may fail on prompt string asserts until Task 4 — if so, continue to Task 4 before re-running).

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/chat_stance.py
git commit -m "fix: exempt relational stance briefs from business-mode compression"
```

---

### Task 3: Stance brief prompt — interface_cost vs connection_seek

**Files:**
- Modify: `orion/cognition/prompts/chat_stance_brief.j2:68-77`

- [ ] **Step 1: Replace LOW-BANDWIDTH block**

Delete lines 68–77 (`LOW-BANDWIDTH / EMBODIED INTERACTION ASSESSMENT` through the `voice later` bullet).

Insert:

```jinja2
INTERACTION POSTURE ASSESSMENT (semantic — no keyword matching)
- Infer two independent dimensions from whole-turn meaning and recent_history (not keyword lists):
  - interface_cost: costly to type, read, screen-use, act, or continue interaction (motion, fatigue, overload, one-handed use)
  - connection_seek: user wants presence, venting, companionship, situated curiosity, mind-off distraction, or explicitly rejects solutioning/fixing
- connection_seek overrides interface_cost minimization when the user invites talk, questions, venting, or says no fixing/no solutioning.
- Represent posture using existing stance fields only (no new JSON keys in v1).

When interface_cost is high and connection_seek is low:
  - set identity_salience low unless identity question
  - prefer task_mode direct_response (or triage if operational/strained)
  - response_priorities: reduce_interaction_load, release_user_from_replying, keep_response_short, offer_voice_pause_or_later
  - response_hazards: avoid_open_ended_questions, do_not_invite_continued_typing, avoid_presence_centering, avoid_poetic_reassurance

When connection_seek is high (any interface_cost):
  - prefer conversation_frame reflective or playful_relational
  - set task_mode reflective_dialogue or playful_exchange
  - response_priorities: companion_presence, situated_curiosity, hold_space, no_solutioning
  - response_hazards: avoid_task_tracking, avoid_next_steps, avoid_transactional_closers, avoid_customer_support_tone
  - if interface_cost is also high, add keep_response_compact and avoid_question_pile_on

Do not infer avoid_open_ended_questions from sickness/recovery alone when connection_seek is present.
```

- [ ] **Step 2: Run prompt contract test**

```bash
./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_chat_relational_stance.py::test_stance_brief_prompt_has_connection_seek_vocabulary -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add orion/cognition/prompts/chat_stance_brief.j2
git commit -m "fix: stance brief semantic interface_cost vs connection_seek posture"
```

---

### Task 4: Update existing plumbing tests for renamed section

**Files:**
- Modify: `services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py:57-66`

- [ ] **Step 1: Update `test_stance_brief_has_semantic_low_bandwidth_guidance`**

Rename test to `test_stance_brief_has_semantic_interaction_posture_guidance` and replace asserts:

```python
def test_stance_brief_has_semantic_interaction_posture_guidance() -> None:
    prompt = Path("orion/cognition/prompts/chat_stance_brief.j2").read_text(encoding="utf-8")
    assert "INTERACTION POSTURE ASSESSMENT" in prompt
    assert "interface_cost" in prompt
    assert "connection_seek" in prompt
    assert "Do not use keyword matching" in prompt
    assert "companion_presence" in prompt
    assert "reduce_interaction_load" in prompt
```

Remove assert for `"LOW-BANDWIDTH / EMBODIED INTERACTION ASSESSMENT"`.

- [ ] **Step 2: Run plumbing tests**

```bash
./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py
git commit -m "test: align stance plumbing asserts with interaction posture prompt"
```

---

### Task 5: Speech pass — relational curiosity + transactional hazards

**Files:**
- Modify: `orion/cognition/prompts/chat_general.j2:41-65`

- [ ] **Step 1: Insert relational block after curiosity/attention-frame rules (after line 44)**

```jinja2
- When chat_stance_brief.task_mode is reflective_dialogue or playful_exchange, or conversation_frame is reflective or playful_relational:
  - attention_frame is advisory only; stance brief controls whether to ask
  - when response_priorities include companion_presence, situated_curiosity, hold_space, or no_solutioning: ask one or two situated questions grounded in the thread; do not use generic reversal questions
  - when response_hazards include avoid_task_tracking, avoid_next_steps, or avoid_transactional_closers: do not offer tracking, next steps, readiness checks, or customer-support closers
  - hold existential or emotional weight without redirecting to action or solutions unless explicitly requested
  - keep responses compact when response_priorities include keep_response_compact
```

- [ ] **Step 2: Narrow the low-bandwidth obey block (line 58) so it does not apply on relational task modes**

Change the opening condition from:

```jinja2
- If chat_stance_brief response_priorities or response_hazards indicate reduced interaction load...
```

To:

```jinja2
- If chat_stance_brief task_mode is not reflective_dialogue or playful_exchange, and response_priorities or response_hazards indicate reduced interaction load...
```

- [ ] **Step 3: Add to ANTI-PATTERNS section (~line 82)**

```jinja2
- Do not close with transactional support phrases (e.g., "let me know when you're ready", "need anything tracked", "what's the next step") when stance hazards include avoid_transactional_closers or avoid_next_steps.
```

- [ ] **Step 4: Run speech prompt test**

```bash
./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_chat_relational_stance.py::test_speech_prompt_relational_curiosity_overrides_attention_frame -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/cognition/prompts/chat_general.j2
git commit -m "fix: speech pass follows relational stance over attention frame"
```

---

### Task 6: Identity policy alignment

**Files:**
- Modify: `orion/cognition/personality/orion_identity.yaml:49-59`

- [ ] **Step 1: Replace clarifying-questions priority**

Change line 49 from:
```yaml
    - "Ask clarifying questions only as a last resort."
```
To:
```yaml
    - "On instrumental or task turns, ask clarifying questions only as a last resort; on relational turns (when stance selects companion/reflective mode), situated curiosity is appropriate."
```

- [ ] **Step 2: Append transactional banned phrases**

Add under `banned_phrases:`:

```yaml
    - "Let me know when you're ready"
    - "What's the next step"
    - "Need anything tracked or noted"
    - "I'm here if you need anything"
```

- [ ] **Step 3: Verify YAML loads**

```bash
python3 -c "import yaml; yaml.safe_load(open('orion/cognition/personality/orion_identity.yaml'))"
```

Expected: exit 0, no output.

- [ ] **Step 4: Commit**

```bash
git add orion/cognition/personality/orion_identity.yaml
git commit -m "fix: scope clarifying-questions rule; ban transactional closers"
```

---

### Task 7: Full verification

- [ ] **Step 1: Run all new and related tests**

```bash
./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_chat_relational_stance.py services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py -v
```

Expected: all PASS

- [ ] **Step 2: Compile cortex-exec**

```bash
python3 -m compileall services/orion-cortex-exec/app/chat_stance.py
```

Expected: exit 0

- [ ] **Step 3: Optional broader stance suite**

```bash
./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_chat_stance_brief.py -q
```

Expected: PASS (no regressions from prompt changes)

---

## Spec coverage self-review

| Spec requirement | Task |
|------------------|------|
| Exempt relational modes from business compressor | Task 2 |
| interface_cost / connection_seek vocabulary | Task 3 |
| Speech pass stance wins curiosity | Task 5 |
| Identity policy + banned transactional phrases | Task 6 |
| Fixture tests through enforce | Task 1–2 |
| No keyword detectors | Enforced by design + anti-slop rule (already committed) |
| v2 interaction_regime schema | Out of scope |

No placeholders. All code blocks are complete.

---

## Remaining risks (post-v1)

- Stance **LLM** may still misclassify turns; v1 fixes Python crushing correct output, not LLM mis-inference.
- Live companion-thread replay (optional v1.1) requires running stack + manual or integration harness — not in this plan.
- `banned_phrases` in identity YAML guide speech pass via summaries; they are output anti-patterns, not user-input triggers (per spec).
