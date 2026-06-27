# Orion relational stance — structural fix (not keyword patches)

**Date:** 2026-06-26  
**Status:** Approved for implementation planning  
**Problem:** Orion chat replies feel transactional, non-curious, and "all business" during relational turns (loneliness, recovery companionship, existential venting, explicit no-solutioning requests).

---

## Root cause (evidence)

Orion `chat_general` uses a two-pass pipeline:

1. **Stance synthesizer** (`chat_stance_brief.j2` → LLM) — semantic inference over turn + history + substrate signals. Already instructed: *"Do not use keyword matching."*
2. **Speech pass** (`chat_general.j2` → LLM) — follows `ChatStanceBrief` contract.
3. **Post-processor** (`enforce_chat_stance_quality` in `services/orion-cortex-exec/app/chat_stance.py`) — **deterministically overwrites relational stance on most turns.**

On every non-identity turn, `enforce_chat_stance_quality` currently:

- Sets `juniper_relevance` to *"Prioritize practical usefulness over relationship labels."*
- Strips `active_relationship_facets` when `identity_salience` is low
- Injects `avoid_identity_recital` hazards

This runs **even when** the stance LLM sets `task_mode=reflective_dialogue` or `conversation_frame=reflective`. The stance synthesizer may infer correctly; Python crushes it back to instrumental mode.

Secondary conflicts:

| Layer | Issue |
|-------|--------|
| `chat_stance_brief.j2` low-bandwidth section | Hazard vocabulary only supports *minimize interaction* (`reduce_interaction_load`, `avoid_open_ended_questions`). No paired vocabulary for *connection under load*. |
| `chat_general.j2` + attention frame | Curiosity hard-gated: questions only when `selected_action.action_type == ask` (score ≥ 0.65). Relational invitation does not override. |
| `orion_identity.yaml` | *"Ask clarifying questions only as a last resort"* biases away from companion curiosity. |

Literal LLM `temperature` (default 0.7 in executor) is **not** the primary lever. Structural stance compression is.

---

## Goals

- Orion can be warm, curious, and present on relational turns without becoming a task tracker.
- Fixes wire through the **existing semantic stance pipeline** — no keyword/phrase ontologies for feelings, medical states, or life events.
- Scales to unseen situations via stance LLM inference + schema/post-processor discipline.

## Non-goals

- Hub UI changes
- Regex/keyword detectors for emotional or medical language
- Global temperature bump
- New microservices
- Phrase-list patches ("when user says surgery…")

---

## Architecture

```text
user turn
  → build_chat_stance_inputs (beliefs, turn_effect, relational drive, attention frame)
  → synthesize_chat_stance_brief (LLM, semantic)
  → enforce_chat_stance_quality (Python — FIX HERE)
  → llm_chat_general (speech, follows brief)
  → identity boundary guard (role inversion only)
```

**Hierarchy (from README):** identity constrains stance → stance shapes style. Style must not replace identity; instrumental compression must not replace relational stance.

---

## Design — v1 (approved)

### 1. Exempt relational modes from business compressor

**File:** `services/orion-cortex-exec/app/chat_stance.py` — `enforce_chat_stance_quality`

Define `_RELATIONAL_TASK_MODES = frozenset({"reflective_dialogue", "playful_exchange"})` and `_RELATIONAL_FRAMES = frozenset({"reflective", "playful_relational"})`.

When `task_mode` or `conversation_frame` is relational:

- **Do not** set `juniper_relevance` to practical-usefulness-over-relationship
- **Do not** strip relationship facets populated by stance synthesizer
- **Do not** inject `avoid_identity_recital` / business-only priorities
- **Do** preserve stance LLM's `response_priorities`, `response_hazards`, `answer_strategy`

When `task_mode == triage` or `conversation_frame == technical`: keep current instrumental compression.

### 2. Expand stance brief semantic vocabulary (LLM-inferred, not keyword triggers)

**File:** `orion/cognition/prompts/chat_stance_brief.j2`

Replace monolithic "LOW-BANDWIDTH" block with two **semantic dimensions** the stance LLM must infer from whole-turn meaning:

- **`interface_cost`** — costly to type/read/act (motion, fatigue, one-handed, screen overload)
- **`connection_seek`** — user wants presence, venting, curiosity, companionship, mind-off distraction, or explicitly rejects solutioning

Output via existing fields only (no new schema in v1):

| Inferred state | `task_mode` | `response_priorities` (examples) | `response_hazards` (examples) |
|----------------|-------------|-----------------------------------|-------------------------------|
| High interface_cost, low connection_seek | `direct_response` | `reduce_interaction_load`, `release_user_from_replying`, `keep_response_short` | `avoid_open_ended_questions`, `do_not_invite_continued_typing` |
| High connection_seek (any interface_cost) | `reflective_dialogue` | `companion_presence`, `situated_curiosity`, `hold_space`, `no_solutioning` | `avoid_task_tracking`, `avoid_next_steps`, `avoid_transactional_closers`, `avoid_customer_support_tone` |
| Both high | `reflective_dialogue` | above + `keep_response_compact` | above + `avoid_question_pile_on` |

Explicit rule: **connection_seek overrides interface_cost minimization** when the user invites talk, questions, venting, or says no fixing.

Remove contradiction: recovery/sickness must not automatically imply `avoid_open_ended_questions`.

### 3. Speech pass — stance wins curiosity on relational turns

**File:** `orion/cognition/prompts/chat_general.j2`

When `chat_stance_brief.task_mode` is `reflective_dialogue` or `playful_exchange`:

- Attention frame is **advisory**, not binding
- Ask 1–2 situated questions when `response_priorities` include curiosity/companion tags
- Do not offer task tracking, next steps, or "let me know when you're ready" closers when hazards include transactional patterns
- Hold existential weight without redirecting to action

When instrumental/triage: keep current attention-frame gating.

### 4. Identity policy alignment

**File:** `orion/cognition/personality/orion_identity.yaml`

- Change *"Ask clarifying questions only as a last resort"* → scoped: *instrumental/task turns only; on relational turns, situated curiosity is appropriate when stance brief selects it*
- Add `banned_phrases` for transactional closers (structural list for speech pass anti-patterns, not user-input triggers):
  - "Let me know when you're ready"
  - "What's the next step"
  - "Need anything tracked or noted"
  - "I'm here if you need anything" (as a standalone closer)

### 5. Tests (required for closure)

**File:** `services/orion-cortex-exec/tests/test_chat_relational_stance.py` (new)

Fixtures replaying transcript turns through:

1. `enforce_chat_stance_quality` with a synthetic `ChatStanceBrief` that has `reflective_dialogue` — assert business compressor does **not** run
2. Stance brief prompt contract strings (vocabulary present, keyword-matching forbidden)
3. Regression: brief with `response_priorities=["companion_presence", "no_solutioning"]` survives enforce with relationship facets intact

Optional v1.1: golden transcript integration test (surgery/lonely thread) through fallback + enforce path.

---

## v2 follow-up (not in v1 scope)

Add `interaction_regime: instrumental | relational | minimal` to `ChatStanceBrief` schema and wire generation profile (temperature delta, max questions) from regime in executor. v1 uses existing `task_mode` + `conversation_frame` as regime proxy.

---

## Agent guardrail

Cursor rule: `.cursor/rules/conversational-behavior-anti-slop.mdc`  
Mirror summary in `AGENTS.md` § Conversational behavior changes.

---

## Acceptance checks

- [ ] Relational `ChatStanceBrief` survives `enforce_chat_stance_quality` without business compression
- [ ] Stance brief prompt documents interface_cost vs connection_seek; no keyword lists for user states
- [ ] Speech prompt subordinates attention frame on relational task modes
- [ ] Transcript fixture tests pass in `orion-cortex-exec`
- [ ] No new regex/keyword detectors for feelings or medical states
- [ ] Live replay of companion thread (optional v1.1): Orion asks situated questions, does not offer tracking/next-steps

---

## Files likely to touch (implementation)

- `services/orion-cortex-exec/app/chat_stance.py`
- `orion/cognition/prompts/chat_stance_brief.j2`
- `orion/cognition/prompts/chat_general.j2`
- `orion/cognition/personality/orion_identity.yaml`
- `services/orion-cortex-exec/tests/test_chat_relational_stance.py`
- `.cursor/rules/conversational-behavior-anti-slop.mdc`
- `AGENTS.md`
