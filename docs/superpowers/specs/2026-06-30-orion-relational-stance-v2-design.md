# Orion relational stance v2 — late contract injection + regime persistence

**Date:** 2026-06-30
**Status:** Approved for implementation planning
**Predecessor:** `docs/superpowers/specs/2026-06-26-orion-relational-stance-design.md` (v1)

---

## Problem

v1 (PR #762) fixed the structural compressor that was crushing relational stance. Live results: turn 1 works; turn 2 still leaks customer-support closers.

Confirmed root causes (evidence from rendered prompt measurement):

1. **Prompt recency deficit** — `avoid_transactional_closers` appears at ~token 340 of a 2200-token speech prompt. The `TASK` section is at ~token 1934. The model must hold the relational contract across 1600+ tokens of competing rules before generating. Strong pretraining prior (support-script closers) wins at generation time.

2. **Brief indirection** — the `TASK` section says "follow `response_hazards` in the brief" by reference to injected JSON. The model must interpret JSON intent 1600 tokens earlier, not act on a flat instruction near the generation point.

3. **Inter-turn regime decay** — the stance synthesizer re-infers from scratch each turn. Turn 2 ("thanks, it's just hard…") scores `connection_seek` lower because the explicit invite is gone. Brief weakens; contract weakens; closer leaks through.

Post-generation stripping (e.g., `strip_transactional_closers`) is out of scope — fixes must be upstream of generation.

---

## Goals

- Relational regime established on turn 1 persists across follow-up turns unless the user pivots to task work
- Speech model receives a flat, late, regime-specific contract immediately before generation — not a JSON reference 1600 tokens earlier
- Closing move is positively specified ("end with a wondering") not just negatively prohibited ("avoid closers")
- All fixes are deterministic Python + prompt wiring — no new LLM calls, no output editing

## Non-goals

- Prompt restructure / section reordering (deferred to v3, see Appendix)
- Regenerate-on-violation retry
- Structured speech output schema
- Keyword/phrase detectors for user emotional states
- Global temperature bump

---

## Architecture

```text
user turn
  → build_chat_stance_inputs  ← [NEW] prior_stance summary injected here
  → synthesize_chat_stance_brief (LLM)  ← [NEW] prior_stance input + carryforward rule
  → enforce_chat_stance_quality (Python)
  → [NEW] compile_speech_contract (Python, deterministic)
  → executor stores speech_contract in ctx
  → chat_general.j2 renders  ← [NEW] TURN CONTRACT block just before TASK
  → llm_chat_general generates  ← sees contract at ~token 1900
  → router guards (identity boundary, recital strip)
```

---

## Design

### 1. Schema changes — `orion/schemas/chat_stance.py`

Two new optional fields on `ChatStanceBrief`:

```python
interaction_regime: Literal["instrumental", "relational", "minimal"] | None = Field(default=None)
companion_closing_move: str | None = Field(default=None)
```

**`interaction_regime`** — explicit regime signal. Replaces inference from `task_mode` + `conversation_frame` in downstream consumers (compiler, enforce). Values:
- `relational` — companion presence, venting, companionship, no-solutioning
- `minimal` — high interface cost, release from replying, low interaction demand
- `instrumental` — task, triage, technical, direct response

**`companion_closing_move`** — positive closing specification, set by the stance synthesizer when `connection_seek` is high. Not "avoid X" but what to end *with*. Example values: `"end_with_a_wondering"`, `"leave_space_without_offer"`, `"ground_observation"`, `"be_with_silence"`. `None` means no specific closing instruction.

Both fields default to `None` — all existing paths are unaffected.

### 2. Prior brief carryforward

**Problem:** stance LLM re-infers regime from scratch each turn. Follow-up vent turns may not re-establish `connection_seek` without the explicit companion invite.

**Fix — executor (`services/orion-cortex-exec/app/executor.py`):**

After `enforce_chat_stance_quality` runs on turn N, store a compact summary in `ctx["prior_chat_stance_brief"]`:

```python
{
    "interaction_regime": brief.interaction_regime,
    "task_mode": brief.task_mode,
    "response_priorities": brief.response_priorities[:4],
    "response_hazards": brief.response_hazards[:4],
}
```

**Persistence across requests:** The executor ctx is rebuilt per request, so this cannot live in `ctx` alone. Two mechanisms, in order of preference:

1. **In-process session cache** — an ephemeral module-level dict keyed by `(session_id, conversation_id)` in `executor.py` (same pattern as `_CANDIDATE_SPARK_META` in orion-spark-introspector). Lost on restart; sufficient for continuous companion threads within a session window. Evict entries older than a configurable TTL (default 30 min).

2. **Hub passthrough** — the Hub includes the prior brief summary in the request ctx payload for the next turn. Survives restarts but requires Hub-side storage and an API contract change. Deferred to v2.1 unless in-process cache proves insufficient.

v2 implements option 1 only. The session cache key is the conversation/session identifier already present in the request ctx.

**Fix — stance inputs (`services/orion-cortex-exec/app/chat_stance.py` → `build_chat_stance_inputs`):**

Read `ctx.get("prior_chat_stance_brief")` and include in `inputs["prior_stance"]` when present.

**Fix — stance synthesizer (`orion/cognition/prompts/chat_stance_brief.j2`):**

New optional input block and rule:

```
{% if prior_stance %}
- prior_stance: {{ prior_stance }}
{% endif %}
```

New instruction under INTERACTION POSTURE ASSESSMENT:

> If `prior_stance.interaction_regime` is `relational` and the current turn continues the same emotional thread (venting, presence-seeking, recovery companionship) without a clear pivot to task or technical work, carry `interaction_regime=relational` forward. Do not re-infer regime as instrumental just because the explicit companion invite is no longer in this message. A task pivot (explicit technical question, deploy/debug/restart request) overrides continuation.

Pivot detection stays fully semantic — stance LLM decides. No keyword lists.

### 3. `compile_speech_contract` — deterministic compiler

**Location:** `services/orion-cortex-exec/app/chat_stance.py`

**Signature:**
```python
def compile_speech_contract(brief: ChatStanceBrief) -> str:
```

Pure Python, no LLM, no I/O. Called in executor after enforce. Result stored in `ctx["speech_contract"]`.

**Regime branches:**

| `interaction_regime` | Compiled contract |
|---|---|
| `relational` | "This is a companion turn. [companion_closing_move → instruction if set, else 'Stay present; do not offer next steps, trackers, or support closers.']. [If `situated_curiosity` in response_priorities: 'Ask one grounded question from this thread — not a generic reversal.']" |
| `minimal` | "Keep this reply very short. Do not ask questions. Release Juniper from replying — offer voice, a pause, or continuation later." |
| `instrumental` / `None` | "Answer directly." [If `triage` task_mode: "Lead with the operational blocker."] |

`companion_closing_move` values map to natural-language instructions:
- `"end_with_a_wondering"` → "End with a wondering, not an offer."
- `"leave_space_without_offer"` → "Leave space. Do not close with an offer to help."
- `"ground_observation"` → "End with a grounded observation from the thread."
- `"be_with_silence"` → "Hold the silence. No closing move required."

Unrecognized or `None` values fall through to the default relational closing: no transactional offers.

### 4. Late injection in `chat_general.j2`

New block inserted immediately before the `TASK` section:

```jinja
{% if speech_contract %}
TURN CONTRACT
{{ speech_contract }}

{% endif %}
```

This renders the compiled contract at ~token 1900 — immediately before generation — instead of buried hazard lists at token 340.

The full brief is still available earlier in the prompt for nuance. The contract is the late authoritative summary the model acts on.

---

## Files to touch

| File | Change |
|---|---|
| `orion/schemas/chat_stance.py` | Add `interaction_regime`, `companion_closing_move` fields |
| `orion/cognition/prompts/chat_stance_brief.j2` | Add `prior_stance` input, carryforward instruction, `interaction_regime` + `companion_closing_move` in output schema |
| `orion/cognition/prompts/chat_general.j2` | Add `TURN CONTRACT` block before `TASK` |
| `services/orion-cortex-exec/app/chat_stance.py` | Add `compile_speech_contract`; update `build_chat_stance_inputs` for prior stance |
| `services/orion-cortex-exec/app/executor.py` | Store prior brief summary in ctx after enforce; call compiler; set `ctx["speech_contract"]` |
| `services/orion-cortex-exec/tests/test_chat_relational_stance.py` | New tests (see below) |
| `services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py` | TURN CONTRACT position regression |

---

## Tests

**New unit tests (`test_chat_relational_stance.py`):**
- `compile_speech_contract` with `interaction_regime=relational` + `companion_closing_move="end_with_a_wondering"` → output contains closing instruction, no "let me know" language
- `compile_speech_contract` with `interaction_regime=relational` + `situated_curiosity` in priorities → output contains question instruction
- `compile_speech_contract` with `interaction_regime=minimal` → short, release-from-replying language
- `compile_speech_contract` with `interaction_regime=instrumental` → direct-answer language
- `build_chat_stance_inputs` with `prior_chat_stance_brief` in ctx → prior stance present in returned inputs dict
- `ChatStanceBrief` roundtrips with both new fields

**Regression guard (`test_chat_general_stance_plumbing.py`):**
- Rendered `chat_general.j2` with `speech_contract` set → `TURN CONTRACT` appears after `ANTI-PATTERNS`, before `TASK`
- Rendered without `speech_contract` → no `TURN CONTRACT` block emitted

## Acceptance checks

- [ ] Turn 1 (companion invite) + turn 2 (vent continuation): turn 2 brief has `interaction_regime=relational`; prior brief from turn 1 was in stance inputs; `TURN CONTRACT` in rendered speech prompt; no support closer in output
- [ ] Technical pivot after relational thread: regime switches to `instrumental` on the pivot turn
- [ ] All existing relational stance tests pass (v1 non-regression)
- [ ] `compile_speech_contract` unit tests pass
- [ ] `TURN CONTRACT` position test passes

---

## Appendix — v3: Prompt restructure (Option C)

*Not in this spec. Captured here as the next phase.*

**Problem this solves:** The `NON-NEGOTIABLE` block in `chat_general.j2` is ~400 tokens of mixed universal + regime-specific rules at the top of the prompt. Every future mode adds to this wall. `compile_speech_contract` (v2) puts the contract late, but the wall stays.

**Design sketch:**

Split `chat_general.j2` into:

1. **Universal early block** (keep at top): identity boundary, memory rules, universal anti-patterns, evidence-gated claims
2. **Regime-gated late blocks** (just before TASK): replace the `TURN CONTRACT` injection with full regime sections, each gated on `interaction_regime`. Each section contains all rules relevant to that regime, in one place, near generation.

```jinja
{% if interaction_regime == "relational" %}
RELATIONAL TURN RULES
...
{% elif interaction_regime == "minimal" %}
MINIMAL TURN RULES
...
{% else %}
INSTRUMENTAL TURN RULES
...
{% endif %}

TASK
Produce exactly one user-facing reply.
```

**Trade-offs:**
- Cleanest long-term architecture; rules are co-located with generation, not scattered through a wall
- Regression risk across all modes during migration — requires full test coverage of each branch
- Removes the `compile_speech_contract` Python indirection once the prompt is restructured — v3 and v2 overlap; v2 can be stripped when v3 is proven

**When to do it:** After v2 is live and verified in production. v2 proves the late-contract principle works; v3 cleans up the architecture without behavioral risk.
