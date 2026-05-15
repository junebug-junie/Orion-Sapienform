# chat_kids_story Verb Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add verb `chat_kids_story` that mirrors `chat_quick`’s fast single-pass mesh and Recall RPC behavior, uses a vector-off SQL-heavy default recall profile, and centralizes shared behavior in `FAST_SINGLE_PASS_CHAT_VERBS` so exec/orch/recall/fanout stay aligned.

**Architecture:** New cognition verb YAML + Jinja + recall profile on disk; shared frozenset `orion/cognition/fast_chat_verbs.py`; `recall_utils` gains `apply_fast_chat_recall_profile_clamp` + extended `resolve_recall_bus_rpc_wait_sec`; `settings` adds `CHAT_KIDS_STORY_RECALL_PROFILE`; router/supervisor pass both profile defaults into `delivery_safe_recall_decision`; `executor.py` threads the frozenset through token caps, metacog timeout, situation prep set, inline Recall clamp, default `llm_route`, and `_should_prepare_brain_reply_context`; orch maps the verb to execution lane `chat`; `fanout_policy` returns `bounded` for the new verb; `orion-actions` maps the verb name to route `quick` like `chat_quick`.

**Tech stack:** Python 3, Pydantic settings, Pytest, YAML verbs under `orion/cognition/verbs/`, Jinja2 prompts, Redis bus (unchanged).

**Worktree:** Recommended: implement in a dedicated git worktree (see repo `using-git-worktrees` / superpowers skill) so `main` stays clean while subagents land each task.

**Commits:** Template steps below include `git commit`. **If your operator policy is “AI commits only when the human asks,” skip those steps** and keep a single local commit or squash at the end.

---

## File map (create / modify)

| Path | Responsibility |
|------|------------------|
| `orion/cognition/fast_chat_verbs.py` | `FAST_SINGLE_PASS_CHAT_VERBS` frozenset; single import surface for orch, exec, autonomy. |
| `orion/cognition/verbs/chat_kids_story.yaml` | Verb definition: two-step plan, `recall_profile: chat.story.kids.v1`, points at kids Jinja. |
| `orion/cognition/prompts/chat_kids_story.j2` | Storytelling tone, digest rules, multi-listener + age-safety rails (no real PII). |
| `orion/recall/profiles/chat.story.kids.v1.yaml` | Vector-off; SQL chat + timeline; modest cards knob. |
| `orion/autonomy/fanout_policy.py` | Treat `chat_kids_story` like plain `chat_quick` for bounded fanout. |
| `services/orion-cortex-orch/app/execution_lanes.py` | Map `chat_kids_story` → lane `chat`. |
| `services/orion-cortex-exec/app/settings.py` | `chat_kids_story_recall_profile` + env alias `CHAT_KIDS_STORY_RECALL_PROFILE`. |
| `services/orion-cortex-exec/.env_example` | Document `CHAT_KIDS_STORY_RECALL_PROFILE`. |
| `services/orion-cortex-exec/app/recall_utils.py` | `resolve_recall_bus_rpc_wait_sec` + clamp + `delivery_safe_recall_decision` extended. |
| `services/orion-cortex-exec/app/router.py` | Pass `chat_kids_story_recall_profile`; branch `prepare_chat_quick_reply_context` for both verbs. |
| `services/orion-cortex-exec/app/supervisor.py` | Same `delivery_safe_recall_decision` kwargs as router. |
| `services/orion-cortex-exec/app/executor.py` | All fast-verb wiring (import frozenset, tokens, metacog, situation set, recall clamp, `llm_route`, `_should_prepare_brain_reply_context`). |
| `services/orion-actions/app/main.py` | `_normalized_llm_route`: treat `chat_kids_story` like quick-chat route key. |
| `services/orion-cortex-orch/tests/test_execution_lanes.py` | New test: `chat_kids_story` resolves to `chat`. |
| `services/orion-cortex-exec/tests/test_recall_delivery_gating.py` | Clamp + explicit profile tests for `chat_kids_story`. |
| `services/orion-cortex-exec/tests/test_recall_bus_rpc_timeout.py` | `chat_kids_story` uses quick cap. |
| `services/orion-cortex-exec/tests/test_executor_runtime_context_skip.py` | `_should_prepare_brain_reply_context` false for new verb. |
| `services/orion-cortex-exec/tests/test_chat_kids_story_plumbing.py` | **Create:** YAML structure + prompt string assertions (mirror `test_chat_quick_plumbing.py`). |
| `services/orion-cortex-exec/tests/test_chat_general_route_mapping.py` | **Append:** async test that `llm_chat_kids_story` sends `route=="quick"`. |
| `orion/autonomy/tests/test_fanout_policy.py` | **Append:** bounded fanout for `chat_kids_story`. |
| `services/orion-hub/README.md` | Short “Story lane” subsection: send `verbs: ["chat_kids_story"]` (v1 hub contract). |

---

### Task 1: Shared frozenset + fanout policy

**Files:**

- Create: `orion/cognition/fast_chat_verbs.py`
- Modify: `orion/autonomy/fanout_policy.py`
- Modify: `orion/autonomy/tests/test_fanout_policy.py`

- [ ] **Step 1: Write failing test**

Append to `orion/autonomy/tests/test_fanout_policy.py`:

```python
def test_fanout_bounded_chat_kids_story() -> None:
    assert autonomy_subject_fanout_from_runtime_ctx({"verb": "chat_kids_story", "mode": "brain", "options": {}}) == "bounded"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest orion/autonomy/tests/test_fanout_policy.py::test_fanout_bounded_chat_kids_story -q --tb=short
```

Expected: `FAILED` (import error or assertion `full != bounded`).

- [ ] **Step 3: Create frozenset module**

Create `orion/cognition/fast_chat_verbs.py`:

```python
from __future__ import annotations

FAST_SINGLE_PASS_CHAT_VERBS = frozenset({"chat_quick", "chat_kids_story"})
```

- [ ] **Step 4: Implement fanout policy**

In `orion/autonomy/fanout_policy.py`, add at top (after imports):

```python
from orion.cognition.fast_chat_verbs import FAST_SINGLE_PASS_CHAT_VERBS
```

Replace the body condition so **both** `chat_quick` (without full stance) **and** `chat_kids_story` return `bounded`:

```python
    verb = str(ctx.get("verb") or "").strip().lower()
    opts = ctx.get("options") if isinstance(ctx.get("options"), dict) else {}
    hub_full = bool(opts.get("chat_quick_full_stance"))
    if verb in FAST_SINGLE_PASS_CHAT_VERBS and not (verb == "chat_quick" and hub_full):
        return "bounded"
    return "full"
```

Wait: original logic was `if verb == "chat_quick" and not hub_full: return bounded`. For `chat_kids_story`, `hub_full` must not apply. Correct logic:

```python
    verb = str(ctx.get("verb") or "").strip().lower()
    opts = ctx.get("options") if isinstance(ctx.get("options"), dict) else {}
    hub_full = bool(opts.get("chat_quick_full_stance"))
    if verb == "chat_quick" and not hub_full:
        return "bounded"
    if verb == "chat_kids_story":
        return "bounded"
    return "full"
```

This preserves exact `chat_quick` semantics (stance option still forces full for **only** `chat_quick`). Alternatively use:

```python
    if verb in FAST_SINGLE_PASS_CHAT_VERBS and not (verb == "chat_quick" and hub_full):
        return "bounded"
```

Because `chat_kids_story in FAST_SINGLE_PASS_CHAT_VERBS` and `(verb == "chat_quick" and hub_full)` is false for kids verb, kids always bounded. Use this one-liner.

- [ ] **Step 5: Run test to verify it passes**

Same pytest command as Step 2. Expected: `1 passed`.

- [ ] **Step 6: Commit (optional)**

```bash
git add orion/cognition/fast_chat_verbs.py orion/autonomy/fanout_policy.py orion/autonomy/tests/test_fanout_policy.py
git commit -m "$(cat <<'EOF'
feat(autonomy): bounded fanout for chat_kids_story

Treat chat_kids_story like fast single-pass chat for autonomy subject fanout.
EOF
)"
```

---

### Task 2: recall_utils — RPC wait + profile clamp + delivery_safe

**Files:**

- Modify: `services/orion-cortex-exec/app/recall_utils.py`
- Modify: `services/orion-cortex-exec/tests/test_recall_delivery_gating.py`

- [ ] **Step 1: Write failing tests**

Append to `services/orion-cortex-exec/tests/test_recall_delivery_gating.py`:

```python
def test_chat_kids_story_clamps_inherited_profile_to_story_default() -> None:
    decision = delivery_safe_recall_decision(
        {"enabled": True, "profile": "reflect.v1"},
        [_memory_step()],
        output_mode="direct_answer",
        verb_profile="chat.general.v1",
        plan_verb_name="chat_kids_story",
        chat_quick_recall_profile="assist.light.v1",
        chat_kids_story_recall_profile="chat.story.kids.v1",
    )
    assert decision["run_recall"] is True
    assert decision["profile"] == "chat.story.kids.v1"
    assert decision["profile_source"] == "fast_chat_latency_default"


def test_chat_kids_story_keeps_explicit_client_profile() -> None:
    decision = delivery_safe_recall_decision(
        {"enabled": True, "profile": "deep.graph.v1", "profile_explicit": True},
        [_memory_step()],
        verb_profile="chat.general.v1",
        plan_verb_name="chat_kids_story",
        chat_quick_recall_profile="assist.light.v1",
        chat_kids_story_recall_profile="chat.story.kids.v1",
    )
    assert decision["profile"] == "deep.graph.v1"
    assert decision["profile_source"] == "explicit"
```

`delivery_safe_recall_decision` gains `chat_kids_story_recall_profile` with default `"chat.story.kids.v1"` so existing callers (e.g. `test_chat_prompt_context_guardrails.py`) need no edits.

- [ ] **Step 2: Run new tests — expect FAIL**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-cortex-exec && PYTHONPATH=. ../../venv/bin/python -m pytest tests/test_recall_delivery_gating.py::test_chat_kids_story_clamps_inherited_profile_to_story_default tests/test_recall_delivery_gating.py::test_chat_kids_story_keeps_explicit_client_profile -q --tb=short
```

- [ ] **Step 3: Implement recall_utils**

In `services/orion-cortex-exec/app/recall_utils.py`:

1. Add import: `from orion.cognition.fast_chat_verbs import FAST_SINGLE_PASS_CHAT_VERBS`

2. Replace `resolve_recall_bus_rpc_wait_sec` lane cap with:

```python
    verb_l = str(verb or "").strip().lower()
    lane_cap = (
        float(chat_quick_recall_rpc_timeout_sec)
        if verb_l in FAST_SINGLE_PASS_CHAT_VERBS
        else float(recall_rpc_timeout_sec)
    )
```

3. Rename `apply_chat_quick_recall_profile_clamp` → `apply_fast_chat_recall_profile_clamp` with signature:

```python
def apply_fast_chat_recall_profile_clamp(
    *,
    plan_verb_name: str | None,
    recall_cfg: Dict[str, Any],
    profile: str,
    profile_source: str,
    chat_quick_recall_profile: str,
    chat_kids_story_recall_profile: str,
) -> Tuple[str, str]:
    """Fast single-pass chat verbs: default to low-latency recall profile unless profile explicit."""
    verb_l = str(plan_verb_name or "").strip().lower()
    if verb_l not in FAST_SINGLE_PASS_CHAT_VERBS:
        return profile, profile_source
    _explicit_profile, explicit_source = _resolve_explicit_profile(recall_cfg)
    if explicit_source == "explicit" or _normalize_bool(recall_cfg.get("profile_explicit"), default=False):
        return profile, profile_source
    if verb_l == "chat_kids_story":
        qp = str(chat_kids_story_recall_profile or "").strip()
    else:
        qp = str(chat_quick_recall_profile or "").strip()
    if not qp:
        return profile, profile_source
    return qp, "fast_chat_latency_default"
```

4. Update `delivery_safe_recall_decision` signature:

```python
def delivery_safe_recall_decision(
    recall_cfg: Dict[str, Any],
    steps: Iterable[ExecutionStep],
    *,
    output_mode: str | None = None,
    verb_profile: Optional[str] = None,
    user_text: str | None = None,
    runtime_mode: str | None = None,
    plan_verb_name: str | None = None,
    chat_quick_recall_profile: str = "assist.light.v1",
    chat_kids_story_recall_profile: str = "chat.story.kids.v1",
) -> Dict[str, Any]:
```

Replace the inner clamp call with:

```python
    base_profile, profile_source = apply_fast_chat_recall_profile_clamp(
        plan_verb_name=plan_verb_name,
        recall_cfg=recall_cfg,
        profile=base_profile,
        profile_source=profile_source,
        chat_quick_recall_profile=chat_quick_recall_profile,
        chat_kids_story_recall_profile=chat_kids_story_recall_profile,
    )
```

5. Update **existing** test `test_chat_quick_clamps_inherited_profile_to_low_latency_default` assertion:

```python
    assert decision["profile_source"] == "fast_chat_latency_default"
```

- [ ] **Step 4: Run full recall_delivery_gating module**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-cortex-exec && PYTHONPATH=. ../../venv/bin/python -m pytest tests/test_recall_delivery_gating.py -q --tb=short
```

Expected: all passed.

- [ ] **Step 5: Commit (optional)**

```bash
git add services/orion-cortex-exec/app/recall_utils.py services/orion-cortex-exec/tests/test_recall_delivery_gating.py
git commit -m "$(cat <<'EOF'
feat(cortex-exec): fast-chat recall clamp and RPC cap for chat_kids_story

Generalize profile clamp and quick Recall timeout to FAST_SINGLE_PASS_CHAT_VERBS.
EOF
)"
```

---

### Task 3: settings + .env_example

**Files:**

- Modify: `services/orion-cortex-exec/app/settings.py`
- Modify: `services/orion-cortex-exec/.env_example`

- [ ] **Step 1: Add setting**

After `chat_quick_recall_profile` field in `settings.py`, add:

```python
    chat_kids_story_recall_profile: str = Field("chat.story.kids.v1", alias="CHAT_KIDS_STORY_RECALL_PROFILE")
```

- [ ] **Step 2: Document in .env_example**

Near `CHAT_QUICK_RECALL_PROFILE` comments, add:

```bash
# Default recall profile for chat_kids_story when profile is not client-explicit (vector-off; see orion/recall/profiles).
# CHAT_KIDS_STORY_RECALL_PROFILE=chat.story.kids.v1
```

- [ ] **Step 3: Commit (optional)**

```bash
git add services/orion-cortex-exec/app/settings.py services/orion-cortex-exec/.env_example
git commit -m "$(cat <<'EOF'
feat(cortex-exec): CHAT_KIDS_STORY_RECALL_PROFILE setting

EOF
)"
```

---

### Task 4: router + supervisor pass new profile into delivery_safe

**Files:**

- Modify: `services/orion-cortex-exec/app/router.py` (recall_policy call ~704–713; brain reply branch ~758–766)
- Modify: `services/orion-cortex-exec/app/supervisor.py` (recall_policy call ~1319–1328)

- [ ] **Step 1: Router kwargs**

In `delivery_safe_recall_decision(...)` add:

```python
            chat_kids_story_recall_profile=settings.chat_kids_story_recall_profile,
```

Extend the `if str(plan.verb_name or "").strip().lower() == "chat_quick":` block to:

```python
            verb_lc = str(plan.verb_name or "").strip().lower()
            if verb_lc == "chat_quick":
                opts_now = ctx.get("options") if isinstance(ctx.get("options"), dict) else {}
                hub_full = bool(opts_now.get("chat_quick_full_stance"))
                if hub_full:
                    prepare_brain_reply_context(ctx)
                else:
                    prepare_chat_quick_reply_context(ctx)
            elif verb_lc == "chat_kids_story":
                prepare_chat_quick_reply_context(ctx)
```

- [ ] **Step 2: Supervisor kwargs**

Mirror `chat_kids_story_recall_profile=settings.chat_kids_story_recall_profile` in `supervisor.py` `delivery_safe_recall_decision` call.

- [ ] **Step 3: Run router tests that touch recall (smoke)**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-cortex-exec && PYTHONPATH=. ../../venv/bin/python -m pytest tests/test_router_chat_quick_recall_rpc_cap.py -q --tb=short
```

- [ ] **Step 4: Commit (optional)**

---

### Task 5: executor.py wiring (import, tokens, metacog, situation, recall clamp, llm_route, brain-skip)

**Files:**

- Modify: `services/orion-cortex-exec/app/executor.py`

- [ ] **Step 1: Imports**

Replace:

```python
from .recall_utils import apply_chat_quick_recall_profile_clamp, resolve_profile, resolve_recall_bus_rpc_wait_sec
```

with:

```python
from orion.cognition.fast_chat_verbs import FAST_SINGLE_PASS_CHAT_VERBS

from .recall_utils import apply_fast_chat_recall_profile_clamp, resolve_profile, resolve_recall_bus_rpc_wait_sec
```

- [ ] **Step 2: `_resolve_llm_chat_max_tokens`**

After the `chat_quick` block, add:

```python
    if step.verb_name == "chat_kids_story":
        return int(settings.llm_chat_quick_max_tokens), requested, "settings.llm_chat_quick_max_tokens"
```

- [ ] **Step 3: `_resolve_llm_max_tokens`**

After `if step.verb_name == "chat_quick":` block (~992), add:

```python
    if step.verb_name == "chat_kids_story":
        return max(1, int(settings.llm_chat_quick_max_tokens)), "quick_default", requested
```

- [ ] **Step 4: Situation / journal logging gate (~2164)**

Change:

```python
        if step.verb_name in {"chat_general", "chat_quick"}:
```

to:

```python
        if step.verb_name in {"chat_general", "chat_quick", "chat_kids_story"}:
```

- [ ] **Step 5: Metacog timeout (~2713)**

Change:

```python
                _metacog_rpc_timeout = 6.0 if str(step.verb_name or "") == "chat_quick" else 20.0
```

to:

```python
                _metacog_rpc_timeout = 6.0 if str(step.verb_name or "") in FAST_SINGLE_PASS_CHAT_VERBS else 20.0
```

- [ ] **Step 6: Inline Recall clamp (~3008)**

Replace `apply_chat_quick_recall_profile_clamp` call with:

```python
                resolved_profile, profile_source = apply_fast_chat_recall_profile_clamp(
                    plan_verb_name=str(plan_verb) if plan_verb is not None else None,
                    recall_cfg=recall_cfg,
                    profile=resolved_profile,
                    profile_source=profile_source,
                    chat_quick_recall_profile=settings.chat_quick_recall_profile,
                    chat_kids_story_recall_profile=settings.chat_kids_story_recall_profile,
                )
```

- [ ] **Step 7: Default `llm_route` (~3143)**

Change:

```python
                        else "quick"
                        if step.verb_name in {"chat_quick", "introspect_spark"}
```

to:

```python
                        else "quick"
                        if step.verb_name in FAST_SINGLE_PASS_CHAT_VERBS or step.verb_name == "introspect_spark"
```

- [ ] **Step 8: `_should_prepare_brain_reply_context` (~3578)**

Change:

```python
    if verb_name == "chat_quick":
        return False
```

to:

```python
    if verb_name in FAST_SINGLE_PASS_CHAT_VERBS:
        return False
```

- [ ] **Step 9: Run executor-heavy tests**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-cortex-exec && PYTHONPATH=. ../../venv/bin/python -m pytest tests/test_chat_general_route_mapping.py tests/test_executor_runtime_context_skip.py tests/test_recall_bus_rpc_timeout.py -q --tb=short
```

- [ ] **Step 10: Commit (optional)**

---

### Task 6: orch execution lane + test

**Files:**

- Modify: `services/orion-cortex-orch/app/execution_lanes.py`
- Modify: `services/orion-cortex-orch/tests/test_execution_lanes.py`

- [ ] **Step 1: Write failing test**

Append:

```python
def test_resolve_chat_kids_story() -> None:
    d = resolve_execution_lane(_req(verb="chat_kids_story"))
    assert d.lane == "chat"
    assert d.reason == "verb_chat"
```

(Use the same `_req` helper pattern as existing tests in that file.)

- [ ] **Step 2: Implement lane**

At top of `execution_lanes.py` add:

```python
from orion.cognition.fast_chat_verbs import FAST_SINGLE_PASS_CHAT_VERBS
```

Replace:

```python
    if verb in {"chat_general", "chat_quick"}:
```

with:

```python
    if verb == "chat_general" or verb in FAST_SINGLE_PASS_CHAT_VERBS:
```

- [ ] **Step 3: pytest**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-cortex-orch && PYTHONPATH=. ../../venv/bin/python -m pytest tests/test_execution_lanes.py -q --tb=short
```

- [ ] **Step 4: Commit (optional)**

---

### Task 7: Verb YAML + recall profile + Jinja prompt

**Files:**

- Create: `orion/cognition/verbs/chat_kids_story.yaml`
- Create: `orion/recall/profiles/chat.story.kids.v1.yaml`
- Create: `orion/cognition/prompts/chat_kids_story.j2`

- [ ] **Step 1: Create `orion/cognition/verbs/chat_kids_story.yaml`**

```yaml
name: chat_kids_story
label: Chat (Kids Story)
description: >
  Fast single-pass storytelling lane for Oríon: lightweight identity grounding,
  SQL-heavy recall (no vector), optional memory cards for listener facts; no stance-brief synthesis.

category: ExecutiveControl
priority: medium

interruptible: true
can_interrupt_others: false

requires_gpu: true
requires_memory: true

recall_profile: chat.story.kids.v1

timeout_ms: 180000
max_recursion_depth: 0

services:
  - LLMGatewayService

prompt_template: chat_kids_story.j2

personality_file: orion/cognition/personality/orion_identity.yaml

plan:
  - name: collect_metacog_context
    description: >
      Hydrate current spark/turn-effect context so story turns can propagate tension signals.
    order: 0
    services:
      - MetacogContextService
  - name: llm_chat_kids_story
    description: >
      Single-pass kids storytelling with anchors, recall digest, and multi-listener fairness.
    order: 1
    prompt_template: chat_kids_story.j2
    services:
      - LLMGatewayService
    requires_gpu: true
    requires_memory: true

safety_rules: []
```

- [ ] **Step 2: Create `orion/recall/profiles/chat.story.kids.v1.yaml`**

```yaml
profile: chat.story.kids.v1
render_transcript_user_only: true
vector_top_k: 0
rdf_top_k: 0
max_per_source: 4
max_total_items: 10
time_decay_half_life_hours: 48
render_budget_tokens: 220
enable_query_expansion: true
enable_sql_timeline: true
sql_since_minutes: 10080
sql_top_k: 8
cards_top_k: 4
relevance:
  backend_weights:
    vector: 0.0
    sql_timeline: 0.9
    sql_chat: 0.85
    rdf_chat: 0.0
    rdf: 0.0
    cards: 0.45
  score_weight: 0.72
  text_similarity_weight: 0.12
  recency_weight: 0.08
  enable_recency: true
  recency_half_life_hours: 72
```

- [ ] **Step 3: Create `orion/cognition/prompts/chat_kids_story.j2`**

```jinja
{# chat_kids_story template_revision=2026-05-14 — no real child names or interests in repo #}
You are Oríon.

This is the **kids storytelling** lane (fast single pass). Do not synthesize a formal stance brief.
Use the lightweight identity context below. Speak as yourself — warm, imaginative, clear — while
telling a story that lands for **children roughly ages six through nine** when that age band fits the audience.

LIGHTWEIGHT IDENTITY CONTEXT
- user_message: {{ user_message }}
- message_history: {{ message_history }}
- memory_digest: {{ memory_digest }}
- orion_identity_summary: {{ orion_identity_summary }}
- juniper_relationship_summary: {{ juniper_relationship_summary }}
- response_policy_summary: {{ response_policy_summary }}
{% if situation_prompt_fragment %}
- situation_prompt_fragment: {{ situation_prompt_fragment }}
{% endif %}
{% if world_context_capsule %}
- world_context_capsule: {{ world_context_capsule }}
{% endif %}

STORY TASK
- You are telling an **oral story** for the household, not lecturing. One cohesive narrative reply.
- If Juniper gave **anchors** (characters, places, props, “start with…”), weave them in early and pay them off.
- **memory_digest** may include prior story turns, timeline snippets, or structured listener facts from Recall.
  Use them as soft continuity — do not contradict obvious prior beats unless Juniper retcons out loud.
- If **memory_digest is empty**, continue from `user_message` and `message_history` only. Do **not** claim you lack memory systems.

MULTI-LISTENER FAIRNESS
- When digest or history implies **more than one child listener**, avoid excluding one for long stretches.
  Rotate focal moments, or use an ensemble frame (“you two…”) without sounding like a classroom lecture.

INTERESTS AND “SCARY” THEMES
- If interests mention spooky or scary media tastes, keep stories at **mild spooky adventure** — curiosity,
  mystery, courage — **no gore, no on-page harm to kids, no adult horror tone.**

FORBIDDEN
- Generic assistant voice (“As an AI…”).
- Refusals that deny `message_history` or `memory_digest` when they are present.
- Inventing private facts about real people not supported by the supplied context.
- Pasting long policy text; keep the story primary.

OUTPUT
Produce exactly one user-facing reply (the story or story segment).
```

- [ ] **Step 4: Commit (optional)**

---

### Task 8: test_chat_kids_story_plumbing.py + route_mapping test + recall_bus test

**Files:**

- Create: `services/orion-cortex-exec/tests/test_chat_kids_story_plumbing.py`
- Modify: `services/orion-cortex-exec/tests/test_chat_general_route_mapping.py`
- Modify: `services/orion-cortex-exec/tests/test_recall_bus_rpc_timeout.py`

- [ ] **Step 1: Create plumbing test**

`services/orion-cortex-exec/tests/test_chat_kids_story_plumbing.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml


def test_chat_kids_story_plan_collects_metacog_then_llm() -> None:
    doc = yaml.safe_load(Path("orion/cognition/verbs/chat_kids_story.yaml").read_text(encoding="utf-8"))
    assert doc["name"] == "chat_kids_story"
    steps = sorted(doc["plan"], key=lambda s: s["order"])
    assert steps[0]["name"] == "collect_metacog_context"
    assert steps[1]["name"] == "llm_chat_kids_story"
    assert steps[1]["prompt_template"] == "chat_kids_story.j2"
    assert doc.get("recall_profile") == "chat.story.kids.v1"


def test_chat_kids_story_prompt_has_identity_placeholders() -> None:
    text = Path("orion/cognition/prompts/chat_kids_story.j2").read_text(encoding="utf-8")
    for needle in (
        "message_history",
        "memory_digest",
        "MULTI-LISTENER",
        "SCARY",
    ):
        assert needle in text, f"missing {needle!r} in chat_kids_story.j2"
```

- [ ] **Step 2: Append route test** to `test_chat_general_route_mapping.py`:

```python
def test_chat_kids_story_step_uses_quick_route() -> None:
    step = ExecutionStep(
        step_name="llm_chat_kids_story",
        verb_name="chat_kids_story",
        services=["LLMGatewayService"],
        order=1,
        prompt_template="{{ raw_user_text }}",
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    with patch("app.executor.LLMGatewayClient.chat", new=AsyncMock(return_value=ChatResponsePayload(content="once"))) as llm_chat:
        result = asyncio.run(
            call_step_services(
                bus=MagicMock(),
                source=source,
                step=step,
                ctx=_base_ctx(),
                correlation_id=str(uuid4()),
            )
        )

    assert result.status == "success"
    sent_req = llm_chat.await_args.kwargs["req"]
    assert sent_req.route == "quick"
    assert sent_req.options["max_tokens"] == 384
```

- [ ] **Step 3: Extend `test_recall_bus_rpc_timeout.py`**

Add:

```python
def test_resolve_recall_bus_wait_chat_kids_story_uses_quick_cap() -> None:
    w = resolve_recall_bus_rpc_wait_sec(
        verb="chat_kids_story",
        step_timeout_ms=120_000,
        rpc_timeout_override=None,
        recall_rpc_timeout_sec=90.0,
        chat_quick_recall_rpc_timeout_sec=33.0,
    )
    assert w == 33.0
```

- [ ] **Step 4: Run tests**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-cortex-exec && PYTHONPATH=. ../../venv/bin/python -m pytest tests/test_chat_kids_story_plumbing.py tests/test_chat_general_route_mapping.py::test_chat_kids_story_step_uses_quick_route tests/test_recall_bus_rpc_timeout.py -q --tb=short
```

- [ ] **Step 5: Commit (optional)**

---

### Task 9: executor runtime skip test + chat_prompt_context (if needed)

**Files:**

- Modify: `services/orion-cortex-exec/tests/test_executor_runtime_context_skip.py`

- [ ] **Step 1: Add test**

```python
def test_chat_kids_story_skips_heavy_brain_reply_context_prep():
    step = ExecutionStep(
        verb_name="chat_kids_story",
        step_name="llm_chat_kids_story",
        order=1,
        services=["LLMGatewayService"],
    )
    assert _should_prepare_brain_reply_context(step=step, ctx={"mode": "brain"}) is False
```

- [ ] **Step 2: Run**

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-cortex-exec && PYTHONPATH=. ../../venv/bin/python -m pytest tests/test_executor_runtime_context_skip.py -q --tb=short
```

- [ ] **Step 3: Commit (optional)**

---

### Task 10: orion-actions route normalization

**Files:**

- Modify: `services/orion-actions/app/main.py`

- [ ] **Step 1: Extend `_normalized_llm_route`**

Change:

```python
    if route in {"chat_quick", "quick_chat"}:
        return "quick"
```

to:

```python
    if route in {"chat_quick", "quick_chat", "chat_kids_story"}:
        return "quick"
```

- [ ] **Step 2: Commit (optional)**

---

### Task 11: Hub README (v1 contract)

**Files:**

- Modify: `services/orion-hub/README.md`

- [ ] **Step 1: Add subsection** under the chat / brain area (pick the section that documents modes):

Add ~6 lines:

```markdown
### Kids story lane (verb override)

Send a single explicit verb: `verbs: ["chat_kids_story"]` with normal `messages` / recall options.
Verb must be active on the hub node (`orion/cognition/verbs/active.yaml`). Default recall profile is
`chat.story.kids.v1` (vector-off; SQL + timeline + optional cards) unless the client sets `profile_explicit`.
```

- [ ] **Step 2: Commit (optional)**

---

## Plan self-review (against spec)

| Spec requirement | Task covering it |
|------------------|------------------|
| Verb mirrors `chat_quick` plan | Task 7 YAML |
| Vector-off recall profile in repo | Task 7 profile YAML |
| `FAST_SINGLE_PASS_CHAT_VERBS` | Tasks 1, 2, 4, 5, 6 |
| `CHAT_KIDS_STORY_RECALL_PROFILE` | Task 3 |
| Same Recall RPC cap as quick | Tasks 2, 8 |
| Router lightweight reply context | Task 4 |
| Bounded autonomy fanout | Task 1 |
| No PII in repo / synthetic tests | Tasks 7–8 (prompt wording + tests use no names) |
| Hub v1 verb override documented | Task 11 |

**Placeholder scan:** None intentional; operator listener card convention remains documented in spec (`listener_profile` / `story_listener`) — implementation of card authoring is **out of scope** for this plan (YAGNI).

**Type / name consistency:** Verb string `chat_kids_story`; step `llm_chat_kids_story`; profile `chat.story.kids.v1`; setting field `chat_kids_story_recall_profile`.

---

## Execution handoff

**Plan complete and saved to** `docs/superpowers/plans/2026-05-14-chat-kids-story-verb.md`.

You asked for **subagent-driven** execution.

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task; review between tasks. **REQUIRED SUB-SKILL:** superpowers:subagent-driven-development.

**2. Inline Execution** — Run tasks sequentially in this session with checkpoints. **REQUIRED SUB-SKILL:** superpowers:executing-plans.

**Which approach?** (You already indicated subagent — start with **1** unless you prefer a single long session.)
