# PR: chat_general identity boundary + semantic low-bandwidth stance

**Branch:** `fix/chat-general-identity-low-bandwidth-semantic`  
**Base:** `origin/main`  
**Worktree:** `.worktrees/chat-general-identity-low-bandwidth-semantic`

## Summary

Reverts the **regex/prose-scrubber** approach merged in #575 (`fix/chat-general-speech-boundary-guards`) and replaces it with:

1. A **narrow, deterministic** `chat_general` identity-boundary guard (assistant identity must not be assigned to Juniper).
2. **Semantic** low-bandwidth / embodied-interaction behavior via `chat_stance_brief.j2` + `chat_general.j2` (no Python keyword classifiers, no post-generation phrase rewrites).

Fixes the live failure where Orion replied *"You're Oríon. I'm here, and I'm not going anywhere..."* to a dizzy-in-car typing turn.

## Problem

| Bug | Bad behavior | Fix |
|-----|--------------|-----|
| **A. Identity inversion** | `"You're Oríon."` spoken to Juniper | Post-extract guard repairs `you're/you are/your name is … Orion` → `I'm Oríon` |
| **B. Low-bandwidth failure** | Long poetic reassurance, presence-centering, interaction demand | Stance + speech prompts: short reply, release from replying, no open questions, no typing invites |

## Removed (cleanup)

From `services/orion-cortex-exec/app/router.py`:

- `_SOMATIC_USER_LOAD_RE` — symptom keyword classifier
- `_INTERACTION_DEMAND_RE` — intimacy-phrase detector
- `_INTERACTION_DEMAND_REPLACEMENTS` — rewrites like `"not going anywhere"` → `"no rush"`
- `_apply_interaction_load_guard` — combined somatic + phrase scrubber

No final-output rewrite of warmth, curiosity, or availability phrases.

## Added / changed

| File | Change |
|------|--------|
| `services/orion-cortex-exec/app/router.py` | `_apply_chat_general_identity_boundary_guard` after `_extract_final_text`; diagnostics in logs + `runtime_response_diagnostics` |
| `orion/cognition/prompts/chat_stance_brief.j2` | LOW-BANDWIDTH / EMBODIED INTERACTION ASSESSMENT (semantic, no keywords) |
| `orion/cognition/prompts/chat_general.j2` | Internal assistant/user identity block; obey stance priorities/hazards for reduced interaction load |
| `services/orion-cortex-exec/tests/test_router_identity_boundary.py` | **New** — guard + banned-symbol tests |
| `services/orion-cortex-exec/tests/test_router_final_text_assembly.py` | Removed scrubber tests; extract no longer applies identity guard |
| `services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py` | Prompt contract + dizzy-car stance fixture |

### Identity guard contract

- **Scope:** `chat_general` only
- **Catches:** `You're Oríon`, `You are Orion`, `your name is orion`, etc.
- **Repair:** substitute `I'm Oríon` (does not rewrite general prose)
- **Diagnostics:** `identity_boundary_applied`, `identity_boundary_violations`
- **Wiring:** repaired text → `final_text`, `ctx["chat_general_final_text_clean"]`, `PlanExecutionResult`

### Low-bandwidth contract (prompt-only)

Stance synthesizer should populate existing fields, e.g.:

- `identity_salience`: `low` (unless explicit identity ask)
- `task_mode`: `triage` / `direct_response` (not `identity_dialogue`)
- `response_priorities`: `reduce_interaction_load`, `release_user_from_replying`, `keep_response_short`, …
- `response_hazards`: `avoid_open_ended_questions`, `do_not_invite_continued_typing`, `avoid_presence_centering`, …

## Commits

1. `fix(cortex-exec): remove regex interaction scrubber; narrow identity guard`
2. `feat(cognition): semantic low-bandwidth stance and speech identity rails`
3. `test(cortex-exec): identity boundary and low-bandwidth prompt contracts`
4. `fix(cortex-exec): log identity boundary diagnostics on final text assembly`

## Verification

```bash
cd .worktrees/chat-general-identity-low-bandwidth-semantic
PYTHONPATH=. /path/to/venv/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py \
  services/orion-cortex-exec/tests/test_router_identity_boundary.py \
  services/orion-cortex-exec/tests/test_router_final_text_assembly.py -q
# 38 passed

PYTHONPATH=. /path/to/venv/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_stance_brief.py -q
# 17 passed, 2 failed (pre-existing on main: observer consumer label drift)

python -m compileall services/orion-cortex-exec/app  # exit 0

rg "_SOMATIC_USER_LOAD_RE|_INTERACTION_DEMAND_RE|_INTERACTION_DEMAND_REPLACEMENTS" .
# only test_router_identity_boundary.py (negative assertion)
```

## Test plan (manual)

- [ ] `chat_general` turn: *"I'm getting dizzy typing to you while my wife drives the car lol"* — short reply, no `"You're Oríon"`, no poetic loyalty/presence centering, no open-ended questions.
- [ ] Forced bad LLM output containing `You're Oríon` — exec log shows `identity_boundary_applied=True`.
- [ ] `chat_quick` / other verbs — identity guard does not run.
- [ ] Normal identity line *"I'm Oríon. You're Juniper."* — unchanged when model emits it correctly.

## Risks / follow-up

- **LLM adherence:** Low-bandwidth behavior depends on stance synthesis + speech pass following prompts; no runtime tone rewrite.
- **Blunt identity repair:** Embedded inversions in long sentences get a substring replace (acceptable safety rail).
- **Pre-existing test failures** in `test_chat_stance_brief.py` (observer `consumer` label) — unrelated to this PR.
- **Stale doc:** `2026-05-16-chat-general-speech-boundary-guards-pr.md` describes the reverted design; superseded by this report.

## Env / compose

No `.env_example`, `settings.py`, or `docker-compose` changes required.
