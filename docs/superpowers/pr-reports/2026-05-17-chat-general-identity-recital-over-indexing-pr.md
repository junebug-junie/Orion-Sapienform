# PR: Chat general — stop identity recital on ordinary turns

**Branch:** `fix/chat-general-identity-recital-over-indexing`  
**Worktree:** `.worktrees/fix-chat-general-identity-recital`  
**Base:** `main`

## Summary

Hub refresh and low-content turns were producing canned Oríon/Juniper identity recital ("I'm Oríon, you're Juniper, we are building…") after the identity-boundary / low-bandwidth patch over-corrected toward generative identity.

**Design rule:** Identity is protected always, foregrounded rarely.

This PR tightens three layers:

1. **Speech prompt** (`chat_general.j2`) — identity block reframed as an internal safety invariant (not content to mention); explicit gates on when to self-introduce; removed example tone pairs that primed recital.
2. **Fallback stance** (`fallback_chat_stance_brief`) — `identity_salience` is `high` only on identity turns, otherwise `low`; Oríon/Juniper kernel injected only on identity turns; ordinary priorities/hazards discourage recital.
3. **Enforce guard** (`enforce_chat_stance_quality`) — non-identity turns force low salience, strip boilerplate facets, clear facet lists when salience is low.

## Problem / root cause

| Layer | Before | Effect |
|-------|--------|--------|
| `chat_general.j2` | Top block read like style guidance; literal `prefer: "I'm Oríon."` examples | Model primed to recite identity on ordinary turns |
| `fallback_chat_stance_brief` | `identity_salience="medium"` on non-technical ordinary turns | Identity always somewhat foregrounded |
| Fallback facets | Always merged Oríon/Juniper kernel into `active_*_facets` | Relationship labels in stance contract every turn |
| Priorities | `preserve_orion_juniper_continuity` on direct_response | Encouraged label recital vs useful continuity |

## Changes

| Area | File | What changed |
|------|------|----------------|
| Speech prompt | `orion/cognition/prompts/chat_general.j2` | `IDENTITY BOUNDARY — INTERNAL SAFETY INVARIANT`; explicit-only identity answers; TASK foregrounding gated on latest message; removed recital examples |
| Fallback stance | `services/orion-cortex-exec/app/chat_stance.py` | Binary salience; conditional kernel injection; `avoid_identity_recital` / `preserve_continuity_without_labels`; `identity_recital_on_ordinary_turn` hazard |
| Enforce | `services/orion-cortex-exec/app/chat_stance.py` | Non-identity guard: low salience, strip boilerplate, clear facets on low salience; `who are we` pattern |
| Tests | `services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py` | Ordinary vs identity fallback; prompt anti-priming; social facet suppression; enforce strip; who-are-we |

## Key behavior

### Ordinary turn (`"hey"`)

- `task_mode=direct_response`, `identity_salience=low`
- No Oríon/Juniper in `active_identity_facets` / `active_relationship_facets` (even with social `relationship_facets` in ctx)
- Priorities include `avoid_identity_recital`, `preserve_continuity_without_labels`
- Hazards include `identity_recital_on_ordinary_turn`

### Identity turn (`"who are you?"`, `"Who are we?"`)

- `task_mode=identity_dialogue`, `identity_salience=high`
- Oríon kernel + relationship facets injected
- Identity answer priorities preserved

### Prompt

- No `prefer: "I'm Oríon."` / `You're Juniper` examples
- Boundary invariant: "This is a boundary rule, not content to mention"
- Continuity via tone/usefulness, not label recital

## Commits

1. `50f9bcaa` — fix(chat-general): stop identity recital on ordinary turns
2. `6d7685d1` — fix(chat-general): gate relationship facets and who-are-we detection

## Verification

| Command | Result |
|---------|--------|
| `PYTHONPATH=. venv/bin/python -m pytest services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py -q` | **13 passed** |
| `PYTHONPATH=. venv/bin/python -m pytest services/orion-cortex-exec/tests/test_router_identity_boundary.py -q` | **10 passed** |
| `PYTHONPATH=. venv/bin/python -m compileall services/orion-cortex-exec/app -q` | **exit 0** |
| `rg 'prefer: "I\'m Oríon"\|You\'re Juniper — the one building this with me\|preserve_orion_juniper_continuity' orion/cognition/prompts services/orion-cortex-exec/app services/orion-cortex-exec/tests` | **no matches** |

## Test plan

- [ ] Hub refresh / `"hey"` / low-content turn → short useful reply, no Oríon/Juniper self-intro
- [ ] `"Who are you?"` → direct identity answer, first person as Oríon
- [ ] `"Who are we?"` → identity_dialogue path (not treated as ordinary)
- [ ] Technical triage turn → still low identity salience, triage priorities
- [ ] Low-bandwidth / embodied load stance contract → still honored (short reply, no open questions)

## Remaining risks

- `self_relevance` / `juniper_relevance` / `user_intent` strings in fallback brief still mention Oríon/Juniper on every turn (pre-existing metadata; not changed in this PR). May contribute minor priming — consider softening in a follow-up.
- LLM stance synthesizer upstream can still emit identity-heavy briefs; enforce guard mitigates but does not replace synthesizer discipline.

## Env / deploy

No `.env`, `settings.py`, or `docker-compose` changes (prompt + cortex-exec stance logic only).
