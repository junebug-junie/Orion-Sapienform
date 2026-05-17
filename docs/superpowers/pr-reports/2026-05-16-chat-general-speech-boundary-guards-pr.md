# PR: chat_general speech boundary guards (identity + somatic load)

**Branch:** `fix/chat-general-speech-boundary-guards`

## Summary

Fixes a live `chat_general` failure where Orion replied to Juniper with *"You're Oríon. I'm here, and I'm not going anywhere..."* — two bugs at the speech boundary:

1. **Identity boundary failure** — assistant identity was assigned to the user (`You're Oríon`).
2. **Somatic / interaction-load failure** — user reported dizziness while typing in a moving car; Orion invited continued conversational intimacy instead of lowering interaction demand.

Small, deterministic guards in `orion-cortex-exec` final-text assembly (no prompt/cathedral changes).

## Commits

| Commit | Description |
|--------|-------------|
| `fefc0b3e` | Identity + interaction-load guards + unit tests |
| *(HEAD)* | Review fixes: `plan_ctx_latest_user_text`, narrower somatic regex |

## Root cause

| Layer | Finding |
|-------|---------|
| **Prompt** | `chat_general.j2` opens with `You are Oríon.` — model can invert roles under load. |
| **Stance** | `enforce_chat_stance_quality` / `fallback_chat_stance_brief` guard generic-assistant drift, not role inversion or somatic load. |
| **Router** | `_extract_final_text` strips think/planning but had no deterministic guard for `You're Oríon` or intimacy-escalating closers under somatic user signals. |

## Changes

| File | Change |
|------|--------|
| `services/orion-cortex-exec/app/router.py` | `_apply_identity_boundary_guard` (chat_general only, in `_extract_final_text`); `_apply_interaction_load_guard` (after extract, uses `plan_ctx_latest_user_text(ctx)`); diagnostics keys on `final_text_diag`. |
| `services/orion-cortex-exec/tests/test_router_final_text_assembly.py` | Regression tests for identity variants, somatic load, `raw_user_text`-only user message, false-positive guard (`laptop in the car`). |

### Identity guard (minimum patterns)

Repairs to `I'm Oríon`:

- `You're Oríon` / `You're Orion` (straight + curly apostrophe)
- `You are Oríon` / `You are Orion`
- `youre orion` / `youre oríon`
- `your name is Orion`

Diagnostics: `identity_boundary_applied`, `identity_boundary_violations`.

### Interaction-load guard

When **user message** (via `plan_ctx_latest_user_text`) matches somatic/high-load signals (`dizzy`, `nauseous`, `carsick`, `moving car`, `while driving`, `hard to type`, …) **and** assistant text contains intimacy-demand phrases (`not going anywhere`, `tell me more`, …):

- Replaces `I'm here, and I'm not going anywhere` → `I'm here — no rush.`
- Strips remaining demand phrases; may append `Go easy — reply when you're steady.`

Diagnostics: `interaction_load_guard_applied`, `interaction_load_violations`.

**Not in scope:** prompt edits, stance-brief synthesis changes, env/docker toggles.

## Verification

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/chat-general-speech-boundary-guards
PYTHONPATH=. /mnt/scripts/Orion-Sapienform/venv/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_router_final_text_assembly.py -q --tb=short
```

- **Exit code:** 0  
- **Result:** 30 passed

## Test plan (live)

- [ ] Hub brain turn with identity question — Orion says `I'm Oríon`, never `You're Oríon`.
- [ ] Turn with somatic load (`dizzy`, `moving car`) — reply is brief, no `not going anywhere` / multi-question intimacy push.
- [ ] Exec logs / `final_text_diag`: `identity_boundary_applied` or `interaction_load_guard_applied` when guards fire.

## Remaining risks

- Guards are regex/heuristic; rare false positives/negatives possible on edge phrasing.
- **UNVERIFIED** against live stack in this session — closure requires a reproduced Hub turn or live verification script per `AGENTS.md`.
