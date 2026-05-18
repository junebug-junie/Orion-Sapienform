# PR: Chat general — withhold identity kernels from ordinary speech turns

**Branch:** `fix/chat-general-speech-kernel-suppress`  
**Base:** `main` (includes merged #585 identity-recital-over-indexing)

## Summary

After #585, hub turns like *"k, did some edits to your chat profile"* still opened with canned recital:

> You're Juniper — the one building this with me. I see the edits.

**Root cause:** `llm_chat_general` still rendered full `orion_identity_summary` / `juniper_relationship_summary` prose on every turn. The model reconstructed the removed prompt examples from kernel text.

**Fix:** Withhold identity kernels from speech render on ordinary turns; align stance synthesizer; router strips leading recital as safety net.

## Changes

| Area | What |
|------|------|
| `executor.py` | `suppress_chat_general_speech_identity_priming()` before `llm_chat_general` render |
| `chat_stance.py` | Speech suppress helper; neutral `self_relevance`/`juniper_relevance` on ordinary fallback; enforce scrub; `strip_identity_recital_leadin()` |
| `chat_general.j2` | Conditional identity kernel blocks; chat-profile edit rule |
| `chat_stance_brief.j2` | Ordinary turns: low salience, empty facets, avoid recital tags |
| `router.py` | Strip leading Juniper/Oríon label sentences on non-identity turns |

## Verification

```bash
PYTHONPATH=. ./venv/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_chat_general_stance_plumbing.py \
  services/orion-cortex-exec/tests/test_router_identity_boundary.py -q
```

**26 passed**

## Test plan

- [ ] `"k, did some edits to your chat profile"` → acknowledges edits, no Juniper label recital
- [ ] `"who are you?"` → identity answer still works; kernels present
- [ ] Hub refresh / `"hey"` → no self-intro

## Deploy

Rebuild **cortex-exec** (and hub if it bundles exec). #585 alone is insufficient.
