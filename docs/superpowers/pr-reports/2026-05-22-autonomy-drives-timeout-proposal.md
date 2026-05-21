# Proposal: Autonomy drives timeout (Orion Fuseki)

**Status:** Partially implemented — see `2026-05-22-autonomy-drives-automation.md` (branch `feat/autonomy-drives-automation`)  
**Companion PR:** goals SPARQL per-`drive_origin` aggregate (`feat/autonomy-goals-sparql-per-origin`)

---

## Problem (plain)

Orion’s graph is **huge** (old drive audits + goals). We already ask for “**latest** drive report only,” but Fuseki still has to **search the pile** to find it. That search hits the **20 second** limit on Orion; relationship barely finishes in time.

Chat correctly uses relationship instead (`fallback_contextual`). The code is working; the **attic is full**.

---

## What we already did (don’t redo)

- Latest-audit drives query shape (`_fetch_drive_audit`)
- Smaller limit on chat path (`AUTONOMY_CHAT_STANCE_DRIVES_QUERY_LIMIT=20`)
- Fallback to relationship when Orion fails
- Manual archive script (operator must run `--apply`)

---

## Options (best first)

### A. Automate cleanup (do this first)

| Action | Why |
|--------|-----|
| Nightly cron: `archive_stale_goal_proposals.py --apply` | Shrinks graph → drives + goals faster |
| One-time `--apply` for orion + relationship | Immediate win |
| Optional: archive after concept-induction tick (feature flag) | Stays clean without you |

**Risk:** auto-delete needs a flag + logs; keep CLI dry-run default.

---

### B. Smarter chat policy (code, ~1–2 days)

- **Skip Orion drives** when we already know we’ll use relationship (saves 20s/wait)
- **Shorter drives-only timeout** (e.g. 12s) so fallback happens faster
- Don’t query all 3 subjects on every `chat_general` when relationship is enough

---

### C. Fuseki index / “latest pointer” on write (bigger)

Index by subject + timestamp, or write `orion:latestDriveAudit` when materializing. Phase 2 if A+B aren’t enough.

---

### D. Raise timeout to 45s (band-aid)

**Don’t** rely on this alone — every chat turn gets slower.

---

## Recommended order

1. **Run archive** (dry-run → apply)  
2. **Ship goals SPARQL fix** (separate branch — per-origin aggregate)  
3. **Cron archive**  
4. **Phase 1b:** skip/defer Orion drives when contextual fallback is likely  
5. **Phase 2:** index / latest-pointer if still slow  

---

## Success = 

- Orion drives p95 **< 10s** (at 20s budget)  
- Goals subquery **< 3s**, `row_count` not 12+  
- Orion drives timeout **< 5%** on chat_general after archive  

---

## Sign-off

- [ ] Archive automation approved  
- [ ] Phase 1b facet skip approved  
- [ ] Phase 2 index deferred or approved  
