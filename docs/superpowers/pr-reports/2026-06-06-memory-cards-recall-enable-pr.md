# PR: Enable memory cards recall rail + Hub review metadata editors

## Summary

- Turn on the memory cards Postgres recall backend (`RECALL_ENABLE_CARDS=true`) in orion-recall env, settings default, and docker-compose wiring.
- Add `cards_top_k` and `backend_weights.cards` to every shipped recall profile so the cards rail can activate under any profile (weights tuned by profile intent).
- Restrict cards recall to **`status=active`** rows only — primary query and 1-hop neighbors (code review caught pending_review leak via edges; fixed in commit 4aa8415b).
- Hub Memory tab: review/all card detail panels now expose editable metadata (confidence, sensitivity, visibility, priority, provenance, evidence, still_true, time_horizon) with **Save metadata** (`PATCH`) and **Approve/Reject** that save first.
- Hub chat dropdown: add `biographical.v1` and `self.factual.v1` recall profiles.

## Context / honest limits

Cards recall scoring remains **lexical regex token overlap** — not embeddings, not LLM. It is a cheap supplement for operator-curated facts with explicit anchors/tags, not a replacement for RDF/SQL recall. Enabling it makes curated facts *reachable* when query tokens overlap; quality still depends on card text and operator metadata (especially `priority=always_inject` via cortex-orch for must-carry facts).

Memory graph **approve** writes RDF to **Fuseki** when `RDF_STORE_GRAPH_STORE_URL` + `RDF_STORE_UPDATE_URL` are set (`MEMORY_GRAPH_APPROVAL_BACKEND=auto`). GraphDB is legacy-only (`MEMORY_GRAPH_APPROVAL_BACKEND=graphdb`).

## Files touched

| Area | Changes |
|------|---------|
| `services/orion-recall/.env_example` | `RECALL_ENABLE_CARDS=true` + cards timeout/neighbor knobs |
| `services/orion-recall/app/settings.py` | default `RECALL_ENABLE_CARDS=True` |
| `services/orion-recall/docker-compose.yml` | pass cards env vars into container |
| `services/orion-recall/app/cards_adapter.py` | active-only SQL filter |
| `orion/recall/profiles/*.yaml` | `cards_top_k` + `cards` weight on all profiles |
| `services/orion-hub/static/js/memory.js` | metadata review editors + PATCH |
| `services/orion-hub/templates/index.html` | biographical/self.factual profile options |
| READMEs + tests | docs and regression tests |

## Test plan

- [x] `pytest tests/test_recall_profiles_cards_knobs.py` — all profiles declare cards knobs
- [x] `pytest services/orion-recall/tests/test_cards_adapter_active_only.py` — active-only filter
- [x] `pytest services/orion-hub/tests/test_memory_review_ui.py` — UI + template wiring
- [ ] Restart orion-recall with updated `.env`; confirm logs show memory cards pool when DSN set
- [ ] Seed + approve a card with overlapping tokens; run chat with `biographical.v1` and confirm `source=cards` in recall debug
- [ ] Hub Memory tab → Review queue → edit priority to `always_inject` → Approve → confirm `[Known facts]` block in cortex prompt (cortex-orch `RECALL_PG_DSN` required)
- [ ] Memory graph approve with Fuseki up → confirm named graph + pending_review cards in Postgres

## Local `.env` sync (not committed)

Updated `services/orion-recall/.env` on operator machine to match `.env_example` cards block.
