# Orion Memory Cards v1 — offboarding / next-phase guide

**Audience:** Engineers continuing Memory Cards after the initial `feat/memory-cards-v1` land (Phases 0–3 + partial 4–5 + scaffolds).  
**Design source of truth:** [Memory Cards v1 design](../specs/2026-05-01-orion-memory-cards-v1-design.md).

This document maps **what exists today**, **what is stub or intentionally thin**, **how services connect**, and **ordered next-phase work** so the next builder does not reverse-engineer the branch from chat history alone.

---

## One-sentence recap

**Memory Cards** are operator-curated Postgres rows (`memory_cards` + edges + history) with shared **contracts + async DAL** under `orion/core/`; **recall** can score them as a `cards` backend when enabled; **cortex-orch** can prepend an always-on **known facts** block from the same DB; **Hub** exposes a **subset** of memory HTTP APIs behind `ensure_session` when `RECALL_PG_DSN` is set. Auto-extraction and distiller are **scaffolded**, not product-complete.

---

## Where the work lives

- **Target branch:** `feat/memory-cards-v1` (branched from `cognitive_substrate/v4` at implementation time).
- **If you use a worktree:** repo-local worktrees should live under `.worktrees/` (ignored in git). Merge or cherry-pick from `feat/memory-cards-v1` into your integration branch; do not assume the primary working tree still matches that branch.

Use:

```bash
git log --oneline feat/memory-cards-v1 -- \
  orion/core/contracts/memory_cards.py \
  orion/core/storage/ \
  services/orion-recall/app/cards_adapter.py \
  services/orion-hub/scripts/memory_routes.py
```

---

## Architecture (runtime)

```text
conjourney Postgres (RECALL_PG_DSN)
  ├── orion-recall: optional asyncpg pool when RECALL_ENABLE_CARDS → cards_adapter.fetch_* → fusion (source=cards)
  ├── orion-cortex-orch: memory_inject (psycopg2 sync) → context["known_facts_block"] in conversation_front
  └── orion-hub: optional asyncpg pool → /api/memory/* → DAL (orion.core.storage.memory_cards)
```

**Ground rules (do not violate without a new spec revision):** design §2 — no default profile churn except `self.factual.v1.yaml`, auto-extractor **off** by default, cards are **data** not identity mutation, Stage 2 LLM paths stay **NotImplemented** in v1.

---

## File inventory (as implemented)

### Contracts + storage (shared)

| Path | Role |
|------|------|
| `orion/core/contracts/memory_cards.py` | `MemoryCardV1`, edges, history, `visibility_allows_card`, type tokens, etc. |
| `orion/core/contracts/recall.py` | `RecallQueryV1.lane`, `profile_explicit`. |
| `orion/core/storage/__init__.py` | Re-exports DAL entrypoints. |
| `orion/core/storage/memory_cards.py` | Asyncpg DAL: schema apply (psycopg2), CRUD, edges, history, `reverse_history`, parent/child cycle guard on `add_edge`. |
| `orion/core/storage/memory_extraction.py` | **Thin Stage-1 stub** (`extract_candidates` / `fingerprint`) — expand per design §11 / §5 patterns. |
| `services/orion-recall/sql/memory_cards.sql` | Canonical DDL (matches design §4). |

### Recall service

| Path | Role |
|------|------|
| `services/orion-recall/app/cards_adapter.py` | Scoring, visibility filter, 1-hop neighbor fan-out, timeout wrapper. |
| `services/orion-recall/app/worker.py` | `lane` + `include_cards` on first expansion signal only; intent routing before `get_profile` when enabled; pool via `set_recall_pg_pool`. |
| `services/orion-recall/app/intent.py` | Keyword Stage-1 intent + `resolve_profile_for_intent`; telemetry payload helper. |
| `services/orion-recall/app/fusion.py` | `cards` in default backend weights; `allowed_sources` bypass for `cards` when cards rail is on in profile; render diagnostics `budget_dropped`. |
| `services/orion-recall/app/render.py` | Returns `(rendered, budget_dropped)`; optional `+N more items dropped` suffix. |
| `services/orion-recall/app/storage/vector_adapter.py` | Per-collection parallel fetch (thread pool), `RECALL_VECTOR_EXCLUDE_COLLECTIONS`, `meta.collection`, dedupe. |
| `services/orion-recall/app/settings.py` | `RECALL_ENABLE_CARDS`, timeouts, neighbors, intent flag, vector exclude regex, render budget indicator. |
| `services/orion-recall/app/main.py` | Lifespan: creates asyncpg pool when `RECALL_ENABLE_CARDS` + DSN, registers pool with worker. |

### Profiles

| Path | Role |
|------|------|
| `orion/recall/profiles/biographical.v1.yaml` | New profile per design §7. |
| `orion/recall/profiles/self.factual.v1.yaml` | **Only** `backend_weights.rdf` + `backend_weights.cards` changed per design. |

### Cortex orchestrator

| Path | Role |
|------|------|
| `services/orion-cortex-orch/app/memory_inject.py` | Sync psycopg2 read for `always_inject` + `active` + visibility; 250ms statement timeout; **not** the spec’s async `pool` signature (conversation front is sync). |
| `services/orion-cortex-orch/app/conversation_front.py` | `ChatTurnPayload.lane`; `known_facts_block` in context + prompt instructions. |
| `services/orion-cortex-orch/app/memory_extractor.py` | Hunter handler: **noop** when disabled; raises `NotImplementedError` if Stage 2 enabled. |
| `services/orion-cortex-orch/app/main.py` | Starts `memory_cards_hunter` on `orion:chat:history:turn`. |
| `services/orion-cortex-orch/app/settings.py` | `RECALL_PG_DSN`, always-inject budget/enable, auto-extractor flags + promote threshold. |
| `services/orion-cortex-orch/requirements.txt` | Adds `psycopg2-binary`. |

### Hub

| Path | Role |
|------|------|
| `services/orion-hub/scripts/memory_routes.py` | **Separate router** (not merged into giant `api_routes.py`): `POST/GET /api/memory/cards`, `GET /api/memory/cards/{id}`, `POST /api/memory/sessions/{id}/distill` → **501**, `POST /api/memory/history/{id}/reverse`. |
| `services/orion-hub/scripts/main.py` | Lifespan asyncpg pool when `RECALL_PG_DSN` set; shutdown closes pool; `include_router(memory_router)`. |
| `services/orion-hub/app/settings.py` | `RECALL_PG_DSN`. |
| `services/orion-hub/requirements.txt` | Adds `asyncpg`. |
| `services/orion-hub/.env_example` | Documents `RECALL_PG_DSN`. |

**Spec still calls for:** `PATCH` card, status endpoint, edges CRUD, history list, neighborhood endpoint, and **full** `api_routes.py` integration — implement or consciously keep split-router; update the design’s path table if you standardize on `memory_routes.py`.

### Scaffolds / tests / ops

| Path | Role |
|------|------|
| `services/orion-recall/scripts/distill_memory_cards.py` | **Argparse scaffold only** — wire DAL + JSONL per design §11. |
| `scripts/smoke_memory_cards_e2e.sh` | **Placeholder** exit 0 — replace with real E2E per design §12. |
| `tests/test_memory_cards_contracts.py` | Minimal contract tests. |
| `services/orion-recall/tests/test_intent_routing.py` | Basic intent routing tests (path hack for `app` imports). |
| `services/orion-cortex-orch/tests/test_memory_extractor.py` | Disabled path + Stage 2 `NotImplementedError`. |

**Spec-listed tests not yet added:** DAL roundtrip, cards backend scoring, visibility lane integration, `memory_inject`, Hub API, extraction tables, distiller CLI, `chat.general.v1` snapshot parity (design §12 / §13).

---

## Done vs next phase (checklist for the next builder)

### Done (high level)

- DDL + DAL + contracts; `RecallQueryV1` extensions; fusion + worker + settings; intent router; vector fan-out + collection filter; render budget metadata; profiles `biographical.v1` + `self.factual` rewire; cortex inject + payload `lane`; Hub pool + partial memory API; Hunter subscription shell; distiller/smoke **files exist** as stubs.

### Next phase — **Hub Phase 4 completion** (highest UX value)

1. **`templates/index.html`** — Memory tab shell: nav button `memoryTabButton`, `data-hash-target="memory"`, section `#memory` with Review Queue / All Cards / Activity Log toggles (design §10).
2. **`static/js/memory.js`** — Fetcher + three panels + highlight-to-remember + Cytoscape neighborhood (CDN); wire to **all** API routes from the design table; load script from template with cache-bust token like other Hub JS.
3. **`memory_routes.py` (or `api_routes.py`)** — Implement missing routes: `PATCH`, status change, edges, `GET /api/memory/history`, neighborhood, align with spec §10.
4. **Hub `index.html` / `app.js`** — Pass `lane` into conversation/cortex payloads for social vs chat where the orchestrator already knows context (design §3 lane invariant).

### Next phase — **Recall / cortex hardening**

5. **`test_visibility_intimate_excluded_from_social_lane`** — cards_adapter + inject + fusion path; assert cards with `visibility_scope=['intimate']` never appear when `lane="social"`.
6. **Graphtri / browse paths** — Confirm whether memory cards should participate in graphtri-only recall paths; spec threaded `lane` through `_query_backends` for the multi-signal path only.
7. **`memory_inject`** — Optional: migrate to async pool if conversation front becomes async; add snapshot test for “empty `known_facts_block` ⇒ prompt byte-identical to baseline” (design §9).

### Next phase — **Phase 5 productization**

8. **`memory_extraction.py`** — Replace stub regex with full §5 pattern tables; add `tests/test_memory_extraction.py`.
9. **`memory_extractor.py`** — Implement the per-candidate pipeline (fingerprint, dedupe, contradiction with `always_inject`, session counting for auto-promote) and DAL inserts; keep defaults **off**.
10. **`distill_memory_cards.py`** — JSONL detect, dry-run YAML, `--apply` via DAL; Hub `501` → `subprocess` or in-process call when ready.
11. **Design §11** — Wire `POST /api/memory/sessions/{id}/distill` to distiller.

### Next phase — **Phase 6 + ops**

12. **Operator runbook** — `docs/superpowers/runbooks/2026-05-01-memory-cards-v1-runbook.md` (design §12).
13. **Safety checklist** — Run design §12 `rg` / `git diff` commands on the final merge branch; attach outputs to PR.
14. **Real `smoke_memory_cards_e2e.sh`** — Schema → DAL insert → recall bundle contains cards → reverse path (design §12).
15. **CI** — Ensure images install updated `requirements.txt` for hub + cortex-orch + recall; run new pytest targets.

---

## Commands (verification)

From repo root with a venv that has service deps + pytest:

```bash
# Contracts + intent + extractor unit tests (paths match current layout)
PYTHONPATH=. python3 -m pytest \
  tests/test_memory_cards_contracts.py \
  services/orion-recall/tests/test_intent_routing.py \
  services/orion-cortex-orch/tests/test_memory_extractor.py -q
```

Apply DDL (once per database) using the same DSN as recall:

```bash
python3 -c "from orion.core.storage.memory_cards import apply_memory_cards_schema; import os; apply_memory_cards_schema(os.environ['RECALL_PG_DSN'])"
```

**Recall cards rail:** set `RECALL_ENABLE_CARDS=true`, `RECALL_PG_DSN=...`, restart recall; use a profile with `cards_top_k` / `backend_weights.cards` > 0 (e.g. `biographical.v1`).

**Hub memory API:** set `RECALL_PG_DSN` for Hub; hit `POST /api/memory/cards` with session header pattern used elsewhere (`ensure_session`).

---

## Order of reading for a new owner

1. [Memory Cards v1 design](../specs/2026-05-01-orion-memory-cards-v1-design.md) — especially §2 ground rules, §3 lane/visibility, §12 safety checklist.  
2. `orion/core/storage/memory_cards.py` — write paths and history semantics.  
3. `services/orion-recall/app/cards_adapter.py` — scoring + neighbor SQL.  
4. `services/orion-recall/app/worker.py` — `process_recall` intent + `_query_backends` cards gating.  
5. `services/orion-cortex-orch/app/memory_inject.py` + `conversation_front.py` — always-on block + `lane`.  
6. `services/orion-hub/scripts/memory_routes.py` — current HTTP surface vs spec table.

---

## Explicit reminders

- **Do not** change `RECALL_DEFAULT_PROFILE` or the other eleven recall profiles without a new approved spec.  
- **Do not** turn on `ORION_AUTO_EXTRACTOR_ENABLED` in production defaults.  
- **`ORION_AUTO_EXTRACTOR_STAGE2_ENABLED=true`** must keep raising `NotImplementedError` until v1.5.  
- **No new external runtime families** — stay on FastAPI / asyncpg / psycopg2 / Chroma / Pydantic v2 / Tailwind CDN per design.

---

## Related internal template

For tone and section layout of “offboarding” docs, see also: [Organ Signal Gateway offboarding](./2026-05-01-organ-signal-gateway-offboarding.md).
