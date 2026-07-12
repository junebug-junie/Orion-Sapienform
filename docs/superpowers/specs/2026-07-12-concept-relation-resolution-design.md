# Concept relation resolution — replacing keyword-cathedral dedup with a bounded LLM judgment seam

**Date:** 2026-07-12
**Status:** Proposal — awaiting approval before implementation (memory/cognition-loop change, proposal mode per AGENTS.md §0A)
**Problem:** The memory crystallization pipeline (`orion/memory/crystallization/`) is live and already does encode → reinforce → decay → retire → multi-rail projection correctly. What it cannot do is recognize that two crystallizations, phrased differently, are about the same thing. Cross-window duplicate detection is not just weak — it is structurally incapable of ever firing, verified against the live corpus, not theorized.

---

## Root cause (evidence, queried live against `conjourney` DB, 2026-07-12)

| Symptom | Choke point | Live evidence |
|---|---|---|
| Byte-identical multi-paragraph crystallization created twice, 70 min apart, both `active`, never merged | `detection.py::detect_duplicates()` `scope_overlap` check | `crys ecea0d6d` / `crys 50127466` — identical subject text, dedup never fired |
| Dedup structurally cannot match across windows | `intake_consolidation_window.py`: every crystallization gets `scope=[f"memory_window:{window_id}"]` | `scope_overlap = bool(A & B) or (not A or not B)` — two non-empty, always-disjoint per-window scopes evaluate `False` for every pair, always |
| Typed relation graph (supports/contradicts/supersedes/refines/...) never populated | `links.py::insert_link()` never called from `intake_pipeline.py` | `SELECT count(*) FROM memory_crystallization_links` → **0 rows**, against 79 live crystallizations |
| Low-info chitchat crystallized as durable "semantic" belief | `consolidation_gate.py`: `novelty_above_floor` / `substantive_shift` short-circuit before the `is_low_info_social` check | Live rows with `subject = "hi"`, `"hey!"`, `"hey, Orion. What's shakin?"` — one `hi` row scored `activation=1.0, salience=1.0`, a second `hi` row scored `activation=0.25, salience=0.625` for equivalent content |
| Old concept-induction extractor mistaken for semantic dedup | `orion/spark/concept_induction/extractor.py` | Confirmed by reading the file: NER-label + noun-chunk text, frequency-counted with a trivial positional weight (`1.0 + idx*0.01`) — no embedding, no IDF, no semantic comparison. Not viable as a concept-identity mechanism at any fix level. |
| Deterministic canonicalization was proposed and rejected | N/A — design decision, not a bug | A `concept_key` derived by rule-based lemma/noun-chunk matching would be wrong at the boundaries by construction (false-merges of distinct concepts sharing a head noun; false-splits of the same concept phrased differently), and every boundary fix is another rule stacked on formation_policy → salience → dynamics → recall_eligibility, which is already five layers deep. This is the keyword-cathedral failure mode AGENTS.md §0A bans, and "is this the same idea, worded differently" is squarely a latent-judgment task under AGENTS.md §4, not a parsing task. |

Live corpus snapshot at time of writing: 79 crystallizations — `semantic active=36, stance rejected=18, stance active=9, stance proposed=6, semantic rejected=6, open_loop rejected=3, open_loop active=1`. No `episode`/`procedure`/`decision`/`contradiction`/`attractor`/`failure_mode` rows exist yet; `_kind_for_gate()`'s `episode` default path has apparently never fired in practice. `MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED=true` and `GRAPHITI_ENABLED=true` are both live defaults; `orion-athena-memory-consolidation` and `orion-athena-graphiti-adapter` containers are both running.

---

## Goals

- **Phase A:** Stop crystallizing low-info chitchat. Cheap, deterministic, ships independently of everything else below.
- **Phase B:** Fix cross-window candidate retrieval so duplicate/related candidates are actually discoverable, replacing the structurally-broken scope-gated Jaccard check.
- **Phase C:** Add one bounded, structured-output LLM call that resolves the genuinely ambiguous judgment — "is this new crystallization the same concept as an existing one, and if so how does it relate?" — and dispatches the typed answer into primitives that already exist (`reinforce()`, `governor.supersede()`, `insert_link()`) rather than inventing new ones.

## Non-goals

- No deterministic `concept_key` canonicalizer, lemma-matching, or alias table. Rejected explicitly (see root-cause table).
- No revival of `orion/spark/concept_induction/` or its noun-chunk extractor at any fix level. `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED` stays `false`.
- No new relation vocabulary. `CrystallizationRelation` in `schemas.py` already has everything needed (`supports/contradicts/supersedes/narrows/expands/refines/depends_on/evidence_for/evidence_against/co_occurs_with/related_to`) and is currently unused — Phase C activates it, it does not extend it.
- No changes to `formation_policy.py`'s gated-kind human-review requirement for `stance/decision/contradiction/attractor/failure_mode`.
- No Graphiti Phase C (`graphiti-core` hybrid search) activation — that rail is separately designed (`docs/superpowers/specs/2026-07-06-graphiti-rail-activation-design.md`) and becomes more valuable *after* this ships, not before.
- No change to `MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED` default or the governor approval flow itself.

---

## Architecture (stable contract)

```text
consolidation window closes
        │
        ▼
consolidation_memory_gate()   ← Phase A: raise/adjust low-info floor here
        │ (action=propose)
        ▼
build_crystallization_from_window()
        │
        ▼
process_consolidation_crystallization()          orion/memory/crystallization/intake_pipeline.py
        │
        ▼
retrieve_candidates()   ← Phase B: chroma vector search, NOT scope-gated   (new, thin)
        │  top-5 similar existing active crystallizations
        ▼
resolve_concept_relation()   ← Phase C: bounded LLM call                   (new)
        │  {relation: same|refines|contradicts|unrelated, target_id, confidence}
        ▼
dispatch on relation:
    same         → reinforce(target)                      (existing, orion/memory/crystallization/dynamics.py)
    refines       → governor.supersede(target, new)         (existing, orion/memory/crystallization/governor.py)
    contradicts   → insert_link(relation="contradicts")      (existing, orion/memory/crystallization/links.py — currently dead code, this activates it)
    unrelated     → insert_crystallization() as new row      (existing path, unchanged)
```

**Invariant:** Phase C never mutates canonical rows directly from the LLM's raw output — the LLM emits a typed, schema-validated decision; only the existing deterministic primitives (`reinforce`, `governor.supersede`, `insert_link`) touch storage. If the LLM call fails or times out, fall open to today's behavior (insert as new, proposed/auto-activate per existing formation_policy) — never block window closing on this call.

**Choke points:**

| File | Role |
|---|---|
| `orion/memory/consolidation_gate.py` | Phase A: significance/novelty floor, `is_low_info_social` ordering |
| `orion/memory/low_info_social.py` | Phase A: existing low-info check, currently bypassed when novelty/significance scores are noisy on short turns |
| `orion/memory/crystallization/detection.py` | Phase B: today's broken scope-gated Jaccard dedup — retained only as a same-window cheap pre-filter, no longer the source of truth for cross-time relation |
| `orion/memory/crystallization/chroma_query.py`, `retriever.py::_embed_query` | Phase B: existing embedding/vector rail, reused unchanged for candidate retrieval |
| `orion/memory/crystallization/concept_relation.py` (new) | Phase C: candidate retrieval + LLM call + typed dispatch |
| `services/orion-memory-consolidation/app/classify.py::_llm_classify` | Phase C precedent: bounded structured LLM call via bus RPC to `LLMGatewayService`, `max_tokens=24`, `purpose=classify` — same shape, new call site |
| `orion/memory/crystallization/intake_pipeline.py::process_consolidation_crystallization` | Wires Phase B + C in ahead of the existing `resolve_formation_policy` dispatch |
| `orion/memory/crystallization/dynamics.py`, `governor.py`, `links.py` | Unchanged — Phase C's dispatch targets, already correct |

---

## Phase A — Significance floor hardening

### A1. Root cause of the "hi" / "hey!" crystallizations

`consolidation_gate.py::consolidation_memory_gate()` checks `substantive_shift`, `novelty_above_floor`, and `significance_above_floor` **before** the `is_low_info_social` / `all_low_info` check. When the turn-change classifier (`classify_turn()` in `app/classify.py`) scores a very short turn with noisy high novelty/significance, the gate proposes regardless of `is_low_info_social` — the low-info filter exists but is short-circuited.

### A2. Fix

Add a `has_substantive_text` requirement as a precondition for `novelty_above_floor` and `significance_above_floor` (not just their own independent fallback branch), so a short low-info turn cannot win on a noisy score alone. Keep `repair_signal` and `substantive_shift` (grammar-evidence-backed) as unconditional overrides — those are the paths with independent corroborating signal.

### A3. Acceptance

- [ ] Unit test: two-turn window of `"hi"` / `"hey, what's shakin"` with a synthetically high novelty score does **not** propose (regression test for the live rows found)
- [ ] Unit test: existing `repair_signal` / `substantive_shift` / `substantive_text` accept-paths unchanged
- [ ] Re-run against the live 79-row sample (read-only query) to confirm no currently-`active` row would retroactively fail the new gate in a way that changes behavior going forward (informational only — no backfill)

---

## Phase B — Candidate retrieval fix

### B1. Remove scope-gating from cross-crystallization candidate search

`detect_duplicates()`'s `scope_overlap` requirement is the confirmed structural bug (root-cause table). Two options, pick one at implementation time:

- **B1a (minimal):** drop the `scope_overlap` condition entirely from `detect_duplicates()` — cheap, but Jaccard-on-raw-text remains a weak signal.
- **B1b (recommended):** replace `detect_duplicates()`'s role in the intake path with a vector-similarity candidate fetch (`retriever.py`'s existing `_embed_query` + `chroma_query.query_chroma_collection`, already wired and unchanged), returning top-5 similar active crystallizations regardless of scope. This becomes Phase C's candidate input directly — no second dedup mechanism to keep in sync.

Recommendation: B1b. Keep `detect_duplicates()`'s Jaccard check only as a cheap same-window pre-filter (its `scope_overlap` condition is *correct* for same-window near-duplicates — the bug is only that it's currently the sole cross-window mechanism too).

### B2. Acceptance

- [ ] Regression test reproducing the live GitNexus-pipeline duplicate (same text, different window) — confirm candidate retrieval now surfaces the prior crystallization
- [ ] Confirm same-window Jaccard pre-filter behavior unchanged

---

## Phase C — Bounded LLM concept-relation resolution

### C1. Call shape (precedent: `classify.py::_llm_classify`)

```python
async def resolve_concept_relation(
    bus: OrionBusAsync,
    *,
    candidate: MemoryCrystallizationV1,
    similar_existing: list[MemoryCrystallizationV1],  # top-5 from Phase B, may be empty
    settings: Any,
) -> ConceptRelationDecision:
    ...
```

```python
class ConceptRelationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relation: Literal["same", "refines", "contradicts", "unrelated"]
    target_crystallization_id: str | None = None   # required unless relation == "unrelated"
    confidence: float = Field(ge=0.0, le=1.0)
```

Bus RPC to `LLMGatewayService` via `CHANNEL_LLM_INTAKE`, same envelope shape as `_llm_classify`: small `max_tokens`, `purpose="classify"`, `skip_spark_candidate_publish=True`. Prompt includes the candidate's subject+summary and up to 5 similar existing crystallizations' subject+summary+status, asks for one of the four typed relations plus target id. Parse and validate against `ConceptRelationDecision` — on parse failure or timeout, fall open to `unrelated` (never block).

### C2. Dispatch

In `intake_pipeline.py::process_consolidation_crystallization`, insert the Phase B/C step ahead of the existing `resolve_formation_policy` branch:

| Decision | Action | Existing primitive |
|---|---|---|
| `same`, confidence ≥ threshold | Reinforce target, do not insert new row | `dynamics.reinforce()` |
| `refines`, confidence ≥ threshold | Insert new row, then `supersedes` link + supersede target (gated kinds still require governor path — Phase C does not bypass `formation_policy.py`'s existing gate) | `governor.supersede()`, `links.insert_link()` |
| `contradicts`, any confidence | Insert new row as `contradiction`-linked, do not auto-resolve | `links.insert_link(relation="contradicts")` |
| `unrelated`, or LLM call failed/degraded | Existing path, unchanged | `resolve_formation_policy()` as today |

Confidence threshold (`CONCEPT_RELATION_CONFIDENCE_FLOOR`, proposed default `0.6`) gates `same`/`refines` from acting automatically; below floor, fall through to `unrelated` (insert as new, let governor/human review handle it via the normal gated-kind path).

### C3. Acceptance

- [ ] Adapter/unit test: candidate matching an existing crystallization's paraphrase → `relation=same`, reinforces target, no duplicate row
- [ ] Test: candidate that supersedes an existing belief → `refines`, target superseded, link written (first-ever row in `memory_crystallization_links`)
- [ ] Test: LLM call timeout/malformed response → falls open to `unrelated`, existing behavior unchanged, no window-closing block
- [ ] Live smoke: replay the two GitNexus-duplicate turns from the root-cause evidence against a test corpus, confirm second one reinforces the first instead of inserting a duplicate

---

## Env / config changes

| Key | Change |
|---|---|
| `CONCEPT_RELATION_RESOLUTION_ENABLED` | New, default `false` — Phase C gate, off until acceptance checks pass in a test environment |
| `CONCEPT_RELATION_CONFIDENCE_FLOOR` | New, default `0.6` |
| `CONCEPT_RELATION_CANDIDATE_LIMIT` | New, default `5` |
| `MEMORY_CONSOLIDATION_MIN_NOVELTY` / `MIN_SIGNIFICANCE` | Unchanged values; gate *ordering* changes in Phase A, not the thresholds themselves |

Run `python scripts/sync_local_env_from_example.py` after `.env_example` changes land.

---

## Testing matrix

| Layer | Phase A | Phase B | Phase C |
|---|---|---|---|
| Unit | gate ordering regression (low-info short-circuit) | candidate retrieval crosses window scope | `ConceptRelationDecision` schema validation, dispatch table |
| Integration | consolidation worker end-to-end on low-info fixture | dedup on live-shaped duplicate fixture | LLM call mocked + degraded-path fallback |
| Live smoke | — | replay live GitNexus-duplicate pair | replay live GitNexus-duplicate pair through full Phase B+C, confirm reinforce not duplicate-insert |

---

## Risks

| Severity | Risk | Mitigation |
|---|---|---|
| Medium | LLM relation call adds latency/cost to every proposed window | Only invoked when Phase B returns ≥1 candidate; empty-candidate case skips the call entirely |
| Medium | `refines`/`same` misclassification silently loses a distinct belief | Confidence floor gates automatic action; `contradicts` never auto-resolves; all four decisions are logged in `provenance` for audit |
| Low | Phase A gate reordering changes what gets skipped vs proposed | Regression tests pin current accept-paths (`repair_signal`, `substantive_shift`) exactly; only the low-info short-circuit changes |
| Low | Second dedup mechanism (Jaccard same-window + vector cross-window) diverges over time | B1b makes vector retrieval the only cross-window path; Jaccard scoped strictly to same-window pre-filter, documented as such |

---

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-memory-consolidation/.env \
  -f services/orion-memory-consolidation/docker-compose.yml up -d --build
```

No Hub or Graphiti-adapter restart required for Phases A–C (crystallization schema/API surface unchanged; `concept_key`-style field explicitly rejected, so no migration needed).

---

## Recommended implementation order

1. **Phase A** — ships alone, no dependencies, fixes the most visible live symptom (chitchat pollution) cheaply.
2. **Phase B** — fixes the confirmed structural dedup bug; can ship before Phase C as a standalone correctness fix (Jaccard-with-real-candidates is still better than Jaccard-that-can-never-match).
3. **Phase C** — the LLM relation-resolution seam; depends on B's candidate retrieval; ships behind `CONCEPT_RELATION_RESOLUTION_ENABLED=false` until live-smoke acceptance passes.

Await explicit go-ahead before implementation per AGENTS.md §0A proposal-mode requirement for memory/cognition-loop changes.
