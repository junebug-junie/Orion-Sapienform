# Orion Memory Consolidation

Subscribes to `orion:memory:turn:persisted` (sql-writer post-commit outbox), classifies each chat turn via LLM gateway quick lane logprobs, patches `chat_history_log.spark_meta`, tracks consolidation windows, and on boundary closure runs a **deterministic consolidation gate** (default) or legacy graph suggest.

## Consolidation output modes

| `MEMORY_CONSOLIDATION_OUTPUT` | Behavior |
|-------------------------------|----------|
| `crystallization_propose` (default) | Run `consolidation_memory_gate`; **skip** low-signal windows (`consolidation_status=skipped`) or **propose** `MemoryCrystallizationV1` for governor review |
| `graph_draft` | Legacy path: LLM `memory_graph_suggest` + pending graph draft insert (manual bridge only) |
| `skip_only` | Run gate for traceability; always mark window skipped — no crystallization or graph draft |

Gate thresholds: `MEMORY_CONSOLIDATION_MIN_NOVELTY` (default `0.35`), `MEMORY_CONSOLIDATION_MIN_SIGNIFICANCE` (default `0.40`). Both floors require the window not be all low-info-social (`is_low_info_social` on every turn's prompt and response) as corroboration — a bare novelty/significance float alone is not sufficient, since a noisy classifier score on a short greeting-only turn was previously enough to crystallize it (`repair_signal` and `substantive_shift` are unaffected; they already require an independent shift-kind classification).

Grammar repair evidence (read-only): `MEMORY_CONSOLIDATION_FETCH_GRAMMAR_EVIDENCE=true` queries `grammar_events` by `hub.chat:{NODE_NAME}:{correlation_id}` trace. Optional override DSN: `MEMORY_CONSOLIDATION_GRAMMAR_DSN`.

**Note:** Proposed crystallization IDs are stored in `memory_consolidation_windows.draft_id` until a dedicated `crystallization_id` column migration lands.

## Cross-window concept-relation resolution (off by default)

Same-window duplicate detection (`orion.memory.crystallization.detection.detect_duplicates`) requires `scope_overlap`, which is structurally always `False` across two different consolidation windows — every crystallization gets a unique per-window `scope`, so two windows can never share one. `orion.memory.crystallization.concept_relation` adds a second, cross-window path: vector-similarity candidate retrieval (`candidate_retrieval.fetch_similar_candidates`, not scope-gated) followed by one bounded, structured-output LLM call that judges `same` / `refines` / `contradicts` / `unrelated` against the nearest existing active crystallizations of the same `kind`.

Dispatch is deliberately conservative: `same` reinforces the existing target (identical mechanism to the same-window path). `refines` and `contradicts` only attach a typed link to the *new* candidate's own `links` (persisted to `memory_crystallization_links` on insert, same as any other crystallization) — they never mutate or supersede the existing target's status. That stays a human decision via the existing `/api/memory/crystallizations/{id}/links` and supersede endpoints. Every decisive branch (`same`/`refines`/`contradicts`) stamps `provenance.concept_relation` (relation, target id, confidence) on the affected row for audit, independent of which branch acted.

**Decision log + belief-revision digest.** Every real LLM decision — including `unrelated` and sub-floor `contradicts`/`refines` that the dispatch above discards — is written to `memory_concept_relation_decisions` (`orion.memory.crystallization.repository.insert_concept_relation_decision`, guarded to never raise). `scripts/concept_relation_digest.py` (repo root, run on demand or via cron — not a live service loop) reads undigested rows and reports call volume / relation distribution / near-miss counts under `CONCEPT_RELATION_CONFIDENCE_FLOOR`, then creates one `reflection`-kind crystallization per real belief-revision decision (`same`/`refines`/`contradicts` that cleared the floor) — a structured, deterministic trace of Orion revising its own beliefs, not just a link nobody reads back.

Ships flag-gated off; flipping the flag alone does not activate anything without also configuring the embed/chroma hosts (both default to empty string):

| Env | Default | Purpose |
|-----|---------|---------|
| `CONCEPT_RELATION_RESOLUTION_ENABLED` | `false` | Master flag for this path |
| `CONCEPT_RELATION_CONFIDENCE_FLOOR` | `0.6` | Minimum LLM decision confidence to act; below this, falls through to the normal formation-policy path unchanged |
| `CONCEPT_RELATION_CANDIDATE_LIMIT` | `5` | Max vector-similar candidates fetched and sent to the LLM prompt |
| `CONCEPT_RELATION_TIMEOUT_SEC` | `8.0` | RPC timeout for the relation-judgment call |
| `CRYSTALLIZER_EMBED_HOST_URL` | *(empty)* | Embedding HTTP endpoint for candidate retrieval — must be set for this feature to do anything |
| `CRYSTALLIZER_EMBED_TIMEOUT_MS` | `8000` | Embed call timeout |
| `CHROMA_HOST` / `CHROMA_PORT` | *(empty)* / `8000` | Chroma vector store for candidate retrieval — must be set alongside the embed host |
| `CRYSTALLIZER_VECTOR_COLLECTION` | `orion_memory_crystallizations` | Chroma collection name (matches Hub's projection collection) |

## Channels

| Direction | Channel |
|-----------|---------|
| In | `orion:memory:turn:persisted` |
| Out | `orion:chat:history:spark_meta:patch` |
| Out (threshold) | `orion:signals:memory_consolidation` (`signal.memory_consolidation.turn_change`) |
| Out (propose) | `orion:memory:crystallization:proposed` (`memory.crystallization.proposed.v1`) |

## Turn change appraisal

Each persisted turn (after the first in a window) gets a logprob-calibrated `turn_change_appraisal` patch on `spark_meta`: novelty score, shift kind, confidence, and baseline mode (`prior_turn` or `session_window` fallback). The first turn in a window uses `turn_change_status=skipped` (no baseline, no LLM call). High-confidence novel turns also emit `OrionSignalV1` on `orion:signals:memory_consolidation`.

| Env | Default | Purpose |
|-----|---------|---------|
| `TURN_CHANGE_CLASSIFY_ROUTE` | `metacog` | Gateway route for classify RPC (`metacog` or `quick`) |
| `TURN_CHANGE_CONFIDENCE_MARGIN` | `0.15` | Re-appraise vs session window when novelty margin is below this |
| `TURN_CHANGE_SUBSTRATE_THRESHOLD` | `0.65` | Minimum novelty to emit substrate signal |
| `TURN_CHANGE_WINDOW_TURNS` | `3` | Prior turns in session-window baseline |
| `CHANNEL_SIGNALS_PREFIX` | `orion:signals` | Bus prefix for organ signal publish |

## Bring-up

```bash
docker compose --env-file ../../.env --env-file ../orion-bus/.env -f docker-compose.yml up -d --build
```

Apply Postgres migration: `services/orion-sql-db/manual_migration_memory_consolidation_v1.sql`

## Smoke

```bash
PYTHONPATH=. python scripts/smoke_memory_consolidation_pipeline.py
bash scripts/smoke_memory_consolidation_gate.sh
```

Gate smoke runs deterministic unit tests (greeting skip + substantive propose). Pipeline smoke requires live stack (bus, sql-writer, postgres, llm-gateway, cortex).
