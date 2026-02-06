# Orion Topic Foundry (Phase 1)

Spec-driven service for building topic artifacts (runs, segments, model registry) without implicit training on startup.

## Endpoints
- `GET /health` — process liveness.
- `GET /ready` — dependency checks (Postgres, embedding endpoint, model dir).
- `GET /capabilities` — supported modes and defaults for UI clients.
- `POST /datasets` — register a dataset spec.
- `GET /datasets` — list datasets.
- `POST /datasets/preview` — preview derived documents (time-gap windowing).
- `POST /models` — register a model spec.
- `GET /models` — list models.
- `GET /models/{name}/versions` — list model versions.
- `GET /models/{name}/active` — fetch the active model (if any).
- `POST /models/{model_id}/promote` — promote or demote a model stage.
- `POST /runs/train` — start a training run.
- `GET /runs` — list recent runs (legacy response).
- `GET /runs?limit=20&offset=0&format=wrapped` — paginated run summaries with filters (`status`, `stage`, `model_name`).
- `GET /runs/{run_id}` — poll run status.
- `GET /runs/compare?left_run_id=...&right_run_id=...` — compare run stats and top aspect diffs.
- `GET /segments?run_id=...` — list segment records.
- `GET /segments?run_id=...&include_snippet=true&include_bounds=true` — include snippet/char counts and time bounds.
- `GET /segments?run_id=...&limit=200&offset=0&format=wrapped` — paginated segments with limit/offset and total.
- `GET /segments?run_id=...&q=...&sort_by=friction&sort_dir=desc` — search and sort segments (searches title/aspects/snippet).
- `GET /segments/facets?run_id=...` — facets for aspects/intents/friction buckets.
- `GET /segments/{segment_id}/raw` — segment provenance (row_ids/timestamps).
- `POST /runs/{run_id}/enrich` — enrich segments for a run.
- `GET /segments/{segment_id}` — segment record including enrichment fields.
- `GET /topics?run_id=...&limit=200&offset=0` — list topic clusters for a run.
- `GET /topics/{topic_id}/segments?run_id=...&limit=200` — list segments for a topic.
- `GET /topics/{topic_id}/keywords?run_id=...` — list topic keywords.
- `POST /drift/run` — run a drift check against the active model.
- `GET /drift?model_name=...` — list drift records (includes thresholds, deltas, and topic share snapshots).
- `GET /edges?run_id=...` — list KG edges for a run.
- `GET /kg/edges?run_id=...&q=...&predicate=...&limit=...&offset=...` — list KG edges with filters.
- `GET /events?limit=...&offset=...&kind=...` — list recent run/enrich/drift alert events.

## Required env vars
- `SERVICE_NAME`, `SERVICE_VERSION`, `NODE_NAME`, `LOG_LEVEL`
- `PORT`
- `TOPIC_FOUNDRY_PG_DSN`
- `TOPIC_FOUNDRY_EMBEDDING_URL`
- `TOPIC_FOUNDRY_MODEL_DIR`
- `TOPIC_FOUNDRY_LLM_BASE_URL`
- `TOPIC_FOUNDRY_LLM_MODEL`
- `TOPIC_FOUNDRY_LLM_ROUTE`
- `TOPIC_FOUNDRY_LLM_TIMEOUT_SECS`
- `TOPIC_FOUNDRY_LLM_MAX_CONCURRENCY`
- `TOPIC_FOUNDRY_LLM_ENABLE`
- `ORION_BUS_ENABLED`, `ORION_BUS_URL`
- `TOPIC_FOUNDRY_DRIFT_DAEMON`, `TOPIC_FOUNDRY_DRIFT_POLL_SECONDS`, `TOPIC_FOUNDRY_DRIFT_WINDOW_HOURS`

## Model lifecycle
The model registry supports staged lifecycle events. Use `POST /models/{model_id}/promote` with a target
stage of `candidate`, `active`, or `archived`. Promoting a model to `active` automatically demotes any
other active model with the same name to `candidate` and records the transition in
`topic_foundry_model_events` for auditability.

## Artifact layout
```
${TOPIC_FOUNDRY_MODEL_DIR}/
  registry/{model_name}/versions/{version}/
    model/clusterer.joblib
    model_meta.json
    settings.json
    manifest.json
  runs/{run_id}/
    run_record.json
    documents.jsonl
    segments.jsonl
    boundary_judgements.jsonl
    segments_enriched.jsonl
    enrichment_meta.json
    topics_summary.json
    topics_keywords.json
    stats.json
```

## Smoke scripts
Run from repo root:
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_health.sh`
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_preview.sh`
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_train_and_poll.sh`
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_enrich.sh`
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_llm_segmentation.sh`
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_drift.sh`
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_drift_list.sh`
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_kg_edges.sh`
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_kg_edges_list.sh`
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_events.sh`
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_capabilities_and_snippets.sh`
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_runs_and_segments_paging.sh`
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_facets_and_search.sh`
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_topics.sh`

Environment overrides (common):
- `TOPIC_FOUNDRY_BASE_URL` (default `http://localhost:${PORT}`)
- `PORT` (default `8615`)
- `START_AT`, `END_AT`, `LIMIT`, `MIN_DOCS`, `TIMEOUT_SECS`, `SLEEP_SECS`

## Troubleshooting
- `docs_generated too low` — widen `START_AT/END_AT` to include more rows.
- `embedding unreachable` — confirm `TOPIC_FOUNDRY_EMBEDDING_URL` and upstream service.
- `pg unreachable` — check `TOPIC_FOUNDRY_PG_DSN` and DB connectivity.

## Phase 2: Enrichment
Phase 2 adds optional semantic segmentation and segment enrichment.

- Set `segmentation_mode` in `WindowingSpec` to `semantic` or `hybrid` to enable semantic breaks.
- Use `llm_judge` or `hybrid_llm` to enable LLM boundary decisions (see knobs below).
- Enrichment is controlled by `EnrichmentSpec` in `/models` or via `POST /runs/{run_id}/enrich`.
- LLM enrichment requires setting `TOPIC_FOUNDRY_LLM_ENABLE=true` and the LLM URL/model env vars.
- If LLM JSON parsing fails, the service falls back to the heuristic enricher.

### LLM boundary judge knobs
- `llm_boundary_context_blocks` (default 3)
- `llm_boundary_max_chars` (default 4000)
- `llm_candidate_top_k` (default 200)
- `llm_candidate_strategy` (`semantic_low_sim` | `all_edges`)
- `llm_candidate_threshold` (defaults to `semantic_split_threshold`)
