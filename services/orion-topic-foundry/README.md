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
- `TOPIC_FOUNDRY_COSINE_IMPL`
- `TOPIC_FOUNDRY_MODEL_DIR`
- `TOPIC_FOUNDRY_LLM_BUS_ROUTE`
- `TOPIC_FOUNDRY_LLM_TIMEOUT_SECS`
- `TOPIC_FOUNDRY_LLM_MAX_CONCURRENCY`
- `TOPIC_FOUNDRY_LLM_USE_BUS`
- `TOPIC_FOUNDRY_LLM_INTAKE_CHANNEL`
- `TOPIC_FOUNDRY_LLM_REPLY_PREFIX`
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

### Preview example (canonical request)
```bash
curl -sS http://localhost:8615/datasets/preview \\
  -H "content-type: application/json" \\
  -d '{
    "dataset_id": "00000000-0000-0000-0000-000000000000",
    "windowing": {
      "block_mode": "turn_pairs",
      "segmentation_mode": "time_gap",
      "time_gap_seconds": 900,
      "max_window_seconds": 7200,
      "min_blocks_per_segment": 1,
      "max_chars": 6000
    },
    "start_at": null,
    "end_at": null,
    "limit": 200
  }'
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
- `scripts/smoke_topic_foundry_all.sh` (runs introspect → preview → train → facets → enrich)
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_health.sh`
- `services/orion-topic-foundry/scripts/smoke_topic_foundry_preview.sh`
- `scripts/smoke_topic_foundry_train_cosine.sh`
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
- `TOPIC_FOUNDRY_BASE_URL` (defaults to Hub proxy if available)
- `HUB_BASE_URL` (derives `TOPIC_FOUNDRY_BASE_URL=$HUB_BASE_URL/api/topic-foundry`)
- `PORT` (default `8615`, only used by service-local scripts)
- `START_AT`, `END_AT`, `LIMIT`, `MIN_DOCS`, `TIMEOUT_SECS`, `SLEEP_SECS`

Smoke base URL resolution for scripts under `/scripts` (in order):
1) CLI arg (BASE_URL)
2) `TOPIC_FOUNDRY_BASE_URL`
3) `HUB_BASE_URL` → `${HUB_BASE_URL}/api/topic-foundry`
4) `http://127.0.0.1:8080/api/topic-foundry`

### Running smokes
**Via Hub proxy (recommended):**
```bash
scripts/smoke_topic_foundry_all.sh https://tailscale-host.example.com/api/topic-foundry
```
or:
```bash
HUB_BASE_URL=https://tailscale-host.example.com scripts/smoke_topic_foundry_introspect.sh
```

**Direct service port (optional):**
```bash
TOPIC_FOUNDRY_BASE_URL=http://127.0.0.1:8615 scripts/smoke_topic_foundry_preview.sh
```

**Inside Docker network (optional):**
```bash
TOPIC_FOUNDRY_BASE_URL=http://orion-topic-foundry:8615 scripts/smoke_topic_foundry_facets.sh
```

## Troubleshooting
- `docs_generated too low` — widen `START_AT/END_AT` to include more rows.
- `embedding unreachable` — confirm `TOPIC_FOUNDRY_EMBEDDING_URL` and upstream service.
- `pg unreachable` — check `TOPIC_FOUNDRY_PG_DSN` and DB connectivity.
- `cosine metric errors` — cosine is implemented via L2-normalize + euclidean by default. Set `TOPIC_FOUNDRY_COSINE_IMPL=generic` to force HDBSCAN’s generic cosine implementation (slower but avoids tree metrics).

## Phase 2: Enrichment
Phase 2 adds optional semantic segmentation and segment enrichment.

- Set `segmentation_mode` in `WindowingSpec` to `semantic` or `hybrid` to enable semantic breaks.
- Use `llm_judge` or `hybrid_llm` to enable LLM boundary decisions (see knobs below).
- Enrichment is controlled by `EnrichmentSpec` in `/models` or via `POST /runs/{run_id}/enrich`.
- LLM enrichment requires setting `TOPIC_FOUNDRY_LLM_ENABLE=true` and the LLM URL/model env vars.
- If LLM JSON parsing fails, the service falls back to the heuristic enricher.
- LLM requests prefer the Orion bus (`TOPIC_FOUNDRY_LLM_USE_BUS=true`) and fall back to the HTTP gateway.

### LLM boundary judge knobs
- `llm_boundary_context_blocks` (default 3)
- `llm_boundary_max_chars` (default 4000)
- `llm_candidate_top_k` (default 200)
- `llm_candidate_strategy` (`semantic_low_sim` | `all_edges`)
- `llm_candidate_threshold` (defaults to `semantic_split_threshold`)

## Capabilities Contract

`GET /capabilities` is implemented in `app/routers/capabilities.py` and returns runtime UI contract values (service/version/node, LLM flags, segmentation/enricher modes, metric support, defaults, and introspection summary).

Example (truncated, representative):

```json
{
  "service": "orion-topic-foundry",
  "version": "0.1.0",
  "node": "...",
  "llm_enabled": false,
  "llm_transport": "bus",
  "llm_bus_route": "LLMGatewayService",
  "llm_intake_channel": "orion:exec:request:LLMGatewayService",
  "llm_reply_prefix": "orion:llm:reply",
  "segmentation_modes_supported": ["time_gap", "semantic", "hybrid", "llm_judge", "hybrid_llm"],
  "enricher_modes_supported": ["heuristic", "llm"],
  "supported_metrics": ["cosine", "euclidean", "l1", "l2", "manhattan"],
  "default_metric": "cosine",
  "cosine_impl_default": "normalize_euclidean",
  "defaults": {
    "embedding_source_url": "http://orion-vector-host:8320/embedding",
    "metric": "cosine",
    "min_cluster_size": 15,
    "llm_bus_route": "LLMGatewayService"
  },
  "introspection": {
    "ok": true,
    "schemas": ["public"]
  },
  "default_embedding_url": "http://orion-vector-host:8320/embedding"
}
```

### Key env knobs backing `/capabilities`
- `TOPIC_FOUNDRY_LLM_ENABLE` → `llm_enabled`
- `TOPIC_FOUNDRY_LLM_USE_BUS` + `ORION_BUS_ENABLED` → `llm_transport` (`bus` vs `http`)
- `TOPIC_FOUNDRY_LLM_BUS_ROUTE` → `llm_bus_route` and `defaults.llm_bus_route`
- `TOPIC_FOUNDRY_LLM_INTAKE_CHANNEL`, `TOPIC_FOUNDRY_LLM_REPLY_PREFIX` → bus fields (only emitted when transport is `bus`)
- `TOPIC_FOUNDRY_EMBEDDING_URL` → `defaults.embedding_source_url` and `default_embedding_url`
- `TOPIC_FOUNDRY_COSINE_IMPL` → `cosine_impl_default`
- `TOPIC_FOUNDRY_INTROSPECT_SCHEMAS` → `introspection.ok` / `introspection.schemas`

## LLM over bus

LLM path is considered usable when:
1. `TOPIC_FOUNDRY_LLM_ENABLE=true` (feature enabled), and
2. `TOPIC_FOUNDRY_LLM_USE_BUS=true` and `ORION_BUS_ENABLED=true` (transport resolves to bus).

Recommended env example:

```env
TOPIC_FOUNDRY_LLM_ENABLE=true
TOPIC_FOUNDRY_LLM_USE_BUS=true
ORION_BUS_ENABLED=true
ORION_BUS_URL=redis://<tailnet-redis-host>:6379/0
TOPIC_FOUNDRY_LLM_BUS_ROUTE=LLMGatewayService
TOPIC_FOUNDRY_LLM_INTAKE_CHANNEL=orion:exec:request:LLMGatewayService
TOPIC_FOUNDRY_LLM_REPLY_PREFIX=orion:llm:reply
TOPIC_FOUNDRY_LLM_TIMEOUT_SECS=60
TOPIC_FOUNDRY_LLM_MAX_CONCURRENCY=4
```

Reply correlation:
- Foundry creates `correlation_id` per request.
- It sets `reply_to = <TOPIC_FOUNDRY_LLM_REPLY_PREFIX>:<correlation_id>`.
- RPC response is matched using that reply channel + envelope correlation metadata.

## Troubleshooting

### `LLM disabled` in Hub Topic Studio
- Verify `TOPIC_FOUNDRY_LLM_ENABLE=true` in Topic Foundry runtime env.
- Check `/capabilities` returns `"llm_enabled": true`.
- If `llm_enabled=true` but `llm_transport` is not `bus`, check `TOPIC_FOUNDRY_LLM_USE_BUS` and `ORION_BUS_ENABLED`.

### Capabilities missing keys
- Confirm you are hitting this service’s `/capabilities` route (not stale proxy target).
- Check service logs for startup/env load errors.
- Validate required env vars above, especially embedding + introspection settings.

### Hub proxy issues
- Hub uses `TOPIC_FOUNDRY_BASE_URL` to proxy `/api/topic-foundry/*`.
- If direct service works but Hub doesn’t, test:
  - `curl "$HUB_BASE/api/topic-foundry/ready"`
  - `curl "$HUB_BASE/api/topic-foundry/capabilities"`

