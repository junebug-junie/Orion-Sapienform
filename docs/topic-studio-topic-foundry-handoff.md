# Hub + Topic Studio + Topic Foundry Wiring Handoff (Rebuild Guide)

_Last updated: 2026-04-07 (UTC)_

## 1) Why this doc exists

This is a **deep handoff report** for rebuilding the Topic Studio / Topic Foundry integration after version-control loss.

The goal is to let a future engineer (or future agent) re-create:

- Hub proxy wiring (`/api/topic-foundry/*`)
- Topic Studio UI wiring (datasets, preview, models, train, run results)
- BERTopic parameter propagation (including `nr_topics`)
- Capability-driven UI defaults (metrics, modes, backend toggles)
- Inspection/reporting flows (run summary + run results)

---

## 2) High-level architecture

```text
Browser Topic Studio UI
    -> Hub endpoint: /api/topic-foundry/*
        -> Topic Foundry service (FastAPI)
            -> Postgres (registry, run/segment/topic storage)
            -> Embedding backend (vector host / sentence-transformers)
            -> BERTopic training pipeline
```

### Hub proxy contract

- Hub proxies Topic Foundry via one catch-all route:
  - `services/orion-hub/scripts/api_routes.py`
  - Route: `/api/topic-foundry/{path:path}`
- Upstream base URL comes from:
  - `services/orion-hub/app/settings.py`
  - `TOPIC_FOUNDRY_BASE_URL`

### Browser base URL

In Topic Studio JS, all calls route through:

- `TOPIC_FOUNDRY_PROXY_BASE = apiUrl("/api/topic-foundry")`
- Then `topicFoundryFetch()` and `topicFoundryFetchWithHeaders()` build requests from that.

---

## 3) Core backend endpoints used by Topic Studio

### Capabilities + health
- `GET /capabilities`
- `GET /ready`

### Dataset lifecycle
- `POST /datasets`
- `GET /datasets`
- `PATCH /datasets/{dataset_id}`
- `POST /datasets/preview`
- `GET /datasets/{dataset_id}/preview/docs/{doc_id}`

### Model lifecycle
- `POST /models`
- `GET /models`
- `GET /models/{name}/versions`
- `GET /models/{name}/active`

### Training + result inspection
- `POST /runs/train`
- `GET /runs/{run_id}`
- `GET /runs`
- `GET /runs/{run_id}/results`
- `GET /runs/{run_id}/results/{segment_id}`

### Additional exploration used by Studio panes
- Segments:
  - `GET /segments`
  - `GET /segments/facets`
  - `GET /segments/{segment_id}`
  - `GET /segments/{segment_id}/raw`
  - `GET /segments/{segment_id}/full_text`
- Topics:
  - `GET /topics`
  - `GET /topics/{topic_id}/segments`
  - `GET /topics/{topic_id}/keywords`

---

## 4) Topic Studio frontend wiring map

Primary files:

- `services/orion-hub/templates/index.html` (UI structure + element IDs)
- `services/orion-hub/static/js/app.js` (state, fetches, actions, rendering)

### 4.1 Dataset wiring

- Dataset form -> `buildDatasetSpec()`
- Create dataset button -> `handleCreateDatasetClick()` -> `POST /datasets`
- Save dataset button -> `handleSaveDatasetClick()` -> `PATCH /datasets/{id}`
- Preview button -> `handlePreviewDatasetClick()` -> `POST /datasets/preview`

### 4.2 Model creation wiring

`handleCreateModelClick()` builds payload:

- `model_spec`:
  - `algorithm`
  - `embedding_source_url`
  - `min_cluster_size`
  - `metric`
  - `params` (raw Params JSON)
- `model_meta`:
  - BERTopic/HDBSCAN/UMAP/vectorizer knobs
  - stop-word extras
  - representation

#### Important UX fix (no dataset pre-selection required)

If no dataset is selected, Studio now auto-creates one from the current dataset form via:

- `ensureDatasetForModelCreate()`
- Calls `POST /datasets`
- Uses returned `dataset_id` in `POST /models`

This removed the prior hard requirement to manually pre-select dataset every time.

### 4.3 Metric selector

- Metric selector is positioned directly under Params JSON.
- Options come from `/capabilities.supported_metrics`.
- Default selected uses `/capabilities.default_metric`.
- Fallback metric list in JS when capabilities fetch fails.

### 4.4 Training + polling

- Train button -> `handleTrainRunClick()` -> `POST /runs/train`
- Poll button -> `handlePollRunClick()` -> `GET /runs/{run_id}`
- Background polling -> `startRunPolling(runId)` updates status cards and auto-loads results once terminal state.

### 4.5 Run results and visualization

- `loadRunResultsSegments(runId)`:
  - fetches `GET /runs/{run_id}` summary
  - fetches paginated `/runs/{run_id}/results?limit=...&offset=...`
  - accumulates all pages
  - renders table rows
- `renderRunResultsTopicChart(items)`:
  - aggregates topic+label counts
  - renders % bar chart above table

---

## 5) BERTopic kwarg propagation (critical rebuild knowledge)

Primary files:

- `services/orion-topic-foundry/app/services/training.py`
- `services/orion-topic-foundry/app/topic_engine.py`

### 5.1 Meta merge strategy before training

Training composes effective model metadata from multiple sources:

1. Stored `model_row.model_meta`
2. Snapshot `run.specs.model.model_meta`
3. Stored `model_row.model_spec.params`
4. Snapshot `run.specs.model.params`
5. Runtime topic-mode params (`seed_topic_list`, `zeroshot_topic_list`, etc.)

Then selected `model_spec.metric` is explicitly forwarded as:

- `model_meta["hdbscan_metric"]`

This ensures UI-selected metric actually influences clustering backend, not just model registry metadata.

### 5.2 Topic engine assembly

`build_topic_engine(model_meta)` creates:

- embedding backend (vector host or sentence-transformers)
- UMAP reducer
- HDBSCAN clusterer
- representation model (`ctfidf`, `keybert`, `mmr`, `pos`, `llm`)
- vectorizer + ctfidf config for ctfidf path
- finalized BERTopic kwargs

### 5.3 `nr_topics` support (reduction target)

`topic_engine.py` now includes `_parse_nr_topics(...)` and wires:

- `nr_topics=<int>`
- `nr_topics="auto"`

into BERTopic kwargs when present (from params/meta).

This is the dynamic topic-reduction capability requested in the rebuild scope.

---

## 6) Capabilities payload contract used by Studio

`/capabilities` response includes:

- `capabilities.topic_modeling.*` booleans
- `backends` (embedding/reducer/clusterer/representation families)
- `defaults.vectorizer.*` and `defaults.ctfidf.*`
- `supported_metrics` (derived from `hdbscan.dist_metrics.METRIC_MAPPING`)
- `default_metric` (from settings)

Studio consumes this to:

- populate metric dropdown
- prefill certain defaults
- reflect topic-mode capability flags

---

## 7) Enrichment panel status in Topic Studio

For this UI branch, the dedicated Topic Studio Enrichment control block was removed because it was not wired in the active Topic Studio action path and caused confusion.

Note:

- Topic Foundry still has enrichment concepts/services in backend domain code.
- This handoff is specifically about current Topic Studio UX wiring and intentional removal of unbaked front-end controls.

---

## 8) Smoke/validation scripts available

Repo contains reusable scripts under `/scripts`, including:

- `smoke_topic_foundry_all.sh`
- `smoke_topic_foundry_preview.sh`
- `smoke_topic_foundry_train.sh`
- `smoke_topic_foundry_run_results_inspector.sh`
- `smoke_topic_foundry_preview_detail.sh`
- `smoke_topic_foundry_bertopic.sh`

Use these as first-line rebuild verification before deeper debugging.

---

## 9) Rebuild checklist (ordered)

1. **Hub proxy**
   - Confirm `/api/topic-foundry/{path:path}` is enabled and points to correct `TOPIC_FOUNDRY_BASE_URL`.
2. **Capabilities handshake**
   - Verify Studio loads `/capabilities` and metric dropdown gets real options.
3. **Dataset flow**
   - Validate create/list/patch/preview flows with one known source table.
4. **Model creation flow**
   - Confirm auto-dataset-create works when no dataset preselected.
5. **Training flow**
   - Start run, poll completion, verify summary stats update.
6. **Result inspector flow**
   - Confirm full pagination load (not first page only).
   - Confirm topic distribution chart renders percentages.
7. **BERTopic kwargs trace**
   - Pass `nr_topics` in Params JSON; confirm it appears in effective training metadata/log path.
8. **Regression tests**
   - Run focused tests (see section 10).

---

## 10) Test coverage that should stay green

- `tests/test_hub_js_syntax.py`
- `tests/test_topic_foundry_capabilities_defaults.py`
- `tests/test_topic_foundry_model_meta_merge.py`
- `tests/test_topic_foundry_vectorizer_stop_words.py`
- `tests/test_topic_foundry_training_json_safe.py`
- `tests/test_topic_foundry_nr_topics.py`

---

## 11) Known gotchas / risk areas

1. **Conflicting kwargs source**
   - Many BERTopic knobs may appear in both `model_meta` and `model_spec.params`.
   - Keep merge order deterministic and documented.

2. **Capabilities drift vs UI assumptions**
   - UI assumes `supported_metrics` and `default_metric` are present.

3. **Dataset auto-create validation**
   - Auto-create requires form completeness (name/source/id/time/text columns).

4. **Result table scale**
   - Full pagination can be heavy on very large runs; tune page size only if needed.

5. **Unbaked feature temptation**
   - Avoid reintroducing dormant controls in UI without a bound handler and tested API path.

---

## 12) “If I had to rebuild from scratch tomorrow” minimal plan

1. Recreate proxy route in Hub.
2. Recreate Topic Studio fetch helpers and state restore/save.
3. Recreate dataset create/preview path.
4. Recreate model creation payload path with auto-dataset-create fallback.
5. Recreate training/polling/results flow with pagination + chart.
6. Recreate topic engine builder and kwargs merge path.
7. Recreate capabilities contract for metrics/defaults.
8. Re-add focused tests listed above.

That sequence gets you from zero to functional parity fastest.

