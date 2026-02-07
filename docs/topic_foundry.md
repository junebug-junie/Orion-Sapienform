# Topic Foundry (Windowing v2, Micro/Macro Topics, Enrichment)

## Windowing v2
Topic Foundry now exposes explicit windowing modes that are persisted with each run:

- `turn_pairs`: pairs rows into two-turn windows.
- `fixed_k_rows`: fixed-size windows (`fixed_k_rows`) with optional `fixed_k_rows_step` stride.
- `time_gap`: windows split when `time_gap_seconds` is exceeded.
- `conversation_bound`: never cross a conversation boundary column.
- `conversation_bound_then_time_gap`: split by boundary, then apply time-gap chunking.

### Boundary configuration (dataset-level)
Datasets can optionally declare a `boundary_column` (and `boundary_strategy="column"`). These are validated
against Postgres introspection metadata and are required for `conversation_bound*` windowing modes.

## LLM gating (optional)
Windowing can optionally apply LLM gating after candidate windows are built:

```json
{
  "llm_filter_enabled": false,
  "llm_filter_prompt_template": "You are filtering candidate topic windows... {window_text}",
  "llm_filter_max_windows": 200,
  "llm_filter_policy": "keep"
}
```

If LLM is disabled/unavailable, the system continues without gating.

## Micro vs Macro runs
Runs now include `run_scope`:

- `macro`: global clustering over the full dataset windows.
- `micro`: clustering within conversation boundaries.

For micro runs, the service attempts to map micro topic centroids to the latest macro topics for the same model,
storing the `parent_topic_id` mapping in `topic_foundry_topics`.

## Enrichment endpoint (segments/topics)
Use `POST /runs/{run_id}/enrich` to enrich segments, topics, or both. The endpoint is idempotent unless `force=true`.

Example:
```bash
curl -sS -X POST http://localhost:8615/runs/${RUN_ID}/enrich \
  -H "Content-Type: application/json" \
  -d '{
    "target": "both",
    "fields": ["title","aspects","meaning","sentiment"],
    "force": false
  }'
```

## Full text detail
Segment detail supports full text:

```bash
curl -sS "http://localhost:8615/segments/${SEGMENT_ID}?include_full_text=true"
```

## Smoke scripts
The repo includes smoke scripts for common flows:

- `scripts/smoke_topic_foundry_introspect.sh`
- `scripts/smoke_topic_foundry_preview.sh`
- `scripts/smoke_topic_foundry_train.sh`
- `scripts/smoke_topic_foundry_facets.sh`
- `scripts/smoke_topic_foundry_enrich.sh`
