# Topic Foundry (Windowing v2, Micro/Macro Topics, Enrichment)

## Windowing v2
Topic Foundry now exposes explicit windowing modes that are persisted with each run:

- `rows`: one document per unit. **The default since 2026-08-28.**
- `turn_pairs`: pairs units into two-turn windows. With `split_text_columns`
  on, a pair is one row's own prompt+response; with it off, two consecutive
  rows. Pairs never straddle a row boundary in split mode.
- `fixed_k_rows`: fixed-size windows (`fixed_k_rows`) with optional `fixed_k_rows_step` stride.
- `time_gap`: windows split when `time_gap_seconds` is exceeded.
- `conversation_bound`: never cross a conversation boundary column.
- `conversation_bound_then_time_gap`: split by boundary, then apply time-gap chunking.

### Column splitting and speakers (2026-08-28)

Two windowing fields control how a row becomes documents, and are part of the
model's frozen `windowing_spec` (and of the model-name fingerprint, so changing
them always mints a new model rather than silently retraining an existing one):

- `split_text_columns` (default `false`): emit one document per
  `(row, text column)` instead of concatenating a row's text columns into one
  blob. A prompt and its response are two different speech acts; fusing them
  averages both into a single vector that represents neither whenever they
  diverge in topic.
- `column_speakers` (default `{}`): maps a text column to the speaker who
  authored it, e.g. `{"prompt": "juniper", "response": "orion"}`. This is
  recorded metadata, not inference -- the column *is* the speaker.

The resolved speakers are carried on each segment's
`provenance.speakers` (and mirrored into `documents.jsonl`), **never** written
into the document text. A speaker label inside the text gets embedded and
tf-idf'd along with the content: on a corpus split roughly in half by speaker
it is a near-perfect high-IDF discriminator, so the clusterer can end up
grouping by who was talking rather than what was said. Consumers that need the
speaker read it from provenance.

`include_roles` filters on these speakers. It defaults to `[]` (no filtering);
setting it to values that match no speaker drops every block and fails training
with "No documents available".

Defaults are deliberately the *old* behavior (`split_text_columns=false`,
`include_roles=[]`) because a model row freezes its `windowing_spec` at
creation and rows written before these fields existed have no key for them --
a non-conservative default would silently change how every pre-existing model
builds documents. Callers that want splitting say so explicitly; `orion-hub`'s
scheduler does.

### Boundary configuration (dataset-level)
Datasets can optionally declare a `boundary_column` (and `boundary_strategy="column"`). These are validated
against Postgres introspection metadata and are required for `conversation_bound*` windowing modes.

#### Conversation-bound preview example (via Hub proxy)
```bash
# Discover columns
curl "http://localhost:8080/api/topic-foundry/introspect/columns?schema=public&table=chat_logs"

# Create dataset with boundary_column
curl -sS -X POST "http://localhost:8080/api/topic-foundry/datasets" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "support_chat",
    "source_table": "public.chat_logs",
    "id_column": "id",
    "time_column": "created_at",
    "text_columns": ["user_text", "assistant_text"],
    "boundary_column": "conversation_id",
    "boundary_strategy": "column",
    "timezone": "UTC"
  }'

# Preview with conversation_bound windowing
curl -sS -X POST "http://localhost:8080/api/topic-foundry/datasets/preview" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "<dataset_id>",
    "windowing": {
      "windowing_mode": "conversation_bound",
      "boundary_column": "conversation_id",
      "fixed_k_rows": 2,
      "time_gap_seconds": 900,
      "max_window_seconds": 7200,
      "min_blocks_per_segment": 1,
      "max_chars": 1200
    },
    "limit": 100
  }'
```

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
