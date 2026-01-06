# Recall Service (orion-recall)

This service is the single canonical hub for retrieving memory across vector and RDF backends. Verbs issue exactly one bus RPC:

```
recall.query.v1  ->  recall.reply.v1
```

Telemetry is emitted on every request as `recall.decision.v1` and persisted when Postgres is configured.

## Flow

```mermaid
sequenceDiagram
    participant Verb as Verb (reflect)
    participant Exec as Cortex Exec
    participant Recall as orion-recall
    participant Vector as Vector backend
    participant RDF as RDF backend
    participant LLM as LLM

    Verb->>Exec: plan step (RecallService, recall_profile)
    Exec->>Recall: recall.query.v1 (corr_id)
    Recall->>Vector: vector search (top_k)
    Recall->>RDF: rdf neighborhood (top_k)
    Recall-->>Recall: fuse + dedupe + render (MemoryBundle)
    Recall-->>Exec: recall.reply.v1 (MemoryBundleV1)
    Exec->>LLM: prompt with memory_bundle.rendered
```

## Bus Contracts

```mermaid
flowchart LR
    Q[recall.query.v1<br/>payload: RecallQueryV1] --> R[recall.reply.v1<br/>payload: RecallReplyV1]
    Q --> T[recall.decision.v1<br/>payload: RecallDecisionV1]
```

### Envelope

All messages use the Titanium envelope fields:

- `corr_id` / `correlation_id`
- `event` / `kind`
- `service` / `source`
- `reply_to`
- `ts`
- `ttl_ms`
- `trace`

### Payloads

- `RecallQueryV1`: fragment, verb, intent, session_id, node_id, profile, reply_to
- `RecallReplyV1`: bundle (MemoryBundleV1)
- `MemoryBundleV1`:
  - `rendered`: prompt-ready bullet list
  - `items[]`: id, source, source_ref, uri, score, ts, title, snippet, tags
  - `stats`: backend_counts, latency_ms, profile
- `RecallDecisionV1`: corr_id, session_id, node_id, verb, profile, query, selected_ids, backend_counts, latency_ms, dropped
- SQL timeline: optional source reading from Postgres (e.g., `collapse_mirror`), controlled via profile (`enable_sql_timeline`, `sql_top_k`, `sql_since_minutes`). Items are emitted with `source="sql_timeline"` and `source_ref` identifying the table.

## Profiles

Profiles live in `orion/recall/profiles/*.yaml`:

- `reflect.v1`: balanced vector+RDF
- `assist.light.v1`: cheap, vector-focused
- `deep.graph.v1`: RDF-heavy neighborhood

Fields:

- `vector_top_k`, `rdf_top_k`
- `max_per_source`, `max_total_items`
- `time_decay_half_life_hours` (reserved)
- `render_budget_tokens`
- `enable_query_expansion`

## Prompt Usage

Prompts should consume the rendered bundle:

```jinja2
{{ memory_bundle.rendered | default("No additional memory context provided.") }}
```

The executor places the full `memory_bundle` in the step context; raw vector/RDF calls in verbs are no longer allowed.
