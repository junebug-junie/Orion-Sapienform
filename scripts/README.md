# Smoke Tests

Run these scripts to validate bus publishing and GraphDB ingestion.

## Chat History → RDF
```bash
python scripts/smoke_chat_to_rdf.py --redis redis://localhost:6379/0
```
Expected output:
```
PASS [...]
```

## Tags Enrichment → RDF
```bash
python scripts/smoke_tags_to_rdf.py --redis redis://localhost:6379/0
```
Expected output:
```
PASS [...]
```

## Juniper Collapse Fanout → Triage + SQL
```bash
python scripts/smoke_juniper_collapse_fanout.py --redis redis://localhost:6379/0
```
Expected output:
```
PASS triage=ok sql-write=ok
```

## ChatGPT Export Import (Bus Fanout)

### What this does
- Publishes chat history MESSAGE events (chat.history.message.v1) to `orion:chat:history:log`
  → sql-writer writes `chat_message`
  → vector-host embeds and publishes `vector.upsert.v1`
  → vector-writer upserts into Chroma
- Publishes ChatGPT TURN events (chat.gpt.log.v1) to `orion:chat:gpt:log`
  → sql-writer writes `chat_gpt_log`

### Preconditions (for full fanout)
These services must be running and connected to the same ORION_BUS_URL:
- orion-sql-writer
- orion-vector-host (otherwise no embeddings will be produced)
- orion-vector-writer
- orion-meta-tags
- orion-rdf-writer

### Idempotence / “no worries about doubling”
This importer is SAFE to re-run. It uses deterministic UUIDs:
- Each conversation turn gets a stable correlation_id and turn id derived from (conversation_id, user_node_id, assistant_node_id).
- Each message gets a stable message_id derived from (conversation_id, node_id).
Downstream writers upsert by primary key/doc_id:
- SQL uses primary keys (id) so rows are overwritten, not duplicated.
- Chroma upserts by doc_id, so vectors overwrite, not duplicate.
- RDF stores triples as a set; re-inserting is safe.
Even if you run the same export twice, you will not “double” the stored history.

Default turn channel: `--channel-turn orion:chat:gpt:log` (override supported).

### Usage (dry run)
Example:
```bash
python scripts/import_chatgpt_export.py \
  --export ~/Downloads/chatgpt-export.zip \
  --dry-run
```

### Usage (full fanout publish)
```bash
python scripts/import_chatgpt_export.py \
  --export ~/Downloads/chatgpt-export.zip \
  --bus-url redis://100.x.y.z:6379/0 \
  --rate-limit 25
```

### Usage (periodic runs with checkpointing)
This avoids reprocessing old conversations (per-conversation update_time/message_time checkpoints):
```bash
python scripts/import_chatgpt_export.py \
  --export ~/Downloads/chatgpt-export.zip \
  --state-file scripts/.state/chatgpt_import.json \
  --rate-limit 25
```

### Include all branches (optional)
```bash
python scripts/import_chatgpt_export.py \
  --export ~/Downloads/chatgpt-export.zip \
  --include-branches \
  --rate-limit 10
```

By default, `--include-branches` publishes MESSAGE events only (to avoid cross-branch turn pairing). To explicitly control output:
- `--only-messages`: publish message events only
- `--only-turns`: publish turn events only (not supported with `--include-branches`)

### Implementation notes
- Do not add new dependencies.
- Do not modify sql-writer/vector-host/vector-writer/rdf-writer/meta-tags.
- Keep changes limited to the importer module + CLI + scripts/README.md.
