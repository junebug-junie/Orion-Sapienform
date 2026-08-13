---
type: "query"
date: "2026-08-13T06:14:39.982640+00:00"
question: "How does a bus event become a SQL row in orion-sql-writer, and is the write an upsert?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["worker.py", "_write_row", "MODEL_MAP", "INSERT_ONLY_MODELS"]
---

# Q: How does a bus event become a SQL row in orion-sql-writer, and is the write an upsert?

## Answer

handle_envelope -> settings.route_map[env.kind] -> MODEL_MAP -> _write (pydantic validate + model_dump) -> _write_row, which filters to mapper columns then sess.merge() (real PK upsert) unless the model is in INSERT_ONLY_MODELS (plain add + duplicate-key catch).

## Outcome

- Signal: useful

## Source Nodes

- worker.py
- _write_row
- MODEL_MAP
- INSERT_ONLY_MODELS