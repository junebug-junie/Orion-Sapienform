---
type: "query"
date: "2026-08-20T00:52:38.675278+00:00"
question: "grammar_events table readers and retention"
contributor: "graphify"
outcome: "dead_end"
source_nodes: ["GrammarEventV1", "store.py", "retention.py"]
---

# Q: grammar_events table readers and retention

## Answer

BFS returned 156 mostly-irrelevant schema/reducer nodes; no table-level read sites. Had to fall back to grep + live psql.

## Outcome

- Signal: dead_end

## Source Nodes

- GrammarEventV1
- store.py
- retention.py