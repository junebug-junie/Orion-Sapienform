# PR Report: Memory graph suggest `no_json_object` after chat turn

**Branch:** `feat/memory-graph-suggest-fix-v1`  
**Date:** 2026-05-25

## Problem

After a Hub chat turn (Grounded Small lane), opening **Memory graph from chat** and clicking **Suggest draft** returned:

- Valid empty `SuggestDraftV1` (no entities/situations/edges)
- `validation_errors=no_json_object` on **both** quick and brain routes
- `api_error: memory_graph_suggest_failed`

The model/gateway often returned JSON in the OpenAI `raw.choices[].message.content` shape while cortex `final_text` and step `content` were empty — especially when `MEMORY_GRAPH_SUGGEST_STRUCTURED_OUTPUT_METHOD=none` (no schema-constrained `response_format`).

## Root cause

1. **Structured output disabled by default** — `none` meant the gateway did not send schema-constrained JSON; cortex-exec rejects non-JSON for `memory_graph_suggest`, yielding empty `final_text`.
2. **Hub text extraction gap** — `hub_memory_graph_suggest_text` did not read gateway `raw.choices` when `content` was empty.

## Fix

| Area | Change |
|------|--------|
| Defaults | `MEMORY_GRAPH_SUGGEST_STRUCTURED_OUTPUT_METHOD` default → `json_object_schema` (settings, docker-compose, `.env_example`) |
| Legacy env | Runtime maps `none` → `json_object_schema` (including `auto` + env `none`) |
| Hub extraction | `cortex_memory_graph_text.py` reads `raw.choices[].message.{content,reasoning_*}`; prefers JSON blobs; checks `inline_think_content` |
| UI coalesce | `memory-graph-draft-ui.js` also checks `reasoning_content` in step blocks |

## Commits

1. `d4075a4a` — structured output default + raw choices extraction  
2. (review) — `auto`+`none` normalization + `suggest_with_escalation` integration test

## Test plan

```bash
cd .worktrees/feat-memory-graph-suggest-fix-v1
PYTHONPATH=. /path/to/venv/bin/python -m pytest \
  services/orion-hub/tests/test_cortex_memory_graph_text.py \
  services/orion-hub/tests/test_memory_graph_structured_output_resolve.py \
  services/orion-hub/tests/test_memory_graph_structured_output.py \
  services/orion-hub/tests/test_memory_graph_suggest_escalation.py \
  services/orion-hub/tests/test_memory_graph_suggest_coalesce_ui.py \
  -q
```

**Manual smoke**

1. Hub chat turn (Grounded Small).
2. **Memory graph** on assistant turn → select turns → **Suggest draft**.
3. Expect populated draft JSON or a non-`no_json_object` diagnostic (e.g. RDF/semantic validation), not empty graph with `no_json_object` on both routes.

## Local `.env` sync

Your `services/orion-hub/.env` already has `MEMORY_GRAPH_SUGGEST_STRUCTURED_OUTPUT_METHOD=json_object_schema`. If it still says `none`, set:

```env
MEMORY_GRAPH_SUGGEST_STRUCTURED_OUTPUT_METHOD=json_object_schema
```

Restart **orion-hub** after changing env.

## Out of scope

- No `orion/bus/channels` or `orion/schemas/registry` changes (no new contracts).
- Cortex-exec `final_text` assembly unchanged; hub now recovers from step `raw` when assembly is empty.
