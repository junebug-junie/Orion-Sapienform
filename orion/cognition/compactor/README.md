# Shared compactor helpers

Common seam for compactor workflows (`github_compactor_pass`, `chat_history_compactor_pass`). Kind-specific packages (`orion/cognition/github_compactor/`, `orion/cognition/chat_history_compactor/`) own their constants, quiet-day builders, and journal-id composition, and delegate the shared mechanics here.

- `budget.py` — `assert_fields_within_budget`: fail-loud `compactor_output_over_budget:<field>` checks for digest outputs.
- `digest.py` — `parse_compactor_digest_json(raw, model_cls)`: LLM JSON → typed digest model; rejects non-object payloads with `compactor_digest_not_object`.
- `index.py` — `build_compactor_index`: stable window keys for indexed (upsert-by-`compactor_index`) memory cards.

The digest verb request/response plumbing both workflows share (`_build_compactor_digest_request`, `_compactor_digest_from_payload`) lives in `services/orion-cortex-orch/app/workflow_runtime.py` because it depends on orch request envelopes.

Rule of thumb: a new compactor kind should add a sibling package with its constants and quiet builder, reuse these helpers, and never fork the budget/parse error tokens.
