# Shared compactor helpers

Common seam for compactor workflows (`github_compactor_pass`, `chat_history_compactor_pass`). Kind-specific packages (`orion/cognition/github_compactor/`, `orion/cognition/chat_history_compactor/`) own their constants, quiet-day builders, and journal-id composition, and delegate the shared mechanics here.

- `budget.py` — `fit_fields_within_budget`: trims over-budget digest prose to its cap at a word
  boundary and reports which fields it touched. The cap is enforced by repair, not rejection:
  by the time it runs the payload has already parsed and passed the digest model's validation,
  so an over-long summary is a formatting miss on sound content and the full narrative survives
  in `journal_body`. The ellipsis is paid for out of the budget, so `len(value) <= max_chars`
  holds exactly.
- `digest.py` — `parse_compactor_digest_json(raw, model_cls)`: LLM JSON → typed digest model; rejects non-object payloads with `compactor_digest_not_object`.
- `index.py` — `build_compactor_index`: stable window keys for indexed (upsert-by-`compactor_index`) memory cards.

The digest verb request/response plumbing both workflows share (`_build_compactor_digest_request`, `_compactor_digest_from_payload`) lives in `services/orion-cortex-orch/app/workflow_runtime.py` because it depends on orch request envelopes.

Rule of thumb: a new compactor kind should add a sibling package with its constants and quiet builder, reuse these helpers, and never fork the budget/parse error tokens.
