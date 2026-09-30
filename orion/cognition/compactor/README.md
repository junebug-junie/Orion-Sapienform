# Shared compactor helpers

Common seam for compactor workflows (`github_compactor_pass`, `chat_history_compactor_pass`). Kind-specific packages (`orion/cognition/github_compactor/`, `orion/cognition/chat_history_compactor/`) own their constants, quiet-day builders, and journal-id composition, and delegate the shared mechanics here.

Both compactors summarize the FULL window (every merged PR / every chat turn of the previous
Denver calendar day on a scheduled run) and store `journal_body` untruncated, because it is
embedded verbatim in the daily letter. Volume is handled by map-reduce, never by dropping input.

- `constants.py` — shared digest-call budgets: `DIGEST_LLM_ROUTES` (`agent`, then an `agent`
  retry; never Juniper's reserved `chat` lane), `DIGEST_INPUT_CHAR_BUDGET` (per-call input,
  sized to the smallest context an agent-class call can land on), `DIGEST_MAX_TOKENS`
  (sent as `options.max_tokens`), and wall-clock budgets (per call and per pass).
- `calendar_day.py` — `previous_local_day_window`: the one definition of "yesterday, Denver".
- `chunking.py` — `chunk_items_by_char_budget`: order-preserving split; every item lands in a chunk.
- `budget.py` — `fit_fields_within_budget`: trims over-budget memory-card prose (`card_summary`,
  `journal_title`) to its cap at a word boundary and reports which fields it touched. The
  journal body is NOT passed through it. The ellipsis is paid for out of the budget, so
  `len(value) <= max_chars` holds exactly.
- `digest.py` — `parse_compactor_digest_json(raw, model_cls)`: LLM JSON → typed digest model.
  Tolerant of raw control characters (`strict=False`) and a markdown code fence; an empty
  completion raises `compactor_digest_empty_completion` (retried by the orch); non-object
  payloads raise `compactor_digest_not_object`.
- `index.py` — `build_compactor_index`: stable window keys for indexed (upsert-by-`compactor_index`) memory cards.

The digest verb request/response plumbing both workflows share (`_build_compactor_digest_request`, `_compactor_digest_from_payload`, `_call_compactor_digest_with_retry`, `_run_compactor_digest` — single call, or chunk digests then one merge call, falling back to a recorded `merge_mode=concatenated` join if every merge attempt fails) lives in `services/orion-cortex-orch/app/workflow_runtime.py` because it depends on orch request envelopes.

Rule of thumb: a new compactor kind should add a sibling package with its constants and quiet builder, reuse these helpers, and never fork the budget/parse error tokens.

Workflow metadata evidence (both passes): `total_count`, `covered_count`, `input_truncated`
(true only if a single item hit its safety cap, the chat fetch hit `COMPACTOR_MAX_TURNS`, or
the GitHub page walk hit `GITHUB_PULLS_MAX_PAGES`), `digest_chunk_count`, `digest_merge_mode`,
`digest_llm_route`, `digest_attempts`, `journal_body_chars`.

Quiet windows: `github_compactor_pass` journals its quiet day; `chat_history_compactor_pass`
writes no journal entry and no card for a quiet window (only the workflow result records it).
