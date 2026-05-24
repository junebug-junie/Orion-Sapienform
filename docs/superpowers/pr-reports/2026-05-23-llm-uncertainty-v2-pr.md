# PR: LLM Uncertainty v2 (native completion, SQL scalars, collapse + journal index)

**Branch:** `feat/llm-uncertainty-v2`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-llm-uncertainty-v2`

## Summary

Extends merged PR #608 summary-only `llm_uncertainty` with four additive surfaces:

1. **Native llama.cpp aligned generation** — `POST /apply-template` → `POST /completion` with `n_probs` (not a side probe; probabilities describe the same completion text returned to Orion).
2. **`chat_history_log` scalar columns** — queryable denormalized fields; `spark_meta.llm_uncertainty` remains JSON source of truth.
3. **Collapse Mirror telemetry** — `state_snapshot.telemetry.llm_uncertainty` + semantics key.
4. **`journal_entry_index`** — JSONB + summary scalars (not `journal_entries`).

Framed throughout as **language surface stability**, not factual confidence (`language_surface_stability_not_truth`).

## Architecture

```text
[opt-in] messages + return_logprobs + logprob_probe_mode=native_completion
  → llama.cpp /apply-template → prompt
  → llama.cpp /completion (n_probs, post_sampling_probs=false)
  → summarize_logprob_content → llm_uncertainty (source=llamacpp_native_completion)
  → ChatResultPayload.meta → cortex/hub spark_meta
  → sql-writer: spark_meta JSON + chat_history_log scalars
  → collapse: state_snapshot.telemetry.llm_uncertainty
  → journal index: llm_uncertainty JSON + scalars

[default unchanged] /v1/chat/completions logprobs → llamacpp_openai_chat source
```

## Changes by area

### orion-llm-gateway
- `native_completion_probs_to_logprob_content` + `extract_llm_uncertainty_from_native_completion`
- `_execute_llamacpp_native_completion` + `run_llm_chat` routing when all flags set
- Settings: `LLM_LOGPROB_NATIVE_COMPLETION_ENABLED`, `LLM_LOGPROB_NATIVE_COMPLETION_MAX_TOKENS`
- `.env_example`, `docker-compose.yml` parity

### orion-sql-writer
- Idempotent `ALTER TABLE` for 8 `chat_history_log` columns + 4 `journal_entry_index` columns
- `_chat_history_llm_uncertainty_scalars()` on chat history write
- Journal index via `stance_metadata` / top-level `llm_uncertainty` merge in worker

### orion-cortex-exec
- `attach_llm_uncertainty_to_collapse_payload()` in `orion/schemas/collapse_mirror.py`
- MetacogDraftService attaches from `ctx.metadata` or `llm_res.meta`

### orion/journaler
- `JournalEntryIndexV1` optional uncertainty fields
- `build_journal_entry_index_payload` populates from `stance_metadata.llm_uncertainty`

## Opt-in gates

| Layer | Requirement |
|-------|-------------|
| Gateway global | `LLM_LOGPROB_SUMMARY_ENABLED=true` |
| Native path | `LLM_LOGPROB_NATIVE_COMPLETION_ENABLED=true` |
| Per request | `options.return_logprobs=true` |
| Native mode | `options.logprob_probe_mode=native_completion` |
| Backend | `llamacpp` route target |

**Local enablement (committed in `.env_example`; sync `.env` on your machine):**

| Service | Variables |
|---------|-----------|
| orion-llm-gateway | `LLM_LOGPROB_SUMMARY_ENABLED=true`, `LLM_LOGPROB_NATIVE_COMPLETION_ENABLED=true` |
| orion-mind | `MIND_LLM_RETURN_LOGPROBS_SEMANTIC=true`, `MIND_LLM_LOGPROB_PROBE_MODE=native_completion` |
| orion-cortex-exec | `CORTEX_METACOG_RETURN_LOGPROBS=true`, `CORTEX_METACOG_LOGPROB_PROBE_MODE=native_completion` |

**Not changed (no new surface):** `orion/schemas/registry.py` (existing `JournalEntryIndexV1` entry covers new fields), `orion/bus/channels`, `requirements.txt` (no new dependencies).

## Test evidence

```bash
cd .worktrees/feat-llm-uncertainty-v2
pytest services/orion-llm-gateway/tests/test_llm_uncertainty.py \
  services/orion-llm-gateway/tests/test_llm_backend.py -q
pytest services/orion-sql-writer/tests/test_llm_uncertainty_spark_meta.py \
  services/orion-sql-writer/tests/test_journal_entry_indexing.py -q
pytest services/orion-cortex-exec/tests/test_collapse_llm_uncertainty_telemetry.py \
  services/orion-cortex-exec/tests/test_llm_uncertainty_metadata.py -q
```

All targeted suites pass (51+ tests across gateway, sql-writer, cortex-exec).

## Known limitations / follow-ups

- **Think-block alignment:** Native (and OpenAI) uncertainty is summarized from raw completion tokens before `_split_think_blocks()`; hidden think spans may be included in metrics while visible text is stripped.
- **Caller wiring:** Enable native path on semantic/metacog chat when ready (`logprob_probe_mode: "native_completion"`).
- **Metacog integration test:** Unit tests cover helper; optional executor mock test for regression.

## Commits

1. `docs: add LLM uncertainty v2 implementation plan`
2. `feat(llm-gateway): extract llm_uncertainty from native llama.cpp completion`
3. `feat(llm-gateway): add native llama.cpp completion path for aligned logprobs`
4. `feat(sql-writer): add chat_history_log llm uncertainty scalar columns`
5. `feat(cortex-exec): attach llm_uncertainty to collapse mirror telemetry`
6. `feat(journal): persist llm_uncertainty on journal_entry_index`
7. `fix(llm-gateway): strip completion_probabilities from native raw when summary-only`
