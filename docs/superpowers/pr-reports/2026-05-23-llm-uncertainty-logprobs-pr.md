# PR: LLM logprob summary (`llm_uncertainty`) at gateway + Mind telemetry

**Branch:** `feat/llm-uncertainty-v1`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-llm-uncertainty-v1`

## Summary

Adds a canonical, **summary-only** `llm_uncertainty` object at the LLM Gateway boundary (OpenAI-compatible `/v1/chat/completions` logprobs), propagates it through execution metadata and chat `spark_meta`, and wires it into **Mind semantic synthesis** phase telemetry with an optional advisory metacog trigger. Framed as **language surface stability**, not factual confidence.

## Architecture

```text
llama.cpp /v1/chat/completions (logprobs opt-in)
  → orion-llm-gateway: summarize → result["llm_uncertainty"]
  → ChatResultPayload.meta["llm_uncertainty"]
  → orion-cortex-exec: PlanExecutionResult.metadata
  → orion-hub: spark_meta (existing gateway_meta merge)
  → orion-sql-writer: spark_meta JSONB
  → orion-mind: MindPhaseTelemetry.llm_uncertainty (semantic_synthesis opt-in)
  → optional: orion.metacog.trigger.v1 (llm_surface_instability, default off)
```

## Changes by service

### orion-llm-gateway
- New `app/llm_uncertainty.py` — extract + summarize from `choices[0].logprobs.content`
- `_execute_openai_chat` forwards `logprobs`/`top_logprobs` when `return_logprobs` + `LLM_LOGPROB_SUMMARY_ENABLED`
- Strips per-token logprobs from `raw` when `logprob_summary_only` (default true)
- `handle_chat` exposes `meta["llm_uncertainty"]`
- Env knobs: `LLM_LOGPROB_*` in settings, `.env_example`, docker-compose

### orion-cortex-exec
- Forwards gateway `meta.llm_uncertainty` → `ctx["metadata"]` → `PlanExecutionResult.metadata`

### orion-sql-writer
- Merges `llm_uncertainty` from payload `meta` or `spark_meta` into chat history `spark_meta`

### orion-mind
- `MindLLMClient` propagates `result.meta.llm_uncertainty`
- `MindPhaseTelemetry.llm_uncertainty` on semantic synthesis when `MIND_LLM_RETURN_LOGPROBS_SEMANTIC=true`
- Advisory metacog publish when `MIND_LLM_UNCERTAINTY_METACOG_ENABLED=true` (after successful synthesis only)

## Opt-in gates (all default off)

| Layer | Flag |
|-------|------|
| Gateway global | `LLM_LOGPROB_SUMMARY_ENABLED=false` |
| Per request | `options.return_logprobs` |
| Mind semantic | `MIND_LLM_RETURN_LOGPROBS_SEMANTIC=false` |
| Metacog trigger | `MIND_LLM_UNCERTAINTY_METACOG_ENABLED=false` |

**Operator note:** Mind semantic logprobs require **both** `MIND_LLM_RETURN_LOGPROBS_SEMANTIC=true` **and** `LLM_LOGPROB_SUMMARY_ENABLED=true` on the gateway. Enabling only the Mind flag is a silent no-op; Mind logs `mind_llm_logprobs_requested_but_no_uncertainty_summary` when that happens.

**Debug only:** Set `logprob_summary_only: false` in request options to retain full per-token logprobs in `ChatResultPayload.raw` (default strips them).

## `llm_uncertainty` v1 shape

```json
{
  "schema_version": "v1",
  "source": "llamacpp_openai_chat",
  "available": true,
  "diagnostic_only": true,
  "confidence_semantics": "language_surface_stability_not_truth",
  "token_count_observed": 128,
  "mean_logprob": -0.71,
  "min_logprob": -4.9,
  "mean_top1_margin": 1.35,
  "low_margin_token_count": 9,
  "low_logprob_token_count": 6,
  "entropy_proxy_mean": 0.42,
  "unstable_span_count": 2
}
```

## Test plan

- [x] `orion-llm-gateway` tests: 50 passed (extractor, passthrough, meta, strip raw, gate)
- [x] `orion-cortex-exec` `test_llm_uncertainty_metadata.py`: 3 passed
- [x] `orion-sql-writer` `test_llm_uncertainty_spark_meta.py`: 2 passed
- [x] `orion-mind` uncertainty tests: 11+ passed (incl. status gate, async envelope)
- [x] Rebased onto `origin/main` (substrate MVP retained from main)
- [ ] Staging: `LLM_LOGPROB_SUMMARY_ENABLED=true` **and** `MIND_LLM_RETURN_LOGPROBS_SEMANTIC=true`; confirm `meta.llm_uncertainty` + `spark_meta`
- [ ] Phase 5 (Collapse Mirror / journal index) deferred per plan

## Local env sync (not committed)

After pull, copy new keys from `.env_example` → `.env` for:
- `services/orion-llm-gateway/.env`
- `services/orion-mind/.env`

## Out of scope (deferred)

- Native llama.cpp `/completion` `n_probs` probe mode
- New SQL columns for `mean_logprob`
- Collapse Mirror `state_snapshot.telemetry` (Phase 5)
- Journal entry index attachment (Phase 5)

## Commits

11 implementation commits + 1 docs plan + 1 review-fix commit on `feat/llm-uncertainty-v1`.
