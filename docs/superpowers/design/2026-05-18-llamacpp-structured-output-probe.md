# llama.cpp structured output probe (design audit)

Date: 2026-05-18

## Pinned runtime

- **Image:** `ghcr.io/ggml-org/llama.cpp:server-cuda-b8740` (Dockerfile `ARG LLAMACPP_IMAGE_TAG=server-cuda-b8740`, overridable at build).
- **Rationale (wrapper README):** Qwen3-related flags (`--reasoning-budget`, `--chat-template-kwargs`) and stable OpenAI-compatible `llama-server` surface.

## How the wrapper launches llama-server

- `services/orion-llamacpp-host/app/main.py` builds argv from `config/llm_profiles.yaml` via `LlamaCppConfig` + `thinking_policy.resolve_thinking_launch_policy`.
- Emits `--chat-template-kwargs`, `--reasoning-budget`, `--jinja` when supported (flag probe via `--help`).
- Container runs upstream `llama-server` binary from the pinned GHCR image.

## Live worker surface

- **Health:** `GET {base}/health` (used by verify scripts and probe).
- **Chat:** `POST {base}/v1/chat/completions` (OpenAI-compatible).
- **Optional (non-fatal in probe):** `/props`, `/slots`, `/models` if present.

## Gateway: `response_format` today

- `services/orion-llm-gateway/app/llm_backend.py` `_execute_openai_chat`:
  - For `vllm`, `llamacpp`, `llama-cola`: forwards `opts["response_format"]` when set.
  - Else if `opts["return_json"]`: sends `{"type": "json_object"}` only.
- No schema-constrained builder existed before this work; callers could pass raw `response_format` but nothing proved which shape b8740 accepts.

## Gateway: `chat_template_kwargs` today

- Same path: if `backend_name in ("llamacpp", "llama-cola")` and `opts["chat_template_kwargs"]` is a non-empty dict, it is forwarded on the JSON body (per-request Qwen3 `enable_thinking` without worker restart).

## Thinking separation

- After completion, gateway:
  - Strips `` / `` blocks from visible `content` via `_split_think_blocks`.
  - Stores provider `reasoning_content` / `reasoning` / `reasoning_text` separately (not merged into artifact text).
- Cortex-exec `router.py` also sanitizes structured JSON for verbs like `memory_graph_suggest`.
- **Risk:** If thinking is enabled on the artifact route, delimiters can appear in `content` before strip; artifact routes must force `enable_thinking: false`.

## `response_format` shape uncertainty (why probe, not docs)

Upstream llama.cpp server README (current master) documents schema-constrained JSON roughly as:

- `{"type": "json_object"}`
- `{"type": "json_object", "schema": {...}}`
- `{"type": "json_schema", "schema": {...}}`

OpenAI’s nested wrapper `{"type":"json_schema","json_schema":{"name","strict","schema"}}` may **not** match what b8740 enforces. Historical issues and doc drift mean we **must not** assume OpenAI parity.

**Probe script:** `services/orion-llamacpp-host/scripts/probe_structured_output.py` tests methods A–F on the **live** worker, records artifacts, and selects `best_method` by real schema enforcement (adversarial `extra` key test).

## Orion wiring (post-probe)

- **Probe** → set `MEMORY_GRAPH_SUGGEST_STRUCTURED_OUTPUT_METHOD=<best_method>` (or `LLM_STRUCTURED_OUTPUT_METHOD` for gateway default).
- **Gateway** `structured_output.py` builds `response_format` from method + schema (no hot-path probe).
- **Hub** `memory_graph_suggest.py` passes `structured_output_*` options + `disabled_for_artifact` thinking policy; Pydantic + semantic lint remain the safety net.

## Fallback if enforcement fails

Report honestly (`best_method=json_object_only` or `none`). Keep parse/validate/retry. Next options: `/completion` + top-level `json_schema`, dedicated llama-cpp-python extractor, image bump + re-probe, or GBNF from compact schema.
