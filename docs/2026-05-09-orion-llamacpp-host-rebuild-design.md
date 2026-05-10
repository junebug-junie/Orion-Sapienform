# Design: Rebuild `orion-llamacpp-host` (llama.cpp pin, thinking policy, live verification)

**Date:** 2026-05-09  
**Path:** `docs/2026-05-09-orion-llamacpp-host-rebuild-design.md` (tracked; `docs/superpowers/specs/` is gitignored in this repo for local-only drafts).  
**Status:** Approved for spec (brainstorming); implementation follows separate plan after spec review.

## 1. Problem

- Operators run an **older** `llama-server` than the repo intends; profile knobs under `config/llm_profiles.yaml` → `llamacpp:` behave like **ghosts** (skipped silently or only logged) when flags are missing from `--help`.
- **Qwen3 “thinking off”** must match **upstream-documented** mechanisms, not guessed completion tags.
- Need **regression tests** (fast CI) plus **live verification** on a real GGUF before claiming success.
- **Secondary (v2):** toggle thinking **per request** without restarting the server.

## 2. Goals and non-goals

### v1 (must ship)

1. **Pinned, known-good image**  
   - `Dockerfile` `LLAMACPP_IMAGE_TAG` reflects the binary we support; README matches Dockerfile defaults and documents override via build arg.

2. **Explicit launch-time thinking policy** in `services/orion-llamacpp-host/app/main.py`  
   - Derive a clear **thinking intent** (on / off / default-from-model) from `LlamaCppConfig` (`chat_template_kwargs.enable_thinking`, `reasoning`, `reasoning_budget`, `reasoning_format` where relevant).  
   - Emit consistent argv: in particular, when applying **Qwen3-style “thinking off”** via `--reasoning-budget 0`, also emit **`--jinja`** when the binary supports it, because upstream discussion on [PR #13771](https://github.com/ggml-org/llama.cpp/pull/13771) shows **`--reasoning-budget 0` alone may be insufficient** on `llama-server` without **`--jinja`**.  
   - **Precedence:** explicit profile fields win over auto-derived defaults (e.g. if `reasoning_budget` is set explicitly, do not overwrite; if only `enable_thinking: false` and budget unset, policy may set effective budget `0` when flag exists). Document full precedence table in the implementation plan.

3. **Version evidence**  
   - For the chosen pin, record `llama-server --version` (numeric **b** build) in release notes or spec addendum.  
   - **Feature floor (documentation, not hand-wavy):**  
     - [`--reasoning-budget 0` for disabling thinking including Qwen3](https://github.com/ggml-org/llama.cpp/pull/13771) merged **2025-05-25**.  
     - [`--chat-template-kwargs` / API `chat_template_kwargs`](https://github.com/ggml-org/llama.cpp/pull/13196) merged **2025-06-29**.  
   - Implementation phase: confirm the pin’s binary contains both behaviors (e.g. `llama-server --help` lists required flags; optional `git tag --contains <merge>` on upstream for the exact tag if we need a hard **b_min**).

4. **Testing (two layers)**  
   - **Fast:** pytest argv construction with monkeypatched `_get_supported_llama_server_flags` / `_get_llama_server_build` for thinking-on and thinking-off profiles (including `--jinja` + `--reasoning-budget` pairing when policy says off).  
   - **Live (required for “thinking off verified”):** see §5.

5. **Live verification**  
   - Not optional for acceptance of thinking-off: see §5 Gates A and B.

### v2 (secondary; out of v1 scope unless explicitly added)

- **Per-request thinking** without relaunch: upstream [PR #13196](https://github.com/ggml-org/llama.cpp/pull/13196) documents **`chat_template_kwargs`** on the OpenAI-compatible API; future **per-request reasoning budget** is discussed in [issue #13272](https://github.com/ggml-org/llama.cpp/issues/13272).  
- **Likely implementation locus:** `orion-llm-gateway` forwards kwargs / future budget fields; `orion-llamacpp-host` remains one model per process. No host-side “router” in v1.

### Non-goals (v1)

- Multi-model dynamic routing inside one `orion-llamacpp-host` process.  
- Proving every profile in `llm_profiles.yaml` on every GPU SKU.  
- String-matching on completion output **without** anchoring to upstream examples or a **pin+GGUF baseline** captured in-repo.

## 3. Architecture (unchanged boundaries)

- **`orion-llamacpp-host`:** resolves `LLM_PROFILE_NAME`, optional HF download, builds `llama-server` argv, supervises process.  
- **`config/llm_profiles.yaml`:** per-worker defaults at launch.  
- **`orion-llm-gateway`:** unchanged in v1; v2 may pass per-request kwargs.

## 4. Thinking policy (behavioral summary)

- **Probe-first:** keep `_get_supported_llama_server_flags` / `_get_llama_server_build`; skip unsupported flags with warnings; retain ERROR logs when profile **requested** kwargs or `reasoning_budget` but flags could not be emitted.  
- **Centralize** “what we intend for thinking” in one place (function or small module) called from `build_llama_server_cmd_and_env`, then map to argv. Reduces drift between YAML comments and code.  
- **Jinja pairing:** when policy emits `--reasoning-budget 0` for Qwen3-style non-thinking, also emit `--jinja` if supported (per #13771 thread). When profile already uses `--reasoning-format` / `--jinja` for other reasons, avoid duplicate/conflicting argv (implementation plan details deduplication).  
- **Logging:** one structured line (or compact JSON log) per boot: intended thinking mode, and emitted/skipped for `--jinja`, `--chat-template-kwargs`, `--reasoning-budget`, `--reasoning`, `--reasoning-format`.

## 5. Live verification gates (acceptance)

Run against the **same Docker image** intended for production and a **reference Qwen3 GGUF** on a GPU host (variables e.g. `MODEL_QWEN3_PATH`, existing `CUDA_VISIBLE_DEVICES`).

**Gate A — `/apply-template` (contract from upstream)**  
- Start `llama-server` with argv equivalent to a **non-thinking** profile (full wrapper container acceptable).  
- `POST /apply-template` with body aligned to [PR #13196](https://github.com/ggml-org/llama.cpp/pull/13196) examples (`chat_template_kwargs: {"enable_thinking": false}`, same message text as in spec or upstream doc).  
- Assert response **`prompt` field** matches the **documented** expected pattern for that example (allowing only explicitly listed normalizations: whitespace, line endings, or HF model id string if we fix `model` in the request). **Do not** invent tags; follow PR #13196 or an appendix that cites Qwen template + that PR.

**Gate B — `/v1/chat/completions` (end-user path)**  
- Same server, fixed prompt and decoding params recorded in spec.  
- Assert “no thinking” using **only**:  
  - substrings **quoted** from official Qwen / llama.cpp documentation or issues, **or**  
  - a **golden excerpt** checked into the repo generated once on the **chosen pin + reference GGUF** (script documents how to refresh goldens when pin changes).

**Script home:** extend `services/orion-llamacpp-host/scripts/validate_llamacpp_upgrade.sh` or add `…/scripts/verify_qwen3_thinking_off_live.sh` invoked by the same workflow; document env vars and failure modes.

## 6. Operator / docs

- Align `services/orion-llamacpp-host/README.md` with `Dockerfile` default `LLAMACPP_IMAGE_TAG` and org (`ghcr.io/ggml-org/llama.cpp`).  
- Document Gates A/B in README under “Verifying thinking off after upgrade.”

## 7. Risks

- **Upstream regressions** or Qwen3.5-class bugs (e.g. open issues on `enable_thinking`); live gates catch breakage on pin bump.  
- **Apply-template vs embedded template:** if server uses bundled vs custom template, Gate A expected string may differ; spec appendix must state which template path the gate assumes (default GGUF embedded template vs `--chat-template-file`).  
- **GPU availability in CI:** argv tests always run; live script is required for human/release verification when CI lacks GPU.

## 8. Self-review (spec quality)

- **Placeholders:** None intentional; numeric **b_min** is deferred to implementation (record actual **b** from chosen image).  
- **Consistency:** v1 launch-only; v2 per-request documented separately.  
- **Scope:** Single service rebuild + tests + scripts; gateway v2 is follow-on.  
- **Ambiguity:** Golden completion path is allowed only with checked-in baseline and pin recorded alongside it.

---

**Next step after you approve this file:** invoke **writing-plans** to produce an implementation plan with file-level tasks and verification commands.

---

## Appendix (implementation, 2026-05-10)

- Default CUDA pin raised to **`ghcr.io/ggml-org/llama.cpp:server-cuda-b8740`** (`Dockerfile` / compose / `.env_example`).
- **`orion-llm-gateway`** forwards **`options.chat_template_kwargs`** on `ChatRequestPayload` to **`llamacpp`** and **`llama-cola`** `/v1/chat/completions` payloads for **per-request** thinking control (no host restart).
- **Atlas live check:** `services/orion-llamacpp-host/scripts/verify_atlas_quick_llamacpp_thinking_off.sh` with `ATLAS_LLAMACPP_QUICK_URL` + `ATLAS_QUICK_CHAT_MODEL` hits the quick worker’s HTTP API directly.
