# Goldens and forbidden markers

This directory holds small, checked-in artifacts used by `verify_qwen3_thinking_off_live.sh`.

## `forbidden_thinking_markers.txt`

Case-sensitive substrings that must **not** appear in the assistant `content` when the server is started with thinking disabled (`--reasoning-budget 0`, `--chat-template-kwargs '{"enable_thinking":false}'`, and matching per-request `chat_template_kwargs` where applicable). The list is intentionally short and conservative: it targets delimiter strings that commonly appear when a Qwen3-class model emits a **thinking** trace (see upstream discussion around [PR #13196](https://github.com/ggml-org/llama.cpp/pull/13196) and Qwen3 templates), not vague English substrings (avoid bare `think`, which matches normal prose).

If you change any of the following, re-run the live script on a GPU host and **review** whether this file still matches real failure modes (or causes false positives):

- **`LLAMACPP_IMAGE_TAG` / rebuilt `orion-llamacpp-host` image** — template or tokenizer behavior can shift delimiters.
- **Reference Qwen3 GGUF** — different quant or checkpoint may change how thinking leaks into `content`.

## How to refresh or extend markers

1. Run the live verifier (from repo root or any cwd; use absolute `MODEL_QWEN3_PATH`):

   ```bash
   export MODEL_QWEN3_PATH=/absolute/path/to/your/Qwen3.gguf
   export IMAGE=orion-llamacpp-host:0.1.0   # optional
   bash services/orion-llamacpp-host/scripts/verify_qwen3_thinking_off_live.sh
   ```

2. If Gate B fails because a **new** thinking delimiter appears in `content`, add a **single literal substring** per line to `forbidden_thinking_markers.txt` (document the source: upstream issue, PR, or a one-line paste from your capture). Prefer tags or reserved tokens over vague words.

3. If a marker causes **false positives** on your pin/GGUF (legitimate text contains the substring), remove or narrow that line and re-run until the script passes.

4. Commit updated `forbidden_thinking_markers.txt` together with any README tweaks.

Optional future golden files (e.g. `qwen3-*_thinking_off_completion.json`) can live here following the same refresh discipline.
