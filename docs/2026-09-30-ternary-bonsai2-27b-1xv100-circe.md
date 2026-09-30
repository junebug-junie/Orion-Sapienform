# Ternary Bonsai 2 27B on one Tesla V100 32GB

**Live local run, 2026-09-30.**
**Status:** loaded, served four concurrent runs, measured against the 35B chat model on the same card. This is not a production recipe.

This is a hardware and llama.cpp field note for Volta, not a full quality eval. The numbers come from one host (`circe`), one card (gpu0), one binary and a small set of prompts. Anything we did not measure is marked **unverified**.

---

## Headline

It runs on Volta. Stock llama.cpp cannot load it; Prism's fork can.

- **Fits on one 32 GB card** with four 65K-token slots: **24.1–24.3 GB** with flash attention on.
- **Decode:** 51 tok/s for one short run. Four runs at once get **27 tok/s each, ~99 tok/s total**.
- **Context depth costs speed.** At 61K tokens, decode is **32 tok/s** with flash attention on and **17.6** with it off. The 35B chat model holds **~70 tok/s** at every depth on the same card.
- **Prompt caching mostly survives four slots, but it misses the model's own replies.** In four concurrent agent-style loops, **73%** of prompt tokens came from cache. Of the tokens that were re-processed, **23.5%** were the model's own previous reply being read again: the template renders an assistant turn differently when it comes back as history. This is Prism's open tool-loop issue, and it does happen here.
- **Recall held to the slot limit.** A fact planted at turn 0 and a fresh fact at each turn were both recalled correctly at every depth up to 61K. This is an easy exact-string test; see the caveats.
- **The "flash attention off on Volta" rule is wrong for this model.** Turning it on wins at depth and saves ~4 GB. The exception: exactly two concurrent runs get slower.

---

## Hardware and software

| Piece | Value |
| --- | --- |
| GPU | gpu0: Tesla V100-PCIE-32GB (sm_70). Chat worker stopped for the test |
| Driver | 580.173.02 |
| Model | `prism-ml/Ternary-Bonsai-2-27B-gguf` · `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7.21 GB; ternary {−1,0,+1} weights on Qwen3.8-27B) |
| Engine | [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) branch `prism` @ `88c4bc6`, reports `b10750` |
| Build | CUDA **12.8.1** devel container, `CMAKE_CUDA_ARCHITECTURES=70`, `GGML_CUDA_FA_ALL_QUANTS=ON`. Host nvcc 13.x dropped Volta, and Prism says 13.3 builds segfault |
| Image | `services/orion-llamacpp-bonsai-host` → `llamacpp-bonsai-prism:server-local-volta`, port **8017** |
| Comparison | chat worker, `Qwen3.6-35B-A3B-UD-Q5_K_M.gguf` (MoE, ~3B active), stock `b10398`, same gpu0, 1 slot × 65,536 |

Build traps we hit, all fixed in the repo:

1. **A shallow clone reports build 1.** llama.cpp's build number is `git rev-list --count`. The Orion wrapper then treats the binary as pre-b5332 and silently drops `--flash-attn` and `--reasoning`. Use a blobless clone.
2. **The fork prints `version: 0.2.0-dev (build 10750, …)`.** The wrapper read the leading `0` as the build number. Its parser now prefers `(build N)`.
3. **The CUDA stubs ship `libcuda.so` without `.so.1`**, so the link fails. Add a symlink in the build stage.
4. **The base wrapper image bakes in `config/`.** An image layered on an old base did not know the new profile (`KeyError`). The Bonsai image now copies the wrapper, config and `orion/` itself.

Load time: **3.7 s** from the NVMe cache once the file was downloaded. The first boot included the 7.2 GB download, and `/health` was up within ~2 minutes of `up`.

---

## Serve recipe (what listened)

```text
llama-server -m Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --ctx-size 262144 --parallel 4 \          # 65,536 per slot
  --n-gpu-layers 99 --threads 16 --batch-size 1024 \
  --reasoning auto --jinja --reasoning-format deepseek \
  --chat-template-kwargs '{"reasoning_effort":"medium","preserve_thinking":false}' \
  --flash-attn on \                          # the soak ran "off" first; see below
  --no-context-shift --n-predict 16384
```

Template facts, measured against this GGUF:

- `reasoning_effort` accepts **`xhigh`, `medium`, `low`**. **`"none"` returns HTTP 500** (`Unexpected reasoning effort none`), even though Prism's KNOWN_ISSUES says it turns reasoning off.
- **`chat_template_kwargs: {"enable_thinking": false}`** turns thinking off: a 7-token answer with no reasoning.
- **`reasoning_budget: 0` does not** stop thinking; it still produced 578 characters of reasoning.

---

## Measured completions

All via `POST /v1/chat/completions`, reading llama.cpp's own `timings`, with `temperature=0.6, top_p=0.95` unless noted. The single-run and quality results below were taken with flash attention **off**.

| Prompt | Prompt / gen tokens | Prefill | Decode | Finish |
| --- | ---: | ---: | ---: | --- |
| "Say hello in five words." (thinking off) | 4 / 7 | — | 46.4 tok/s | stop |
| "Tell me the weirdest thing you ever heard." (medium) | 20 / 1953 | 67 tok/s | **51.0 tok/s** | stop |
| ~31.6K-token doc + question (thinking off) | 31,573 / 21 | **535 tok/s** | 27.2 tok/s | stop |
| same, repeated | 4 new (31,569 cached) | — | 28.0 tok/s | stop, 1.1 s wall |

### Quality spot checks (medium effort, flash attention off)

| Check | Result |
| --- | --- |
| Logic (height ordering) | **Correct** ("Bob"), 350 reasoning chars |
| Code (`merge_intervals`) | **Correct**, idiomatic |
| Tool call (`read_file`) | **Correct**: right function, right path, `finish=tool_calls` |
| Math (sum of n<100 where n²+n+41 is not prime) | **Failed: hit the 8,192-token cap still thinking** (11,420 reasoning chars, no answer). This is Prism's documented "thinks forever" issue |
| "Weirdest thing you ever heard" | Fluent, but the "dancing mushroom that spreads spores" story looks **fabricated**. Unverified either way |
| Ambiguous "metric returns to rest" | Read it as a spacetime metric. Coherent, but not our sense of the word; the prompt had no Orion context |

That is too small a sample to rank quality. It says: fluent, tool calls work, and it can burn its whole output budget thinking.

---

## Concurrency (thinking off, 512 forced tokens, `ignore_eos`)

| Runs at once | FA off, per run | FA off, total | FA on, per run | FA on, total |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 51.0 | 51 | 51.7 | 52 |
| 2 | **41.1–41.7** | ~82 | **23.4** | ~47 |
| 4 | 25.4–25.8 | ~102 | 27.0 | ~99 |

The two-runs-with-flash-attention-on dip reproduced on two passes. It looks like a slow Volta kernel path for a batch of two. Three concurrent runs were **not measured**.

The 35B chat worker has one slot, so it cannot serve concurrent runs at all; requests queue.

---

## Context depth (the degradation question)

One conversation grows by ~5.5K tokens a turn until it hits the 65,536-token slot. Each turn asks for a fact planted at turn 0 (far) and one planted in the newest chunk (near). Thinking was off; a medium-effort pass matched within ~1.7 tok/s and recalled identically.

| Depth (tokens) | Bonsai FA off: decode / prefill | Bonsai FA on: decode / prefill | 35B chat: decode / prefill | Recall (all three) |
| ---: | ---: | ---: | ---: | --- |
| 14,321 | 39.2 / 602 | 43.8 / 662 | 74.2 / 796 | far ✓ near ✓ |
| 20,555 | 33.3 / 500 | 41.5 / 577 | 74.7 / 746 | ✓ ✓ |
| 26,509 | 29.8 / 439 | 39.9 / 540 | 74.2 / 718 | ✓ ✓ |
| 31,871 | 27.6 / 381 | 38.5 / 500 | 73.5 / 690 | ✓ ✓ |
| 37,819 | 24.5 / 347 | 37.6 / 474 | 72.1 / 670 | ✓ ✓ |
| 43,226 | 22.9 / 318 | 36.4 / 439 | 72.0 / 631 | ✓ ✓ |
| 49,174 | 21.0 / 296 | 34.8 / 423 | 70.8 / 626 | ✓ ✓ |
| 55,324 | 19.8 / 273 | 33.5 / 404 | 69.0 / 602 | ✓ ✓ |
| 60,695 | **17.6** / 247 | **32.3** / 375 | **69.8** / 578 | ✓ ✓ |

Bonsai loses speed with depth: **−55%** decode from 14K to 61K with flash attention off, and **−26%** with it on. The 35B barely moves (−6%). The 35B is a sparse MoE with ~3B active parameters. Bonsai runs all 27B every token, and its attention layers get more expensive as the cache grows. **Recall did not degrade** in any run.

---

## Four concurrent agent-style loops (prompt-cache reuse)

Four conversations ran at once, 6 steps each. Every step re-sent the whole history plus a ~1–1.7K-token "tool result", like a curiosity run. This was measured with flash attention off.

- The history up to the end of the previous *prompt* was reused every step. The four slots did not evict each other.
- **The previous assistant reply was never reused.** Each step's `cache_n` is the previous prompt total minus 4, so the prior reply (up to 512 tokens) was re-processed. This happened even with thinking off and only `content` sent back, so it is template re-rendering, not a harness artifact.
- Totals: **36,597 prompt tokens processed, 99,138 reused (73%)**. **8,607 (23.5%)** of the processed tokens were the model's own prior replies; the rest was new tool-result text. Wall time: 211 s.
- Prefill on processed tokens ran at 350–490 tok/s per slot, and decode at 13.7–29.5 tok/s per slot, while four ran together.

Cross-slot eviction, the risk Cursor raised, did not happen. The known issue (Bonsai-demo #183, "tool calls re-rendered differently from how they were generated") did. `preserve_thinking: false` does not prevent it. A long curiosity run pays this re-read on every step: the cost grows with reply length, not with context length.

---

## GPU memory and power

| | FA off | FA on |
| --- | ---: | ---: |
| VRAM after load | 27,170 MiB | 24,098 MiB |
| VRAM peak under load | 28,612 MiB | 24,296 MiB |
| Max power / util | 279.6 W / 100% | not sampled |

The FA-off figures come from 1,639 one-second samples (`vram.csv`). The **FA-on figures are two hand readings of `nvidia-smi`**, one after load and one after the depth sweep. They were not continuously sampled, so the FA-on peak could be higher. The KV cache for all four 65K slots is allocated at boot, so memory barely moves with load.

---

## What we did not measure

- Real curiosity replays. The loops above are synthetic stand-ins.
- Three concurrent runs, and whether the two-run dip also hits 3.
- Quality at depth beyond exact-string recall.
- A like-for-like quality comparison against the 35B or the Q4 27B.
- The 35B was the live chat worker and **did serve Orion traffic during its run**. Seven of its ten depth turns waited 55–66 s wall against ~8–9 s of compute, which is queueing behind real requests on a 1-slot worker. Per-request tok/s comes from llama.cpp's own timings and is unaffected, but its wall times are not comparable.

---

## Next

- Profile default is now `flash_attn: on`.
- If curiosity would typically have exactly two runs active, measure 2 and 3 concurrent runs before relying on four slots.
- Tie the math "thinks forever" failure to an output budget or a `reasoning_budget` in the curiosity caller before trusting it for real runs.

Raw data and the scripts that produced it: `docs/bench/2026-09-30-bonsai2-circe/`.
