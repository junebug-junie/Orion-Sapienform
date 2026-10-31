# Strata: Qwen3.8-Flash-Next on one V100 (hecate, 2026-10-30)

One V100 served Qwen3.8-Flash-Next and passed every reasoning check. Its output was the fastest of any model we have run on one card.
In September, llama.cpp needed all four of circe's V100s to hold this model family (UD-IQ4_XS, `config/llm_profiles.yaml`).

Strata ([github.com/Niko1221/Strata](https://github.com/Niko1221/Strata), MIT) keeps the most-used experts of a mixture-of-experts model on the GPU.
It keeps the rest in system RAM and computes them on the CPU. Source: Reddit r/v100 "V100 with Strata" and the
[PC Watch write-up](https://pc.watch.impress.co.jp/docs/column/nishikawa/2147537.html).

## Setup

- Host: hecate, one V100-SXM2-32GB on NUMA node 1, 2x Xeon Gold 6148, 183 GB RAM, driver 580.178.04.
- Image: `ghcr.io/niko1221/strata:0.2.7-cuda12` (engine 0.1.52). No host install, no sudo.
- Model: ISTA-DASLab Qwen3.8-Flash-Next IQ3_S (`FAMILY=qwen MODEL=IQ3_S`), 54.8 GB of weights. Context 128K, 1 request at a time, int8 KV.
- Model files: `/mnt/models/strata` on the NVMe. Do not use `/mnt/telemetry`: it is a spinning disk.
- Pinned to the GPU's socket: `--cpuset-cpus 20-39,60-79 --cpuset-mems 1`.
- The VRAM was full (32,051 MiB). hecate's `agent-deep` vLLM container was stopped for the run, then restarted and smoke-tested.
- Harness: the 2026-09-30 circe kit (`docs/bench/2026-09-30-bonsai2-circe/`), with the same prompts and seeds. `run_hecate.sh` runs it.
  Strata returns no llama.cpp `timings`, so the kit streams the reply. Output speed is measured after the first token.
- One difference from September: the old kit nested the reasoning setting
  (`chat_template_kwargs={"reasoning_effort": {"reasoning_effort": "medium"}}`), a malformed value.
  This run sends `reasoning_effort: "medium"` correctly.
- The committed `out-pin-IQ3_S` JSONL predates a fix: its `prompt_per_second` divides the whole prompt,
  cached part included, by the first-token time. The real figure is new tokens (`usage.prompt_tokens_details.cached_tokens`
  subtracted) over `ttft_s`, as in Results below. The kit now records it that way.

## Results

| | Strata Flash-Next IQ3_S | 35B-A3B Q5 | Bonsai 27B | 27B Q4 |
| --- | --- | --- | --- | --- |
| Reasoning at depth (8K-50K) | 12/12 | 12/12 | 12/12 | 12/12 |
| Hard reasoning (8K-50K) | **20/20** | 19/20 | 19/20 | 20/20 |
| Avg output speed, hard set | **94 tok/s** | 89 | 41 | 26 |
| Recall sweep to 75K (far + near) | 22/22 | — | — | — |

The September columns are from circe gpu0 (a V100-PCIE) running llama.cpp with flash-attn on. hecate's card is the SXM2 version,
so part of the speed gap may be the card, not Strata.

- **Prompt reading:** about 960-1,280 tok/s on new tokens. The cold turn at 6K tokens read at 1,221 tok/s. Speed eases slightly with depth.
  Strata reuses the cached conversation, so a turn at 75K context that adds about 6K new tokens returns its first token in about 6 s.
- **Output, short replies (512 tokens):** 62-71 tok/s. Output with reasoning on: 83-106 tok/s.
- **Socket pinning:** pinned averaged 67.3 tok/s and unpinned 65.9 (3 runs each, short replies). That is about 2%, within noise.
  The pin costs nothing, so keep it. The likely reason it barely matters: the 32 GB card holds most of the hot experts.

## What this does not show

- **Concurrency.** Every run here sent one request at a time. Orion's lanes send many at once, and Strata serves 1-2 (`PARALLEL`, max 8).
  This is the open question before Strata could replace a lane.
- **Tool calls and the Orion gateway.** These were not tested. Strata claims an OpenAI- and Anthropic-compatible API (`strata connect`).
- **Context above 128K** and the 262K setting from the Reddit post were not tested.
- Each depth was measured once, with no repeats.

## Files

- `run_hecate.sh`: the run script. Usage: `run_hecate.sh <data_dir> [pin|nopin] [model]`. `SHORT_ONLY=1` runs only the short-reply check.
- `out-pin-IQ3_S/`: full pinned run (depth / reason / hard JSONL).
- `out-nopin-IQ3_S/`: unpinned short-reply A/B.
- The `run.log` files are gitignored. They are on hecate under `/mnt/telemetry/strata-bench/kit/out-*/`.
