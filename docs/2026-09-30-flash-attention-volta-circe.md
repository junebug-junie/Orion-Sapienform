# Flash attention on Volta, and three 27B-class models head to head

**Live run on circe gpu0 (Tesla V100-PCIE-32GB), 2026-09-30.**
**Status:** measured. It overturns the repo's "flash attention off on Volta" rule for every model we re-benched.

We ran three models on the same card with the same harness and one slot of 65,536 tokens each. Each ran with `--flash-attn off` and `--flash-attn on` explicitly:

| Label | Model | Bits | Engine |
| --- | --- | --- | --- |
| Bonsai | Ternary-Bonsai-2-27B PQ2_0 (Qwen3.8-27B base) | ~1.6 | PrismML fork b10750 |
| Q4 27B | Qwen3.8-27B-UD-Q4_K_XL (the live agent model) | ~4.5 | stock b10398 (`orion-llamacpp-host:0.1.0`) |
| 35B | Qwen3.6-35B-A3B-UD-Q5_K_M (the live chat model, MoE ~3B active) | ~5.5 | stock b10398 |

Raw data and scripts: `docs/bench/2026-09-30-bonsai2-circe/`.

---

## Headline

1. **Flash attention on is never slower, and about 2x faster deep in context, for all three.** At 61K tokens of context, decode speed:
   - Bonsai: 18.1 off → 32.4 on
   - Q4 27B: 12.8 → 25.4
   - 35B: 45.0 → 84.7

   Short-prompt speed is identical either way, and memory drops 2–3 GB with it on.
2. **The live lanes were already running it on, by accident.** The wrapper parsed b10398's `version: 0.1.0-dev (build 10398, …)` as build 0 and treated the binary as pre-b5332. It then silently dropped `--reasoning` and any `--flash-attn` value other than "on". So llama.cpp ran its default `auto`, and for the Q4 27B that logs `flash_attn = auto` → `Flash Attention enabled`, confirmed with `-lv 4`. The parser fix (PR #2420) would have made "off" take effect at the next rebuild, halving deep-context speed. Profiles now say "on".
3. **Ternary did not visibly lose reasoning at depth.** On a hard multi-hop test to 50K tokens, Q4 27B scored 20/20, Bonsai 19/20 and 35B 19/20. Bonsai was about 1.5x faster than Q4 27B at every depth and used about 9 GB less memory.

---

## Speed (1 slot, 65,536 ctx, thinking off, tok/s)

Short prompt, 512 forced tokens:

| | FA off | FA on |
| --- | ---: | ---: |
| Bonsai | 51.2 | 51.7 |
| Q4 27B | 33.0 | 33.2 |
| 35B | 97.1 | 96.9 |

Decode and prefill as one conversation grows (about 5.5K new tokens per turn):

| Depth | Bonsai off | Bonsai on | Q4 27B off | Q4 27B on | 35B off | 35B on |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14K decode / prefill | 38.4 / 609 | 42.6 / 661 | 28.9 / 596 | 31.7 / 684 | 84.2 / 595 | 93.2 / 613 |
| 32K | 27.8 / 386 | 39.1 / 506 | 22.0 / 394 | 29.2 / 517 | 62.1 / 491 | 91.4 / 554 |
| 49K | 21.7 / 296 | 34.8 / 426 | 13.5 / 303 | 27.0 / 433 | 50.4 / 426 | 86.5 / 509 |
| 61K | 18.1 / 248 | **32.4** / 380 | 12.8 / 252 | **25.4** / 385 | 45.0 / 390 | **84.7** / 479 |

A planted fact from turn 0 and one from the newest turn were recalled correctly at every depth in all six runs.

GPU memory with 1 slot × 65K, after load → after the run:

| | FA off | FA on |
| --- | ---: | ---: |
| Bonsai | 14.4 → 15.9 GB | 11.4 → 11.6 GB |
| Q4 27B | 23.6 → 25.0 GB | 20.5 → 20.7 GB |
| 35B | 28.6 → 29.6 GB | 26.5 → 26.6 GB |

The live chat worker in auto mode measured about 70 tok/s at 61K in an earlier run, against 84.7 here. It also loads a vision projector and uses different batch and thread settings. We did not isolate which of those costs the ~15 tok/s.

---

## Reasoning at depth (flash attention on, 1 slot)

Facts were planted at fixed fractions of a filler context built from repo docs. Values are fresh and random per depth, and answers are scored exactly. Qwen3.8 models ran with `reasoning_effort: medium`; the 35B ran with `enable_thinking: true` at its own default.

**Basic** (facts at 10/50/90%: a 3-hop chain, arithmetic across two distant facts, a later correction overriding an earlier fact; depths 8K/24K/40K/50K):

| | Score | Mean generated tokens |
| --- | ---: | ---: |
| Bonsai | 12/12 | 268 |
| Q4 27B | 12/12 | 263 |
| 35B | 12/12 | 963 |

**Hard** (three look-alike vaults, cities and keepers; a keeper replaced at 75%; a 4-hop chain to the *current* keeper; a decoy vault; a pounds-to-kg sum across three shipments; two stacked date shifts; "who was keeper before"):

| | Score | Miss | Mean gen | Mean wall |
| --- | ---: | --- | ---: | ---: |
| Q4 27B | **20/20** | — | 328 | 24.8 s |
| Bonsai | 19/20 | date shift at ~49K (said Monday, want Tuesday; only 180 tokens of thought) | 336 | 20.0 s |
| 35B | 19/20 | date shift at ~40K (said Tuesday, want Sunday) | 1,261 | 28.4 s |

With 20 items each, this shows no visible reasoning loss from ternary weights at depth. It cannot rank the three. Both misses were the same type: a date-arithmetic slip, not a retrieval failure.

---

## Four concurrent sessions: Bonsai vs Q4 27B (4 slots × 32K, flash attention on)

The Q4 27B cannot fit 4 × 65K on one card: about 16.5 GB base plus about 2.1 GB per 32K of session. So both models ran at 4 × 32K.

| Runs at once | Bonsai per run (total) | Q4 27B per run (total) |
| ---: | ---: | ---: |
| 1 | 51.7 | 33.0 |
| 2 | 42.8 (~80) | 26.4 (~51) |
| 3 | 33.7 (~91) | 22.1 (~62) |
| 4 | 26.8 (~97) | 18.0 (~67) |

These are two passes each, 512 forced tokens with thinking off. The first Q4 pass includes a cold warm-up, so its totals are not shown. The two-runs dip seen at 4 × 65K (23 tok/s each) did not appear at 32K slots.

| | Bonsai | Q4 27B |
| --- | ---: | ---: |
| VRAM, 4 × 32K loaded | 15.9 GB | 25.0 GB |
| 4 concurrent agent loops (6 steps), wall | 187 s | 231 s |
| Prompt-cache reuse in those loops | 79.7% | 79.4% |

**The prompt cache misses the prior assistant reply on both models.** This is the Qwen3.8 chat template, not Bonsai, so the live agent lane pays it too.

In four concurrent depth sweeps to about 26K, recall held for every session on both models. The decode figures in those rows (~1 tok/s) are not meaningful: each answer was about 15 tokens, generated while three other slots were prefilling 6K-token prompts.

### Sessions per 32 GB card at today's agent context (131K)

The live agent lane's own logs (1,485 requests since 2026-09-24) show prompts of: median 24K, p90 60.5K, p95 68K, max 82K. **6.8% exceed 65K**, so 65K slots would cut off the deep end of long runs, and 131K is the safe size.

| Model | Sessions × 131K per card | VRAM |
| --- | ---: | ---: |
| Q4 27B (live) | 1 | 24.9 GB (live gpu1) |
| Bonsai | **2** | ~24.1 GB (same total cache as the measured 4 × 65K) |

Two options are untested: a `q8_0` KV cache (about 3 Bonsai × 131K), and `--kv-unified` (4 sessions sharing one 262K pool).

## Where the "off on Volta" rule came from

Every "off" in `config/llm_profiles.yaml` traced back to one measurement: a 2026-07-09 llama-optimus tuning run of `qwen3-coder-next-q5km-2xv100-32gb-agent-depth`. That run was split across two GPUs on an older llama.cpp build, and found flash attention on "collapses TG ~20x". Other profiles copied the rule without their own bench, and the DeepSeek and BF16 field notes repeated it.

On b10398, single card, it does not hold for the Q4 27B or the 35B, and it does not hold for the Bonsai fork either.

## What changed in config

- Every stock-image profile is now `flash_attn: "on"`. For the live lanes (agent, chat, fast, metacog) this matches what was actually running, so there is no behaviour change today. The difference is that the next wrapper rebuild will not flip them to "off".
- Two profiles are kept at their own measured "off": `qwen3-coder-next-q5km-2xv100-32gb-agent-depth` (the source bench) and `deepseek-v41-flash-…` (a different fork, from its own soak).

## Not measured

- Multi-GPU split (`split_mode: layer/row`) with flash attention on, on this build.
- The 8B fast and metacog lanes: they are "on" by inference from the same build, not probed.
- What auto chose for the 35B: the direct probe failed to load on gpu2. Its live speed is consistent with on.
