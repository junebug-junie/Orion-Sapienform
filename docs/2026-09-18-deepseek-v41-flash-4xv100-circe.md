# DeepSeek-V4.1-Flash on 4× Tesla V100 32GB

**Live local run — 2026-09-18.**
**Status:** loaded, served tokens, bottleneck measured. Not a production recipe.

This is a hardware + llama.cpp field note for people trying V4.1-Flash on Volta, not a quality eval. Numbers below come from one host (`circe`), one binary, one GGUF pack, and a handful of completions. Where we did not measure something, it is marked **unverified**.

---

## Headline

It runs. Upstream llama.cpp does not.

- **Load from NVMe:** ~6.5 minutes to `/health`
- **Decode:** **2.24 → 3.18 → 3.36 tok/s** at `--moe-stream-cache` 18 / 48 / 64
- **Prefill:** 1.31 → 2.07 tok/s (10-token then 14-token prompts)
- **Why the GPUs look idle:** ~**79% of decode time** was MoE expert *load stall*, not matmul. Cards sat at ~45–60 W (PCIe) / ~115 W (SXM2) with SM util often under 50%, often near 0% on two of the four.

Official card (`temperature=1.0`, `top_p=0.95`, 1M context, `max_tokens ≥ 256K`) is a *serving* target for DeepSeek’s own stack. This box served **8K context**, **reasoning off**, and cannot spend 32 hours emitting 256K tokens at 2 tok/s.

---

## Hardware

| Piece | What we actually had |
| --- | --- |
| Board | Gigabyte G481-HA0-class dual-socket (AMI MegaRAC BMC) |
| CPU | 2× Intel Xeon Gold 6150 @ 2.70 GHz (18c/36t per socket) |
| RAM | 251 GiB. After evicting other GPU services, ~220 GiB available before load. During serve: 96 GiB **page-locked** for MoE L2, process RSS ~110–115 GiB, **5 GiB swap in use** |
| GPUs used | 0: V100-PCIE-32GB · 1: V100-SXM2-32GB · 2: V100-class 32GB (PG500-216) · 3: V100-PCIE-32GB |
| GPUs not used | 4–5: V100-PCIE-16GB (left running other work) |
| Driver | 580.173.02 |
| Weights disk | Lexar SSD NM610 PRO 2TB (DRAM-less), PCIe **Gen3 x4** (8.0 GT/s × 4), ext4 `noatime,discard` |
| What failed | First load from a 2.7T Hitachi 7.2K SAS HDD. Process went disk-wait (`D`), ~159 GiB RSS, GPUs idle, never reached `/health` before a reboot for the NVMe install |

The Lexar is not a 990 Pro. Sequential copy Hitachi→Lexar held ~170 MB/s (HDD-limited). During decode, NVMe util was low (~8% in one `iostat` window). The crawl is **not** “slow SSD saturating.”

---

## Software

| Piece | Pin |
| --- | --- |
| Model | DeepSeek-V4.1-Flash (552B backbone + Engram; HF reports ~763B-class param count). Server meta: `n_params=754,638,981,608`, `ftype=MXFP4 MoE`, `n_vocab=129280`, `n_embd=5120`, `n_ctx_train=1,048,576` |
| GGUF | `JigSawPT/DeepSeek-V4.1-Flash-GGUF` · `DeepSeek-V4.1-Flash-MXFP4-engram-00001-of-00011.gguf` (11 shards) |
| On-disk bytes | **501,809,988,803** (~468 GiB `du`; ~502 GB advertised) |
| Engine | [JigSawPT/llama.cpp](https://github.com/JigSawPT/llama.cpp) branch **`dsv41-porte`** @ **`3b6fcfe`** (“48 engram rows per token, not 56”) |
| Why a fork | Circe’s stock image `ghcr.io/ggml-org/llama.cpp:server-cuda-b10398` has no V4.1 graph and no `--moe-stream`. It cannot load this checkpoint. |
| Build | CUDA **12.8.1** devel container, `CMAKE_CUDA_ARCHITECTURES=70`. Host toolkit is 13.2, which **dropped Volta** — do not compile sm_70 with 13.x |
| Linux patch | Fork is marked Linux-untested. Compile died at ~47% in `llama-moe-stream.cpp`: `std::isfinite` without `#include <cmath>`. One-line include fix, then it linked. |
| Runtime | `nvidia/cuda:12.8.1-devel-ubuntu24.04`, binary + `libllama-server-impl.so` bind-mounted, `--gpus device=0,1,2,3`, host port **8099** |

Community Q2_K (`vcruz305`, ~246 GiB) was **not** run. Its Engram layout is not interchangeable with this converter. Do not mix packs.

---

## Serve recipe (what actually listened)

```text
llama-server \
  -m DeepSeek-V4.1-Flash-MXFP4-engram-00001-of-00011.gguf \
  --n-gpu-layers 99 \
  --ctx-size 8192 \
  --moe-stream \
  --moe-stream-cache 48 \   # first soak was 18; 48 is the measured faster pin
  --moe-stream-l2 96 \
  --moe-stream-io-threads 4 \
  --split-mode layer \
  --tensor-split 1,1,1,1 \
  --flash-attn off \
  --reasoning off \
  --parallel 1
```

`--moe-stream-cache` is GiB of **VRAM expert cache**. `18` is the fork’s single-5090 minimum. First soak used 18; a second boot used **48**. After load, VRAM went from ~8.6–10.4 GiB/card to ~16.6–17.3 GiB/card. Still headroom on 32 GiB.

`--moe-stream-l2 96` allocated **96.00 GiB PINNED** host RAM (16,426 slots × 6,275,072 bytes, CLOCK eviction). That memory is unavailable to the rest of the system. Combined with ~110 GiB RSS this host went into **5 GiB swap**. Raising L2 further on 256 GiB-class boxes is likely a footgun until RAM grows.

`--flash-attn off` is not optional on these Voltas. Same host has previously seen ~20× token-generation collapse with flash-attn on V100. We did not re-enable it.

Vision and tools: weights are multimodal; this run was **text-only**. `supports_vision` was left off.

---

## Measured completions

All via `POST /v1/chat/completions` on `127.0.0.1:8099`. `temperature=0` unless noted. Server default sampling in our profile file is `temperature=1.0, top_p=0.95` (card-aligned) but the timed curls used greedy.

### Cold (first request after NVMe load)

| | |
| --- | --- |
| Prompt / gen | 10 / 8 tokens |
| Prefill | 7624 ms · **1.31 tok/s** |
| Decode | 4588 ms · **1.74 tok/s** |
| GPU expert cache | hits 2229 / misses 1369 · **61.95%** hit · 1873 cold |
| Load stall | 7685 ms total · 19.2 ms / remap |
| L2 slabs | **6.61%** hit (cold fill; 0 evictions) |

Reply text: `Hello there, how are you?`

### Warm-ish (user prompt, `max_tokens=5000`)

Prompt: *“Tell me the weirdest thing you ever heard.”*
Stopped on EOS, not the cap.

| | |
| --- | --- |
| Prompt / gen | 14 / **615** tokens |
| Prefill | 6751 ms · **2.07 tok/s** |
| Decode | 274288 ms · **2.24 tok/s** (446 ms/token) · 3s window 2.0–2.8 tok/s |
| GPU expert cache | hits 89556 / misses 63410 · **58.55%** hit · 9687 cold |
| Remaps | 25040 |
| Load stall | **216744 ms (79.0% of decode)** · 8.656 ms / remap |
| L2 slabs | hits 149389 / fills 45186 / evictions 28760 · **76.78%** hit |
| Graph reuse | **0** (`graphs reused = 0`) |

At 2.24 tok/s, a card-faithful **256K** completion is ~32 hours. A **1M** prefill is not a chat turn.

---

## Bottleneck (this is the result)

The cards are waiting for experts.

```text
decode wall time     274.3 s
load stall           216.7 s   ← 79%
everything else       57.6 s   ← compute + overhead
```

During that 615-token run:

- GPU 0/2 flickered 26–48% SM; GPU 1/3 often **0%** in the same `nvidia-smi` snapshot
- PCIe V100s ~45–49 W of 250 W; SXM2 ~116 W of 300 W
- Host CPU ~98% idle, iowait ~0
- Lexar not saturated

So: **not** thermal, **not** PCIe power limit, **not** “NVMe too slow to read sequential GGUF,” **not** a disabled lane. `--moe-stream` is doing what it says — keep a small hot set, fetch the rest. With `cache=18` the GPU expert hit rate stayed ~59–62%. Misses cost ~9–19 ms each and dominate tok/s.

Raising cache **18 → 48 → 64** (same prompt). 18 and 48 were `temperature=0` after a 8-token warmup. 64 was the operator curl: `temperature=1, top_p=0.95`, first request after load (colder).

| | cache 18 | cache 48 | cache 64 |
| --- | ---: | ---: | ---: |
| Decode | 2.24 tok/s (615) | 3.18 tok/s (619) | **3.36 tok/s** (543) |
| Prefill | 2.07 tok/s | 2.79 tok/s | 1.58 tok/s (cold) |
| GPU expert hit | 58.55% | 78.78% | **83.79%** |
| Load stall | 216.7 s (79%) | 145.3 s (75%) | 109.0 s (67%) |
| Stall / remap | 8.66 ms | 5.77 ms | 5.00 ms |
| VRAM / card | 8.6–10.4 GiB | 16.6–17.3 GiB | 20.8–21.6 GiB |
| L2 slab hit | 76.8% | 59.1% | 46.8% |
| Wall | ~281 s / 615 tok | ~200 s / 619 tok | ~170 s / 543 tok |

18→48 was the real jump (+42% decode). 48→64 is diminishing returns (+6%). Still stall-bound. Cards at 64 have ~10–11 GiB left — enough for a modest ctx bump, not a second 64 GiB of cache.

---

## Official card vs this run

| Card | This run |
| --- | --- |
| `temperature=1.0` | Timed curls used `0`. Use 1.0 in the request body if you want card behavior. |
| `top_p=0.95 or 1.0` | Not set on the timed curls. |
| `context_window=1M` | Weights: `n_ctx_train=1048576`. **Served `n_ctx=8192`.** KV is small (~890 B/token, sparse attention) so memory is not the 1M blocker — wall time and graph/alloc are. |
| `max_tokens ≥ 256K` | For `reasoning_effort=100`. We ran **`--reasoning off`**. 256K output at 2.2 tok/s is a day-long job. |
| Reasoning 1–100 | Off. Not measured. |

---

## Reproduce

```bash
# health
curl -sS http://127.0.0.1:8099/health

# card-like sampling, short cap (do not send 256K)
curl -sS http://127.0.0.1:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dsv41",
    "messages": [{"role":"user","content":"Tell me the weirdest thing you ever heard."}],
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 256
  }'
```

Watch the server log for `print_stats` / `load stall` / hit rate after each request. Those three lines are the report.

---

## What we would tell someone else on Volta

1. **You need `dsv41-porte` + the JigSawPT MXFP4+Engram pack.** Stock llama.cpp and the Q2 community pack are different animals.
2. **Compile with CUDA 12.8 / sm_70.** CUDA 13 dropped Volta. Add `#include <cmath>` if `isfinite` fails.
3. **Do not mmap this GGUF from a spinning disk** and expect `/health`. We didn’t get it. NVMe made load ~6.5 min.
4. **`--flash-attn off` on V100.**
5. **`--moe-stream-cache` is the knob.** 18 → 48 → 64 moved decode **2.24 → 3.18 → 3.36 tok/s** and GPU expert hit **59% → 79% → 84%**. 64 is diminishing returns. Still ~67% stall. Do not blame the SSD first.
6. **96 GiB pinned L2 on a 256 GiB host is tight.** We swapped. More L2 without more RAM can make this worse.
7. **4-way `--split-mode layer` looks quiet on `nvidia-smi`.** One or two cards work; the others wait. That is on top of the expert-fetch stall.
8. This is a **proof of load + tokens**, not a claim that 4×V100 is a good V4.1-Flash server.

---

## Open / not done

- Cache above 64 (little left; 48→64 was only +6% decode)
- Modest `--ctx-size` bump (16K/32K) now that cache 64 left ~10 GiB/card
- `temperature=1.0` quality pass
- Reasoning effort on
- Context above 8K
- Q2_K pack
- Image inputs
- Graph reuse (`graphs reused = 0` on the 615-token call — may be a fork issue; **unverified** cause)
- Cold vs warm tok/s after a cache bump

---

*Host: Circe · 2026-09-18 · one evening, one checkpoint, one fork pin.*
