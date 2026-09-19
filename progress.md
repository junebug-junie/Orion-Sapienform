# DeepSeek-V4.1-Flash on Circe — experiment progress

Branch / worktree: `feat/deepseek-v41-flash-circe` @ `/mnt/scripts/Orion-Sapienform-deepseek-v41-flash-circe`
Host under test: **Circe** (`circe@circe`)
Started: 2026-09-18 (UTC)

Goal: stand up a `config/llm_profiles.yaml` card for DeepSeek-V4.1-Flash on nvidia-smi GPUs **0,1,2,3**, figure out MoE/Engram offload, download weights, free those GPUs for the soak, and keep a restore checklist.

---

## Current verdict (read this first)

| Claim | Status |
| --- | --- |
| Profile added | **DONE** — `deepseek-v41-flash-mxfp4-engram-4xv100-32gb-circe-test` |
| Host forwards `--moe-stream*` | **DONE** in `orion-llamacpp-host` (skips cleanly if binary lacks flags) |
| Circe GPUs 0–3 freed | **DONE** (chat/agent/diffusion/vision stopped; fast+metacog left on 4/5) |
| Weights downloaded | **DONE** — 11/11 shards, 468G on Lexar at `/mnt/storage-fast/llm-cache/gguf/DeepSeek-V4.1-Flash-MXFP4-engram/` (Hitachi copy deleted) |
| Lexar NVMe | **DONE** — `nvme1` Lexar NM610 PRO 2TB, ext4 `storage-fast`, mounted `/mnt/storage-fast` (UUID `d951cece-4221-4870-8a53-0abc6cffe4f2`) |
| Fork source | **DONE** — `dsv41-porte` @ `3b6fcfe` |
| Volta binary | **DONE** — `llama-server` + `--moe-stream*` on Circe (`build/bin`, sm_70 / CUDA 12.8). Repo recipe: `Dockerfile.dsv41-porte` + `scripts/build-dsv41-volta.sh` |
| Smoke load | **DONE / torn down** — cache-64 last pin; `dsv41-smoke` stopped 2026-09-19 ~00:48 UTC |
| Restore | **agent only** — `orion-circe-atlas-llamacpp-agent` up on `:8015` GPU1 (Qwen3.8 27B). Chat/diffusion/vision still stopped. |
| Community report | **DONE** — `docs/2026-09-18-deepseek-v41-flash-4xv100-circe.md` (also Circe `community-report.md`) |

Upstream Orion llama.cpp **cannot** load V4.1-Flash yet. The published working path is the **dsv41-porte** fork + the **JigSawPT MXFP4+Engram** GGUF (~502 GB), not Circe’s current `ghcr.io/ggml-org/llama.cpp:server-cuda-b10398` image.

Community Q2_K (~246 GiB, `vcruz305`) matches the “192+128 volatile” spreadsheet better, but its Engram layout is **not** interchangeable with the fork converter. Not wired as the primary profile.

---

## Actions taken

### 2026-09-18 — inventory

Circe GPUs before eviction:

| Index | Card | Was used by |
| --- | ---: | --- |
| 0 | V100-PCIE-32GB | `orion-atlas-llamacpp-chat` (`CUDA=0,3`, profile `qwen36-35b-a3b-udq5km-2xv100-32gb-deep-cognition`) |
| 1 | V100-SXM2-32GB | `orion-circe-atlas-llamacpp-agent` (`CUDA=1`, profile `qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex`) |
| 2 | V100-class 32GB (PG500-216) | `orion-circe-diffusion-host` + `orion-circe-circe-vision-host-qwen` |
| 3 | V100-PCIE-32GB | chat (shared with 0) |
| 4 | V100-PCIE-16GB | `orion-atlas-llamacpp-fast` — **left running** |
| 5 | V100-PCIE-16GB | `orion-atlas-llamacpp-metacog` — **left running** |

RAM after eviction: ~**220 GiB available** / 251 GiB total.
Disk: `/mnt/telemetry` only ~79 GiB free (too small for this model). Scratch + weights go on **`/mnt/storage-warm`** (~2.7 TiB free).

Full pre-eviction dump:
`/mnt/storage-warm/deepseek-v41-flash-circe/pre-eviction-snapshot.txt` on Circe.

### 2026-09-18 — stopped for the test

```bash
docker stop \
  orion-atlas-llamacpp-chat \
  orion-circe-atlas-llamacpp-agent \
  orion-circe-diffusion-host \
  orion-circe-circe-vision-host-qwen
```

Left up on purpose: `orion-atlas-llamacpp-fast`, `orion-atlas-llamacpp-metacog` (GPUs 4 and 5).

### 2026-09-18 — repo changes (Athena worktree)

- `config/llm_profiles.yaml` — new profile `deepseek-v41-flash-mxfp4-engram-4xv100-32gb-circe-test`
- `services/orion-llamacpp-host/app/profiles.py` — `moe_stream`, `moe_stream_cache`, `moe_stream_l2`, `moe_stream_io_threads`, `override_tensor`
- `services/orion-llamacpp-host/app/main.py` — forward those flags when `llama-server --help` advertises them
- This file

### 2026-09-18 — download

Target on Circe:

```text
/mnt/storage-warm/llm-cache/gguf/
  DeepSeek-V4.1-Flash-MXFP4-engram-00001-of-00011.gguf
  ... through ...-00011-of-00011.gguf
```

Repo: `JigSawPT/DeepSeek-V4.1-Flash-GGUF` (~502 GB).

Optional later: `vcruz305/DeepSeek-V4.1-Flash-GGUF` Q2_K (`DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf`, ~246.3 GiB).

### Still TODO before a real boot

1. Build or image the **`dsv41-porte`** CUDA server for **sm_70 (Volta)** — Linux build is marked untested by the fork author; treat first boot as experimental.
2. Point Circe agent (or a dedicated compose service) at:
   - `LLM_PROFILE_NAME=deepseek-v41-flash-mxfp4-engram-4xv100-32gb-circe-test`
   - `CUDA_VISIBLE_DEVICES=0,1,2,3`
   - volume mount `/mnt/storage-fast/llm-cache` → `/models` (Hitachi copy remains on `/mnt/storage-warm/llm-cache`)
   - custom `LLAMACPP_IMAGE_TAG` / local binary that has `--moe-stream`
3. Smoke: `/health`, one short completion, measure cold vs warm tok/s + NVMe pressure.
4. Only then consider RAM upgrade (192→256) or Q2 swap.

---

## Restore Circe to pre-test service layout

Do this when the soak is done (order matters only insofar as chat/agent expect free VRAM):

```bash
ssh circe@circe

# 1) Confirm GPUs 0-3 are idle
nvidia-smi

# 2) Restart the four stopped containers
docker start \
  orion-atlas-llamacpp-chat \
  orion-circe-atlas-llamacpp-agent \
  orion-circe-diffusion-host \
  orion-circe-circe-vision-host-qwen

# 3) Health checks
curl -fsS http://127.0.0.1:8011/health   # chat
curl -fsS http://127.0.0.1:8015/health   # agent
curl -fsS http://127.0.0.1:8014/health || true  # diffusion (port mapping)
curl -fsS http://127.0.0.1:6602/health || true  # vision

# 4) Confirm profiles/GPU bindings still match .env
grep -E '^ATLAS_|^LLAMACPP_IMAGE' \
  /mnt/scripts/Orion-Sapienform/services/orion-llamacpp-host/.env
```

Expected bindings after restore (from live `.env` at eviction time):

| Container | Host port | CUDA | Profile |
| --- | ---: | --- | --- |
| `orion-atlas-llamacpp-chat` | 8011 | 0,3 | `qwen36-35b-a3b-udq5km-2xv100-32gb-deep-cognition` |
| `orion-circe-atlas-llamacpp-agent` | 8015 | 1 | `qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex` |
| `orion-circe-diffusion-host` | 8014 | 2 | (diffusion) |
| `orion-circe-circe-vision-host-qwen` | 6602 | 2 | (vision) |
| `orion-atlas-llamacpp-fast` | 8013 | 4 | `qwen3-8b-q4km-v100-16gb-balanced` (was never stopped) |
| `orion-atlas-llamacpp-metacog` | 8012 | 5 | `qwen3-8b-q5km-v100-16gb-atlas-metacog-16k` (was never stopped) |

If a container was recreated during the test, prefer compose bring-up from Circe’s checked-out repo instead of bare `docker start`:

```bash
cd /mnt/scripts/Orion-Sapienform
# pull the branch only if you intentionally want the profile/host patches on Circe
git -C /mnt/scripts/Orion-Sapienform status

docker compose \
  --env-file services/orion-llamacpp-host/.env \
  -f services/orion-llamacpp-host/docker-compose.atlas-workers.yml \
  up -d atlas-chat atlas-agent

# diffusion + vision: use their own compose files / usual Circe bring-up path
```

Snapshot reference: `/mnt/storage-warm/deepseek-v41-flash-circe/pre-eviction-snapshot.txt`

---

## Offload topology this profile encodes

```text
4× V100 32GB (128 GB HBM)  → dense/shared + hot experts + KV  (--moe-stream-cache)
~96–120 GiB host RAM       → warm expert L2                   (--moe-stream-l2)
NVMe (/mnt/storage-fast)   → Engram tables + cold experts     (mmap / stream)
HDD  (/mnt/storage-warm)   → Hitachi GGUF deleted 2026-09-18 22:44 UTC
```

Circe after eviction had ~220 GiB available RAM, so L2=96 is intentional headroom vs the 5090 recipe’s 72.

---

## Log

| When (UTC) | What |
| --- | --- |
| 2026-09-18 ~18:15 | Worktree created; Circe inventory; storage-warm dirs created via Docker root |
| 2026-09-18 ~18:16 | Stopped chat/agent/diffusion/vision; GPUs 0–3 near-idle; RAM ~220 GiB avail |
| 2026-09-18 ~18:17 | Profile + `moe_stream*` forwarding landed in Athena worktree |
| 2026-09-18 ~18:19 | MXFP4+Engram download started via `docker run --name deepseek-v41-dl` → `/mnt/storage-warm/llm-cache/gguf/DeepSeek-V4.1-Flash-MXFP4-engram/` |
| 2026-09-18 ~18:23 | Download ~50G / ~502G and climbing; `dsv41-porte` shallow clone in progress on Circe |
| 2026-09-18 | YAML profile schema verified (`scripts/verify_deepseek_v41_profile.py`) |
| 2026-09-18 ~19:32 | MXFP4+Engram download finished (`DONE`, 11 shards) |
| 2026-09-18 ~20:05 | Compile died at 47%: `llama-moe-stream.cpp` `std::isfinite` without `<cmath>` (Linux untested). Patched on Circe clone; rebuild started. |
| 2026-09-18 ~20:17 | Volta `llama-server` linked; `--moe-stream*` present. Smoke container `dsv41-smoke` started on `:8099`, loading the 11-shard GGUF. |
| 2026-09-18 ~21:55 | Lexar enumerated as `nvme1` (Gen3 x4). Formatted ext4, mounted `/mnt/storage-fast`, fstab added. rsync of 468G GGUF Hitachi → Lexar started (~157 MB/s, ~50 min). Log: `/mnt/storage-warm/deepseek-v41-flash-circe/lexar-copy.log` |
| 2026-09-18 ~22:00 | Juniper: delete Hitachi copy. Waiter will wipe `/mnt/storage-warm/llm-cache/gguf/DeepSeek-V4.1-Flash-MXFP4-engram` only after size+file-list match. Log: `hitachi-delete.log` |
| 2026-09-18 ~22:43 | rsync 100% (501,809,988,803 bytes both sides). Hitachi tree deleted. |
| 2026-09-18 ~23:08 | `dsv41-smoke` restarted from Lexar `/mnt/storage-fast`, port 8099, GPUs 0–3. |
| 2026-09-18 ~23:16 | Model loaded (~6.5 min). `/health` ok. First chat: 10 prompt / 8 gen tokens, ~1.3 tok/s prefill, ~1.7 tok/s decode. |

### Monitor download

```bash
ssh circe@circe 'du -sh /mnt/storage-warm/llm-cache/gguf/DeepSeek-V4.1-Flash-MXFP4-engram; docker logs --tail 20 deepseek-v41-dl; tail -5 /mnt/storage-warm/deepseek-v41-flash-circe/logs/download-mxfp4.log'
```

### Next after download finishes

1. Finish `dsv41-porte` clone/build for **sm_70** (Volta) — Linux untested by fork author.
2. On Circe: `git -C /mnt/scripts/Orion-Sapienform fetch` + checkout/pull this branch (or rsync worktree), rebuild `orion-llamacpp-host` with the fork binary or a custom image.
3. Point agent worker at profile `deepseek-v41-flash-mxfp4-engram-4xv100-32gb-circe-test`, mount `/mnt/storage-warm/llm-cache` → `/models`, `CUDA_VISIBLE_DEVICES=0,1,2,3`.
4. Smoke load; then restore via checklist above.
