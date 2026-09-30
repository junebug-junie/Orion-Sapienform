# Design: move the gpu1/gpu2 agent lanes from Qwen3.8-27B Q4 to Ternary Bonsai 2

**Status:** proposal. No code changed. Evidence: `docs/2026-09-30-flash-attention-volta-circe.md` and `docs/2026-09-30-ternary-bonsai2-27b-1xv100-circe.md` (PR #2434).

## Arsonist summary

The agent lanes on gpu1 and gpu2 serve agent, curiosity and reading work. Each serves one session at 131K. On the same card, Bonsai (a ternary build of the same Qwen3.8-27B base) reasons about as well on our depth tests:
- 19/20 against 20/20;
- ~45% faster at every concurrency level;
- fits **2 sessions × 131K per card** in ~24 GB, against 1 for today's Q4.

The Bonsai "service" differs from `orion-llamacpp-host` only in its `llama-server` binary: same launcher, same profile file, same pool announce. So the migration is not a service swap. It points the **existing** `atlas-agent` and `atlas-agent-burst` workers at a different image and profile. Callers never notice: they reach the gateway, then the `agent`/`agent-gpu2` roles on ports 8015/8016, which do not move.

Result: agent capacity goes from 2 concurrent sessions (1 per card) to 4. Each is faster, and the per-session context the harness budgets for is unchanged.

## Current architecture

- **gpu1:** `atlas-agent` in `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`, image `orion-llamacpp-host:0.1.0` (stock llama.cpp b10398), profile `qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex` via `ATLAS_AGENT_PROFILE_NAME`. It runs 1 slot × 131,072, uses 24.9 GB live, and announces role `agent`, port 8015.
- **gpu2:** `atlas-agent-burst`, the same service shape. It is loaded and unloaded by `orion-gpu-lane-controller` through `config/gpu_pool.yaml` → `agent-gpu2.launch`. The pool sends the first entry of `launch.profiles` as `ATLAS_AGENT_BURST_PROFILE_NAME`. It evicts diffusion, `max_hold_sec` 9000.
- **Pool sizing:** llm roles have no configured slots. `orion/gpu_pool/discovery.py` reads `total_slots` and ctx-per-slot from each worker's `/props`, and `scheduler.py` leases by those slots.
- **Bonsai today:** `services/orion-llamacpp-bonsai-host` is a Volta build of PrismML's fork, layered on the llamacpp-host wrapper, plus a manual-only bake-off compose on gpu0. It is in no auto-rebuild list.
- **Measured demand:** 1,485 agent-lane requests since 2026-09-24 had prompt sizes of median 24K, p90 60.5K, p95 68K, max 82K. 6.8% exceed 65K, so **131K per session is required**. Runs: median 44 steps, p90 134.

## Missing questions

1. **Holds per role.** The scheduler allows at most one *hold* per role (H1). Curiosity and self-sense holds reserve a seat, so with 2 slots a second curiosity hold on the same card would still wait. Keep one hold per card, or allow one per slot?
2. **Thinking effort.** The live agent profile uses `reasoning_effort: xhigh`. Bonsai was tested mostly at `medium`, and once thought past an 8K cap at medium. Keep `xhigh` for parity, or move to `medium`? (`n_predict: 16384` already bounds it.)
3. **`preserve_thinking`.** Live agent uses `true`. The prompt cache misses the prior assistant reply on *both* models (79.4% vs 79.7% reuse), so this is not a Bonsai-specific cost. Keep `true` unless the rollout shows otherwise.
4. **One image or two.** The fork is llama.cpp b10750 plus ternary kernels, so it can probably also serve the stock GGUFs. That is untested. For now, keep two images.

## Proposed schema / API changes

None. No bus, schema, channel or gateway API change. Roles, ports, announce payload (`LlmWorkerAnnounceV1`) and routing stay the same.

Config changes, in the implementation PR:

1. **Image.** Move `services/orion-llamacpp-bonsai-host/Dockerfile` into `services/orion-llamacpp-host/` as `Dockerfile.prism`, mirroring `Dockerfile.dsv41-porte`. Give `atlas-agent` and `atlas-agent-burst` an image and Dockerfile selectable by env (e.g. `ATLAS_AGENT_IMAGE`, default the stock image). Chat, fast and metacog stay on stock.
2. **Profile.** Add `ternary-bonsai2-27b-pq2-v100-32gb-circe-agent`, derived from the agent-flex profile:
   - `ctx_size: 262144`, `n_parallel: 2` (131,072 per session), `flash_attn: "on"`;
   - `max_model_len: 131072`, `max_concurrent_requests: 2`;
   - the same sampling as agent-flex, and `reasoning_effort` per question 2.
3. **Selection.**
   - gpu2: put the new profile first in `agent-gpu2.launch.profiles`.
   - gpu1: set `ATLAS_AGENT_PROFILE_NAME`.
   - Both are one-line reverts.
4. **Retire** `services/orion-llamacpp-bonsai-host`'s compose and bake-off profile once the image lives in llamacpp-host. Keep its README facts in the llamacpp-host README.

## Files likely to touch

- `services/orion-llamacpp-host/Dockerfile.prism` (moved), `scripts/build-prism-volta.sh`
- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml` (per-worker image for the two agent services)
- `services/orion-llamacpp-host/.env_example` (+ local `.env` sync), `README.md`
- `config/llm_profiles.yaml` (agent Bonsai profile)
- `config/gpu_pool.yaml` (`agent-gpu2.launch.profiles`), `scripts/check_gpu_pool_config.py` if it validates profiles or images
- `services/orion-llamacpp-host/tests/` (image-selection + profile argv tests, moved Bonsai contract tests)
- `orion/gpu_pool/scheduler.py`, only if question 1 is answered "one hold per slot"
- `services/orion-llamacpp-bonsai-host/` (removed)

## Non-goals

- Moving chat (gpu0, the 35B MoE) or fast/metacog (gpu3). The 35B is 2–3× faster per session and is not a Bonsai target.
- Quantizing the KV cache (`q8_0`) or `--kv-unified`: separate experiments.
- Changing the harness context budget (stays 131K per session).
- Replacing stock llama.cpp everywhere with the fork.

## Acceptance checks

1. **Build.** `Dockerfile.prism` builds on circe, and the build asserts `build > 5332` and the flags the profile uses (already in today's Bonsai Dockerfile).
2. **Boot.** `atlas-agent-burst` loaded by the pool with the Bonsai profile: `/props` reports `total_slots: 2` and per-slot ctx 131072, and the argv shows `--flash-attn on`.
3. **Pool.** Discovery shows `agent-gpu2` `discovery_confirmed` with 2 slots, and two concurrent agent leases are granted on gpu2.
4. **Same-traffic A/B** from `harness_turn_trace.run_artifact` (`fcc_served_model`, `turn_ok`, `step_count`, `fcc_elapsed_sec`): gpu2 Bonsai vs gpu1 Q4 over at least 1 day or 30 turns each. Bonsai's `turn_ok` rate must not be below Q4's by more than a set margin (proposed: 5 points), and its median `fcc_elapsed_sec` should be lower.
5. **No regression for callers.** No HTTP 500s from template kwargs, via the gateway error rate for `agent-gpu2`.
6. **Rollback drill.** Revert the `launch.profiles` entry; the next pool load serves the Q4 profile again.

## Recommended next patch

The gpu2-only slice. It moves the Dockerfile into llamacpp-host, adds per-worker image selection for `atlas-agent-burst` only, adds the agent Bonsai profile, and makes it `agent-gpu2`'s first launch profile. Run acceptance checks 1–5 on live traffic, then a second patch flips gpu1. Decide questions 1 and 2 before that patch.

## Proposal-mode notes (cognition-adjacent: agent/curiosity execution)

- **Capability change:** the same agent role, a different model build; 2 concurrent sessions per card instead of 1.
- **Data touched:** none new. It reads existing `harness_turn_trace` for the A/B.
- **Privacy boundary:** unchanged (same lanes, same callers).
- **Trace proving it worked:** `fcc_served_model = Ternary-Bonsai-2-27B-PQ2_0` rows with `turn_ok`, plus pool discovery showing 2 slots.
- **Dangerous failure mode:** fluent but wrong tool use or findings on real curiosity runs that synthetic tests did not catch. One story looked fabricated in the bake-off. Mitigation: gpu2 first, compare `turn_ok` against Q4, and read a sample of findings by hand.
- **Disable / rollback:** a one-line profile revert per card. The stock image stays the default.
