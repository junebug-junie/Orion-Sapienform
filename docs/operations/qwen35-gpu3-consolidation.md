# Circe GPU3: one Qwen3.5 worker for fast and metacog

## Contract

`atlas-fast` on Circe GPU3 serves the `fast` and `metacog` GPU pool classes.
Its Unsloth `Qwen3.5-9B-UD-Q5_K_XL.gguf` profile has ten slots, each with
32,768 tokens (`--parallel 10 --ctx-size 327680`). Kev remains on GPU3.
`atlas-metacog` and port 8012 are retired. The `quick*`, `metacog*`, and
`operator_metacog` route names retain their priority and wait/backlog policy.
Both classes own the one physical `fast` role, so perception and interactive
priority still beat lower-priority fast calls while capacity is counted once.

The gateway defaults thinking off for both classes on llama.cpp. An explicit
`chat_template_kwargs.enable_thinking=true` is preserved; structured artifact
policy can force it off. Metacog defaults remain temperature 0.35, top-p 0.90,
presence penalty 0.20, and 1,024 output tokens. A metacog call explicitly
enabling thinking without an output budget gets 4,096 tokens. Caller-supplied
values win. Cortex-exec no longer pins the old model; the gateway selects the
metacog profile when a caller omits one.

No event schema or new metric is introduced. Use `llm_call_usage`, pool events,
worker `/props` and `/slots`, and the resulting cognition artifacts as evidence.

## Baseline and open hardware check

Seven days of `llm_call_usage` before this change showed `quick_turn` prompts up
to 16,722 tokens, `quick_background` up to 22,925, and `metacog` up to 7,521.
Larger calls spilled from the old 4k GPU3 slots to agent roles. GPU3 was using
about 19.2 GiB of 32 GiB: fast 7.0, metacog 8.0, Kev 4.1. Removing metacog
frees its share. [Qwen's model config](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json)
implies roughly 10 GiB of FP16 attention KV for 327,680 total tokens (8 full
attention layers × 4 KV heads × 256 dimensions × K/V × 2 bytes). That leaves
plausible headroom, but hybrid state, graph buffers, peak KV, and ten-way
decode performance are **UNVERIFIED** until the live run. The worker's current llama.cpp
pin does not expose `--kv-unified-per-slot`; use the explicit total context.

## Deployment after merge and production-write approval

Deploy only from clean, synchronized primary `main` on each host. The change
requires the same commit on Athena and Circe. Athena goes first: the old
four-slot fast profile remains in the registry solely so the new pool can
discover Circe's still-running worker during the host update. Circe's
post-pull helper does not rebuild the Atlas workers automatically. Keep
`ORION_BUS_URL=redis://100.92.216.81:6379/0`.

1. On Athena, set the gateway's `ATLAS_METACOG_PROFILE_NAME` to
   `qwen35-9b-udq5kxl-v100-32gb-circe-shared-32k` and remove cortex-exec's
   retired `ATLAS_METACOG_PROFILE_NAME` key from its local `.env`. Sync other
   `.env` keys from the examples, then pull clean primary `main`. Confirm the
   post-pull rebuild brought up `orion-gpu-pool`, `orion-llm-gateway`, and
   `orion-cortex-exec`; use `scripts/safe_docker_build.sh` for any missed
   service. The new pool should still discover the old four-slot fast profile.
2. Pull clean primary `main` on Circe. Sync its worker `.env`, set
   `ATLAS_FAST_PROFILE_NAME=qwen35-9b-udq5kxl-v100-32gb-circe-shared-32k`
   `LLAMACPP_IMAGE_TAG=server-cuda-b10398`, and
   `ATLAS_FAST_CUDA_VISIBLE_DEVICES=3`. Confirm the card index against
   `nvidia-smi` before starting it. Stop and remove only
   `orion-circe-atlas-llamacpp-metacog`, with
   no volume deletion. Start the shared worker explicitly with
   `scripts/safe_docker_build.sh orion-llamacpp-host --compose-file docker-compose.atlas-workers.yml up -d --build atlas-fast`.
   Circe's post-pull helper excludes this Atlas compose file, so this explicit
   restart is required.
3. Confirm Circe's `orion-gpu-lane-controller` uses the same pool config digest
   as Athena's pool; deploy it from clean main with `scripts/safe_docker_build.sh`
   if the post-pull rebuild did not do so.
4. Check worker `/health`, `/props` (model file and ten slots), `/slots` (ten
   32,768-token slots), and GPU3 memory beside Kev. Run
   `python3 services/orion-llamacpp-host/evals/run_qwen35_fast_eval.py --base-url http://127.0.0.1:8013 --gateway-url http://100.92.216.81:8210`
   on Circe. It exercises thinking on/off, 24k prompt text, JSON, logprobs,
   and tools, then starts ten distinct long requests behind a barrier and
   samples `/slots` to require ten simultaneously occupied seats. The optional
   gateway pass mixes `quick` and `metacog_perception` and prints observed
   median/p95 call time. Compare those times with live route baselines; this
   smoke proves overlap, not sustained production latency.
5. Check real gateway calls on `quick`, `quick_turn`, `quick_embodied`,
   `metacog`, `metacog_background`, `metacog_turn`, and
   `metacog_perception`. Their new `llm_call_usage.pool_role` should be
   `fast` when served on GPU3; no new call should name physical role
   `metacog`. Inspect JSON parse results for cortex metacog drafts, Topic
   Foundry labels, AI Town replies, and vision-council artifacts. Compare
   failures, p95 wait, and empty/truncated completions with the pre-cutover
   baseline. A healthy container alone does not pass this check.

If the worker cannot load or the real callers fail, stop the rollout and
report the measured failure. Do not mark the migration verified from config or
container health alone.
