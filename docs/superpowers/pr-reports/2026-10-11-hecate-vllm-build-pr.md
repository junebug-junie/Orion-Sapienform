# Hecate SM70 vLLM build and live gateway integration

## Outcome

Hecate now serves `Qwen/Qwen3-4B-Instruct-2507` through the patched vLLM image and
Athena's existing `agent-deep` gateway route. Live discovery confirms one slot,
8,192 tokens, exact alias `qwen3-4b-instruct` and NVMe model path. HTTP, streamed
tool calls and bus RPC all reached Hecate. The maintenance hold was released;
a subsequent ordinary request returned `{"answer":391}` with no leaked test lease.

Status: **DONE_WITH_CONCERNS**. The requested GLM weights are downloaded, but the
live inventory has one V100, not the four required by the documented recipe.
Juniper authorized the smaller model. Its strict arithmetic-formatting eval
fails: it returns `17 × 23 = 391` instead of only `391`. We retained the failing
results and did not weaken that gate. Structured arithmetic, recall, JSON,
automatic tool calls and streamed tool calls work. This is an engine/integration
acceptance exercise, not evidence that 4B replaces the old 27B model's quality.

## Problem and changes

- Reconstructed the required source from a pinned base, six cherry-picks and
  the EXL3 merge, verifying every tree and excluding the known MTP regression.
  Final source tree: `b321701ed88fc9882ae7c71cdb5968a6635505ab`.
- Built CUDA 12.8/GCC14/Python3.12/SM70 vLLM on Hecate. Fixed upstream Docker-ignore
  omissions by reconstructing the source from Git objects; added Python dev headers.
- Added explicit Hecate Compose/profile launch with a real NVMe mount check.
  Athena's existing pull/rebuild exclusion was verified on shared main and pinned
  by regression tests; the additional Compose opt-in prevents generic startup.
- Selected the one-card Qwen validation profile and `agent-deep.backend=vllm`.
  Other role definitions, Circe models and launch digests remain unchanged.
- Added configured and live image-storage checks. Docker data-root was on SSD,
  but containerd still unpacked into `/var/lib/containerd` on root. The new gate
  detects this split and config/runtime disagreement. It reads the mount table
  without accessing private directories; inspection/shutdown bypass the gate.

No new schema, bus channel, cognition loop or metric was introduced by this PR.
The earlier gateway/discovery contract patch was merged in PR #2610.

## Live deployment and storage

Juniper explicitly approved NVMe erasure, stopping the old GPU worker, the
containerd storage switch, and the named Athena/Circe integration deployment.

- NVMe: `/mnt/models`, ext4 UUID `19d66b9a-2234-4a60-9180-1862285cd74c`, persistent
  fstab entry; both locked snapshots verified. GLM: 13 weight shards,
  104,011,208,249 bytes. Qwen: 3 shards, 8,044,982,000 bytes.
- Containerd: `/mnt/docker/containerd-store`, with a systemd mount dependency.
  Root recovered from 3.5 to **79 GiB free**; the root LV remains 100 GiB.
  Old store retained intact at `/mnt/docker/containerd-root-backup-20261011`.
  Docker SSD has 686 GiB free; model NVMe has 341 GiB free at verification.
- Migration attempt one exercised successful recovery after a transient
  biometrics startup connection reset. Attempt two used bounded retries and
  curl-managed output; immutable evidence from both attempts remains on Hecate.
  Biometrics is healthy and the image identity was preserved. No image pruning
  or root expansion occurred.
- Hecate image config: `8d4eac1c38a7e8db61cc0add69404a825e5bc14ca1acd3a05e160d62548c03f7`.
  Runtime: torch `2.10.0+cu128`, CUDA 12.8, vLLM `0.1.dev20452+g08b29dcd5`.
- Deployed prepared images: Athena pool `562b90014859`, gateway `c86645e65422`,
  durable-runs `870f7945992b`, mesh-guardian `f23a5ef16030`; Circe lane-controller
  `126f18023f68`. Health checks pass. Circe's model workers were not restarted.
- The old Hecate llama.cpp container/model remain available for rollback.

## Env/config parity

`VLLM_SOURCE_DIR` was added to the template and synchronized locally. Hecate's
worktree and primary ignored envs select the verified source, Qwen profile,
`LLM_ROLE=agent-deep`, `LLM_ANNOUNCE_PORT=8021`. Bus URL is
`redis://100.92.216.81:6379/0`. No env or secrets were committed.

Circe's existing missing `GPU_POOL_FENCE_STATE_PATH` and
`GPU_LANE_DRAIN_TIMEOUT_SEC` keys were synchronized to their existing runtime
defaults. No operator-required skipped key remains for this patch.

## Tests and evals

- **80 upstream EXL3/SM70 GPU tests passed** on the actual V100 in 67.53 seconds.
- Host suite: 35 passed before the final mount-table correction; all 17 focused
  preparation/storage tests passed after it. Earlier source/rebuild checks passed.
- Real source assembly/conflict resume, ancestry, Compose/env gates and Athena
  automatic-start exclusion passed. All ten CI checks passed on `d75ed9c80`.
- Direct strict smoke: recall and JSON pass; arithmetic formatting fails as above.
  Streaming delivers the same equation and a final `[DONE]`; exact-text assertion
  still fails. Automatic and streamed `get_weather(city=Paris)` calls pass.
- Direct and gateway structured arithmetic pass. Both reject excess context
  with HTTP 400 and the actual 8,192-token limit.
- Bus RPC correlation `1b29fb78-10e7-4701-ab92-ec4ed953ff18`: exact model and
  `served_by=hecate-worker-agent-deep`, correct nonempty structured answer.
- Normal post-maintenance request correlation
  `a6eabc40-bbf8-40cc-8b80-9eaac5d94da9`: correct answer via `agent-deep`.
- During warm-up the pool reported down/zero slots. A queued client's timeout
  produced `passthrough_client_gone_while_queued` and no remaining child lease
  (`cbd6e36f-9581-4114-916d-303cb2415380`). This verifies cancellation while unavailable,
  not a completed server-side timeout/error response.
- Circe `quick` route returned nonempty `READY.` from `circe-worker-fast`,
  correlation `0fb5c43c-36cb-421c-81b0-75dbe3ef8ed0`.

Committed synthetic evidence: `docs/superpowers/evidence/2026-10-11-hecate-vllm/`.

## Review findings fixed

- Mixed source merge/conflict resume lacked a regression: added real Git graph,
  dirty rejection and committed-resolution resume checks; independent review passed.
- Hecate wrapper host/mount rejection lacked tests: added subprocess regressions.
- Compose override omitted required journald block: corrected; static gate passes.
- Storage gate trusted config alone: now verifies live snapshot mount paths and
  config/runtime disagreement. Shutdown and inspection remain available.
- Migration could strand stopped services or overwrite recovery evidence:
  added immutable evidence, preflight drop-in checks and phase-aware recovery;
  successful recovery was observed live on the first attempt.
- Nonroot directory access broke storage detection: use the flat mount table;
  nested/missing mounts and sibling prefixes are covered. Independent reviewer
  passed 17 focused tests and found no remaining material code issues.
- Final documentation/evidence review found no material issues; all nine JSON
  artifacts parse and contain synthetic test data without credentials.

## Rollback and limits

Deployment used the reviewed worktree. Merge this PR before the next main-based
pool/gateway rebuild, which would otherwise restore main's previous backend config.
Athena's existing worker exclusion was verified on main.

Drain Hecate, stop the vLLM worker, restore the previous pool/gateway config/images,
then start `orion-hecate-hecate-llamacpp-agent-deep`. Its original 27B GGUF remains
on telemetry storage. Compatible reader upgrades can stay in place. Do not run
both workers on GPU0/port8021. Athena's prepared rollback image tags are recorded
in the runbook. Do not rerun the completed one-time storage migration.

GLM inference/TP4, Anthropic Messages, vision and broad model-quality acceptance
remain **UNVERIFIED**. The live fallback is Qwen 4B with 8K context, replacing a
27B worker that advertised 128K. No claim of equivalent capability is made.

## PR

https://github.com/junebug-junie/Orion-Sapienform/pull/2615
