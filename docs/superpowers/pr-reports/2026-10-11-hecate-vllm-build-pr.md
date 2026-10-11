## Summary

- Correct GLM source assembly to cherry-pick compatible patches and reject the known MTP regression.
- Add an explicit Hecate CUDA 12.8 / SM70 image build and a single-V100 Qwen validation profile.
- Protect Athena's pull/rebuild path with a required worker Compose profile and regression tests.
- Record live hardware inventory, authorized NVMe provisioning and pinned model downloads.

## Outcome moved

The old source helper imported an incompatible upstream base. The corrected
helper reproduces the reviewed tree and safely resumes manual conflict resolution.
The Hecate worker now has a reproducible Docker build path outside Athena's
normal deployment sweep. GPU compilation/inference are still in progress.

## Current architecture

Athena routes through the GPU pool and gateway. Hecate has one V100-SXM2-32GB,
currently serving Qwen3.8-27B GGUF through llama.cpp. GLM's documented TP4 setup
cannot run on that inventory. The operator authorized a smaller vLLM test model
and erasing the identified Toshiba NVMe for model storage.

## Architecture touched

vLLM host build/preparation scripts, unselected profiles and pull/rebuild tests.
No live pool routing, Circe workload or cognition contract changed.

## Files changed

- `services/orion-vllm-host/scripts/{prepare_source.py,glm53.lock.json}`: per-step tree and ancestry verification.
- `services/orion-vllm-host/Dockerfile.sm70` and its ignore file: pinned base and source, GCC14/Python3.12/SM70 build.
- `services/orion-vllm-host/docker-compose.hecate.yml`, `scripts/hecate.sh`: explicit Hecate build/run with real NVMe mount.
- `services/orion-vllm-host/docker-compose.yml`: require `vllm-worker` profile.
- `config/llm_profiles.yaml`: one-card Qwen acceptance profile, exact model path/alias.
- Host and rebuild tests: wrong ancestry/resume, hardware profile and Athena auto-start exclusion.
- Runbook, README and earlier PR report: retire incorrect assembly and record current operations.

## Schema / bus / API changes

- Added/removed/renamed: none.
- Behavior changed: source assembly and explicit worker launch selection.
- Compatibility: manual stock-image launches now also require `--profile vllm-worker`.

## Env/config changes

- Added key: `VLLM_SOURCE_DIR` for Hecate's source build context.
- Removed/renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python3 scripts/sync_local_env_from_example.py`: yes; service-specific `--all-keys` sync also run.
- Skipped keys requiring operator action: none for this patch. Athena source path stays empty; Hecate points at its verified checkout.
- Hecate service env created in its worktree and synchronized to its primary checkout's ignored env for the parity gate.

## Tests run

40 focused vLLM host/rebuild tests passed, followed by 4 preparation tests including
two additional host/mount rejection cases. Env and compose parity gates passed.
The stale real-repository exact import fan-out test now uses an explicit miniature
import graph, preserving an exact assertion without unrelated repository growth.

## Evals run

Actual fresh network source assembly reached the documented conflicts; both
additions were inspected and retained. Manual commit plus helper `--resume` passed
with tree `b321701ed88fc9882ae7c71cdb5968a6635505ab`. Forbidden upstream commit is
not an ancestor. Model acceptance remains UNVERIFIED pending image completion.

## Docker/build/smoke checks

- Hecate override `config --quiet`: PASS through the safety wrapper.
- Athena generic `--dry-run up -d`: `no service selected`, proving the worker opt-in blocks routine startup.
- Full CUDA image build: IN PROGRESS at `/mnt/scripts/hecate-vllm-build-headers.log`. Initial integrity check caught an upstream Docker-ignore omission; the image now reconstructs the pinned tree from Git objects, and that gate passes. The next CMake pass identified missing Python development headers; `python3.12-dev` is now explicit in the image.
- Qwen pinned snapshot: downloaded to Hecate NVMe.
- GLM pinned snapshot: downloaded successfully; downloader exited 0. All 13 shards exist (104,011,208,249 weight bytes); 24 metadata files match the locked revision. Qwen has all 3 shards (8,044,982,000 weight bytes), with 13 matching metadata files.
- Existing Hecate worker `/health`: `{"status":"ok"}` before testing.
- NVMe `/dev/nvme0n1p1` mounted ext4 at `/mnt/models`, UUID `19d66b9a-2234-4a60-9180-1862285cd74c`; fstab verified. The operator explicitly approved erasing its prior BitLocker contents. Saved layout metadata is not a data backup.

## Review findings fixed

- Finding: one-step source tests did not exercise mixed merge/conflict resume.
  - Fix: real Git graph with conflicting merge, dirty rejection and committed-resolution resume.
  - Evidence: independent reviewer ran 21 host tests successfully.
- Finding: wrapper host/mount failure paths lacked tests (minor).
  - Fix: wrong-host and wrong-mount subprocess tests.
  - Evidence: focused preparation tests pass.
- Independent review found no material issues in the source fix or expanded Docker/Compose/profile changes; reviewer independently ran 40 focused tests.

- CI finding: override Compose files must repeat the journald block. Added the block; focused logging gate passes.

## Restart required

No Athena or Circe restart required for the staging patch. On Hecate, after image
completion and draining its existing role:

```bash
services/orion-vllm-host/scripts/hecate.sh up -d
```

Stop the old worker before starting the new one; preserve it for rollback.
Role announcement remains disabled during direct acceptance.

## Risks / concerns

- Severity: GLM hardware blocker.
- Concern: one physical V100 cannot satisfy the documented four-card setup.
- Mitigation: authorized smaller Qwen checkpoint tests the new engine; keep GLM weights for later capacity.
- GPU build, inference quality, streaming/tool calls and gateway integration remain UNVERIFIED until live evidence is appended.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2615
