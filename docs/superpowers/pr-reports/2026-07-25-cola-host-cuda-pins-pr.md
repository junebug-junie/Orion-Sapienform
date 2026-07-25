# PR Report: Pin torch/cu126 + transformers==4.44.2 for Volta GPUs (`fix/cola-host-cuda-pins`)

## Summary

- `services/orion-llama-cola-host/requirements.txt` only floored `torch>=2.2.0` and `transformers>=4.40.0`, so a fresh build resolved torch's default cu13.x wheel and transformers 5.x.
- Pins `torch==2.13.0+cu126` (via `--extra-index-url https://download.pytorch.org/whl/cu126`) -- cu126 is the newest published index that still ships sm_70 (Volta/V100) kernels.
- Pins `transformers==4.44.2` -- `intention.py` (vendored from the LAMDA-RL HF repo) targets this version's `LlamaAttention`/RoPE call signature; 5.x reshapes q/k differently and breaks the custom forward pass.
- Documents both errors, plus a pre-existing fp32-OOM footgun in the downloaded model cache, in the service README's troubleshooting section.

## Outcome moved

`orion-llama-cola-host` goes from crash-looping on every V100 node in the fleet (unpinned deps silently drift to incompatible major versions) to a clean, reproducible `docker compose build` that actually serves `/v1/understand` against the real 10B checkpoint.

## Current architecture

`requirements.txt` had no upper bounds on `torch`/`transformers`. A rebuild today pulls whatever's newest on PyPI, which happened to be a cu13-only torch build (no Volta kernels) and transformers 5.x (breaking `intention.py`'s vendored RoPE code) -- neither incompatibility is visible until the model actually loads and runs a forward pass on real V100 hardware.

## Architecture touched

- `services/orion-llama-cola-host/requirements.txt`: dependency pins only, no code changes.
- `services/orion-llama-cola-host/README.md`: new troubleshooting section.

## Files changed

- `services/orion-llama-cola-host/requirements.txt`: pin `torch==2.13.0+cu126` (with `--extra-index-url`) and `transformers==4.44.2`.
- `services/orion-llama-cola-host/README.md`: documents the CUDA-kernel error, the RoPE/head_dim error, and the fp32-OOM model-cache fix (the last one is data, not code -- no requirements.txt change for it).

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: none (dependency pins only).
- Compatibility notes: n/a.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: n/a, no env keys changed.
- skipped keys requiring operator action: none.

## Tests run

```text
docker run (llama-cola-host:0.1.0, tests/ + app/ mounted) -> pytest tests -q
  4 passed, 12 warnings (pydantic v2 deprecation warnings, pre-existing/unrelated)
```

## Evals run

No eval harness exists for this service (same gap noted in the prior `feat/cola-novelty-signal` PR).

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-llama-cola-host build   -> clean build, resolves torch-2.13.0+cu126 and transformers-4.44.2
scripts/safe_docker_build.sh orion-llama-cola-host up -d   -> container starts, /health -> {"status":"ok"}
curl -X POST localhost:8005/v1/understand -d '{"text": "..."}'
  -> real 64-dim cola_action_distribution response against the live 10B checkpoint on Circe's GPU 2 (V100)
```

Live-verified end to end on Circe against the real model, not just a compile/import check -- this is the exact failure sequence (CUDA kernel error, then a RoPE shape mismatch, then a stale-index/fp32-OOM cache issue) that was hit and fixed ad hoc in a running container during an interactive latent-space probing session, before being turned into this permanent pin. That same session also ran a first-look validation of what the CoLA action codebook actually encodes (dominant code tracks certain sentence-level speech-acts -- "report" and "exclamation" -- far more than topic, at rates well above chance across a 5-topic x 5-intent x 3-phrasing grid); that result lives in the conversation transcript, not in this repo, since it's an experiment output rather than a code change.

## Review findings fixed

Not run through the code-review skill subagent for this pass -- pure dependency pins + docs, no logic changed, and both fixes were already live-verified against the real model on real V100 hardware before being committed.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-llama-cola-host/.env \
  -f services/orion-llama-cola-host/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Low
  - Concern: `--extra-index-url` in `requirements.txt` means pip checks two indexes; if PyPI ever ships a package name collision with the same version on both indexes, resolution could pick the wrong one. Standard/common practice for pinning CUDA-variant torch builds, low real risk.
  - Mitigation: none needed currently; flagging for awareness.
- Severity: Low
  - Concern: pinning `transformers==4.44.2` is a compatibility workaround, not a real fix -- `intention.py`'s vendored RoPE code will eventually need porting to newer transformers, or this pin will start blocking upgrades needed by other tooling in the container.
  - Mitigation: documented in the README as the root cause; a proper fix would port `intention.py`'s attention code, out of scope for this patch.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/cola-host-cuda-pins
