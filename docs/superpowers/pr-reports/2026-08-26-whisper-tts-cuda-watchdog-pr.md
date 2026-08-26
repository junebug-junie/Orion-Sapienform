# whisper-tts: fail loud at boot, self-heal mid-uptime, on CUDA staleness

## Summary

- Closes a real, live incident (2026-08-26): a docker+nvidia-container-toolkit
  staleness quirk (`Failed to initialize NVML: Unknown Error`) silently broke
  Coqui TTS mid-uptime while STT (same container, same GPU) kept working.
  Found while verifying the [mic + affect bracket PR](../pr-reports/2026-08-26-orion-hears-juniper-affect-bracket-pr.md)
  live -- TTS crashed on the very first real synth call after that PR's
  redeploy.
- Wires `main.py`'s `_require_cuda_or_die()` into `startup()` -- it existed
  as dead code (defined, documented with "call this during startup", never
  called) for an unknown period. Now fails loud at boot if `TTS_USE_GPU=true`
  and CUDA is genuinely unavailable.
- New `app/cuda_watchdog.py`: a background task, same shape as the existing
  `heartbeat_loop`, that polls `torch.cuda.is_available()` and self-restarts
  the process (clean `SIGTERM`, not `os._exit`) after `N` consecutive
  failures -- `restart: unless-stopped` then recovers the container
  automatically instead of the outage sitting silent until a human notices.
- Both gated on `TTS_USE_GPU` (skipped entirely for a deliberate CPU-mode
  deployment) and independently toggleable (`CUDA_WATCHDOG_ENABLED`).

## Outcome moved

Before: a CUDA staleness event was silent. STT kept working (masking the
problem), TTS crashed on first use with no automatic recovery -- found only
when a human tried to use voice and it failed. Root cause required manual
investigation (checking `torch.cuda.is_available()` inside the container,
then `nvidia-smi` there for the actual NVML error) and a manual
`docker compose restart`.

After: the same failure mode is caught within `CUDA_WATCHDOG_POLL_SEC *
CUDA_WATCHDOG_FAILURE_THRESHOLD` seconds (default 60s) and self-heals with
no human action. A boot with CUDA already broken fails immediately and
loudly instead of serving requests that will crash later.

## Current architecture (before this patch)

- `app/main.py`'s `startup()` never called the already-defined
  `_require_cuda_or_die()` -- confirmed by reading the function; the
  call site simply did not exist anywhere in the file.
- No liveness check existed for CUDA at all after boot. The only existing
  background task was `heartbeat_loop` (publishes `orion:system:health`
  every 30s) -- it does not probe CUDA, only that the process is alive.
- `app/stt.py`'s `STTEngine` picks `"cuda" if torch.cuda.is_available() else
  "cpu"` -- an explicit fallback. `app/tts.py`'s `CoquiBackend` has none;
  the underlying `TTS` library hard-asserts CUDA
  (`TTS/utils/synthesizer.py:90`).

## Architecture touched

| Seam | Change |
| --- | --- |
| `app/cuda_watchdog.py` | **new** -- the watchdog loop + pure decision function + restart trigger |
| `app/main.py` | wires `_require_cuda_or_die()` into `startup()`; starts/stops the watchdog task alongside the other background tasks |
| `app/settings.py` | 3 new settings |
| `.env_example` / `.env` | synced |
| `docker-compose.yml` | comment explaining why the new keys are deliberately NOT added to the `environment:` allowlist |
| `README.md` | new "GPU liveness" section |

## Files changed

- `services/orion-whisper-tts/app/cuda_watchdog.py`: new module -- `should_trigger_restart` (pure), `cuda_watchdog_loop` (the task), `restart_process` (the real on_trigger).
- `services/orion-whisper-tts/app/main.py`: `_require_cuda_or_die()` call wired into `startup()`, gated on `settings.tts_use_gpu`; watchdog task created/cancelled alongside `heartbeat_task`.
- `services/orion-whisper-tts/app/settings.py`: `cuda_watchdog_enabled`, `cuda_watchdog_poll_sec`, `cuda_watchdog_failure_threshold`.
- `services/orion-whisper-tts/.env_example`, root-checkout `.env`: the 3 new keys, synced.
- `services/orion-whisper-tts/docker-compose.yml`: comment only -- explains why the 3 keys are deliberately absent from the `environment:` list (this service already declares `env_file: .env`, and the review finding from PR #1881 the same day showed an explicit `${VAR:-default}` entry there can *override* a service-`.env` value with whatever the *compose invocation's* env-file context resolves to).
- `services/orion-whisper-tts/README.md`: new "GPU liveness" section documenting the incident and both checks.
- `services/orion-whisper-tts/tests/test_cuda_watchdog.py`: new, 12 tests.
- `services/orion-whisper-tts/tests/test_main_startup_cuda_gating.py`: new, 4 tests -- exercises the real `startup()` coroutine, not a reimplementation of its logic (see `feedback_test_the_lifecycle_not_just_the_arithmetic` in project memory).

## Schema / bus / API changes

None. No channel, schema, or payload changes.

## Env/config changes

- Added keys (orion-whisper-tts): `CUDA_WATCHDOG_ENABLED=true`,
  `CUDA_WATCHDOG_POLL_SEC=30`, `CUDA_WATCHDOG_FAILURE_THRESHOLD=2`.
- `.env_example` updated: yes.
- Local `.env` synced: yes, directly (this worktree's own `.env_example`
  is invisible to `scripts/sync_local_env_from_example.py`, which reads
  `.env_example` from the primary checkout -- a known trap, see
  `feedback_env_sync_reads_example_from_primary_checkout` in project
  memory -- so the primary checkout's `services/orion-whisper-tts/.env`
  was hand-edited and verified).
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-whisper-tts/tests/test_cuda_watchdog.py             12 passed
services/orion-whisper-tts/tests/test_main_startup_cuda_gating.py   4 passed
services/orion-whisper-tts/tests/ (full service suite)             54 passed, 2 failed
  -- the 2 failures (test_tts_worker_replies.py, a pre-existing pydantic
     UUID-format issue unrelated to this patch) are confirmed identical
     on origin/main in isolation; not introduced here.
```

## Evals run

```text
No eval harness exists for orion-whisper-tts. Not added here -- this patch
is deterministic infrastructure (a liveness check + a restart trigger), not
model-quality behavior an eval would measure.
```

## Docker/build/smoke checks

```text
docker compose --env-file <primary>/.env --env-file services/orion-whisper-tts/.env \
  -f services/orion-whisper-tts/docker-compose.yml config
  => CUDA_WATCHDOG_ENABLED: "true", CUDA_WATCHDOG_FAILURE_THRESHOLD: "2",
     CUDA_WATCHDOG_POLL_SEC: "30"   (resolves correctly through env_file -> compose -> settings)

docker compose ... build
  => built clean, image sha256:f06faec711ec...

docker compose ... up -d   (REAL deploy, replacing the live orion-athena-whisper-tts container)
  => boot log:
     [WHISPER-TTS] INFO - TTS configured backend=coqui ... gpu=True ...
     [WHISPER-TTS] INFO - Heartbeat loop started. boot_id=1d019cc4-...
     [WHISPER-TTS] INFO - cuda_watchdog_started poll_sec=30.0 failure_threshold=2
     Application startup complete.
  => _require_cuda_or_die() passed silently (real GPU was healthy) -- no
     regression on the healthy path.
  => Real TTS synth call against the new image succeeded:
     "CUDA watchdog deployed and running." -> audio_b64 len=187152
```

The watchdog's actual restart-on-staleness behavior was NOT re-triggered
live (that would mean deliberately breaking CUDA on the production GPU
again) -- it is covered by `test_cuda_watchdog.py`'s loop tests with fully
injected `is_cuda_available`/`on_trigger`, exercising the real async loop
and real debounce logic, just not a real NVML failure.

## Review findings fixed

Self-reviewed before commit (no separate reviewer agent run for this
smaller, mechanical patch -- flagging honestly rather than claiming a review
pass that didn't happen):

- Finding: `_require_cuda_or_die()` unconditional would force a GPU on a
  deliberate CPU-mode deployment.
  - Fix: gated on `settings.tts_use_gpu`, matching the watchdog's own gate.
- Finding: `docker-compose.yml`'s `environment:` allowlist is the exact
  shape that broke a kill switch on a sibling PR earlier the same day
  (`environment:` overriding `env_file:` with the wrong interpolation
  context).
  - Fix: did not add the new keys there at all; left a comment explaining
    why, referencing the concrete incident.
  - Not a live risk either way for THIS feature specifically (worst case
    if it recurred would be the watchdog staying at its safe default,
    not a kill switch reverting to "recording on") but the pattern itself
    is worth never repeating.

## Restart required

```bash
# Already deployed live during verification -- see Docker/build/smoke
# checks above. For a future redeploy from a clean worktree:
cd <this-worktree>
docker compose --env-file <repo-root>/.env \
  --env-file services/orion-whisper-tts/.env \
  -f services/orion-whisper-tts/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low -- a false-positive restart. Mitigated by the
  `CUDA_WATCHDOG_FAILURE_THRESHOLD` debounce (default 2 consecutive
  failures, not 1) and by the check itself raising being treated as
  "unknown", not "failed" (a broken check must not count toward the
  threshold).
- Severity: low -- this cannot fix the underlying host/nvidia-container-
  toolkit quirk, only make its symptom self-healing. If the quirk is
  itself caused by something recurring frequently (e.g., a sibling GPU
  container being rebuilt often), the container could restart somewhat
  often. Worth watching `cuda_watchdog_triggering_restart` log frequency
  after this ships.

## PR link

<filled in after push>
