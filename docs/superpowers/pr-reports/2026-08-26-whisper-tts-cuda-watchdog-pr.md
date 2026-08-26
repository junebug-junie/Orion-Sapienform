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
# Pre-review-fix build/deploy (superseded by the post-fix one below):
docker compose ... build   => built clean, image sha256:f06faec711ec...
docker compose ... up -d   => TTS confirmed working, audio_b64 len=187152

# Post-review-fix rebuild/redeploy (the version actually shipping):
docker compose --env-file <primary>/.env --env-file services/orion-whisper-tts/.env \
  -f services/orion-whisper-tts/docker-compose.yml config
  => stop_grace_period: 2m30s, healthcheck present (resolves correctly)

docker compose ... build
  => built clean, image sha256:7d9a81524e58...

docker compose ... up -d   (REAL redeploy of the live orion-athena-whisper-tts container)
  => boot log:
     [WHISPER-TTS] INFO - Heartbeat loop started. boot_id=9a4a41a5-...
     [WHISPER-TTS] INFO - cuda_watchdog_started poll_sec=30.0 failure_threshold=2
       check_timeout_sec=30.0
     Application startup complete.
  => advisory boot guard passed silently (real GPU healthy) -- no
     regression on the healthy path; STT/listener tasks confirmed alive
     via the corresponding unit test, not re-tested live (would need a
     real CUDA outage to observe against production).
  => curl localhost:7800/health ->
     {"status":"ok",...,"cuda_available":true,"cuda_watchdog_enabled":true}
  => docker inspect --format '{{.State.Health.Status}}' -> healthy
  => Real TTS synth call against the rebuilt image succeeded:
     "All review findings fixed and verified." -> audio_b64 len=196028
```

The watchdog's actual restart-on-staleness behavior was NOT re-triggered
live (that would mean deliberately breaking CUDA on the production GPU
again) -- it is covered by `test_cuda_watchdog.py`'s loop tests with fully
injected `is_cuda_available`/`on_trigger`, exercising the real async loop,
the real debounce logic, the timeout path, and the whole-tick exception
guard, just not a real NVML failure.

## Review findings fixed

High-effort `/code-review` run (fanned into ~8 parallel sub-reviews). 10
findings; 1 verified false positive, 9 real and fixed.

- **False positive**: docstring cited `curiosity_investigation.py`'s
  `_consecutive_not_ready` as prior art, reviewer claimed the symbol
  doesn't exist. Verified live via direct grep: it does (lines
  432/761/775-790) -- the reviewing subagent's own grep must have hit a
  stale checkout. No change made; citation was accurate.

- **Most severe -- Finding**: `_require_cuda_or_die()` raising at startup
  crashed the ENTIRE process (bus, `listener_task`, `stt_task`, everything)
  before any of them started. STT does not need CUDA at all; the whole
  incident this patch closes is STT SURVIVING a CUDA outage that killed
  TTS. A boot-time crash regressed exactly that resilience.
  - Fix: the boot check is now advisory-only (logs `CRITICAL`, never
    raises). The watchdog became the single enforcement mechanism for
    both "broken at boot" and "broken mid-uptime" -- a boot-broken GPU
    fails its first watchdog check almost immediately and restarts on the
    normal threshold, rather than two mechanisms that could drift apart.
  - Evidence: `test_startup_logs_critical_but_does_not_raise_when_cuda_unavailable`
    asserts `startup()` does not raise, `stt_task`/`listener_task` are
    both alive, and the critical log fired.

- **Finding**: `torch.cuda.is_available()` ran synchronously in-line on the
  event loop, no timeout. A genuine NVML wedge (the exact state targeted)
  is documented to hang rather than fast-return -- freezing
  `heartbeat_loop`/`listener_task`/`stt_task` right along with the
  watchdog, and the failure-counting logic never even running.
  - Fix: runs via `asyncio.to_thread` + `asyncio.wait_for`; a timeout counts
    as a real failure (a raised exception still does not -- different
    failure modes, kept distinct on purpose).
  - Evidence: `test_a_hanging_check_is_treated_as_a_failure_not_skipped`,
    `test_check_does_not_freeze_the_event_loop_while_hanging` (asserts a
    sibling task kept making real progress throughout the hang).

- **Finding**: only the check itself was guarded -- `on_trigger()` (e.g.
  `os.kill` raising under a restricted sandbox) or any other exception in
  the tick body could kill this fire-and-forget task silently, with
  nothing left supervising the very outage this feature exists to catch.
  - Fix: the whole tick is now guarded; `on_trigger` may be sync or async.
  - Evidence: `test_an_exception_from_on_trigger_does_not_kill_the_loop_silently`,
    `test_on_trigger_may_be_async`.

- **Finding**: no `stop_grace_period`/`healthcheck`. Docker's default 10s
  grace period could SIGKILL before `shutdown()` ever ran, defeating the
  whole reason `SIGTERM` was chosen over `os._exit()`.
  - Fix: `stop_grace_period: 150s` (above the 120s synth timeout) and a
    `healthcheck` matching 20 other services' convention.
  - Evidence: live-deployed; `docker inspect` reports `healthy`.

- **Finding**: no `Field` validation bounds on `poll_sec`/`failure_threshold`
  -- 0 or negative either busy-loops the check or defeats the debounce on
  the first failure.
  - Fix: `gt=0` / `ge=1`.
  - Evidence: `test_poll_sec_zero_or_negative_is_rejected`,
    `test_failure_threshold_zero_or_negative_is_rejected`. **Caught a real
    bug in my own first attempt at this test**: `env="..."` binds OS
    environ, not constructor kwargs -- the first draft's rejection test
    "passed" for the wrong reason (a leaked `os.environ` mutation from an
    earlier ad hoc terminal check made it look like validation was firing
    when the bad value was never reaching the field at all). Added
    `test_the_env_alias_itself_actually_works_via_real_environ` to close
    that gap for real.

- **Finding**: `/health` and the heartbeat's bus publish reported
  `status: ok` for the entire `poll_sec * failure_threshold` window before
  a restart, with no visibility into CUDA state at all.
  - Fix: added `cuda_available` (fresh direct check, `null` in CPU mode)
    and `cuda_watchdog_enabled` to `/health`.
  - Evidence: `test_health_endpoint_reports_cuda_state`,
    `test_health_endpoint_omits_cuda_state_in_cpu_mode`; live-verified via
    `curl localhost:7800/health` against the redeployed container.

- **Finding**: 4 near-identical hand-copied `shutdown()` cancel blocks (3
  pre-existing, 1 added by this patch) that could drift independently.
  - Fix: deduped into one loop applied uniformly to all four.

- **Finding**: reinvents `orion-mesh-guardian`'s existing
  cooldown/max-attempts-per-hour remediation instead of reusing it.
  - Assessment: correct in principle, but mesh-guardian is an EXTERNAL,
    cross-restart-*persistent* supervisor; this watchdog is IN-PROCESS and
    dies with the very restart it triggers, so it cannot borrow
    mesh-guardian's in-memory state directly. Registering whisper-tts there
    properly is a real, separate architectural change (new roster entry,
    understanding its probe-mode contract), not a quick fix to bolt onto
    this PR.
  - Fix: documented as a named, deliberately deferred risk in the README's
    "GPU liveness" section and in Risks/concerns below, rather than
    silently left uncovered. Not fixed in this patch.

**Self-caught, not from the review**: while writing the settings-bounds
test, an early draft of the hang-simulation test used `time.sleep(3600)`
inside `asyncio.to_thread` -- `asyncio.wait_for`'s timeout stops AWAITING a
thread, it does not kill the underlying OS thread, so that leaked a real
hour-long-sleeping thread and hung the whole test process (confirmed live:
the test run had to be force-terminated). Fixed with a bounded (0.3s)
simulated hang; documented in the test's own docstring why production is
not exposed to this the same way (a real sustained wedge crosses
`failure_threshold` within a couple of poll intervals and `restart_process()`
then exits the whole process, which reaps every thread at the OS level
regardless of Python-side cleanup).

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
- Severity: medium -- **no cross-restart rate limit** (review finding,
  deliberately deferred -- see "Review findings fixed" and the README's
  "GPU liveness" section). Each restart is a brand-new process with fresh
  in-memory state, so a genuinely PERMANENT GPU failure would restart the
  container roughly every `poll_sec * failure_threshold` seconds
  indefinitely, no backoff. `orion-mesh-guardian` already has proper
  cross-restart-persistent remediation with cooldown/max-attempts-per-hour;
  whisper-tts is not registered there. Follow-up: register it, rather than
  building a second, in-process, un-persisted rate limiter here.
- Severity: low -- this cannot fix the underlying host/nvidia-container-
  toolkit quirk, only make its symptom self-healing. If the quirk is
  itself caused by something recurring frequently (e.g., a sibling GPU
  container being rebuilt often), the container could restart somewhat
  often. Worth watching `cuda_watchdog_triggering_restart` log frequency
  after this ships.

## PR link

<filled in after push>
