# Roll the gpu2 agent seat back from Bonsai to the Q4 27B

## Summary

- The pool's gpu2 agent seat (`agent-gpu2`) loads the Q4 27B first again, and Bonsai drops to second place. Bonsai stays on the allowed list, so a seat already running it is confirmed, not refused.
- This undoes the 7.2 profile order (#2477). The prism image and the profile support stay, so promoting Bonsai again later is the same one-line reorder.
- The tests that pin which profile loads first are updated to match.

## Outcome moved

Since 10-02, Bonsai on gpu2 produced no usable agent output. The cause is in the worker log on circe:

```text
Jinja Exception: No user query found in messages.   (HTTP 500)
```

Bonsai's built-in chat template refuses any request that has no user message. Many cortex-exec and durable-run agent steps send only system and tool messages; Q4's template accepts that, Bonsai's does not. Evidence from the read-only check, 10-02 to 10-06:

- **Chat calls:** 44 of 44 chat-completions calls from cortex-exec failed with HTTP 500.
- **Multi-step runs:** 26 of 30 Bonsai-held runs failed. The other 4 "completed" with 16 of 16 blank answers.
- **Curiosity turns:** 0 of 5 Bonsai-only turns had turn_ok, against 28 of 28 on Q4.
- **Hidden failure:** the gateway reports these 500s back to callers as `ok` with empty text, so they read as successes. A follow-up fix is needed.

Both cases are evidence from live data. The quality comparison could not run, because Bonsai never produced a usable answer.

## Files changed

- `config/gpu_pool.yaml`: the profile order is Q4 then Bonsai, with a comment explaining why.
- `orion/gpu_pool/tests/test_stage7_2_bonsai_seat.py`, `orion/gpu_pool/tests/test_stage5_config.py`, `services/orion-gpu-lane-controller/tests/test_launch_exec.py`, `services/orion-gpu-lane-controller/tests/test_actuator_bus.py`, `services/orion-gpu-pool/tests/test_stage5_3_cutover_e2e.py`: tests that pinned Bonsai as the default now pin Q4.

## Schema / bus / API changes

None. The launch digest for `agent-gpu2` changes, so the pool on athena and the controller on circe must be on the same commit (see "Restart required").

## Env/config changes

None.

## Tests run

```text
orion/gpu_pool/tests + orion-gpu-lane-controller/tests + orion-gpu-pool/tests: 615 passed, 15 skipped
check_gpu_pool_config: ok
```

The tests in `services/orion-llamacpp-host/tests/test_prism_seat.py` also fail locally on origin/main, because they depend on local setup. This PR doesn't touch that file.

## Evals run

None. This is a config rollback; the evidence above is from live data.

## Review findings fixed

A review subagent was not run for this PR. The change is two lines of config plus the tests that pin it.

## Restart required

Run the circe line first, then the athena line:

```bash
ssh circe@circe 'cd /mnt/scripts/Orion-Sapienform && git pull --ff-only'
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
```

Between the two pulls, the pool refuses gpu2 loads and unloads, which is safe. Bonsai isn't loaded right now, so the next gpu2 load will be Q4.

## Risks / concerns

- **Severity: medium. Concern:** the gateway reports a worker's HTTP 500 back to callers as `ok` with empty text, so failures stay hidden.
  - **Mitigation:** a follow-up PR.
- **Severity: low. Concern:** Bonsai can't be promoted again until it accepts requests with no user message.
  - **Mitigation:** either give it the Q4 model's chat template, or have the gateway always send a user message. Then run the replay test.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2529

🤖 Generated with [Claude Code](https://claude.com/claude-code)
