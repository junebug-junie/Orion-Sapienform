## Summary

- `orion.gpu_pool.client` never set `reply_to` on its lease RPC envelope. The Rabbit chassis only replies to `env.reply_to`, so the pool processed every lease and then stayed silent. Every live `gpu_lease()` call timed out, even though the pool had granted it.
- The fix sets `reply_to` to the same channel `rpc_request` listens on.
- The round-trip test's fake bus used to reply whether or not `reply_to` was set, and that is why the bug passed CI. It now behaves like the real chassis: no `reply_to` means no reply.

## Outcome moved

A live lease over the real bus now completes. It is granted on metacog `:8012` (`Qwen_Qwen3-8B-Q5_K_M.gguf`, ctx 4096/slot) after 67 ms in line, and released `ok`.

## Current architecture

orion-gpu-pool stage 1 (#2318) is deployed in observe mode. No production caller uses the client yet, so nothing in production was affected by the bug.

## Architecture touched

`orion/gpu_pool/client.py` only (plus its test).

## Files changed

- `orion/gpu_pool/client.py`: set `reply_to=reply_channel`.
- `services/orion-gpu-pool/tests/test_client_roundtrip.py`: the fake bus now enforces `reply_to` the way Rabbit does.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: lease RPCs now receive their replies.
- Compatibility notes: none.

## Env/config changes

None. Local `.env` sync is not applicable.

## Tests run

```text
test_client_roundtrip.py with the OLD client + new fake: FAILS ("responder would reply to None, caller listens on orion:gpu_pool:reply:…")
services/orion-gpu-pool tests with the fix: 25 passed, 2 skipped (Postgres tests skip without GPU_POOL_TEST_POSTGRES_URI; CI runs them)
```

## Evals run

No change to scheduling policy. `run_pool_day_eval.py` is unaffected (it does not use the client).

## Docker/build/smoke checks

```text
live bus smoke (fixed client -> deployed orion-athena-gpu-pool):
  gpu_pool_events: smoke3 admitted -> granted metacog (66.8 ms wait) -> released ok (70.6 ms held)
the two earlier smokes with the broken client: granted, then expired after 30 s without heartbeat, then released
  (expire:heartbeat_lost) and never re-granted -- #2318's review fix working live
```

## Review findings fixed

- Finding: the test double diverged from the real chassis contract, so it passed for the wrong reason.
  - Fix: the double now enforces `reply_to`.
  - Evidence: the test fails against the old client and passes against the fix.

## Restart required

```text
No restart required. The client is library code; its first production caller (the gateway, stage 3) will ship with it.
```

## Risks / concerns

- Severity: low
  - Concern: none beyond the above.
  - Mitigation: n/a

## PR link

(see PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
