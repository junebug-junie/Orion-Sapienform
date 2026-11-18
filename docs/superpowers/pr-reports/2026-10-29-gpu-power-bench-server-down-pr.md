## Summary

- The bench crashed with a traceback when gpu2's model server was down (Juniper's first run, 10-29 ~01:45 UTC: `Connection refused` on :8016).
- Cause: a dream painting (reverie-visual diffusion hold, 01:35) took gpu2, which unloads the agent model; it reloads after ~15 quiet minutes and goes again on the next painting.
- Now the bench waits up to 60 min for the server before touching the card, and says why.
- If the server goes away mid-run, the bench stops, keeps the samples so far, restores the limit and exits 1.

## Outcome moved

The power bench can run on gpu2, a card that switches between the agent model and image generation.

## Current architecture

PR #2945's bench assumed the server was up at start and treated a lost server as an unhandled error (limit restored, data kept, but a traceback).

## Architecture touched

`scripts/gpu_power_bench.py` only.

## Files changed

- `scripts/gpu_power_bench.py`: `wait_for_server`, `--server-wait-s`, clean stop on URLError/ConnectionError/BenchAbort with `meta.aborted`, exit 3 (server never up) / 2 (refused) / 1 (stopped early) / 130 (Ctrl+C while waiting).
- `tests/test_gpu_power_bench.py`: 3 tests.
- `docs/superpowers/pr-reports/2026-10-29-gpu-power-bench-server-down-pr.md`: this report.

## Schema / bus / API changes

- None.

## Env/config changes

- None.

## Tests run

```text
pytest tests/test_gpu_power_bench.py -q -> 30 passed
```

## Evals run

```text
None: operator measurement script.
```

## Docker/build/smoke checks

```text
Live: :8016 refused at 01:50 and 02:51 UTC, up at 01:55 (pool role agent-gpu2 "unloaded" while diffusion held gpu2).
```

## Review findings fixed

- Finding: a read timeout (bare TimeoutError on 3.12), half-sent reply or missing timings escaped as a traceback.
  - Fix: `SERVER_GONE` tuple (URLError, OSError, HTTPException, JSONDecodeError, KeyError) for the clean stop.
  - Evidence: `test_server_leaving_mid_run_keeps_samples_and_restores` x4.
- Finding: /slots failing before the start exited 1 ("stopped early") though nothing changed.
  - Fix: BenchAbort -> 2, server gone -> 3. Evidence: `test_slots_failing_before_start_is_a_clean_refusal`.
- Finding: Ctrl+C while waiting for the server gave a traceback.
  - Fix: exit 130 with a plain message. Evidence: `test_ctrl_c_while_waiting_for_the_server_is_clean`.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: a dream painting can still end a run early. Mitigation: samples kept; rerun picks up from nothing but takes ~10 min.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2959

🤖 Generated with [Claude Code](https://claude.com/claude-code)
