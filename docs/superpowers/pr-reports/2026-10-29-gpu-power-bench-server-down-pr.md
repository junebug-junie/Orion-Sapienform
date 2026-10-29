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

- `scripts/gpu_power_bench.py`: `wait_for_server`, `--server-wait-s`, clean stop on URLError/ConnectionError/BenchAbort with `meta.aborted`, exit 3 (never started) / 1 (stopped early).
- `tests/test_gpu_power_bench.py`: 3 tests.
- `docs/superpowers/pr-reports/2026-10-29-gpu-power-bench-server-down-pr.md`: this report.

## Schema / bus / API changes

- None.

## Env/config changes

- None.

## Tests run

```text
pytest tests/test_gpu_power_bench.py -q -> 25 passed
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

- See PR comments / follow-up commit.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: a dream painting can still end a run early. Mitigation: samples kept; rerun picks up from nothing but takes ~10 min.

## PR link

<link>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
