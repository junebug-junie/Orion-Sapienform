## Summary

- Hold circe's gpu1 (lane 1, the `agent` role) at 150 W, as gpu2 has been since 2026-10-29. Juniper accepted the slowdown.
- One entry in `config/gpu_power_limits.json`. The checked-in-config test now pins gpu1 and gpu2 and asserts that the chat card (gpu0) is never capped.

## Outcome moved

Lower power and heat on gpu1. For scale: PR #2963 measured gpu2 at 150 W writing 0.86x as fast for 0.80x the energy per token, at a realistic load. gpu1 at 150 W itself is not measured.

## Current architecture

`orion-gpu-power-limits` (`scripts/ops/gpu_power_limits.py`, a 5-minute timer on circe) holds each card listed in the JSON at its wattage. Cards not listed are never touched.

## Architecture touched

`config/gpu_power_limits.json`, `tests/scripts/test_gpu_power_limits.py`.

## Files changed

- `config/gpu_power_limits.json`: adds `"1": {"watts": 150, ...}`.
- `tests/scripts/test_gpu_power_limits.py`: expects `{1: 150, 2: 150}`; gpu0 never capped.

## Schema / bus / API changes

- None.

## Env/config changes

- None (JSON config only).

## Tests run

```text
tests/scripts/test_gpu_power_limits.py  24 passed
gpu1 limits (nvidia-smi): min 100 W, default 250 W -> 150 W is in range
```

## Evals run

```text
n/a
```

## Docker/build/smoke checks

```text
Live check after merge: ssh circe@circe nvidia-smi -i 1 --query-gpu=power.limit --format=csv  -> 150.00 W
```

## Review findings fixed

- One-line config change; no separate review.

## Restart required

```bash
# circe: git pull on the primary main checkout; the 5-minute timer applies it
```

## Risks / concerns

- Severity: low. Concern: agent-lane work on gpu1 slows (gpu2 measured 0.86x write speed at this cap). Mitigation: Juniper accepted it; rollback is to delete the entry.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3018

🤖 Generated with [Claude Code](https://claude.com/claude-code)
