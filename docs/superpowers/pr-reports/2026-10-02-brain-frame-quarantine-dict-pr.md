## Summary

- The brain-frame tick (the per-tick snapshot of Orion's substrate that drives the hub Self Brain panel) crashed on every tick starting 2026-10-02 08:26 UTC.
- Cause: the lane-health "quarantine" map holds a small dict per lane (`{"unacknowledged_count": N, "recent_examples": [...]}`), but the brain-frame producer called `float()` on that dict.
- Fix: read `unacknowledged_count` from each entry (`_quarantine_count()` in `brain_frame_producer.py`). No try/except, no dict-to-zero coercion.
- Regression test uses the exact store shape sampled live today; it fails on old code and passes on the fix.

## Outcome moved

Brain frames resume being written/published, and each lane region now carries the real unacknowledged quarantine count in `detail.quarantine` instead of crashing.

## Root cause / timeline

- The dict shape is not new. `store.quarantine_summary()` has returned per-lane dicts since `425512e86` (2026-06-16, "Harden substrate poison quarantine"). The consumer in `brain_frame_producer.py` (2026-07) assumed bare numbers.
- It stayed latent because `substrate_reducer_quarantine` had zero unacknowledged rows until today, so the map was always `{}`. Every existing test also passed `{}`.
- Live evidence (read-only, `orion-athena-sql-db`):
  - First-ever quarantine rows: `storage_write` at 08:26:13.08 UTC and `vision_organ` at 08:26:56.65 UTC (1 row each, unacknowledged).
  - Last row in `substrate_brain_frame_log`: 08:26:12.90 UTC, 0.2 s before the first quarantine row. Still the last row at 09:15 UTC.

## What was dark (08:26 UTC to deploy)

- `substrate_brain_frame_log` stopped receiving rows, so the hub Self Brain panel (`/api/self-brain`, `self_brain_routes.py`) served a frozen 08:26 frame.
- Hub mood-arc status (`mood_arc_status_routes.py`) reads the field-anomaly region from the same table: no new points.
- `orion:substrate:brain_frame` bus publishes stopped (the tick returns `None` before publish).

## Other consumers checked

- `scripts/grammar_truth_gate.py`: only checks the key is present; fine.
- `grammar_truth.py` / `reducer_health.py`: use the separate numeric `unacknowledged_quarantine_count_by_reducer`; fine.
- Hub JS/routes: never read `detail.quarantine`; fine.
- No other consumer does `float()`/numeric math on `quarantine_by_reducer` values.

## Files changed

- `services/orion-substrate-runtime/app/brain_frame_producer.py`: `_quarantine_count()` reads the count from the per-lane dict.
- `services/orion-substrate-runtime/app/worker.py`: docstring correction only.
- `services/orion-substrate-runtime/tests/test_brain_frame_producer.py`: regression tests (live store shape; quarantine-only lane; bare-number fallback).

## Schema / bus / API changes

None. `detail.quarantine` stays a float, now with the real count.

## Env/config changes

None.

## Tests run

```text
pytest tests/test_brain_frame_producer.py -k quarantine   (old code)  -> 1 failed (TypeError, reproduces crash)
pytest tests/test_brain_frame_{producer,worker,store,settings}.py + evals/test_brain_frame_substance_eval.py -> 36 passed
```

`tests/test_grammar_consumer_integration.py` fails to collect locally (needs Postgres on localhost:5432); pre-existing, unrelated.

## Evals run

```text
evals/test_brain_frame_substance_eval.py -> 2 passed
```

## Docker/build/smoke checks

Not deployed (per task). Live smoke after deploy: `SELECT max(created_at) FROM substrate_brain_frame_log` should advance past 08:26, and `brain_frame_tick_failed` should stop in `docker logs orion-athena-substrate-runtime`.

## Review findings fixed

Reviewer (subagent) found no blocking issues. Minor findings, all fixed:

- Finding: a lane present only in the quarantine map would be silently dropped (`lane_keys` built from lag/backlog/labels only).
  - Fix: `lane_keys` now includes `set(quarantine)`.
  - Evidence: `test_lane_quarantine_only_lane_and_bare_number_fallback`.
- Finding: `_brain_frame_lane_health` docstring said quarantine is keyed by cursor name; it is keyed by reducer_key already (remap is a no-op for it).
  - Fix: docstring corrected in `worker.py`.
- Finding: bare-number fallback branch untested.
  - Fix: covered by the same new test.
- Note: the brain-frame evals use only empty quarantine maps, which is why this bug was invisible; the new tests close that gap.

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build substrate-runtime
```

## Risks / concerns

- Severity: low. The two quarantined events (storage_write, vision_organ) are still unacknowledged; they are a separate question from this crash.

## PR link

(filled in on the PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
