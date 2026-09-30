## Summary

- `tool_progress` heartbeat frames from the FCC stream are no longer recorded as grammar step atoms (started+completed pair + 2 edges each) by `HarnessGrammarCollector`.
- They are also excluded from `step_char_sum` / `step_char_max`, so `avg_step_chars` keeps numerator and denominator over the same population.
- Untouched on purpose: the live stream, per-step receipts, `publish_harness_run_step`, the runner's `step_count`, the source-level frame loop in `fcc_motor.py`.

## Outcome moved

2026-09-29: a 183-step harness run (137 of them `tool_progress`) flushed 743 grammar events in ~20 ms and overflowed a sql-writer lane (722 events shed to `bus_fallback_log`; see PR #2410). Heartbeats were 21% of all recorded harness step atoms over 48 h (2,168 / 10,389 across 294 runs) and ~75% inside the largest runs. Largest bursts shrink ~4x; total grammar volume ~-21%.

## Current architecture

`fcc_motor.py` yields one `step` event per parsed CLI stream line (only `thinking_tokens` was previously filtered, 2026-09-14, same class of bug). `runner.py` turned each into `record_step_started/completed`; `build_harness_grammar_events` flushed atoms+edges at run end; `grammar_extract.py` counts `exec_step_started` -> `harness_started_step_count` -> `harness_step_load`, and `step_char_sum / completed_step_count` -> `avg_step_chars`.

## Architecture touched

`orion/harness/runner.py`, `orion/harness/fcc_motor.py` (helper only). No bus, schema, channel or env change.

## Files changed

- `orion/harness/fcc_motor.py`: `is_progress_frame()`.
- `orion/harness/runner.py`: skip collector step atoms + char accounting for progress frames.
- `orion/harness/tests/test_harness_progress_frames_not_grammar_steps.py`: new.

## Metric quality gate (harness_step_load, avg_step_chars_pressure)

1. **Provenance:** `runner.py` loop -> `HarnessGrammarCollector.record_step_*` -> `grammar_extract.py:146` (`harness_step_load`), `:153` (`avg_step_chars`).
2. **Independence:** unchanged; the two channels already share the frame count and this change moves both consistently.
3. **Theory anchor:** `harness_step_load` is documented as a compute/thermal/power *cost proxy*. Heartbeat frames cost ~nothing, so counting them was measuring stream chatter, not work.
4. **Live-data check (48 h, 294 harness runs, same formula replayed):** mean `harness_step_load` 0.75 -> 0.72; runs pinned at 1.0: 16% -> 12%. Still varies; not degenerate.
5. **Existing mechanism:** same fix shape already used for `thinking_tokens` (`fcc_motor.py`, 3bf2a697e).
6. **Reversibility:** a two-condition revert; no schema/manifest/training default touched. Definition-drift gate: 685 definitions, 0 changed (formula code unchanged; only its input population shrinks). Anomaly model v3 ignores these FCC-motor channels (README, 2026-07-24).
Note: the saturation constant (60) was already flagged "not calibrated"; not recalibrated here.

## Tests run

```text
orion/harness/tests/test_harness_progress_frames_not_grammar_steps.py: 3 passed
  (runner test verified to FAIL without the fix)
pytest orion/harness: 397 passed
scripts/check_definition_drift.py --gate: PASS (0 changed); check_metric_dead_wiring: rc 0
```

## Evals run

None (no harness eval for grammar volume). Live verification after deploy: see below.

## Docker/build/smoke checks

Not run yet. UNVERIFIED live: after `orion-harness-governor` restart, largest new `harness_motor` traces should show no `Step started: ... tool_progress` atoms.

## Review findings fixed

- Finding (low): after a run that dies on trailing progress frames, `record_step_failed` used the count of ALL frames as its order, so the failure atom had no started atom to link to.
  - Fix: failure atom now uses the order of the last frame actually recorded (`last_recorded_step_order`).
  - Evidence: `test_error_after_trailing_progress_frames_links_failure_to_last_real_step` (fails with the old order).
- Reviewer noted, not changed: `record_result_assembled(step_count=...)` still carries the all-frames count; nothing reads it.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform-harness-grammar-skip-progress-frames && scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Low: `harness_step_load` / `avg_step_chars_pressure` values shift (avg -0.03 / more runs below saturation). Intentional; documented above.
- Low: atom `order` values now have gaps (frame index is preserved). `temporal_successor` chaining is unaffected (tested).
- Not done: a source-level filter in `fcc_motor.py` (would also change budget/receipts/step_count seen by Hub and durable runs).

## PR link

(see PR)
