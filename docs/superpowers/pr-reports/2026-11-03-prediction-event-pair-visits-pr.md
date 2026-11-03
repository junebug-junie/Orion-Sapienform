## Summary

- The "does X make Y more likely?" check (`POST /effects/event-pair`) can now count visits instead of rows. Each event selector takes `visit_gap_sec` (whole seconds, 0-3600, default 0 = rows as before).
- A visit ends when the next row comes more than the gap after the previous one. Only the first row of each visit counts.
- A stored answer's label gains `~visit<gap>s`, so answers by rows and by visits stay apart in `prediction_effect_estimate`.
- The live eval runs walkway -> room both ways.

## Outcome moved

The room camera (cam0) logs one person as a burst of rows (median row gap ~57 s). Walkway person -> room person, 1-15 min, last 13 days:

- Rows: 3.02x, p = 0.133 (looked like "maybe").
- 300 s visits: 1.16x, p = 0.894 (no effect).

The 3x was the bursts. Someone outside does not bring a person into the room.

## Current architecture

`Effects._event_seconds` (`app/effects.py`) read raw `vision_events` rows and passed every row to `event_pair_effect`. The README said "Counts are rows, not visits."

## Architecture touched

orion-prediction only: `app/event_pair.py` (`visits()`), `app/effects.py` (applies it, label suffix), `app/main.py` (`EventSelector.visit_gap_sec`; `stream_id` limit 40 -> 33 so the longest label still fits the 120-char columns).

## Files changed

- `services/orion-prediction/app/event_pair.py`: pure `visits()` helper.
- `services/orion-prediction/app/effects.py`: collapse rows to visits on read; `~visit<gap>s` label.
- `services/orion-prediction/app/main.py`: `visit_gap_sec` field; `stream_id` <= 33 chars.
- `services/orion-prediction/README.md`: documents the field.
- `services/orion-prediction/evals/run_event_pair_live_eval.py`: walkway -> cam0 as rows and as visits.
- `services/orion-prediction/tests/test_event_pair.py`: visits helper, bursty-B end to end, label length bound.

## Schema / bus / API changes

- Added: optional `visit_gap_sec` on each event-pair selector.
- Behavior changed: `stream_id` longer than 33 chars is refused (422). Live stream ids are `walkway`, `cam0`, `doorbell`.
- Compatibility notes: default 0 gives the old answer and the old label.

## Env/config changes

None.

## Tests run

```text
pytest services/orion-prediction/tests/test_event_pair.py -q   -> 25 passed
pytest services/orion-prediction/tests -q                      -> 8 failed (same 8 fail on main: backtest/causal/events Prophet tests, local env), rest pass
```

## Evals run

```text
POSTGRES_URI=... python evals/run_event_pair_live_eval.py --days 13  (read-only)
walkway -> cam0 rows:            n_b 957, 3.02x, p=0.133
walkway -> cam0 ~visit300s:      n_b 304, 1.16x, p=0.894
```

## Docker/build/smoke checks

Not run (no dependency or config change). Live endpoint check after deploy: POST with `"visit_gap_sec": 300` on the cam0 selector.

## Review findings fixed

- Finding: label suffix could push `action` past varchar(120) (129 chars at max bounds), failing the insert after the full placebo run.
  - Fix: `visit_gap_sec` is an int; `stream_id` limit 33.
  - Evidence: `test_longest_allowed_selector_label_fits_the_120_char_columns` (exactly 120).
- Finding: `:g` formatting gave different gaps the same label.
  - Fix: whole seconds only.
  - Evidence: same test refuses `300.5`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build
```

## Risks / concerns

- Severity: low. Concern: choosing the gap is a judgement call per stream. Mitigation: the label records it; rows stay the default.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3038

🤖 Generated with [Claude Code](https://claude.com/claude-code)
