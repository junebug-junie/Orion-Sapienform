# feat(prediction): "does event A make event B more likely?" (event_pair)

## Summary

- New endpoint `POST /effects/event-pair` in orion-prediction. It answers one question: after event A, does event B happen more often within a lag window than it does in general?
- The answer is always labelled `observational_effect`. It is a correlation with a lag attached, never an experiment.
- The p-value comes from placebos: move all of A by one random offset many times and count how often the shifted ratio beats the real one.
- The default placebo keeps each A at its time of day (whole-day shifts, plus or minus 1 hour). A free shift gave false positives in 38 % of seeds when two independent streams both happen only in daytime. Street events follow the day, so this matters.
- Too little data gives a structured refusal (`422`, `error: refused`, with counts), never a made-up answer.
- Each answer is stored in the existing `prediction_effect_estimate` table. No migration, no env keys.

Spec: `docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md` section 7 (PR #2803).

## Outcome moved

Orion had no way to ask "does X affect Y" across two event streams. Now it can, with a lag window, honest counts, and a calibrated p-value. Measured false-positive rate at alpha 0.05:

| null case | default (`same_time_of_day`) | `circular` |
|---|---|---|
| independent Poisson streams, 300 seeds | 13/300 (4.3 %) | 15/300 (5.0 %) |
| both daytime-only, independent, 120 seeds | 7/120 (5.8 %) | 45/120 (37.5 %) |

## Current architecture

orion-prediction had three effect levels: `conditional` (`/scenario`), `observational_effect` (interrupted time series over a numeric series, `/effects/estimate`), and `experimental_effect` (switchback). All of them work on numeric telemetry series. Nothing read event tables.

## Architecture touched

- `services/orion-prediction` only. It now reads `vision_events` (same `conjourney` database it already uses, user already has SELECT) in a read-only transaction with a 10 s statement timeout.
- Selector shape `{source, event_type, stream_id?}`. `source` accepts only `vision_events` today. Adding a table means one entry in `store.EVENT_SOURCES` and one value in `EventSelector.source`.

## Files changed

- `services/orion-prediction/app/event_pair.py`: the pure statistic, the two placebos, and the refusals.
- `services/orion-prediction/app/effects.py`: `Effects.event_pair` (read, refuse, store). Comment in `best_effect` explains why event_pair rows stay out of the pooled answer.
- `services/orion-prediction/app/store.py`: `event_times` (read-only, statement timeout, parameterised).
- `services/orion-prediction/app/main.py`: `POST /effects/event-pair`; 422 refusal; 504 on a timed-out read.
- `services/orion-prediction/tests/test_event_pair.py`: 22 tests.
- `services/orion-prediction/evals/run_event_pair_live_eval.py`: live pairs, read-only, both placebos.
- `services/orion-prediction/README.md`: endpoint contract and live numbers.

## Schema / bus / API changes

- Added: `POST /effects/event-pair`. Body: `a`, `b` (`{source="vision_events", event_type, stream_id?}`), `start`, `end` (<= 90 days, not in the future), `lag_min_sec` (default 0), `lag_max_sec`, `n_placebo` (99-9999, default 999), `seed` (default 0), `placebo` (`same_time_of_day` default, or `circular`). Returns the stored row (`effect` = rate ratio, `p_value`, `n` = A count, `details` with counts, both rates, covered seconds, placebo quantiles) plus a plain `summary`.
- Removed / renamed: none.
- Behavior changed: none for existing endpoints. Rows use `action = event_pair:<A>`, `series_id = <B>`, `ci_lo`/`ci_hi` NULL, so `GET /effects/{action}` never pools them.
- Compatibility notes: no migration. Bus untouched.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed (no template change). `check_env_template_parity.py`: PASS.
- skipped keys: none.

## Tests run

```text
docker run ... orion-prediction-prediction:latest python -m pytest tests -q
103 passed (22 new in test_event_pair.py)
```

Covered: union/clip math; B 30 s after A is significant (ratio > 20, p <= 0.002) under both placebos; a lag window that misses the delay finds nothing; null false-positive rate across 300 seeds for each placebo; bursty B; shared daily rhythm (the free shift is fooled, the default is not); part-day range regression (14.5 days); work budget; refusals with counts; overlapping selectors need lag_min > 0; refusal stores nothing; endpoint returns structured 422.

## Evals run

```text
POSTGRES_URI=<prediction .env> python evals/run_event_pair_live_eval.py --days 14   (2026-10-20, read-only)
doorbell_ring -> walkway person_presence [0,300]s      REFUSED: 0 doorbell_ring rows (n_b=223). Correct: no producer writes doorbell_ring yet.
doorbell person_presence -> walkway presence [0,300]s  n_a=12, 5 seen vs 0.19 expected, ratio 26.8, p=0.02 (tod) / 0.01 (circular)
walkway presence -> walkway camera_caption [0,60]s     n_a=223, 25 seen vs 0.77 expected, ratio 32.4, p=0.006 / 0.001 (sanity: captions follow presence)
room cam0 presence -> walkway attention_worthy [0,600]s n_a=1848, 12 seen vs 2.64 expected, ratio 4.5, p=0.001 / 0.005
```

The last pair was meant as a placebo. It comes out significant under both placebos, so time of day does not explain it. It may be a real path (someone leaves the room, then appears on the walkway), or a confound this method cannot see. This eval cannot tell which. It is not a clean negative control. The CI-safe negative controls are the synthetic null tests above.

## Docker/build/smoke checks

```text
Tests and the live eval ran inside the existing orion-prediction image with the worktree mounted.
No image rebuild and no deploy from this worktree.
```

## Review findings fixed

- Finding (must-fix): the default placebo wrapped A around a part-day range, which moved wrapped events to a different time of day (false-positive rate 15.8 % at 3.5 days).
  - Fix: `same_time_of_day` cuts the range to whole days before counting.
  - Evidence: `test_time_of_day_placebo_keeps_time_of_day_on_a_part_day_range` (14.5-day range, <= 9 %).
- Finding: A = `person_presence` (all streams) and B = `person_presence@walkway` passed the self-pair guard, so each A counted itself.
  - Fix: refuse `lag_min <= 0` whenever the two selectors can match the same rows.
  - Evidence: parametrised `test_overlapping_selectors_need_a_positive_lag_min`.
- Finding: 2-day ranges gave only 2 distinct day shifts.
  - Fix: `same_time_of_day` needs >= 7 whole days.
- Finding: placebo CPU had no limit.
  - Fix: refuse when `n_placebo x (n_a + n_b) > 5e7`. Evidence: `test_refuses_more_placebo_work_than_the_budget`.
- Nits: `isinstance` check on `psycopg.errors.QueryCanceled`; `best_effect` comment; matching numbers in comments.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-prediction up -d --build
```

## Risks / concerns

- Severity: low. Concern: deviation from the brief. The brief named the circular shift. The default is the time-of-day placebo because the circular one is badly miscalibrated on daily-rhythm data. `circular` stays available. Mitigation: both tested; README says when to use which.
- Severity: low. Concern: counts are rows, not visits. Bursts inflate the ratio, not the p-value. Mitigation: documented; the bursty-B null test holds.
- Severity: info. Concern: `doorbell_ring` has no producer yet, so the spec's first pair refuses until one exists.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2812

🤖 Generated with [Claude Code](https://claude.com/claude-code)
