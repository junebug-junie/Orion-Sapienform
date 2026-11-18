## Summary

- The dream asks about fragments that happened near each other in different parts of Orion, not
  about the most unrelated ones. Pairs must come from different organs, and their times must fall
  within 2 hours of each other.
- Every replay fragment now carries its time as data: `occurred_at`, plus `occurred_until` for arcs,
  which span a period.
- There is no fallback to distant pairs. A sleep with no coincidences asks fewer questions.
- The random control arm is unchanged, so the dream-vs-chance comparison still holds.
- Juniper decided this on 2026-11-17 (#3664). Fixes #3664.

## Outcome moved

- **Before:** the dream arm picked the pair with the highest weight × tag distance. That means
  unrelated fragments, often hours apart, and an honest judge declines them (#3412).
- **Live check, read-only, last 6 completed sleep windows:** the new rule finds 3 pairs in every
  window, 1–56 minutes apart, and every replay fragment had a time. Earlier single-sleep test on
  real content: max distance gave 0/3 linked, coincidence gave 1/4.
- **The real test is the 14-day readout** of link rate and claim checkability, dream against
  control, from `dream_cycle.cycle_json->'pair_answers'` (stored since #3636).

## Current architecture

`recombine.dream_pairs` scored every replay pair by `weight_a * weight_b * (1 - tag Jaccard) + 0.1`
when the sources differed, then picked greedily. Fragments had no time field; a time appeared only
inside the text (#3665).

## Architecture touched

orion-dream only, plus the shared `ReplayItemV1` model. One new env key.

## Files changed

- `orion/schemas/dream_cycle.py`: `ReplayItemV1.occurred_at`, `occurred_until` (optional, aware).
- `services/orion-dream/app/replay.py`: `_as_utc()`; every builder sets the time (arc: began to ended;
  walkway: scored_at).
- `services/orion-dream/app/cycle_store.py`: resonance and crystallization queries return `occurred_at`.
- `services/orion-dream/app/recombine.py`: `time_gap_minutes()`; coincidence `dream_pairs`; removed
  `jaccard`, `pair_score`, `CROSS_SOURCE_BONUS` (no other callers).
- `services/orion-dream/app/cycle.py`: passes the window.
- `services/orion-dream/app/settings.py`, `.env_example`, `docker-compose.yml`: `DREAM_PAIR_WINDOW_MINUTES=120`.
- `services/orion-dream/tests/test_dream_cycle_v2.py`: fixtures carry times. New tests: coincidence
  only, same organ never paired, untimed never paired, no fallback, weight then nearness, an arc span
  overlapping a point, every builder sets a time.
- `services/orion-dream/evals/test_dream_cycle_eval.py`: the fake loader returns `occurred_at`, as
  the real queries do.

## Schema / bus / API changes

- Added: `ReplayItemV1.occurred_at`, `ReplayItemV1.occurred_until` (defaulted None; stored in `cycle_json`).
- Removed: none from contracts. Renamed: none.
- Behavior changed: the dream arm's pair choice (above).
- Compatibility notes: old `cycle_json` rows load with no times.

## Env/config changes

- Added keys: `DREAM_PAIR_WINDOW_MINUTES=120` (orion-dream).
- `.env_example` updated: yes. Compose passes it with the same default.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys` (the default
  key list doesn't include dream keys). `services/orion-dream/.env` line 143. A rerun adds 0 keys.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-dream tests+evals: 411 passed, 9 skipped (after review fixes)
git diff --check: clean
```

## Evals run

`services/orion-dream/evals` ran in the suite above. The behavior eval is the live pair-count check
(above) plus the 14-day readout after deploy.

## Docker/build/smoke checks

Live read-only check of the new rule against the last 6 real sleep windows (source queries + replay
selection + `dream_pairs`): 3 pairs per window, all within 56 minutes. No Docker build (pure Python).

## Review findings fixed

Review by a subagent (by hand; the skill was not loadable there). It found nothing material. It
confirmed that:

- every builder sets a time;
- `keyed_candidates` keeps the first row's time;
- no outside reader re-validates `ReplayItemV1`;
- the removed helpers have no callers;
- a sleep with zero dream pairs still completes and closes its window.

Fixed:

- Finding: a long arc overlaps everything inside it (gap 0), so a night-long body arc would take
  every pair.
  - Fix: a span longer than the window counts only by its end (`time_gap_minutes(...,
    max_span_minutes=window)`).
  - Evidence: `test_a_night_long_arc_does_not_take_every_pair`, plus span tests.
- Finding: the walkway time was the grading time, not the street event.
  - Fix: the walkway fragment spans `window_start` to `scored_at`.
- Finding: a stale doc line said "(salient, distant)". Fixed in `orion/schemas/dream_cycle.py`.
- Not changed: a recurring metacog problem keeps one time (the newest critical row). That is
  consistent with "one thing, one fragment".

Live recheck after the fixes: the same 6 windows still give 3 pairs each.

## Restart required

Merge, then the post-pull rebuild rebuilds orion-dream. Check after the next sleep:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select started_at, hypothesis_count, no_link_count, jsonb_pretty(cycle_json->'pair_answers') from dream_cycle order by started_at desc limit 1"
```

## Risks / concerns

- Severity: medium. Concern: nearly every coincidence pair is metacog with compaction-request,
  because those two organs fill most replay slots. Mitigation: arcs and walkway keep their reserved
  slots; the 14-day readout shows whether the mix is too narrow.
- Severity: low. Concern: a sleep with no coincidences asks no dream-arm questions. That is by
  design: no fallback to pairs we know get declined.
- Severity: low. Concern: the dream-vs-control scorecard now compares a different dream rule; rows
  before 2026-11-17 are a different experiment (and before 11-13 an always-yes judge).
- Issues opened: none. Fixes #3664. Refs #3412.
