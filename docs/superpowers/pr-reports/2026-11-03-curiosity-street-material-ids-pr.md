# fix(curiosity): record street material in offer material_ids

## Summary

- Curiosity's kickoff prompt shows walkway captions and a day card. Before this patch nothing recorded that. Now each run's offer row lists them.
- New `kickoff_material_ids(material)` in `orion/curiosity/kickoff_prompt.py`. It reads the same lists the material section renders, so only shown material is recorded.
- New ids: `street_caption:<vision_events.event_id>` for each caption shown, and `street_day:<stream_id>:<day>@<computed_at>` for the day card.
- The hub's `_record_offer` now calls that function. The crystallization and relation ids are unchanged and stay first.
- `StreetDayCard` gains `computed_at`. The newest day row is rewritten all day, so this says which version Orion read.

## Outcome moved

Before: "walkway captions or the day card reached a real investigation" was UNVERIFIED and could not be checked. The walkway was offered in 23 of 49 offer decisions in 7 days, but the stored ids held no trace of it. After: a SQL query on `curiosity_offer_decisions.material_ids` shows which runs saw which street material.

## Current architecture

`CuriosityInvestigation.tick` reads `StudyMaterial`. It includes street captions (deduped, capped at 3, from the last 6 hours, in `build_street_caption_cards`) and the newest `vision_rollup` day row. Then `_record_offer` writes one `curiosity_offer_decisions` row with `material_ids` (jsonb list of strings). Then `build_kickoff_prompt` renders the same `material` object. The material section is always rendered. Readers of `material_ids`: `orion/personality/choices.py` (passes them through as `situation_refs`, capped at 50), `services/orion-durable-runs/app/personality_sources.py`, `scripts/analysis/measure_personality_choices.py`. None of them parse the prefix.

## Architecture touched

- `orion/curiosity` (shared library): id builder and the day card field.
- `orion-hub` curiosity loop: calls the builder.
- No bus, schema registry, env, or table change. `material_ids` is untyped jsonb with no Pydantic schema, so no registry entry applies.

## Files changed

- `orion/curiosity/kickoff_prompt.py`: `kickoff_material_ids()` beside the street sections it mirrors.
- `orion/curiosity/study_material.py`: `StreetDayCard.computed_at`, filled from the `computed_at` column the SQL already selects.
- `services/orion-hub/scripts/curiosity_investigation.py`: `_record_offer` uses `kickoff_material_ids`.
- `services/orion-hub/tests/test_curiosity_street_material_ids.py`: regression tests.
- `docs/superpowers/pr-reports/2026-11-03-curiosity-street-material-ids-pr.md`: this report.

## Schema / bus / API changes

- Added: two new id prefixes inside the existing `material_ids` jsonb list: `street_caption:` and `street_day:`.
- Removed: none.
- Renamed: none.
- Behavior changed: offer rows for runs that saw street material carry more ids.
- Compatibility notes: same column and same list-of-strings shape. Personality `situation_refs` gets up to 4 more refs per run. That is about 24 at the default sample sizes, under the cap of 50.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
services/orion-hub: pytest tests/test_curiosity_street_material_ids.py -> 3 passed
  With main's curiosity_investigation.py swapped in: the real-tick test FAILS (no street ids recorded)
services/orion-hub: pytest tests -k "curiosity or street" -> 587 passed, 4 skipped, 0 failed
pytest orion/curiosity orion/personality -> 346 passed, 23 skipped
services/orion-durable-runs: pytest tests/test_personality_loop.py tests/test_personality_postgres.py -> 34 passed, 5 skipped
```

## Evals run

```text
None. This change only adds observability ids and does not change behavior, so no eval applies.
The post-deploy SQL check below is the live proof.
```

## Docker/build/smoke checks

```text
Not run. No dependency or compose change. Rebuilding orion-hub is enough.
```

## Post-deploy SQL check

Run this after the first investigation that follows the deploy:

```sql
SELECT run_id, decided_at,
       (SELECT count(*) FROM jsonb_array_elements_text(material_ids) m WHERE m LIKE 'street_caption:%') AS captions,
       (SELECT string_agg(m, ',') FROM jsonb_array_elements_text(material_ids) m WHERE m LIKE 'street_day:%') AS day_card
FROM curiosity_offer_decisions
WHERE decided_at > now() - interval '2 days'
ORDER BY decided_at DESC
LIMIT 20;
```

What to expect: the runs after the deploy that had walkway material show `captions` from 1 to 3 and/or a `day_card` value. The runs before the deploy show 0 and NULL. To trace a caption back to its source: `SELECT event_id, narrative FROM vision_events WHERE event_id = '<id after street_caption:>'`.

## Review findings fixed

- Finding: the `street_day` id did not say which version of the row was read. The newest day row is rewritten all day (`complete=false`).
  - Fix: `StreetDayCard.computed_at` was added, and the id now ends with `@<computed_at>`.
  - Evidence: the real-tick test asserts `street_day:walkway:2026-11-02@<computed_at>`.
- Finding: test 2 checked only the id prefixes.
  - Fix: test 2 now asserts the exact old id list.
  - Evidence: `test_without_street_material_the_ids_are_unchanged`.
- Finding: a caption with an empty event_id is shown but not recorded. Unresolved percepts are also not recorded. A downstream cap would drop the street ids first.
  - Fix: all three are documented in the `kickoff_material_ids` docstring. The test is renamed so it no longer says "exactly". No material issues were found.
  - Evidence: review report. It also found no path where a recorded id points at material that was not in the prompt.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

(Run from the clean primary main checkout after the merge.)

## Risks / concerns

- Severity: low
- Concern: the offer row is written before the turn is dispatched. A cancelled run still lists its material. This was already true for crystallizations.
- Mitigation: join `curiosity_run_outcomes` when you need runs that actually ran.
- Severity: low
- Concern: unresolved percepts are still not recorded.
- Mitigation: follow-up. Their cards need a stable id first.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3069

🤖 Generated with [Claude Code](https://claude.com/claude-code)
