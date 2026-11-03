## Summary

- When Juniper answers "are these the same person?", the saved answer now keeps the face similarity Orion wrote on the card when it asked. That number is Orion's guess, made before the answer.
- Before this, the answer row recomputed the similarity from the two faces' current averages at answer time. Those averages keep moving as new sightings arrive, so the saved number was not always the guess.
- The old recompute stays as the fallback when the card stored no face similarity.
- Handoff cards (`rule:handoff`) never use the stored number. Their similarity compares two walks, not the two people, so the answer row recomputes, as before.
- This is one of the source fixes for the self-monitor organ (PR #2827). It makes the face-match bet gradeable: guess first, answer second.

## Outcome moved

The face-match bet now holds the real pre-answer guess. Live data on 2026-11-03, for the 22 answers about the exact pair on the card:

- 22 of 22 have a similarity saved.
- 3 of 22 had drifted away from the guess stored on the card (up to 0.056 off, 0.005 on average).
- Nothing else would have caught this drift. A face average missing at answer time would also have saved NULL. Live, that has hit 0 of these 22, but the new test reproduces it.

What the original "10 of 16 filled" audit number really was: 7 of the 8 rows without a similarity are **other** pairs. These are pairs joined by a name Juniper typed on the card, and they carry the card's ask id. Orion made no face guess about those pairs, so NULL is the honest value for them. Copying the card's number onto them would be wrong, and this patch does not do it. A grader should grade only the row whose pair equals the card's `source_ref`. The 8th row is a handoff card with a body similarity only.

## Current architecture

`_open_asks` (services/orion-sql-writer/app/vision_individuals.py) opens a "same person?" card. It stores `face_cosine:<sim>` in `orion_ask.evidence_refs`. `_apply_pair_answer` writes the `vision_individual_pair_verdict` row. Until now it recomputed the cosine from `vision_individual.face_centroid` at answer time.

## Architecture touched

- orion-sql-writer only: `_apply_pair_answer` and a new pure helper `face_cosine_from_refs`.
- No schema, bus, or env change.

## Files changed

- `services/orion-sql-writer/app/vision_individuals.py`: `face_cosine_from_refs`. `_apply_pair_answer` reads the card's stored guess first. The log line names the source (`cosine_source=ask|recomputed|none`).
- `services/orion-sql-writer/tests/test_vision_confirm_asks.py`: a parser test, a Postgres regression test (a face average removed and another drifted between ask and answer), and a handoff test.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: `vision_individual_pair_verdict.face_cosine` for the asked pair = the value stored on the card (3 decimals). It was the answer-time recompute (full float).
- Compatibility notes: every reader prints or bands at 3 decimals or coarser (`scripts/report_vision_individuals.py`, `evals/run_vision_split_replay_eval.py`). No backfill of old rows.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
ORION_TEST_PG_URI=<throwaway pg, CI migrations applied> pytest services/orion-sql-writer/tests/test_vision_confirm_asks.py test_vision_split.py test_vision_handoff_pg.py test_vision_handoff.py
-> 125 passed
pytest services/orion-sql-writer/tests (no pg)  -> 1162 passed, 180 skipped
New pg regression test on the old _apply_pair_answer -> FAILED (as intended); on the new code -> passed
```

## Evals run

```text
None run. services/orion-sql-writer/evals/run_vision_split_replay_eval.py reads face_cosine, but it replays live data
and is not a gate for this change. Grading the face-match bet belongs to the self-monitor organ (PR #2827).
```

## Docker/build/smoke checks

```text
No image change beyond Python source. Live check after deploy (UNVERIFIED until then):
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select v.ask_id, v.face_cosine, a.evidence_refs from vision_individual_pair_verdict v join orion_ask a on a.ask_id=v.ask_id where a.source_kind='vision_same_individual' and a.source_ref=v.individual_a||'|'||v.individual_b order by v.created_at desc limit 5"
sql-writer log: [VISION_PAIR_VERDICT] ... cosine_source=ask
```

## Review findings fixed

- Finding (material): a face-channel handoff card also stores `face_cosine:`. That value compares two walks' crops, not the two people, so it would have mixed two different quantities into the calibration data.
  - Fix: `face_cosine_from_refs` returns None for any card with `rule:handoff`. The answer row then recomputes, as before. The comment and docstring are corrected.
  - Evidence: `test_handoff_pair_answer_still_recomputes_the_individuals_cosine` and the new parser case. Live, the one handoff card is body-channel, so no live row was affected.
- Nits (not changed): 3-decimal precision (no reader depends on the 4th decimal). The parser stops at the first malformed `face_cosine:` ref (only one is ever written).

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

## Risks / concerns

- Severity: low
- Concern: old rows keep their answer-time recompute. 3 of 22 differ slightly from the guess.
- Mitigation: a grader can read the guess straight from `orion_ask.evidence_refs` for old rows. No backfill was done on purpose: the task says never backfill after the outcome.

## PR link

(added after the PR is opened)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
