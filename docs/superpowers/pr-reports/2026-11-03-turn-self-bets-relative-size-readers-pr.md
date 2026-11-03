## Summary

- PR A of 2 for the approved relative-size proposal (spec `docs/superpowers/specs/2026-11-02-turn-self-bets-relative-size-design.md`, supersedes proposal PR #3035).
- The bet schema accepts `turn.self_prediction.v2` beside v1. v2 holds four probabilities: asks a question, fetches a page, **more steps than usual**, **longer reply than usual**.
- orion-prediction settles the two new bets. A bet is a hit when the turn's true size is above the caller's median over the 7 days **strictly before** the bet. Stance never sees that median.
- One new nullable column, `prediction_turn_self_bet.reference`.
- Nothing writes v2 yet. The models are `extra="forbid"`, so every reader ships first. PR B (the prompt and the producer) follows once this is deployed.

## Outcome moved

The step and reply-length range bets were right 1–8% of the time against a stated 80%, and nothing could be learned from them. A replay of the new bet on 2,250 stored turns, using only earlier turns for each, gave AUC 0.59–0.84 on 9 of 10 caller × faculty cells (0.50 is chance; shuffle test p < 0.02). World-pulse steps gave 0.52 (no signal). This PR makes those bets gradeable live.

## Current architecture

- Stance writes v1 bets (`TurnSelfPredictionV1`): two probabilities and two 80% ranges.
- A3 (#2976) settles four faculties per turn into `prediction_turn_self_bet` and grades them at `GET /turn-self-bets/live`.

## Architecture touched

- Shared schema: `orion/schemas/thought.py` (V2 and the union), `registry.py`.
- orion-prediction: settler, store, migration.
- No bus channel change. No producer change.

## Files changed

- `orion/schemas/thought.py`:
  - `TurnSelfPredictionV2`;
  - the `TurnSelfPrediction` left-to-right union (an unversioned bet is v1; v2 must name itself);
  - used by `ThoughtEventV1` and `ThoughtDecisionRecordV1`.
- `orion/schemas/registry.py`: registers V2.
- `orion/schemas/tests/test_turn_self_prediction_versions.py`: new. Covers both versions round-tripping, an unversioned bet staying v1, and mixed or out-of-range bets being refused.
- `services/orion-prediction/app/turn_self_bets.py`:
  - two relative faculties;
  - `SizeHistory`, the reference median over earlier turns only;
  - `observed` is now recorded for every executed turn, even when nothing was stated, so the size history carries across the version switch;
  - the void reason `ungradeable_no_reference` (fewer than 20 earlier turns).
- `services/orion-prediction/app/store.py`: the `reference` column, `turn_size_history()`, and the `check_schema` entry.
- `services/orion-prediction/tests/test_turn_self_bets.py`: 5 new tests, including "a turn never counts in its own reference".
- `services/orion-sql-db/manual_migration_prediction_turn_self_bet_reference_v1.sql`: new, additive, `REQUIRED-BY: orion-prediction`.
- `docs/superpowers/specs/2026-11-02-turn-self-bets-relative-size-design.md`: the approved spec, with the as-built note.

## Schema / bus / API changes

- Added:
  - `TurnSelfPredictionV2` (`turn.self_prediction.v2`);
  - faculties `turn.self.more_steps_than_usual` and `turn.self.longer_reply_than_usual`;
  - column `reference`.
- Removed: none (v1 still parses and settles).
- Behavior changed: `observed` is now filled on `ungradeable_no_stated` rows of executed turns. Grading still excludes every void row.
- Compatibility notes: consumer-first by design. Merge and deploy this PR before PR B.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no change needed
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: no env change
- skipped keys requiring operator action: none

## Tests run

```text
services/orion-prediction: pytest tests/test_turn_self_bets.py -q          -> 31 passed
orion/schemas/tests -q                                                      -> 78 passed
orion/schemas/tests/test_turn_self_prediction_versions.py + stance + motor-leak tests -> 39 passed
services/orion-sql-writer: pytest tests/test_thought_decision_sql_shape.py -> 13 passed
scripts/check_migration_required_by.py                                      -> OK
git diff --check                                                            -> clean
```

## Evals run

```text
Leak-free backtest over 2,250 stored v1 bets (spec, metric gate 4c): AUC per caller x faculty,
early/late halves, 300-shuffle p-values. Read-only.
```

## Docker/build/smoke checks

```text
apply_post_pull_migrations.py --dry-run: NOT APPLIED, would apply (single transaction) -> auto-apply on pull
ALTER TABLE ... ADD COLUMN reference inside BEGIN/ROLLBACK on the live DB: ok (7,988 rows, all NULL)
```

## Review findings fixed

Code review in a subagent: **nothing material.** It confirmed no leak into a bet's own reference
(strict `<` on `placed_at`, history added after the reference is taken), no v1/v2 mis-parse, no
consumer broken by `observed` on void rows, aware-vs-aware timestamps, and that the migration
auto-applies before the boot schema check.

- Finding (low): `SizeHistory.reference` scanned the whole caller history per bet (worst case 5.5 s
  on a 2000-bet backlog).
  - Fix: keep each list sorted (`bisect.insort`) and slice the window with `bisect`.
  - Evidence: `test_reference_uses_only_earlier_turns_inside_the_window` still passes (31 passed).
- Not changed (low): v2 rows store `coverage` 0.8. The column is NOT NULL and no relative faculty
  reads it.
- Not changed (low): the reference depends on settle timing (a turn still running is not yet
  history). The reference is stored per row, and the effect on a median of 20 or more turns is negligible.
- Carried to PR B: the normalizer must emit v2 (done there), and every image that bakes
  `orion/schemas/thought.py` must rebuild before v2 is written (the reason for the PR order).

## Restart required

```text
Post-pull rebuild handles it: the migration auto-applies before orion-prediction rebuilds, and every
service importing orion/schemas/thought.py rebuilds with the v2-aware schema.
```

## Risks / concerns

- Severity: low
  - Concern: the reference counts only turns already settled. A turn still running when a later bet settles is missing from that bet's reference.
  - Mitigation: the live settle lag is minutes; the median over at least 20 turns barely moves.
- Severity: low
  - Concern: chat and self turns sit exactly at their median 55–57% of the time, so "bigger than usual" happens only 30–35% of the time there.
  - Mitigation: skill is measured against each caller's own rate, not against 50%.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
