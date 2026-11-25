# PR report: Juniper's newest answer on a picture wins

Arc: the walkway "who is this?" card work. PR #3847 made confirmed pictures
teach the person model. Its review found that an older "someone else" answer
beat a newer "yes, that's her" on the same picture forever (#3854). Juniper
approved the fix on 2026-11-25: her newest answer wins.

## Summary

- When Juniper answers the same picture twice for the same person, only the
  newer answer counts. "Someone else" then "yes, that's her" means it is her.
  "Yes" then "someone else" means it is not her.
- "Not sure" never replaces a yes or a no.
- One rule, in one place (`orion/person_model/picture_verdicts.py`), used by
  the three readers that act on who a picture shows: the recognition model's
  examples, the split that moves "someone else" pictures out of a group, and
  the People panel pictures.
- A split that already ran is not undone automatically.

## Outcome moved

The live case from 2026-11-24 now reads correctly. Juniper said "someone
else" on a picture of herself at 05:28:13 and corrected it to "yes" at
05:28:36. Before: the model held a "not Juniper" example for that picture. It
also never guessed Juniper for the group that picture now sits in. After: the
picture is a "Juniper" example and the block is gone.

Live dry run (read-only, 2026-11-25): **1 picture flips**. One "not Juniper"
example is revoked and one "Juniper" example is added. No other example
changes. That is the only picture with both answers today.

## Current architecture

Picture answers live in `vision_ask_picture_verdict`, one row per picture per
card. Readers before this patch:

- Recognition model (`services/orion-sql-writer/app/vision_person_model.py`):
  every `different` row was a -1 example, whatever she said later.
- Split (`services/orion-sql-writer/app/vision_split.py`): every `different`
  row on a card moved its picture out of the group. In the live case the split
  ran at 05:28:51, 15 seconds after her correction.
- People panel (`orion/person_model/pictures.py`): any `different` row on a
  picture hid it.

## Architecture touched

- New: `orion/person_model/picture_verdicts.py`, `overridden_sql(alias)`. It
  is true when a newer opposite firm answer exists for the same picture (same
  `crop_id` or same `image_ref`) and the same person (same asked group, or a
  group with the same current name).
- No schema, bus, API, or env change.

## Files changed

- `orion/person_model/picture_verdicts.py`: the rule.
- `services/orion-sql-writer/app/vision_person_model.py`: `picture_different`
  and `picture_same` queries skip answers she replaced.
- `services/orion-sql-writer/app/vision_split.py`: a "someone else" picture
  she later confirmed is treated as confirmed (it stays). The rest of the
  card still splits.
- `orion/person_model/pictures.py`: the panel hides a picture only when her
  "someone else" was not replaced.
- `services/orion-sql-writer/README.md`: the rule.
- Tests: `test_vision_person_model.py`, `test_vision_split.py`,
  `test_person_record_pg.py`.

## Readers left alone, and why

- `services/orion-sql-writer/app/person_record.py` (card reasons). It decides
  whether a typed "why" on one card is about one person or two. That is a
  fact about the card, not about who a picture shows.
- `orion/self_monitor/adapters.py`. It grades one ask by what she said on that
  card. A later card is graded on its own.
- `services/orion-sql-writer/evals/run_vision_split_replay_eval.py`. It
  replays the split on history and asserts fixed numbers from 2026-10-16,
  which no correction touches. Its calibration pairs are per card.
- `services/orion-sql-writer/evals/run_osnet_vs_siglip_eval.py`. It builds
  same/different pairs within each card. The correction is a separate card
  and contributes its own pair.

## Schema / bus / API changes

- Added / Removed / Renamed: none.
- Behavior changed: a replaced picture answer no longer counts in the
  recognition model, the split, or the People panel.
- Compatibility notes: the panel SQL is shared by orion-sql-writer and the
  Hub. The fragment has no parameters, so both drivers take it unchanged.

## Env/config changes

- None. No `.env_example` changed; nothing to sync.

## Tests run

```text
orion-sql-writer suite: 1357 passed, 284 skipped
Throwaway Postgres 16 with the walkway, split, reanchor, person-model and
person-record migrations:
  test_vision_person_model, test_vision_split, test_person_record_pg,
  test_vision_reanchor, test_vision_confirm_asks -> 216 passed
Mutation checks: removing the rule from the model, from the split's
per-picture step, or from the panel each makes a new test fail.
orion/person_model/tests: the only failures are linkage tests that need
rapidfuzz, which is missing from the local venv (same on main).
```

New tests:
- The 22-second shape: "someone else" then "yes": +1, no -1. The reverse
  order gives -1, no +1. A later "not sure" changes nothing.
- The split: a corrected picture never splits. On one card, the corrected
  picture stays while the rest splits. No, then yes, then no again: it splits.
- The panel: the corrected picture shows. The reverse order hides it.

## Evals run

```text
No eval harness covers picture answers directly. The live dry run above
(read-only) is the check: 1 picture flips; exactly the expected -1 -> +1.
```

## Docker/build/smoke checks

```text
No build: Python/SQL only, no new dependency. orion-sql-writer and orion-hub
import orion/person_model already. Live effect is UNVERIFIED until deploy.
Proof: after the next person-model tick, vision_person_exemplar has a revoked
picture_different row and an active picture_same row for Juniper on crop
cropobs:940f3184-...:0.
```

## Review findings fixed

REVIEW_SECTION

## Restart required

The post-pull rebuild redeploys orion-sql-writer and orion-hub. By hand, from
the clean primary main checkout:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium. Concern: the live split from 2026-11-24
  (`split-46796799f8ef4a30839e95c679019f9f`) moved the picture of Juniper she
  corrected into a new group, `9268573a`. It also wrote a "different" link
  between that group and her group. This patch does not undo it. Mitigation:
  `scripts/undo_vision_split.py` reverses it by hand. That is Juniper's call.
- Severity: low. Concern: two answers with the exact same timestamp both stay,
  so the "someone else" answer still wins. Mitigation: answers come from
  separate clicks, seconds apart.
- Issues: Fixes #3854. Opened: none.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
