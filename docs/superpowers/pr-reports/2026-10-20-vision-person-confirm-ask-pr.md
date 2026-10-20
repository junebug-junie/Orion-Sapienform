## Summary

- Orion now asks **"Is this <name>?"** about its strongest "unsure" person guesses. The card shows the cluster's pictures with three choices: **Yes, it is <name>** / **No** / **Can't tell**.
- **Yes** names the cluster through the usual typed-name path. On its next cycle the album takes the cluster's crops as that person, and orion-substrate-runtime writes the person entity.
- **No** marks every crop of the cluster "not <name>" (kept across album reconciliation) and withdraws that guess. Orion stops guessing that name for that cluster.
- **Can't tell** dismisses the card. That person-cluster pair is never asked again.
- Same rules as the other openers: no picture, no ask; a skipped candidate never uses up a slot; one open question per individual. It shares `ORION_ASK_DAILY_CAP` with the slow "who is this?" rule and uses what that rule leaves.
- Spec PR #2767, Phase 1, second half. The "probably X" display and its "not X" button are deferred: no accept can happen at today's street size (see PR #2797).

## Outcome moved

This is how the albums grow. On 2026-10-20 only 4 named clusters had a visit on another day to learn from, so nobody could earn a "probably" bar. A "yes" on a different day's visit adds exactly the cross-day example that is missing, and a "no" adds a hard negative. Live: 253 unsure guesses across 7 people; 21 have a picture and at least 2 sightings (askable).

## Current architecture

PR #2797 writes one guess per unnamed cluster into `vision_person_attribution` (all `unsure` today), and nothing reads them. Asks: face rule, "same person?", the slow 10-sightings "who is this?" rule, story-card names. The Hub enforces yes/no only on "same person?" cards.

## Architecture touched

- orion-sql-writer: a new opener and answer path in `vision_individuals.py`.
- orion-hub: answer route (the yes/no rule now covers a list of kinds) and the card in `vision-asks.js`.
- orion-substrate-runtime: the answered listener names the individual on a "yes".
- Contract: a new `source_kind` value, documented in `orion/schemas/ask.py`. No schema field change, no new channel, no migration, no env change.

## Files changed

- `services/orion-sql-writer/app/vision_individuals.py`: `SOURCE_KIND_PERSON_CONFIRM`, `_open_person_confirm_asks`, `_apply_person_confirm`, `person_confirm_question`; `ASKED_TODAY_SQL` counts the new kind.
- `services/orion-sql-writer/tests/test_vision_person_model.py`: 9 real-Postgres tests and 2 pure tests (opens with a picture only; shares the cap; yes names and the album takes it; no rules the name out, marks only the guessed crops and survives reconciliation; can't tell is never re-asked; seen-once is not asked; a name elsewhere closes the card; an open card keeps the slow rule off; wording; casing).
- `services/orion-sql-writer/app/vision_person_model.py`: names differing only in casing are one person in the album and in blocks.
- `services/orion-sql-writer/tests/test_vision_individuals.py`: budget SQL expectation updated.
- `services/orion-sql-writer/README.md`: step 4 of person learning.
- `services/orion-hub/scripts/ask_routes.py`: `PERSON_CONFIRM_KIND`, `YES_NO_KINDS`; `CLOSE_ASK_SQL` `$5` is now a text array.
- `services/orion-hub/static/js/vision-asks.js`: `renderPersonConfirm`, `personConfirmName`, `personConfirmSubmission`.
- `services/orion-hub/tests/test_ask_routes.py`, `tests/test_vision_asks_js.py` (new), `tests/test_orion_is_asking_browser_smoke.py`: route, node and browser coverage.
- `services/orion-substrate-runtime/app/ask_answered_listener.py` and its tests: a "yes" writes the entity; "no" or a dismissal writes nothing.
- `orion/schemas/ask.py`: comment documenting the new kind.

## Schema / bus / API changes

- Added: `source_kind="vision_person_confirm"` on `orion_ask` rows and on `orion:ask:answered` events (`source_ref="<individual_id>|<name>"`).
- Removed: none
- Renamed: none
- Behavior changed: `ASKED_TODAY_SQL` counts the new kind; the Hub refuses anything but yes/no on the new kind (422 `answer_must_be_yes_or_no`).
- Compatibility notes: **deploy orion-hub before orion-sql-writer.** An old Hub would render the new card as a free-text "who is this?" card, and a typed name would be stored but discarded by sql-writer (it is not yes/no). orion-substrate-runtime can go in any order: an old one ignores the kind, so a "yes" then reaches the album but not the substrate entity.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced: not needed
- skipped keys requiring operator action: none

## Tests run

```text
sql-writer  ORION_TEST_PG_URI=<throwaway pg16, 7 migrations> pytest tests/test_vision_person_model.py -> 37 passed
            pytest tests -k "vision or ask or settings" (PG) -> 407 passed; 22 errors = motion/re-anchor/story-strip
            migrations not loaded (same on main)
            mutations caught: blank card allowed (4 fail), own cap instead of shared, yes ignored,
            "no" stored under a reconciled source, no sighting floor, card not closed on a name,
            slow rule ignores an open card, "no" marks every crop, casing splits a person
hub         pytest tests -k ask -> 92 passed, 13 skipped; tests/test_vision_asks_js.py (node) -> 2 passed
            test_orion_is_asking_browser_smoke.py in real headless Chromium -> 1 passed (new card: 1 picture,
            3 choices, Send refused until a choice, "No" posts {"answer": "no"})
substrate   pytest tests/test_ask_answered_listener.py -> 26 passed; full suite: same 14 failed / 9 errors as
            main (local environment), +4 passing
```

## Evals run

```text
No new eval: an ask opener has no quality score beyond what its tests pin. The effect is measured by
evals/run_person_recognition_eval.py (PR #2797): scorable clusters (with another-day visit) and accept bars
should rise as "yes" answers land. Baseline 2026-10-20: 4 scorable clusters, 0 accept bars.
```

## Docker/build/smoke checks

```text
Not built (no dependency change). Live read-only: 253 unsure guesses, 56 with a picture; top scores 0.94 / 0.91 / 0.90.
After deploy: a "[ORION_ASK] opened ... source_ref=<id>|<name>" log line from sql-writer and the card in the Hub.
```

## Review findings fixed

- Finding: guesses rank by raw similarity, so one-off passers-by (1 sighting, 1 crop) took the daily slots; the question read "1 times over 1 days".
  - Fix: only individuals seen at least twice (`PERSON_CONFIRM_MIN_SIGHTINGS`); plural-correct wording.
  - Evidence: `test_someone_seen_once_is_not_asked_about`, `test_is_this_x_question_counts_in_plain_english`; mutation "no floor" fails. Live: 21 askable candidates remain.
- Finding: "one open question per individual" held only one way. A "who is this?" card could open next to an open "Is this X?" card, and a name given elsewhere left the "Is this X?" card open. A later "yes" would then relabel the substrate entity against Postgres.
  - Fix: the face and slow openers skip an individual with an open "Is this X?" card; `_close_who_is_this` also closes open "Is this X?" cards (answered, no answer, applied) when a name lands.
  - Evidence: `test_a_name_given_elsewhere_closes_the_open_is_this_x_card`, `test_an_open_is_this_x_card_keeps_the_slow_rule_off_that_individual`; both mutations fail.
- Finding: "no" marked every crop the individual ever had, permanently, though she saw three pictures.
  - Fix: only the newest 20 crops (the ones the guess was made on).
  - Evidence: `test_no_marks_only_the_crops_orion_guessed_on`; mutation fails.
- Finding: names were compared by exact spelling, so a casing flip lost a "no" and allowed a re-ask.
  - Fix: `build_album` merges casings into one person; blocks are casefolded; "already asked" compares lower-cased refs.
  - Evidence: `test_a_not_x_written_under_another_casing_still_counts`; mutation fails.
- Finding: some unusable answers were dropped silently.
  - Fix: logged (`person_confirm_answer_unusable`, `person_confirm_yes_not_applied`, `person_confirm_no_not_applied`).
- Finding: deploy order. Already in Compatibility notes and Restart required (Hub first).

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d
scripts/safe_docker_build.sh orion-substrate-runtime up -d
scripts/safe_docker_build.sh orion-sql-writer up -d
```

In that order: the Hub first.

## Risks / concerns

- Severity: low
  - Concern: with no bars yet, guesses rank by raw body similarity; some asks will be "no".
  - Mitigation: a "no" is useful too (a hard negative that also stops that guess); asks are capped per day, need 2+ sightings, and each pair is asked once.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2807

🤖 Generated with [Claude Code](https://claude.com/claude-code)
