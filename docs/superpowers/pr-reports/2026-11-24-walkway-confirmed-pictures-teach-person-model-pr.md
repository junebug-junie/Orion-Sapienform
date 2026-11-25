# PR report: confirmed pictures teach the person model

Arc: the walkway "who is this?" card redesign
(`docs/superpowers/specs/2026-11-24-walkway-who-card-redesign.md`). Juniper
approved Missing question 6 on 2026-11-24: a "Yes, that's X" answer should
teach the camera's person-recognition model.

## Summary

- When Juniper says a picture is a person ("Yes, that's <name>" on a story card,
  or "same person" on an older multi-picture card), that one picture now
  becomes a training example for that name.
- This works even when the picture's group looks like two people. Before, such
  a group taught nothing at all. Now the pictures she confirmed still teach;
  the rest of the group still does not.
- A picture is never counted twice, never counted for a name she also said it
  is not, and is skipped when the group has been renamed since she answered.
- One switch turns it off (`VISION_PERSON_PICTURE_SAME_POSITIVES`, on by
  default). Off removes these examples on the next tick.
- A picture that no longer belongs to any group now stands as its own group,
  instead of being pooled with every other such picture.

## Outcome moved

The model learns from more of Juniper's answers. On live data today 8 new
positive examples are added (5,975 -> 5,983 album rows), spread over 4
people (3, 2, 2 and 1 pictures). 3 of the 8 come from today's story-card yes answers; 5
come from older multi-picture cards. Small today; it grows with every yes.

## Current architecture

The person model (`services/orion-sql-writer/app/vision_person_model.py`,
`desired_exemplars`) built positives only from whole named groups
(`cluster_label`, `pair_same`). A named group that looks like two people
(`split_cosine` below the same-sighting floor) was skipped entirely: 9 groups
live today. Picture verdicts only fed negatives (`picture_different`). The
`same` verdicts were recorded and then unused by recognition.

## Architecture touched

- Service: `orion-sql-writer`, person-model tick inside the individuals loop.
- New exemplar source `picture_same` in `vision_person_exemplar` (the
  `source` column is free text; no migration).
- No bus, schema, or API change.

## Files changed

- `services/orion-sql-writer/app/vision_person_model.py`: new pure
  `picture_same_exemplars`, DB reader `_picture_same_rows`, `desired_exemplars`
  takes `picture_same`, config field `picture_same_positives`, source added to
  `RECONCILED_SOURCES`, orphan crops get group `orphan:<crop_id>`, summary
  counter `person_picture_same_exemplars`.
- `services/orion-sql-writer/app/settings.py`, `app/vision_individuals_loop.py`:
  the flag.
- `services/orion-sql-writer/.env_example`: the flag, `true`.
- `scripts/sync_local_env_from_example.py`: `VISION_PERSON_PICTURE_SAME_`
  prefix, so the sync script actually adds the key (no `VISION_PERSON_*`
  prefix existed).
- `services/orion-sql-writer/evals/run_person_recognition_eval.py`:
  `--no-picture-same` arm for before/after.
- `services/orion-sql-writer/tests/test_vision_person_model.py`: tests.
- `services/orion-sql-writer/README.md`: the rule.

## Schema / bus / API changes

- Added: exemplar source value `picture_same` (data value, not a column).
- Removed / Renamed: none.
- Behavior changed: recognition album gains confirmed-picture positives.
- Compatibility notes: no migration. Off-switch revokes rows cleanly.

## Env/config changes

- Added keys: `VISION_PERSON_PICTURE_SAME_POSITIVES=true` (orion-sql-writer).
- Removed / renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`:
  yes, `+VISION_PERSON_PICTURE_SAME_POSITIVES='true'`.
- skipped keys requiring operator action: none.

## Metric quality gate (new exemplar source)

1. **Provenance.** Rows come from `vision_ask_picture_verdict` with
   `verdict = 'same'`. Writers: `picture_rows` / `_record_pictures` in
   `services/orion-sql-writer/app/vision_individuals.py` (~line 1883), fed by
   `answer_detail.pictures`. Story-card yes: `motion_story_routes.py` confirm
   action writes one `same` row per picture the card showed. Older rows:
   multi-picture "who is this?" cards. The person is the group's current
   `vision_individual.label`.
2. **Independence from `cluster_label`.** Not independent: both are Juniper
   naming the same people, and a confirmed picture is usually inside a group
   she already named. Live: of 56 `same` rows, 34 are already positives for
   that name through their named group. Those are skipped (deduped on
   casefolded name + crop), so nothing counts twice. The new signal is only the
   remainder: pictures in groups that look like two people, or pictures that
   left their group.
3. **Theory anchor.** A per-picture human answer is a direct label for that
   picture. A group label is an inferred label for every crop in it, which is
   why impure groups were skipped. The per-picture answer does not carry that
   doubt, so it is safe to use where the group label is not.
4. **Live-data check (read-only, 2026-11-24).** 56 `same` rows. Outcome
   through the new code: 8 new positives, 34 already counted, 10 not a person
   group (dog, vehicles), 3 skipped because the picture's name differs from the
   group's current name (a picture named for one person on a group since renamed to another), 1 ruled
   out (she also said that picture is not that person). Not degenerate: it is a
   small, growing count fed by each yes.
5. **Existing mechanism.** `picture_different` already turns `different`
   verdicts into negatives in the same function; this is its positive twin.
   `confirm_pick_yes` is a separate positive source (Is-this-X asks) and is
   included in the dedupe.
6. **Reversibility.** Cheap. Flag off revokes every `picture_same` row on the
   next tick (tested). No schema, no migration, no training default baked in.

## Tests run

```text
cd services/orion-sql-writer && PYTHONPATH=../..:. python -m pytest tests -q -p no:cacheprovider
1344 passed, 265 skipped

# against a throwaway Postgres 16 with the person-model migrations:
ORION_TEST_PG_URI=... python -m pytest tests/test_vision_person_model.py -q
82 passed

# mutation: disabling the dedupe makes 2 of the new tests fail (they catch it)
python scripts/check_env_template_parity.py   -> PASS
python scripts/check_integration_seam.py      -> OK
git diff --check                              -> clean
```

New tests: same -> positive; different -> never positive (and a same+different
picture stays out); unsure -> ignored; already counted -> not twice; renamed
group / non-person group -> nothing; impure group still contributes only its
confirmed pictures (real Postgres); flag off -> rows revoked, old album
(real Postgres); orphaned confirmed crop -> its own group (real Postgres).

## Evals run

```text
evals/run_person_recognition_eval.py --live (read-only), two arms:
with the new examples, and with --no-picture-same.
NOT COMPLETED: both arms ran in parallel (~20 cores each) and were stopped
after ~80 minutes without output. Before/after recognition is UNVERIFIED.
Follow-up: #3867 (run one arm at a time overnight; pass = 0 wrong accepts,
same or more correct accepts, fewer "no album").

Live check after deploy (2026-11-24 23:43Z): 8 active picture_same rows in
vision_person_exemplar; sql-writer tick log: person_picture_same_exemplars: 8.
```

## Docker/build/smoke checks

```text
No Docker build: pure Python change inside an existing service, no new
dependency. Live effect is UNVERIFIED until orion-sql-writer is rebuilt from
main and a person-model tick logs person_picture_same_exemplars > 0.
```

## Review findings fixed

A code-review subagent found nothing material. Minor findings:

- Finding: a picture no longer in any group counted as a group of its own, so
  one walk-past looked like several groups.
  - Fix: `regroup_confirmed_orphans` puts it back in the group she confirmed
    it in.
  - Evidence: Postgres test asserts the orphan's group is the original group.
- Finding: a confirmed picture that moved into a group with another name
  would mix two people.
  - Fix: skipped (`picture_same_moved`).
  - Evidence: `test_a_picture_now_in_a_group_with_another_name_teaches_nothing`.
- Finding: the database test did not cover dedupe with names in different
  capitals.
  - Fix: added a "same" on a "not them" picture and an "Is this X?" yes row.
  - Evidence: 83 passed against real Postgres.
- Finding: an older "different" answer overrides a newer "same" one.
  - Not fixed here: it changes the "different is never a positive" rule.
    Filed #3854 (needs Juniper).
- Finding: the eval's version comparison does not cover an album change.
  - Covered by the two-arm eval; not completed, see #3867.

## Restart required

After merge, the post-pull rebuild redeploys orion-sql-writer. By hand, from
the clean primary main checkout:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

## Risks / concerns

- Severity: low. Concern: the album changes, so bars refit; a person with few
  examples may move. Mitigation: only 8 rows today; the flag reverts.
- Severity: low. Concern: the 5 older multi-picture `same` answers are
  included, not only story-card yes answers. Their meaning is the same ("this
  picture is this person") and the name must still match the group's current
  name. Mitigation: set the flag off if that is not wanted.
- Severity: low. Concern: `MODEL_VERSION` is unchanged (scoring did not
  change, only the album), so the eval's McNemar comparison against the
  previous stored row does not trigger. The before/after arms above stand in.
- Issues opened: #3854 (newer "same" cannot override older "different"), #3867 (verify-live eval). Closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3847

🤖 Generated with [Claude Code](https://claude.com/claude-code)
