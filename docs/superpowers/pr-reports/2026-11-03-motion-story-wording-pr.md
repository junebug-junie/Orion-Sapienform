# Motion story wording: counts, not repeats

## Summary

- Walkway motion story sentences no longer repeat companions. "together with a dog, a dog, a dog" now reads "moving alongside 3 dogs".
- Other people and other companions share one clause: "together with 1 other person, together with a dog, a dog" now reads "moving alongside another person and 2 dogs".
- A companion of the same kind as the subject reads "other" ("A dog ..., moving alongside a person and 2 other dogs"), so a count never reads like the total on the street.
- "moving alongside" says the companions moved with this one, in this one passage. The old "together with 1 other person" could read like a head count.
- Companion kinds read in a fixed order (people, then dog, bicycle, stroller, vehicle, then others alphabetically).
- Plurals for every kind that occurs: person/people, dog/dogs, bicycle, stroller, vehicle; any other kind gets "s".
- Spot-check gate version (`RULES_VERSION`) is unchanged on purpose (see below). Stored stories are not rewritten.

## Outcome moved

The sentences reach Orion's chat street line, walkway rollups, journals and reveries. They now say how many companions there were, in one short clause, without repeated items.

## Current architecture

`services/orion-sql-writer/app/vision_motion.py` `build_stories` derives each story's companion kinds from its "together" partners (passages that moved with it), and `sentence_for` writes the sentence. Before this patch, other people got ", together with N other person/people", and each non-person companion was listed one by one: ", together with a dog, a dog".

## Architecture touched

- One service: `orion-sql-writer`, sentence text only. No schema, bus, env, or Docker change.

## Files changed

- `services/orion-sql-writer/app/vision_motion.py`: `PLURALS` table and new `companion_phrase()`; `sentence_for` uses it.
- `services/orion-sql-writer/tests/test_vision_motion.py`: regression tests for repeats, plurals, same-kind "other", named subject, and the updated two-people assertion.
- `orion/temporal_self/sources.py`, `orion/temporal_self/README.md`, `orion/temporal_self/tests/test_street_arcs.py`: comments that quoted the old wording (no behavior change).
- `docs/superpowers/pr-reports/2026-11-03-motion-story-wording-pr.md`: this report.

## Before / after (live rows, re-rendered with the new code)

```text
BEFORE: A person walked left to right and away from the house on the walkway, over 15 s (7 detections), together with 1 other person, together with a dog, a dog.
AFTER:  A person walked left to right and away from the house on the walkway, over 15 s (7 detections), moving alongside another person and 2 dogs.

BEFORE: A dog walked left to right and away from the house from the mailbox to the walkway, over 5 s (3 detections), together with a person, a person.
AFTER:  A dog walked left to right and away from the house from the mailbox to the walkway, over 5 s (3 detections), moving alongside 2 people.

BEFORE: A dog walked right to left and toward the house from the walkway to the mailbox, over 5 s (3 detections), together with a dog, a person, a dog.
AFTER:  A dog walked right to left and toward the house from the walkway to the mailbox, over 5 s (3 detections), moving alongside a person and 2 other dogs.

BEFORE: A person with a stroller walked left to right and away from the house on the walkway, over 5 s (2 detections), together with 1 other person.
AFTER:  A person with a stroller walked left to right and away from the house on the walkway, over 5 s (2 detections), moving alongside another person.
```

## Gate decision: `RULES_VERSION` stays 1

The spot-check gate counts Juniper's verdicts for one rules version at a time (`orion/vision/motion_story_gate.py`; the rollup reader and Hub card both read `max(rules_version)`). Live tally today: 131 checked, 129 right, 2 wrong_other, all on version 1. A bump would make new stories version 2 with zero verdicts, close the gate, and stop story sentences from reaching the rollups until 50 more were checked.

The rule's own comment says to bump "when a rule below changes meaning". This patch changes no rule: who is in a passage, which passages are together, the motion, the zones, the name, and the publish/drop decision are all byte-for-byte the same. Only the words for the same companion facts change, and the new words say less that could be wrong (no repeats, no reading as a total). Verdicts judged those facts, so they still vouch for them. The gate logic does not require a bump.

Named people: the subject keeps its name. Companions were always kinds, never names, so there are no different names to collapse.

Consumers checked: `orion/temporal_self/sources.py` builds street labels from story fields, never the sentence. `rename_subject` only matches the sentence start. No code parses "together with".

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: new `vision_motion_story.sentence` text for stories with companions.
- Compatibility notes: stored sentences keep the old wording. A day can mix old and new wording.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-sql-writer/tests -q -k "motion or story"   -> 161 passed, 29 skipped
pytest services/orion-sql-writer/tests -q                         -> 1177 passed, 181 skipped
New tests against main's vision_motion.py: 6 sentence tests FAIL on main (repeats, plurals, same-kind, two-people, named subject); pass on branch.
```

## Evals run

```text
None. No eval harness covers motion story wording; the sentence is deterministic and the unit tests cover every reachable kind combination shape.
```

## Docker/build/smoke checks

```text
None run. Pure text change in one function; no deps, env, or compose change. Not deployed (per task).
```

## Post-deploy SQL check

New stories with companions use the new wording and no repeats:

```sql
SELECT ended_at, sentence FROM vision_motion_story
WHERE created_at > '<deploy time>' AND cardinality(companion_kinds) > 0
ORDER BY ended_at DESC LIMIT 20;
-- expect ", moving alongside ..." and no "together with"
SELECT count(*) FROM vision_motion_story
WHERE created_at > '<deploy time>'
  AND (sentence LIKE '%together with%' OR sentence ~ '(a (\w+)), \1');
-- expect 0
SELECT rules_version, count(*) FROM vision_motion_story WHERE created_at > '<deploy time>' GROUP BY 1;
-- expect only 1 (gate tally unchanged)
```

## Review findings fixed

Review subagent found no material bugs. Fixed:

- Finding: kind order for non-person subjects followed companion story-id hashes, so the same companions could read in a meaningless order ("2 people and another dog").
  - Fix: fixed order (person, then `NOUNS` order, then alphabetical).
  - Evidence: `test_same_kind_companions_read_other_never_a_total` asserts the order.
- Finding: three comments in `orion/temporal_self/` quoted the old wording.
  - Fix: updated to the new wording, noting older rows.
  - Evidence: `orion/temporal_self/tests/test_street_arcs.py` 16 passed.
- Finding (judgment, not fixed in code): the counts are track counts. "a dog, a dog" looked like a list; "2 dogs" asserts two animals, so one dog split into two tracks now reads as a confident wrong count.
  - Decision: kept the change (Juniper asked for counts); noted in the docstring and in Risks below.

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

Run from the clean primary `main` checkout after merge.

## Risks / concerns

- Severity: low-medium
- Concern: counts are track counts. One dog split into two tracks reads "2 dogs". Old wording repeated the item, which hid the claim; new wording states it.
- Mitigation: Juniper's spot-checks will catch it; she should decide whether a split-track count marks a story `wrong_other`.

- Severity: low
- Concern: Juniper's next spot-checks on version 1 see the new wording on new stories.
- Mitigation: the facts checked are unchanged; if she wants the wording judged on its own, a later bump is a one-line change.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
