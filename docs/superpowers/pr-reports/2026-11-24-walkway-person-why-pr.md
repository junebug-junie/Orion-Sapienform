## Summary

- Each person box on the Hub's Walkway stories card now has an optional "How could you tell?" box.
- Whatever Juniper types there goes with her answer (name / someone else / not sure).
- It is saved on the same answer row (`orion_ask.answer_detail.why`). No new table, no new identity store.
- The card shows it back under what she said: "How you could tell: ...".
- Saving one person no longer wipes half-typed text in another person's boxes on the same story.

## Outcome moved

Juniper's context clues ("pushing a stroller; Jason has no young kids; not his build; not who he walks with") were lost. They are now kept as raw evidence for a later identity primitive (#3768). Nothing reasons over them yet.

## Current architecture

`POST /api/vision/motion-stories/{story_id}/people/{individual_id}` took `{action, name}` and wrote one already-answered `orion_ask` row (`vision_individual` for a name, `vision_story_person` for someone else / not sure). sql-writer applies those rows. The card re-renders the story from the reply.

## Architecture touched

Hub only: `scripts/motion_story_routes.py` and `static/js/walkway-stories.js`. `answer_detail` is free JSON; every reader picks named keys, none uses a strict model (checked in review). The bus event `OrionAskAnsweredV1` does not carry `answer_detail`, so no bus contract change.

## Files changed

- `services/orion-hub/scripts/motion_story_routes.py`: `PersonBody.why` (max 1000), `clean_why`, store `answer_detail.why`, read it back into `story_action.why`, log `why_chars`.
- `services/orion-hub/static/js/walkway-stories.js`: reason textarea, `whyLine`, send `why`, poll guard covers the reason box, keep other boxes' typing across a save.
- `services/orion-hub/tests/test_motion_story_routes.py`: stored + read back + cleaned, over-long refused, control chars dropped.
- `services/orion-hub/tests/test_walkway_stories_js.py`: `whyLine`.
- `services/orion-hub/tests/test_walkway_stories_browser_smoke.py`: reason is sent and shown back; other box's typing survives; still fits 360 px.

## Schema / bus / API changes

- Added: optional `why` on the person POST body; `story_action.why` in the list/person responses; `answer_detail.why` key.
- Removed / Renamed: none.
- Behavior changed: none for callers that omit `why`.
- Compatibility notes: additive; old rows simply have no `why`.

## Env/config changes

- None. `.env_example` not touched.

## Tests run

```text
pytest tests/test_motion_story_routes.py tests/test_walkway_stories_js.py tests/test_walkway_stories_browser_smoke.py
34 passed  (Postgres tests against a throwaway test container; browser tests with local chromium_headless_shell-1254)
```

## Evals run

```text
None. This stores human input; there is no model behavior to evaluate yet. #3768 holds the eval question for the primitive that will read it.
```

## Docker/build/smoke checks

```text
Not run. Hub-only Python + static JS; no dependency, env, or compose change.
```

## Review findings fixed

- Finding: a NUL in the reason would make the jsonb insert fail (500).
  - Fix: `clean_why` drops control characters except line breaks.
  - Evidence: `test_clean_why_trims_spaces_and_blank_lines_but_keeps_line_breaks`.
- Finding: reason cleaned twice (save + log).
  - Fix: cleaned once before the transaction, reused.
  - Evidence: diff.
- Finding: a save re-renders the whole story and wiped half-typed text in the other person's boxes (old for names, likelier with a reason box).
  - Fix: carry the other boxes' typed values into the re-rendered story.
  - Evidence: browser smoke asserts the dog box keeps "on a red leash".
- Nit not changed: the `[:1000]` slice is a guard only (Pydantic refuses over-long input first); commented.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```
(post-pull rebuild does this after merge.)

## Risks / concerns

- Issues opened: #3768. Closed: none.
- Severity: low. Concern: the reasons are stored but unused. Mitigation: #3768 (opened) designs the reader once real reasons exist.
- Live path UNVERIFIED until the Hub is rebuilt and one reason is saved.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3769

🤖 Generated with [Claude Code](https://claude.com/claude-code)
