## Summary

Three fixes Juniper hit on the first live use of the new walkway card (2026-11-25):

- An unnamed person ("Who is this?") now shows the known names and a type box at once. Before, they hid behind an "It's ..." button that read as a label, so "Can't see them" looked like the only answer.
- The sentence parts quote Orion's own sentence and say what Orion claimed: "Right person? (Orion said Addy)" or "Right number of people?". Before, the card showed the sentence renamed by her answer ("Juniper walked ...") and asked "Right people?" again.
- A name correction on the walker pre-marks "Right person?" as No, held or sent (the server already writes this row when the answer is sent).
- No lone whole-story "Can't tell" under the parts. A result line says what the parts add up to and why ("counts as can't tell (you said can't tell for: ...)"), or "Counted as X so far" for an old-button verdict with parts still open.

## Outcome moved

Juniper can name an unnamed person on the card. Live evidence of the bug: 2026-11-25 17:56Z and 17:57Z, two `unsure` answers on unnamed persons; she says she had no way to give the name.

## Current architecture

Card: `services/orion-hub/static/js/walkway-stories.js` (Patches A-D, #3859 #3863 #3875 #3877). Parts and the automatic people-no live in `scripts/motion_story_routes.py`; unchanged here.

## Architecture touched

Frontend only. No route, schema, bus or env change.

## Files changed

- `services/orion-hub/static/js/walkway-stories.js`: picker open for unnamed persons; `partQuestion`, `correctedName`, `resultLine`; parts box quotes `story.sentence`; no whole-story button with parts.
- `services/orion-hub/static/js/walkway-stories.test.js`: unit tests for the three helpers.
- `services/orion-hub/tests/test_walkway_stories_browser_smoke.py`: picker visible without a click; no whole-story button under parts; the quoted sentence.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
node --test static/js/walkway-stories.test.js   -> 22 pass, 0 fail
pytest services/orion-hub/tests/test_walkway_stories_browser_smoke.py (playwright, chromium) -> 7 passed
pytest services/orion-hub/tests/test_walkway_stories_js.py -> passed
```

## Evals run

No eval: UI wording and layout. The live proof is her next story-card answers naming a person (#3857).

## Docker/build/smoke checks

Browser smoke at 360 px (`_fits`) passes. Hub rebuild on deploy.

## Review findings fixed

- Finding: an old-button verdict became invisible on the parts path until every part was answered.
  - Fix: "Counted as X so far; answering every part replaces it."
  - Evidence: unit test `a verdict from the old buttons shows while parts are still open`.
- Finding: stale comment, unused `row`, `Orion wrote: "null"` when the sentence is missing.
  - Fix: comment rewritten, `row` moved to the old path, fallback heading.
- Finding (filed, not fixed): the card's pre-marked No uses the linked person's role/label; the server checks each camera group. Display only, clears on refresh. #3986.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: #3986 (pre-marked No can disagree with the server in a rare linked-person case until refresh). Mitigation: the server stays the source of truth.
- Issues opened: #3986. Closed: none.

## PR link

(see PR)
