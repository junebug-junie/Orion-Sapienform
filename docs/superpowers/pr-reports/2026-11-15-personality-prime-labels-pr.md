## Summary

Orion's tendency labels on the Personality tab said where a choice came from, not what it was about. Live 11-14 a
tendency read "a thread from my letter over a promise I made and a thread from my letter, 2 of 2". This patch makes
labels name the subject and shows the full option words on each card.

- Prime's choice record now leads with the subject. A letter thread starts with its headline ("The walkway render
  break is unresolved and still live each morning — 59 http_500 ... (a thread from my letter for 11-13)"). A chat
  promise starts with Juniper's own line ("\"howdy, just saying hi…\" — a reply I owe Juniper from chat"). Any
  other promise says where it was made ("Curiosity run promise — the words stay in my own thinking"). Markdown and
  trace refs are removed from this text. (`orion/prime/choice.py` `option_text`)
- What Orion promised is NOT put in the record. That sentence is kept in Prime's own thinking frame only (spec
  acceptance S-1.8), because record text becomes tendency labels, and labels reach every turn's Prime block.
  Juniper asked for "what was promised"; this patch cannot give it without breaking S-1.8. See Risks.
- Prime's own prompt is unchanged. The concern text with its refs still goes to Prime, so the citation gate sees
  the same refs. The refs stay on the option key and the concern.
- The label maker drops a leading category phrase ("a thread from my letter", "a promise I made", "A read,",
  "A daydream:") and takes a quoted read title on its own. No model. (`orion/personality/graph.py` `subject_text`,
  `gist`)
- Stored labels: a side that still has its subject is tidied on the next fold (`A read, "The Definitive
  Fine-Tuning Guide` -> `The Definitive Fine-Tuning Guide`). A side that is only a category cannot be recovered
  from the label, so the next choice that names a subject replaces that label, even when it is less similar than
  the founding choice (`reinforce`). Before this, `disp-fba9ebe96512f7f1` (similarity ~1.0) could never be replaced.
- Hub Personality tab: each tendency card has a collapsed "About" block with the full text of what Orion chose and
  passed over in its example choices, as stored. (`services/orion-hub/static/js/personality.js` `aboutView`)

## Outcome moved

Live labels, before -> after (from the replay eval on live 11-14 data):

- `a thread from my letter over a promise I made and a thread from my letter` ->
  `The walkway render break is unresolved over Curiosity run promise and Chat memory has never been promoted`
- `a thread I opened over ...` -> `howdy, just saying hi over My hire channel has returned nothing`
- `A read, "The Definitive Fine-Tuning Guide` -> `The Definitive Fine-Tuning Guide`
- `A daydream: an installation over Say nothing to Juniper this time` ->
  `an installation over Say nothing to Juniper this time`

## Current architecture

Prime (`services/orion-mind/app/prime_loop.py`) snapshots its held concerns when a frame is built
(`shown_concerns`) and records a `PersonalityChoiceV1` per thought (`prime_choice`). The option text was the
concern's `text`, which day letters and promises build with a category lead (`orion/prime/day_letter.py`,
`orion/prime/handback.py`). durable-runs folds choices into tendencies; the label is
`pair_label(chosen, declined)` (`orion/personality/graph.py`), the first six words of each side. A label only
moved to a later choice with a higher similarity.

## Architecture touched

- `orion/prime/choice.py`: record text only. No bus, schema or prompt change.
- `orion/personality/graph.py`: label maker and the label-replace rule in `reinforce`.
- Hub Personality tab JS (view only; the durable-runs `/personality/tendencies` payload already carries the
  full texts).

## Files changed

- `orion/prime/choice.py`: `option_text` (subject-first record text from the concern's public text).
- `orion/personality/graph.py`: `subject_text`, `category_led`, `gist` cut at " — ", `tidy_label` per side,
  stale-label rule.
- `orion/prime/tests/test_prime_choice.py`, `orion/personality/tests/test_graph.py`: tests on live texts.
- `orion/prime/evals/run_choice_label_replay.py` + `fixtures/choice_labels_live_2026-11-14.json`: replay eval.
- `services/orion-hub/static/js/personality.js`, `personality.test.js`,
  `tests/test_personality_panel_browser_smoke.py`: About block.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: `personality_choice.chosen/alternatives[].text` for lane prime is subject-first and has no
  markdown or trace refs. Keys are unchanged.
- Compatibility notes: older prime rows keep their old text. The text is embedded for choice directions, so new
  rows embed differently from old ones (the category prefix no longer dominates the vector).

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/personality/tests orion/prime/tests                  419 passed
pytest services/orion-mind/tests                                  386 passed
pytest services/orion-durable-runs/tests -k personality           73 passed, 13 skipped (postgres)
pytest services/orion-hub/tests (personality, outreach choice/cue) 110 passed
pytest services/orion-hub/tests/test_personality_panel_browser_smoke.py  4 passed (scratch venv with playwright)
node --test services/orion-hub/static/js/personality.test.js     15 passed
```

## Evals run

```text
PYTHONPATH=. python orion/prime/evals/run_choice_label_replay.py
PASS prime_subject_first
PASS no_refs_in_text
PASS promise_words_private
PASS outreach_titles
PASS stored_tidy
```

## Docker/build/smoke checks

```text
Not run. No compose, dependency or env change. Live path UNVERIFIED until deploy.
```

## Review findings fixed

Code review ran in a subagent on af670ecf9. All findings fixed except the embedding note (in Risks).

- Finding (blocker): the promise sentence (`promise_text`) went into the record text, so it reached tendency
  labels and the Prime block in every turn (breaks S-1.8).
  - Fix: option text is built from the concern's public `text` only; `promise_text` is no longer read.
  - Evidence: `test_choice_record_text_is_clean_but_prime_concern_text_is_untouched`, eval `promise_words_private`.
- Finding: the promise category regex cut the front off real subjects ("A promise I made to fix the cache").
  - Fix: the regex only matches the real category shape (known places, then a parenthesis, colon or end).
  - Evidence: `test_real_subjects_are_not_read_as_categories`.
- Finding: the stale-label rule could flap and lowered `label_similarity`.
  - Fix: a category excerpt never replaces a subject base; a stale swap keeps the higher similarity.
  - Evidence: `test_subject_label_never_flaps_back_to_a_category`, updated stale test.
- Finding: chat labels led with "Reply to Juniper's", which used up the six words.
  - Fix: chat options lead with Juniper's line itself.
- Finding: a dash inside a quoted line cut the label; brackets like `[note: x]` were removed; stripping was not
  idempotent; a stray quote survived the " and " split; a long letter thread lost its "(from my letter)" tail.
  - Fix: quote-aware dash cut, id-shaped refs only, strip to a fixed point, quote chars removed per side, the
    middle of the thread is cut instead of the tail.
  - Evidence: `test_label_base_is_idempotent`, `test_only_id_shaped_refs_are_removed`,
    `test_letter_option_keeps_its_tail_when_the_thread_is_long`, gist dash test.

## Restart required

Deploy order (from the clean primary main checkout after merge):

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

durable-runs first, so it folds the new subject-first texts with the new label maker. Order between mind and Hub
does not matter for correctness.

## Risks / concerns

- Severity: medium. Concern: Juniper asked for labels that say what was promised. A non-chat promise label now
  reads "Curiosity run promise", because the promise sentence must stay out of every turn's prompt (S-1.8).
  Mitigation: the sentence is still in Prime's frame. Showing it on the Hub card only (not in labels) is possible
  but needs a separate field on the record; needs Juniper's call.
- Severity: low. Concern: new prime records embed differently from old ones (the category prefix no longer
  dominates the vector), so new choices may start new tendencies instead of reinforcing old ones. Only 5 prime
  choices and 4 candidate nodes exist live (11-15), so the cost is small. Mitigation: none needed now.
- Severity: low. Concern: stored labels that are only categories (e.g. `disp-fba9ebe96512f7f1`) stay as they are
  until that tendency gets its next choice. Labels whose subject survived (reads, daydreams) tidy on the next fold.
- Live path UNVERIFIED until deploy and the next Prime choice.
- Issues opened: none. Issues closed: none.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
