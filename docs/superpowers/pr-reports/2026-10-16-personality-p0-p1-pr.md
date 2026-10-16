## Summary

- **Tone dials (P0):** a pure measurer that turns one Orion reply into structural numbers: sentence count and length, list lines, headings, bold, questions, dash rate, first-person rate, preamble length, word count. It uses no vocabulary word lists.
- **Tone baseline (P0):** a read-only script over 60 days of chat. It keeps chat replies and outreach separate, and it compares Juniper's `good_tone` turns with her `bad_tone` turns. When either side has fewer than 5 rows it says "insufficient n" instead of printing a number.
- **Choice evidence contract (P1):** `PersonalityChoiceV1` holds what Orion picked, what else was offered, the shown rank, a cost and situation refs. It is registered and has no bus channel, table or writer yet.
- **Choice reducers (P1):** pure reducers rebuild choices from the two lanes that already save the options not picked: curiosity prior selection (Orion picks) and the voluntary attention override (a formula biased by Orion's goal). A read-only script measures them.
- **CI:** a new `Personality` workflow runs every new test with only pytest and pydantic installed.

## Outcome moved

The personality graph (#2726) now has measured inputs instead of assumptions.

- **Orion's chat voice today** (247 replies, 60 days):
  - 77% open with a separate preamble paragraph (median 21 words).
  - 83% contain a dash, about 2 per 100 words.
  - Sentences are short, median about 11 words.
  - Bold appears in 19% of replies and lists in 13%.
- **Real choices by Orion:** 40 in 14 days, all from curiosity. Orion took the top-ranked prior in 16 of 40. In 8 of 40, nothing Orion picked came from the ordered list. Every other lane is a formula, a gate, or an LLM output with no listed alternatives, so the evidence base is thin.
- **Tone separation:** cannot be tested yet. There are 2 `good_tone` rows and 0 `bad_tone` rows.

## Current architecture

No personality code existed. Stance is regenerated every turn, and its carry-over is a 30-minute in-process cache. Curiosity already saved its offered menu and its per-prior outcomes, and the attention broadcast already saved override pairs. Nothing read either one as evidence about Orion.

## Architecture touched

- New pure package `orion/personality/`.
- New contract `orion/schemas/personality.py`. `PersonalityChoiceV1` is in both registries. The nested `ChoiceOptionV1` is in `_REGISTRY` only, per the registry convention for nested models.
- Two read-only analysis scripts.
- No service, env, bus, compose or migration changes.

## Files changed

- `orion/personality/__init__.py`, `orion/personality/tone_dials.py`: the tone measurer
- `orion/personality/choices.py`: the choice reducers
- `orion/personality/tests/test_tone_dials.py`, `orion/personality/tests/test_choices.py`: tests, with fixtures copied from real live rows
- `orion/schemas/personality.py`, `orion/schemas/registry.py`: the contract and its registration
- `tests/test_personality_schemas_registered.py`: checks registration with `resolve()`
- `scripts/analysis/measure_tone_dial_baseline.py` and its test: the tone baseline
- `scripts/analysis/measure_personality_choices.py` and its test: the choice measure
- `.github/workflows/personality-tests.yml`: CI
- `docs/superpowers/pr-reports/2026-10-16-personality-p0-p1-pr.md`: this report

## Schema / bus / API changes

- Added: `PersonalityChoiceV1` (`personality.choice.v1`) and `ChoiceOptionV1`, both registered
- Removed: none
- Renamed: none
- Behavior changed: none at runtime
- Compatibility notes: nothing publishes or consumes these on the bus yet

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/personality/tests scripts/analysis/tests/test_measure_tone_dial_baseline.py scripts/analysis/tests/test_measure_personality_choices.py tests/test_personality_schemas_registered.py -> 75 passed
Same set in a fresh venv with only pytest + pydantic==2.11.5 (the CI install) -> passed
tests/test_agent_trace_schema_registry.py tests/test_no_personal_terms.py -> passed
scripts/check_metric_lineage.py --gate -> PASS
```

## Evals run

```text
Live read-only measurements (P0 is itself the measurement):
measure_tone_dial_baseline.py --window-days 60 -> chat_replies=247 outreach=243; good_tone n=2, bad_tone n=0 (insufficient n)
measure_personality_choices.py (14 d) -> curiosity 40 choices (rank 1: 16/40, 8 off the ordered list, median cost 3582 s); override 355 choices (rank 1: 0/355, by construction)
```

## Docker/build/smoke checks

```text
N/A. No service code changed.
```

## Review findings fixed

- Finding: a code fence opened mid-line deleted every sentence after its closing line. Inline ```code``` spans swallowed whole replies.
  - Fix: fences are now tracked line by line, inline spans are removed, and a longer fence is not closed by a shorter one.
  - Evidence: regression tests in `test_tone_dials.py`. The live baseline was essentially unchanged, so the bug hit few real replies.
- Finding: words were counted on two different bases (list markers and URLs inflated `word_count`).
  - Fix: every word dial now counts the same marker-stripped text, and a URL counts as one word.
  - Evidence: tests.
- Finding: questions wrapped in bold or italics were not counted, and a leading `# Title` was counted as a preamble.
  - Fix: emphasis is treated as a closer, and headings are left out of paragraphs.
  - Evidence: tests.
- Finding: a turn rated both good and bad would land on both sides of the split.
  - Fix: feedback is folded per turn, and a contested turn goes on neither side.
  - Evidence: test.
- Finding: random "coverage" priors kept a continuation rank, so they read as Orion going down the order.
  - Fix: coverage rows are unranked, matching the "drawn at random, not ranked" prompt section.
  - Evidence: test. Live, 0 coverage picks so far.
- Finding: overflowing numbers dropped a whole row, string `source_refs` were split into characters, duplicate loop ids gave the wrong rank, and a filled-in loop could duplicate rank 1.
  - Fix: guarded each case.
  - Evidence: tests.
- Finding: `cost_seconds` spans retries and queue waits (max about 20 h).
  - Fix: documented as "first start to last scoring" and summarized by median.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium. Concern: about 3 real Orion choices a day is too thin to grow a personality graph. Mitigation: #2726 now names outreach topic and stance strategy as the next lanes where Orion becomes the chooser.
- Severity: low. Concern: the override lane is a formula, and its `effort_spent` equals `applied_bias` on every row, so it cannot tell overrides apart. Mitigation: it is labelled `goal_biased_formula`, and the design says not to weight it like an Orion choice.
- Severity: low. Concern: the tone separation check is waiting on Juniper's tags. Mitigation: re-run `measure_tone_dial_baseline.py` once each side has 5 or more rows.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2729

🤖 Generated with [Claude Code](https://claude.com/claude-code)
