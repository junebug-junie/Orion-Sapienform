## Summary

- New read-only eval: how often a stroller or dog is next to each named person vs the street (`run_clue_rates_eval.py`).
- Strollers and dogs are already detector boxes, so the clue design's "new SigLIP reader" is not needed for them.
- Clue-evidence design (patch 2) revised with the measured numbers.
- Clothes, hair and build still have no answer key; their reader waits for a labelled set.

## Outcome moved

Patch 3 (clues in the who-is-this guess) now has measured starting points instead of guesses: a stroller is evidence "not them" for the most-seen regulars (weight of evidence -3.4 for the most-seen one), dog clues are weak and can point either way.

## Current architecture

The clue design planned a SigLIP text read per clue. The detector already emits `stroller` (since 2026-11-18) and `dog` boxes; `together_pairs` links only companions moving together over >= 2 frames, which misses most one-frame dogs.

## Architecture touched

None at runtime. Eval + spec only.

## Files changed

- `services/orion-sql-writer/evals/run_clue_rates_eval.py`: the eval (same-frame, within one person-height; per passage and per crop).
- `services/orion-sql-writer/evals/test_clue_rates_eval.py`: tally, weight of evidence, report.
- `docs/superpowers/specs/2026-11-24-walkway-clue-evidence-design.md`: patch 2 revised with numbers and caveats.

## Schema / bus / API changes

- None.

## Env/config changes

- None.

## Tests run

```text
pytest services/orion-sql-writer/evals/test_clue_rates_eval.py   3 passed
pytest tests/test_no_personal_terms.py                            7 passed
```

## Evals run

```text
PYTHONPATH=. python services/orion-sql-writer/evals/run_clue_rates_eval.py   (live, read-only)
per passage: street 372; stroller 29 (7.8%), dog 42 (11.3%), named dog 15 (4.0%)
most-seen regular: stroller 0/189 (woe -3.40); others with >= 10 passages: -0.53 to -1.41
```

## Docker/build/smoke checks

```text
Not applicable: read-only eval script.
```

## Review findings fixed

- Finding: per-crop counts overstated the evidence (one walk past = many frames; all stroller boxes in ~23 minutes).
  - Fix: unit is a passage (cluster, stream, minute); per-crop printed beside it; spec says ~20 street stroller events.
  - Evidence: eval output above; stroller woe for the most-seen regular -4.11 per crop -> -3.40 per passage.
- Finding: the answer key was described as one-by-one confirmations; it is ~99.8% named clusters that passed the purity check.
  - Fix: docstring and spec reworded.
- Finding: a crop that is also -1 for the same name, or under two names, still counted.
  - Fix: both dropped in SQL.
- Noted in spec: street includes unnamed regulars (shrinks weights toward 0).

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. The stroller weight rests on about 20 street events in one week; patch 3 must learn weights per person from passages, not hard-code these. Refs #3768.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3835

🤖 Generated with [Claude Code](https://claude.com/claude-code)
