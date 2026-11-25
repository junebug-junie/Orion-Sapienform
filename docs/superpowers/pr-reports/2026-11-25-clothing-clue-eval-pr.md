## Summary

- New read-only eval: do Juniper's clothing and hair words ("puffy purple jacket", "blue denim shirt", "tight ponytail", "tights", "shorts", "red hair") separate a named person from the street?
- The phrases are turned into SigLIP text vectors once (`scripts/generate_clothing_phrase_vectors.py`, run in the vision-host image) and scored against image vectors already stored for every crop. No new labels, no model at runtime.
- Result: no measurable lead. Clothes and hair are not wired into guesses. Spec updated.

## Outcome moved

The last open part of the clue design (#3768) now has an answer from live data instead of a guess: these words do not yet separate people. It saves building a clothes clue that would not work.

## Current architecture

Stroller and dog clues are live (veto, #3862). Clothes, hair and build had "no answer key yet".

## Architecture touched

None at runtime. Eval, its text side, a one-off generator, spec.

## Files changed

- `services/orion-sql-writer/evals/run_clothing_clue_eval.py`: the eval (per passage, AUC album vs street, `--combine` per person at run time).
- `services/orion-sql-writer/evals/clothing_phrases.json`, `clothing_phrase_vectors.json`: generic phrases (no names) and their vectors.
- `scripts/generate_clothing_phrase_vectors.py`: regenerates the vectors.
- `services/orion-sql-writer/evals/test_clothing_clue_eval.py`: AUC, passage scores, vectors match phrases.
- `docs/superpowers/specs/2026-11-24-walkway-clue-evidence-design.md`: results and caveats.

## Schema / bus / API changes

- None.

## Env/config changes

- None.

## Tests run

```text
pytest services/orion-sql-writer/evals/test_clothing_clue_eval.py   3 passed
pytest tests/test_no_personal_terms.py                               7 passed
```

## Evals run

```text
PYTHONPATH=. python services/orion-sql-writer/evals/run_clothing_clue_eval.py --days 30 --combine ...   (live, read-only)
2,375 street passages, 14 album labels; best stated set combined AUC 0.67 (rank 1, next 0.59); 14 days: tie 0.64
summer phrase in November 0.33; decoration control 0.70 on "red hair"
With 10-110 passages SE ~0.05-0.15: no AUC distinguishable from 0.5.
```

## Docker/build/smoke checks

```text
Generator ran in orion-vision-host-vision-host (offline weights): 6 phrases, dim 1152.
```

## Review findings fixed

- Finding: the spec gave no sample sizes or uncertainty, so "points the right way" overstated it.
  - Fix: n, standard error, rank-by-chance and selection effect stated; verdict reworded to "no measurable lead".
- Finding: a decoration is among the 14 labels ranked.
  - Fix: named as a control in the spec.
- Finding: street side is the newest 20 crops per cluster; album orphan crops count per frame.
  - Fix: both stated as caveats (both conservative or shrink n).

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Seasonal: re-run in another season before ruling clothes out for good. Refs #3768.

## PR link

(filled below)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
