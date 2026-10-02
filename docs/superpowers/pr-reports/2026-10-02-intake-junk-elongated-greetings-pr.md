## Summary

- First live chat after the Stage 0A deploy: Juniper's "wasssuppppp" was saved and auto-approved as a semantic memory. The junk filter only matched exact spellings, so "wassup", "heyyyy", "hiiii", "yooo", "hmmm" and "okkk" also passed.
- Runs of 3+ identical characters are now collapsed before matching (`_STRETCH_RE` in `orion/memory/intake_junk.py`). Doubles are deliberately untouched, so "good" never becomes "god".
- Added the wassup spellings (wassup, wasup, whassup, wazzup, wazup, whatsup, heyo) to filler.

## Outcome moved

Stretched greetings stop becoming memories. Real stretched content stays: "noooo my flight got cancelled" and "I feeeel awful" are kept.

## Current architecture

`_words()` lowercased and tokenized only, so the filler and stopword sets saw "wasssuppppp" as an unknown content word.

## Architecture touched

`orion/memory/intake_junk.py` only (orion-memory-consolidation).

## Files changed

- `orion/memory/intake_junk.py`: stretch collapse in `_words`; wassup variants in `_FILLER`.
- `services/orion-memory-consolidation/tests/test_intake_junk_gate.py`: two regression tests.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
orion-memory-consolidation tests + evals: 231 passed
new stretched-greeting test: fails on the previous intake_junk.py, passes now
```

## Evals run

```text
intake gate replay: kept 101, dropped 6 (unchanged); greeting/command summaries 0; synthetic keepers 13/13
```

## Docker/build/smoke checks

Live miss verified in Postgres: memory_crystallizations summary='wasssuppppp', kind=semantic, active, gate_reasons=["substantive_shift"], history op auto_activate.

## Review findings fixed

None. Review skipped: a two-line normalization with regression tests and an unchanged replay.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-memory-consolidation up -d --build
```

## Risks / concerns

- Severity: low
- Concern: a real word containing 3+ identical letters in a row is collapsed before the filler check. It only matters if the collapsed form is filler, and then only for an otherwise-empty message.
- Mitigation: the keep-real test pins the obvious cases.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
