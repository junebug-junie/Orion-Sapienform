## Summary

Handoff step 4: before the personality graph grows "in situation S, theme T" subnodes (CAPS if-then), check whether any theme actually shows up in particular situations more than chance. **Answer today: no.** So this PR ships the measurement only, with no reducer change.

- New read-only script `scripts/analysis/measure_personality_situation_specificity.py`, with tests and CI.
- Verdict: do not build situation subnodes yet. Run the script again once there are about 2 more weeks of evidence.

## Outcome moved

The question "are themes situation-specific?" now has a repeatable test with known power, instead of a guess.

Live run (2026-10-20, 1,008 stamped rows, which fold into 401 episodes):

```text
tested 36 (node, dimension) pairs; untestable 249 (too few episodes in strata that vary)
     q      p   tvd  eps dim        over       status     label
...
significant at q <= 0.1: 0 of 36
```

Smallest p-values: bus-synaptic prediction error, day_phase, p 0.008 (q 0.31). Chat prediction error, company, p 0.022. None of the 3 active themes comes close (smallest p 0.22).

**Sensitivity:** with a 5-minute burst window instead of 30 minutes, 5 of 38 pairs pass q ≤ 0.1. All 5 are candidate prediction-error themes from reverie. The bus-synaptic theme's preferred phase flips from "afternoon" to "night" between the two windows. That flip is the sign of counting one reverie loop as many observations. A stable situation preference would not flip. The 30-minute result is the one to trust.

## Current architecture

Every `personality_evidence` row carries `meta.situation` (arousal, day_phase, company) from #2739. The Hub panel shows each theme's raw situation mix (#2743). Nothing compared that mix against what the theme's lanes produce anyway.

## How the test works

- **Episodes, not rows.** A theme's rows in one lane that fall within 30 minutes of each other count as one episode.
- **Lane-and-day baseline.** Each episode's expected situation mix is the mix of every episode in the same lane on the same UTC day. A theme seen only on a strained day is not a "strained" theme.
- **Shuffle test.** Situation values are shuffled within each (lane, day) group, 2,000 times. The Benjamini-Hochberg q-value corrects for testing many pairs.
- **Untestable is not null.** A theme without 8 episodes in groups whose situation varies is reported as untestable, never as "no effect".
- **Power is checked in tests.** A planted preference across 15 episodes is found at p < 0.01. Across 20 seeds with no planted preference, at most 3 reach p < 0.05.

## Architecture touched

Read-only analysis script plus the CI workflow. No service, schema, bus or env change.

## Files changed

- `scripts/analysis/measure_personality_situation_specificity.py`: the measurement.
- `scripts/analysis/tests/test_measure_personality_situation_specificity.py`: burst collapse, planted-effect power, no-effect false-positive rate, untestable handling.
- `.github/workflows/personality-tests.yml`: runs the new test.
- This report.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: none.
- Compatibility notes: none.

## Env/config changes

None. Not needed: `.env_example` update, local `.env` sync.

## Tests run

```text
pytest scripts/analysis/tests/test_measure_personality_situation_specificity.py -q -> 4 passed
pytest orion/personality/tests scripts/analysis/tests -q                         -> see CI
```

## Evals run

```text
Live read-only run (30-min bursts, 2000 permutations): 0 of 36 significant at q <= 0.1
Sensitivity run (5-min bursts, 1000 permutations): 5 of 38 significant, unstable direction (burst artifact)
```

## Docker/build/smoke checks

```text
Not applicable (read-only script).
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

```text
No restart required.
```

## Risks / concerns

- **Severity:** low.
  - **Concern:** only about 4 days of stamped evidence exist (since 10-16 23:36). Most (node, dimension) pairs are untestable (249 of 285).
  - **Mitigation:** run the script again around 11-03. Build subnodes only if active themes pass at q ≤ 0.1, and only if the burst window does not flip the preferred situation.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
