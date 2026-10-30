# docs(spec): Orion's energy allowance (design only)

## Summary

- Design proposal: split Orion's electricity bill by who can move it. The rest body (hardware at rest, cooling, Juniper-sponsored work such as the Zelda dojo) is Juniper's lever: reported, never priced. Energy above rest is Orion's activity, priced against a monthly allowance with the existing dual ascent.
- Answers Juniper's question: use cost-function optimization (the price mechanism already is an online primal-dual solver for a budget-limited knapsack), not RL. A day-ahead solver gives nothing on the flat Schedule 1 tariff. Fixed caps stay as safety rails.
- Replaces the misleading "not by what I do" text with the real split.
- States what the design needs from the whole-machine meter (sibling branch `fix/act-energy-whole-machine`): one pinned rest number, per-host hourly attribution rows, a coverage figure.

## Outcome moved

Design only. When built, the energy price stops being stuck at its ceiling with no information (`exogenous_bind`, $184 vs $100) and starts to answer "is Orion's own activity on pace for its allowance".

## Current architecture

One `energy` constraint: Orion's whole bill against a $100 envelope. Orion's resting body alone costs ~$146 per 30 days, so the envelope is broken before Orion acts and the price is clamped. Choice charges no energy (`unpriced:energy`). The per-act meter (#2977) covers under 1% of above-rest energy and uses a rolling 24 h idle floor.

## Architecture touched

None (docs). Proposed: `config/physiology/rest_body.yaml`, `orion/physiology/{body,budget,choice,report_text}.py`, a new constraint `energy:activity`, durable-runs env keys.

## Files changed

- `docs/superpowers/specs/2026-10-30-orion-energy-allowance-proposal.md`: the design.
- `docs/superpowers/pr-reports/2026-10-30-orion-energy-allowance-proposal-pr.md`: this report.

## Schema / bus / API changes

- Added: none (proposed: `energy:activity` constraint name, using the existing `detail` dict; no model fields).
- Removed / Renamed: none.
- Behavior changed: none.
- Compatibility notes: proposal keeps `PhysiologyConstraintReadingV1` unchanged because it is `extra="forbid"`.

## Env/config changes

- None in this PR. Proposed keys are listed in the doc.
- local `.env` synced: not needed (no template change).

## Tests run

```text
git diff --check   (clean)
```

## Evals run

```text
None: design only. Acceptance checks and an October replay are specified in the doc.
```

## Docker/build/smoke checks

```text
None: design only. Live numbers come from read-only psql on conjourney (2026-10-30).
```

## Review findings fixed

- Finding (must): the doc dropped the energy clamp and moved `energy:activity` into self-spent while also charging crash loops and Juniper turns to it, which would hide a crash-loop price from strain.
  - Fix: `energy:activity` stays out of `SELF_SPENT_CONSTRAINTS` and moves strain; strain test rewritten.
  - Evidence: `budget.py:344-348` comment gives the same reason for the existing energy line.
- Finding (must): rest guard compared against nothing (seed shipped `approved_at: null`).
  - Fix: per-host `approved_rest_w` baseline in the file; loader refuses `rest_w > approved_rest_w`; CI refuses raising it without `approved_at`.
- Finding (should): curiosity cost upper bound used 550 W; the doc's own p90 gives 818 W.
  - Fix: bound is now $0.21/run, ~$43/month at 7/day; text no longer claims money cannot bind near 7.
- Finding (should): week of 09-21 is partial; uplift cap of 4 never acts under a 50% coverage gate; "freezes at ceiling" ignored `bind_ramp`.
  - Fix: week marked partial and p75 recomputed ($48.8, still $50); cap removed (gate implies uplift <= 2); wording now says bounded ramp.
- Nits: decile row rounding and hecate coverage hours explained in the doc.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium. Concern: the pinned rest numbers (circe 541 W, athena 340 W) come from 10-01..10-13 only. Mitigation: re-measure script plus Juniper sign-off for any increase.
- Severity: medium. Concern: curiosity energy per run is bracketed ($0.21 upper bound), not measured (meter coverage under 1%). Mitigation: labelled `UNVERIFIED` in the doc.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2989
