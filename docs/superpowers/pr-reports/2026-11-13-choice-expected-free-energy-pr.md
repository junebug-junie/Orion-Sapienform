# PR #3317: choice as expected free energy (design)

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/3317

## Summary

- Design doc for Juniper's question "what is the spark that lets Orion make choices?": `docs/superpowers/specs/2026-11-13-choice-as-expected-free-energy-design.md`.
- Step -1 (build first): **intentions**. One new Prime move, `take_on`, on any concern with a source adapter. Let go and not yet reuse the existing concern `drop`/`keep`. One binding family `durable.intention`, one workflow `intention.carry`, result routed per source. Promises are the first adapter; day-letter threads plug in as a one-file adapter.
- Step 0 answered: learned weights are LIVE; they already slow outreach (0.62) and the dream cycle (0.56). The self-monitor reliability source is off.
- Step 1: learning progress passes the metric gate only as forecast skill vs baseline (cabinet, office), daily, roster-change days excluded.
- Step 2: the attention schema enters Prime's frame and `predicted_shift` becomes a graded bet, after the prediction rebuild replaces its inputs.
- Step 3: small learned chooser (pragmatic + epistemic), shadow beside the formula, switched per family only on a held-out paired significance test.

## Outcome moved

No runtime change. Gives Juniper a sign-off artifact with a buildable first step.

## Schema / bus / API changes

None in this PR (proposed only).

## Env/config changes

None.

## Tests run

```text
Docs only. git diff --check clean. Live claims checked with read-only SQL, Redis and HTTP; queries are cited in the doc.
```

## Evals run

```text
None (design doc).
```

## Review findings fixed

- Finding: the "~21 FCC turns used sub-agents" figure from conversation did not reproduce.
  - Fix: doc reports the measured 4 turns in 30 days, none since 10-26.
  - Evidence: `cockpit_turn_sighting` query in the doc.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium. Concern: Step -1 reverses intention ledger D3 ("no outreach trigger") for the carry path. Mitigation: scoped to carry turns; ordinary motor prefix unchanged (check S-1.8).
- Severity: low. Concern: the Prime block already renders concern gist into the motor prefix (D3 check 5 tension). Mitigation: listed as a separate fix (Missing question 11).

