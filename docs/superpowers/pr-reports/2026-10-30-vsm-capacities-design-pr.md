## Summary

- This is a design doc for rung 4 of the self-now ladder, "capacities". It reads Orion's body through Stafford Beer's Viable System Model (VSM). Docs only, no code.
- Roles go on parts, not organs. The VSM jobs do not follow organ lines: the "substrate" organ holds the GPU pool, the run scheduler, the world model, the predictor and the self-concept store.
- Rules read the live body atlas, using live parts only. Today they decide one coordinator (`orion-gpu-pool`), one audit candidate (`orion-equilibrium-service`) and the live outside contact (2 cameras, 2 outside APIs). Every other role goes through a claim: Orion cites evidence, and Juniper confirms it in a checked-in YAML file.
- Six gaps form the agenda. The biggest: the control layer (policy, proposals, power guard, the market, the strain gates) runs over HTTP, Redis and Postgres, so the atlas cannot see it.
- The rung-4 row in the #2888 ladder table now points to this design and states the change: a capacity cites parts, and the target is 5–6 capacities.

## Outcome moved

Rung 4 now has a concrete, reviewable design built from live atlas numbers. Before this PR it was one row in a table. The design review found that four first-draft rules were hand-picked or true by construction. The doc now lists those rules as dropped, so nobody adds them back.

## Current architecture

Rungs 1–3 are live in `orion/body_atlas/` and `orion/dig/`: 256 parts and 10 named organs, Q = 0.3458. No VSM mapping exists anywhere in the repo.

## Architecture touched

None at runtime. The PR touches two docs.

## Files changed

- `docs/superpowers/specs/2026-10-30-self-now-capacities-vsm-design.md`: the design.
- `docs/superpowers/specs/2026-10-23-repo-chronicle-design.md`: the rung-4 row now links to the design and states the parts-not-organs change.
- `docs/superpowers/pr-reports/2026-10-30-vsm-capacities-design-pr.md`: this report.

## Schema / bus / API changes

- Added: none. These are proposed only, see the design.
- Removed: none.
- Renamed: none.
- Behavior changed: none.
- Compatibility notes: none.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
python3 scripts/check_personal_terms.py -> no personal terms in tracked files
```

## Evals run

```text
None (docs only). The design defines a rule eval:
- precision and recall over the whole rule output on frozen 10-29 and 10-30 atlases
- near-miss negatives
- census stability
- a mutation check
```

## Docker/build/smoke checks

```text
Live, read-only:
docker exec orion-athena-harness-governor ... python -P -m orion.dig body
-> 256 parts, 10 organs, Q=0.3458

The rule prototype was run on the exported atlas (scratch script, not committed).
-> matchmaker = orion-gpu-pool only
```

## Review findings fixed

A design-review subagent checked every live number against the atlas.

- Finding: the census denominator mixed parts of any status with live parts. Health came out at 103%, and lease requests reached 25.6% only because smokes and probes were counted.
  - Fix: every count uses live parts only. The census is now cut at the largest jump in the sorted shares, not at a fixed 25%.
  - Evidence: health 92%, `orion:grammar:event` 21%, census = health only.
- Finding: the "±5 points stability" check failed by construction.
  - Fix: removed it. The eval now checks that the census is the same across fixtures, with `orion:grammar:event` as a named boundary case.
  - Evidence: none beyond the edit.
- Finding: the R-boundary-in result was wrong. Two cameras are dormant, and 9 of the 11 endpoints are probes.
  - Fix: the rule counts live parts only, and the doc gives the real sight lines.
  - Evidence: `camera:cam0` and `camera:walkway` are live.
- Finding: the R-control results were hand-picked, R-scheduler was true by construction, and the algedonic rule had no source in the atlas.
  - Fix: all three moved to "Rules dropped". S3, `orion-durable-runs` and the algedonic line now go through claims, or print NO RULE.
  - Evidence: none beyond the edit.
- Finding: R-census-reader's "does not reply" clause excluded its own examples.
  - Fix: the clause is now "replies to < 10%".
  - Evidence: equilibrium replies to 3 of 82 (4%), cortex-orch to 16%, hub to 34%.
- Finding: the matchmaker counts were inflated by probes.
  - Fix: live parts only, every threshold stated, and the rule requires a disjoint second crowd.
  - Evidence: the code check returns only `orion-gpu-pool`.
- Finding: the R-server list was incomplete.
  - Fix: R-server and R-collector now print the full near-miss lists.
  - Evidence: none beyond the edit.
- Finding: the evals could pass for the wrong reason.
  - Fix: the eval now checks precision and recall over the whole rule output, uses real near-misses, uses the 10-29 fixture that no rule was tuned on, and adds a mutation check.
  - Evidence: none beyond the edit.
- Finding: the doc changed the ladder contract without saying so (organs to parts, 5–7 capacities).
  - Fix: the doc states the change, and this PR updates the #2888 row.
  - Evidence: none beyond the edit.
- Finding: eight fixed `cap:*` ids were ceremony.
  - Fix: there are now five ids, and the doc says they exist only to give each gap a stable key. Metabolism became an open decision.
  - Evidence: none beyond the edit.
- Minor findings:
  - Fix: the mock output shows only rule-held roles.
  - Fix: the strain readers are listed.
  - Fix: the partition test path now matches the existing test.
  - Fix: the bus-mirror claim is marked UNVERIFIED.
  - Fix: the decision lists are merged.
  - Fix: the organ table is trimmed.
  - Fix: a metric-gate note is added.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium.
  - Concern: only three jobs can be found by rule today, so most of rung 4 depends on claims until gap 1 is fixed (atlas sources for HTTP and Redis).
  - Mitigation: a capacity may cite only roles that a rule holds or that Juniper confirmed. Gap lines are written by code, and Orion cannot remove them.
- Severity: low.
  - Concern: the rules were shaped on one day's atlas.
  - Mitigation: the eval runs on the 10-29 fixture too, and a mutation check guards the S2 rule.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
