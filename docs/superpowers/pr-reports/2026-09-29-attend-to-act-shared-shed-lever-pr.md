## Summary

Docs-only amendment to the attend → act → learn design (PR #2394). Orion's first world action no longer gets its own GPU pause; it shares the hardware-watch reflex's shedding lever, ships after it, and fires earlier and gentler.

- Ordering: the hardware-watch reflex (spec 2026-09-28 Part 4 + rule U4) ships first; the learned action is built on top of it.
- One lever: Orion sets a second, lower-precedence named reason (`orion_self_shed`) on the same pool shedding guard; the reflex's `cooling_incident` always wins. Same U4 semantics (stop new grants, running work finishes, no recall), background only (argued: system work is Orion's own cognition/execution).
- Mutual exclusion: not proposed while any hardware-watch incident is open or hardware-watch health is unknown; an active Orion reason is cleared when the reflex fires (`preempted_by_reflex`). Holdback drawn only among reflex-idle decisions.
- Learning: reflex-overlap rows excluded from fitting (AC failure is exogenous and dominates); render-gate-overlap rows kept and reported (excluding them would select on the outcome). Intention-to-treat on a clock from decision time, drained-only contrast alongside.
- Corrections: elevated trips at 29.5 °C (28.0 is re-arm); cabinet temperature *is* persisted (80,395 rows in `orion_biometrics_summary` since 2026-08-29), so the new column, zwave poll and Phase 0 are dropped; the rise computation is reused from the reflex, not duplicated.

## Outcome moved

Removes a planned two-controller collision on the GPU pool before any code is written, and fixes two factual errors that would have shaped the first patch.

## Current architecture

Design doc listed the hardware-watch reflex as a "later action" and chose recall/pause semantics via #2385 for Orion's shed.

## Architecture touched

Docs only. No code, schema, env, bus or config change.

## Files changed

- `docs/superpowers/specs/2026-09-29-attend-to-act-loop-design.md`: amendment section + inline corrections.
- `docs/superpowers/pr-reports/2026-09-29-attend-to-act-shared-shed-lever-pr.md`: this report.

## Schema / bus / API changes

None in this PR (design proposals only, described in the doc).

## Env/config changes

None. No `.env_example` changed.

## Tests run

```text
git diff --check  -> clean
```

## Evals run

```text
None: docs-only.
```

## Docker/build/smoke checks

```text
None: docs-only. Live evidence queried read-only from production Postgres (orion_biometrics_summary, gpu_pool_leases), 2026-09-29 ~22:45Z.
```

## Review findings fixed

Consistency review in a subagent: 0 must, 5 should, 4 nits. All fixed.

- Finding: named-reason guard stated as fact; the hardware-watch spec defines only one guard.
  - Fix: reworded as a dependency/ask to the reflex PR.
- Finding: `overlap:reflex` defined two ways; excluding CPU/GPU heat incidents would select on the outcome.
  - Fix: `overlap:reflex` = `cabinet_ac` incident only (excluded); new `overlap:heat_incident` kept as a covariate.
- Finding: 5.6 episodes/day read as an expected rate.
  - Fix: labelled a ceiling (winner bind + reflex-idle not counted); first patch's eval measures the joint rate.
- Finding: stale "learns within ~2 weeks" contradicted the ~47-per-arm volume.
  - Fix: withdrawn.
- Finding: `cooling_incident` clear rule misdescribed (re-evaluated each tick).
  - Fix: table corrected; Orion's reason is cleared on `cabinet_ac` incident open, not on guard tick.
- Nits: leftover "paused", "~2 months" -> "6+ weeks", "shipped" -> "implemented", check-7 citation phrased as an added expectation.
  - Evidence: `git diff --check` clean after fixes.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: the reflex PR (`feat/hardware-watch-and-shedding`) is not pushed yet; the named-reason guard shape and the pure rise function are assumed from the brief. The doc says the Orion reason adapts to whatever guard shape ships, and never adds a second guard.
- Mitigation: explicit prerequisite (acceptance check 0) and a stated ask to that PR.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2417
