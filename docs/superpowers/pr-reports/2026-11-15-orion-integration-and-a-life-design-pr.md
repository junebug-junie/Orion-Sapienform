## Summary

- Design spec for Juniper's sign-off: `docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md`. No code.
- Part A maps 15 organs with live numbers, checks the existing integration seams, and proposes one primitive built on Temporal Self: every organ's output is an experience, every organ reads "Orion's now" and cites what it used, and the cite is checked against what it was shown.
- Part A.6 turns that into a deterministic gate: a proposed `CLAUDE.md` "Integration gate" section, `scripts/check_integration_seam.py`, and a live coverage report.
- Part A.7 designs a daily dig roundup (`dig organs <date>`) that replaces the glossing digests.
- Part B rebuilds Prime hunger as needs with content, and gives the root cause of #3366 (a 4-second timeout against a 4-44 second dashboard endpoint).
- Part C proposes projects as the next intention source, with the exact question to ask Orion.

## Outcome moved

A signed-off design and patch order. Four defects found and ticketed; #3366 root-caused and reproduced.

## Current architecture

12 of 15 organs feed another organ through private pipes. Prime reads 4 organs directly. 91% of acts in 7 days have no path by which use can be detected; 2.3% were used.

## Architecture touched

None (design only).

## Files changed

- `docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md`: the design.
- `docs/superpowers/pr-reports/2026-11-15-orion-integration-and-a-life-design-pr.md`: this report.

## Schema / bus / API changes

- Added: none (proposed only: 6 Temporal Self source kinds, 2 columns, 3 tables, 2 endpoints; no bus channel).

## Env/config changes

- None.

## Tests run

```text
git diff --check: clean
```

## Evals run

```text
None: design only.
```

## Docker/build/smoke checks

```text
None: design only. Live reads: Postgres, Redis, durable-runs /physiology/report, Hub curiosity endpoint, container logs (see spec).
```

## Review findings fixed

- Design-only PR; no code review run.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: two of three soup-map helpers were cut off by a usage limit; their organs were re-checked directly, and lines not checked live are marked UNVERIFIED.
- Issues opened: #3412 (dreams 0 hypotheses since 11-13), #3413 (`dream_compaction_delta.applied_at` never written), #3414 (resonance alerts silent since 11-03), #3415 (`curiosity_hop_reading` 0 rows).
- Issues commented: #3366 (root cause).
- Issues closed: none.

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
