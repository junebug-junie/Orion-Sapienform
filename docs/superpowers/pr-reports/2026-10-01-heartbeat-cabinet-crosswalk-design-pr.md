## Summary

- Design-only spec for a passive experiment: log heartbeat `/h1` before/after attend→act triggers next to the real cabinet temperature outcome.
- Verified in code: heartbeat never sees a cabinet temperature; temperature is folded into `cabinet_climate_activity` and arrives only as one unlabeled biometrics-site atom; the `cabinet.ambient.spike.v1` event is audio, and its atom comes from orion-substrate-runtime, which heartbeat's allowlist skips.
- Records metric-gate findings: fails independence and theory anchor today, so no detector, no publishing, no new channel.
- Names dependencies: PR #2451 (time windows) and the attend→act action shipping.
- Reheat-driver floor (raw ~1.1 vs sqrt(2/pi) ~0.8) is an open item with a test plan.

## Outcome moved

Turns a vague "does heartbeat track the cabinet" idea into a bounded, null-allowed experiment.

## Current architecture / Architecture touched

None changed (docs only). See the spec.

## Files changed

- `docs/superpowers/specs/2026-10-01-heartbeat-cabinet-crosswalk-design.md`: design
- `docs/superpowers/pr-reports/2026-10-01-heartbeat-cabinet-crosswalk-design-pr.md`: this report

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

N/A, docs only. Code claims checked by reading the cited files/lines.

## Evals run

N/A.

## Docker/build/smoke checks

N/A.

## Review findings fixed

Docs-only; no code review run.

## Restart required

No restart required.

## Risks / concerns

- Low. Live heartbeat reheat history was not pulled; floor claim is UNVERIFIED until the test in the spec runs.

## PR link

(see PR)
