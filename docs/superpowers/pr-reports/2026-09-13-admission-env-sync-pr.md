# Admission env sync follow-up

After PR #2205 deployed, four live service `.env` files lacked 21 admission
keys. The previous implementation synchronized its worktree files only under
the original no-production-config constraint. The reported deployment warnings
also exposed a tooling gap: the default sync skipped the runner and Orch
services, and had no prefixes for their new keys or Gateway lease validation.

## Fix and evidence

- Add runner and Orch to default services and include `DURABLE_RUNS_`,
  `CORTEX_DURABLE_` and `LLM_GATEWAY_LEASE_`. Existing `HUB_CURIOSITY_` covers Hub.
- Make the warning's remedy name the affected services and use `--all-keys`;
  mention protected exclusions instead of promising a filtered default scan
  can repair every possible missing key. Existing values are never forced.
- Add a regression using the real admission templates and a temporary primary
  checkout: default scan reaches all four services, preserves a lease override
  and the Tailscale bus address, and the second sync is idempotent. Add a warning
  regression and run both suites in static CI.

The live correction added exactly 2 Orch, 13 runner, 2 Hub and 4 Gateway keys.
Programmatic before/after comparison confirmed every existing value unchanged.
`check_env_template_parity.py` now reports PASS for all four with no warnings.
All four running containers already have equivalent effective defaults; the
runner's empty lane-policy value falls back to `{}`. No restart or feature
activation was needed. No secrets or local `.env` files are committed.

## Review findings fixed

- Finding: default service selection and key filtering both excluded admission.
  - Fix: cover both lists and test the actual default CLI path.
  - Evidence: temporary primary-checkout regression and real 21-key correction.
- Finding: the warning's bare sync command could leave unlisted keys missing.
  - Fix: explicit service arguments and `--all-keys`, with protected-key caveat.
  - Evidence: warning-command regression without any `--force` recommendation.

Independent subagent review confirmed service/key coverage, preserved overrides
and correct internal service URLs. Verification uses the existing sync/parity
test suites plus the live operational parity check; no cognition behavior or
schema changed, so no model eval or Docker rebuild is required.
