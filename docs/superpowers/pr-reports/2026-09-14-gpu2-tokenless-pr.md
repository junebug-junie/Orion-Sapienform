# GPU2 internal control without bearer tokens

GPU2 activation, diffusion lifecycle drain and durable elastic intent now use the
existing internal service/tailnet boundary without a shared bearer token. Removed
all three new token settings from callers, settings, Compose and templates.
GPU1 authentication remains unchanged. Intent/generation validation, lease and
permit fencing, eligibility, atomic draining and restoration remain enforced.

## Validation

- Controller: 43 tests passed, including tokenless activation and GPU1 regression.
- Diffusion: 41 tests passed, including tokenless atomic drain/resume and cancelled GPU work occupancy.
- Durable: 85 tests passed against disposable Postgres, including tokenless endpoint gates.
- Elastic fairness eval: 20 FIFO admissions, restoration passed (physical facts are fixtures).
- All three affected Docker images built through safe_docker_build.sh.
- Removed retired keys from primary ignored envs and ran standard env sync.
- Enabled local GPU2, durable elastic/assignments/restoration/thermal, Thought status and Hub opt-in; shadow false.
- Applied additive admission, gateway-capacity and GPU2 schemas to the configured checkpoint database. Durable startup recovered.

## Review findings fixed

- Finding: GPU1 slot route delegation altered raw-token rejection and failure codes.
  - Fix: Retained its original checks and result mapping inside the GPU1 branch.
  - Evidence: Four route-specific regressions pass; independent review recheck has no material findings.
- Removed stale token documentation and added durable API gate regressions.

## Runtime limits

Production GPU2 round-trip and automatic borrowing remain UNVERIFIED. Circe's
GPU2 status endpoint returns 404; SSH as athena is denied by tailnet policy.
Circe needs deployment, physical mapping/headroom checks and the controlled
round-trip before end-to-end activation can be claimed. Local env flags alone
are not runtime proof. Existing main templates still mention retired keys until
this patch merges; deploy parity warns about those keys but passes without bypass.
