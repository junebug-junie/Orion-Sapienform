# chore: enable reasoning telemetry + seed-v4 corpus

**Status:** IMPLEMENTED, tested. Flips the two switches deliberately held off
through the phi corpus-honesty initiative now that all 3 specs (reasoning
telemetry adapter, corpus hygiene, seed-v4 feature set) are merged and
deployed.

## Summary

- `PUBLISH_REASONING_TELEMETRY=true` (`orion-cortex-exec`) — cortex-exec now
  emits per-call `ReasoningCallV1` telemetry, so orion-thought's
  `reasoning_activity` projection starts accruing real data instead of
  reading empty.
- `INNER_FEATURES_VERSION=seed-v4` (`orion-spark-introspector`) — the phi
  corpus starts writing the honest 8-dim seed-v4 trainable feature set going
  forward (theater trio excised; `execution_load`/`reasoning_load`/
  `reasoning_present` sourced from `reasoning_activity`). Existing seed-v3
  rows on disk are untouched and remain separately fit-able.
- `ORION_PHI_ENCODER_ENABLED` stays `false`. No encoder/promote change in
  this patch — that's a separate, later step per the standing "hold for ONE
  re-version" decision.

## Gaps found and fixed while flipping the switches

- `test_inner_features_settings_defaults` asserted the pre-flip `seed-v3`
  default (it instantiates a live `Settings()`, which reads local `.env`) —
  updated to `seed-v4`.
- `services/orion-spark-introspector/docker-compose.yml`'s
  `INNER_FEATURES_VERSION: ${INNER_FEATURES_VERSION:-seed-v3}` fallback was
  out of sync with the new `.env_example` default — updated to `seed-v4`.
- **4th occurrence of the same recurring failure mode this initiative keeps
  hitting**: `ORION_THOUGHT_BASE_URL` (added by the seed-v4 PR) was never
  added to `orion-spark-introspector/docker-compose.yml`'s explicit
  `environment:` allowlist, so it was silently unconfigurable in Docker
  (falling back to the Python `settings.py` default — which happened to
  match, so it "worked," but wasn't actually operator-tunable). Added.
- Corresponding `docker-compose.yml`-shape gate test renamed/updated
  (`test_compose_defaults_inner_features_to_seed_v3` →
  `..._to_seed_v4`) to assert both.
- Fixed a stale `seed-v2` reference in `services/orion-spark-introspector/README.md`'s
  deployment-requirements table.

## Files changed

- `services/orion-cortex-exec/.env_example` — `PUBLISH_REASONING_TELEMETRY=true`
- `services/orion-spark-introspector/.env_example` — `INNER_FEATURES_VERSION=seed-v4`
- `services/orion-spark-introspector/docker-compose.yml` — fallback + `ORION_THOUGHT_BASE_URL` passthrough
- `services/orion-spark-introspector/tests/test_inner_state_emit.py`,
  `test_compose_seed_v2_telemetry_mount.py` — updated assertions
- `services/orion-spark-introspector/README.md` — stale value fix

## Env/config changes

- Local `.env` synced via `python scripts/sync_local_env_from_example.py`
  for both `orion-cortex-exec` and `orion-spark-introspector` (not committed,
  gitignored).

## Tests run

```text
pytest services/orion-cortex-exec/tests/test_reasoning_emit.py -q        → 15 passed
pytest services/orion-spark-introspector/tests -q                        → 97 passed, 1 pre-existing unrelated failure
pytest services/orion-thought/tests -q                                   → 123 passed, 4 pre-existing unrelated failures
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml config --quiet  → clean
```

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low. Both flags were built default-off/seed-v3 specifically so
  they could be flipped independently once ready; nothing else in the
  pipeline depends on them being off. `ORION_PHI_ENCODER_ENABLED` stays
  false, so no φ computation or Δφ reward path is affected.

## What comes next (not part of this patch)

1. Let corpus accrue — fit gate needs `min_rows=500`, `min_hours=4.0` on
   fresh seed-v4 rows (`scripts/fit_phi_encoder.py`).
2. Run `scripts/diag.py` against the accrued corpus, confirm ≥8 of the 8
   seed-v4 trainable dims clear `variance_fraction=0.8`, `variance_eps=1e-6`.
3. Run a single `scripts/fit_phi_encoder.py` fit + promote gate
   (`PROMOTE_MIN_RECON_RATIO=2.0`) on seed-v4.
4. If it passes: promote (flip `ORION_PHI_ENCODER_ENABLED=true`, point
   `ORION_PHI_ENCODER_WEIGHTS` at the new symlink) — this is a
   cognition-touching change and needs explicit proposal-mode sign-off
   before implementation, per CLAUDE.md.

## PR link

Branch pushed: `chore/enable-reasoning-telemetry-seedv4`.
Compare: https://github.com/junebug-junie/Orion-Sapienform/compare/main...chore/enable-reasoning-telemetry-seedv4
