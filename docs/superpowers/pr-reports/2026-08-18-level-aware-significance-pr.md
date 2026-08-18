## Summary

- `orion.field.significance.sustained_load_pressure`: the level-aware axis `orion.attention.tension`'s `DeviationGate` structurally cannot see. A channel steadily overloaded for hours re-centers its own EWMA baseline and reads calm, by design — this reads level+dispersion as separate axes (`orion.field.regime.channel_regime()`, PR #1622/#1633, previously unwired to anything), scoped to `loaded_steady` only.
- Wired as a 6th `PRESSURE_DIMENSIONS` entry, real-data-derived variance floor (24h/1,395-point/34,316-real-row replay, full metric-gate writeup in `orion/proposals/scoring.py`'s own comment).
- **No consumer/action wiring** — sensing-only, same staged shape PR #1699/#1701's own tension package used. No target-binding, no policy-template scoring on it yet.
- Architecture ended up simpler than the design doc originally guessed: no new async loop, no in-memory buffer. Runs inline in field-digester's existing hot tick, throttled by a Postgres-persisted timestamp (`sustained_load_computed_at`), querying recent history directly — same pattern `tension_outreach_trigger.py`/`field_channel_glossary_routes.py` already use.
- Live-traced the real signal by hand: `disk_capacity_pressure` on `node:athena`, sustained ~0.77, never spiking — exactly the "looks peaceful but running high load" case named as the reason for this metric, structurally invisible to `deviation_pressure`.

## Outcome moved

Orion now has a real, live-validated signal for sustained-but-non-spiking load, an axis nothing else in the system could see. Confirmed independent of `deviation_pressure` on real data (r=-0.0313, 24h replay).

## Current architecture

`orion.field.regime.channel_regime()` computed level+dispersion as separate axes since PR #1622/#1633 but had exactly one consumer: Hub's debug panel, recomputed fresh per HTTP request, nothing persisted. This patch is the first real producer.

## Architecture touched

- `orion/field/significance.py` (new) — pure computation
- `orion/schemas/field_state.py`, `orion/field/pressure.py`, `orion/proposals/scoring.py` — schema + scoring contract
- `services/orion-field-digester/app/{store.py, worker.py, tensor/update_rules.py, digestion/significance.py (new), settings.py}` — producer
- `scripts/analysis/measure_sustained_load_pressure.py` (new) — the real-data replay script behind every number in this PR

## Files changed

- `orion/field/significance.py` (new): `compute_tick()`/`sustained_load_pressure()` — pure, no Borda (see its own docstring for why), `voting_regimes` is a real parameter not a hardcoded constant.
- `orion/schemas/field_state.py`: `sustained_load_pressure`, `sustained_load_computed_at` fields.
- `orion/field/pressure.py`: `sustained_load_pressure` present in `field_pressures()` ONLY on the tick that actually recomputed it (not unconditionally, unlike `deviation_pressure` — see Review findings below).
- `orion/proposals/scoring.py`: 6th `PRESSURE_DIMENSIONS` entry + real-data-derived `DIMENSION_PRECISION_MIN_VARIANCE` floor, full CLAUDE.md 0A metric-gate writeup.
- `services/orion-field-digester/app/store.py`: `load_recent_field_json()`.
- `services/orion-field-digester/app/digestion/significance.py` (new): throttled-inline producer.
- `services/orion-field-digester/app/{worker.py, tensor/update_rules.py, settings.py}`: wiring.
- `services/orion-field-digester/{.env_example, docker-compose.yml}`: `FIELD_SIGNIFICANCE_WINDOW_SECONDS`, `FIELD_SIGNIFICANCE_CHECK_INTERVAL_SEC`.
- `scripts/analysis/measure_sustained_load_pressure.py` (new): the replay script, with a real `--include-volatile` flag.
- Tests: `tests/test_field_significance.py` (new, 10), `services/orion-field-digester/tests/test_digestion_significance.py` (new, 6), plus updates to `test_dimension_precision_baseline.py`, `test_field_pressure_provenance.py`, `test_field_channel_ratchets.py`, `test_proposal_scoring.py` for the new always-present-conditionally dimension.
- `config/metrics/metric_definitions.lock.json`: 1 new metric locked.
- `docs/superpowers/specs/2026-08-16-level-aware-significance-design.md`: full design + review-findings writeup.

## Schema / bus / API changes

- Added: `FieldStateV1.sustained_load_pressure: float`, `FieldStateV1.sustained_load_computed_at: datetime | None`.
- Behavior changed: `field_pressures()` gains a 6th key, present only on ticks that actually recomputed it (not every tick).
- Compatibility notes: new fields default to `0.0`/`None`; existing consumers unaffected. No consumer reads the new dimension yet.

## Env/config changes

- Added keys: `FIELD_SIGNIFICANCE_WINDOW_SECONDS` (900), `FIELD_SIGNIFICANCE_CHECK_INTERVAL_SEC` (30).
- `.env_example` updated: yes (`services/orion-field-digester`).
- local `.env` synced: yes, directly.
- `docker-compose.yml` updated: yes (field-digester lists env vars individually, not `env_file:` wholesale — `check_service_env_compose_parity.py orion-field-digester`: OK).

## Tests run

```
services/orion-field-digester: 198/198 pass, verified live inside the running
  orion-athena-field-digester container (its real dependency set).
repo-root: 94/94 pass (test_field_significance.py, test_proposal_scoring.py,
  test_field_pressure_provenance.py, test_field_channel_glossary.py, test_field_regime.py).
```

## Evals run

`scripts/analysis/measure_sustained_load_pressure.py` is the closest equivalent — real 24h/1,395-point/34,316-real-row replay, full output in `orion/proposals/scoring.py`'s metric-gate comment.

## Docker/build/smoke checks

```
python scripts/check_definition_drift.py --gate  -> PASS, 0 changes
python scripts/check_service_env_compose_parity.py orion-field-digester  -> OK, all keys exposed
Full pipeline (store.load_recent_field_json -> compute_tick -> update_significance_pressure)
  verified live end-to-end against real Postgres inside orion-athena-field-digester.
```

## Review findings fixed

6 finder agents surfaced 8 findings. One was verified FALSE (claimed `run_digestion_tick`'s new required kwargs broke 5 test call sites in 4 named files — those files don't exist anywhere in this repo; discarded as hallucinated). The other 7:

- Finding: `load_recent_field_json`'s `ORDER BY ASC LIMIT` silently keeps the oldest rows and drops the newest once `row_cap` triggers.
  - Fix: switched to `DESC LIMIT` + `reverse()`, matching `field_channel_glossary_routes.py`'s own already-documented fix for this exact failure mode.
  - Evidence: code diff; currently masked in practice (900s window ≈ 360 rows, well under the 4000 cap) but the window is an exposed operator env knob.
- Finding: duplicate-sample bias — `sustained_load_pressure` throttled to ~30s but fed into the per-~2s-tick EWMA precision baseline as a "fresh" observation on every carried-forward tick.
  - Fix: present in `field_pressures()` only on the tick that actually recomputed it.
  - Evidence: `test_sustained_load_pressure_only_scores_on_the_tick_it_actually_recomputed`.
- Finding: unjustified Borda-ranking ceremony — zero real consumers of `.winner`/`.borda`.
  - Fix: removed; `compute_tick()`/`TickResult` simplified to a plain dict + `max()`, matching `deviation_pressure()`'s own scalar computation (which also doesn't use Borda).
  - Evidence: code diff; `test_scalar_is_the_max_across_all_loaded_ballots`.
- Finding: the `loaded_volatile` exclusion was justified by an unmeasured independence claim.
  - Fix: measured it (`--include-volatile`, new real flag). Both scopes are ~zero correlation with `deviation_pressure` (r=-0.0313 vs r=-0.0021) — reworded every writeup to state the exclusion as conceptual, not independence-driven.
  - Evidence: `orion/proposals/scoring.py`'s metric-gate comment; `test_loaded_volatile_votes_when_included_via_voting_regimes`.
- Finding: a perfectly flat-repeating channel misclassifies as `no_new_input`, missing the exact case this metric targets.
  - Fix: disclosed as a known, real, unfixed limitation (module docstring) — checked against real data, doesn't trigger today (real jitter in the live driver channel) but could for a more coarsely-quantized one. Full fix (threading real per-channel write timestamps) is real follow-up work.
- Finding: `extra="forbid"` + 2 new fields means a rollback to pre-patch code crashes on a post-patch row.
  - Fix: confirmed pre-existing risk class (every additive `FieldStateV1` field already carries it), not new — noted, not fixed (out of scope for a sensing-only patch).
- Finding: 3 near-duplicate hand-rolled SQL fetches across 2 services with no shared helper.
  - Fix: disclosed as known debt — a shared helper would need to touch an already-merged file this patch has no other reason to change.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-field-digester/.env \
  -f services/orion-field-digester/docker-compose.yml \
  up -d --build orion-field-digester
```

## Risks / concerns

- Severity: low
- Concern: the flat-repeat misclassification limitation (disclosed above) and the 3-file SQL duplication (disclosed above) are both real, unfixed-in-this-patch gaps.
- Mitigation: both fully disclosed in code comments and the design doc; neither currently manifests against live data; both are real, scoped follow-up work, not silent.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1718
