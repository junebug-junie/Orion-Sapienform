## Summary

- Hub Attention Organ ensemble H1 gauge no longer paints mean_ratio 0–0.2 / 0.2–0.6 / 0.6–1 as concentrated / mixed / redundant.
- Three gauges now draw live `/health.config` classifier edges: mean (silence floor + redundant conjunct), std (`std_mixed` / `std_redundant_max`), bulk (`bulk_low` / `bulk_redundant_min`, zoomed).
- Fallback "why" names the settled-agreement conjuncts that actually missed, instead of always blaming bulk.
- Dissipation config rows list the six ensemble edges. Heartbeat `/health` test locks the producer keys Hub now reads.

## Outcome moved

Operator surface matches `classify_ensemble_verdict()`. Live 2026-09-01: mean 0.94 sat in the old "redundant" color band while verdict was mixed (bulk cliff / std middle band). That lie is gone.

## Current architecture

`orion-heartbeat` classifies from std + bulk after a mean silence floor (`reconstruction.classify_ensemble_verdict`). `verdict_thresholds()` already exports those edges on `/health`. Hub `renderBandGauge` still colored the mean bar with the retired 0.2/0.6 split.

## Architecture touched

- `services/orion-hub` operator tab JS + wiring tests
- `services/orion-heartbeat` `/health` contract test (producer keys)

## Files changed

- `services/orion-hub/static/js/attention-organ.js`: three-axis gauge + why line
- `services/orion-hub/scripts/attention_organ_routes.py`: docstring matches ensemble classifier
- `services/orion-hub/tests/test_attention_organ_page.py`: regression tests for the lie
- `services/orion-heartbeat/tests/test_http_endpoints.py`: assert std/bulk keys on `/health`

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: Hub render only
- Compatibility notes: older heartbeat builds missing `std_mixed` etc. draw gray tracks without those ticks

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced: n/a
- skipped keys requiring operator action: none

## Tests run

```text
orion_dev/bin/pytest services/orion-hub/tests/test_attention_organ_page.py -q
39 passed
```

Heartbeat `test_http_endpoints.py` not run in this session (host venv has no `quimb`). Assertions are equality against `verdict_thresholds()`, which live `/health` already returns.

## Evals run

```text
Hub has no eval harness for the Attention Organ tab. Follow-up: none required for this render fix.
```

## Docker/build/smoke checks

```text
Live /health already exports std_mixed, std_redundant_max, bulk_low, bulk_redundant_min.
Hub UI not live-verified from this worktree (static comes from the deployed hub-app image / main mount).
```

## Review findings fixed

- Finding: fallback why always blamed bulk
  - Fix: name missed settled-agreement conjuncts (mean / std / bulk)
  - Evidence: `test_attention_organ_js_fallback_why_names_failing_conjuncts`
- Finding: Hub tests passed on dissipation rows alone
  - Fix: assert three `drawLinearGauge` calls and `thresholds.std_mixed` etc. inside `renderBandGauge`
  - Evidence: `test_attention_organ_js_draws_std_and_bulk_classifier_thresholds`
- Finding: bulk 0.840/0.875 ticks collided on a 0–1 axis
  - Fix: `domainMin` zoom around bulk, same idea as std's auto-span
  - Evidence: `domainMin: bulkMin` in gauge slice

## Restart required

```bash
# after merge, from main checkout (Juniper deploy path)
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

Hub cache-busts static via file mtime (`HUB_UI_ASSET_VERSION`). Rebuild or restart hub-app so `attention-organ.js` is the new file.

## Risks / concerns

- Severity: low
- Concern: `ensembleVerdictWhy` still mirrors Python priority in JS. Badge remains `h1.verdict` from the producer.
- Mitigation: why is caption only; wrong why cannot change classification. Follow-up: put `verdict_reason` on `/h1` if this drifts.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2018
