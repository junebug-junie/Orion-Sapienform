## Summary

- The attention self-model's `prediction_error_confidence` averaged every domain's prediction error with no age check. Only chat is ever stale (it writes only when a turn lands), and a stale chat reading of up to 0.83 could shift the 5-domain mean by up to ~0.17.
- A domain whose node `temporal.observed_at` is older than the shared 1800 s horizon is now left out of the average and out of the trend buffer. Omit, not fade: a faded value reads as calm.
- The basis string records `[from N of M domains; omitted ...]`. If every domain is omitted, confidence is `None`, which equilibrium's reader already skips.
- Rollback: `SUBSTRATE_ATTENTION_SELF_MODEL_OMIT_STALE_PE=false`.

## Outcome moved

Gates that read `prediction_error_confidence` (equilibrium flow/insight metacog gates) no longer see an hours-old chat error as current.

## Current architecture

`worker.py::_brain_frame_prediction_error_and_evidence_by_domain` read node metadata for execution, biometrics, chat, route, bus_synaptic with no age check and fed `reduce_attention_self_model`. Live audit 2026-10-02: chat >30 min old on ~40% of ticks, stale and non-zero on ~3.8%; other four domains always fresh.

## Architecture touched

substrate-runtime worker (two call sites: broadcast tick, brain-frame tick), `orion/substrate/attention_self_model.py` (optional arg, basis string only), new `orion/substrate/prediction_error_freshness.py`. No schema change; `AttentionSelfModelV1` untouched. Horizon imported from `PressureConfig().prediction_error_decay_horizon_seconds` (same source curiosity uses). Does not use the #2480 history table.

## Files changed

- `orion/substrate/prediction_error_freshness.py` (new): freshness rule; missing/unparseable `observed_at` is omitted with age None and traced, never treated as fresh or zero.
- `services/orion-substrate-runtime/app/worker.py`: `_fresh_prediction_error_and_evidence_by_domain`, used by both call sites; logs when the omitted set changes.
- `orion/substrate/attention_self_model.py`: optional `prediction_error_omitted_by_domain`; note goes in `prediction_error_confidence_basis`.
- `settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: new setting.
- tests: new `test_worker_stale_prediction_error_omit.py` (11), fixture update in `test_worker_attention_self_model_tick.py`.

## Schema / bus / API changes

None. Compatibility: `parse_confidence_samples` already skips `None` rows (pinned by a test).

## Env/config changes

- Added: `SUBSTRATE_ATTENTION_SELF_MODEL_OMIT_STALE_PE` (default true).
- `.env_example` updated: yes. Local `.env` sync: NOT done, UNVERIFIED.

## Tests run

```
101 passed, 1 failed (4 files). The failure, test_self_model_tick_reads_broadcast_lane_field_frame,
also fails on the clean primary checkout. check_env_template_parity.py: PASS. git diff --check: clean.
```

## Evals run

None. No replay of how `prediction_error_confidence` would have differed over 24 h (the table stores no per-domain ages). Follow-up.

## Docker/build/smoke checks

None run. UNVERIFIED live.

## Review findings fixed

Code review NOT yet run at time of opening. Status of this PR is DONE_WITH_CONCERNS until it has.

## Restart required

```bash
python scripts/sync_local_env_from_example.py
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

## Risks / concerns

- Severity: low-medium. Omitting a domain changes the average's denominator; the basis string and `None` handling make that visible but a gate near its threshold could move.
- `brain_frame_producer._node_pressure` (display-only, reads dynamic_pressure first) not touched; follow-up.
- Dropped from scope: spec breadcrumbs (Juniper).
