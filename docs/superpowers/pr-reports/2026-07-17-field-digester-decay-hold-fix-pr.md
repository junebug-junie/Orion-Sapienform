# PR report: field-digester decay-hold fix

## Summary

- `orion-field-digester`'s `apply_decay()` previously decayed every `NODE_DECAY_CHANNELS`
  entry unconditionally every 2s tick, regardless of whether fresh biometrics data arrived —
  producing a mechanical sawtooth (~44% loss between real ~15-30s publishes, then a hard snap
  back) confirmed as a polling-architecture artifact, not real host load
  (`orion/autonomy/drives_and_autonomy_retrospective.md` §5b).
- Fixed: new `FieldStateV1.node_vector_updated_at` tracks each `(node_id, channel)`'s last
  real perturbation timestamp; `apply_decay()` now holds a channel flat while it's within
  `FIELD_DECAY_STALENESS_THRESHOLD_SEC` (default 90s) of its last real write, and only decays
  once genuinely stale.
- 8-angle code review found two direct-write paths (`worker.py`'s `field_coherence_warning`,
  `suppression.py`'s `staleness` reset) that bypassed the new tracking entirely, plus a
  `docker-compose.yml` env-parity gap and some cleanup — all fixed in a follow-up commit.
- Updated `services/orion-field-digester/README.md`, `orion/autonomy/README.md`, and
  `orion/autonomy/drives_and_autonomy_retrospective.md` (new §5c) to record the fix.
- `DriveEngine`'s separate fold-batch clamp collapse (retrospective §5b/§6 item 5) is
  deliberately **not** touched here — deferred pending a live post-deploy gate re-run to see
  whether this fix alone drops fold-batch tension volume enough.

## Outcome moved

Every channel in `NODE_DECAY_CHANNELS` (19 channels: `cpu_pressure`, `memory_pressure`,
`gpu_pressure`, `thermal_pressure`, `disk_pressure`, `staleness`, `execution_load`,
`execution_friction`, `reasoning_load`, `failure_pressure`, `egress_confidence_deficit`,
`repair_pressure`, `conversation_load`, `transport_pressure`, `contract_pressure`,
`catalog_drift_pressure`, `observer_failure_pressure`, `reliability_pressure`,
`field_coherence_warning`, `prediction_error`) stops mechanically sawtoothing between real
publishes. This removes an artificial source of `tension.distress.v1` volume from the drive
economy downstream — not yet re-measured live (see Risks/concerns).

## Current architecture

`perturb -> decay -> diffuse -> suppress`, ticking every `RECEIPT_POLL_INTERVAL_SEC=2.0`s
(`app/tensor/update_rules.py`, `app/worker.py`). `apply_decay()` multiplied every
`NODE_DECAY_CHANNELS` entry by `BIOMETRICS_FIELD_DECAY_RATE` every tick unconditionally.
`apply_perturbations()` applied fresh readings via `mode="replace"` full overwrite. No
per-channel freshness tracking existed.

## Architecture touched

`services/orion-field-digester`'s digestion pipeline (`decay.py`, `perturbation.py`,
`suppression.py`, `update_rules.py`, `worker.py`, `settings.py`, `.env_example`,
`docker-compose.yml`) and the shared `FieldStateV1` schema (`orion/schemas/field_state.py`).
Docs: `services/orion-field-digester/README.md`, `orion/autonomy/README.md`,
`orion/autonomy/drives_and_autonomy_retrospective.md`.

## Files changed

- `orion/schemas/field_state.py`: new `node_vector_updated_at` field.
- `services/orion-field-digester/app/digestion/decay.py`: staleness-gated decay, `_aware_utc()`
  helper.
- `services/orion-field-digester/app/digestion/perturbation.py`: records
  `node_vector_updated_at` on every write.
- `services/orion-field-digester/app/digestion/suppression.py`: records
  `node_vector_updated_at` for the `staleness` reset (review fix).
- `services/orion-field-digester/app/worker.py`: records `node_vector_updated_at` for
  `field_coherence_warning` (review fix); threads the new setting into `run_digestion_tick`.
- `services/orion-field-digester/app/tensor/update_rules.py`: threads `now`/
  `staleness_threshold_sec` into `apply_decay`.
- `services/orion-field-digester/app/settings.py`, `.env_example`, `docker-compose.yml`: new
  `FIELD_DECAY_STALENESS_THRESHOLD_SEC` (default 90.0), including the compose passthrough
  (review fix).
- `services/orion-field-digester/tests/test_decay_hold.py` (new),
  `tests/test_worker.py`: regression tests for the fix and the two review-found gaps.
- 5 pre-existing test files (`test_field_channel_ratchets.py`,
  `tests/test_field_deterministic_replay.py`, `tests/test_field_digestion_rules.py`,
  `tests/test_field_execution_perturbations.py`, `tests/test_field_transport_perturbations.py`):
  call-site updates for the new required kwargs, no assertion changes.
- `docs/superpowers/specs/2026-07-17-field-digester-decay-hold-fix-design.md`: quick spec.
- `services/orion-field-digester/README.md`, `orion/autonomy/README.md`,
  `orion/autonomy/drives_and_autonomy_retrospective.md`: status updates.

## Schema / bus / API changes

- Added: `FieldStateV1.node_vector_updated_at: dict[str, dict[str, datetime]]`
  (`default_factory=dict`, backward compatible — a state persisted before this fix loads with
  an empty dict, treated as "unknown freshness," safe-default decay applies unchanged).
- Removed: none.
- Renamed: none.
- Behavior changed: `apply_decay()` and `run_digestion_tick()` gained new required keyword
  parameters (`now`, `staleness_threshold_sec`); all real call sites updated.
- Compatibility notes: `FieldStateV1` is not registered in `orion/bus/channels.yaml` /
  `orion/schemas/registry.py` as a bus-published schema in its own right (it's persisted
  directly to Postgres by this service, not published as a bus event payload) — no channel
  registry update needed.

## Env/config/settings changes

- Added keys: `FIELD_DECAY_STALENESS_THRESHOLD_SEC` (default `90.0`).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (`services/orion-field-digester/.env_example`).
- `docker-compose.yml` updated: yes (review-found gap, fixed same PR).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran; reported no
  local `.env` exists anywhere in this worktree (expected — `.env` is gitignored and not part
  of `git worktree add`; nothing to sync in this worktree). Needs syncing on whatever host runs
  the live service.
- skipped keys requiring operator action: none.

## Tests run

```text
$ pytest services/orion-field-digester/tests -q
100 passed, 15 warnings

$ PYTHONPATH=.:services/orion-field-digester pytest tests/test_field_*.py -q
29 passed, 1 failed (test_field_topology_config.py::test_canonical_and_alias_lattice_load_same_edges)
```

The one failure is pre-existing and unrelated (a lattice-edge-count mismatch between
`config/field/orion_field_topology.v1.yaml` and `config/field/biometrics_lattice.yaml`) —
confirmed present on `main` independent of this branch.

## Evals run

None. `services/orion-field-digester` has no `evals/` directory (pre-existing gap, not
introduced by this PR). Per CLAUDE.md §11's fallback clause, reporting this rather than
building an eval harness from scratch in this patch — flagging as a follow-up.

## Docker/build/smoke checks

Not run — Docker was not available to exercise in this environment. `docker-compose.yml`
change reviewed manually (env passthrough line added, matches existing pattern exactly). No
port, health-check, or dependency changes.

## Review findings fixed

- Finding: `field_coherence_warning` (`worker.py`) is a `NODE_DECAY_CHANNELS` entry written
  directly to `node_vectors`, bypassing `apply_perturbations()` — never got a
  `node_vector_updated_at` stamp, so it silently kept decaying every tick unconditionally,
  missing the fix entirely (found independently by 3 of 8 review angles).
  - Fix: record the stamp at the direct-write site in `worker.py`.
  - Evidence: `test_worker.py::test_field_coherence_warning_records_node_vector_updated_at`.
- Finding: `staleness` (`suppression.py`'s reset-to-0.0 on suppression) has the same
  direct-write gap. Currently inert (decaying 0.0 is a no-op) but a latent risk.
  - Fix: record the stamp at the direct-write site in `suppression.py`.
  - Evidence: `test_decay_hold.py::test_suppression_staleness_reset_records_node_vector_updated_at`.
- Finding: `docker-compose.yml` never passed `FIELD_DECAY_STALENESS_THRESHOLD_SEC` through to
  the container — an operator's `.env` override would be silently ignored in real deployment
  (CLAUDE.md §7 env parity).
  - Fix: added the passthrough line alongside `BIOMETRICS_FIELD_DECAY_RATE`.
  - Evidence: manual diff review, matches existing pattern.
- Finding: `perturbation.py`'s `node_vector_updated_at` write was triplicated identically
  across all three write branches.
  - Fix: hoisted to a single line after the `if`/`elif`/`else`.
  - Evidence: `pytest services/orion-field-digester/tests -q` still green post-refactor.
- Finding: `decay.py` duplicated tz-awareness normalization inline for `now` and
  `last_updated_at`, and had two separate `vec[ch] = vec[ch] * decay_rate` call sites.
  - Fix: extracted `_aware_utc()` helper, merged both decay branches into one `is_fresh` check.
  - Evidence: `test_decay_hold.py`'s 4 original tests still pass unchanged post-refactor.
- Finding (not fixed, reported): `tests/test_field_deterministic_replay.py`'s `_replay()`
  helper never advances `state.generated_at`, so it can't exercise the staleness-crossing
  branch through that particular test.
  - Fix: none — `test_decay_hold.py` already covers that branch directly and thoroughly; not a
    production bug, just a blind spot in one pre-existing test's fixture.
  - Evidence: n/a.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-field-digester/.env \
  -f services/orion-field-digester/docker-compose.yml \
  up -d --build
```

## Risks / concerns

- Severity: medium.
  Concern: the altitude review angle flagged that `FIELD_DECAY_STALENESS_THRESHOLD_SEC=90s` is
  a single global threshold applied across all 19 `NODE_DECAY_CHANNELS`, derived only from
  biometrics' publish cadence (15-30s) — the 13 execution/transport channels' real update
  cadence was never measured. If any of those channels legitimately update sparser than ~90s,
  they'd still sawtooth, just on a longer period.
  Mitigation: the spec's stated rationale still holds (this can only make staleness detection
  more honest, never worse, versus the prior unconditional-every-2s decay) — but this is
  unverified for non-biometrics channels, not resolved. Flagged as a follow-up if live data
  shows a problem.
- Severity: medium.
  Concern: this fix's real-world impact on the drive economy (whether it drops fold-batch
  tension volume enough to make `DriveEngine`'s fold-batch clamp collapse a non-issue) has not
  been measured live yet.
  Mitigation: retrospective §5c and §6 item 5 explicitly record this as the next step before
  deciding on the `DriveEngine` fix — re-run `scripts/analysis/measure_autonomy_gate.py`
  against a post-deploy window.
- Severity: low.
  Concern: no eval harness exists for this service.
  Mitigation: reported per CLAUDE.md §11 rather than built in this patch; a follow-up if this
  service's failure modes prove hard to catch with unit tests alone.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/field-digester-decay-hold-fix
