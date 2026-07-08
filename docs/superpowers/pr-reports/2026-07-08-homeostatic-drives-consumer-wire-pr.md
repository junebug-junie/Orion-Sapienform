# PR: Homeostatic drives — live signal consumer (drive-only rail)

**Status:** IMPLEMENTED + reviewed. Completes Task 6 (live consumer) and Task 9
(review). Task 10 (host live-verify) is the remaining step and needs the running
host — instructions below.

## Summary

- Concept-induction worker now **subscribes to specific organ/failure channels**
  (`orion:signals:biometrics|spark|equilibrium`, `orion:system:error`,
  `orion:rdf:error`, `orion:vision:edge:error`) — **never** the `orion:signals:*`
  wildcard, so the 55/s `scene_state` flood is excluded at the subscription.
- Those envelopes route to a new **`_handle_signal_drive_tick`** — a thin
  drive-only rail: mint deviation/failure tensions → update + publish drive
  state/audit → **return before concept induction** (no windowing, goals, or
  identity snapshot on a ~1/s biometric tick).
- Health degradation arrives as a `mesh_health` `OrionSignal` on the signal rail
  (deviation-gated `level↓`); no separate unverified-shape `equilibrium:snapshot`
  branch was wired.
- Flag-gated on `ORION_HOMEOSTATIC_DRIVES_ENABLED`; degrades to a no-op, never
  raises into the bus loop.

## Outcome moved

The homeostatic substrate (merged in #879) is now **fed by real bus traffic**.
Before this PR the leaky-math engine only saw the existing spark/feedback/metabolism
tensions; now biometrics, spark, mesh-health, and system failures drive pressure
through the deviation gate — the "make it real, not tick-based" goal.

## Architecture touched

- `orion/spark/concept_induction/bus_worker.py` — `_handle_signal_drive_tick`,
  `_parse_signal`, `_homeostatic_source`, `_homeostatic_channels`,
  `_pubsub_patterns` (+homeostatic channels when enabled), `handle_envelope`
  early-return branch.
- `orion/spark/concept_induction/settings.py` — `homeostatic_signal_channels`,
  `homeostatic_failure_channels`, `homeostatic_failure_severity`.
- `services/orion-spark-concept-induction/.env_example`, `orion/bus/channels.yaml`.
- Tests: `orion/spark/concept_induction/tests/test_signal_drive_consumer.py`.

## Schema / bus / API changes

- No new schemas. Reuses `OrionSignalV1`, `TensionEventV1`, `SignalTensionSource`.
- Consumed channels are specific organ/failure channels (documented in channels.yaml).

## Env/config changes

- Added: `HOMEOSTATIC_SIGNAL_CHANNELS` / `HOMEOSTATIC_FAILURE_CHANNELS` (JSON list
  overrides; defaults apply), `HOMEOSTATIC_FAILURE_SEVERITY=0.8`.
- `.env_example` updated. Run `python scripts/sync_local_env_from_example.py` on
  the host post-merge (worktree has no local `.env`).

## Tests run

```text
pytest orion/spark/concept_induction/tests orion/autonomy/tests -q
261 passed  (incl. 7 new consumer tests: subscription/no-wildcard, source
classification, biometric-drop→drives-no-induction, steady=no-op, failure→tension,
flag-off fall-through, never-raise-on-broken-store)
```

## Review findings fixed (Task 9, subagent)

- **MAJOR** — `_handle_signal_drive_tick` drive-update/publish section was unwrapped;
  a tz-naive stored `updated_at` or store fault would raise into the bus loop
  (which re-raises) and tear down the subscription, contradicting the never-raise
  contract on a ~1/s rail.
  - Fix: wrapped the section in `try/except → log + return`; normalize
    `previous_ts` to tz-aware.
  - Evidence: new `test_never_raises_when_drive_update_breaks` (tz-naive state +
    `save_drive_state` raising → no exception). 261 tests green.
- **NIT** — channels.yaml comment read self-contradictory (service on the wildcard
  entry vs "never the wildcard"). Reworded to clarify registration vs runtime
  subscription.
- **NIT** — flag defaults ON: intended (Juniper authorized flags-on).

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml up -d --build
```

## Task 10 — live host verification (remaining)

After restart, tap live for ~60s and confirm:
- `drive:audit` / drive_state shows **differentiated, non-pinned** pressures that
  move with biometrics/mesh-health/failures and decay toward ~0 in quiet windows
  (NOT all six pinned at 0.7309).
- `orion:signals:vision` / scene_state flood mints **0** homeostatic tensions
  (channel never subscribed).
- `dominant_drive` reflects real events (not constant "autonomy"/None).

## Risks / concerns

- Severity: low — consumer degrades to no-op and is flag-gated; scene_state flood
  structurally excluded; 261 tests + subagent review clean.
- Live behavior against the real 55/s reality is unverified until Task 10 on host.

## PR link

<to be filled by `gh pr create` / GitHub UI>
