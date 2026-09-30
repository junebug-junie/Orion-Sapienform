## Summary

- The transport lane can now learn what "normal" looks like for `bus_synaptic_pressure` and trigger earlier than the static yaml number when normal is low. It can never trigger later: the static yaml value is a hard ceiling on every rung.
- Two clocks per channel, both from `orion.bus.ewma.compute_ewma_update`: fast (30 min half-life, reports "just changed" z) and slow (2 day half-life, the chronic level and spread). Thresholds come from the slow clock; z is reported, never used to quiet anything.
- Only `bus_synaptic_pressure` is wired. `contract_pressure` (flat, 5 distinct values) and `observer_failure_pressure` (96% exact zeros) failed the metric gate and stay static.
- Cold start (under 2880 samples, 24 h) or stale state (over 600 s old) or Redis down: static thresholds apply, z reads unknown (None), never 0.0.
- One flag, `TRANSPORT_THRESHOLDS_DERIVED_ENABLED=false`, restores pure static yaml in hub and mind.
- Hub lattice tab shows a provenance line per channel (source static|derived, half-life window, sample count, z_fast or "unknown", CHRONIC-HOT marker). The mind recall sentence says "(learned from recent history)" when derived.

## Outcome moved

Replayed on the real 3.4 days of data: default k=5 makes the watch rung fire on 0.77% of ticks (derived ~0.19-0.22) versus 0.19% at the static 0.25, i.e. earlier and still rare. A channel that has been hot for days keeps tripping at the static number even though its z drops to ~0 (regression test).

## Current architecture

Static per-channel thresholds in `config/substrate-lattice/transport_lattice_policy.v1.yaml`, read by the hub lattice routes (gate overlay, panel, simulator) and the mind recall resolver (display rungs). No baseline of any kind.

## Architecture touched

- New shared pure module + tiny Redis IO: `orion/field/transport_thresholds.py` (state in Redis hash `orion:lattice:transport_thresholds:v1`, field per channel).
- Producer: `services/orion-substrate-runtime/app/worker.py::_bus_synaptic_tick` folds `error * 0.85` (the channel's own scale, `capability:transport.pressure`) into the state each 30 s tick, fail-open.
- Readers: hub `_effective_channels` (gates, panel rows, simulator), mind `_effective_bus_synaptic_rungs`. Both call the same `fetch_effective_thresholds`.
- No FieldStateV1, schema, bus channel, or yaml change. The draft-policy-patch still diffs the static yaml (that is the file being edited).

## Files changed

- `orion/field/transport_thresholds.py`: new; state update, effective thresholds, Redis IO.
- `services/orion-hub/scripts/substrate_lattice_routes.py`, `static/js/substrate-lattice.js`: use effective thresholds, show provenance.
- `services/orion-mind/app/recall_signal_resolver.py`: use effective rungs.
- `services/orion-substrate-runtime/app/worker.py`: producer hook.
- `.env_example` and `docker-compose.yml` in those three services: 8 new keys.
- `tests/test_transport_thresholds.py` and service tests/conftests (conftests force the flag off so unit tests never touch live Redis).
- `docs/superpowers/specs/2026-09-30-ewma-transport-thresholds-gate.md`: Step 0 metric gate with live numbers.

## Schema / bus / API changes

- Added: `threshold_provenance` on each row of `/api/substrate-lattice/transport/latest` `lattice_channels` (additive JSON; consumer is the same-PR JS).
- Removed / Renamed: none.
- Behavior changed: hub gate/panel/simulator `watch_at` for `bus_synaptic_pressure` may be lower than yaml once warm.
- Compatibility: flag off is byte-identical to previous values.

## Env/config changes

- Added keys (hub, mind, substrate-runtime): `TRANSPORT_THRESHOLDS_DERIVED_ENABLED=true`, `_FAST_HALF_LIFE_SEC=1800`, `_SLOW_HALF_LIFE_SEC=172800`, `_MIN_SAMPLES=2880`, `_K_WATCH=5.0`, `_K_STEP=2.0`, `_MAX_STATE_AGE_SEC=600`, `_STATE_KEY=orion:lattice:transport_thresholds:v1`.
- `.env_example` updated: yes, all three. Local `.env` synced with `python3 scripts/sync_local_env_from_example.py --all-keys orion-hub orion-mind orion-substrate-runtime` (default mode skips these prefixes). Note: that script writes to the primary checkout's local .env files (gitignored).
- Skipped keys: none. Two unrelated pre-existing diverged keys (HUB_ROOM_CLAUDE_ENABLED, MIND_LLM_SYNTHESIS_ENABLED) left alone.

## Tests run

```text
tests/test_transport_thresholds.py                       17 passed
services/orion-hub/tests/test_substrate_lattice_routes.py + hub_tab   all passed (3 new + 1 new)
services/orion-mind/tests                                108 passed
services/orion-substrate-runtime/tests/test_worker_bus_synaptic_tick.py  10 passed
python3 scripts/check_env_template_parity.py             PASS
```
Full substrate-runtime suite: 16 failures + 1 collection error exist identically on unmodified main (compared before/after); none from this change.

## Evals run

No separate eval harness for this seam. The live-data replay in the gate doc (3.4 days, 124k rows) stands in; it is a one-off script, not a committed eval. UNVERIFIED as a repeatable eval.

## Docker/build/smoke checks

Not run (no deploy allowed). Live path UNVERIFIED: no state has been written to Redis yet, so hub/mind will report static (`no_state`) until substrate-runtime is restarted and 24 h of samples accrue.

## Review findings fixed

See the review section appended below.

## Restart required

```bash
# after merge, from the deployed checkout (Juniper runs these; not run here)
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```
Order: substrate-runtime first so state starts accruing. Verify: `redis-cli -u $ORION_BUS_URL hget orion:lattice:transport_thresholds:v1 bus_synaptic_pressure` shows a growing `n`; hub lattice tab shows "source=static (cold_start)" for 24 h then "derived".

## Risks / concerns

- Severity: medium. Only 3.4 days of history exist; a 2-day half-life and any weekly cycle are unproven. k and half-lives are knobs, not findings.
- Severity: low. Pre-existing scale mismatch: mind renders node error (0..1) against rungs defined on the 0.85x M4 scale; thresholds themselves are identical in both readers.
- Severity: low. Redis hiccup adds up to ~0.5 s on a hub gates request (then static).

## PR link

TBD
