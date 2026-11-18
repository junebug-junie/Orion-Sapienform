# PR report: missing perception ticks leave the field channel absent, not 0.0

## Summary

- When the camera input is missing, the field no longer holds a calm-looking 0.0 for the two camera surprise nodes. The channel is now absent, which every reader treats as "unmeasured".
- The camera tick (orion-substrate-runtime) still sends one receipt every 30 s. On a missing tick the receipt names the reason and carries no value.
- The field digester drops the node's `prediction_error` channel and its write stamp when it sees that reason (new perturbation mode `clear`).
- Regression tests on both halves. One old test pinned the leak (it asserted the 0.0) and is updated.
- Implements retire step 8 of the PE suite rebuild spec (PR #3168, `docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md`, unmerged): "missing ticks must not write 0.0 into the field".

## Outcome moved

A dead or warming camera no longer reads as a calm camera in the field. Before: 0.0. After: no value.

Live evidence of the leak (before), from Postgres:

- The room camera (`cam0`) and walkway camera sent no frames from 2026-11-08 08:59:58 to 09:01:33 UTC (95 s), per `substrate_perception_embedding_baseline`, which gets one row per scored frame.
- The field (`substrate_field_state`) held `prediction_error = 0.0` for both `node:substrate.perception` and `node:substrate.perception_walkway` on the writes at 09:00:33 (no frame in the window), 09:01:03 (input stale) and 09:01:34 (walkway stale).
- Over the 3 days of retained field state (2026-11-07 06:58 to 11-10 07:31), 8,574 field writes per node were classified by input age. 3 writes fell in an input gap, and all 3 were 0.0. Every other write had live input.
- The graph node and the 7-day history were already protected on 2026-10-21 (`prediction_error_missing_reason`). The field was the remaining leak. A larger instance (no embeddings for 49 h, 2026-10-18 to 10-20) is older than the field-state retention, so it cannot be re-checked in the field table.
- Most field zeros are real. 5,160 of 8,572 room writes with live input are 0.0, because the score clips negative z-scores to 0. Those stay as real 0.0 readings. This fix touches only writes with no input.

## Current architecture

- `_publish_perception_group` (substrate-runtime) runs every 30 s for each camera node. On missing input it wrote a 0.0 placeholder to two places: the FalkorDB graph node, marked with `prediction_error_missing_reason`, and the field receipt, with no mark.
- The digester turned every `prediction_signal` receipt into a `replace` perturbation, so the unmarked 0.0 landed in `node_vectors[node]["prediction_error"]`.

## Architecture touched

- `services/orion-substrate-runtime`: the producer of the field receipt.
- `services/orion-field-digester`: the consumer of the receipt and the owner of the field write.
- Receipt payload: one optional key in `StateDeltaV1.after` (a free-form dict). No schema model, registry or bus channel change.

## Files changed

- `services/orion-substrate-runtime/app/worker.py`: `_prediction_error_receipt` takes `missing_reason` and needs exactly one of value or reason. The perception tick passes the reason and no value on missing ticks.
- `services/orion-field-digester/app/ingest/state_deltas.py`: a `prediction_signal` delta with a missing reason produces a `clear` perturbation. A reason wins over a value sent with it.
- `services/orion-field-digester/app/digestion/perturbation.py`: the `clear` mode drops the channel and its stamp, and never creates a node.
- `services/orion-field-digester/app/digestion/decay.py`: the camera nodes' `prediction_error` expires after 180 s with no write (node-scoped), so a stopped producer cannot hold a spike.
- `services/orion-field-digester/tests/test_prediction_signal_missing_is_absent.py`: new regression tests for the digester.
- `services/orion-substrate-runtime/tests/test_worker_perception_honest_input.py`: new regression tests for the producer receipt.
- `services/orion-substrate-runtime/tests/test_worker_perception_prediction_error.py`: the old assertion of a 0.0 receipt is replaced by the new contract.
- `services/orion-field-digester/README.md`, `orion/substrate/prediction_error_freshness.py`, `orion/attention/world_first.py`, `services/orion-attention-runtime/app/worker.py`: docs and docstrings brought in line with the change. No logic change in attention-runtime.

## Schema / bus / API changes

- Added: optional `after["prediction_error_missing_reason"]` on `prediction_signal` state deltas. It uses the existing key `PREDICTION_ERROR_MISSING_REASON_KEY`. When present, `pressure_hints` carries no `prediction_error`.
- Removed: none.
- Renamed: none.
- Behavior changed: the digester removes the field channel on such a delta. Before, it wrote 0.0. The camera nodes' field `prediction_error` also expires after 180 s with no write.
- Compatibility notes:
  - New producer with old digester: the old digester sees no value and writes nothing, so the field holds the last real reading. The new digester expires it after 180 s, so rebuild the digester first.
  - Old producer with new digester: behaviour is the same as today.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, because no template changed.
- Skipped keys requiring operator action: none.

## Consumer audit (absence handling)

Every production reader of field `node_vectors` that could see these nodes was checked. None crashes on a missing key or on an empty vector `{}`. None turns the missing perception value into a 0.0 that matters. The camera nodes are not in the static lattice, so `reconcile._ensure_node_vector` never seeds a default 0.0 back. The attention runtime admits a node only when `"prediction_error" in node_vectors[k]` or when history has a value. Pressure, commensurability, tension and diffusion loop over the keys that are present. Two readers fall back to 0.0 only when no node at all has the channel (mood-arc encoder features, the Hub glossary series). That does not happen, because other substrate nodes always carry the channel. A side effect is that `orion/field/pressure.py` no longer lets the perception placeholder win provenance of the merged `prediction_error` on all-zero ties.

Out of scope, unchanged: the graph node still stores a 0.0 placeholder plus the reason. Its readers already omit it through `classify_reading`.

## Tests run

```text
PYTHONPATH=services/orion-field-digester:. pytest -q services/orion-field-digester/tests
  297 passed, 6 skipped
PYTHONPATH=services/orion-substrate-runtime:. pytest -q services/orion-substrate-runtime/tests --ignore=tests/test_grammar_consumer_integration.py
  565 passed, 17 failed, 5 skipped
  The same 17 tests fail on main at b79fe1a02 (diffed, identical). test_grammar_consumer_integration fails to collect on main too (app.models import).
Focused perception tests: 47 passed.
The new digester tests run against the pre-fix digester code: 3 of 6 fail, which proves they catch the leak.
```

## Evals run

```text
None. Neither service has an eval for the field camera channel. The live check below is the acceptance check.
```

## Docker/build/smoke checks

```text
Not run. This PR is not deployed (the task says do not merge or deploy).
Live acceptance after deploy, during the next camera gap: field_json->'node_vectors'->'node:substrate.perception' has no 'prediction_error' key, and the substrate-runtime log shows `..._tick_completed ... missing=stale|warming|no_scored_frame_since_last_tick` at the same time. UNVERIFIED until then.
```

## Review findings fixed

A code review ran in a subagent. It found no must-fix issues. Each finding and what was done:

- Finding (should-fix): deploy order. A new producer with an old digester sends no value, and the old digester holds the last real reading forever, because `prediction_error` never decays or expires. The same happens when substrate-runtime stops.
  - Fix: node-scoped expiry. `EXPIRING_NODE_CHANNELS_BY_NODE` in `decay.py` drops the camera nodes' `prediction_error` after 180 s (six missed ticks) with no write. It is scoped by node, so slow writers such as `node:substrate.route` still hold their value. The restart section now says to rebuild the digester first.
  - Evidence: `test_a_stopped_producer_expires_instead_of_holding_the_last_spike` and `test_node_scoped_expiry_leaves_other_nodes_prediction_error_alone`.
- Finding (nit): the new attention-runtime comment gave the wrong reason why perception cannot win. The candidate magnitude comes from history, not from the field key.
  - Fix: comment reworded to give the history reason.
  - Evidence: `services/orion-attention-runtime/app/worker.py` docstring of `_camera_absent_reason`.
- Finding (nit): `orion/attention/world_first.py` said that perception writes a 0.0 placeholder, with no qualifier.
  - Fix: the docstring now says "graph node only; the field key is dropped".
  - Evidence: diff.
- Finding (nit): one edited comment line ran to 117 characters.
  - Fix: rewrapped.
  - Evidence: diff.
- Finding (nit): if a camera node is ever added to the lattice `nodes:`, reconcile puts 0.0 back.
  - Fix: `test_the_camera_nodes_stay_off_the_lattice`.
  - Evidence: test passes.
- Finding (nit): no test ran a clear through the full digestion tick.
  - Fix: `test_missing_stays_absent_through_a_full_tick` runs reconcile, the digestion tick and a stored-form round trip against the live topology.
  - Evidence: test passes.
- Finding (not applied): the reviewer said attention-runtime's receipt-history precision baseline would stop folding the placeholder zeros. That baseline reads only `PREDICTION_ERROR_NATIVE_TARGETS` (biometrics, execution, chat, route, bus_synaptic). Perception is not in that set, so nothing changes there. Both receipt readers in `services/orion-attention-runtime/app/store.py` already skip NULL values.

## Restart required

```bash
scripts/safe_docker_build.sh orion-field-digester up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Run these from the clean primary `main` checkout after merge. Rebuild the digester first.

## Risks / concerns

- Severity: low. Concern: while the camera input is missing, the field holds no value for these nodes. A future reader that does `vec["prediction_error"]` would crash. Mitigation: the audit found no such reader. The README documents absence.
- Severity: low. Concern: the deploy order gap described under compatibility. Mitigation: rebuild the digester first. After that, the 180 s expiry covers any producer that stops.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3171

🤖 Generated with [Claude Code](https://claude.com/claude-code)
