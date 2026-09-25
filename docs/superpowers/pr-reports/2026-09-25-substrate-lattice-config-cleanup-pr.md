## Summary

- Deleted two substrate-lattice config files that nothing read: `action_ceiling_policy.v1.yaml` and `grammar_producer_registry.v1.yaml`. The registry was also wrong.
- Removed keys that nothing read. From `transport_lattice_policy.v1.yaml`: `dimension_weights`, `required_windows` and `healthy_idle`. From `gate_policy.v1.yaml`: `confidence`, `lineage` and `action_ceiling`.
- Deleted the Hub's hand-copied `_TRANSPORT_CHANNELS` mirror, which was still keyed on the retired `stream_backlog_pressure` channel.
  - The Substrate Lattice tab's threshold simulator, Lattice Values panel and simulator inputs now read the channel list and thresholds straight from the policy YAML.
  - They read real values. Before this change they read fields that do not exist and always showed 0.0 or "—".
- Fixed the producer list for `orion:grammar:event` in `orion/bus/channels.yaml`: removed three vision services that never emit grammar events, and added `orion-substrate-runtime`.
  - A new CI gate compares that list against a static scan of the code.
- Removed the dead L6 (self-state) rung from the proof-ladder doc and the smoke script, and recorded the production flag state.
- Committed the 2026-09-22 substrate lattice audit as a design spec.

## Outcome moved

- **The simulator was blind and now measures.** It used to read M3's top level, which has no pressure fields, so it always said "suppressed, salience 0" whatever the bus was doing.
  - Live read against production after this patch: `bus_synaptic_pressure` 0.029 (from M4), `catalog_drift_pressure` 0.0036 (M3 per-bus), both quiet. The four channels come straight from the YAML.
- **"Unmeasured" no longer reads as "calm".**
  - If M3 or M4 is stale, or a bus row's own `observed_at` is stale, the channel shows as unmeasured and cannot trigger anything.
  - This excludes the phantom `bus:rpc_timeout` row, which was live today with a timestamp 4 hours old.
- **The producer catalog can no longer drift silently.** A CI gate now fails if it does. I checked this by mutation: re-adding a phantom producer or dropping a real one turns the test red.

## Current architecture

- `config/substrate-lattice/` held four YAMLs.
  - The Hub read two of them, and only partly: `transport_lattice_policy` for `channels`, `gate_policy` for freshness and evidence.
  - `orion-mind`'s recall resolver reads the `bus_synaptic_pressure` thresholds.
  - Nothing read the other two files, or the weights.
- The Hub's simulator used a hardcoded channel dict with weights, and the JS kept a third copy.
- The `orion:grammar:event` producer list was edited by hand and was wrong in both directions.

## Architecture touched

- **Hub, Substrate Lattice tab.** Changed the API, the JS and the HTML.
- **Bus catalog.** Changed the `producer_services` list only. No schema or payload change.
- **Config.** Deleted keys and files.
- **Docs, smoke script and CI workflow.**

## Files changed

- `config/substrate-lattice/action_ceiling_policy.v1.yaml`: deleted (no loader).
- `config/substrate-lattice/grammar_producer_registry.v1.yaml`: deleted (no loader, and wrong).
- `config/substrate-lattice/transport_lattice_policy.v1.yaml`: removed the unread keys. The header now names the live readers.
- `config/substrate-lattice/gate_policy.v1.yaml`: removed the unread gate keys.
- `services/orion-hub/scripts/substrate_lattice_routes.py`: this is the main code change.
  - Removed: the `_TRANSPORT_CHANNELS` mirror.
  - Added:
    - `_policy_channels`, which loads the channel list from the YAML.
    - `_channel_value`, which reads each channel's value and checks per-bus freshness.
    - `_lattice_channel_rows`, which builds the rows for the Lattice Values panel.
  - Salience is now the strongest promoted reading.
  - `/transport/latest` gains `lattice_channels`.
  - The simulator reports unmeasured and ignored channels.
  - Gate reasons now name what they actually read.
  - The lane rail statuses are corrected.
- `services/orion-hub/static/js/substrate-lattice.js` and `static/substrate-lattice.html`:
  - The panel and the simulator inputs render from the server's data.
  - The M3 card reads per-bus rows.
  - Unmeasured channels are surfaced.
  - Quotes are escaped.
- `services/orion-hub/tests/test_substrate_lattice_routes.py` and `test_substrate_lattice_hub_tab.py`:
  - The fixture now uses the real M3 shape.
  - New tests cover:
    - kill-means-kill (the deleted mirror and weights stay gone)
    - single-source reads from the YAML
    - stale and phantom bus rows
    - unmeasured channels never promoting
    - retired threshold keys being ignored
    - a malformed policy file
    - the UI hardcoding no policy values
- `orion/bus/channels.yaml`: corrected the `orion:grammar:event` producers.
- `tests/test_grammar_event_producer_catalog.py`: new static-scan gate.
- `.github/workflows/orion-static-gates.yml`: runs the new gate.
- `config/metrics/metric_definitions.lock.json`: re-locked. The drift gate flagged the producer change as `routing_changed`, which is expected.
- `orion/core/bus/async_service.py`, `services/orion-heartbeat/app/substrate/routing.py` and `orion/sentience_striving_program/README.md`: comments that pointed at deleted files or made false claims.
- `services/orion-attention-runtime/.env_example`: a stale comment only. No key changed.
- `docs/transport_substrate_proof_ladder.md` and `scripts/smoke_orion_bus_transport_full_stack.sh`: L6 removed; production flag state added.
- `docs/superpowers/specs/2026-09-22-substrate-lattice-audit.md`: the audit.

## Schema / bus / API changes

- **Added:**
  - `/api/substrate-lattice/transport/latest` returns `lattice_channels`, plus `lattice_policy_error` if the policy file is malformed.
  - `/transport/simulate` returns `promoted_channels`, `unmeasured_channels`, `ignored_thresholds` and `channel_values`.
  - A `route` lane on `/lanes`.
- **Removed:** from the `orion:grammar:event` producers: orion-vision-retina, orion-vision-edge, orion-vision-window.
- **Added (catalog):** orion-substrate-runtime.
- **Behavior changed:**
  - Simulator salience is now the strongest reading among channels at or above their watch threshold. It was a weighted sum.
  - A threshold key that names no policy channel is reported and not applied.
  - A malformed policy makes `/transport/simulate` return 503.
- **Compatibility:** the Hub is the only client of these endpoints, and its JS ships in the same change. No bus payload changed.

## Env/config changes

- Added, removed or renamed keys: none.
- `.env_example` updated: comment only, in `services/orion-attention-runtime/.env_example`.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, a no-op for this change. Env template parity PASS.
- Skipped keys: none.

## Tests run

```text
services/orion-hub: pytest tests/test_substrate_lattice_routes.py tests/test_substrate_lattice_hub_tab.py  -> 68 passed
pytest tests/test_grammar_event_producer_catalog.py                                                      -> 3 passed
  mutation: phantom orion-vision-retina + drop orion-substrate-runtime                                    -> 1 failed (named both)
services/orion-mind: pytest tests/test_recall_signal_resolver.py                                         -> 22 passed
services/orion-heartbeat: PYTHONPATH=<root> pytest tests                                                 -> 119 passed
pytest tests/test_grammar_truth_gate.py tests/test_agent_trace_schema_registry.py \
       tests/test_vision_scene_inventory_contract.py tests/test_vision_edge_activity_bus_catalog.py \
       orion/core/bus/tests/test_rpc_timeout_grammar_emit.py                                             -> passed
scripts/check_metric_lineage.py --gate      -> PASS
scripts/check_definition_drift.py --gate    -> PASS (after --update)
scripts/check_env_template_parity.py        -> PASS
node --check static/js/substrate-lattice.js -> ok
bash -n scripts/smoke_orion_bus_transport_full_stack.sh -> ok
```

## Evals run

```text
No eval harness covers the Substrate Lattice tab. Instead, the patched route
functions were run read-only against production Postgres: real M3/M4 values
appear in lattice_channels, and the stale bus:rpc_timeout row is excluded.
Live grammar_events source_service counts (7d) match the new catalog exactly;
vision services have 0 rows in 60 days.
```

## Docker/build/smoke checks

```text
Not rebuilt. Hub runtime change is Python route + static JS/HTML; no dependency,
port, or compose change. Hub restart required to pick it up (below). UNVERIFIED in
the running container until restarted.
```

## Review findings fixed

The review ran in a subagent against commit 2cf272d14.

- **Finding:** The per-bus max included stale and phantom bus rows. `bus:rpc_timeout` was 4 hours old and would have held its last value in the max.
  - **Fix:** `_bus_row_is_fresh()` skips any row whose own `observed_at` is older than the freshness threshold. If every row is stale, the channel is unmeasured.
  - **Evidence:** `test_channel_value_ignores_stale_bus_rows` and `test_channel_value_unmeasured_when_every_bus_row_stale` pass, and the live read excludes `bus:rpc_timeout`.
- **Finding:** The gate overlay labels M4 `reliability_pressure` as `observer_failure_pressure`, and labels M4 `contract_pressure` (which is fed by catalog drift) as contract pressure. The panel and the gates can disagree under the same name.
  - **Fix:** gate reasons now name what they read. The underlying vocabulary bug is recorded as an open item in the audit.
  - **Evidence:** live gate reasons.
- **Finding:** The simulator UI showed "✓ no change" when every channel was unmeasured.
  - **Fix:** the result panel now shows unmeasured channels and ignored thresholds. When any channel is unmeasured it says "no change among measured channels" instead.
  - **Evidence:** `test_lattice_simulator_surfaces_unmeasured_and_ignored` passes.
- **Finding:** On Refresh, the simulator inputs kept stale values it had never been told to keep, and a failed fetch wiped the operator's edits.
  - **Fix:** only inputs marked `data-dirty` survive a Refresh, and the error path leaves the inputs alone.
- **Finding:** `_esc` did not escape quotes but is used inside HTML attributes, and `_fmt` returned strings unescaped.
  - **Fix:** both are escaped now.
- **Finding:** A malformed policy YAML would return a 500 from `/transport/latest`.
  - **Fix:** that endpoint now returns `lattice_channels: []` plus `lattice_policy_error`, and simulate returns 503.
  - **Evidence:** `test_latest_survives_malformed_policy` passes.
- **Finding:** The catalog gate keyed dynamic call sites by file only, so a second dynamic site in the same file would pass unnoticed.
  - **Fix:** it now counts dynamic sites per file.
- **Finding:** The lanes test forced every lane to "live".
  - **Fix:** it now checks that a lane is "live" exactly when its producer is in the catalog.
- **Finding:** The audit doc overstated "four unread files", and its L11 input attribution was muddled.
  - **Fix:** corrected. L11 reads M4 `contract_pressure`, which is fed by catalog drift.

## Restart required

```bash
# Hub (route + static assets). Run from a worktree per safe_docker_build.sh policy:
scripts/safe_docker_build.sh orion-hub up -d --build
```

No other service needs a restart. The bus catalog is data read by gates and tests. `orion-mind` still reads the same `bus_synaptic_pressure` keys.

## Risks / concerns

- **Medium: merge conflict.** `orion/core/bus/async_service.py` touches a docstring line next to the one the sibling `fix/transport-rpc-timeout-phantom-node` edits. Git will flag a trivial conflict; keep both sides.
- **Low: sibling branch may trip the new gate.** `feat/llm-gateway-grammar-lane` will add a grammar emitter. The new catalog gate will fail that branch unless it also adds `orion-llm-gateway` to `orion:grammar:event`. That is intended.
- **Low: leftover directory not deleted.** The untracked, gitignored `services/orion-self-state-runtime/` in the primary checkout holds only `.env` and `__pycache__`. It was not deleted because that needs `rm -rf`, which needs Juniper's approval.
  - Command: `rm -rf /mnt/scripts/Orion-Sapienform/services/orion-self-state-runtime`
- **Info: schema kept.** `orion/schemas/self_state.py` is kept because other modules still import it.
- **Open items, carried in the audit and not fixed here:**
  - the dead capability→orchestration edge (needs the metric quality gate)
  - the transport proposal-template dimensions (needs proposal mode)
  - the unreachable L11 threshold
  - the narrow M3 observer, which watches only two world_pulse streams
  - the three names used for one transport signal

## PR link

See the GitHub PR for this branch (`chore/substrate-lattice-config-cleanup`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
