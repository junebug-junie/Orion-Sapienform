## Summary

- circe was described as a "burst" GPU. It carries the bulk of Orion's daily cognition. `config/biometrics/node_catalog.yaml` role `burst_gpu` -> `primary_cognition_gpu`.
- The topology file Orion learns "my body" from (`config/field/orion_field_topology.v1.yaml`, read by cortex-exec `self_study.py` as `physical_topology` facts) had no hecate. Added `- node_id: hecate` as the on-demand deep agentic-thinking node, with no edge.
- The chat profile was labelled "2x V100 32GB" but runs on one card. `config/llm_profiles.yaml` display_name only -> "1x V100 32GB". Profile key unchanged (`LLM_PROFILE_NAME` on circe references it).
- `tests/test_static_body_descriptions.py` pins each fact to its file, and checks the field-digester loader parses hecate.
- athena's Tesla P4 + T10 (vision) are NOT added to `config/gpu_pool.yaml`: every section of that file is pool control, there is no descriptive inventory section, and Juniper has not decided whether they become pool-managed.

## Outcome moved

Orion's static self-description stops teaching three falsehoods (circe is a side burst box; hecate does not exist; chat uses two cards) until the live body map (PR #2648, `docs/superpowers/specs/2026-10-14-live-body-map-and-self-improvement-reading-design.md`) replaces these files.

## Current architecture

Static YAML files are the source of Orion's hardware self-facts: node_catalog (biometrics roles/expected_online), field topology (lattice nodes/edges, also self_study physical_topology), llm_profiles (profile labels).

## Architecture touched

Config only. No schema, bus, env, or code changes.

## Live verification (2026-10-14)

- `ssh circe@circe nvidia-smi -L`: 4 GPUs (2x V100-PCIE-32GB, PG500-216, V100-SXM2-32GB); compute apps on all four (llama-server x4, python3 on gpu2).
- circe `orion-circe-atlas-llamacpp-chat`: `LLM_PROFILE_NAME=qwen36-35b-a3b-udq5km-2xv100-32gb-deep-cognition`, `CUDA_VISIBLE_DEVICES_OVERRIDE=0` (one card, ~28 GB used on gpu0).
- hecate: no ping reply (on-demand, off).
- athena `nvidia-smi -L`: Tesla P4, Tesla T10.
- Live field (`substrate_field_state`, latest tick 2026-10-14 01:14Z): `node:hecate` already exists off-lattice since 2026-10-11 with `expected_offline_suppression=1.0`. Putting it on the lattice only adds the default node channels reconcile seeds for every lattice node.

## Files changed

- `config/biometrics/node_catalog.yaml`: circe role.
- `config/field/orion_field_topology.v1.yaml`: hecate node, no edge (an edge would feed a capability from seeded defaults while hecate is off).
- `config/llm_profiles.yaml`: chat display_name.
- `services/orion-cortex-exec/app/executor.py`: `_KNOWN_FIELD_NODES` gains hecate (review finding).
- `tests/test_static_body_descriptions.py`: pins.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: node_catalog role string appears in biometrics node-context summaries; nothing branches on its value.
- Compatibility notes: profile key unchanged.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced: not needed
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=services/orion-field-digester:. pytest tests/test_static_body_descriptions.py tests/test_field_topology_reconciliation.py tests/test_field_topology_edges.py tests/test_field_transport_perturbations.py tests/test_transport_thresholds.py tests/test_fleet_roster_coverage.py  -> pass
PYTHONPATH=services/orion-field-digester:. pytest services/orion-field-digester/tests  -> 284 passed, 6 skipped
PYTHONPATH=services/orion-cortex-exec:. pytest services/orion-cortex-exec/tests/test_self_study_layer1_broadening.py -> 11 passed
PYTHONPATH=. pytest orion/gpu_pool/tests -> 436 passed
ATLAS_CHAT_PROFILE_NAME=<chat key> scripts/check_fcc_context_parity.py -> ok
```

Pre-existing failures, unchanged by this patch (same on main):
- `tests/test_field_deterministic_replay.py` (2): `run_digestion_tick()` signature drift.
- `services/orion-biometrics/tests/test_node_catalog.py::test_circe_expected_offline` and `test_biometrics_grammar_emit.py::test_circe_node_availability_reflects_expected_offline`: assert circe `expected_online: false`, which has been `true` since 2026-07-18.
- `services/orion-llamacpp-host/tests` prism/metacog profile tests: other profiles and local env.

## Evals run

```text
None. Config-truth fix; the pinning test is the gate. The real eval belongs to the live body map (PR #2648).
```

## Docker/build/smoke checks

```text
Not run (do not deploy). Config is read at boot by field-digester, cortex-exec, biometrics, substrate-runtime, llamacpp hosts.
```

## Review findings fixed

Review ran in a subagent against commit dbfee80e8. No must-fix findings.

- Finding: the test claimed to cover "both consumers" but only ran the field-digester loader, and left a probe module in `sys.modules`.
  - Fix: renamed it; it now also checks `orion.evals.self_sense.load_field_node_ids()` and cleans up the module.
  - Evidence: `tests/test_static_body_descriptions.py` 4 passed.
- Finding: `services/orion-cortex-exec/app/executor.py` `_KNOWN_FIELD_NODES` mirrors the topology's node set and lacked hecate, so a reply hecate served would credit its reasoning load to the orchestrating node instead of hecate.
  - Fix: added `hecate`. Kept `atlas`, because circe's workers still carry `atlas-*` served_by labels.
  - Evidence: `test_llm_uncertainty_metadata.py` + `test_self_study_layer1_broadening.py` 18 passed.
- Finding (nit): the topology comment should say that availability 1.0 is the existing offline-suppression floor, not something new.
  - Fix: comment extended.
- Not fixed (nit, historical): four old docs still say `burst_gpu`. Left as history.

## Restart required

field-digester and substrate-runtime copy `config/field` / `config/biometrics` into their images (rebuild needed). cortex-exec has a code change. circe's biometrics container reads `config/biometrics` read-only (pull + restart on circe).

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-field-digester up -d --build
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-cortex-exec up -d --build
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
ssh circe@circe 'cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-biometrics up -d --build'
```

The chat label is display-only; no restart is needed for it. None of these restarts were verified.

## Risks / concerns

- Severity: low
- Concern: hecate's node_catalog capabilities are all `false` though it serves `agent-deep` when on; other agent-lane profiles still say "2x V100" in their labels. Not in this patch's scope.
- Mitigation: the live body map replaces these static descriptions.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2650
