## Summary

- Each metric's registry entry can now record why it reads what it reads. Five new fields cover this: `value_kind`, `rest`, `sparsity`, `absent_means` and `polarity`. Before this, nothing could tell "calm" from "dead" without reading docstrings.
- **Where the fields live.** They go on the existing registries: the field-channel glossary and the inner-state registry. No new registry. `polarity` is never declared on field channels; it is worked out from the two sets the pressure merge already uses.
- **Lock and lineage card.** The metric lock records the new fields, so the drift gate reports any change to them. The lineage card (`check_metric_lineage.py --metric X`) prints them, and says "(not recorded)" when a field is missing.
- **New CI gate.** `--prompt-semantics` fails when a metric that reaches an Orion prompt has no `value_kind`, `rest` or `sparsity`. This follows Juniper's 2026-10-10 scope.
- **Entries filled in:** 8 new node-qualified prediction_error entries, plus field-level entries on 4 existing signals and 4 newly registered ones. Every claim was checked against producer code and 3 days of live data.
- **Hub Field Channel Glossary.** It now shows one row per channel. Each node's own meaning is nested under that row, where before it was a duplicate row.

## Outcome moved

The 2026-10-07 sweep over-called 7 designed signals as "dead" because their rest state was recorded nowhere. That state is now one command away: `check_metric_lineage.py --metric <token>`. The examples below are from that sweep:

| Signal | What the entry now records |
|---|---|
| harness_closure | a 0.65 placeholder |
| cabinet | a trigger that is 97% zeros by design |
| route | a count of changed decisions |
| perception | 0 does not prove the camera works |

This closes R6 ("can a metric express rest?") with a gate. Temporal Self rev 4 (#2369) can use these lock fields for its immune sweep.

## Current architecture

The semantic layer (glossary YAML, inner-state registry, `orion/metrics/lineage.py`, metric lock, drift gate) recorded meaning, lineage and consumers. It recorded no rest value, value shape or sparsity. The glossary had one node-qualified `prediction_error` entry (bus_synaptic).

## Architecture touched

- `orion/metrics/semantics.py` (new): vocabulary, polarity derivation, the gate checks, and the prompt inventory pin.
- `MetricNode` gains `value_kind`, `rest`, `sparsity`, `absent_means`, `polarity` and `prompt_sites`.
- Glossary entries gain optional `semantics:`, `consumers:` and `producer:` keys.
- `InnerStateSignal.semantics`, plus 4 new registrations: `attention_salience_trace.v1`, `frontier_invocation_signal.v1`, `repair_pressure_appraisal.v1` and `equilibrium_snapshot.v1`.
- **Lock fields.** The first five fields are recorded in the lock as `semantics` changes, which are high severity. `prompt_sites` is recorded as a `routing` change.
- **Hub `/api/field-channel-glossary/channels`.** Node-qualified entries are nested as `node_variants`. The verdict route skips them, because they have no separate data series.

## Files changed

- `orion/metrics/semantics.py`: new; the vocabulary, the gate and the pin.
- `orion/metrics/lineage.py`: projects the new fields onto metric nodes, and derives polarity.
- `orion/metrics/definitions.py`: the lock records the new fields.
- `orion/metrics/gate.py`: adds `check_prompt_semantics` / `run_prompt_semantics_gate`. Prompt sites now count as consumers.
- `orion/inner_state_registry.py`: field semantics, plus the 4 registrations.
- `orion/field/channel_glossary.py`: entry carries `semantics`.
- `config/field/field_channel_glossary.v1.yaml`: the 8 new PE entries, plus semantics on bus_synaptic and the two vision_organ channels.
- `scripts/check_metric_lineage.py`: adds `--prompt-semantics`; the card prints the new fields.
- `services/orion-hub/scripts/field_channel_glossary_routes.py` and its test: adds `node_variants`.
- `.github/workflows/orion-static-gates.yml`: new step.
- `tests/test_metric_prompt_semantics.py` (new, 42 tests) and `tests/test_field_channel_glossary.py`.
- `config/metrics/metric_definitions.lock.json`: re-locked, twice (once after the build, once after the review fixes).
- Docs: semantic-layer spec section; phase-5 R6 closed.

## Schema / bus / API changes

- **Added:**
  - the registry fields above;
  - 8 glossary entries;
  - 4 inner-state registrations;
  - the `node_variants` and `semantics` keys on the Hub glossary `/channels` rows.
- **Removed:** none.
- **Renamed:** none. bus_synaptic keeps its digester URN. The new PE entries use `orion-substrate-runtime`, the service that actually computes them.
- **Behavior changed:** the Hub glossary lists 49 channel rows. Before, it listed 50, including a duplicate `prediction_error` row.
- **Compatibility:** no bus, channel or pydantic schema contract changed.

## Env/config changes

None. No `.env_example` was touched, so the env sync was not needed.

## Tests run

```text
Metric/glossary/registry suites (drift, lineage, gate, glossary, prompt semantics, inner-state gate,
generic consumers, corpus schema, lattice sources): 253 passed
Hub glossary routes + outreach vocabulary + mood-arc + tension trigger: all passed
Mutation check on the gate: 12/12 mutants killed (2 survivors in the first round got new tests)
```

## Evals run

No eval harness exists for the metric layer. It is a static gate, and its tests are mutation-checked. Live data was checked by hand instead (3 days to 2026-10-10):

- `substrate_node_prediction_error_history` zero fractions: cabinet 97%, codebase 91%, route 87% (4 distinct values), execution 72%, chat 69%, perception 61%.
- harness_closure: 113 rows, all 0.650.
- biometrics: median 0.026. bus_synaptic: median 0.033.
- self-model `confidence`: only ever 0.9, 0.6 or 0.3.
- `top_down_effort_used`: 0 (3,959 rows) or 1.0 (1,591 rows), with about 30 other values.
- `prediction_error_confidence`: median 0.977.
- `heartbeat_smear`: median 2.6, max 1.46M.

## Docker/build/smoke checks

```text
No Docker: static change, nothing read at boot by a running service except Hub's glossary route (code-only).
All 27 run-steps of .github/workflows/orion-static-gates.yml executed locally in a clean venv
(pydantic, pydantic-settings, PyYAML, pytest, requests): 27/27 OK.
```

## Review findings fixed

- **Finding (must-fix):** polarity was invented for about 26 channels. For example, it labelled `expected_offline_suppression` and the `cabinet_*_activity` channels "higher is worse".
  - Fix: polarity now comes only from `HIGHER_IS_BETTER_CHANNELS` and `PRESSURE_CHANNELS`. Any other channel, and any trigger, gets none.
  - Evidence: `test_channel_in_neither_merge_set_gets_no_polarity`, `test_trigger_gets_no_polarity`; re-locked.
- **Finding:** the repair appraisal's declared consumer was upstream of what it published.
  - Fix: the consumer is now the equilibrium metacog gate (`EquilibriumService`, which subscribes at `service.py:1261`). The notes say "Hub publishes, the v2 paradigm computes".
- **Finding:** the curiosity signal's consumer reads only `world_coverage_gap`.
  - Fix: the notes now limit the claim to that signal type.
- **Finding:** the gate passes for any prompt site that exists, so a marker could be dropped from an unpinned metric without anything failing.
  - Fix: a reverse pin. Any marker on a URN that is not in `PROMPT_INVENTORY_URNS` now fails. The "exists, not proven to render" limit is documented.
- **Finding:** Hub lost the per-node meanings.
  - Fix: they are now nested as `node_variants`, with their semantics.
- **Finding:** a whitespace-only or non-string `rest` passed the gate.
  - Fix: the value must now be non-empty text. Test added.
- **Finding:** `sparsity` mixed two properties.
  - Fix: a precedence rule is documented, and chat, codebase and route are aligned to `designed_sparse`.
- **Finding:** execution, chat, route and biometrics did not declare endogenous curiosity as a consumer.
  - Fix: added.

## Restart required

```text
No restart required for the gate. Hub's glossary panel picks up node_variants on its next deploy.
```

## Risks / concerns

- **Medium: the gate can't see five prompt-reaching numbers**, because they have no URN. They are:
  - the mind frontier score;
  - the curiosity prior's confidence;
  - the metacog biometrics cue's strain, homeostasis, stability and fleet_watts;
  - metacog transport severity.

  They are listed next to the pin; registering them is follow-up work.
- **Low: "reaches a prompt" is declared and pinned, not derived.** The consumer scan cannot read templates.
- **Low: bus_synaptic is on a different producer URN from its 8 siblings.** Moving it is a separate rename.

## PR link

(filled after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
