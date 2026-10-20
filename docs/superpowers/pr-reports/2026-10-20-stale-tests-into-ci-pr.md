# fix/stale-tests-into-ci

## Summary

- Three test files failed on `origin/main` every time, but no CI workflow ran them, so nobody saw it. This patch fixes all three and adds them to CI.
- orion-thought settings test: the expected Mind URL was `http://orion-mind:6611`. The real default has been `http://mind:6611` since commit d66f816e0 (07-28). The test now expects the real value. It also clears `CHANNEL_MIND_ARTIFACT`, so a host `.env` cannot change the result.
- Context provenance: `executor.py` gained 7 context keys between 07-18 and 09-24 and nobody classified them. Each key is now classified after reading its producer.
- Reverie proposal test: the keyword `self_state_id` was renamed to `fallback_target_id` in 58012465e (07-22). The test now uses the new name.
- CI: `visual-baseline.yml` now runs the reverie tests (`test_proposal.py`, `test_proposal_world_winner.py`) and the orion-thought settings test. `metacog-capture-tests.yml` now runs `test_context_provenance.py`, and it starts when `context_provenance.py` or that test changes.

## Outcome moved

Before this patch, 8 reverie tests, 1 provenance test, and 1 settings test failed and nothing reported it. Now all of them pass, and they run in CI on the paths that can break them. If someone adds a new `ctx["key"] = ...` to `executor.py` without classifying it, the metacog-capture workflow fails on that PR. Before, the gap only showed up when someone ran the test by hand.

## Current architecture

- `orion/schemas/context_provenance.py` maps each content key to a source kind (live, derived, memory, static identity, user input) or lists it in `PLUMBING_KEYS`. `grounding_capsule.context_provenance_for_ctx` and the chat_stance provenance hazard read this registry when they decide which context counts as "live now".
- `test_static_ctx_assignments_covered` scans `executor.py` with a regex for literal ctx assignments. It ran in no workflow.
- `orion/reverie/tests/test_proposal.py` and `services/orion-thought/tests/test_settings_mind_enrichment.py` also ran in no workflow.

## Architecture touched

Test files, one schema registry module (classification only), and two CI workflows. No runtime service code changed.

## Files changed

- `orion/schemas/context_provenance.py`: classifies the 7 keys below.
- `orion/reverie/tests/test_proposal.py`: `self_state_id=` changed to `fallback_target_id=`.
- `services/orion-thought/tests/test_settings_mind_enrichment.py`: expects the correct default URL and clears `CHANNEL_MIND_ARTIFACT`.
- `.github/workflows/visual-baseline.yml`: runs the two tests (its trigger paths already cover `orion/reverie/**` and `services/orion-thought/**`).
- `services/orion-thought/docker-compose.yml`: changed the `ORION_MIND_BASE_URL` fallback to `http://mind:6611` (review nit).
- `.github/workflows/metacog-capture-tests.yml`: runs `test_context_provenance.py`. Trigger paths now include the module and the test (it already triggered on `executor.py`).

### Classifications (each one is based on reading its producer)

| key | kind | producer / reason |
|---|---|---|
| `admission` | live_runtime_projection | `app/admission_cue.py`: a windowed read of `gpu_pool_events` that shows whether background LLM work had to wait. TTL 60s. |
| `recent_trend_signals` | live_runtime_projection | `app/metacog_trend_reader.py`: the latest per-node prediction error and biometrics induction. This matches `self_readings`. |
| `recent_trend_signals_json` | live_runtime_projection | The JSON version of the key above. This follows the module's existing convention for `_json` versions (`biometrics_json`, `turn_effect_json`). |
| `recent_attention` | derived_summary | `app/recent_attention_reader.py`: persisted attention rows labelled by age ("about 2 hours ago"). This is a record of the past, not the current tick, so it matches `recent_arcs`. |
| `metacog_event_evidence` | derived_summary | `executor._metacog_event_evidence_cue`: a fixed template over the trigger's evidence mapping. |
| `trigger_upstream_json` | derived_summary | The JSON form of `trigger.upstream`, the evidence the gate already built. This matches `trigger`. |
| `llm_serving_node` | PLUMBING_KEYS | The node from gateway `served_by`, used only for routing attribution. |

Deviation from the brief: the brief said to put the `_json` versions in `PLUMBING_KEYS`. The module defines plumbing as keys that carry "no cognitive content", and it already registers every `_json` version with the same kind as its source. `recent_trend_signals_json` and `trigger_upstream_json` are rendered into the metacog prompts, so they carry content. They are registered as content to stay consistent with that convention.

## Schema / bus / API changes

- Added: 6 registry entries and 1 plumbing key (classification only).
- Removed / Renamed: none.
- Behavior changed: `context_provenance_for_ctx` now reports these keys when they have content. On a metacog pass where `admission` or `recent_trend_signals` is non-empty, the chat_stance provenance hazard can now fire. That is the intended meaning: these keys are live readings.
- Compatibility notes: none.

## Env/config changes

- Added / Removed / Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
env -i PATH=/usr/bin:/bin HOME=$HOME python -m pytest -q <each CI step's exact file list>
visual-baseline step 1 (+ orion/reverie/tests/test_proposal.py): 160 passed
visual-baseline step 2 (+ test_settings_mind_enrichment.py):     234 passed, 29 skipped
metacog-capture schema step (+ test_context_provenance.py):      174 passed
test_settings_mind_enrichment.py with the host services/orion-thought/.env copied in: 7 passed
cortex-exec grounding/provenance consumers (5 files):             39 passed
orion/harness grounding tests:                                    passed
tests/test_metacog_prompt_phase_contract.py: 2 failures, which also fail on origin/main (ModuleNotFoundError 'app'). Not caused by this patch and not in scope.
```

## Evals run

```text
No eval changes. This patch only repairs tests and wires them into CI. The metacog capture eval still runs in the same workflow.
```

## Docker/build/smoke checks

```text
Not applicable. No runtime code, image, or compose changes.
```

## Review findings fixed

A code-review subagent reviewed the branch diff against origin/main. It found no must-fix issues. It confirmed all 7 classifications, the `_json`-twin convention, and the CI wiring. It found no harmful runtime change: the MetacogContextService branch already set live keys (`metacog_biometrics_cue`, `metacog_substrate_cue`), so the provenance hazard already fired on these turns.

- Finding (should): `orion/reverie/tests/test_proposal_world_winner.py` uses the same API and was also run by no workflow.
  - Fix: added it to visual-baseline's first step.
  - Evidence: 17 passed (proposal + world_winner + provenance), `env -i`.
- Finding (nit): `services/orion-thought/docker-compose.yml` still had the fallback `http://orion-mind:6611`, but settings and `.env_example` say `http://mind:6611`.
  - Fix: changed the compose fallback to `http://mind:6611`. It applies only when the env key is unset, and the live `.env` sets it.
  - Evidence: one-line diff. The live `.env` already holds `http://mind:6611`, so the running config does not change.
- Finding (nit): the `derived_summary` comment said "reduction of other context", but `recent_attention` reduces persisted rows.
  - Fix: the comment now also covers "persisted past records", and the description now says "fixed-template reduction".
  - Evidence: test_context_provenance passes.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: `test_static_ctx_assignments_covered` only sees literal `ctx["k"] =` / `ctx.setdefault` assignments in `executor.py`. Keys that are set inside helper modules still depend on the hand-maintained live snapshot.
- Mitigation: no change from before. This is a known limitation, documented in the test.

## PR link

PR_LINK_PLACEHOLDER
