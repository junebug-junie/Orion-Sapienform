# Fix a policy-runtime stall, then arm mutating dispatch

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1594
Branch: `feat/maintenance-dispatch-wiring`
Status: **DONE_WITH_CONCERNS**

## Summary

- Asked to flip `allow_mutating_dispatch` and correct `.env_example`. Did both — but code review found the chain being armed was **non-functional**, with a failure mode worse than a no-op, so this PR grew a blocker fix first.
- **Blocker:** `PolicyDecisionV1`'s `allowed_scope`/`autonomy_tier` Literals omitted `maintenance_bounded` while the `maintain` policy rule sets both to it. A real prune candidate raised `ValidationError` during decision *construction*, which the policy runtime's escape hatch does not cover → **permanent FIFO stall of the whole policy runtime**, L8–L11 dark. Reproduced live, fixed, regression-tested red-before-green.
- **Honesty:** a mutating prune was being approved as `approved_read_only` / reason `read_only_low_risk`. Added `approved_maintenance` as its own decision literal and made read-only-ness a declared per-kind property.
- **Misclassification:** the one skill in this repo that deletes host data advertised itself as `read_only, idempotent, no-confirmation` in two registries.
- **Theater:** a prune that reclaimed zero bytes recorded as success and would re-fire every tick forever.
- **Arming:** `allow_mutating_dispatch: false → true`; `.env_example` corrected to the live `dispatch_read_only`.

## Outcome moved

Orion can take a real, mutating, world-changing action on its own initiative — the first in this arc with a real per-action outcome (bytes reclaimed). Before this PR, arming the flag would have stalled cognition instead.

## Current architecture (before)

`proposal → policy decision → execution dispatch → cortex verb`, entirely read-only. PR #1594's earlier commit added a `maintain` kind, a `maintenance_bounded` scope, a route, and a template — but nothing had ever run a `maintain` candidate through the real policy evaluator.

## Architecture touched

`orion/policy/*` (evaluator, rules, builder, policy model), `orion/schemas/policy_decision_frame.py`, `orion/execution_dispatch/*`, both skills-manifest registries, the prune verb adapter, and three config files.

## Files changed

- `orion/schemas/policy_decision_frame.py`: added `maintenance_bounded` to `allowed_scope`/`autonomy_tier`; added `approved_maintenance` decision literal
- `orion/policy/builder.py`: per-candidate fault isolation; surfaces `candidate_unevaluable:*` warnings; new approved bucket
- `orion/policy/evaluator.py`: `approved_maintenance` in `DecisionLiteral`; maps it to the `execution_policy` gate, not `read_only`
- `orion/policy/rules.py`: `is_read_only_candidate` checks the kind's declared `mutating` flag first
- `orion/policy/policy.py`: `ProposalKindRuleV1.mutating: bool = False`
- `orion/execution_dispatch/builder.py`: dispatch reason string no longer hardcodes `approved_read_only_dispatch_v1`
- `orion/execution_dispatch/envelopes.py`: `no_external_side_effects`/`no_file_writes` derived from scope
- `orion/cognition/skills_manifest.py`, `services/orion-cortex-exec/app/actions_skill_registry.py`: `builder_prune` → `high_impact`, `runtime_housekeeping`, confirmation + execute-opt-in
- `services/orion-cortex-exec/app/verb_adapters.py`: zero-reclaim → `pruned_nothing`, escalates
- `orion/inner_state_registry.py`: `l7_l11_ladder` rehearsal justification marked false (label left as-is pending measurement)
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`: **`allow_mutating_dispatch: true`**; `approved_maintenance` allowed
- `config/policy/substrate_policy.v1.yaml`: `maintain` → `approved_maintenance`, `mutating: true`; removed a comment that documented the old lie
- `services/orion-execution-dispatch-runtime/.env_example`: `EXECUTION_DISPATCH_MODE` → `dispatch_read_only`
- `services/orion-execution-dispatch-runtime/docker-compose.yml`: `NOTIFY_URL` fallback → `http://notify:7140`
- Tests: 3 files, +6 tests, 2 guards inverted, 1 renamed, 2 fixtures corrected

## Schema / bus / API changes

- Added: `PolicyDecisionV1.decision` gains `approved_maintenance`; `allowed_scope`/`autonomy_tier` gain `maintenance_bounded`; `ProposalKindRuleV1.mutating`
- Behavior changed: a `maintain` candidate now yields `approved_maintenance` instead of `approved_read_only`
- Compatibility: **no SQL migration** — `policy_decision_frame_json` is `jsonb` with no typed column, enum, or CHECK. No consumer breaks; `orion/consolidation/motif.py` correctly stops counting maintain approvals as read-only-policy-loop samples.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: yes — `EXECUTION_DISPATCH_MODE` `dry_run` → `dispatch_read_only`
- local `.env` synced: yes, `python scripts/sync_local_env_from_example.py` run; no change needed for this service (live `.env` already carried the correct value — the template was the drifted one)
- Skipped keys requiring operator action: none

## Tests run

```text
pytest tests/test_maintenance_dispatch_gating.py tests/test_action_warrant.py
       tests/test_execution_dispatch_*.py tests/test_proposal_*.py tests/test_policy_*.py
       services/orion-cortex-exec/tests/test_builder_prune_skill.py
-> 182 passed

pytest tests/test_execution_dispatch_runtime_worker.py   -> 47 passed
pytest services/orion-cortex-exec/tests/test_builder_prune_skill.py -> 19 passed
```

Red-before-green verified for: the blocker regression (reverting the schema fix fails 3 tests), and both arming guards (reverting the flag + template fails exactly those 2).

## Evals run

```text
No eval harness exists for orion-policy-runtime or orion-execution-dispatch-runtime.
Not claimed as covered. Follow-up: the closest thing to an eval here is a live
observation of the first real prune's bytes_reclaimed, which cannot run until deploy.
```

## Docker/build/smoke checks

```text
docker compose ... -f services/orion-execution-dispatch-runtime/docker-compose.yml config
-> EXECUTION_DISPATCH_MODE: dispatch_read_only
-> NOTIFY_URL: http://notify:7140

docker exec orion-athena-execution-dispatch-runtime env | grep EXECUTION_DISPATCH_MODE
-> EXECUTION_DISPATCH_MODE=dispatch_read_only   (live, pre-existing)

docker exec ... python -c "socket.gethostbyname('orion-notify')" -> gaierror
docker exec ... python -c "socket.gethostbyname('notify')"       -> 172.18.0.28
```

`scripts/safe_graphify_update.sh` **refused and auto-restored** (node count 28306 → 2471, ~91%). Known recurring destructive-update bug, wrapper working as designed; nothing to commit for the graph.

## Review findings fixed

- Finding: SEV-1 — `maintain` candidates raise `ValidationError`, permanently wedging the policy-runtime FIFO.
  - Fix: added `maintenance_bounded` to both Literals; fault-isolated `build_policy_decision_frame`.
  - Evidence: reproduced the exact `ValidationError` pre-fix; post-fix `decision: approved_maintenance, scope: maintenance_bounded, dispatchable: True`. `test_maintain_candidate_survives_policy_evaluation` fails on the reverted schema.
- Finding: SEV-2 — the mutating skill registered as `read_only`/`idempotent`/no-confirmation in two registries.
  - Fix: `high_impact`, `runtime_housekeeping`, confirmation + execute-opt-in, in both.
  - Evidence: manifest now prints `builder_prune | high_impact | confirm: True | execute_opt_in: True`, matching its sibling prune skill.
- Finding: a prune reclaiming zero bytes recorded as success and would re-fire forever.
  - Fix: `pruned_nothing` + escalate.
  - Evidence: new test; it also turned `test_acts_only_when_both_conditions_hold` red, which had been asserting `"pruned"` on a zero-byte reclaim.
- Finding: `test_no_mutating_scope_in_envelopes` asserted a now-false repo-wide invariant, passing only by fixture accident.
  - Fix: renamed/narrowed to `test_read_only_routes_keep_read_only_envelopes`.
- Finding: my own `test_policy_default_dispatch_mode_...` documented a protection that never executes (`_resolve_dispatch_mode` short-circuits on the always-truthy override).
  - Fix: rewritten to assert the real source of the default (`settings.py`'s field default + the compose fallback).
- Finding: my own `.env_example` test pinned `dispatch_read_only`, so an operator using the emergency stop would fail CI.
  - Fix: asserts the key appears once and names a mode the runtime accepts.
- Finding: `envelopes.py` asserted `no_external_side_effects`/`no_file_writes` on a verb that deletes host files.
  - Fix: derived from scope; comment corrected to say these are metadata, not gates (the worker forwards only `context`).

## Restart required

```bash
cd services/orion-policy-runtime && ../../scripts/safe_docker_build.sh orion-policy-runtime up -d --build
cd services/orion-proposal-runtime && ../../scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
cd services/orion-execution-dispatch-runtime && ../../scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
cd services/orion-cortex-exec && ../../scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

All four: policy-runtime and proposal-runtime carry the schema/evaluator change, execution-dispatch-runtime the policy config and `NOTIFY_URL`, cortex-exec the skill manifest and zero-reclaim branch.

## Risks / concerns

- **Severity: HIGH.** `EXECUTION_DISPATCH_RPC_TIMEOUT_SEC` is a single global 120s and the verb's `timeout_ms` is 120000. A real prune of 142.5 GB has never been timed. If it exceeds 120s, Docker still deletes but the RPC records a failure — reintroducing the "acted with no attributable outcome" hole this arc exists to close. Raising the global value is wrong: the consumer is single-threaded, so a 15-minute ceiling lets one hung inspect stall dispatch for 15 minutes. **Mitigation: a per-route timeout — `ExecutionDispatchCortexClient.send()` already takes a per-call `timeout_sec`. Deliberately not in this PR.**
- **Severity: MEDIUM.** No cooldown. `prune_build_cache` is rebuilt every tick whenever `resource_pressure >= 0.2`, and `stable_dispatch_id` embeds `field_tick_id` so cross-tick dedup does not apply. The `pruned_nothing` branch now stops the *zero-outcome* loop from reading as success, but does not rate-limit attempts. Mitigation: a per-template cooldown keyed on last successful prune.
- **Severity: MEDIUM.** The skill measures `/hostfs/docker` but mutates through the docker socket. If the mount is not the daemon's data-root, `used_pct` describes one filesystem while the prune destroys cache on another. The `:ro` flag bounds nothing about the deletion.
- **Severity: MEDIUM.** Fresh-deploy widening: copying `.env_example` now yields `dispatch_read_only` + flag on, so a new node is armed on first boot. The skill's host-measured gate is the only remaining barrier.
- **Severity: LOW.** This PR was **not** split into "safe fixes" + "one-line arming", which review recommended for clean revert. The affected files interleave both, so splitting meant a red intermediate commit. The arming is still a single line (`allow_mutating_dispatch`) that can be reverted alone.
- **Severity: LOW.** `scripts/check_service_env_compose_parity.py` compares keys only, not values, so a compose default contradicting `.env_example` is invisible to CI. Not fixed here.
- **Severity: LOW.** `orion/proposals/builder.py` hardcodes `execution_intent.mode = descriptive_only` for all kinds, including this mutating one. Making it honest would trip `hard_blocks` (`destructive_action`, `file_write`) and block the action entirely — needs a deliberate design decision, not a rename.
- **Pre-existing, unrelated, verified identical on the untouched checkout:** 13 cortex-exec test-collection errors; `check_inner_state_registry.py` failing on two Attention schemas.
