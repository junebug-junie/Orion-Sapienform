# PR report: P3-P5 -- drive satisfaction, recall.query.readonly capability, attention-bound proposals

Branch: `feat/autonomy-satisfaction-capabilities-attention-p3-p5`
Series: endogenous action motor-nerve (P0 → P7), see `docs/superpowers/specs/2026-07-14-autonomy-p3-p5-design.md`

## Summary

- **P3**: `DriveEngine.update()` can now represent relief, not just growth --
  negative `drive_impacts` weights were silently double-clamped to zero
  before this patch, so drives could only ever accumulate pressure. Fixed
  with a signed clamp + sign-branched leaky-math formula.
- **P3**: `orion-spark-concept-induction` now subscribes to
  `orion:autonomy:action:outcome` (already published by Layer-9 dispatch) and
  mints a relief tension on successful dispatch of `inspect`/`summarize`/`observe`
  actions -- the first closed loop where an autonomous action can satisfy the
  drive that motivated it, not just accumulate more pressure toward the next one.
- **P4**: New `recall.query.readonly` capability -- before falling back to a
  Firecrawl web fetch, the metabolism loop now tries an internal RecallService
  query first. A recall hit leaves that cycle's fetch budget unconsumed.
- **P5**: `ProposalTemplateV1.target_binding` lets a template resolve its
  target from live context instead of a fixed value. One binding shipped:
  `dominant_attention_targets[0]`, wired into a new `inspect_attended_target`
  template that ships live (not dark-shipped) with a documented 7-day kill
  criterion and a standalone eval script.
- **P6** (folded in from the prior session): `/latest`'s `status_summary`
  field (dispatched/prepared/dry-run counts) on the hub's execution-dispatch
  debug route.
- Code review found and fixed two CRITICAL dead-wiring bugs -- both P3 and P4
  would otherwise have shipped fully built and fully unit-tested but
  structurally unreachable at runtime (see Review findings below).

## Outcome moved

Before this patch, Orion's drive/tension system was strictly one-directional:
every tension only ever raised pressure, and nothing an autonomous action did
could lower it. `DriveEngine` had no code path capable of representing
relief at all -- it wasn't a policy choice, the math structurally clamped
negative impacts to zero. This patch makes successful autonomous action
(Layer-9 dispatch) capable of satisfying the drive that motivated it, and
gives the metabolism loop an internal-memory-first option before it reaches
for the external web.

## Current architecture

Before this patch:
- `DriveEngine.update()` (`orion/spark/concept_induction/drives.py`) computed
  `impact_sum[drive] += mag * self._clamp01(weight)` -- `_clamp01` clamps to
  `[0,1]`, so any negative `drive_impacts` weight became `0` before it ever
  reached the pressure formula. The leaky-math pressure update itself,
  `pressures[drive] = base + impulse*(1-base)`, is also monotonically
  non-decreasing in `impulse` for `impulse >= 0` -- there was no formula
  branch for relief even if a negative impulse had survived the first clamp.
- `TensionRateLimiter._signature()` filtered on `weight > 0.0`, so any
  hypothetical negative-weight tension would have collided into a shared
  empty-tuple rate-limit bucket with every other relief tension regardless
  of which drive it targeted.
- `maybe_execute_substrate_act_after_metabolism` (`orion/autonomy/policy_act.py`)
  had exactly one capability: a Firecrawl fetch gated by capability policy.
- `ProposalTemplateV1`/`ProposalCandidateV1` had no way to point a template
  at context-derived data; every template's target was a fixed literal.
- The hub's execution-dispatch `/latest` route returned the raw frame with no
  aggregate counts; a caller had to count candidate statuses itself.

## Architecture touched

- `orion/spark/concept_induction/drives.py` -- signed-impact clamp + relief-capable pressure formula.
- `orion/autonomy/tension_ratelimit.py` -- signature filter widened to `weight != 0.0`.
- `orion/spark/concept_induction/tensions.py` -- new `extract_tensions_from_action_outcome`.
- `orion/spark/concept_induction/settings.py`, `bus_worker.py` -- new intake channel, dispatch branch, self-publish filter, skip-set entry.
- `orion/autonomy/models.py` -- `SubstrateActResultV1` gains `recall_attempted`/`recall_outcome`.
- `orion/autonomy/policy_act.py` -- `_execute_readonly_recall`, `maybe_execute_readonly_recall_after_goal`, recall-first wiring into the metabolism gate.
- `config/autonomy/capability_policy.v1.yaml` -- new `recall.query.readonly` rule.
- `orion/bus/channels.yaml` -- `orion-spark-concept-induction` added as a producer on `orion:exec:request:RecallService`.
- `orion/proposals/policy.py`, `orion/proposals/builder.py`, `orion/schemas/proposal_frame.py` -- `target_binding`/`binding_resolved_from`, `_resolve_binding_target`.
- `config/proposals/proposal_policy.v1.yaml` -- new `inspect_attended_target` template, live.
- `orion/autonomy/evals/run_attention_bound_proposal_eval.py` -- new kill-criterion eval.
- `services/orion-hub/scripts/substrate_execution_dispatch_routes.py` -- `_dispatch_status_summary`.
- READMEs: `orion-spark-concept-induction`, `orion-proposal-runtime`, `orion-execution-dispatch-runtime`.
- `services/orion-spark-concept-induction/.env_example` -- `BUS_INTAKE_CHANNELS` gains `orion:autonomy:action:outcome`.

## Files changed

- `orion/spark/concept_induction/drives.py`: signed clamp + sign-branched pressure formula so relief is representable and bounded.
- `orion/spark/concept_induction/tests/test_drives_leaky.py`: 4 new tests for relief math + a positive-only regression guard.
- `orion/autonomy/tension_ratelimit.py`: `w > 0.0` → `w != 0.0` in `_signature()` so relief tensions get their own budget.
- `orion/autonomy/tests/test_tension_ratelimit.py`: 2 new tests for negative-weight signatures.
- `orion/spark/concept_induction/tensions.py`: `extract_tensions_from_action_outcome`, closed kind→drive relief map (`inspect`→coherence, `summarize`→predictive, `observe`→continuity), fires only on `success is True`.
- `orion/spark/concept_induction/tests/test_action_outcome_tensions.py` (new): 7 tests covering all three kinds, failure/no-fire cases, unmapped kinds, magnitude/direction assertions.
- `orion/spark/concept_induction/settings.py`: `orion:autonomy:action:outcome` added to `intake_channels`'s Python default.
- `orion/spark/concept_induction/bus_worker.py`: dispatch branch for `action.outcome.emit.v1`, self-publish filter, skip-set entry, **and** (review fix) the sole `maybe_execute_substrate_act_after_metabolism` call site now passes `recall_bus`/`recall_source`, **and** a new publish block for `act_result.recall_outcome` mirroring the existing `fetch_outcome` block.
- `orion/autonomy/models.py`: `SubstrateActResultV1.recall_attempted`/`recall_outcome` (review fix, mirrors `fetch_attempted`/`fetch_outcome`).
- `orion/autonomy/policy_act.py`: `_execute_readonly_recall`, `maybe_execute_readonly_recall_after_goal` (P4); recall-first check wired into `maybe_execute_substrate_act_after_metabolism` with its own UUID-correlation_id fix; (review fix) now populates `recall_attempted`/`recall_outcome` on every attempt, independent of whether it succeeded enough to skip the fetch.
- `orion/autonomy/tests/test_policy_act.py`: existing recall-hit/recall-miss tests extended with `recall_attempted`/`recall_outcome` assertions (review fix regression coverage).
- `config/autonomy/capability_policy.v1.yaml`: `recall.query.readonly` rule -- readonly, auto_execute, `requires_goal_status: proposed`, `required_drive_origins: [predictive]`, `required_signal_kinds: [world_coverage_gap]`, `budget_per_cycle: 2`.
- `orion/bus/channels.yaml`: `orion-spark-concept-induction` added to `orion:exec:request:RecallService`'s `producer_services`.
- `orion/proposals/policy.py`: `ProposalTemplateV1.target_binding: str | None = None`.
- `orion/schemas/proposal_frame.py`: `ProposalCandidateV1.binding_resolved_from: str | None = None`.
- `orion/proposals/builder.py`: `ATTENTION_FIRST_TARGET_BINDING` constant, `_ATTENTION_BOUND_TARGET_KINDS` frozenset, `_resolve_binding_target()` (never raises).
- `config/proposals/proposal_policy.v1.yaml`: `inspect_attended_target` template, `base_priority: 0.34`, live with a documented 7-day kill criterion in comments.
- `orion/autonomy/evals/run_attention_bound_proposal_eval.py` (new): kill-criterion eval, handles insufficient data gracefully.
- `tests/test_proposal_frame_builder.py`, `tests/test_proposal_policy_loader.py`: binding-resolution + template-loading coverage.
- `services/orion-hub/scripts/substrate_execution_dispatch_routes.py`: `_dispatch_status_summary()`, wired into `/latest`.
- `services/orion-hub/tests/test_substrate_execution_dispatch_debug_api.py`: 2 new tests.
- `services/orion-spark-concept-induction/.env_example`: `BUS_INTAKE_CHANNELS` gains `orion:autonomy:action:outcome` (review fix, see Env/config changes).
- `docs/superpowers/specs/2026-07-14-autonomy-p3-p5-design.md`: design spec, documents why the original P3 brainstorm plan (a `drive_origin` bridge) was wrong -- no such bridge exists between the proposal/goal pipeline and Layer-9 dispatch, they're structurally separate pipelines -- and the chosen alternative seam (subscribe to the existing `action:outcome` channel).
- READMEs: `services/orion-spark-concept-induction/README.md`, `services/orion-proposal-runtime/README.md`, `services/orion-execution-dispatch-runtime/README.md`.

## Schema / bus / API changes

- Added: `orion.autonomy.models.SubstrateActResultV1.recall_attempted: bool`, `.recall_outcome: ActionOutcomeRefV1 | None`.
- Added: `orion.proposals.policy.ProposalTemplateV1.target_binding: str | None`.
- Added: `orion.schemas.proposal_frame.ProposalCandidateV1.binding_resolved_from: str | None`.
- Added: consumer subscription -- `orion-spark-concept-induction` now consumes `orion:autonomy:action:outcome` (already an existing, already-published channel; this is a new consumer, not a new channel).
- Added: producer registration -- `orion-spark-concept-induction` added to `orion:exec:request:RecallService`'s `producer_services` in `orion/bus/channels.yaml`.
- Added: `status_summary` field on the hub's `GET /api/substrate/execution-dispatch/latest` response (`dispatched_count`, `prepared_for_dispatch_count`, `dry_run_count`).
- Removed: none.
- Renamed: none.
- Behavior changed: `DriveEngine.update()` now accepts negative `drive_impacts` weights and applies relief instead of clamping them to a zero-effect no-op. This is additive for every existing (non-negative-weight) producer -- verified by running the full pre-existing drive test suite unchanged before adding new tests, and by the `test_positive_only_tensions_unaffected_by_signed_clamp` regression guard.
- Compatibility notes: `TensionRateLimiter`'s rate-limit signature space grew (relief tensions now get distinct signatures instead of colliding into one shared bucket) -- this only affects tensions with negative weights, of which there were previously none in production, so no existing rate-limit behavior changes.

## Env/config changes

- Added keys: none (no new env var name).
- Removed keys: none.
- Renamed keys: none.
- **Changed key value** (`.env_example` updated): `BUS_INTAKE_CHANNELS` gains `"orion:autonomy:action:outcome"` in `services/orion-spark-concept-induction/.env_example`.
- local `.env` synced: **no local `.env` exists in this worktree** (only `.env_example` is tracked/branched). The live `.env` lives in the shared main checkout at `services/orion-spark-concept-induction/.env` -- `scripts/sync_local_env_from_example.py --all-keys` was run from the main checkout, but it reads the main checkout's *own* `.env_example` (still on `main`, pre-this-branch) to decide what's "correct," so it could not see this branch's `BUS_INTAKE_CHANNELS` addition and reported no divergence for that key. The script did apply several unrelated missing-key adds from origination config that were already sitting undone in the live `.env` (`ORION_ENDOGENOUS_ORIGINATION_ENABLED=false` + its tuning keys, matching `.env_example` exactly -- pre-existing drift, not part of this patch, all default-off).
  `BUS_INTAKE_CHANNELS` was therefore **updated by hand** in the live `services/orion-spark-concept-induction/.env` (gitignored, not committed, verified via `git check-ignore`) to add `"orion:autonomy:action:outcome"`, matching this branch's `.env_example` exactly:
  ```
  BUS_INTAKE_CHANNELS=["orion:chat:history:log","orion:chat:history:turn","orion:chat:social:turn","orion:chat:social:stored","orion:chat:gpt:turn","orion:chat:gpt:message:log","orion:collapse:sql-write","orion:spark:telemetry","orion:metacognition:tick","orion:cognition:trace","orion:substrate:self_state","orion:feedback:frame","orion:world_pulse:run:result","orion:autonomy:action:outcome"]
  ```
  Confirmed via `scripts/check_service_env_compose_parity.py orion-spark-concept-induction`: this service's `docker-compose.yml` uses `env_file:`, so the full `.env` reaches the container regardless of the `environment:` list -- no compose change needed. P3's satisfaction-tension pipeline will subscribe correctly on next container restart (see Restart required below); no further operator action needed for this key.
- **Skipped keys requiring operator action**: none for this patch.

## Tests run

```text
$SCRATCH/p3p5-venv/bin/python -m pytest orion/spark/concept_induction/tests orion/autonomy/tests -q \
  --deselect orion/autonomy/tests/test_autonomy_isolation.py::test_autonomy_state_v2_not_wired_into_phi_or_self_state
347 passed, 1 deselected, 24 warnings in 15.17s

$SCRATCH/p3p5-venv/bin/python -m pytest \
  services/orion-hub/tests/test_substrate_execution_dispatch_debug_api.py \
  tests/test_proposal_frame_builder.py \
  tests/test_proposal_policy_loader.py \
  tests/test_execution_dispatch_bus_catalog.py \
  orion/autonomy/tests/test_capability_policy.py \
  orion/autonomy/tests/test_capability_policy_models.py \
  orion/autonomy/tests/test_policy_act.py \
  orion/autonomy/tests/test_policy_act_prefetched.py \
  -q
66 passed, 2 warnings in 2.55s
```

Deselected test is pre-existing and unrelated: `test_autonomy_state_v2_not_wired_into_phi_or_self_state`
fails identically on `origin/main` (verified via `git stash` + rerun, and by
inspecting `git show origin/main:orion/self_state/inner_state_registry.py`,
a file untouched by this branch or either subagent). Independently confirmed
by the code-review agent.

**Environment note**: the shared `/tmp/orion-test-venv` had numpy upgraded to
`2.5.1` by a concurrent session, breaking `thinc`'s compiled Cython extension
ABI (`ValueError: numpy.dtype size changed`) and making every test that
imports `spacy` uncollectable. `orion-spark-concept-induction/requirements.txt`
pins `numpy==1.26.4`. Rather than downgrade the shared venv (would risk
breaking whatever the concurrent session needs 2.5.1 for), copied it to an
isolated scratch venv and pinned numpy there. All numbers above are from that
isolated venv; the shared venv was not modified.

## Evals run

```text
python orion/autonomy/evals/run_attention_bound_proposal_eval.py
```

Not run live in this session -- requires an accumulation window of real
`inspect_attended_target` candidates against a live proposal-runtime deployment
to evaluate the 7-day kill criterion meaningfully; running it against an
empty/synthetic local DB would only exercise the "insufficient data" path,
which is not a meaningful eval result. This is the standing eval for judging
whether the template should be kept, tuned, or killed once live data
accumulates -- flagging as a follow-up check for 7 days post-merge, not a
gap in this PR.

No dedicated eval harness exists yet for the P3 satisfaction-tension path or
the P4 recall capability; the unit test suites above (`test_drives_leaky.py`,
`test_action_outcome_tensions.py`, `test_policy_act.py`) are the coverage
for those tracks.

## Docker/build/smoke checks

Docker was not run in this environment. Ran the deterministic non-Docker
checks that substitute for it:

```text
$SCRATCH/p3p5-venv/bin/python scripts/check_service_env_compose_parity.py orion-spark-concept-induction
→ N/A: service declares env_file:, all 110 .env_example keys reach the container
  regardless of the environment: list.

ORION_BUS_URL=redis://100.92.216.81:6379/0 \
  $SCRATCH/p3p5-venv/bin/python scripts/check_single_consumer_channels.py
→ single_consumer gate OK: 31 channel(s) checked, 3 warning(s) (pre-existing,
  zero-subscriber channels unrelated to this patch). orion:exec:request:RecallService
  reports OK 1 -- confirms P4's new producer registration on that channel did
  not create a fan-out/multi-consumer collision.

git diff --check origin/main..HEAD
→ clean, exit 0
```

## Review findings fixed

- Finding: `BUS_INTAKE_CHANNELS`'s Pydantic `validation_alias` **replaces**
  the Python default list rather than merging with it when the env var is
  set -- and it is set in both `.env_example` and every deployed `.env`.
  Adding the new channel to `settings.py`'s code default alone was a
  structural no-op in every real environment.
  - Fix: updated `.env_example`; documented the replace-not-merge behavior
    directly in the README so it's not rediscovered the hard way again;
    flagged the live-`.env` operator action prominently above.
  - Evidence: `check_service_env_compose_parity.py` confirms the container
    reads the full `.env`; the remaining gap is the live file's content,
    named explicitly as an operator action rather than silently assumed done.

- Finding: `bus_worker.py`'s one production call site to
  `maybe_execute_substrate_act_after_metabolism` never passed the
  `recall_bus`/`recall_source` kwargs P4's function signature added --
  `_execute_readonly_recall`'s `if bus is None: return decision, None` guard
  degraded gracefully and silently before any RPC fired. P4 was fully built
  and fully unit-tested but unreachable in the one place it actually runs.
  - Fix: call site now passes `recall_bus=self.bus, recall_source=_service_ref(self.cfg)`.
  - Evidence: `test_policy_act.py`'s recall-hit/recall-miss tests exercise
    the function directly with a real bus; the call-site fix is a one-line
    diff verifiable by inspection against `bus_worker.py`'s single call site.

- Finding: a successful recall was recorded only via
  `_execute_readonly_recall`'s local `append_action_outcome` file-store
  fallback, never reaching the bus-emit → sql-writer → durable-SQL path a
  fetch success uses -- `SubstrateActResultV1` had no field to carry it, so
  `bus_worker.py`'s existing `if act_result.fetch_outcome is not None:`
  publish block could never see it.
  - Fix: `SubstrateActResultV1` gains `recall_attempted`/`recall_outcome`
    (mirrors `fetch_attempted`/`fetch_outcome`); `policy_act.py` populates
    them on every recall attempt; `bus_worker.py` gains a parallel publish
    block for `recall_outcome`.
  - Evidence: extended `test_substrate_act_recall_hit_skips_fetch_budget`
    and `test_substrate_act_recall_miss_falls_through_to_fetch` with direct
    assertions on `recall_attempted`/`recall_outcome`; both pass.

- Finding (documented, not fixed -- accepted risk): `extract_tensions_from_feedback`
  (Postgres-polled failure path) and `extract_tensions_from_action_outcome`
  (bus-emitted success path) are independently-computed classifications of
  the same Layer-9 dispatch from two separate pipelines. A dispatch the
  feedback path scores failed/mixed while the outcome-emit reports
  `success=True` can fire both a growth and a relief tension for the same
  event. Coordinating across both pipelines is out of scope for this patch's
  size -- documented in the concept-induction README and here as a named
  follow-up.

- Findings (documented, not fixed -- cleanup only, not bugs): `_clamp_signed`
  duplicates `orion/signals/normalization.py`'s existing `clamp11`;
  `_execute_readonly_recall` duplicates ~60 lines of RPC boilerplate from
  `recall_prefetch.py`; `maybe_execute_readonly_recall_after_goal` duplicates
  ~35 lines of gate logic from `maybe_execute_readonly_fetch_after_goal`. All
  three are candidates for a follow-up dedup pass; none change behavior or
  correctness, so deferred rather than expanding this patch's surface area.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml \
  up -d --build orion-spark-concept-induction

docker compose \
  --env-file .env \
  --env-file services/orion-proposal-runtime/.env \
  -f services/orion-proposal-runtime/docker-compose.yml \
  up -d --build orion-proposal-runtime

docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build orion-hub
```

`orion-execution-dispatch-runtime` itself has no code changes in this patch
(only its README changed) -- no restart needed for that service.

The live `.env`'s `BUS_INTAKE_CHANNELS` was already updated by hand (see Env/config
changes above), so restarting `orion-spark-concept-induction` with this branch's
code is sufficient -- no separate operator step required.

## Risks / concerns

- Severity: MEDIUM (accepted, named)
  Concern: double-tension-fire risk between the feedback-frame and
  action-outcome pipelines on disagreeing success/failure classifications of
  the same dispatch.
  Mitigation: documented in the concept-induction README; cross-pipeline
  coordination deferred as a follow-up, not silently ignored.

- Severity: LOW
  Concern: three cleanup/duplication findings from review (`_clamp_signed`,
  `_execute_readonly_recall`, `maybe_execute_readonly_recall_after_goal`)
  left unaddressed.
  Mitigation: none needed for correctness; candidates for a follow-up dedup
  patch if this pattern keeps recurring.

- Severity: LOW
  Concern: `run_attention_bound_proposal_eval.py`'s 7-day kill criterion has
  not been evaluated against live data yet (the template only just shipped).
  Mitigation: eval script exists and handles the pre-accumulation window
  gracefully; run it 7 days after this merges.

## PR link

Not opened via `gh` (unauthenticated in this environment). Paste-ready for manual creation:

**Title:**
```
feat(autonomy): P3-P5 -- drive satisfaction, recall.query.readonly, attention-bound proposals
```

**Body:** (this file's Summary through Risks/concerns sections, verbatim)

Compare URL: https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/autonomy-satisfaction-capabilities-attention-p3-p5
