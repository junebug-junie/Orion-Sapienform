# Action-outcome ledger — make dispatched actions falsifiable

**Branch:** `feat/action-outcome-ledger`
**Status:** `DONE_WITH_CONCERNS` — merge is gated on the deploy order in
"Restart required" below, and the value definition is superseded by a
committed design (see "Known-wrong by design").

## Summary

- Every dispatched action now records, **before it runs**, what it expects to
  change: `ExpectedEffectV1` on the dispatch candidate.
- After the action's field window closes, the claim is scored: observed
  delta, prediction error, whether the declared direction held, and Bayesian
  surprise in nats.
- Templates declare only `(signal, direction)`. Magnitude comes from the
  measured posterior, never from YAML — repeating the hand-typed-`risk_score`
  defect was the specific thing to avoid.
- Two new tables, written in the same transaction as the feedback frame and
  its pending-marker clear, with the ledger write on a savepoint so it cannot
  wedge the pipeline.
- 57 new tests. Every fixture hand-computed from the closed forms.
- **Nothing about what Orion dispatches changes.** Measurement only.

## Outcome moved

There was no measurement of autonomous-action value at all. Verified by grep:
neither `orion/proposals/` nor `orion/execution_dispatch/` referenced a
feedback frame or an outcome score. 32,000 outcome observations a day were
written and none were read back into any decision.

The one number that claimed to be this — `action_outcomes.surprise`, 133,058
rows — is `latest_bus_synaptic_prediction_error()`: a global bus-traffic
reading fetched once per tick and stamped identically onto every candidate.
Three different actions at 2026-08-20 22:42:57 all carry `0.0157`. It
describes the message bus, not the action.

After this patch the dispatch builder reads the posterior back into the next
prediction, so the *belief* loop closes. The *decision* loop does not — the
allocator still takes the first 5 by priority. That is phase 2 and is not
claimed here.

## Current architecture (before this patch)

Measured live 2026-08-21:

- 38,138 real dispatches in 7 days across **15 targets and 7 verbs**.
- Upstream, 190,430 proposals in 3 days from **16 templates and 6 kinds** in
  `config/proposals/proposal_policy.v1.yaml`.
- Top two actions: inspect `orion/bus/channels.yaml` (7,583) and summarize
  `capability:transport` (7,439). Docker pruning is ~19% of all dispatch.
- 75% of blocks are `max_dispatch_candidates:5`, a fixed count — the risk
  budget was not the binding constraint.
- Every dispatched action carried one of **five** distinct `risk_score`
  values, 67% of them exactly `0.05`, all hand-written in YAML.

## Architecture touched

`orion/autonomy/` (new pure math), `orion/feedback/` (new resolver),
`orion/proposals/` + `orion/execution_dispatch/` (declaration carry-through),
`orion-feedback-runtime` and `orion-execution-dispatch-runtime` stores and
workers, one SQL migration, one analysis script.

## Files changed

- `orion/autonomy/prediction.py` (new): Normal-Normal conjugate update and
  Bayesian surprise (Itti & Baldi 2009 — the epistemic-value term of expected
  free energy). Nats, so pragmatic and epistemic value can compete on one
  scale later.
- `orion/schemas/action_prediction.py` (new): `ExpectedEffectV1`,
  `ActionOutcomeRecordV1`, closed `PredictableSignal` literal.
- `orion/feedback/outcome_resolution.py` (new): scores claims against the
  real field window; `claim_upheld()`.
- `orion/schemas/execution_dispatch_frame.py`, `orion/schemas/proposal_frame.py`,
  `orion/proposals/policy.py`, `orion/proposals/builder.py`,
  `orion/execution_dispatch/builder.py`: declaration plumbing + a load-time
  validator that rejects a half-declared or misspelled claim.
- `config/proposals/proposal_policy.v1.yaml`: 11 of 16 templates declare a
  claim.
- `services/orion-feedback-runtime/app/{store,worker}.py`: persistence.
- `services/orion-execution-dispatch-runtime/app/{store,worker}.py`: posterior
  read-back.
- `services/orion-sql-db/manual_migration_action_outcome_ledger.sql` (new).
- `scripts/analysis/report_action_value.py` (new).
- `docs/superpowers/specs/2026-08-21-action-value-control-arm-design.md` (new).
- `tests/test_action_prediction.py`, `tests/test_action_outcome_resolution.py`,
  `tests/test_expected_effect_declaration.py` (new).

## Why 5 templates declare nothing, deliberately

`inspect_bus_channel_catalog`, `summarize_transport_contract_drift`,
`watch_transport_backpressure`, `inspect_field_topology_catalog`,
`inspect_attended_target` have no motivating dimensions at all and fire on
`base_priority` alone. They are **72% of live dispatch volume over 24h.**
Inventing claims for them would have manufactured exactly the kind of
unfalsifiable label this patch exists to remove. They record as
`no_declared_signal` and that absence is the audit result.

(The commit message says "~62%". That figure was computed over the 7-day
dispatch mix; the 24h volume figure is 72.0%. Corrected here and in the spec.)

## Metric gate

1. **Provenance.** `surprise_nats` is computed in
   `orion/autonomy/prediction.py::bayesian_surprise_nats` from field values
   read by `orion/field/pressure.py::field_pressures`, the same channel-merge
   the proposal layer already uses.
2. **Independence.** Not another pressure. Zero when the posterior does not
   move, regardless of the pressure's level — a saturated pressure with a
   perfectly-predicted delta scores 0; a mid-range pressure with a surprising
   delta scores high.
3. **Theory anchor.** Bayesian surprise, Itti & Baldi, *Vision Research*
   49(10), 2009. Named, with a closed form, not "seems related".
4. **Live-data sanity.** Pressure deltas over 68,715 real feedback frames are
   non-degenerate and signed both ways (sd 0.057–0.293).
   `deviation_pressure` reaches genuine rest — 46% nonzero over 20,000
   samples, full 0–1 range — rather than sitting on a permanent floor.
   **This check was run on a pooled statistic and that was not good enough:**
   per channel, `reliability_pressure` is below 1e-12 in 91.1% of 50,680
   frames (live values ~3.7e-190, a decay artifact). It carries 16 of 17,983
   dispatches/day so it is not urgent, but it is recorded in the spec's
   "Known bad instruments" and must not be declared on a high-volume template.
5. **Existing mechanism.** `orion/autonomy/action_outcomes.py` and the
   `action_outcomes` table already exist. Read before building; its
   `surprise` column is the bus-traffic stand-in described above, so it does
   not answer this question. Retiring it in favour of this ledger is recorded
   in the spec, not done here.
6. **Reversibility.** Two new tables, one optional schema field, no change to
   dispatch behaviour. Dropping both tables and reverting the branch removes
   it completely.

## Schema / bus / API changes

- **Added:** `ExpectedEffectV1`, `ActionOutcomeRecordV1`;
  `ExecutionDispatchCandidateV1.expected_effect` (optional);
  `ProposalCandidateV1.expected_signal` / `.expected_direction` (optional);
  `ProposalTemplateV1.expected_signal` / `.expected_direction` (optional).
- **Removed / renamed:** none.
- **Behaviour changed:** `load_proposal_policy` now *rejects* a template with
  a half-declared or misspelled claim instead of loading it. A typo would
  otherwise silently turn a scored action back into an unscored one.
- **Bus:** no new channels, no envelope changes.
- **Compatibility:** all new model fields are optional, so frames written
  before this patch parse unchanged. The reverse is **not** true — see
  "Restart required".

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: not needed (no new keys).
- Local `.env` sync: not applicable this patch.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest tests/test_action_prediction.py tests/test_action_outcome_resolution.py \
       tests/test_expected_effect_declaration.py -q
57 passed

pytest tests/test_feedback_builder.py tests/test_proposal_frame_builder.py \
       tests/test_proposal_policy_loader.py tests/test_dispatch_starvation.py \
       tests/test_proposal_scoring.py tests/test_feedback_runtime_store.py -q
110 passed          (regression check on touched areas, no failures)
```

## Evals run

```text
None. orion/autonomy/evals/ has no harness for action value, because no
measure of action value existed before this patch. The spec's acceptance
check 3 (replay the live 3-day prune data through the contrast and confirm it
lands near zero, not near -0.15) is the eval this needs, and it belongs with
the control-arm patch that makes the contrast exist. Recorded as a follow-up,
not claimed as coverage.
```

## Docker/build/smoke checks

```text
Migration applied live and verified:
  psql -f services/orion-sql-db/manual_migration_action_outcome_ledger.sql
  -> substrate_action_outcomes         (4 valid indexes)
  -> substrate_action_effect_posterior (1 valid index)

python scripts/check_sql_migrations_applied.py
  -> 79 migration file(s): 69 applied, 2 skipped, 3 superseded, 5 unknown
  -> Every declaratively checkable object in every migration is present and valid.

python scripts/analysis/report_action_value.py --days 7
  -> "No scored action outcomes in the last 7 day(s)" (correct; not deployed)

No container was rebuilt or restarted. See "Restart required" -- the order
matters and is not safe to improvise.
```

## Review findings fixed

Adversarial review found 13 items. Fixed in-branch:

- **Finding 3 (HIGH) — `direction` was never scored.** It had a schema, a
  producer and a persister and no consumer, which made the patch's central
  claim false: an action declaring `decrease` that produced `+0.4` earned
  identical nats to one declaring `increase`.
  - Fix: `claim_upheld()` in `outcome_resolution.py`, a `claim_upheld` column,
    and a 1e-6 dead band reused from `classify_pressure_deltas` so "moved"
    means one thing across the feedback path. A directional claim inside the
    dead band returns `None` (undecidable), never a soft pass.
  - Evidence: 9 new tests, including
    `test_opposite_directions_on_the_same_delta_do_not_score_alike`.
- **Finding 4 (HIGH) — a ledger write failure would wedge the FIFO forever.**
  The write shares a transaction with the frame insert and the
  `feedback_pending = false` clear, so a raise rolled back all three; the next
  tick re-selects the same oldest row and fails identically. This service has
  already suffered that stall once.
  - Fix: `conn.begin_nested()` savepoint around the ledger write. The frame
    and marker commit; a ledger failure costs one row, not the loop.
  - Evidence: transaction scoping re-read; the worker's existing try/except
    was confirmed to cover only the pure computation, not the write.
- **Finding 6 (MEDIUM) — `prediction_error` did not match its own row.** It
  was the residual against `prior.mean` (the belief at *scoring* time) while
  `predicted_delta` on the same row is the claim made at *dispatch* time.
  A reader recomputing `observed_delta - predicted_delta` got a different
  number, sometimes with the opposite sign.
  - Fix: `error = observed_delta - effect.predicted_delta`.
  - Evidence: `test_error_is_recomputable_from_the_row` pins the exact case
    that used to disagree (stored -0.15 vs recomputable +0.10).
- **Finding 11 (LOW) — a copy-pasted comment claimed the three mutating prune
  templates were read-only.**
  - Fix: distinct banner naming the confound risk and pointing at the spec.
- **Finding 12 (LOW) — stray triple blank line** in the dispatch store.

Confirmed correct by the review, recorded so it is not re-litigated: the
Normal-Normal update and closed-form KL re-derived independently; the
hand-computed fixture sequence reproduces exactly; the convergence property
holds across four unrelated constant pairs;
`ON CONFLICT DO NOTHING RETURNING id` yields `None` on conflict through the
real SQLAlchemy stack; the `posterior_n <` guard cannot walk a posterior
backwards; `dispatch_id` is unique per tick. Every priority mutation test was
caught by an existing test.

## Known-wrong by design, and specced

- **Finding 1 (CRITICAL) — the value definition is confounded.**
  `observed_delta` is the unconditional field delta. Actions fire *because* a
  pressure is high, and high pressures fall on their own. Raw: prune ticks
  -0.148 vs non-prune -0.026, apparently a 5.8x effect. Conditioned on
  baseline decile it inverts — in 6 of 8 bands the prune arm falls *less*.
  Left alone the ledger would print a confident "docker prune reduces
  resource_pressure by 0.15", which is regression to the mean.
  - Full design, with the live decile table, the control arm
    (`max_dispatch_candidates:5` blocks — 58,285 in 7 days), the matched
    contrast, and a randomized-holdback upgrade:
    `docs/superpowers/specs/2026-08-21-action-value-control-arm-design.md`.
  - **No budget may read `posterior_mean` until that lands.**
- **Finding 5 (HIGH) — no tests on the store layer.** The double-counting
  guards were verified by hand against live Postgres and behave as claimed,
  but nothing would catch them breaking. Belongs with the control-arm patch,
  which changes those writes anyway.
- **Finding 7 (MEDIUM, latent) — in-frame double absorption.** Two candidates
  in one frame sharing `(kind, target, signal)` would feed the same delta into
  the posterior twice. Unreachable today only because
  `inspect_attended_target` declares no signal — and it collides with
  `inspect_execution_pressure` on `(inspect, capability:orchestration)` 404
  times in 7 days. Becomes live the moment that template gets a declaration.
- **Findings 8, 9, 13 — instrument health.** `reliability_pressure` decayed to
  1e-190 in 91% of frames; `deviation_pressure`'s absence guard structurally
  cannot fire; `resource_pressure` was frozen at exactly 0.85 for the two
  hours in which this patch's live checks ran. All three recorded in the
  spec's "Known bad instruments".
- **Finding 10 (LOW) — `MIN_VARIANCE` guards only the low end**; an infinite
  prior variance yields an infinite KL that pydantic accepts. Unreachable from
  `cold()`, which is 0.25 and only shrinks.

## Restart required

**Order is load-bearing. Consumers before producers.**

`ProposalCandidateV1` and `ExecutionDispatchCandidateV1` are `extra="forbid"`,
and the readers do not stall on a validation error — they **retire the row**
so the FIFO advances (`services/orion-policy-runtime/app/store.py:64-77`,
`services/orion-feedback-runtime/app/store.py:81-101`, each with its own
documented prior incident). If a producer ships first, every proposal and
dispatch frame written during the deploy window is silently discarded behind
a `logger.warning`.

```bash
# 1. readers first
./scripts/safe_docker_build.sh orion-hub up -d --build
./scripts/safe_docker_build.sh orion-policy-runtime up -d --build
./scripts/safe_docker_build.sh orion-feedback-runtime up -d --build

# 2. execution-dispatch-runtime: reads proposal frames, writes dispatch frames
./scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build

# 3. producer last
./scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
```

Then confirm the ledger is filling:

```bash
psql -h localhost -p 55432 -U postgres -d conjourney \
  -c "SELECT arm_check.* FROM (SELECT count(*) rows, count(DISTINCT dispatch_kind) kinds,
      count(*) FILTER (WHERE claim_upheld IS TRUE) upheld
      FROM substrate_action_outcomes) arm_check;"
python scripts/analysis/report_action_value.py --days 1
```

## Risks / concerns

- **Severity: high.** The headline value number is confounded (finding 1).
  Mitigation: phase 1 is measurement-only and changes no dispatch behaviour,
  the spec is committed alongside, and this report states plainly that no
  budget may read `posterior_mean` until the control arm lands. The risk is
  not that the code misbehaves — it is that someone reads the report output
  as a causal result.
- **Severity: high.** Deploy ordering (finding 2). Mitigation: exact order
  above; wrong order loses proposals for the duration of the deploy.
- **Severity: medium.** `substrate_action_outcomes` grows at roughly the real
  dispatch rate (~5,400/day, less the 72% that declare nothing). Indexed on
  `observed_at` from day one, but it has no retention policy yet and this repo
  has ~8.3 GB of unbounded substrate tables already.
- **Severity: low.** `OutcomeResolution.posteriors` has no non-test consumer.
  Kept as the natural return of a resolver; flagged rather than hidden.

## PR link

`gh` is unauthenticated in this environment. Branch is pushed; open with:

```bash
gh pr create --base main --head feat/action-outcome-ledger \
  --title "feat(autonomy): action-outcome ledger -- make dispatched actions falsifiable" \
  --body-file docs/superpowers/pr-reports/2026-08-21-action-outcome-ledger-pr.md
```
