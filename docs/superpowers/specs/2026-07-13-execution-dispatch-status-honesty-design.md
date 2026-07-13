# Execution dispatch status honesty (P0 of the motor-nerve spec) — design

**Date:** 2026-07-13
**Status:** Implementation, this session
**Mode:** Thin patch. Implements P0 of `docs/superpowers/specs/2026-07-13-endogenous-action-motor-nerve-spec.md` exactly — no scope beyond it.

## Arsonist summary

`orion/execution_dispatch/builder.py:198-203` sets `dispatch_status="dispatched"` when policy allows `dispatch_read_only` mode, but the builder never sends anything — it only constructs a `request_envelope` dict. No worker, transport, or send exists anywhere in the repo (confirmed by repo-wide grep). The status is a lie today: it claims a send happened when only a plan was drawn up. This patch makes `dispatched` mean only "a send was attempted and evidenced" and gives the honest "ready but not sent" state its own name, `prepared_for_dispatch`.

## Current architecture

- `ExecutionDispatchCandidateV1` (`orion/schemas/execution_dispatch_frame.py:9-52`): `dispatch_status: Literal["prepared", "dry_run", "blocked", "dispatched", "skipped"]`. No evidence fields exist for a real send (no result ref, no error, no timestamp).
- `build_execution_dispatch_frame` (`orion/execution_dispatch/builder.py:63-257`): for `dispatch_mode="dispatch_read_only"` candidates that clear every gate (hard-block, policy-decision, route-scope, candidate-limit checks), line 198-203 sets `dispatch_status="dispatched"` if `policy.mode.allow_dispatch_read_only`, else `"dry_run"` with a warning. Status-`"dispatched"` items are routed into `dispatched_candidates`/`dispatch_count` (line 223-234); everything else lands in `candidates`.
- `orion/feedback/builder.py`: `OutcomeKind` Literal (line 31-46) mirrors the dispatch-status vocabulary. `_candidate_outcome_kind` (line 78-87) maps `candidate.dispatch_status` 1:1 to an `OutcomeKind` (falls to `"unknown"` for anything unrecognized). `_score_for_outcome_kind` (line 107-120) scores `"dispatched"` on `scoring.prepared_score` — i.e. the scoring lane already treats a "dispatched" candidate as merely "prepared," which is honest about the *scoring* even though the *label* was not.
- `orion/schemas/feedback_frame.py`: `OutcomeObservationV1.outcome_kind` carries an independent copy of the same Literal (line 26-40) — must stay in sync with `orion/feedback/builder.py`'s `OutcomeKind`.
- Live policy (`config/execution_dispatch/execution_dispatch_policy.v1.yaml`): `allow_dispatch_read_only: false` — the lie is not reachable in production today (mode defaults to `dry_run`), but is reachable by any caller that overrides `dispatch_mode` and flips the policy flag, and is directly exercised by `tests/test_feedback_builder.py`'s six hand-built fixtures.
- Downstream consumers of `dispatch_status`: `orion/consolidation/tensorize.py:152` (reads the raw string into a tensor feature, no literal-set assumption — unaffected), `orion/schemas/registry.py` (schema registration only — unaffected), Hub debug routes/tests (pass the frame through, assert on `dispatch_attempted`/empty lists in `dry_run`-mode fixtures only — unaffected).
- No producer of a genuinely evidenced `dispatched` status exists anywhere (grep-confirmed) — nothing has ever really dispatched. No data migration is needed; this is asserted, not assumed (see Acceptance checks).

## Missing questions

None — P0's scope is fully specified by the motor-nerve spec and confirmed against the live code in this session. The two judgment calls made explicit here so they're reviewable:

1. **Routing, not a new list.** `ExecutionDispatchFrameV1` gets no new `prepared_for_dispatch_candidates` list. Once the builder never emits `"dispatched"`, the `if dispatch_status == "dispatched": dispatched.append(item)` check in the builder naturally never matches, so `prepared_for_dispatch` items fall through to the existing `candidates` list (same as `"prepared"`/`"dry_run"` already do) — no code change needed at that check itself, only at the status-assignment line. `dispatched_candidates`/`dispatch_count` become honestly empty/zero from this builder until P1 adds a real sender. This is the correct minimal fix, not a gap: a new list would be scope creep for a naming/honesty patch.
2. **`dispatch_attempted` is left unchanged.** It's computed as `dispatch_mode == "dispatch_read_only" and policy.mode.allow_dispatch_read_only` — true whenever the policy permits read-only dispatch, independent of whether a send happened. This is a related imprecision but is explicitly out of P0's scope (the spec's P0 section doesn't name it, and P1's worker is what will make "attempted" meaningful). Traced its two downstream branches in `_aggregate_outcome_status` (`orion/feedback/builder.py:123-146`): neither the old `"dispatched"`-only nor the new `"prepared_for_dispatch"`-only case matches any early-return branch, so both fall through identically to `return "unknown"` — **confirmed no behavior change** to aggregate outcome status from this patch. Not fixed here; flagged in the PR report as a known follow-on for P1.

## Proposed schema / API changes

**`orion/schemas/execution_dispatch_frame.py`**
- `ExecutionDispatchCandidateV1.dispatch_status` gains `"prepared_for_dispatch"` in the Literal, alongside the existing five values. `"dispatched"` stays but its meaning narrows.
- New optional fields on `ExecutionDispatchCandidateV1`: `result_ref: str | None = None`, `dispatch_error: str | None = None`, `dispatched_at: datetime | None = None`.
- New `model_validator(mode="after")`: if `dispatch_status == "dispatched"`, require `dispatched_at is not None` AND (`result_ref is not None` OR `dispatch_error is not None`). Raise `ValueError` otherwise. This is the evidence bar — a claimed send must carry a timestamp and either a result or a recorded failure.

**`orion/schemas/feedback_frame.py`**
- `OutcomeObservationV1.outcome_kind` Literal gains `"prepared_for_dispatch"`, kept in sync with `orion/feedback/builder.py`'s `OutcomeKind`.

**`orion/execution_dispatch/builder.py`**
- Line 200: `dispatch_status = "dispatched"` → `dispatch_status = "prepared_for_dispatch"`. This is the entire behavioral change in this file. No other line moves.

**`orion/feedback/builder.py`**
- `OutcomeKind` Literal (line 31-46): add `"prepared_for_dispatch"`.
- `_candidate_outcome_kind` (line 78-87): add `if candidate.dispatch_status == "prepared_for_dispatch": return "prepared_for_dispatch"` — without this, candidates carrying the new status silently fall to `"unknown"`, a real misclassification (this is the specific case the task scope asked to check for and fix).
- `_score_for_outcome_kind` (line 107-120): add `"prepared_for_dispatch": scoring.prepared_score` — same scoring lane as `"prepared"` and (unchanged) `"dispatched"`.
- `"dispatched"`'s existing mapping and scoring are untouched — nothing produces an evidenced `"dispatched"` candidate yet, so no behavior change there; the mapping stays correct for when P1's worker exists.

## Files likely to touch

- `orion/schemas/execution_dispatch_frame.py` — literal, fields, validator.
- `orion/schemas/feedback_frame.py` — literal.
- `orion/execution_dispatch/builder.py` — one status-string change.
- `orion/feedback/builder.py` — literal, classification, scoring.
- `tests/test_execution_dispatch_frame_schemas.py` — new validator tests (evidenced `"dispatched"` passes, un-evidenced fails; `"prepared_for_dispatch"` needs no evidence).
- `tests/test_execution_dispatch_builder.py` — new test exercising `dispatch_mode="dispatch_read_only"` with `allow_dispatch_read_only=True`, asserting `dispatch_status == "prepared_for_dispatch"` and the item lands in `frame.candidates`, not `frame.dispatched_candidates`.
- `tests/test_feedback_builder.py` — six existing fixtures construct `dispatch_status="dispatched"` directly without evidence (`test_missing_cortex_result_absence`, `test_successful_cortex_result_completed`, `test_failed_cortex_result`, `test_partial_dispatch_completed_and_absent_is_mixed` ×2, `test_completed_and_failed_is_mixed`) — each needs `dispatched_at=NOW` and a `result_ref` (or leave the failed-result one to use `dispatch_error` where more honest) added so they remain valid under the new validator. These fixtures intentionally keep testing the feedback layer's handling of a genuinely-dispatched candidate — that scenario is still real and still needs coverage, it just now needs to supply the evidence the schema requires.
- `services/orion-execution-dispatch-runtime/README.md`, `services/orion-feedback-runtime/README.md` — document the status vocabulary change if either currently describes `dispatch_status` values.

## Non-goals

- No sending, no bus wiring, no transport, no new table (`substrate_dispatch_results`) — that is P1, explicitly deferred.
- No change to `dispatch_attempted`'s semantics (see Missing questions §2) — tracked as a known follow-on, not fixed here.
- No new `ExecutionDispatchFrameV1` list field for `prepared_for_dispatch` candidates — routes through the existing `candidates` list.
- No policy flag flips (`allow_dispatch_read_only` stays `false` in the live `.env`/policy default).
- No `.env`/`.env_example` changes — this patch touches no runtime configuration.

## Acceptance checks

1. `pytest tests/test_execution_dispatch_frame_schemas.py tests/test_execution_dispatch_builder.py tests/test_execution_dispatch_envelopes.py tests/test_feedback_builder.py -q` green.
2. `pytest orion/feedback -q` — no dedicated test dir exists under `orion/feedback/` today (confirmed); this becomes a no-op collection, not a failure. Real feedback coverage lives in `tests/test_feedback_builder.py` and `tests/test_feedback_transport_outcomes.py` at repo root.
3. `grep -rn '"dispatched"' orion services --include="*.py"` shows the string only in: the Literal definitions (schema files), the unchanged scoring-map key in `orion/feedback/builder.py`, `orion/consolidation/tensorize.py`'s generic pass-through, and the unrelated `orion/schemas/workflow_execution.py`/`orion-actions` workflow-schedule status (a different, pre-existing, unrelated status vocabulary — confirmed by inspection, not touched). No remaining code path constructs `dispatch_status="dispatched"` without evidence.
4. A read-only check (grep across `tests/`, and reasoning from the "no sender exists" fact already established) confirming no stored/historical `ExecutionDispatchFrameV1` data could contain a genuine evidenced `dispatched` status — asserted explicitly in the PR report per the task's compatibility-check requirement.
5. Model-validator regression test: constructing `ExecutionDispatchCandidateV1(dispatch_status="dispatched", ...)` with no `dispatched_at`/`result_ref`/`dispatch_error` raises `ValidationError`.

## Recommended next patch

P1 (the motor nerve itself — Layer 9 actually sends) per the parent spec, once this lands.
