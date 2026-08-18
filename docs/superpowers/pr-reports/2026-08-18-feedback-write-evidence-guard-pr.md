# PR report — R5b: gate the feedback-credit loop on real write evidence

Branch: `feat/feedback-write-evidence-guard`

## Summary

- R5a (`orion/field/credit_integrity.py`, shipped #1700) is report-only: it detects when a feedback-credited channel's value has no real write evidence behind it, but cannot stop the credit from being assigned, since it needs a whole window to find a silent run after the fact.
- This PR is R5b, the actual gate on the live credit-assignment loop (`orion/feedback/builder.py::build_feedback_frame`), built per Juniper's direct instruction ("build it") after the roadmap doc's open decision #1 was resolved (design against the current loop, not a hypothetically-fixed offline-suppression config) — satisfying CLAUDE.md 0A's "unless Juniper directly asks to implement" exception to proposal mode.
- New single-tick primitive `channel_write_backed()` in `orion/field/credit_integrity.py` reuses R5a's own mechanisms (`_classify_tick`, `regime._refresh_from_timestamps`) verbatim — no new staleness heuristic. It answers "should THIS credit decision trust THIS tick's value" at exactly the tick a decision is about to use, not swept over history.
- Withholds — never silently drops — positive/negative pressure-delta evidence and the `reliability_pressure` improved/worsened observation (reusing the previously-unused `outcome_kind="stale"` enum value) when the credited AFTER tick has no fresh write evidence, including the missing-`field_after` case, which is the most acute form of the same trap (100% missing evidence, not partial staleness).
- One-line rollback: `FeedbackPolicyV1.write_evidence_guard_enabled` (default `true`), flip to `false` in `config/feedback/feedback_policy.v1.yaml` + restart `orion-feedback-runtime`.
- Investigation before building this confirmed there is currently **no live action-value/ranker consumer** of `FeedbackFrameV1.outcome_score`/`positive_evidence` downstream — the real, live consequence of the trap this closes is `orion/consolidation/motif.py`'s pattern/expectation building reading `outcome_status`/`negative_evidence`, not a reward signal. Recorded honestly rather than overstating scope: this closes a real defect in what the system currently does (build false patterns from decayed data), not a hypothetical RL exploit that isn't wired yet.

## Outcome moved

Live gate check (`POSTGRES_URI=... scripts/check_merge_domination.py --gate`, same real Postgres this arc has used throughout): R5a's watch, unaffected by this PR, still finds real live drift (a genuine 31s `reliability_pressure` no-write window at 2026-08-18T04:27:03Z) — confirming both that R5a's existing surface is untouched and that this is not a hypothetical scenario: a feedback decision landing on that exact tick would previously have been credited (or penalized) off a value R5b would now correctly withhold.

Two review passes on this PR's own code (below) found and fixed one real bug (a duplicate `withheld_evidence` entry) and one real test-coverage gap (negative-direction gating was untested) before this shipped — same review discipline the rest of this arc has used, applied to itself.

## Current architecture

`config/feedback/feedback_policy.v1.yaml` → `orion/feedback/policy.py::load_feedback_policy` → loaded once at `orion-feedback-runtime` startup (`services/orion-feedback-runtime/app/worker.py:24`) → `FeedbackRuntimeWorker._tick()` calls `orion/feedback/builder.py::build_feedback_frame` once per tick with two `FieldStateV1` snapshots (`field_before`, `field_after`, from `FeedbackRuntimeStore.load_field_for_tick`/`load_latest_field_after` — both single-tick loaders, no window method exists) → published to `orion:feedback:frame` and stored.

## Architecture touched

- `orion/field/credit_integrity.py`: new `channel_write_backed()` — the single-tick sibling of R5a's batch `analyse_credit_integrity()`.
- `orion/feedback/builder.py`: `_gate_positive_delta_channels()` (new) gates `classify_pressure_deltas`'s input; the `reliability_pressure` improved/worsened block gains a `stale` branch.
- `orion/feedback/policy.py`: `FeedbackPolicyV1.write_evidence_guard_enabled` (new, default `true`).
- `orion/schemas/feedback_frame.py`: `FeedbackFrameV1.withheld_evidence` (new, additive, `extra="forbid"` on the model is unaffected since this has a default — old stored rows deserialize fine).
- `config/feedback/feedback_policy.v1.yaml`: explicit `write_evidence_guard_enabled: true`.

## Files changed

- `orion/field/credit_integrity.py`: `channel_write_backed()` + module docstring's "R5B, THE GATE" section.
- `orion/feedback/builder.py`: gate wiring, `stale` observation branch, `build_feedback_frame` docstring.
- `orion/feedback/policy.py`: `write_evidence_guard_enabled` field.
- `orion/schemas/feedback_frame.py`: `withheld_evidence` field.
- `config/feedback/feedback_policy.v1.yaml`: explicit guard flag.
- `tests/test_credit_integrity.py`: 6 new tests for `channel_write_backed()`.
- `tests/test_feedback_builder.py`: `_field()` helper gains `node_vector_updated_at` support (defaults to fresh, matching R5a's own live-measured 100%-stamped finding); 8 new tests; 2 pre-existing tests updated to pass realistic stamps (an unstamped fixture now correctly reads as unbacked — the guard doing its job on a fixture that didn't look like real data, not a bug).
- `tests/test_feedback_policy_loader.py`: `write_evidence_guard_enabled` default test; fixed one pre-existing, unrelated broken assertion (`agency_readiness`, a key removed in the 2026-07-22 SelfStateV1 burn that the test never got updated for) — disclosed drive-by, not scope creep.

## Schema / bus / API changes

- Added: `FeedbackFrameV1.withheld_evidence: list[str]` (default `[]`, additive). `orion:feedback:frame` (`feedback.frame.v1`) channel definition itself unchanged — confirmed via `scripts/check_definition_drift.py --gate`: 0 changes, PASS.
- Added: `FeedbackPolicyV1.write_evidence_guard_enabled: bool` (default `true`).
- No removals, no renames, no channel changes.
- Compatibility: checked every `FeedbackFrameV1` consumer (`orion/consolidation/{motif,expectation,tensorize,windows}.py`, `services/orion-feedback-runtime/app/{store,worker}.py`, `services/orion-hub/scripts/substrate_feedback_routes.py`, `services/orion-consolidation-runtime/app/store.py`) — none pattern-match exhaustively on `outcome_kind`, so the new `"stale"` producer is safe everywhere. One motif rule (`_detect_stable_after_dry_run`) now correctly fails to match a decayed-to-plateau reading it previously could have spuriously matched as `"unchanged"` — a real behavior change, and a positive one (closes exactly the false-motif trap this arc exists to close).

## Env/config changes

None (no env keys added/removed/renamed). `config/feedback/feedback_policy.v1.yaml`'s new `write_evidence_guard_enabled: true` key is config, not env — no `.env_example` sync needed. Confirmed via `scripts/check_service_env_compose_parity.py orion-feedback-runtime`: OK, all 14 keys still exposed.

## Tests run

```
/tmp/r4venv/bin/python -m pytest tests/test_feedback_builder.py tests/test_credit_integrity.py \
  tests/test_feedback_policy_loader.py tests/test_feedback_frame_schemas.py tests/test_feedback_extractors.py \
  tests/test_feedback_scoring.py tests/test_feedback_runtime_store.py -q
    81 passed
```

Full explicit feedback-surface sweep (avoids an unrelated pytest whole-tree collection quirk on this venv — spot-checked one flagged file standalone, it passes fine in isolation, confirmed unrelated to this diff):
```
/tmp/r4venv/bin/python -m pytest tests/test_chat_response_feedback_schema.py tests/test_credit_integrity.py \
  tests/test_feedback_builder.py tests/test_feedback_extractors.py tests/test_feedback_frame_schemas.py \
  tests/test_feedback_policy_loader.py tests/test_feedback_runtime_store.py tests/test_feedback_scoring.py \
  tests/test_feedback_transport_outcomes.py tests/test_measure_proposal_feedback_correlation.py \
  tests/test_sql_writer_chat_feedback.py -q
    109 passed
```

## Evals run

Mutation testing (targeted, following this arc's established method — hand-flip, confirm test failure, restore via `cp` from a saved original):

- `channel_write_backed()`: flipped `if verdict == "unknown": return None` → `return True`. **Killed** by `test_channel_write_backed_none_for_never_stamped_node_write`.
- `_gate_positive_delta_channels()`: flipped `if backed is not True:` → `if backed is False:` (lets a `None`-backed/unmapped channel fall through as if credited). **Killed** by `test_unmapped_channel_in_present_field_after_is_withheld` — added specifically to close this gap after the first mutation run showed it surviving with 0 failures across all 18 then-existing tests in the file.

Both mutants restored via `cp` from a pre-mutation saved copy; `diff -q` confirmed exact restoration before continuing.

No dedicated eval harness exists for `orion/feedback/` or `orion/field/` beyond `tests/` — consistent with the rest of this arc (R1-R5a also had none); the live `check_merge_domination.py --gate` run below is this module's closest thing to a periodic eval.

## Docker/build/smoke checks

```
POSTGRES_URI=postgresql://postgres:postgres@127.0.0.1:55432/conjourney \
  /tmp/r4venv/bin/python scripts/check_merge_domination.py --gate
    analysed 6000 ticks across 38 merge points
    feedback-credit watch: 1 finding(s) (policy window 30s)
        reliability_pressure no_write_in_window for 31s (13 ticks) [timestamp]
    merge domination gate: PASS
```
R5a's existing watch is unaffected by this PR (this script never calls the new gate — the gate lives in the live feedback-runtime path, not the cron watch) and found real live drift during this check, confirming the trap this PR closes is not hypothetical.

```
/tmp/r4venv/bin/python scripts/check_definition_drift.py --gate
    600 metric definitions (0 changed, 0 high severity) -- PASS
/tmp/r4venv/bin/python scripts/check_service_env_compose_parity.py orion-feedback-runtime
    OK -- all 14 .env_example keys are exposed via environment:
```

No Docker rebuild required — pure Python logic + config change, no new dependency, no Dockerfile/compose change.

## Review findings fixed

Independent subagent review (`orion-repo-agent`), adversarial per this arc's own established pattern:

- **Finding: duplicate `withheld_evidence` entry for `reliability_pressure`.** `channel_write_backed()` was called twice for the same channel/tick (once in `_gate_positive_delta_channels`, once in the dedicated `reliability_pressure` observation block), each independently appending an identical withheld-entry string — `execution_pressure`/`resource_pressure` got one entry each, `reliability_pressure` got two.
  - **Fix:** `_gate_positive_delta_channels` now returns the per-channel `backed_by_channel` result alongside the gated dict; the `reliability_pressure` block reuses that result instead of recomputing, and only appends to `withheld_evidence` itself in the (currently unreachable given the live policy yaml) fallback case where `reliability_pressure` isn't a policy-configured credited channel at all.
  - **Evidence:** `test_stale_reliability_write_produces_exactly_one_withheld_entry` — asserts exactly one entry, reproduces the reviewer's exact repro before the fix, passes after.
- **Finding: negative-direction gating was untested.** Every staleness test used a decrease (0.6→0.2); the code path suppressing a worsening (increase) delta on a stale write had no regression test.
  - **Fix:** `test_stale_write_withholds_negative_evidence_too` added.
  - **Evidence:** passes; confirmed via the same mutation-style reasoning that a future edit special-casing negative deltas to bypass the gate would now be caught.
- **Finding, cosmetic: dangling docstring reference.** `_gate_positive_delta_channels`'s docstring pointed at "`build_feedback_frame`'s docstring note", but `build_feedback_frame` had no docstring at all.
  - **Fix:** added a real docstring to `build_feedback_frame` documenting the "withhold both directions" design choice, and fixed the cross-reference.
- **Confirmed correct by review, not taken on my word:** the True/False/None semantics of `channel_write_backed()` (traced against `regime.py::_refresh_from_timestamps` directly, including the single-sample coverage-floor degenerate case); the dimension-vs-channel namespace handling (`resource_pressure` fed by raw channel `pressure`, correctly absorbed inside the primitive so no caller needs to know); the `field_after=None` path in both integration points; the `write_evidence_guard_enabled=False` full-restore rollback; no circular import; schema/config conventions match this repo's existing patterns exactly; every other `FeedbackFrameV1` consumer checked directly for silent breakage, none found; the module docstring's factual claims spot-checked against the real code.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-feedback-runtime/.env \
  -f services/orion-feedback-runtime/docker-compose.yml up -d --build
```
Only `orion-feedback-runtime` (policy is loaded once at worker startup) needs a restart for the new `write_evidence_guard_enabled` policy key or its rollback to take effect. Nothing else in this PR is a long-running service.

## Risks / concerns

- **Severity: low.** The `"stale"` observation carries the same `confidence=0.7` constant as a genuinely-observed `"unchanged"` reading — arguably a withheld/no-evidence observation should carry lower confidence. Minor calibration question, not a correctness bug; flagged by review, not fixed here to keep this patch thin (reuses an existing constant rather than inventing a new one for a single call site).
- **Severity: low, inherited from R5a, restated honestly.** An outage straddling the exact tick where a channel's winner type switches (node ↔ capability) is still invisible — same residual limit R5a already documented, unchanged by this PR since `channel_write_backed()` reuses the same per-tick classification.
- **Severity: informational.** No live action-value/ranker consumer of this credit signal exists today (confirmed by direct investigation before building this) — this closes a real defect in the system's current pattern/expectation building (`orion/consolidation/motif.py`), not a currently-active reward-hacking exploit. If a ranker is ever wired to `outcome_score`/`positive_evidence`, this gate is already in place ahead of it, which is the point.

## Follow-up (not in this PR)

`docs/superpowers/specs/2026-08-13-phase5-liveness-scope.md`'s status table still needs an update marking R5b shipped — deliberately NOT touched here because PR #1702 (still open as of this PR) already has an in-flight rewrite of the same "R5" section (splitting it into R5a/R5b and resolving open decision #1). Editing it here too would either duplicate that work or conflict with it. Once #1702 merges, a small follow-up doc-only PR should add R5b's shipped status and this PR's link to the table.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1709
