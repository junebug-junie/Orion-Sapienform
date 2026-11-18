# PR: open strain alone no longer rewrites an accepted reply (#3618)

## Summary

- The response-repair pass no longer runs when the 5b judge accepted the draft (`aligned`) and only flagged open strain (`strain_unresolved`). The draft ships as written.
- The skip is recorded: `response_repair_ran=false`, `response_repair_reason=strain_only_not_repaired`, and the log line `response_repair_skipped reason=strain_only_not_repaired`.
- An `unsupported claim:` note now sends the draft to repair even under an `aligned` verdict. Before, that happened only when strain was open too.
- `is_repair_no_change` now also reads a reply that is only the verdict in words ("No change needed.", "No changes required."). Live 363a9723 shipped "No change needed." in place of a 1,567-char answer. A short reply that opens with "No change" and goes on is still a reply.
- Removed `repair_may_keep_draft_on_truncation` and its keep-on-truncation branch. After this change it could never return True. A cut-off repair always fails closed.

## Outcome moved

About 9 repair calls a day stop. They used about 209 GPU-seconds, 4.8k output tokens and about 113k prompt tokens a day, and added 22.7 s on average (p90 48.5 s) to each turn they ran on. Three kinds of harm stop: a PASS decision turned into an unprompted message, a reply replaced by meta text, and a caring question cut from a reply to Juniper.

## Current architecture

`needs_response_repair` (`orion/harness/finalize.py`) returned True for `misaligned`, `uncertain`, or `strain_unresolved`. `strain_unresolved` is a boolean the 5b judge LLM writes ("true when open strain remains after draft", `orion/cognition/prompts/harness_finalize_reflect.j2:218`). The strain itself is `strain_refs` from the stance step (`orion/thought/stance_react.py:178`): ids of arcs, open loops, earlier turns and grammar events, which are Orion's own open loops. `orion/substrate/appraisal/finalize_draft_v1.py:108` passes the ones the coalition did not attend as `strain_shift_refs`. A reply rewrite cannot close any of them. The repair prompt (`orion_response_repair.j2`) tells the model the draft was "REJECTED" and to fix it to match the verdict and notes. On a strain-only turn, the verdict and the notes accepted the draft.

## Evidence (live, read-only)

All repairs with draft and final text, `cockpit_turn_sighting` stage `finalize`, 2026-09-13..11-17 (927 repairs, 464 strain-only):

| strain-only outcome | all 464 | since 11-05 (119) |
|---|---|---|
| identical to draft | 259 (56%) | 78 (66%) |
| under 3% changed | 78 | 19 number/date edits + 1 punctuation |
| 3-15% changed | 35 | 9 rewrites / additions |
| over 15% changed | 92 | 9 sentence cuts, 2 PASS flips, 1 meta-text reply |

- None of the edits added or removed content about the strain.
- Over 15% changed (92, all time): 24 reasoning leaks ("We need answer user's request..."), 2 PASS → message, 2 message → PASS, 16 shortened, 43 other rewrites. Most of these predate the guards from 10-04..11-03.
- Real fixes: about 4 of 119 since 11-05 match a slip the judge's own notes named (0f5abec3, 100701df, 17508cfb). They were fixed by accident: aligned drafts without strain and with the same kind of note never repaired. Follow-up #3682.
- Juniper's own chat turns: 21 strain-only repairs, 15 identical, 5 tiny, 1 harmful (ee38b338 cut "How's the bedroom holding up temperature-wise?"). Repair added 7.8 s on average.
- Cost (`llm_call_usage` verb `orion_response_repair`, 10-30..11-17): strain-only 175 calls, 3,970 GPU-s, 91.7k completion tokens, mean prompt 12.3k tokens, mean 22.7 s.
- Later judges: nothing judges the repaired text on this path. The unsupported-claim recheck is the only re-judge, and it does not run here. No human feedback on these turns was found.
- Replay of the new rules on all 927 repairs: 464/464 strain-only skip, 463/463 others still repair. The widened NO_CHANGE check catches 1 new reply (363a9723) and no false positives.

## Architecture touched

- `orion/harness/finalize.py`: repair gate, skip reason, NO_CHANGE reader, dead truncation branch removed.
- `orion/hub/turn_orchestrator.py`, `services/orion-harness-governor/app/bus_listener.py`: a reason labels a revision only when the repair ran.

## Files changed

- `orion/harness/finalize.py`: gate, `response_repair_skip_reason`, `STRAIN_ONLY_NOT_REPAIRED_REASON`, `is_repair_no_change`, removed `repair_may_keep_draft_on_truncation`.
- `orion/hub/turn_orchestrator.py`: revision label guard.
- `services/orion-harness-governor/app/bus_listener.py`: same guard in the revision log.
- `orion/cognition/verbs/orion_response_repair.yaml`, `services/orion-harness-governor/README.md`: describe the new trigger.
- `orion/cognition/prompts/orion_response_repair.j2`: "rejected or flagged" (aligned drafts with a claim note now reach it).
- `docs/superpowers/specs/2026-09-12-kill-voice-finalize-design.md`: new gate and reason table.
- `services/orion-hub/static/js/draft-revision.js` (+ test): comment that the `strain_unresolved` label is kept only for old turns.
- Tests: `orion/harness/tests/test_response_repair_gate.py`, `test_response_repair_budget.py`, `test_response_repair_scaffold_echo.py`, `test_perception_claim_recheck.py`, `services/orion-hub/tests/test_draft_first_display.py`, `services/orion-harness-governor/tests/test_draft_preview.py`.

## Schema / bus / API changes

- Added: response_repair_reason value `strain_only_not_repaired` (with `response_repair_ran=false`).
- Removed: reason values `strain_unresolved` and `strain_unresolved_kept_draft:*` are no longer produced.
- Renamed: none.
- Behavior changed: an aligned draft with an `unsupported claim:` note is repaired.
- Compatibility notes: field types unchanged. `strain_unresolved` still reaches `surprise_resolved` on the outcome molecule and the equilibrium chat-turn metacog gate.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no. Local `.env` sync: not needed. Skipped keys: none.

## Tests run

```text
pytest orion/harness + test_draft_first_display       999 passed
node --test draft-revision.test.js                    8 passed
pytest services/orion-harness-governor/tests          106 passed, 1 failed (test_fcc_warm_pool_wiring, fails on main too)
pytest orion/harness/evals services/orion-hub/tests -k "orchestrator or draft or revis or repair or finalize"   210 passed, 6 skipped
pytest services/orion-equilibrium-service/tests       266 passed
```

## Evals run

```text
Live replay of the new gate and NO_CHANGE reader over 927 real repairs (cockpit_turn_sighting): see Evidence.
No checked-in eval added: the gate tests pin the rule, and the replay needs live rows.
```

## Docker/build/smoke checks

```text
Not run. Pure Python logic in the governor and Hub images; no compose, env or dependency change.
Live effect is UNVERIFIED until deploy: check that response_repair_reason='strain_only_not_repaired' appears and no 'strain_unresolved%' repairs follow.
```

## Review findings fixed

Review ran in a subagent on ebdb7207c. All findings fixed:

- Finding: the "no change" words check matched short real replies ("No change on the cabinet: still 31 C."), which would ship a rejected draft.
  - Fix: the reply must be only the verdict (fullmatch of "no change(s) [is/are] [needed/required/necessary]").
  - Evidence: new cases in `test_response_repair_scaffold_echo.py`; live replay over 927 replies still catches only 363a9723.
- Finding: a test passed only because of the 80-char cap.
  - Fix: cap removed; short real-reply cases added.
  - Evidence: same test file.
- Finding: aligned + claim with no Situation notes recorded `unsupported_claim`, which reads as "re-check passed".
  - Fix: that case now records `unsupported_claim_unchecked`.
  - Evidence: `test_an_aligned_claim_repair_without_notes_says_unchecked`.
- Finding: no chain test for aligned + claim through the re-check.
  - Fix: `test_an_aligned_draft_with_a_claim_is_repaired_and_rechecked`.
  - Evidence: passes.
- Finding: design spec still stated the old gate.
  - Fix: spec updated (gate, reason table, truncation note).
- Finding (nit): Hub JS label and its test fixture used `strain_unresolved`.
  - Fix: fixture uses `misaligned`; label kept with a comment for old turns.
- Finding (nit): repair prompt said "rejected" only.
  - Fix: "rejected or flagged".

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run from the clean primary `main` checkout after merge (the post-pull rebuild does this).

## Risks / concerns

- Severity: low. Concern: about 3% of old strain-only repairs fixed a small slip the judge named. Mitigation: follow-up #3682 (opened).
- Severity: low. Concern: aligned drafts with an unsupported-claim note now cost a repair (live: 2 in 30 days). Mitigation: that is the intended defect path.
- Severity: low. Concern: the hourly self-signal rollup (`orion/attention/self_signal_outcomes.py`, outcome `response_repair` = `response_repair_ran`) will see its repair rate step down by about the strain-only share (244 of 524 in 30 days) from deploy on. Mitigation: the step is a real change in behavior, not noise. Read associations across the deploy date with that in mind.
- Issues opened: #3682. Issues referenced: #3618.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3686

## Follow-up: Orion's review questions (session thread 4cd4d35e603e)

- Spec acceptance test 4 still said strain triggers repair. Rewritten for the strict gate (commit b7b9554dc). The code never kept strain as a trigger.
- The spec now says where the unsupported-claim re-check lives and how it is bounded.
- `surprise_resolved` is unchanged by this PR. It reads the 5b reflection, not whether a repair ran. 14 days: 275 of 2,375 aligned turns had strain open and were unresolved before and after.
- A failed required repair is now recorded as `response_repair_ran=true` with its reason (outcome and run artifact). Live 30 days: 14 `gpu_pool_unavailable` + 4 truncated repair failures were all recorded `ran=false`. New acceptance test `test_a_failed_required_repair_is_recorded_as_ran_and_failed`.
- Issue opened: #3690. On a failed repair, Hub labels a draft that 5b rejected as "not checked".
