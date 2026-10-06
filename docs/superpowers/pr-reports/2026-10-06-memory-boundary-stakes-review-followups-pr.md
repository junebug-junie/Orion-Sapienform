# #2502 review follow-ups: live-effect measurement, walker test, rendered prompt version

Supersedes the "shadow-only" claim in `2026-10-06-memory-boundary-and-stakes-prompts-pr.md` (#2502, already merged).

## Summary

- **#2502's boundary definition changes the LIVE memory path, not only the shadow episodes.** The #2502 report said the score only mattered under the shadow Rule 3. That was wrong. This PR measures the live effect and corrects the record. The live change is kept on purpose (orchestrator decision, in line with Juniper's "episodes, not turns").
- The re-score eval now also does two more things. It replays the live window rule. It logs the other three answer lines from the same call (NOVEL, SHIFT, MEMORY) and every live decision that reads them, with two runs per prompt so real changes can be told apart from run-to-run noise.
- A new test feeds real token streams from the classify lane, for the new prompt, through the live logprob walker. The old test only exercised the text fallback.
- The distill run now records the version of the prompt template it actually rendered, not the version the brief claims. The validator judges each answer by that version's rules. An answer to the old v2 prompt keeps the distiller's own low/high instead of being forced to "high, no reason". Escalations under v3 are labelled `unjudged`, and the report no longer prints "high: None".

## Outcome moved

**Live window path, replayed over 30 days (106 of Juniper's turns, live `metacog_background` lane, same turns for both prompts, two runs each):**

| | before #2502 | after #2502 |
|---|---|---|
| live windows | 89 | 49 |
| mean / median turns per window | 2.18 / 2 | 3.14 / 3 |
| closed by the judge (score at least 0.85) | 85 | 31 (33 on run 2) |
| closed by the 90-minute gap fallback | 3 | 17 (15 on run 2) |

What this means in practice:
- Fewer crystallization intake runs per day, and each window holds more of one conversation.
- More windows now close on the clock instead of on the judge.
- The old path still takes one prompt per window as the row's summary (`_window_summary`, `orion/memory/crystallization/intake_consolidation_window.py:97`, which picks the last non-junk prompt). So longer windows mean fewer candidate rows, and the earlier turns of a window are not summarized on their own. That is the intended "episodes, not turns" direction, but it is a visible change to the crystallization queue.

**Other answer lines from the same call** (first-pass values, 105 turns). Each entry gives turns flipped before-to-after, then run-to-run flips:

| line / consumer | before | after | flips | noise |
|---|---|---|---|---|
| SHIFT labels (STANCE / TOPIC / NONE / REPAIR) | 30 / 62 / 9 / 4 | 26 / 68 / 5 / 6 | 9-12 | 3-4 |
| `retrieval_intent` relational (STANCE, novelty at least 0.35) | 30 | 26 | 4-7 | 3-4 |
| `retrieval_intent` open_loop (REPAIR, novelty at least 0.35) | 4 | 6 | 2 | 0-2 |
| turn-change substrate signal (novelty at least 0.65, confidence at least 0.15) | 91 | 95 | 10 | 1-3 |
| consolidation gate `substantive_shift` | 95 | 98 | 3 | 0 |
| recall-skip novelty floor (novelty under 0.25) | 9 | 7 | 2 | 0 |
| session reappraisal | 3 | 2 | 2-3 | 0-1 |
| MEMORY significance at least 0.40 (mean 0.60 to 0.75) | 65 | 80 | 15 | 1 |

**Verdict: the only material live behavior change is the window closing.** The SHIFT and NOVEL lines move a little more than noise, but no consumer's decision changes by more than a few turns in 105.
- The turn-change substrate signal fires about 4% more often.
- MEMORY significance moved clearly, but nothing live decides on it. It sets the gate's reason label (since Stage 0A the gate cannot skip a non-junk window), the `[sig=]` transcript prefix, and the evidence note text.
- Not traced: whether cortex-exec's `recall_skip_gate` and `retrieval_intent` read this consolidation appraisal for the same turn. If they read another source, their rows above do not apply.

## Current architecture

- The LIVE window rule is `window_fetch.legacy_close_decision`, called from `worker.py:402-447`. `legacy_view()` hides the wall-clock phase from it, so every turn looks like phase "unknown". A window closes when the boundary score is at least `MEMORY_BOUNDARY_LLM_ONLY_THRESHOLD` (0.85). Failing that, it closes when the gap between its last two turns is at least 5400 s. Each close runs the live crystallization intake. The shadow Rule 3 is a second reader of the same score.
- Scores come from the logprob walker, `app/boundary.py::scores_from_llm_result`. `parse_classify_lines` is only the text fallback.
- The distill graph rendered the current template but stored `brief.prompt_version`. That version comes from memory-consolidation's image, which can differ during a deploy. The validator always required a stakes category, so a v2-answered checkpoint would have been stored all-high with a NULL reason.

## Architecture touched

- Shared library: `orion/memory/episode/{distill,validate,report}.py` and the template's first line.
- durable-runs: `app/episode_distill_graph.py`. It adds a `rendered_prompt_version` state key and a helper.
- Evals and tests only in memory-consolidation. No env, bus, or SQL change.

## Files changed

- `services/orion-memory-consolidation/evals/run_boundary_prompt_rescore_eval.py`:
  - `legacy_live_windows` / `legacy_summary`, the live rule replay;
  - `consumer_flags` / `other_lines_report`, for the NOVEL/SHIFT/MEMORY lines and their consumers;
  - `--runs` (default 2).
- `services/orion-memory-consolidation/evals/test_boundary_prompt_rescore_eval.py`: tests for the replay, the consumer thresholds, and the flip counting.
- `services/orion-memory-consolidation/evals/results/2026-10-06-boundary-prompt-rescore-summary.json`: the re-run with two runs per prompt. Scores and counts only.
- `services/orion-memory-consolidation/tests/test_boundary.py` + `tests/fixtures/boundary_prompt_logprob_streams.json`: real lane token streams for two synthetic turn pairs (tomato seedlings; a car's check-engine light), fed through the walker. No private text.
- `orion/cognition/prompts/memory_episode_distill.j2`: first-line comment `prompt_version: memory_episode_distill.v3`.
- `orion/memory/episode/distill.py`: `template_prompt_version()` and `UNMARKED_TEMPLATE_VERSION` (v2).
- `services/orion-durable-runs/app/episode_distill_graph.py`: stamps `rendered_prompt_version` at render time; `rendered_prompt_version(state)` helper. It stores and validates against the rendered version. A checkpoint without the stamp counts as v2.
- `orion/memory/episode/validate.py`:
  - `stakes_category_required(prompt_version)`, true from v3 on;
  - `UNJUDGED_STAKES_LABEL`;
  - a `prompt_version` argument on `validate_distillation`.
- `orion/memory/episode/report.py`: "no category" instead of "high: None"; lists the stakes events.
- `services/orion-memory-consolidation/evals/run_episode_distill_eval.py`: validates against the rendered template version.
- `docs/superpowers/pr-reports/2026-10-06-memory-boundary-and-stakes-prompts-pr.md`: a correction pointer to this report.

## Schema / bus / API changes

- Added: durable-run state key `rendered_prompt_version` (additive; a missing key reads as v2).
- Behavior changed:
  - `episode_distill_run.prompt_version` is now the version actually rendered;
  - for a v3 answer, a stakes escalation without a category stores `stakes_reason='unjudged'` (#2502 stored NULL);
  - a v2 answer keeps the distiller's stakes, logged as `stakes_uncategorized`, and still escalates on a named high category or on `asks_direction`.
- Removed / renamed: none.

## Env/config changes

- None. `.env_example` is unchanged; no sync needed.

## Tests run

```text
pytest orion/memory/episode/tests tests/test_turn_change_classify.py services/orion-memory-consolidation/tests services/orion-memory-consolidation/evals -q
  467 passed, 12 skipped
(services/orion-durable-runs) pytest tests/ --ignore=tests/test_admission_review_regressions.py -q
  269 passed, 62 skipped
  (test_admission_review_regressions.py fails to COLLECT when run from the service dir: ModuleNotFoundError 'orion',
   a sys.path issue in that file, unrelated to this change)
Static gates from .github/workflows/orion-static-gates.yml: see the PR checks; run locally, all PASS (below).
```

## Evals run

```text
python services/orion-memory-consolidation/evals/run_boundary_prompt_rescore_eval.py --runs 2 \
  --summary-json services/orion-memory-consolidation/evals/results/2026-10-06-boundary-prompt-rescore-summary.json
  106 turns, 105 scored x 2 runs x 2 prompts, 0 errors (numbers above)
```

The distill stakes eval was not re-run. Its committed numbers are the model's proposals before validation, so the validator change does not affect them.

## Docker/build/smoke checks

```text
No build or deploy (instructed). Evals called the live gateway over the bus, read-only on Postgres.
```

## Review findings fixed

- Finding (HIGH): the #2502 report said the boundary prompt only matters under shadow Rule 3, but the live legacy rule closes windows on a score of at least 0.85.
  - Fix: the live rule is replayed in the eval; this report corrects the claim; #2502's report points here.
  - Evidence: live windows 89 to 49, mean 2.18 to 3.14 turns, closes judge/gap 85/3 to 31/17 (run 2: 33/15).
- Finding (MEDIUM): NOVEL / SHIFT / MEMORY were unmeasured.
  - Fix: their distributions and each live consumer's decision are logged, two runs per prompt.
  - Evidence: the table above. SHIFT flips 9-12 against noise 3-4. No consumer moves materially.
- Finding (LOW): the test exercised the text fallback, not the walker.
  - Fix: `test_logprob_walker_scores_real_streams_for_the_defined_boundary_prompt`, using real lane streams with BPE-split " TOP"+"IC" and trailing spaces.
  - Evidence: scoring_source is logprobs, and the scores are not the 0.15/0.85 fallback constants. The continuation scores under 0.5; the unrelated switch scores at least 0.85.
- Finding (LOW): wrong stored version on a mixed deploy, v2 checkpoints forced to "high, no reason", and "high: None" in the report.
  - Fix: the rendered version is stamped and validated against; escalations are labelled `unjudged`; the report reads "no category".
  - Evidence: `test_stored_prompt_version_is_the_rendered_template_not_the_brief`, `test_checkpoint_without_a_stamp_was_rendered_from_v2`, `test_v2_answer_is_not_forced_high_but_real_signals_still_escalate`, `test_report_never_prints_high_none`.

## Restart required

Run after merge, from the primary checkout on main (one line). This also deploys #2502 if it is not live yet:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-memory-consolidation up -d --build && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- **Medium. Fewer live crystallization rows.** Live windows drop about 45%, and the earlier turns of a long window are not summarized on their own (one prompt per window). This is deliberate. Watch the crystallization queue volume after deploy; the shadow episode writer is the replacement for this path.
- **Medium. The boundary wording was tuned on the same 30 days it is measured on** (from #2502, unchanged). Re-run the eval on a fresh month.
- **Low. Consumer source not traced.** Whether cortex-exec's recall gates read this appraisal is not traced.
- **Low. A missing version string is treated as current (strict).** Only the eval and tests call the validator without a version.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2506

🤖 Generated with [Claude Code](https://claude.com/claude-code)
