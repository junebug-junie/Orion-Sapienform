## Summary

- When a queued investigation finally runs, Hub now re-reads every prior its prompt shows. It adds a short block at the top naming any prior whose confidence, test count or closed status changed since the prompt was written (old -> now).
- New knob `HUB_CURIOSITY_MAX_QUEUED_INVESTIGATIONS` (default `1`, ON): a scheduled tick admits no new investigation while that many earlier ones are still waiting for a GPU slot.
- The investigation prompt now lists up to 5 open lived questions with their current answer. Each comes with a ready-to-run `:LivedAnswer` MERGE, so an investigation can revise an answer without waiting for that question to be picked again. Hub copies any answers the run wrote into `self_concept_history`.
- The `:LivedAnswer` template now lives in one place (`orion/curiosity/self_inquiry.py`) and is used by both prompts. It is keyed on `(run_id, question_id)` and fills in `revises` itself.
- `revises` (and the "previous answer" shown to a self-inquiry run) now comes from the newest answer in the graph. Before, it came from the mirror table, which skips drafts that have no evidence.

## Outcome moved

- Stale numbers: on 10-09, seven runs waited 8-12 h and were shown a prior at 0.55 / never tested after it had moved six times. Replaying run 334d78's real stored brief against the live graph, the new block names that prior as `confidence 0.55 -> 0.82, tested 0 -> 6`, plus 6 other priors that moved.
- Queue: replaying 10-09 against `durable_admission_runs`, each dispatch saw 0, 1, 2 ... 6 earlier briefs still queued. With the cap at 1, six of those seven admissions would have waited for the queue to drain.
- Lived lineage: `lived.her_team_unnamed`'s 10-09 answer revised 453e (10-01) instead of e511 (10-04, an evidence-less draft that was never mirrored). The new graph read returns e511 for that shape, and it returns the real newest answer for all 19 open lived questions live.

## Current architecture

Briefs are built and frozen when the run is admitted. `_prompt_for_attempt` sent the prompt byte-for-byte on attempt 1. Answers to lived questions could only be written from the self-inquiry prompt. The previous answer was read from `self_concept_history`.

## Architecture touched

Hub curiosity loop (`services/orion-hub/scripts/curiosity_investigation.py`), the shared prompt modules (`orion/curiosity/kickoff_prompt.py`, `self_inquiry.py`, `self_inquiry_prompt.py`), and Hub config. Hub reads orion-durable-runs' `durable_admission_runs`/`durable_resource_events` read-only through its own pool.

## Files changed

- `orion/curiosity/kickoff_prompt.py`: offered-prior parser, drift block, lived-questions section.
- `orion/curiosity/self_inquiry.py`: shared MERGE template, newest-answer reader, per-run answers reader, `OpenLivedQuestion`, per-question mirror entry ids.
- `orion/curiosity/self_inquiry_prompt.py`: uses the shared template and fills in `revises` (including the early write, which used to hard-code `""`).
- `services/orion-hub/scripts/curiosity_investigation.py`: drift at turn start, backlog gate, open lived questions in the kickoff, graph-first previous answer, mirrors investigation answers (skips rows already mirrored).
- `services/orion-hub/{app/settings.py,.env_example,docker-compose.yml,scripts/main.py,README.md}`: the new knob, plus docs.
- Tests: `services/orion-hub/tests/test_curiosity_fresh_priors_and_answers.py` (new), `orion/curiosity/tests/test_self_inquiry_prompt_drawn_question.py`.

## Schema / bus / API changes

- Added: none (no new channel or schema).
- Behavior changed: the `:LivedAnswer` MERGE key is now `(run_id, question_id)`. This is backward compatible, because older nodes already carry `question_id`. Investigation-written mirror rows use entry id `self-lived:<run>:<question>`. Self-inquiry ids are unchanged.

## Env/config changes

- Added keys: `HUB_CURIOSITY_MAX_QUEUED_INVESTIGATIONS=1`
- `.env_example` updated: yes
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (primary checkout `services/orion-hub/.env`)
- Skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-hub/tests/test_curiosity_fresh_priors_and_answers.py  -> 19 passed
pytest orion/curiosity/tests + resume prompt tests                          -> 139 passed
pytest services/orion-hub/tests -k "curiosity or self_inquiry or lived"     -> 461 passed, 3 skipped
pytest services/orion-durable-runs/tests                                    -> 362 passed, 76 skipped
root tests importing the touched modules (12 files)                        -> 363 passed
check_settings_defaults orion-hub, env parity, env single source, definition drift, metric lineage gates -> PASS
(test_fcc_model_labels_api fails on main too; it depends on the environment and is unrelated)
```

## Evals run

```text
No eval harness covers prompt freshness. Live replays (read-only) stand in:
- 334d78 stored brief vs live graph -> drift block names 7 moved priors
- QUEUED_INVESTIGATIONS_SQL live -> 0 now (2.9 ms); historical replay 0..6 on 10-09
- read_current_lived_answers on 19 open lived questions -> newest per question
```

## Docker/build/smoke checks

```text
Not deployed (per instruction). No Docker build run.
```

## Review findings fixed

- Finding: question ids from Orion-minted rows were not checked before going into the MERGE.
  - Fix: invalid ids are dropped with a warning, and a graph-sourced `revises` that doesn't match the run-id format is left as the placeholder.
  - Evidence: `test_invalid_graph_run_id_is_not_spliced_into_revises`
- Finding: a redelivered completion event could republish answers at version+1.
  - Fix: skip any entry_id that is already in `self_concept_history`.
  - Evidence: `test_redelivered_completion_does_not_republish_mirrored_answers`
- Finding: the drift check compared raw values against two-decimal display values.
  - Fix: compare the displayed strings.
  - Evidence: `test_rounding_to_the_same_display_is_not_drift`
- Finding: on a retry, the drift wording blamed the queue for moves the earlier attempt may have made, and the drift + resume ordering had no test.
  - Fix: retry-specific wording, plus an ordering test.
  - Evidence: `test_retry_puts_drift_before_resume_and_names_the_earlier_attempt`
- Finding: the cap misses retried runs and self-inquiry briefs.
  - Fix: documented as a deliberate non-goal (the drift block covers both).

## Restart required

```bash
./scripts/safe_docker_build.sh orion-hub up -d --build
```

orion-durable-runs needs no restart (no change in what it consumes).

## Risks / concerns

- Severity: low. Concern: the offered-prior parser depends on `Prior.preview`'s format. Mitigation: a round-trip test fails if the format changes.
- Severity: low. Concern: Hub reads orion-durable-runs' registry table directly. Mitigation: read-only, fails open with a warning, and the drift block still corrects stale numbers.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2601

🤖 Generated with [Claude Code](https://claude.com/claude-code)
