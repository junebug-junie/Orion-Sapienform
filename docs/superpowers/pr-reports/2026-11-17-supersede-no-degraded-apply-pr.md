## Summary

Orion's nightly memory step retires an old memory when a newer one corrects it. This patch stops that step from retiring anything when it is running half-blind, or before its accuracy has been measured.

- Write time (nightly episode distill): the step applies nothing when the meaning index is not configured, fails, or cannot answer. It no longer falls back to word overlap. It also stops asking the model about more pairs.
- Write time: the step applies nothing when any model answer fails or does not parse, for either question.
- Write time: the step applies only while the latest precision eval for the current check version passed. The eval table arrives in the follow-up PR. Until then the gate reads as closed, so the step proposes, holds its links in the checkpoint, and logs why.
- The persist step applies only a plan that says `apply == "on"`. Checkpoints written before this patch apply nothing.
- One-time pass: `apply` refuses pairs found without the meaning index, unless `--allow-no-dense` is given. Each pair in `pairs.jsonl` now records `dense_on`.

## Outcome moved

Before this patch, the nightly step could retire a true memory using a word-overlap search with no measured precision. After it, nothing is retired until an eval run says the check is precise. Every refusal is logged with its reason (`episode_supersede_planned ... apply=refused refused_because=...`).

## Current architecture

`AdmissionRuntime._plan_supersede` (services/orion-durable-runs) loads live rows, then calls `plan_for_new_rows` (orion/memory/episode/supersede.py). That function caught any meaning-index failure and went on with shared words alone. `persist_node` passed every planned link to `persist_episode`, which called `apply_link`. Live as of 2026-11-17: 0 `superseded` events, so nothing has been applied yet.

## Architecture touched

- `orion/memory/episode/supersede.py`: `plan_for_new_rows` refusal reasons, `links_to_apply`, `eval_gate`, `check_failed`.
- `services/orion-durable-runs/app/admission_runtime.py`: reads the eval gate on its own connection; logs the refusal reason.
- `services/orion-durable-runs/app/episode_distill_graph.py`: persist applies only `links_to_apply(plan)`.
- `scripts/memory_supersede_pass.py`: `dense_on` per pair; apply refuses word-only pairs.

## Files changed

- `orion/memory/episode/supersede.py`: the never-degraded rule and the eval gate reader.
- `services/orion-durable-runs/app/admission_runtime.py`: the gate and the log line.
- `services/orion-durable-runs/app/episode_distill_graph.py`: persist reads `apply`.
- `scripts/memory_supersede_pass.py`: `dense_on` and the `--allow-no-dense` refusal.
- `orion/memory/episode/tests/test_supersede.py`, `test_supersede_pass_script.py`, `services/orion-durable-runs/tests/test_episode_distill_graph.py`: tests for each refusal.

## Schema / bus / API changes

- Added: plan keys `held_links`, `apply`, `refused_because`, `check_version` (checkpoint only).
- Removed: the word-overlap fallback for write-time apply.
- Renamed: none.
- Behavior changed: write time applies nothing until an eval passes (table `memory_supersede_eval_run`, follow-up PR).
- Compatibility notes: old checkpoints apply nothing; a missing eval table reads as a closed gate.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. pytest -q orion/memory/episode/tests/                       322 passed, 26 skipped (Postgres-backed; run in CI)
PYTHONPATH=.:services/orion-durable-runs pytest -q services/orion-durable-runs/tests/test_episode_distill_{graph,referent_policy,reconcile}.py
                                                                          24 passed, 1 skipped
```

## Evals run

```text
None in this patch. The precision eval and its gate table land in the follow-up PR (#3659 part 2).
Until then the gate is closed, so write time cannot apply.
```

## Docker/build/smoke checks

```text
Not run: no dependency, compose, or env change. Logic change only, covered by the tests above.
```

## Review findings fixed

- Finding: an unparseable second answer did not count as a check error, so other links in the same plan could still apply.
  - Fix: `run_check` marks it `bad_answer`; `check_failed` counts it.
  - Evidence: `test_a_malformed_second_answer_holds_every_link`.
- Finding: the pass's no-dense refusal ran before `--only`/`--exclude`.
  - Fix: filter first, then refuse.
  - Evidence: `test_apply_refuses_pairs_found_without_the_meaning_index` (exclude case).
- Finding: `degraded_pair_ok` read backwards.
  - Fix: renamed `found_with_dense`.
  - Evidence: same test.
- Finding: an unfinished eval run (NULL `finished_at`) would sort first.
  - Fix: `AND finished_at IS NOT NULL`.
  - Evidence: `EVAL_GATE_SQL`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

(From the clean primary `main` checkout after merge; the post-pull rebuild does this.)

## Risks / concerns

- Severity: low.
- Concern: write-time correction is effectively paused until the follow-up eval passes. This is the intended state (proposals are still logged and checkpointed).
- Mitigation: the follow-up PR adds the eval, the classifier, and the gate table.
- Issues: part of #3659 (closed by the follow-up PR).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3669
