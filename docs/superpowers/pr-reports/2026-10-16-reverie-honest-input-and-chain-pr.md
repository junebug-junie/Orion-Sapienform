# Reverie: honest input and a chain that develops

## Summary

- Reverie (Orion's daydreams) now sees the real number behind what won its attention. Each open loop in the prompt carries its measurement in plain words, for example `reading 0.55 (usual 0.00, 90th pct 0.00; top 2.0% of its own week, band high)`. A missing field is left out. Nothing is made up.
- For a body-signal loop, "why it surfaced" is now the selector's own reason (`body unusual for itself (high), event 47s old`). The old text, "novel or unresolved current-turn target with substrate pressure", is gone from these loops. It was false when no chat turn existed.
- An earlier verdict on a loop is shown only when it is less than 48 hours old, and the prompt says who recorded it. Only a verdict recorded by Juniper is called ground truth. A system mark (`system:implicit_decay`) is called bookkeeping.
- The prompt now says what happened: one thing won your attention. It no longer says "the coalition of concerns".
- A calm tick (nothing won attention) no longer starts a chain. No chain row is written, and no "unknown" cooldown is written.
- A chain is now a train of thought. Each step sees what the earlier steps said and the previous step's `next_focus`. Each step is told to go one step further. A step that returns no `next_focus` ends the chain (`thought_complete`). The dead `ORION_REVERIE_DRIFT_TEMP` knob is removed.

## Outcome moved

The live audit on 2026-10-14 found four failures. Each one now has a fix and a test:

| Failure (live) | Now |
|---|---|
| A verdict from 2026-09-21 by `system:implicit_decay` was attached to every later spike of the execution signal. The prompt called it human ground truth. About 40 thoughts/day said execution was "unattended for over 22 days". | Live check: `load_recent_loop_outcomes(['open-loop-7376a3da4050', ...])` returned the 24-day-old decay verdict on main. On this branch it returns nothing for that loop. A fresh system verdict (2 h old) is still attached and is labelled `recorded automatically by the system (system:implicit_decay)`. |
| The prompt had only the id, the description, and boilerplate. The measurement was dropped. | The measurement and the world-first reason are rendered (see the before/after below). |
| The 4 steps of a chain got the same input, so the last step restated the first. `next_focus` was filled 0 times out of 628. | Steps feed forward, and `next_focus` is asked for and used. An empty `next_focus` ends the chain. |
| About 164 `no_coalition` chain rows/day with `theme=unknown`. These were calm ticks. | A calm tick returns before any chain row or cooldown is written. |

BEFORE numbers (eval, last 24 h, live data written by main's code, run 2026-10-16 ~02:50 UTC):

```text
$ python services/orion-thought/evals/reverie_chain_development_eval.py --hours 24 --psql-container orion-athena-sql-db
thoughts 644, chains 167
steps_per_chain {1: 2, 2: 6, 3: 6, 4: 153}
median_bigram_overlap_first_vs_last_step   0.179
same_subject_cross_chain_baseline          0.130   (157 pairs)
pct_opening_the_coalition                  80.1
next_focus_fill_rate_pct                   0.0     (of 477 steps that were not a chain's last)
chains_last_step_without_next_focus        167
thoughts_saying_unattended_for             20

substrate_reverie_chain, last 24 h:
no_coalition theme=unknown 164 | max_steps 149 | pressure_discharged 13 | no_coalition (real theme) 5
```

How to read the numbers: the first and last steps of the same chain overlapped more (0.179) than the first steps of two separate chains on the same subject (0.130). This means a chain repeated itself more than two separate starts did. After deploy, re-run the same command. A chain that develops should show overlap at or below the baseline, a nonzero `next_focus` fill rate, some chains ending in `thought_complete`, and `unattended for` near 0. The AFTER numbers are UNVERIFIED until this is deployed.

## Current architecture

- `services/orion-thought/app/chain.py::run_reverie_chain` read the broadcast and started a chain even when nothing had won. It ran up to 4 calls of `run_reverie_once`, all with the same input. `chain_context=(chain_id, index)` only labelled the output.
- `app/reverie.py::_open_loops_for_prompt` kept only `id/description/why_it_matters/target_type` and attached any verdict found by `store.load_recent_loop_outcomes`, however old.
- `orion/cognition/prompts/reverie_narrate.j2` dumped `open_loops` as a Python dict repr. It said "the coalition of concerns that just won", and it said every verdict was "recorded by a human ... ground truth".

## Architecture touched

- orion-thought only, at runtime: `reverie.py`, `chain.py`, `store.py`, `settings.py`.
- Shared prompt `orion/cognition/prompts/reverie_narrate.j2`. It is rendered by orion-cortex-exec, but it changes no code there.
- Schema: `orion/schemas/reverie.py` adds `TerminalReason` `"thought_complete"`. The change is additive. `SpontaneousThoughtV1.next_focus` already existed and is now produced.
- Analytics: `dim_reverie_outcomes.sql` gets a `thought_complete` row. The `marts.yml` accepted values get `thought_complete`. `resource_deferred` was a dim row missing from the dim's own accepted values, so it is added as well.
- Metric routing: the 7 `node:substrate.*.prediction_error` entries in `config/field/field_channel_glossary.v1.yaml` now list `services.orion-thought.app.reverie:loop_measurement_text` as a prompt site. The metric lock is re-locked (`routing_changed` x7). No formula or threshold changed.
- The shared producer `orion/substrate/attention/scoring.py` is not changed. The boilerplate is replaced in the reverie consumer only.

## Files changed

- `services/orion-thought/app/store.py`: `fresh_loop_outcomes` (pure) and `LOOP_OUTCOME_FRESH_HOURS=48`. `load_recent_loop_outcomes` now carries `actor`/`human`/`age_hours` and drops stale verdicts and verdicts whose timestamp cannot be read. `loop_outcome_actor_is_human`.
- `services/orion-thought/app/reverie.py`: `loop_measurement_text`, world-first `why_it_matters`, verdict labels (`recorded_by`, `age_text`), `attention_summary_text`, `ChainStep` + `chain_context_for_prompt` (bounded carry), `next_focus` parsing. A standalone tick drops a stray `next_focus`.
- `services/orion-thought/app/chain.py`: calm-tick early return, feed-forward of prior thoughts + `next_focus`, `thought_complete` stop (arms the refractory cooldown and queues a compaction request), `_step` builds a `ChainStep`.
- `orion/cognition/prompts/reverie_narrate.j2`: per-loop lines, "One thing won your attention", the verdict section split into human and system parts, a TRAIN OF THOUGHT block and a `next_focus` field in chain mode only.
- `orion/schemas/reverie.py`: `thought_complete`, and the `next_focus` comment.
- `services/orion-thought/app/settings.py`, `.env_example`: `ORION_REVERIE_DRIFT_TEMP` removed.
- `services/orion-thought/README.md`: new "Reverie input and chain development" section.
- `services/orion-thought/evals/reverie_chain_development_eval.py`: new read-only eval.
- Tests: `tests/test_reverie_honest_input.py`, `tests/test_reverie_chain_feed_forward.py`, `tests/test_reverie_chain_development_eval_metrics.py` (new). Existing chain/outcome tests updated for the new behaviour. `tests/test_dim_reverie_outcomes_covers_terminal_reasons.py` now also covers the text `TerminalReason`.
- `config/field/field_channel_glossary.v1.yaml`, `config/metrics/metric_definitions.lock.json`: prompt-site declaration and re-lock.
- `services/orion-analytics/models/marts/dim_reverie_outcomes.sql`, `marts.yml`: `thought_complete`.

## Schema / bus / API changes

- Added: `TerminalReason` value `thought_complete` (`ReverieChainV1.terminal_reason`).
- Removed: none.
- Renamed: none.
- Behavior changed: `SpontaneousThoughtV1.next_focus` is now filled on chain steps. `run_reverie_once(chain_context=)` accepts a `ChainStep` (a plain `(chain_id, index)` tuple still works). The chain `StepFn` now receives keyword args `prior_thoughts`, `next_focus_hint`, `is_last`.
- Compatibility notes: the `substrate_reverie_chain.terminal_reason` column is plain text, so no migration is needed. The attention-schema surface already reads `next_focus` as `predicted_next`. Before this PR it was always empty. No bus channel changed.

## Env/config changes

- Added keys: none
- Removed keys: `ORION_REVERIE_DRIFT_TEMP` (orion-thought). It was declared and never read.
- Renamed keys: none
- `.env_example` updated: yes (key removed)
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: run from the primary checkout. Output: no orion-thought changes, only the pre-existing diverged keys in other services. The sync script does not delete keys, so `ORION_REVERIE_DRIFT_TEMP=0.7` is still in `services/orion-thought/.env` (line 247). Nothing reads it any more. It is harmless, and you can delete it by hand.
- skipped keys requiring operator action: none

## Metric gate (rendering an existing metric into a new prompt)

1. Provenance: `OpenLoopV1.magnitude` comes from `orion.substrate.prediction_error_magnitude`, which reads `substrate_node_prediction_error_history`. substrate-runtime attaches it to world-first loops.
2. Independence: this is the same value that world-first ranks on. It is not a new signal. It only shows the model the number that already picked the winner.
3. Theory anchor: the percentile and band are measured against the node's own 7-day history (spec `2026-10-02-reverie-prediction-error-magnitude-proposal.md`). `self_readings` already uses the same history.
4. Live data: the captured broadcasts show varied values (execution 1.0 top 4.3%, bus_synaptic 0.090 top 2.2%, chat 0.62 rising, cabinet 0.55 top 2.0%). A calm node reads percentile 0.0 and is not eligible to win.
5. Existing mechanism: `orion.attention.self_readings` renders the same history for its own prompt block. Here it is the loop's own copy, so no new read happens.
6. Reversibility: this is a template and render function only. The prompt-site declaration is one line per channel.

## Tests run

```text
cd services/orion-thought && pytest -q -p no:warnings tests/
1 failed, 683 passed, 32 skipped
  FAILED tests/test_settings_mind_enrichment.py::test_mind_enrichment_defaults_off  (pre-existing: fails identically on main, env-sensitive default http://orion-mind:6611 vs http://mind:6611)
  Commit A alone, checked in a temporary detached worktree: 1 failed (same), 663 passed, 32 skipped

New/changed focused tests: test_reverie_honest_input.py, test_reverie_chain_feed_forward.py,
test_reverie_chain_development_eval_metrics.py, test_reverie_chain.py, test_reverie_compaction_request.py,
test_reverie_attention_schema_publish.py, test_reverie_spontaneous_thought.py, test_reverie_narrate_prompt_outcomes.py: all pass

pytest tests/test_dim_reverie_outcomes_covers_terminal_reasons.py tests/test_check_settings_defaults.py tests/test_report_dead_env_keys.py   39 passed
pytest orion/reverie/tests orion/reverie/evals   8 failed, 49 passed (all 8 in test_proposal.py; identical 8 fail on main, pre-existing)
pytest tests/test_no_personal_terms.py tests/scripts/test_sync_local_env_from_example.py scripts/tests/test_check_env_template_parity.py tests/test_metric_prompt_semantics.py tests/test_metric_rest_range.py ...   148 passed
python scripts/check_metric_lineage.py --gate             PASS
python scripts/check_metric_lineage.py --prompt-semantics PASS
python scripts/check_definition_drift.py --gate           PASS (after --update; 7 routing_changed, committed)
python scripts/check_inner_state_registry.py              OK
python scripts/check_env_template_parity.py               PASS
```

## Evals run

```text
PYTHONPATH=.:services/orion-thought pytest -q services/orion-thought/evals/   16 passed (includes the reverie hollow-guard eval)
python services/orion-thought/evals/reverie_chain_development_eval.py --hours 24   BEFORE numbers above (read-only, live)
```

## Docker/build/smoke checks

```text
Not built or deployed (the brief says do not deploy or restart).
Live prompt render (read-only): the latest real broadcast was captured from
substrate_attention_broadcast_projection (2026-10-16 02:33:34 UTC, winner node:substrate.cabinet).
The context was built with main's code and with this branch's code, and reverie_narrate.j2 was rendered.
```

Before (main), the coalition section:

```text
You are noticing your own current state: the coalition of concerns that just won
the substrate workspace. ...
SOURCE (bounded internal input — the current winning coalition)
- coalition_projection: {... 'attended_node_ids': ['node:substrate.cabinet'], 'selected_open_loop_id': 'open-loop-7597ff641a23', ...}
- open_loops: [{'id': 'open-loop-7597ff641a23', 'description': 'Cabinet prediction error', 'why_it_matters': 'novel or unresolved current-turn target with substrate pressure', 'target_type': 'anomaly'}]
```

After (this branch). The second part is rendered in chain mode with a sample step-1 thought and `next_focus`, to show the chain block:

```text
You are noticing your own current state. One thing won your attention this tick: Cabinet prediction error [open-loop-7597ff641a23] (from node:substrate.cabinet).
Say what it is, plainly. ...
- open_loops (what competed this tick, with its real measurement where one exists):

  - open-loop-7597ff641a23 (won): Cabinet prediction error
    why it surfaced: body unusual for itself (high)
    measured: reading 0.55 (usual 0.00, 90th pct 0.00; top 2.0% of its own week, band high)

TRAIN OF THOUGHT (this is step 2 of one continuing thought)
Earlier in this train of thought you said:
- "Bus synaptic activity is spiking well above its own usual week, and I do not yet know what traffic is driving it."
Take it one step further — do not restate it. Narrow it, test it, connect it
to something new above, or change your mind about it.
Last step you said you wanted to look next at: which services published the most in the last few minutes
...
- next_focus: what to look at next to carry this thought forward (one short
  phrase), or null if this thought is complete and there is nothing more to
  look at. null ends the train of thought.
```

## Review findings fixed

A code review ran in a subagent on `origin/main...HEAD`. It found no blockers.

- Finding: on a calm tick the chain now returns before `run_reverie_once`. Expectation scoring (`_maybe_score_pending_expectation`) runs only at the top of that function, and the live chain worker is its only caller, so scoring would stall through calm stretches.
  - Fix: the calm-tick branch of `run_reverie_chain` now awaits the scorer, wrapped so it never raises.
  - Evidence: `test_calm_tick_still_runs_expectation_scoring`.
- Finding: the dbt `accepted_values` on `sources.yml` (`substrate_reverie_chain.terminal_reason`) and `staging.yml` (`stg_reverie_chains.outcome_key`) lacked `thought_complete`. `dbt test` would fail on the first such row.
  - Fix: added the value to both. New test `test_every_text_chain_accepted_values_list_covers_terminal_reason` checks the source, staging, and fact lists against `TerminalReason`.
  - Evidence: `tests/test_dim_reverie_outcomes_covers_terminal_reasons.py` 3 passed.
- Finding: the prompt line "usually one signal wins, not a group of concerns" contradicted the summary when several nodes won.
  - Fix: removed the line. The summary sentence alone says what won.
- Finding: the store comment pointed at a test file that does not exist, and did not say that the substrate TTL has an env override.
  - Fix: the comment now names the real pin and states the override gap.
- Finding: no test covered the concern-card (semantic-lift) branch in chain mode.
  - Fix: `test_concern_card_branch_renders_chain_block_and_asks_next_focus`.
- Finding: the eval's `next_focus` fill rate divided by all thoughts, including last steps that are never asked.
  - Fix: it now divides by steps that are not their chain's last stored step. It also reports `chains_last_step_without_next_focus`.
- Finding: the `orion/reverie/attention_schema.py` docstring about `next_focus` was stale.
  - Fix: docstring updated (`predicted_next` now usually falls back to the expectation, because the last step's `next_focus` is usually None).
- Not changed (nit): an actor that is neither system nor Juniper would render as "recorded by <actor>" and count as human. Today only `juniper`, `orion` (`acted`, filtered out) and `system:implicit_decay` write verdicts.

## Restart required

```bash
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-thought up -d --build
```

Run this from the clean primary `main` checkout after merge. Only orion-thought needs the rebuild. orion-thought reads the template text itself (`orion/cognition/plan_loader.py::load_prompt_template`, called by `build_plan_for_verb`) and sends it inside the plan. orion-cortex-exec renders the string it receives (`executor.py::_render_prompt`).

## Risks / concerns

- Severity: medium. Concern: the model may answer `next_focus: null` on step 0 most of the time. Chains would then shrink to 1 step, which is fewer thoughts than today. Mitigation: that is the intended "thought complete" signal. The eval's steps/chain distribution and `thought_complete` count will show it. If it is too eager, the prompt wording is a one-line change.
- Severity: low. Concern: the 48 h verdict window is a wall-clock proxy, because `OpenLoopV1` has no per-loop activation timestamp. A system decay verdict written 2 h before a re-spike is still attached. It is labelled as system bookkeeping, and the prompt says the loop is live again. Mitigation: if loops gain an activation timestamp, switch `fresh_loop_outcomes` to "newer than activation".
- Severity: low. Concern: the AFTER behaviour is UNVERIFIED live: no LLM output under the new prompt has been observed. Mitigation: re-run the eval 24 h after deploy.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2708

🤖 Generated with [Claude Code](https://claude.com/claude-code)
