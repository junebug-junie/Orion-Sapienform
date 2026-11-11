## Summary

- Orion can now read the exact steps of its earlier turns in a chat: `python -P -m orion.dig turn-steps <chat> [<n>]` (Patch 1 of spec #3139).
- Each Hub chat reply gets a notes file. Orion writes one line per step while it works. At turn end the harness keeps only lines in the fixed step shape and carries them, quoted, to the next turn of the same chat (Patch 2).
- A turn that read outside content has its notes held back. The next turn sees only that they were held back.
- New cockpit stage `working_notes` shows what each turn was carried in and what it wrote.
- Baseline eval for the before/after count: `orion/harness/evals/turn_repeat_baseline.py`.
- Includes the spec (#3139), now marked implemented with the deviations listed.

## Outcome moved

The failure: turn 2 relearns what turn 1 found out, for example the Postgres container name (2026-11-05). Baseline over the last 14 days of live data:
- 50 tool-using chat turns, 44 back-to-back pairs.
- 6 pairs redo a step from the turn before: 29 repeated steps, 0 of them repeats of a failure.
- The taint rule holds back 16 of 50 turns (32%).

Re-run after 7 days live: `python -m orion.harness.evals.turn_repeat_baseline --days 7`.

## Current architecture

Each chat message spawns a new CLI session. RECENT CONVERSATION carries replies only. Every step was already in `cockpit_turn_sighting` (`motor_hop`), but Orion could not read it. The only per-chat carry was `last_tool_fetch_cache` (Redis, tool names only).

## Architecture touched

- `orion/harness/working_notes.py` (new): notes path, shape check, taint check, Redis carry, prompt block.
- `orion/harness/runner.py`: reads the carry, creates the notes file, watches steps for outside content, and settles in a `finally` (success, error event, exception, cancel).
- `orion/harness/prefix.py`: renders the block after RECENT CONVERSATION.
- Hub: marker step `cockpit.working_notes.v1` -> stage `working_notes`. Not sent to the chat websocket. Labeled "working notes" in runtime activity.
- `orion/dig/turn_steps.py` + `orion/dig/__main__.py`: read-only query through `ReadOnlyPsql`. Tool inputs and results are trimmed in SQL.

## Files changed

- `orion/harness/working_notes.py`, `orion/harness/runner.py`, `orion/harness/prefix.py`: Patch 2.
- `orion/dig/turn_steps.py`, `orion/dig/__main__.py`: Patch 1.
- `orion/harness/evals/turn_repeat_baseline.py`: acceptance checks 1, 2 and 7.
- `orion/cockpit/markers.py`, `orion/cockpit/builders.py`, `orion/schemas/cockpit_sighting.py`, `orion/hub/cockpit_emit.py`, `orion/hub/turn_orchestrator.py`, `orion/hub/runtime_activity.py`: cockpit stage.
- `services/orion-harness-governor/{.env_example,docker-compose.yml,README.md}`: flag, directory, docs.
- Tests: `orion/harness/tests/test_working_notes.py`, `orion/dig/tests/test_turn_steps.py`, `orion/hub/tests/test_cockpit_working_notes_hop.py`. CI: `dig-kit-tests.yml` runs them.
- `docs/superpowers/specs/2026-11-05-turn-working-notes-design.md`: status, deviations, baseline.

## Schema / bus / API changes

- Added: `CockpitStageV1` value `working_notes`. Redis key `orion:harness:working_notes:<session>` (TTL 24 h). Marker `cockpit.working_notes.v1` on the existing harness step channel.
- Removed / renamed: none.
- Compatibility: sql-writer stores the stage as a string. No DB constraint or JS list holds the stage set (checked in review). Deploy the Hub before the governor. If the governor goes first, an old Hub records the marker as a plain `motor_hop`, which does no harm.

## Env/config changes

- Added keys: `HARNESS_FCC_WORKING_NOTES_ENABLED=true`, `HARNESS_FCC_WORKING_NOTES_DIR=/mnt/orion-fcc/notes` (governor).
- `.env_example` updated: yes. Local `.env`: both keys added to `services/orion-harness-governor/.env` in the primary checkout. The worktree has no `.env`, so the sync script had nothing to write there.
- Skipped keys: none.

## Tests run

```text
pytest orion/harness/tests orion/dig/tests orion/hub/tests orion/cockpit   946 passed (before final review fixes); after fixes: 41 + 175 targeted passed
pytest (orion-hub cockpit/runtime_activity tests)                          187 passed
pytest services/orion-harness-governor/tests                               100 passed, 1 failed (test_fcc_warm_pool_wiring: host permission on /root/.fcc/.env; fails the same way on main)
check_env_template_parity                                                  PASS
tests/test_no_personal_terms.py, test_compose_logging_journald.py, test_no_private_eval_artifacts.py  12 passed
```

## Evals run

```text
python -m orion.harness.evals.turn_repeat_baseline --days 14   (live, read-only)
tool_using_turns 50, consecutive_pairs 44, pairs_with_a_repeat 6, repeated_failed 0, repeated_ok 29, tainted_share 0.32
python -P -m orion.dig turn-steps <live chat> 5                 (live) printed real steps from 2026-11-08
```

## Docker/build/smoke checks

```text
Not run. The live path is UNVERIFIED until deploy: no working_notes row, no notes file written by a real CLI turn.
```

## Review findings fixed

- Finding: `dig turn-steps` replayed fetched web/GitHub text, so a clean turn could copy it into its notes.
  - Fix: results of outside-content calls show as "(outside content; not shown here)".
  - Evidence: `test_outside_results_are_never_replayed`.
- Finding: the denylist missed live tools (`mcp__github__get_file_contents` on a PR ref, `recommend_reading`), `/usr/bin/curl`, `gh pr diff`, and `git fetch`.
  - Fix: every MCP tool is outside content except `orion-introspect` and `gitnexus`. The Bash check is wider.
  - Evidence: parametrized taint tests. Live taint share went from 18% to 32%.
- Finding: a resumed CLI session hides earlier outside reads.
  - Fix: `resume_key` marks the turn tainted (`resume`).
  - Evidence: `test_resumed_turn_is_held_back`.
- Finding: settle skipped when the motor raises or is cancelled.
  - Fix: settle runs in `finally` under `asyncio.shield`.
  - Evidence: `test_motor_exception_still_harvests_notes`.
- Finding: carry race between overlapping turns; carry unbounded in Redis.
  - Fix: re-read the carry before folding; store at most the 3,000-char budget.
  - Evidence: `test_fold_caps_stored_lines_to_the_prompt_budget`.
- Finding: the marker step counted as a motor step in runtime activity.
  - Fix: `step_index=-1` and its own label.
- Not changed: with the flag off, the `dig turn-steps` line stays in chat prompts. The spec makes Patch 1 unflagged.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: medium. Concern: Orion may not write notes, or may write them in the wrong shape. Mitigation: empty and dropped counts sit on every `working_notes` row. Acceptance 3 (95% coverage) is measured after deploy.
- Severity: medium. Concern: the Bash taint check is a floor (`python -c urllib` passes). Mitigation: the quarantined-reader proposal in the spec is the real boundary.
- Severity: low. Concern: two turns in one chat at the same moment can still race for a few milliseconds. Mitigation: re-read before write.
- UNVERIFIED: the CLI can write `/mnt/orion-fcc/notes/...` (skip-permissions is on, and the directory is under the governor's `/mnt/orion-fcc` mount).

## PR link

TBD
