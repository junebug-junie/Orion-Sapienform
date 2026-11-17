## Summary

Orion's curiosity "hire a helper" channel now hires Claude directly. Cursor is gone from it (Juniper chose "Claude-only helper" on 2026-11-17, #3397).

- The peer worker no longer runs the Cursor Agent CLI. A hire is: Claude budget gate, then episode commitment, then exactly one Claude request through orion-room-companion, then persist.
- The real reason Claude never ran: the container could not see Claude's usage meter, so every hire was refused `claude_budget_unobserved`. Compose now mounts `~/.claude/projects` read-only at the same path, and the meter reads `clear` (checked in the built image).
- Removed: Cursor invoker, Cursor error classifier, argv re-export, Cursor model/binary/budget/repo env keys, Cursor host mounts and the repo mount.
- The hire text Orion reads says Claude, and the role choice is `hire_peer` (matches the `chosen_action: "hire_peer"` the schema already required).
- Room companion no longer offers Claude a "[pass]" option on a contractor hire, so a hire gets an answer, not an empty brief.
- The Claude wait went from 120 s to 420 s, because a room-companion turn with tools may take up to 360 s.

## Outcome moved

Live, 2026-11-05..16: 0 ok briefs. `cursor_auto` failed every day; since 11-14 `claude_room` rows were all `refused_budget` with "Cursor unavailable: it is not logged in ... Claude fallback refused: this service cannot see Claude's usage meter". After deploy, a hire reaches Claude and nothing depends on Cursor.

## Current architecture

`services/orion-curiosity-peer` consumed `HelpRequestV1`, gated on a Cursor budget meter, ran the Cursor CLI, and asked Claude only after a Cursor token failure. That Claude path was gated on `rate_limit_events.observe()`, which read `/root/.claude/projects` inside the container (not mounted), so it always refused.

## Architecture touched

- orion-curiosity-peer worker, settings, compose, Dockerfile comments, README, tests, evals.
- orion-room-companion `append_system_prompt_for` (contractor hires skip the auto-invite clause).
- Prompt text: `orion/curiosity/kickoff_prompt.py`, `role_teach_disclosure.py`, `queue_contention_disclosure.py`, `services/orion-mind/app/stance_handoff.py`.
- Out of scope, unchanged: the grading-side Cursor reviewer (`orion/curiosity/supervisor.py`, `hire_cursor_review`, `cursor_policy.py`, `cursor_limit_events.py`); filed #3601.

## Files changed

- `services/orion-curiosity-peer/app/worker.py`: Claude-only order; meter reads `CURIOSITY_PEER_CLAUDE_PROJECTS_PATH`.
- `services/orion-curiosity-peer/app/brief_prompt.py` (was `cursor_invoker.py`): sealed prompt and parse only.
- `services/orion-curiosity-peer/app/claude_peer.py` (was `claude_fallback.py`): refusal text, 420 s wait.
- `services/orion-curiosity-peer/app/cursor_errors.py`, `app/policy.py`: deleted.
- `services/orion-curiosity-peer/app/settings.py`, `.env_example`, `docker-compose.yml`, `Dockerfile`, `README.md`, `app/main.py`: Cursor keys out; meter path in; mount.
- `services/orion-curiosity-peer/tests/*`, `evals/*`: rewritten for Claude-only; regression tests.
- `tests/test_curiosity_cursor_policy.py`: argv-policy tests moved here (supervisor still uses that policy).
- `services/orion-room-companion/app/room_prompt.py` + test: no "[pass]" on contractor hires.
- `orion/curiosity/*.py`, `services/orion-mind/app/stance_handoff.py`, related tests: hire text.
- `orion/schemas/curiosity_peer.py`: comment only; `cursor_auto` stays valid for old rows.
- `.github/workflows/agency-ask-episodes.yml`: runs the moved policy test and the room-companion prompt test.

## Schema / bus / API changes

- Added: none.
- Removed: none (`cursor_auto` is still accepted in `CuriosityPeerNameV1`; it is no longer produced).
- Renamed: none in schemas. Prompt vocabulary `hire_cursor` -> `hire_peer` (free text; nothing branches on it; Hub atlas renders any non-`local_crawl` value).
- Behavior changed: `refusal_reason` format is now `<plain sentence> [claude_*]` (no Cursor link).
- Compatibility notes: old rows render unchanged.

## Env/config changes

- Added keys: `CURIOSITY_PEER_CLAUDE_PROJECTS_PATH` (orion-curiosity-peer). Compose requires it (`:?`), so a missing value fails loudly.
- Removed keys: `CURIOSITY_PEER_AGENT_BIN`, `CURIOSITY_PEER_MODEL`, `CURIOSITY_PEER_REPO_ROOT`, `CURIOSITY_PEER_CURSOR_BUDGET_STATE`, `CURIOSITY_PEER_CURSOR_BUDGET_FILE`.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, the new key was added to the primary checkout's `services/orion-curiosity-peer/.env`.
- skipped keys requiring operator action: the sync script does not delete keys. The removed Cursor keys are still in the primary checkout's local `.env`. They are inert (compose no longer passes them), but `CURIOSITY_PEER_CURSOR_BUDGET_STATE=clear` is misleading. Delete them by hand.

## Tests run

```text
pytest services/orion-curiosity-peer/tests                 57 passed
pytest services/orion-room-companion/tests                 100 passed
pytest tests/test_curiosity_cursor_policy.py tests/test_curiosity_peer_*.py tests/test_role_teach_disclosure.py orion/curiosity/tests tests/test_curiosity_supervisor.py   376 passed, 29 skipped
pytest services/orion-mind/tests/test_prime_self_monitor_gate.py   26 passed (one failure seen only in a combined multi-service run; passes alone; not touched here)
scripts/check_integration_seam.py                          OK
scripts/check_env_template_parity.py                       PASS
scripts/check_service_env_compose_parity.py orion-curiosity-peer   OK
scripts/check_env_key_single_source.py                     OK
scripts/check_compose_no_relative_mounts.py                PASS
git diff --check                                           clean
```

## Evals run

```text
evals/run_contractor_peer_eval.py   4/4 passed (unobserved refuse, limited refuse, hire Claude once, Claude failed)
evals/run_agency_episode_eval.py    passed: 8 (disposable FalkorDB)
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-curiosity-peer build     Built
scripts/safe_docker_build.sh orion-room-companion build     Built
scripts/safe_docker_build.sh orion-curiosity-peer config    bind /home/athena/.claude/projects -> same path, read_only
docker run --rm --network none (built image, mount as compose) :
  root /home/athena/.claude/projects files 43 msgs 8431 state clear refusal None
  without mount (old behaviour) -> observed False, budget_unobserved
```

## Review findings fixed

- Finding: moved cursor-policy test no longer ran in CI.
  - Fix: added to `agency-ask-episodes.yml` run list and paths.
  - Evidence: workflow diff.
- Finding: nothing guarded the mount whose absence caused the outage.
  - Fix: `test_compose_mounts_the_claude_meter_read_only_at_the_same_path`.
  - Evidence: peer tests 57 passed.
- Finding: `${HOME}` fallback could silently mount an empty dir under sudo/cron.
  - Fix: `:?` required-variable form on the mount and env line.
  - Evidence: `config` output above.
- Finding: contractor hires carried the auto-invite "[pass]" option, so a hire could come back empty.
  - Fix: contractor marker replaces the auto-invite clause in room-companion.
  - Evidence: `test_contractor_marker_adds_peer_clause` updated; room-companion tests 100 passed.
- Finding: stale `--tools ""` comment in `_map_claude_result`.
  - Fix: reworded.
- Finding: stale Cursor keys in local `.env`.
  - Fix: removed from the worktree copy; primary checkout listed under operator action above.

## Restart required

Deploy order: room-companion first (so the first hire gets the no-pass prompt), then the peer. From the clean primary main checkout after merge:

```bash
scripts/safe_docker_build.sh orion-room-companion up -d --build
scripts/safe_docker_build.sh orion-curiosity-peer up -d --build
```

orion-mind and Hub pick up the prompt text on their next normal rebuild (text only).

## Risks / concerns

- Severity: medium. Concern: live proof is UNVERIFIED. No real hire has returned `status=ok` yet. Mitigation: after deploy, check `select peer,status,created_at from curiosity_peer_brief order by created_at desc limit 5`.
- Severity: low. Concern: the meter reads `clear` only when a Claude Code session wrote in the last 5 hours; after a long idle stretch a hire is refused `claude_budget_unobserved` (fail-closed, by design). Mitigation: reason text says so plainly.
- Severity: low. Concern: the peer container now reads all of `~/.claude/projects` read-only (same pattern as orion-cocreation-signals, approved 2026-08-11). Only an in-memory limit state is computed.
- Issues opened: #3601 (grading side still offers a Cursor reviewer). Issues referenced: #3397 (stays open until the live check passes).

## PR link

See the PR this report is attached to.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
