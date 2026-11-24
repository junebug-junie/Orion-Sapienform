## Summary

- The curiosity helper now calls Claude when the Claude Code transcript meter reads `unknown` (nobody used Claude Code in 5 h, so the account is idle). Before, it refused.
- A transcript-observed `limited` still refuses (`claude_budget_limited`).
- New back-off: when Claude itself answers "You've hit your session/weekly/usage limit · resets <time>", every later hire is refused (`refused_budget`, code `claude_limit_backoff`) with no Claude call, until the reset Claude named. The refusal sentence names the reset time.
- No reset time in the text (or an unknown zone such as "(PST)"): wait `CURIOSITY_PEER_CLAUDE_LIMIT_BACKOFF_DEFAULT_SEC` (default 3600). Any later Claude outcome (ok, empty, or a non-limit failure) clears the hold.
- The hold lives in process memory. On boot it is seeded from the newest `claude_room` PeerBrief node in the worldview graph from the last 5 h (the nodes this service already writes), bounded by a 10 s timeout.
- `observe()` and `ask_claude_trigger._budget_refusal` are unchanged for their other callers.

## Outcome moved

The meter was wrong both ways (#3655):
- 2026-11-19..23: 13 hires refused `claude_budget_unobserved` while the account was idle. These now reach Claude.
- 2026-11-17 20:28Z and 2026-11-24 09:12Z: the meter read `clear` and Claude said "session limit · resets ...". The 11-24 11:23Z hire repeated the same failure. That second hire is now refused naming "2026-11-24 12:10Z", with no Claude call.

## Current architecture

`services/orion-curiosity-peer/app/worker.py::handle_help_request` called `observe(root=CURIOSITY_PEER_CLAUDE_PROJECTS_PATH)` and refused on anything `_budget_refusal` rejected, including `budget_unobserved`. Claude's own limit refusal became a `failed` brief and was never an input to the next gate.

## Architecture touched

Only `orion-curiosity-peer`. No bus, schema, or API change. New reason code `claude_limit_backoff` inside the existing `refused_budget` status.

## Files changed

- `services/orion-curiosity-peer/app/claude_limit_backoff.py`: new. Limit-text detection (reuses the meter's session/weekly phrases), reset parsing (reuses `rate_limit_events.parse_reset_at`, adds IANA zone support and a guard against a just-passed reset rolling a day), in-process hold, boot seed cypher.
- `services/orion-curiosity-peer/app/worker.py`: back-off check first, `unknown` meter tries, outcome recording after each real Claude call, `seed_claude_limit_memory`.
- `services/orion-curiosity-peer/app/main.py`: boot seed with a 10 s timeout.
- `services/orion-curiosity-peer/app/settings.py`, `.env_example`, `docker-compose.yml`: `CURIOSITY_PEER_CLAUDE_LIMIT_BACKOFF_DEFAULT_SEC`.
- `services/orion-curiosity-peer/README.md`: gate documented.
- `services/orion-curiosity-peer/tests/test_claude_limit_backoff.py`: new regression tests.
- `services/orion-curiosity-peer/tests/{conftest,test_acceptance,test_claude_peer,test_refusal_reason_chain}.py`: tests that expected an idle meter to refuse now expect a call; refusal-text tests use the `limited` meter.
- `services/orion-curiosity-peer/evals/run_contractor_peer_eval.py`: `unobserved_tries`, `limit_backoff`, `after_reset` cases.

## Schema / bus / API changes

- Added: none (reason code `claude_limit_backoff` in `refusal_reason` text only)
- Removed: none
- Renamed: none
- Behavior changed: idle meter now hires; Claude's limit refusal holds later hires.
- Compatibility notes: none.

## Env/config changes

- Added keys: `CURIOSITY_PEER_CLAUDE_LIMIT_BACKOFF_DEFAULT_SEC=3600`
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`orion-curiosity-peer: +CURIOSITY_PEER_CLAUDE_LIMIT_BACKOFF_DEFAULT_SEC='3600'`)
- skipped keys requiring operator action: none

## Tests run

```text
pytest services/orion-curiosity-peer/tests -q                      81 passed
pytest tests/test_curiosity_peer_*.py services/orion-hub/tests/test_turn_orchestrator_role_teach_disclosure.py tests/test_curiosity_supervisor.py   74 passed
pytest tests/test_role_teach_disclosure.py tests/test_curiosity_cursor_policy.py orion/autonomy/tests/test_agency_episode.py   41 passed
scripts/check_integration_seam.py                                  OK
scripts/check_env_template_parity.py                               PASS
scripts/check_service_env_compose_parity.py orion-curiosity-peer   OK (20 keys)
```

## Evals run

```text
evals/run_contractor_peer_eval.py   6/6 passed (unobserved_tries, limited_refuse, hire_claude, claude_failed, limit_backoff, after_reset)
evals/run_agency_episode_eval.py    8 passed
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-curiosity-peer build   -> Image Built
docker run --rm --entrypoint python <image> -c "parse_limit_reset(... 11pm (America/Denver) ...)"  -> 2026-11-25 06:00:00+00:00 (tzdata present)
Live graph read (read-only, GRAPH.RO_QUERY) of the seed shape returned the 11-24 failed rows with written_at ms epoch.
```

## Review findings fixed

- Finding: a dateless reset read just after it passed ("resets 12:10pm" at 12:10:20) rolled to tomorrow and held ~24 h.
  - Fix: a dateless reset more than 5 h out falls back to the bounded default.
  - Evidence: `test_reset_read_just_after_it_passed_does_not_hold_a_day`.
- Finding: boot seed skipped `empty` outcomes, so a restart could resurrect a cleared hold.
  - Fix: seed cypher includes `empty`.
  - Evidence: `test_boot_seed_empty_outcome_clears`.
- Finding: unknown zone name ("(PST)") silently read as UTC.
  - Fix: return None, bounded default applies.
  - Evidence: parametrized case in `test_parse_limit_reset`.
- Finding: limit phrases duplicated the meter's constants.
  - Fix: import `_SESSION_LIMIT` / `_WEEKLY_LIMIT`.
  - Evidence: code.
- Finding: 200-char clip on the failure reason could cut the reset text.
  - Fix: hold reads the unclipped exception text.
  - Evidence: `test_long_error_naming_reset_late_still_reads_the_reset`.
- Finding: boot seed had no timeout.
  - Fix: `asyncio.wait_for(..., 10 s)`.
  - Evidence: code.
- Finding: tests never used the module-level memory.
  - Fix: `test_default_module_memory_is_the_production_wiring`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-curiosity-peer up -d --build   # from clean primary main after merge
```

## Risks / concerns

- Severity: medium. Concern: an idle meter now spends Claude quota shared with Juniper. Mitigation: Juniper approved "try, back off on refusal" 2026-11-24; the first limit refusal stops all hires until the stated reset.
- Severity: low. Concern: a dateless reset with no zone is read as UTC (matches every live row so far, all "(UTC)"). Mitigation: zone parsing when named.
- Live check: UNVERIFIED. After deploy: a hire during an idle stretch returns `ok`; a limit failure is followed by `refused_budget` naming the reset (`curiosity_peer_brief.refusal_reason` ends `[claude_limit_backoff]`).
- Issues: Refs #3655 (left open until the live check passes).

## PR link

(filled after creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
