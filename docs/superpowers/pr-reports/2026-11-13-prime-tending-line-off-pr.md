# Ship the Prime tending line off until labels read cleanly

## Summary

- `PRIME_PERSONALITY_LINE_ENABLED` now defaults to `false` in settings, `.env_example` and compose for orion-hub and orion-mind. Each place has a comment that says why.
- #3330 merged with the line ON before this change reached it. The live choice-built labels still read as noise ("turn start context is stale snapshot over camera layer mediation not closeness"). A label fix is in progress.
- All tending-line code and tests stay. The tests turn the flag on explicitly.
- The Prime choice producer is not behind this flag. It stays on.

## Outcome moved

Orion does not see unreadable tendency labels in the Prime block or in Prime's frame. Choice evidence keeps accumulating.

## Current architecture

The tending line exists from #3330 and is on by default.

## Architecture touched

Only config defaults and docs. The Hub's missing-attribute fallback in `orion/hub/turn_orchestrator.py` is now `False`.

## Files changed

- `services/orion-{hub,mind}/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: the default is now false, with the reason.
- `orion/hub/turn_orchestrator.py`: the getattr fallback is now False.
- `services/orion-hub/tests/test_turn_orchestrator_cockpit_hops.py`: tests set the flag on explicitly and assert that the default is disabled.
- `docs/superpowers/pr-reports/2026-11-13-personality-prime-choice-pr.md`: the #3330 report now notes the flag ships off.

## Schema / bus / API changes

- None.

## Env/config changes

- Changed default: `PRIME_PERSONALITY_LINE_ENABLED=false` (orion-hub, orion-mind).
- The local `.env` was set to `false` by hand in both services, because the sync script keeps existing values.

## Tests run

```text
orion/prime/tests -> 186 passed
services/orion-mind tests -> 365 passed
services/orion-hub tests/test_turn_orchestrator_cockpit_hops.py -> 21 passed
```

## Evals run

```text
None needed. The block eval passes tending explicitly and does not read the flag.
```

## Docker/build/smoke checks

```text
Not run (no deploy from a worktree).
```

## Review findings fixed

- None. This is a config-only change. The code was reviewed in #3330.

## Restart required

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: if #3330 was deployed before this change, the line is live until hub and orion-mind are rebuilt. Mitigation: rebuild both, or set the flag to false in their `.env`. That is already done locally.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
