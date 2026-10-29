## Summary

- The post-pull rebuild now rebuilds orion-harness-governor when `orion/dig/`, `orion/body_atlas/`,
  or `orion/self_inventory.py` change.

## Outcome moved

- **The bug:** PR #2951 (organ names) merged, and the post-pull rebuild rebuilt 20 services, but
  not the harness. Orion's kit stayed on old code.
- **Why:** the rebuild picks services by Python imports, and Orion runs these packages as CLIs
  through Bash (`python -P -m orion.dig`), so no service imports them. A diff touching only
  them mapped to no service at all.
- **Why #2920 rebuilt the harness anyway:** it also touched `orion/dig/brief.py`, which the harness
  does import.

## Current architecture

`scripts/rebuild_affected_services.py` maps changed paths to services through the import graph,
plus overrides in `scripts/service_rebuild_paths.yaml`. `orion_package_extra_services` was empty.

## Architecture touched

`scripts/service_rebuild_paths.yaml`: `orion_package_extra_services` gains 3 entries. The
classifier code is unchanged.

## Files changed

- `scripts/service_rebuild_paths.yaml`: the mapping.
- `tests/scripts/test_rebuild_affected_services.py`: one parametrized real-repo test.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
pytest tests/scripts/test_rebuild_affected_services.py -> 91 passed
same new test with the mapping reverted                -> 3 failed (it catches the bug)
rebuild_affected_services.py --list-only --paths orion/dig/__main__.py orion/body_atlas/names.py
  before: (nothing)   after: orion-harness-governor
```

## Evals run

Not applicable (deploy routing config).

## Docker/build/smoke checks

None. Takes effect on the next post-pull rebuild.

## Review findings fixed

The change is 3 YAML lines in an existing, documented override slot, and the test above shows it
works. I judged a separate review subagent not useful for it.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low.
  - Concern: the harness rebuilds a bit more often, about 14 minutes when it does.
  - Mitigation: only when these three paths change.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
