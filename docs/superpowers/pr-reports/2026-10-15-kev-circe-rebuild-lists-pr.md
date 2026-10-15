## Summary

- The pull hook on circe never built Kev after #2695. circe's rebuild allowlist did not list `orion-kev`.
- athena had the opposite bug. A future pull that touches `services/orion-kev/` would rebuild and start the old `orion-athena-kev` from athena's leftover `.env`.
- This PR adds `orion-kev` to `include_services_circe.txt` and to `exclude_services.txt`. Only athena reads the exclude file, and so does athena's `up_all_services.sh`.
- A contract test pins both entries.

## Outcome moved

The rebuild resolver's output for a change to `services/orion-kev/entrypoint.sh`:

| Host | Before | After |
|---|---|---|
| circe | 0 services | `orion-kev` |
| athena | `orion-kev` | 0 services |

## Current architecture

- `scripts/rebuild_affected_services.py` (`:864-881`) works like this:
  - A host with `mesh-utilities/common/include_services_<host>.txt` rebuilds only the services listed in that file.
  - Every other host rebuilds everything except the services in `exclude_services.txt`.

## Architecture touched

- Service lists only. No code changes.

## Files changed

- `mesh-utilities/common/include_services_circe.txt`: adds `orion-kev`.
- `mesh-utilities/common/exclude_services.txt`: adds `orion-kev`, with the reason.
- `services/orion-kev/tests/test_orion_kev_contract.py`: `test_rebuild_lists_place_kev_on_circe_only`.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
services/orion-kev/tests + tests/scripts/test_rebuild_*.py + bonsai contract: 66 passed
rebuild_affected_services.py --host circe|athena --paths services/orion-kev/entrypoint.sh
  before: circe 0, athena 1
  after:  circe 1, athena 0
```

## Evals run

```text
Not applicable. This is a deploy-list change.
```

## Docker/build/smoke checks

```text
Kev was deployed on circe by hand on 2026-10-15: orion-circe-kev on GPU 3, /docs returns 200 from athena.
The substrate logged substrate_system_one_appraisal_completed (latency 580ms) after the deploy.
```

## Review findings fixed

See the PR comments.

## Restart required

```text
No restart required.
To stop the old Kev on athena, run this from athena's primary checkout:
scripts/safe_docker_build.sh orion-kev down
```

## Risks / concerns

- Severity: low
  - Concern: none beyond the old athena container still running.
  - Mitigation: the `down` command above.

## PR link

<link>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
