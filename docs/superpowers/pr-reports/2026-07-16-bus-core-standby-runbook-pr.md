# PR report — bus-core cold-standby target + cutover runbook

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1084
Branch: `chore/bus-core-standby-runbook`
Status: **DONE_WITH_CONCERNS**

## Summary

- `bus-core` (`services/orion-bus/docker-compose.yml`) is a single
  `redis:7-alpine` instance on `athena` and a real single point of failure.
  Added a passive, cold-standby deploy target,
  `services/orion-bus/docker-compose.atlas-standby.yml`, scoped to the
  `atlas` mesh node.
- Node choice: `atlas`, not `circe`. `config/biometrics/node_catalog.yaml`
  marks `circe` `expected_online: false` (burst/training box, not reliably
  up); `atlas` is `expected_online: true` and was confirmed live-reachable
  over Tailscale from this session (`ping`/`tailscale status`).
- Deploy pattern matches the repo's existing precedent for node-specific
  service definitions — a separate, node-scoped compose file
  (`services/orion-llamacpp-host/docker-compose.atlas-workers.yml`) run
  from the target node's own checkout — rather than a profile inside the
  primary `docker-compose.yml`, since the standby runs on different
  physical hardware.
- Wrote the cutover/cutback runbook,
  `docs/notes/2026-07-16-bus-core-standby-cutover-runbook.md`: real,
  reproduced AOF-restore verification steps; a grep-verified, mesh-wide
  audit of which services' `ORION_BUS_URL` wiring actually follows a root
  `.env` change (51 do via compose-level pass-through, 3 don't because the
  IP is hardcoded directly in their own `docker-compose.yml`, 20 have no
  visible reference at all and need individual verification); and a
  cutback procedure that explicitly accounts for cold-standby (non-
  replicating) data-loss risk.
- Real, live-verified AOF restore test on `athena`: snapshotted the live
  `bus-core` data directory read-only, restored it into a throwaway
  container and separately into the actual `docker-compose.atlas-standby.yml`
  compose target (with a synthetic node identity so it never touched
  `/mnt/telemetry/orion-athena/bus/`), and confirmed `DBSIZE`/`KEYS *`
  matched the live primary exactly (20/20, real Orion mesh keys). Live
  primary confirmed untouched and healthy throughout and afterward.
- **Could not test anything atlas-specific** — this session runs on
  `athena` and has no SSH access to `atlas`
  (`ssh atlas.tail348bbe.ts.net` → `Permission denied
  (publickey,password)`, confirmed live). This is stated plainly in the
  runbook's own "what was tested" section, not glossed over.
- Code review (high effort, orion-repo-agent subagent) caught a real,
  reproduced bug in the first draft: the documented deploy command only
  chained one `--env-file`, so `PROJECT`/`TELEMETRY_ROOT` interpolated to
  empty strings and the data volume silently bound at the filesystem root
  instead of under `/mnt/telemetry/`. Fixed in a follow-up commit; re-
  verified with `docker compose config`.

## Outcome moved

`bus-core`'s single-point-of-failure risk now has a real, documented,
partially-tested recovery path — a place to restore data to and a runbook
for the manual cutover — instead of nothing. Does not itself reduce the
probability of `bus-core` going down (that's the separate AOF-auto-repair
effort); reduces the blast radius/recovery time when it does.

## Current architecture

Before this patch: `bus-core` was a single Redis instance on `athena`
with no standby, no documented recovery path beyond restarting the
container and hoping the AOF wasn't corrupted. `ORION_BUS_URL` is a root
`.env` key most (but, per the mesh audit in the runbook, not all)
services pick up via a compose-level `${ORION_BUS_URL}` pass-through.

## Architecture touched

`services/orion-bus/` only (compose, `.env_example`, `README.md`) plus a
new docs/notes runbook. No schema, bus channel, or application-code
changes. No changes to the parallel `fix/bus-core-aof-auto-repair` effort
or its data-repair logic — verified by re-reading the diff, this branch
touches none of it.

## Files changed

- `services/orion-bus/docker-compose.atlas-standby.yml` (new): passive
  `bus-core-standby` Redis service. Same redis flags as the primary
  (`--appendonly yes`, `--maxmemory-policy allkeys-lru`,
  `--client-output-buffer-limit pubsub 64mb 32mb 120`) so a future cutover
  doesn't also change broker behavior. Network alias `bus-core-standby`
  (never `bus-core` — no collision risk with the live alias). No
  `bus-exporter`/`bus-observer` — cold data target only. Header comment
  documents the exact deploy command and the two-`--env-file` requirement.
- `services/orion-bus/.env_example`: new `BUS_STANDBY_PORT` key (default
  `6380`, deliberately different from the primary's `REDIS_PORT` 6379 so
  the standby can be co-located with the primary for testing without a
  port collision — the exact test this session actually ran). Only read
  by the standby compose file, not the primary.
- `services/orion-bus/README.md`: new "Cold standby" section documenting
  the deploy pattern, why it follows the `docker-compose.atlas-workers.yml`
  precedent, and the corrected two-`--env-file` deploy command.
- `docs/notes/2026-07-16-bus-core-standby-cutover-runbook.md` (new):
  full manual cutover/cutback runbook.

## Schema / bus / API changes

None. No bus channels, schemas, or payload shapes changed — this is
transport-infrastructure config only.

## Env/config changes

- Added keys: `BUS_STANDBY_PORT` (default `6380`) in
  `services/orion-bus/.env_example`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced: yes, manually. The repo's
  `scripts/sync_local_env_from_example.py` only syncs keys that already
  exist in a service's `.env_example` at the time it runs, but this
  session's worktree only has the `.env_example` edit committed to a
  branch, not yet merged to the checked-out state the primary checkout's
  `.env_example` reads from — so the sync script found nothing new to add
  when run against the primary checkout. Added `BUS_STANDBY_PORT=6380` to
  `/mnt/scripts/Orion-Sapienform/services/orion-bus/.env` by hand,
  mirroring the exact comment/value from `.env_example`, and re-verified
  with `grep BUS_STANDBY_PORT`. Once this branch merges, a future
  `sync_local_env_from_example.py` run would be a no-op for this key
  (already present and matching).
- skipped keys requiring operator action: on the real `atlas` host, an
  operator must `cp services/orion-bus/.env_example
  services/orion-bus/.env.atlas-standby` and confirm atlas's own root
  `.env` has real `PROJECT`/`NODE_NAME`/`TELEMETRY_ROOT` values before
  first deploy — documented in both the compose file header and the
  runbook.

## Tests run

```text
docker compose -f services/orion-bus/docker-compose.atlas-standby.yml config --quiet
=> exit 0, valid syntax

docker compose --env-file .env_example --env-file services/orion-bus/.env_example \
  -f services/orion-bus/docker-compose.atlas-standby.yml config
=> confirmed, post-fix: container_name orion-orion-athena-bus-core-standby,
   volume source /mnt/telemetry/orion-athena/bus-standby/data (correct,
   no longer root-bound as it was pre-fix)

git diff --check          => clean
git check-ignore services/orion-bus/.env .env   => both correctly ignored
python3 scripts/check_service_env_compose_parity.py orion-bus
=> pre-existing limitation, not caused by this patch: the checker only
   supports single-service compose files; orion-bus's primary
   docker-compose.yml declares 3 services (bus-core, bus-exporter,
   bus-observer) and the checker can't disambiguate. Confirmed this is a
   tool limitation, not a parity gap introduced here (BUS_STANDBY_PORT is
   intentionally not referenced by the primary docker-compose.yml at all).
```

No `pytest` suite applies — this service has no Python application code,
only Redis config.

Live AOF-restore test (see Summary and runbook's "what was tested"
section for full detail):

```text
Snapshot: docker run --rm -v /mnt/telemetry/orion-athena/bus/data:/src:ro ... 
Restore test 1 (raw redis-server against snapshot): DBSIZE 20, matches live primary
Restore test 2 (actual docker-compose.atlas-standby.yml, synthetic node identity): 
  DBSIZE 20, healthy healthcheck, distinct bus-core-standby network alias
Post-test: live orion-orion-athena-bus-core confirmed still Up (healthy), DBSIZE 20
```

## Evals run

None applicable — no eval harness exists for `orion-bus`, and this patch
is Redis broker config + documentation, not application logic that
quality-evals would meaningfully cover.

## Docker/build/smoke checks

Ran directly (not through `scripts/safe_docker_build.sh`): that wrapper
hardcodes `-f services/$SERVICE/docker-compose.yml` (the primary compose
file only) and has no way to target an alternate, node-scoped compose
file like `docker-compose.atlas-standby.yml` — the same limitation
applies to the existing `docker-compose.atlas-workers.yml` precedent,
whose own README documents running raw `docker compose -f
docker-compose.atlas-workers.yml` directly, not through the wrapper. Ran
from a worktree throughout (never the shared checkout), which is the
actual risk the wrapper protects against.

```text
docker compose --env-file <scratch synthetic env> \
  -f services/orion-bus/docker-compose.atlas-standby.yml up -d
=> Started, healthy, DBSIZE 20 (see Tests run / Summary)

docker compose ... down
=> Clean teardown, confirmed live bus-core untouched throughout
```

`scripts/safe_graphify_update.sh` was run per repo convention after code
changes: reproduced the known 2026-07-14 destructive-update bug again
(node count would have dropped ~92%), correctly refused and restored
`graph.json`/`manifest.json`. Manually reverted the one file the wrapper
doesn't restore (`graphify-out/GRAPH_REPORT.md`) and removed stray
byproduct files (`.graphify_labels.json`, `.graphify_root`, `graph.html`)
so nothing from the failed update attempt is in this branch.

## Review findings fixed

Code review (orion-repo-agent subagent, high effort — correctness,
security, cleanup, conventions, altitude angles):

- **Finding (Critical)**: the documented deploy command
  (`docker compose --env-file services/orion-bus/.env.atlas-standby -f
  docker-compose.atlas-standby.yml up -d`) only chains one `--env-file`.
  `PROJECT`/`TELEMETRY_ROOT` live in the root `.env_example`, not
  `services/orion-bus/.env_example`, so a freshly copied
  `.env.atlas-standby` never contains them — reviewer reproduced this
  directly with `docker compose ... config` and got
  `container_name: orion--bus-core-standby` and a data volume bound at
  `/bus-standby/data` (filesystem root), not under `/mnt/telemetry/`.
  - **Fix**: compose file header, README, and runbook all now chain
    `--env-file .env` first, then the service env file, matching
    AGENTS.md section 8's mandatory dual-`--env-file` pattern used
    everywhere else in the repo.
  - **Evidence**: re-ran `docker compose ... config` with the corrected
    two-file chain — `container_name: orion-orion-athena-bus-core-standby`,
    `source: /mnt/telemetry/orion-athena/bus-standby/data` (correct).
- **Finding (Important)**: Step 1's raw shell commands
  (`mkdir -p ${TELEMETRY_ROOT}/${PROJECT}/...`) referenced
  `$TELEMETRY_ROOT`/`$PROJECT` as if they were exported shell variables,
  but they only ever existed inside `.env.atlas-standby`, a file
  `docker compose --env-file` reads internally — never sourced into the
  calling shell. As written, a literal copy-paste under incident pressure
  would seed data to the wrong path.
  - **Fix**: rewrote Step 1 to have the operator `grep` and manually
    substitute real `<PROJECT>`/`<TELEMETRY_ROOT>` values rather than rely
    on shell interpolation that was never wired up.
  - **Evidence**: re-read the corrected runbook section; no remaining
    unexported `${VAR}` references outside actual `docker compose
    --env-file` invocations.
- **Finding (Important)**: the compose header cited a `rebuild-services.sh`
  script (for auto-detecting `PROJECT`/`NODE_NAME`) that doesn't exist
  anywhere in this repo checkout — an unverified claim not disclosed in
  the runbook's own honesty section.
  - **Fix**: removed the claim; header now says plainly there's no
    verified auto-detection mechanism and the operator must set
    `PROJECT`/`NODE_NAME` by hand if not already correct.
  - **Evidence**: `find . -iname rebuild-services.sh` returns nothing;
    claim removed from `docker-compose.atlas-standby.yml`.
- **Finding (Minor)**: `BUS_STANDBY_PORT` defaulted to `6379`, identical to
  the primary's `REDIS_PORT` — a landmine if the standby is ever
  co-located with the primary on one host (exactly what this session did
  for testing) without remembering to override it.
  - **Fix**: changed default to `6380` in both `.env_example` and the
    synced local `.env`; runbook's Step 2 now explains the default and how
    to override back to `6379` for a real cutover.
  - **Evidence**: `services/orion-bus/.env_example:20`,
    `/mnt/scripts/Orion-Sapienform/services/orion-bus/.env:31`.

Not fixed, accepted and documented as an explicit gap in the runbook's
"Known gaps" section (would require actual atlas access to resolve, not
fixable from this session):

- Whether atlas's disk space, `app-net` network, root `.env`, and
  Tailscale firewall rules actually match what this runbook assumes.
- Whether the 20 "no visible `ORION_BUS_URL` reference" services in the
  mesh audit actually get the var some other way this session's grep
  didn't catch, or genuinely fall back to a stale hardcoded Python
  default during a real cutover.

## Restart required

No restart of any live service is required by this patch — nothing
consumes `bus-core-standby` automatically, and `ORION_BUS_URL` was not
changed. The only "restart" implied is the runbook's own manual cutover
procedure, which is deliberately not something to run outside a real
incident:

```text
No restart required.
```

## Risks / concerns

- Severity: medium
- Concern: nothing about the actual `atlas` deployment was tested from
  this session (no SSH access — confirmed via a live `Permission denied`
  response, not assumed). The compose file and AOF-restore mechanism are
  proven; whether `atlas` itself has the disk space, network, and
  firewall configuration this runbook assumes is not.
- Mitigation: the runbook's "What was actually tested vs. not" section
  states this plainly and up front, before any step-by-step instructions,
  so a human reads the caveat before following the procedure. A human
  with real `atlas` access needs to run Step 1 end-to-end (data restore +
  verification) on the actual node before this runbook should be trusted
  during a real incident.

- Severity: medium
- Concern: the mesh-wide `ORION_BUS_URL` audit (51/3/20 service
  breakdown) found 20 services with no visible bus-URL wiring in their
  `docker-compose.yml`, several of which are real cognition/runtime
  services (not just backing stores) that may silently keep talking to a
  dead primary during a cutover if they fall back to a hardcoded Python
  default.
- Mitigation: documented explicitly in the runbook rather than glossed
  over, with the exact `docker exec <container> env | grep
  ORION_BUS_URL` command to verify each one individually before trusting
  a cutover. Full remediation (wiring every one of those services to
  respect `ORION_BUS_URL` properly) is out of scope for this patch — it's
  a mesh-wide audit task, not a bus-core-standby task.

- Severity: low
- Concern: this is a cold standby with no live replication. Any writes
  that land on `bus-core-standby` during a cutover window are not
  automatically reconciled back to the primary on cutback.
- Mitigation: explicitly documented as accepted, by design (task scope
  excludes live replication/leader election). Cutback section describes
  the two real recovery paths (repoint back and accept the gap, or
  promote the standby's data if the primary's own data was lost).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1084
