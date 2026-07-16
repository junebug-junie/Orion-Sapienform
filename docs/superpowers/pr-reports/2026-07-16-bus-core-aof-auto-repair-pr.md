# PR report — bus-core AOF auto-repair on boot

PR: (create at https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/bus-core-aof-auto-repair)
Branch: `fix/bus-core-aof-auto-repair`
Status: **DONE**

## Summary

- `bus-core` (`services/orion-bus`, `redis:7-alpine`, `--appendonly yes`)
  periodically crash-loops after an unclean shutdown (host reboot, OOM, disk
  pressure) because its AOF ends up corrupted and Redis refuses to start.
  `restart: unless-stopped` just retries the identical broken state forever
  — no auto-recovery. Juniper has had to manually
  `rm -rf /mnt/telemetry/orion-athena/bus/data/*`, which destroys ALL bus
  state (not just the corrupted part) while cascading every other service
  that depends on the bus.
- Added a custom `entrypoint.sh` that runs `redis-check-aof --fix` against
  the AOF before `redis-server` starts, every boot, and wired it in via a
  new `Dockerfile.bus-core` (bus-core previously used the stock
  `redis:7-alpine` image directly with no `build:` context).
- Verified redis-check-aof's real behavior empirically (not assumed) —
  including a load-bearing landmine: `--fix` is interactive whenever it
  actually finds something to repair (`Continue? [y/N]:` on stdin), and
  there is no `--yes`/`--force` flag. Feeding it closed stdin does **not**
  hang — it reads EOF, defaults to "N", and silently **aborts without
  fixing** (exit 1). The entrypoint explicitly feeds `y` on stdin to make
  this a real repair instead of a silent no-op.
- Built an end-to-end regression smoke test
  (`services/orion-bus/tests/test_bus_core_aof_repair.sh`) that builds the
  real image, reproduces the actual production failure mode (mid-file AOF
  corruption, not just an EOF truncation Redis already tolerates on its
  own), and proves repair + clean boot + data preservation — plus a
  fail-closed case for genuinely unrecoverable corruption. All against a
  throwaway `/tmp` directory; the live
  `/mnt/telemetry/orion-athena/bus/data` volume was never touched.
- Code review (orion-repo-agent, high effort, no dedicated `code-review`
  skill found in this repo) found 3 material findings; all fixed (see
  below).

## Outcome moved

`bus-core` now self-heals from the common AOF-corruption failure mode
instead of crash-looping forever on it. Recoverable corruption (the
overwhelmingly common case — a torn tail write from an unclean shutdown)
now repairs automatically on the next boot with no data loss beyond the
torn tail itself. Genuinely unrecoverable corruption still refuses to
start (same terminal state as today, since Redis's own AOF loader already
refused pre-patch) but now logs a distinguishable `[entrypoint] FATAL:`
line instead of a bare Redis panic, making root-causing it fast instead of
requiring someone to already know the `redis-check-aof --fix` incantation.

## Current architecture

`bus-core` (`services/orion-bus/docker-compose.yml`) ran the stock
`redis:7-alpine` image directly (`image:`, no `build:`), started via
`command: ["redis-server", "--appendonly", "yes", ...]`, with `/data`
bind-mounted to `${TELEMETRY_ROOT}/${PROJECT}/bus/data`. No entrypoint
customization existed for this service. Two precedents for custom
entrypoints on stock/base images already existed elsewhere in the repo
(`services/orion-fcc/entrypoint.sh`, `services/orion-hub/entrypoint.sh`),
both following the same shape: `COPY entrypoint.sh /usr/local/bin/`,
`chmod +x`, `ENTRYPOINT ["entrypoint.sh"]`, ending in `exec "$@"` to hand
off to the compose-supplied `command:` unchanged. This patch follows that
same shape.

## Architecture touched

`services/orion-bus/` only — `docker-compose.yml` (bus-core service
definition), plus new `Dockerfile.bus-core`, `entrypoint.sh`,
`.dockerignore`, and a new test. No cross-service contract touched: bus
channels, schema registry, and the `command:`/ports/healthcheck/volumes
bus-core exposes to `bus-exporter`/`bus-observer` are all unchanged.

## Files changed

- `services/orion-bus/docker-compose.yml`: `bus-core`'s `image:
  redis:7-alpine` replaced with `build: { context: ., dockerfile:
  Dockerfile.bus-core }` so the custom entrypoint can be baked in.
  `command:`, `ports:`, `volumes:`, `healthcheck:`, `networks:` all
  unchanged.
- `services/orion-bus/Dockerfile.bus-core` (new): `FROM redis:7-alpine`,
  `COPY entrypoint.sh`, `chmod +x`, `ENTRYPOINT ["entrypoint.sh"]`.
- `services/orion-bus/entrypoint.sh` (new): runs `redis-check-aof --fix`
  against `/data/appendonlydir/appendonly.aof.manifest` (falling back to a
  legacy single-file `/data/appendonly.aof` defensively) before `exec
  "$@"`. Preflights that `redis-check-aof` exists on `PATH` so a future
  base-image change that removes/moves the tool fails with a
  distinguishable message instead of being conflated with real AOF
  corruption. Fails loudly (non-zero exit, `[entrypoint] FATAL:` log
  lines) if the repair itself fails, rather than silently starting on
  unknown/lossy data or silently skipping the check.
- `services/orion-bus/.dockerignore` (new, force-added — this repo's root
  `.gitignore` blanket-ignores `.dockerignore`, but `services/orion-hub`
  and `services/orion-rag` both already have one tracked via the same
  force-add, so this follows existing precedent): scopes the
  `Dockerfile.bus-core` build context defensively so a future accidental
  `COPY . .` (easy copy-paste from the sibling Python `Dockerfile` in the
  same directory) can't pull the unrelated `bus-observer` Python app tree
  into the Redis image. No effect on the current build, which only `COPY`s
  `entrypoint.sh` explicitly.
- `services/orion-bus/tests/test_bus_core_aof_repair.sh` (new):
  self-contained end-to-end smoke test — builds `Dockerfile.bus-core`
  fresh, seeds a real healthy multi-part AOF via a live container,
  confirms it's left byte-for-byte unmodified by the repair step, corrupts
  a copy with a mid-file length/data mismatch (the real fatal-error shape,
  not a tolerated EOF truncation), confirms a plain `redis-server`
  actually dies on it, confirms the new image repairs it and boots clean
  with pre-corruption data intact, and confirms a manifest referencing a
  missing file (a distinct, genuinely unrecoverable corruption shape) is
  refused rather than silently started. Container/image names are
  PID-suffixed so concurrent runs across this repo's routinely-parallel
  agent worktrees don't collide.

## Schema / bus / API changes

None. No bus channel, schema registry, or payload shape touched.

## Env/config changes

- Added keys: none required. `entrypoint.sh` reads an optional
  `REDIS_DATA_DIR` (defaulting to `/data`) purely as internal defensive
  generality — it is not referenced anywhere in `docker-compose.yml`'s
  `environment:` block and does not need a `.env_example` entry, since
  nothing wires it as a required/expected operator-facing key.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: not applicable, no key added.
- local `.env` synced: not applicable, no `.env_example` change.
- skipped keys requiring operator action: none.

## Tests run

```text
sh services/orion-bus/tests/test_bus_core_aof_repair.sh
=> == ALL CHECKS PASSED ==  (exit 0)
   6 scenarios: healthy-AOF-boots-clean, healthy-AOF-left-byte-identical,
   mid-file-corruption-reproduces-real-fatal-error,
   corrupted-AOF-repaired-and-boots-with-prior-data-intact,
   unrecoverable-manifest-refused-not-silently-started (post-review addition)

PYTHONPATH=. .venv/bin/pytest services/orion-bus/tests -q \
  --ignore=services/orion-bus/tests/test_bus_core_aof_repair.sh
=> 15 passed (pre-existing suite, confirming no regression)

docker build -f services/orion-bus/Dockerfile.bus-core -t orion-bus-core-test:local services/orion-bus
=> builds clean

PROJECT=orion-athena TELEMETRY_ROOT=/tmp docker compose \
  -f services/orion-bus/docker-compose.yml config
=> resolves correctly: build.dockerfile=Dockerfile.bus-core,
   command: unchanged (redis-server --appendonly yes ...)

git diff --check
=> clean
```

## Evals run

None applicable — this is infrastructure/boot-behavior code, not a
cognition/quality surface with an eval harness. The regression smoke test
above is the closest equivalent and is the primary verification artifact.

## Docker/build/smoke checks

```text
docker build -f services/orion-bus/Dockerfile.bus-core -t orion-bus-core-test:local services/orion-bus
=> #8 naming to docker.io/library/orion-bus-core-test:local done

Live redis-check-aof behavior verified directly against redis:7-alpine (Redis 7.4.9):
- `--appendonly yes` with no other aof-* flags produces Redis 7's
  multi-part AOF layout: /data/appendonlydir/appendonly.aof.manifest +
  appendonly.aof.<N>.base.rdb + appendonly.aof.<N>.incr.aof. Confirmed by
  seeding real writes and inspecting the resulting directory.
- `strings $(which redis-check-aof)` confirms no --yes/--force/non-interactive
  flag exists in the redis:7-alpine binary.
- `--fix` on an already-healthy AOF: no prompt, "All AOF files and
  manifest are valid", exit 0, file byte-identical before/after (confirmed
  via sha1sum) -- idempotent and cheap on normal boots.
- `--fix` on a real mid-file corruption (length header no longer matches
  the bytes that follow, with more valid-looking data after the corruption
  point -- NOT a clean EOF truncation, which Redis already tolerates via
  its own aof-load-truncated default): prints "Continue? [y/N]:" and reads
  stdin. Confirmed a closed/empty stdin does NOT hang -- it reads EOF,
  defaults to "N", prints "Aborting...", and exits 1 WITHOUT repairing the
  file (silent-looking failure). Confirmed piping "y\n" makes it truncate
  to the last valid record and exit 0, and that a plain redis-server then
  starts clean with pre-corruption data intact.
- `--fix` on a manifest referencing a genuinely missing file (simulating a
  crash mid-manifest-rotation): reports the missing file, exits nonzero;
  entrypoint.sh correctly logs FATAL and refuses to start (fail-closed).
```

No `docker compose up` against the live `bus-core` service was run this
session — see Restart required below for why that's deliberate.

## Review findings fixed

Ran `orion-repo-agent` (high effort) as a substitute for a dedicated
`code-review` skill — none exists in this repo's `.claude/agents`,
`.claude/commands`, or `~/.claude/skills` under that name. The reviewer
verified claims empirically (built the image, reproduced the fatal error,
reproduced the regression by testing a no-op entrypoint variant against
the same fixture) rather than just reading the diff.

- **Finding (material, M1)**: test container names and the default image
  tag were fixed literals (`bus-core-aof-test-c1`, `orion-bus-core-test:local`).
  This repo routinely runs multiple agent worktrees concurrently against
  the same host Docker daemon — a second concurrent run would collide
  (`docker run --name` failing "already in use", or worse, one run's
  cleanup trap killing a container belonging to a different concurrent
  run).
  - **Fix**: image tag and all container names suffixed with `$$` (this
    process's PID), matching the pattern `$WORKDIR` already used.
  - **Evidence**: re-ran the test twice; each run builds
    `orion-bus-core-test:local.<pid>` and uses `bus-core-aof-test-cN.<pid>`
    container names, no collision possible.
- **Finding (material, M2)**: a real, distinct AOF corruption shape —
  a manifest referencing a base/incr file that never finished being
  written (crash mid-manifest-rotation) — was not covered by the test.
  The reviewer confirmed live that `entrypoint.sh`'s current behavior on
  this case is already correct (fails closed), but a future refactor of
  the if/else could silently break that path with nothing to catch it.
  - **Fix**: added Step 6 to `test_bus_core_aof_repair.sh` — a manifest
    referencing a deliberately-omitted incr file, asserting the container
    logs `[entrypoint] FATAL: redis-check-aof --fix failed` and never
    reaches `Ready to accept connections`.
  - **Evidence**: `sh services/orion-bus/tests/test_bus_core_aof_repair.sh`
    now runs 6 scenarios, all pass.
- **Finding (material, M3)**: the FATAL log message didn't distinguish
  "genuinely unrecoverable AOF" from "the redis-check-aof tool itself is
  missing/moved" (e.g. a future `redis:7-alpine` base bump). Given this is
  the mesh's Redis broker and the log line is the only diagnostic signal a
  human gets, that ambiguity matters.
  - **Fix**: added a `command -v redis-check-aof` preflight in
    `entrypoint.sh` that fails with a distinct message
    ("binary not found on PATH ... NOT that the AOF itself is corrupt")
    before attempting the repair.
  - **Evidence**: `sh -n services/orion-bus/entrypoint.sh` (syntax check)
    plus the full smoke test still passing (preflight doesn't fire on the
    happy path since `redis-check-aof` is present in the real image).
- **Nit (N1)**: no `.dockerignore` for `services/orion-bus/`, which also
  hosts the unrelated `bus-observer` Python app — latent risk if
  `Dockerfile.bus-core` is ever copy-paste-edited to `COPY . .`.
  - **Fix**: added `services/orion-bus/.dockerignore`, force-added per
    existing repo precedent (`orion-hub`, `orion-rag`).
  - **Evidence**: re-ran the smoke test after adding it — still passes,
    confirming no effect on the current build.
- **Nit (N2)**: `REDIS_DATA_DIR` env override in `entrypoint.sh` is unused
  dead generality (compose never sets it). Left as-is — harmless default,
  documented above under Env/config changes.
- **Nit (N3)**: test didn't `docker rmi` the built image on exit, so
  images would accumulate across repeated runs once M1's fix moved to a
  per-PID image tag.
  - **Fix**: added `docker rmi -f "$IMAGE"` to the test's cleanup trap.
  - **Evidence**: `docker images | grep bus-core-test` empty after a test
    run.

No blocking findings.

## Restart required

**bus-core is the mesh's shared Redis pub/sub broker — restarting it
briefly takes down messaging for every service on the bus, not just
`orion-bus` itself (`bus-exporter` and `bus-observer` both `depends_on:
condition: service_healthy` against it, and every other service in this
repo talks to it via `ORION_BUS_URL`).** This was NOT run against the live
service this session — only against a throwaway image/volume in `/tmp`.
Coordinate the restart window before running this:

```bash
scripts/safe_docker_build.sh orion-bus up -d --build
curl -fsS -o /dev/null -w '%{http_code}\n' http://localhost:${REDIS_EXPORTER_PORT:-9121}/metrics  # bus-exporter sanity
docker compose -f services/orion-bus/docker-compose.yml logs --tail=50 bus-core
```

The rebuild pulls no new base image (still `redis:7-alpine`, same floating
tag as before this patch — not a new pinning regression, see Risks below)
and only adds the entrypoint layer, so the running container's actual
Redis behavior is unaffected except for the new pre-boot repair step. On
a healthy AOF (the expected case, since the live volume isn't currently
corrupted) the repair step is a fast no-op, confirmed idempotent above.

## Risks / concerns

- Severity: low
- Concern: `redis:7-alpine` was already an unpinned floating tag before
  this patch (not pinned to a digest). Switching from `image:` to `build:
  FROM redis:7-alpine` carries the same floating-tag risk as
  status quo — not a new regression, but also not improved by this patch.
- Severity: low
- Concern: `docker compose up` without `--build` (or without going through
  `scripts/safe_docker_build.sh`) won't pick up future `entrypoint.sh`
  edits once the image exists locally — an operational gotcha worth
  knowing when iterating on this file later.
- Severity: low
- Concern: genuinely unrecoverable AOF corruption still ends in the same
  terminal crash-loop state as before this patch (Redis's own loader
  already refused to start pre-patch) — this patch only fixes the common
  *recoverable* corruption case and improves diagnosability of the
  unrecoverable case, it does not eliminate manual intervention for truly
  unrecoverable AOF damage.

## PR link

`gh` is authenticated in this environment — see below.
