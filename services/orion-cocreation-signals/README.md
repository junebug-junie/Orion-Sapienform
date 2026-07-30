# orion-cocreation-signals

Owns all external I/O (git, GitHub, graphify) for the codebase-mass Predictive
Processing domain. Design: `docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md`
("Producer + consumer patch design" section).

No FastAPI app, no HTTP surface -- three independent async producer loops
(`app/producers/{git_delta,pr_lifecycle,graph_delta}.py`), each on its own
interval, each with its own enable flag, each publishing a
`CodebaseDeltaV1` event to `orion:substrate:codebase_delta` when it has a
real delta to report. Consumed by `orion-substrate-runtime`'s existing fast
tick (not yet wired as of this service's first patch -- see the design spec).

## What's real, what's not (as of this patch)

Real: this service, its three producers, publishing real bus events with
real computed deltas from `orion/structural_mass/{git_delta,pr_lifecycle,
graph_delta}.py` (all shipped, tested, replay-verified against this repo's
own real history in earlier PRs).

Not yet real: no consumer reads these events. `node:substrate.codebase`
does not exist in any live store. This service running does not yet change
any Orion-facing behavior -- it only proves the producer half of the arc
works end-to-end (real bus publishes), per this program's "measure before
minting" discipline.

## Why one service for three producers

See the design spec's "Dedicated service" section: credential/mount
consolidation (this is the one service with a read-only repo mount and `gh`
CLI access, not scattered across services that shouldn't need them) at a
scale (originally scoped for 6+ eventual producers) that justifies the
upfront cost of a new compose service over a slow task bolted onto
`orion-substrate-runtime`.

## Producers

| Producer | Cadence | Publishes when |
|---|---|---|
| `git_delta` | 60s poll (cheap `git rev-parse HEAD` check) | Real SHA change since last check |
| `pr_lifecycle` | 15min poll (`gh pr list`, rate-limit-aware) | Every tick -- a time window always has a real answer, even all-zero |
| `graph_delta` | 5min poll (`graphify-out/graph.json`'s own `built_at_commit` tag) | Real graphify re-run since last check |

Each producer's in-process "last seen" state resets on container restart --
accepted simplification, since a missed window just produces a bigger real
diff on the next successful check (git_delta/graph_delta) or a normal-sized
window starting from restart time (pr_lifecycle), not a lost/incorrect
reading. See each producer module's own docstring for the full reasoning.

## Env vars

See `.env_example` for the full list and inline comments. Notable:

- `COCREATION_SIGNALS_REPO_HOST_PATH` / `COCREATION_SIGNALS_REPO_PATH`: host
  path (operator-specific, varies per deployment node) bind-mounted
  read-only into the container. `git_delta`/`graph_delta` need a real,
  live `.git` directory and `graphify-out/` -- not baked into the image at
  build time, which would freeze history at build time.
- `COCREATION_SIGNALS_GH_TOKEN`: `gh` CLI auth (via `GH_TOKEN`, which `gh`
  honors natively). Read-only `repo`/`pull_request` scope is sufficient --
  this service never writes to GitHub. No `gh auth login` step, no mounted
  credential file.
- `COCREATION_SIGNALS_{GIT_DELTA,PR_LIFECYCLE,GRAPH_DELTA}_ENABLED`: each
  producer independently toggleable, so a GitHub API/rate-limit problem in
  `pr_lifecycle` can never block the other two.

## Run

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-cocreation-signals/.env \
  -f services/orion-cocreation-signals/docker-compose.yml \
  up -d --build
```

Prefer `scripts/safe_docker_build.sh orion-cocreation-signals ...` over a raw
`docker compose` invocation -- see root `CLAUDE.md` §8.

## Tests

```bash
pytest services/orion-cocreation-signals/tests -q
```

Producer loop tests use a fake in-memory bus (records published envelopes,
no real Redis connection) and monkeypatch the underlying `orion/
structural_mass/*.py` functions -- they test scheduling/state-machine logic
(cold start, publish-on-change vs. publish-every-tick, enable-flag
independence), not the pure functions themselves (already covered by
`orion/structural_mass/tests/`).
