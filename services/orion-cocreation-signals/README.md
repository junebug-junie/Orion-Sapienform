# orion-cocreation-signals

Owns all external I/O (git, GitHub, graphify, Juniper's local Claude Code
transcripts) for the codebase-mass and Juniper affective-state Predictive
Processing domains. Design docs:
`docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md`
("Producer + consumer patch design" section) and
`docs/superpowers/specs/2026-07-30-juniper-affective-state-signal-proposal.md`.

No FastAPI app, no HTTP surface -- four independent async producer loops
(`app/producers/{git_delta,pr_lifecycle,graph_delta,affective_state}.py`),
each on its own interval, each with its own enable flag. The first three
publish `CodebaseDeltaV1` events to `orion:substrate:codebase_delta`;
`affective_state` publishes `JuniperAffectiveStateV1` events to
`orion:substrate:juniper_affective_state`. `codebase_delta` is consumed by
`orion-substrate-runtime`'s existing fast tick; `juniper_affective_state` has
no consumer yet -- a deliberate shadow write (see "What's real, what's not").

## What's real, what's not (as of this patch)

Real: this service, `git_delta`/`pr_lifecycle`/`graph_delta`, publishing real
bus events with real computed deltas from `orion/structural_mass/{git_delta,
pr_lifecycle,graph_delta}.py` (shipped, tested, replay-verified against this
repo's own real history in earlier PRs). `affective_state` is also real and
tested, with its own real offline replay against 113 local transcripts
(`docs/superpowers/pr-reports/2026-08-11-juniper-affective-state-signal-
replay.md`) -- but ships with `COCREATION_SIGNALS_AFFECTIVE_STATE_ENABLED`
default `false`, since only `swear_frequency` passed the replay's live-data
sanity check (`typo_rate` did not, and is deliberately not on the wire
schema at all -- see `orion/schemas/affective_state.py`).

Not yet real: no consumer reads `codebase_delta`'s `node:substrate.codebase`
in any live store, and nothing consumes `juniper_affective_state` at all.
This service running does not yet change any Orion-facing behavior -- it
only proves the producer half of each arc works end-to-end (real bus
publishes), per this program's "measure before minting" discipline.

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
| `affective_state` | 15min poll (scans Juniper's real local transcript tree) | Every tick -- same all-zero-is-real-data reasoning as `pr_lifecycle`. Default **disabled**. |

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
- `COCREATION_SIGNALS_{GIT_DELTA,PR_LIFECYCLE,GRAPH_DELTA,AFFECTIVE_STATE}_ENABLED`:
  each producer independently toggleable, so a GitHub API/rate-limit problem
  in `pr_lifecycle` can never block the others.
- `COCREATION_SIGNALS_CLAUDE_PROJECTS_HOST_PATH` / `COCREATION_SIGNALS_CLAUDE_PROJECTS_PATH`:
  host path (operator-specific -- Juniper's real local `~/.claude/projects`)
  bind-mounted **read-only** into the container for `affective_state`.
  Deliberately the whole tree, not scoped to just this repo's own sessions
  (Juniper's explicit call, 2026-08-11 -- see the PR report linked above for
  the privacy tradeoff this decided, including the precedent in
  `orion-harness-governor/docker-compose.yml` this choice knowingly departs
  from). Raw transcript content is read only to compute an in-memory
  aggregate score (`orion/cocreation/affective_signals.py`) and never
  persisted or logged -- see that module's and `claude_code_ingest.py`'s own
  docstrings for the privacy boundary this is built to.

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
`orion/structural_mass/tests/`). `test_affective_state_producer.py` follows
the same pattern for scheduling, plus one real (not mocked) end-to-end test
against a real transcript file on disk, exercising the actual wiring to
`orion/dev_economics/claude_code_ingest.py` and
`orion/cocreation/affective_signals.py`.
