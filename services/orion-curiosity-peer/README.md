# orion-curiosity-peer

Read-only contractor peer for Orion curiosity / self-inquiry runs.

Consumes `HelpRequestV1` on `orion:curiosity:help:request`, runs a
Cursor Auto investigation (Claude room fallback when Cursor tokens are
dry), and publishes `PeerBriefV1` via `persist_peer_brief`. Bus publish
always runs (sql-writer / Postgres consumers). Worldview `:PeerBrief`
MERGE runs only when `ORION_CURIOSITY_GRAPH_HOST` + port + user + password
are set; otherwise the worker logs
`curiosity_peer_persist_graph_unconfigured` and continues bus-only.
Orion alone still writes `:Prior` / `:Finding` / `:SelfDefinition`.

Design: `docs/superpowers/specs/2026-09-14-orion-contractor-peer-design.md`

**Status:** Patch 1 worker armed (Tasks 12–13). Kill switches still default
off. Cursor meter remains fail-closed until a live reading exists.

## Why this is a separate service

`CURSOR_API_KEY` lives here, **not** in `orion-hub`. Hub runs as root with
`/var/run/docker.sock`, SSH keys, and FCC Bash — a Cursor key there is one
Orion can read and spend. Same credential-isolation pattern as
`orion-room-companion` for Claude.

This separation is defense in depth, not a hard wall while Hub holds the
docker socket (`docker inspect` can still read this container's env).

## Kill switches

Two flags, both required for a live hire:

| Flag | Where | Role |
| --- | --- | --- |
| `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED` | Hub | Enqueue / hire gate (default `false`) |
| `CURIOSITY_PEER_ENABLED` | this service | Subscribe / process gate (default `false`) |

Turning only one on is a deliberate no-op. Typo alias
`CURIOUSITY_PEER_ENABLED` is accepted by Settings.

Disable either flag and behavior returns to today's curiosity path; unused
PeerBrief rows/nodes are additive.

## Read-only policy

Cursor Auto jobs must stay investigation-only: allowlist tools
(`read` / `grep` / `glob` / `ls`), no shell/edit/delete, no graph belief
writes, no docker mutate. Enforced in Task 12 (`app/policy.py` + tests).

Self-inquiry mode: peer may point at evidence; never draft `:SelfDefinition`.

## Bus

- Consumes: `orion:curiosity:help:request`
- Produces: `orion:curiosity:peer:brief` (via `persist_peer_brief`)
- Claude fallback transport (Task 13): `orion:room:claude:request` /
  `orion:room:claude:utterance`

## Run

```bash
# From a worktree — never the shared checkout:
scripts/safe_docker_build.sh orion-curiosity-peer up -d --build
```

Default `CURIOSITY_PEER_ENABLED=false` keeps the container idle (heartbeat
optional once enabled). Do not enable until Task 12–13 + Cursor budget
observation are ready.
