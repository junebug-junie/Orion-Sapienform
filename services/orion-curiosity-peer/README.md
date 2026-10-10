# orion-curiosity-peer

Read-only contractor peer for Orion curiosity / self-inquiry runs.

Consumes `HelpRequestV1` on `orion:curiosity:help:request`, runs a
Cursor Agent CLI investigation (`agent -p --mode ask`, Claude room
fallback when Cursor desktop auth is dry), and publishes `PeerBriefV1`
via `persist_peer_brief`. Bus publish always runs (sql-writer /
Postgres consumers). Worldview `:PeerBrief` MERGE runs only when
`ORION_CURIOSITY_GRAPH_HOST` + port + user + password are set; otherwise
the worker logs `curiosity_peer_persist_graph_unconfigured` and continues
bus-only. Orion alone still writes `:Prior` / `:Finding` / `:SelfDefinition`.

Design: `docs/superpowers/specs/2026-09-14-orion-contractor-peer-design.md`

**Status:** Patch 1 worker armed (Tasks 12–13). Kill switches may be
armed in local `.env_example` on this host; leave `CURIOSITY_PEER_ENABLED`
false until you intend processing.

## Contested Cursor budget

Hires fail closed until a meter reading exists (`budget_unobserved`).
Until a first-party Cursor usage API lands, set one of:

- `CURIOSITY_PEER_CURSOR_BUDGET_STATE=clear` — operator asserts pool headroom
- `CURIOSITY_PEER_CURSOR_BUDGET_FILE=/path` — JSON
  `{"state":"clear","observed_at":"<iso>"}` or plain text `clear`

File wins over STATE. `limited` / `unknown` / garbage / missing all refuse.
`decide_cursor_budget` still requires observed `clear` with a staleness value.

## Claude fallback budget

When Cursor fails for lack of tokens, the peer may try Claude once. That
spend is Claude quota Orion shares with Juniper, so it is gated on the
**Claude** meter (`orion.dev_economics.rate_limit_events.observe`), which
fails closed on anything short of a fresh, observed `clear`.

Until 2026-09-25 this gate read the **Cursor** meter by accident. A local
variable shadowed the Claude reader, so a clear Cursor reading was enough to
spend Claude. That is fixed. The meter reads Claude Code transcripts on disk,
and this container does not mount them, so in production the Claude fallback
now refuses with `claude_budget_unobserved`. That is the intended fail-closed
behaviour, not a new outage.

Re-enabling the fallback is a separate decision. Either give this container a
readable Claude meter (for example by consuming the observation the
`orion-cocreation-signals` Claude-limit publisher emits, which is currently
off: `COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED=false`), or accept that it stays
off. D2 in `docs/superpowers/specs/2026-09-25-attention-with-stakes-design.md`
has the full account.

### What the refusal says

When Cursor is out and Claude is refused (or fails, or is not wired), the
brief's `refusal_reason` names the whole chain, plain sentences first:

```text
Cursor unavailable: it hit its usage limit (resets 2026-10-14). Claude fallback
refused: this service cannot see Claude's usage meter, so it refuses rather than
spend blind. [cursor_token_unavailable:usage_limit; claude_budget_unobserved]
cursor said: <first 400 chars of Cursor's own error>
```

The Cursor cause is one of `usage_limit`, `auth`, `binary_missing`, or
`unknown`, matched on Cursor's error text (`app/cursor_errors.py`). The reset
date is parsed only from an explicit `reset ... on M/D/YYYY`; otherwise the
brief says `reset date unknown`. Orion reads this text in the kickoff "COULD
NOT HIRE" section (`format_soft_nudge`) and, for `refused_budget`, in the
role-teach progress line. Until 2026-10-10 the brief carried only
`claude_budget_unobserved`, and Orion built a "regime break" theory around what
was Cursor's monthly billing cap.

Non-token Cursor failures are unchanged: `failed` with
`cursor_other: <error>`.

## Why this is a separate service

Cursor desktop CLI auth (`agent login` → `~/.config/cursor/auth.json`)
lives here via a host bind mount, **not** in `orion-hub`. Hub runs as
root with `/var/run/docker.sock`, SSH keys, and FCC Bash — a Cursor
credential there is one Orion can read and spend. Same credential-
isolation pattern as `orion-room-companion` for Claude. There is no
`CURSOR_API_KEY` / `cursor-sdk` path.

This separation is defense in depth, not a hard wall while Hub holds the
docker socket (`docker inspect` can still read this container's env).

## Kill switches

Two flags, both required for a live hire:

| Flag | Where | Role |
| --- | --- | --- |
| `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED` | Hub | Enqueue / hire gate (default `false`) |
| `CURIOSITY_PEER_ENABLED` | this service | Subscribe / process gate (Settings default `false`; may be `true` in local `.env_example`) |

Turning only one on is a deliberate no-op. Typo alias
`CURIOUSITY_PEER_ENABLED` is accepted by Settings.

Disable either flag and behavior returns to today's curiosity path; unused
PeerBrief rows/nodes are additive.

## Read-only policy

Cursor Agent jobs must stay investigation-only: CLI argv requires
`-p` / `--print`, `--mode ask`, `--workspace`, and `--trust`. Forbidden:
`--force`, `--yolo`, `--approve-mcps`, `--mode plan`. Enforced in
`app/policy.py` + tests (argv assertions; no live agent).

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

Authenticate on the host first: `agent login` (or `cursor-agent login`).
Compose mounts `${HOME}/.local/share/cursor-agent` and
`${HOME}/.config/cursor` into the container. Set
`CURIOSITY_PEER_AGENT_BIN` to the versioned binary under `/opt/cursor-agent`.

## Ask commitments and later decisions

`CURIOSITY_PEER_EPISODES_ENABLED` is true in the operator template and local
`.env` (the code fallback when unset is false). When enabled, eligible
requests need an explicit expectation and an acknowledged worldview commitment
before provider invocation. Duplicate claims do not resend. Deploy this consumer
before the updated Hub producer. See [contracts, rollout and rollback](../../docs/peer-ask-episodes.md).
