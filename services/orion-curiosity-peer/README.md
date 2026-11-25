# orion-curiosity-peer

Read-only contractor peer for Orion curiosity / self-inquiry runs. **Claude
is the only peer.**

Consumes `HelpRequestV1` on `orion:curiosity:help:request`, asks Claude once
through `orion-room-companion` (one `RoomClaudeRequestV1` on
`orion:room:claude:request`, one matching utterance back), and publishes
`PeerBriefV1` (`peer=claude_room`) via `persist_peer_brief`. Bus publish
always runs (sql-writer / Postgres consumers). Worldview `:PeerBrief` MERGE
runs only when `ORION_CURIOSITY_GRAPH_HOST` + port + user + password are set;
otherwise the worker logs `curiosity_peer_persist_graph_unconfigured` and
continues bus-only. Orion alone still writes `:Prior` / `:Finding` /
`:SelfDefinition`.

Design: `docs/superpowers/specs/2026-09-14-orion-contractor-peer-design.md`

## Cursor retired (2026-11-17, #3397)

This service used to run the Cursor Agent CLI first and try Claude only when
Cursor failed for lack of tokens. Juniper no longer uses Cursor. From
2026-11-05 Cursor failed every hire, and from 2026-11-14 the Claude fallback
was refused too, so Orion got no brief at all. The Cursor path is gone: no
CLI call, no Cursor model or budget keys, no Cursor error classifier, no
Cursor host mounts. Old rows with `peer=cursor_auto` stay readable (the
schema still accepts the value).

The separate grading-side Cursor caller (`orion/curiosity/supervisor.py`,
`hire_cursor_review`) is not part of this service and is not changed here.

## Order of a hire

1. Claude budget gate (fail-closed, below). Refused → `refused_budget` brief.
2. Episode commitment, when `CURIOSITY_PEER_EPISODES_ENABLED=true`.
3. Exactly one Claude request. Reply → `ok` / `empty` brief; error or
   timeout → `failed` brief.

## Claude budget: try, back off on refusal (#3655)

Claude quota is shared with Juniper. Two inputs gate a hire:

1. **Claude's own last answer.** When a Claude call fails with "You've hit
   your session/weekly/usage limit · resets 12:10pm (UTC)", every later hire
   is refused (`refused_budget`, code `claude_limit_backoff`) until the reset
   Claude named, with no Claude call. No reset time in the text: wait
   `CURIOSITY_PEER_CLAUDE_LIMIT_BACKOFF_DEFAULT_SEC` (default 3600). Any later
   Claude outcome (ok, or a non-limit failure) clears it. The hold lives in
   process memory; on boot it is seeded from the newest `claude_room`
   PeerBrief in the worldview graph from the last 5 h.
2. **The transcript meter** (`orion.dev_economics.rate_limit_events.observe`)
   over Claude Code transcripts under `CURIOSITY_PEER_CLAUDE_PROJECTS_PATH`
   (compose mounts that host directory read-only at the same path). A
   transcript-observed `limited` refuses (`claude_budget_limited`). `unknown`
   (no Claude Code session in 5 h, so the account is idle) tries Claude.

Why: the meter alone was wrong both ways. 2026-11-19..23 it read `unknown`
and 13 hires were refused while nobody used Claude. 2026-11-17 and 11-24 it
read `clear` while Claude was at its session limit, and the 11-24 11:23Z hire
repeated the 09:12Z failure. `observe()` and `ask_claude_trigger._budget_refusal`
are unchanged for their other callers.

### What a refusal says

`refusal_reason` is a plain sentence, then a code block:

```text
Claude helper refused: Claude said at 09:12Z that its usage limit was hit and
resets at 2026-11-24 12:10Z, so this hire waits until then. [claude_limit_backoff]
```

Codes: `claude_limit_backoff` (Claude said limit), `claude_budget_*` (meter), `claude_failed` (Claude was asked and the
request failed or timed out), `claude_unwired` (no bus in this process).
Orion reads the plain part in the kickoff "COULD NOT HIRE" section
(`format_soft_nudge`) and, for `refused_budget`, in the role-teach progress
line.

## Why this is a separate service

Hub runs as root with `/var/run/docker.sock`, SSH keys, and FCC Bash. This
worker holds no Claude credential itself (room-companion does) and mounts only
the read-only transcript tree it needs for the meter.

## Kill switches

Two flags, both required for a live hire:

| Flag | Where | Role |
| --- | --- | --- |
| `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED` | Hub | Enqueue / hire gate (default `false`) |
| `CURIOSITY_PEER_ENABLED` | this service | Subscribe / process gate (Settings default `false`; may be `true` in local `.env_example`) |

Turning only one on is a deliberate no-op. Typo alias
`CURIOUSITY_PEER_ENABLED` is accepted by Settings.

## Read-only policy

The sealed prompt (`app/brief_prompt.py`) tells Claude to use only read-only
tools and return a JSON PeerBrief. Claude's reply is marked
`[conversation-only; peer could not look at the repo]` and its
`evidence_pointers` are dropped (`_map_claude_result`).

Self-inquiry mode: peer may point at evidence; never draft `:SelfDefinition`.

## Bus

- Consumes: `orion:curiosity:help:request`, `orion:curiosity:peer:brief:consumed`
- Produces: `orion:curiosity:peer:brief` (via `persist_peer_brief`)
- Claude transport: `orion:room:claude:request` / `orion:room:claude:utterance`

## Run

```bash
# Build from a worktree; deploy (`up`) only from the clean primary main checkout:
scripts/safe_docker_build.sh orion-curiosity-peer build
scripts/safe_docker_build.sh orion-curiosity-peer up -d --build
```

`orion-room-companion` must be running for a hire to get an answer.

## Ask commitments and later decisions

`CURIOSITY_PEER_EPISODES_ENABLED` is true in the operator template and local
`.env` (the code fallback when unset is false). When enabled, eligible
requests need an explicit expectation and an acknowledged worldview commitment
before provider invocation. Duplicate claims do not resend. Deploy this consumer
before the updated Hub producer. See [contracts, rollout and rollback](../../docs/peer-ask-episodes.md).
