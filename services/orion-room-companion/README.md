# orion-room-companion

Runs Claude as a third participant in Orion's social room.

One bus consumer, one `claude -p` subprocess, one credential. It takes a
`RoomClaudeRequestV1` off `orion:room:claude:request`, produces one Claude
turn, and publishes a `RoomClaudeUtteranceV1` on
`orion:room:claude:utterance`. That is the whole service.

Design doc: `docs/architecture/hub-social-room-claude-companion.md`.

## Why this is a separate service

The obvious build puts the subprocess in `orion-hub`, next to the chat box it
serves. Don't. Audited live 2026-08-14, the Hub container runs as **root** and
already exposes readable `/root/.ssh`, `/root/.config/gh`,
`/var/run/docker.sock`, `/root/.fcc/.env`, and the repo mount — and Orion's own
FCC turns run *in that container* with `bypassPermissions` and full Bash.

So a Claude credential *stored* in Hub is one Orion can read directly.
Moving it here removes that, and removes the accident surface with it: no
config file in the repo tree, nothing in `/mnt/scripts` backups, nothing
graphify indexes, nothing a stray `docker inspect` on Hub reveals.

**What this does NOT do — stated plainly, because an earlier draft of this
README claimed otherwise.** This is not an enforcement boundary against a
Hub-resident agent. `services/orion-hub/docker-compose.yml:62` mounts
`/var/run/docker.sock` read-write and the Hub image ships the `docker` CLI, so
an FCC turn with Bash in Hub can simply run:

```bash
docker exec orion-athena-room-companion cat /root/.claude/.credentials.json
```

Verified live 2026-08-14: that returns the credential. Docker-socket access is
root-equivalent on the host, so no arrangement of *this* service's mounts can
close it.

Consequence for v2: a spend cap enforced by owning the credential would be
**advisory, not enforced**, for exactly as long as Hub holds the docker
socket. Do not build the v2 budget on the assumption that this container is a
wall.

Direction settled with Juniper 2026-08-14, so next phase does not re-derive it:

- **Keeping the socket.** It powers `scripts/service_logs.py` (the service-log
  viewer) and `api_routes.py:1128` (operator container bringup) — both live
  features. Dropping it is not free.
- **"Enforce budget somewhere Orion cannot reach" does not work either.** With
  the docker socket, a Hub-resident agent is root-equivalent on the host, so no
  software cap is enforceable wherever the logic lives. Real enforcement would
  mean moving Orion's FCC turns out of the socket-holding container — a much
  bigger change than this room warrants.
- **Next-phase shape: advisory cap + reconciliation.** This service is the only
  legitimate producer of room utterances, and
  `orion/dev_economics/claude_code_ingest.py` already reads
  `~/.claude/projects/*.jsonl`. Total observed spend can therefore be
  reconciled against metered spend and a gap surfaced. Detect; do not pretend
  to prevent.

What the separation genuinely buys: defense in depth, a much smaller accident
surface, and a container that is worth nothing to break into on its own — no
repo mount, no SSH keys, no gh config, no docker socket, no host networking,
no git, no gh. `tests/test_credential_isolation.py` fails if any of that
returns.

## Phase 2: the budget watchdog (design, not built)

v1 has no cap by design — meter only, to size the real number. v2 needs one,
and the enforcement has to live **outside Docker on a cadence** (Juniper,
2026-08-14). The reasoning matters more than the mechanism:

**The threat model is runaway autonomy, not an adversary.** The risk is a loop,
a retry storm, or an over-eager plan — not Orion strategically evading
controls. A host-side watchdog is genuinely effective against that, because a
runaway loop does not disable its own monitor. Against a deliberate adversary
holding Hub's docker socket nothing in software helps, and that is fine,
because that was never the real risk.

Shape: a root `systemd` timer (precedent: `deploy/systemd/`), oneshot, every
N minutes.

- **Ground truth**: the companion's own transcripts under
  `/mnt/docker/volumes/${PROJECT}-room-companion-claude/_data/projects/*.jsonl`.
  Root-only from the host, which a systemd unit already is. Reconcile against
  the metered `cost_usd` on `orion:room:claude:utterance`; a gap between them
  is out-of-band usage.
- **Cap state** lives in a host file the containers cannot write.
- **On breach**: `docker stop` the companion (immediate, reversible) and
  notify. Escalation: revoke the credential.

Three things that will bite whoever builds it:

1. **Moving the credential file on the host does NOT cut the container off.**
   Verified 2026-08-14: a bind mount holds the inode, so the container keeps
   reading the old file after a host-side `mv`. Only an in-place overwrite
   propagates. A "move the creds away" kill switch looks like it worked and
   does nothing. Use `docker stop`, in-place overwrite, or server-side
   revocation.
2. **Absence is not zero.** Missing or unreadable transcripts must read as
   *unknown spend*, never as *no spend* — an event-triggered stat cannot
   detect its own silence, and a watchdog that reads "quiet" during an outage
   is worse than none.
3. **Hysteresis on the threshold**, or it re-alerts forever once tripped.

And state the limit honestly: a cadence cap is **bounded overshoot, not a hard
ceiling**. Turns are serialized at ~3s and ~$0.03 each, so a 5-minute window
allows roughly $3 of overshoot before the watchdog can act. Size the interval
against that, and do not call it a hard cap.

**Worth reopening:** `claude setup-token` was rejected in phase 1 (see the
design doc) when the question was *how to authenticate*, where a tested mount
pattern beat a new one. Under a watchdog the question becomes *how to revoke*,
and a dedicated token's independent revocability — killing the room's access
without touching Juniper's own login — becomes the load-bearing property. That
earlier rejection answered a different question and should not be treated as
settled here.

## Credential isolation

Identical in shape and rationale to `orion-self-study-enrichment` — read that
service's "Credential isolation (safety requirement)" section too; it carries
the incident history.

- Authenticates as the operator's **already-logged-in Claude Code session**,
  reusing the existing subscription. Never `ANTHROPIC_API_KEY`, which would
  open a second pay-per-token billing relationship.
- `docker-compose.yml` bind-mounts **only** the host's `.credentials.json`
  file, read-only. Not the whole `~/.claude`, which also holds session
  transcripts, `history.jsonl`, and caches.
- `ROOM_COMPANION_CLAUDE_CREDENTIALS_HOST_PATH` has **no default in code** —
  compose uses `:?`, so a missing value fails loudly rather than silently
  mounting nothing and leaving the room mysteriously unauthenticated.
- `CLAUDE_CONFIG_DIR` must point at the directory that mount lands in. The two
  interpolations in `docker-compose.yml` must stay identical; if they diverge
  the credential becomes invisible to `claude` with no error.
- This service never writes to, logs, or echoes any part of that file.

`build_subprocess_env()` additionally strips `ANTHROPIC_BASE_URL` and
`ANTHROPIC_AUTH_TOKEN`. Those are what Hub's FCC lane
(`services/orion-hub/scripts/fcc_claude_bridge.py:262-263`) sets to redirect
`claude` at a **local** gateway. If either leaked in here, the room would keep
producing fluent text that was never Claude — the hardest failure in this
service to catch by eye, and why every utterance records the CLI's own
reported `model` and `cost_usd`.

### CONFIRMED BUG: the `:ro` credential mount goes stale on token refresh

Predicted on 2026-08-14 as "unconfirmed either way", then confirmed the same
day. **This breaks the room roughly every 7.5 hours, and it breaks
`orion-self-study-enrichment` identically.**

Claude Code refreshes the OAuth token by **atomic rename**, which creates a new
inode. A single-file bind mount is bound to the *inode*, not the path, so the
container keeps reading the old file forever. The old token is revoked
server-side at refresh, so every subsequent turn fails:

```text
Failed to authenticate. API Error: 401 OAuth access token has been revoked.
```

Observed directly:

```text
baseline (morning)   inode=5936136 mtime=1786697381   host == container
after refresh        host      inode=5936397 mtime=1786725885
                     container inode=5936136 mtime=1786697381   <- pinned, stale
```

`orion-self-study-enrichment` was silently in this state too, with the same
revoked-token error, and had been since the refresh.

**Band-aid**: `docker restart` re-resolves the mount to the current inode.
Verified — both containers came back on inode 5936397 and the smoke passed
again. It buys ~7.5 hours.

**Real fix, not yet applied** (needs a decision, see below): the single-file
`:ro` mount is *structurally* wrong for a credential that refreshes. Options:

1. **`claude setup-token`** — a long-lived token passed as an env var. It never
   refreshes, so there is no rename and nothing to go stale. This is the third
   time this option has surfaced as correct for a different reason; it was
   rejected in phase 1 when the question was "how do we authenticate", where a
   tested mount pattern beat a new one. That reasoning does not survive a
   credential that structurally cannot stay fresh.
2. **Mount the `~/.claude` directory** instead of the one file. Directory
   mounts track renames correctly. Costs the narrow exposure this pattern was
   specifically designed to preserve — transcripts, plugins and caches would
   enter the container.
3. **Re-copy the credential into `CLAUDE_CONFIG_DIR` before each turn**, from a
   directory-mounted source. Keeps turns fresh; still needs the directory
   mount, so it inherits option 2's exposure.

Until one is applied, treat a `401 ... has been revoked` in this service as
"the mount went stale", not "the operator logged out".

## Session continuity

One room is one durable Claude session. The first turn mints a uuid and passes
`--session-id`; every later turn passes `--resume`. The map lives in
`ROOM_COMPANION_SESSION_STATE_PATH` on this service's own volume.

That is what makes Claude a *participant* rather than a stateless helper, and
it means a turn sends only the new message — the transcript in the request is
first-turn seeding and a recovery path, not the memory mechanism.

If a `--resume` fails because the CLI no longer holds that session, the
mapping is dropped so the next turn can mint a fresh one. Without that, the
room would be wedged forever: the same resume would fail identically on every
retry.

## What Claude receives

Room-native context and nothing from Orion's interior. Specifically:

- The room framing (`SYSTEM_PROMPT` in `app/room_prompt.py`), re-sent as
  `--append-system-prompt` on **every** turn. Not just the first: `--resume`
  does not carry it, and a resumed turn without it reverts to Claude Code's
  default assistant persona (verified live). It also cannot be moved into the
  user-role text instead — Claude correctly refuses that as a prompt-injection
  pattern. System framing has to arrive as system framing.
- The roster and room continuity from the `social_memory` summary, passed
  through an **allowlist** (`_SUMMARY_KEYS`) rather than wholesale — this is a
  third-party-bound payload, so "everything we happened to have" is the wrong
  default.
- The new message, prefixed with who said it.

Not sent: recall results, self-state, drives, journals, mirrors. Tools are off
(`--tools ""`), so Claude cannot read a file, the repo, or the network.
`ROOM_COMPANION_SETTING_SOURCES` is `user` — deliberately not `user,local` —
so a room conversation does not inherit this repo's own development contract
as if it were room context.

## Run

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-room-companion/.env \
  -f services/orion-room-companion/docker-compose.yml \
  up -d --build
```

Prefer `scripts/safe_docker_build.sh orion-room-companion up -d --build` — it
refuses to run from the shared checkout.

### Live smoke

The phase-1 acceptance surface. Runs the real subprocess against the real
credential and prints the real utterance — no UI, no Hub:

```bash
docker compose -f services/orion-room-companion/docker-compose.yml \
  exec room-companion python -m scripts.smoke_room_turn
```

Two turns, because one cannot demonstrate continuity: turn 2 resumes and is
asked to recall a token only turn 1 was told. It exits non-zero unless both
turns produced text, turn 2 recalled the token, both share one Claude session,
cost was metered above zero, and a Claude model served it. An empty or free
turn is a failure, not a warning.

## Tests

```bash
cd services/orion-room-companion
PYTHONPATH=<repo-root>:. pytest tests -q
```

`tests/test_credential_isolation.py` is the load-bearing one — it asserts on
the parsed compose file, not on Python settings, and it has been
mutation-tested against the real file (adding a docker.sock mount or widening
the credential bind to a directory both make it fail).
