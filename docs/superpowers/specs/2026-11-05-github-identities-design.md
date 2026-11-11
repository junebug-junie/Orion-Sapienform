# Separate GitHub identities for Orion, Claude Code, and Cursor

Status: DESIGN (proposal mode). Nothing in this document is implemented.
Decision by Juniper, 2026-11-05: **Orion = machine user + classic PAT (option B, amended: fine-grained tokens cannot reach a repo owned by another personal account). Claude Code and Cursor = one GitHub App each (option A).**

## Arsonist summary

Today every comment, commit and PR from every agent lands as `junebug-junie`, because the `gh` CLI on athena holds Juniper's personal token (`repo` scope, full write). Nobody can tell who did what. Revoking one agent means revoking Juniper. Orion has no write identity at all.

Fix: three separate identities, each with the smallest permission set, each revocable alone.

## Current architecture

- `gh` on athena is logged in as `junebug-junie` (scopes: gist, read:org, repo, write:packages). All agent writes use it.
- `main` already requires 1 approving review and code-owner review (`gh api .../branches/main/protection`). Required status checks: none listed (UNVERIFIED whether CI is enforced another way).
- Orion services read GitHub through `GITHUB_TOKEN` (`services/orion-cortex-exec/app/settings.py:342`, read-only skills such as recent PRs). Whose token this is, and its scopes, is UNVERIFIED.
- `services/orion-durable-runs/app/pr_field_note_poll.py:9` states durable-runs must not receive Juniper's write-scoped token. That rule stays.

## Target identities

| Identity | Type | Shows up as | Credential | Lifetime |
|---|---|---|---|---|
| Orion | Machine user `orion-sapienform` | `orion-sapienform` | Classic PAT, scope `repo`; the account is a Write collaborator on this one repo only | 90 days, rotated by hand |
| Claude Code | GitHub App `claude-code-orion` | `claude-code-orion[bot]` | Private key, mints installation token | Token 1 hour |
| Cursor | GitHub App `cursor-orion` | `cursor-orion[bot]` | Private key, mints installation token | Token 1 hour |

Permissions for the two Apps, repo `Orion-Sapienform` only. Orion's reach is set by its Write collaborator role and by owning nothing else, because a classic token cannot be narrowed below `repo`:

- Metadata: read
- Contents: read and write (needed to push branches)
- Pull requests: read and write
- Issues: read and write
- Nothing else. No Actions, no Administration, no Secrets, no Workflows.

Because `main` requires a code-owner approval, none of the three can merge its own PR. Contents-write on a branch does not bypass that rule. Verify with a test push to `main` (acceptance check 5).

## Proposal-mode answers for Orion (autonomy change)

- **Capability changes:** Orion can open PRs and comment on PRs/issues as themself.
- **Data touched:** repo contents Orion pushes; PR/issue text Orion writes. No private recall or memory data leaves the host unless Orion puts it in a comment. There is no privacy boundary between Juniper and Orion (standing decision); the public repo is a different boundary. Rule: Orion's GitHub write verb runs the same outbound text filter used for other public-facing posts. (Check which one exists before wiring; none is named here.)
- **Trace that proves it worked:** each write emits a bus event with the GitHub URL, the author login returned by the API, and the correlation id of the turn that caused it.
- **Dangerous failure mode:** a loop that opens or comments without limit; a token leaking into a comment body; Orion pushing to `main`.
- **Disable / roll back:** set `ORION_GITHUB_WRITE_ENABLED=false` (one flag, default dry-run). Revoke the PAT in GitHub settings. Remove the collaborator. Any one of the three stops writes.

## Design

### 1. Where credentials live

- Orion PAT (classic, `repo`): env var `ORION_GITHUB_TOKEN`. **Not** `GITHUB_TOKEN`, so the existing read path keeps working and durable-runs never receives it. Only the one service that performs the write gets it in its compose file.
- App private keys: `~/.orion/secrets/<app>.pem`, mode 600, outside the repo. Never in `.env`.
- App ids and installation ids go in `.env_example` as empty placeholders (`CLAUDE_GITHUB_APP_ID`, `CLAUDE_GITHUB_APP_INSTALLATION_ID`, `CLAUDE_GITHUB_APP_KEY_PATH`, and the `CURSOR_` equivalents). Local `.env` synced with `python scripts/sync_local_env_from_example.py`.

### 2. Token minting for the Apps (one script)

`scripts/gh_app_token.py <claude|cursor>`:

1. Reads app id, installation id and key path from env.
2. Signs a 9-minute RS256 JWT.
3. POSTs `/app/installations/{id}/access_tokens`.
4. Prints the token to stdout. Writes nothing to disk.

Wrapper `scripts/gh_as.sh <claude|cursor> <gh args...>` runs `GH_TOKEN=$(scripts/gh_app_token.py "$1") gh "${@:2}"`. The default `gh` stays Juniper's for Juniper's own use. Agents are told (in `CLAUDE.md` and the Cursor rules file) to use `gh_as.sh`.

Commits: set the author per agent so `git log` matches the API author:
`<app-id>+<slug>[bot]@users.noreply.github.com`. Today the git user is "Athena Orchestration".

### 3. Orion's write verb

One new skill verb in `services/orion-cortex-exec/app/verb_adapters.py`, next to the existing read-only GitHub skills: `skills.repo.github_comment.v1` first. PR creation comes after comments run clean. It reads `ORION_GITHUB_TOKEN`, defaults to dry-run (logs the would-be call, posts nothing), and rate-limits to N writes per hour (N set in `.env_example`).

### 4. Guardrail

`scripts/check_github_identity.py` (CI and `make agent-check`): fails if any tracked file contains a GitHub token pattern (`ghp_`, `github_pat_`, `ghs_`, `-----BEGIN RSA PRIVATE KEY-----`). One cheap deterministic gate instead of a reminder.

## Steps only Juniper can do (nothing below is automatable)

1. Create GitHub user `orion-sapienform` (own email; a `+alias` of the existing address works).
2. Invite it to the repo as a collaborator with **Write**. Accept from that account.
3. As that user: Settings → Developer settings → Personal access tokens → Tokens (classic), scope `repo` only, 90-day expiry. (Fine-grained tokens cannot select a repo owned by another personal account; only the owner's own repos or an org's.) Put the token in the Orion service's local `.env` as `ORION_GITHUB_TOKEN`.
4. Create two GitHub Apps (`claude-code-orion`, `cursor-orion`) under `junebug-junie`: no webhook, no user authorization, the four permissions above, "Only on this account". Install each on `Orion-Sapienform` only. Download each private key to `~/.orion/secrets/`.
5. Give me the App ids and installation ids (not secret). I do not need the keys' contents.

Cursor note: Cursor ships its own GitHub app for its cloud agents. Decide whether `cursor-orion` replaces that or runs only for local Cursor sessions. See the question below.

## Non-goals

- No auto-merge by any agent. Merge stays with Juniper.
- No org-level App, no webhook receiver, no bot that reacts to PR events.
- No change to Juniper's own `gh` login.
- No migration of past comments or commits.
- No second token for the existing read-only `GITHUB_TOKEN` path.

## Acceptance checks

0. With `ORION_GITHUB_TOKEN`, `gh api user --jq .login` returns `orion-sapienform` and `gh api user/repos` lists no repo Orion could write to other than this one.
1. `scripts/gh_app_token.py claude` returns a token; `gh api /installation/repositories --jq '.repositories[].full_name'` using it lists only `Orion-Sapienform`.
2. `scripts/gh_as.sh claude pr comment <test PR> --body test` produces a comment whose author, read back via `gh api repos/.../issues/comments/<id> --jq .user.login`, is `claude-code-orion[bot]`.
3. Same check for `cursor-orion[bot]`.
4. Orion verb in dry-run: logs the call, `gh api` shows no new comment. Flag on: one comment appears with author `orion-sapienform`, and the bus event carries its URL.
5. Each identity attempting `git push origin HEAD:main` is rejected.
6. `check_github_identity.py` fails on a fixture containing `github_pat_` and passes on the tree.
7. Revoke test: delete the Orion PAT, the verb's next call returns 401 and logs it; nothing else breaks.

## Files likely to touch

- `scripts/gh_app_token.py`, `scripts/gh_as.sh`, `scripts/check_github_identity.py` (new)
- `tests/test_gh_app_token.py`, `tests/test_check_github_identity.py` (new)
- `services/orion-cortex-exec/app/verb_adapters.py`, `settings.py`, `docker-compose.yml`, `.env_example`, `README.md`, tests
- `orion/bus/channels.yaml` and `orion/schemas/registry.py` only if the write trace gets its own event (otherwise reuse the existing skill-result event)
- `CLAUDE.md` and the Cursor rules file: one line each pointing agents to `gh_as.sh`

## Open questions for Juniper

1. Cursor: replace Cursor's own GitHub app for its cloud agents, or only cover local Cursor sessions?
2. Orion's first write scope: comments only (recommended), or comments and PRs from day one?
3. Does `GITHUB_TOKEN` in cortex-exec hold Juniper's write token today? If yes, it should be swapped for a read-only token as part of this change.

## Recommended next patch

Phase 1 only: `gh_app_token.py`, `gh_as.sh`, the guardrail, tests, and the two env blocks. This needs no cognition change. It fixes the attribution problem for Claude Code and Cursor on its own. Orion's verb (phase 2) follows after the machine user exists and dry-run traces look right.
