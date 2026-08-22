# orion-self-study-enrichment

Thin, isolated service that generates real, evidence-grounded "what is this
and why" prose summaries for architectural clusters touched by a real
commit, using an actual `claude -p` subprocess authenticated via a
long-lived OAuth token against the operator's own Claude subscription (not
a separate API key -- see "Credential isolation" below for the exact
mechanism). Producer side for a self-study consumer capability that does
not exist yet -- see "Fast-follow: self_study.py consumer" below.

## Why this exists

`services/orion-cortex-exec/app/self_study.py`'s Layer 2
(`induce_self_concepts()`) clusters Orion's authoritative self-model facts
into named concepts, but is fully hardcoded to ~5 fixed service/channel
names and has zero semantic understanding of *what* a cluster is *for* --
only structural metadata, no real "what is this" prose. This service
produces that prose, grounded only in real evidence (never free-associated
from general repo knowledge -- see "Evidence-only prompting" below), and
caches it for a low-cost future consumer to read.

## Trigger

`scripts/git_hooks/post-commit` (installed by
`scripts/install_git_safety_hooks.sh`) runs
`scripts/self_study_enrichment_hook.py` after every real commit. That script:

1. Computes a real `git_churn_delta` (`orion/structural_mass/git_delta.py`)
   between the last-enriched SHA (state file, cold-starts to `HEAD~1`) and
   the new `HEAD`.
2. Checks whether any changed path matches self-study-relevant patterns
   (`services/`, `orion/bus/channels.yaml`, `orion/schemas/`,
   `orion/cognition/verbs/` -- mirrored from `self_study.py`'s own scan
   surface).
3. If nothing qualifies: does nothing, exits 0.
4. If something qualifies: applies a cheap hook-side publish-count belt
   check (`SELF_STUDY_ENRICHMENT_MAX_PER_DAY`, default 8/day -- shared
   default with the service's own authoritative ceiling below, not the same
   enforcement point), then publishes a `SelfStudyEnrichmentRequestV1` event
   on `orion:self_study:enrichment:requested` via a one-shot
   `redis.Redis(...).publish(...)` call (no existing lightweight one-shot-
   publish-from-shell helper was found in this repo; this is new and
   deliberately minimal, mirroring how `agent_board.py`'s heartbeat is
   invoked from the same hook family).

The hook fragment itself never blocks or fails the commit -- see
`scripts/git_hooks/post-commit`'s own comments.

## What this service does on a qualifying event

1. **Evidence bundle assembly** (`app/evidence.py`): graphify structural
   nodes for the affected cluster(s) (read directly from
   `graphify-out/graph.json` -- see the module docstring for why a direct
   file read was chosen over shelling out to `graphify query`/`explain`),
   the structural_mass delta that triggered the run, and any nearby
   `README.md` text already in the repo for the touched cluster
   directories (cheap file read, not another LLM call).
2. **One real `claude -p` subprocess call** (`app/claude_runner.py`):
   `--output-format json` (non-streaming), model `claude-sonnet-5`, effort
   `medium` by default, `--tools ""` (no tool use at all -- this is a
   one-shot evidence-in/prose-out call, not an agentic turn). Reuses
   `orion/fcc/claude_spawn.py`'s `setting_sources_argv`/
   `claude_permission_argv` helpers, but does **not** reuse
   `orion/harness/fcc_motor.py`'s interactive-turn machinery (streaming
   stream-json, MCP config, tool permissions) -- none of that applies here.
3. **Content-hash-keyed disk cache** (`app/cache.py`): mirrors
   `graphify-out/cache/semantic/`'s directory-tree shape, lives on this
   service's own writable Docker volume (not the read-only repo mount),
   gitignored, never committed.

### Evidence-only prompting

The prompt built by `evidence.render_evidence_prompt()` states, as a hard
requirement, that the model must synthesize only from the evidence block and
must say so explicitly rather than filling a gap with general knowledge.
This mirrors self_study.py's own evidence-grounding design principle and
CLAUDE.md's "no empty-shell cognition" rule -- an ungrounded "what is this
for" summary is worse than none.

## Credential isolation (safety requirement)

**Corrected 2026-08-12 -- see the PR report for the full story of what was
wrong.** This service does NOT use `ANTHROPIC_API_KEY` and never should.
The original version of this patch authenticated the `claude -p` subprocess
via a service-local `ANTHROPIC_API_KEY` (a separate, pay-per-token Anthropic
API billing relationship). That was a real mistake, not a style choice: the
entire point of wiring in a `claude -p` subprocess was to reuse the
operator's **already-logged-in Claude Code CLI session** -- the same
subscription auth this repo's own dev sessions run under -- not to open a
second billing relationship. Any `ANTHROPIC_API_KEY` reference anywhere in
this service (code, `.env_example`, `settings.py`, `docker-compose.yml`) is
a regression; `tests/test_main.py::test_claude_subprocess_env_has_no_api_key`
guards against it silently coming back.

**FIXED 2026-08-21.** The credential mechanism above (bind-mounting the
host's `~/.claude/.credentials.json` into the container, read-only) carried
a real bug shared with every service that used the same pattern: Claude
Code's login credential refreshes internally on roughly a 7.5h cycle, and
nothing ever re-wrote the bind-mounted file inside the container after that
happened, so the mount silently went stale and calls started failing with
auth errors that looked nothing like "the credential expired." First
confirmed and fixed in `orion-room-companion` on 2026-08-18 (see that
service's README for the incident); this service inherited the identical
bug from the same design and gets the identical fix here.

The fix: `claude -p` now authenticates via `CLAUDE_CODE_OAUTH_TOKEN`, a
long-lived (1-year) token from `claude setup-token`
(`SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN` in `.env` -- `SecretStr` in
`settings.py`, so it never appears in a repr/log by accident).
`app/main.py`'s `build_subprocess_env()` injects it into the subprocess env
explicitly from `Settings`, the same pattern `CLAUDE_CONFIG_DIR` already
used. This is deliberately not just a denylist: the subprocess env is built
from an **allowlist** (`_ENV_ALLOWLIST`) of generic passthrough names plus a
belt-and-braces `_ENV_DENY_PREFIXES` covering `ANTHROPIC_`,
`CLAUDE_CODE_OAUTH`, `AWS_`, `GOOGLE_`, `GCP_` -- so neither
`SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN` (this container's own ambient
copy of the secret, present because `docker-compose.yml` has to pass it in
via `environment:` for `Settings` to read it at all) nor an ambient
`ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` (what orion-hub's FCC lane sets
to redirect `claude` at a local gateway -- see
`services/orion-hub/scripts/fcc_claude_bridge.py`) can reach the subprocess.
The real secret exists exactly once in the subprocess env, under the
literal `CLAUDE_CODE_OAUTH_TOKEN` name `claude` actually reads -- an earlier
version of this fix used a plain `dict(os.environ)` copy plus a single-key
`ANTHROPIC_API_KEY` denylist, which a live-code review caught leaking all
four of the above; `orion-room-companion` had already hit and fixed the
identical gap on 2026-08-18, and this service now mirrors that fix exactly
rather than repeating the mistake.

`docker-compose.yml` no longer bind-mounts anything credential-shaped at
all; the token is `${VAR:?err}`-guarded so an unset value fails the
container at startup instead of silently starting with no auth. Rotating
the token needs `docker compose up -d` (recreate), not `restart` -- Compose
resolves `${SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN:?...}` into the
container's environment once, at container-creation time; `restart` reuses
the already-created container and its already-frozen env, it does not
re-read `.env` or re-interpolate anything (this is a Compose-level fact,
independent of `Settings` also being `@lru_cache`-d and read once at
process start inside the container). Revoke at
https://claude.ai/settings/claude-code (no CLI revoke command exists as of
this writing).

This is still a real, sensitive credential tied to the operator's actual
account/subscription -- the OAuth token form doesn't change that, only how
it's transported. It's why `SELF_STUDY_ENRICHMENT_CLAUDE_OAUTH_TOKEN` has
no safe default baked into code (operator sets it explicitly in their own
local `.env`, same as any other secret in this repo) and why it's
`SecretStr`, not a plain `str`, on the settings model.

`orion-cortex-exec` and `orion-cortex-orch` are not touched by this patch
at all except (possibly) channel/schema registration in
`orion/bus/channels.yaml` / `orion/schemas/registry.py`, which are
repo-wide contract files, not credential surfaces -- neither service gains
any new capability or credential from this patch. Orion's own FCC gateway
(`orion/fcc/`, `orion/harness/fcc_motor.py`) is untouched; this service
spawns its own independent `claude -p` process authenticated via the same
underlying Claude Code login credential FCC's own `claude_spawn.py`-adjacent
tooling assumes is already present on the host, not routed through FCC's
internal model gateway.

## Safety backstop

`app/rate_limit.py` is the authoritative daily ceiling on real `claude -p`
runs (`SELF_STUDY_ENRICHMENT_MAX_PER_DAY`, default 8/day) -- separate from
the hook-side belt check, since only this service knows how many runs it
actually executed today. Once the ceiling is hit, the service logs and
skips (never crashes) for the rest of the UTC day.

## Fast-follow: self_study.py consumer

**Not implemented in this patch, disclosed per the feature spec.**
`self_study.py`'s `induce_self_concepts()` does not read this service's
cache yet. Producer side shipped solid; wiring a consumer that reads
`graphify-out/cache/self_study_enrichment/...`-shaped entries (well, this
service's own `/data/cache/self_study_enrichment/` volume -- not the
graphify cache dir itself, a separate but shape-mirroring location) into a
Layer 2 concept's `description` field is a follow-up patch.

## Local dev

```bash
python scripts/sync_local_env_from_example.py
scripts/safe_docker_build.sh orion-self-study-enrichment config
scripts/safe_docker_build.sh orion-self-study-enrichment build
```

## Tests

```bash
pytest services/orion-self-study-enrichment/tests -q
```
