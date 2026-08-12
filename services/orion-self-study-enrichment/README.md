# orion-self-study-enrichment

Thin, isolated service that generates real, evidence-grounded "what is this
and why" prose summaries for architectural clusters touched by a real
commit, using an actual `claude -p` subprocess authenticated as the host's
already-logged-in Claude Code CLI session (not a separate API key -- see
"Credential isolation" below). Producer side for a self-study consumer
capability that does not exist yet -- see "Fast-follow: self_study.py
consumer" below.

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

The real fix: Claude Code CLI resolves its login credential from
`$CLAUDE_CONFIG_DIR/.credentials.json` if `CLAUDE_CONFIG_DIR` is set, else
`~/.claude/.credentials.json`. `app/main.py`'s `handle_request_payload` sets
`CLAUDE_CONFIG_DIR` in the subprocess env to
`SELF_STUDY_ENRICHMENT_CLAUDE_CONFIG_DIR` (default
`/root/.claude` -- container-side, matching the container's real default
`HOME` since the Dockerfile runs as root with no `USER` directive), and
`docker-compose.yml` bind-
mounts **only** the host's `~/.claude/.credentials.json` file -- read-only,
matching its real 600/owner-only host perms -- into that directory at
`.credentials.json`. The rest of the host's real `~/.claude/` (history,
plugins, session transcripts, `stats-cache.json`, `gh-pr-status-cache.json`,
etc) is never mounted into the container; only the one credential file is.
This service never writes to, logs, or echoes any part of that file.

This is a real, sensitive credential tied to the operator's actual logged-
in account/subscription -- arguably a bigger blast radius than a scoped API
key would have been, which is exactly why the mount is a single read-only
file bind, not a directory mount, and why `SELF_STUDY_ENRICHMENT_CLAUDE_CREDENTIALS_HOST_PATH`
has no safe default baked into code (operator sets it explicitly in their
own local `.env`, same as any other host-path override in this repo).

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
