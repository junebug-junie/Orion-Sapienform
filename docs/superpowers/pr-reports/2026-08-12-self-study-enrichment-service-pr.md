# orion-self-study-enrichment: claude -p producer/consumer for self_study concept induction

## Summary

- New service `services/orion-self-study-enrichment`: subscribes to `orion:self_study:enrichment:requested`, assembles a real evidence bundle (graphify structural nodes + `structural_mass` git delta + nearby README text), spawns a one-shot `claude -p` subprocess (model `claude-sonnet-5`, `--tools ""`, no MCP, no tool use) to synthesize an evidence-grounded "what is this and why" summary, and caches results content-hash-keyed on a dedicated writable Docker volume.
- New bus channel `orion:self_study:enrichment:requested` + `SelfStudyEnrichmentRequestV1` schema registered in `orion/schemas/registry.py` and `orion/bus/channels.yaml`.
- `scripts/git_hooks/post-commit` gets a new append-only, self-contained subshell fragment that runs `scripts/self_study_enrichment_hook.py` after every real commit: computes a real `git_churn_delta` since the last enrichment run, checks whether any self-study-relevant path was touched, and one-shot publishes via `redis.Redis(...).publish(...)` if so. Never blocks or fails the commit.
- Safety backstop: authoritative daily ceiling (default 8/day, env-configurable) enforced service-side (`app/rate_limit.py`), plus a cheap hook-side belt check.
- Real Anthropic credentials isolated to this service's own `.env`/`.env_example` only — `orion-cortex-exec`/`orion-cortex-orch` untouched.
- `self_study.py`'s `induce_self_concepts()` does **not** read this service's cache yet — explicitly disclosed fast-follow, not shipped in this patch.

## Outcome moved

`services/orion-cortex-exec/app/self_study.py`'s Layer 2 (`induce_self_concepts()`) had zero semantic understanding of *what* a cluster is *for* — only structural metadata, no real "what is this" prose, and hardcoded to ~5 fixed service/channel names. This patch ships the producer side of a real, evidence-grounded prose-generation capability for that gap: a triggered-by-real-commits, cost-bounded, credential-isolated `claude -p` call that writes grounded summaries to a content-hash-keyed cache.

## Current architecture

Before this patch: `self_study.py` had no mechanism to generate semantic ("what is this for") content at all — only structural fact extraction and rule-based clustering (`induce_self_concepts()`'s five hardcoded concept blocks). No bus channel, schema, or service existed for LLM-generated architectural summaries.

## Architecture touched

- New service: `services/orion-self-study-enrichment` (bus consumer, no HTTP surface).
- New bus channel: `orion:self_study:enrichment:requested` (`orion/bus/channels.yaml`).
- New schema: `SelfStudyEnrichmentRequestV1` (`orion/schemas/self_study_enrichment.py`, registered in `orion/schemas/registry.py`).
- New host-side hook script: `scripts/self_study_enrichment_hook.py`, invoked from `scripts/git_hooks/post-commit`.
- No changes to `orion-cortex-exec`, `orion-cortex-orch`, or any other existing service's runtime.

## Files changed

- `orion/bus/channels.yaml`: registered `orion:self_study:enrichment:requested` (producer `orion-git-hooks`, consumer `orion-self-study-enrichment`, `single_consumer: true`, `stability: experimental`).
- `orion/schemas/registry.py`: imported and registered `SelfStudyEnrichmentRequestV1`.
- `orion/schemas/self_study_enrichment.py`: new — the request schema.
- `scripts/git_hooks/post-commit`: appended a new, independently-subshelled fragment that invokes the hook script; never blocks the commit.
- `scripts/self_study_enrichment_hook.py`: new — deterministic qualifying-change check, `git_churn_delta` reuse, state file, hook-side belt rate-limit, one-shot bus publish.
- `scripts/test_self_study_enrichment_hook.py`: new — 8 tests (path qualification, real-git-repo delta, state roundtrip, rate limit).
- `services/orion-self-study-enrichment/`: new service — `README.md`, `.env_example`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `app/{settings,main,evidence,claude_runner,cache,rate_limit}.py`, `tests/{test_evidence,test_cache,test_rate_limit,test_claude_runner}.py`.

## Schema / bus / API changes

- **Added:** channel `orion:self_study:enrichment:requested` (kind `request`, schema `SelfStudyEnrichmentRequestV1`, `message_kind: self_study.enrichment.request.v1`). Producer service name in the channel catalog is `orion-git-hooks` — a documented pseudo-producer name for the host-side hook script, since it is not itself a containerized Orion service; no existing convention in `channels.yaml` covers a script/host producer, so this is a new (disclosed) naming choice.
- **Added:** schema `SelfStudyEnrichmentRequestV1` — `repo_root`, `prev_sha`, `head_sha`, `commit_count`, `files_changed`, `lines_changed`, `touched_paths` (tuple of repo-relative paths that qualified), `requested_at`.
- **Removed:** none.
- **Renamed:** none.
- **Behavior changed:** none (new capability only).
- **Compatibility notes:** `single_consumer: true` — only `orion-self-study-enrichment` should ever consume this channel.

## Env/config changes

- **Added keys** (`services/orion-self-study-enrichment/.env_example`, empty/safe placeholders per CLAUDE.md sec 7): `SERVICE_NAME`, `SERVICE_VERSION`, `SELF_STUDY_ENRICHMENT_NODE_NAME`, `ORION_BUS_ENABLED`, `ORION_BUS_ENFORCE_CATALOG`, `HEARTBEAT_INTERVAL_SEC`, `CHANNEL_SELF_STUDY_ENRICHMENT_REQUESTED`, `SELF_STUDY_ENRICHMENT_REPO_HOST_PATH`, `SELF_STUDY_ENRICHMENT_REPO_PATH`, `SELF_STUDY_ENRICHMENT_GRAPH_JSON_PATH`, `ANTHROPIC_API_KEY` (empty placeholder — real key operator-supplied, isolated to this service only), `SELF_STUDY_ENRICHMENT_CLAUDE_BIN`, `SELF_STUDY_ENRICHMENT_MODEL`, `SELF_STUDY_ENRICHMENT_EFFORT`, `SELF_STUDY_ENRICHMENT_TIMEOUT_SEC`, `SELF_STUDY_ENRICHMENT_SETTING_SOURCES`, `SELF_STUDY_ENRICHMENT_CACHE_DIR`, `SELF_STUDY_ENRICHMENT_MAX_PER_DAY`, `SELF_STUDY_ENRICHMENT_RATE_LIMIT_STATE_PATH`.
- **Removed keys:** none.
- **Renamed keys:** none.
- **`.env_example` updated:** yes (new file, new service).
- **local `.env` synced with `python scripts/sync_local_env_from_example.py`:** ran clean; the new service has no existing `.env` in the primary checkout yet (it's a brand-new, unmerged service), so there is nothing to sync there until this PR merges. No divergences reported for this service's own keys. Existing pre-existing divergences for other services (`orion-memory-consolidation`, `orion-hub`, `orion-field-digester`, `orion-harness-governor`) were reported by the sync script but are unrelated to this patch and were left untouched.
- **Skipped keys requiring operator action:** `ANTHROPIC_API_KEY` — operator must set a real Anthropic API key in `services/orion-self-study-enrichment/.env` for this service specifically. Not shared with FCC's own credentials.

## Tests run

```text
$ .venv/bin/python -m pytest services/orion-self-study-enrichment/tests scripts/test_self_study_enrichment_hook.py -q
..............................                                           [100%]
30 passed in 0.48s
```

Breakdown: `test_evidence.py` (9, incl. the review-driven regression test), `test_cache.py` (4), `test_rate_limit.py` (3), `test_claude_runner.py` (6, subprocess mocked — no real API credits spent), `test_self_study_enrichment_hook.py` (8, incl. real temp git repos for `git diff`/`git_churn_delta`).

## Evals run

No eval harness exists for this new service — this is dev-tooling-adjacent infrastructure (a producer/consumer pair with deterministic qualifying logic and a mocked subprocess boundary), not a scored cognition capability, so an eval harness doesn't apply cleanly yet. Follow-up: once `self_study.py`'s consumer side is wired (see Concerns below), an eval could grade generated-summary groundedness against the evidence bundle.

## Docker/build/smoke checks

```text
$ docker compose --env-file <root> --env-file services/orion-self-study-enrichment/.env \
    -f services/orion-self-study-enrichment/docker-compose.yml config
# validated cleanly -- full rendered service block confirmed

$ docker compose --env-file <root> --env-file services/orion-self-study-enrichment/.env \
    -f services/orion-self-study-enrichment/docker-compose.yml build
# succeeded: installed git, Node 20, @anthropic-ai/claude-code, python deps

$ docker run --rm orion-self-study-enrichment-self-study-enrichment python -c "..."
imports OK
['claude', '-p', 'hi', '--output-format', 'json', '--model', 'claude-sonnet-5', '--tools', '', '--effort', 'medium', '--setting-sources', 'user,local', '--permission-mode', 'dontAsk']

$ docker run --rm orion-self-study-enrichment-self-study-enrichment claude --version
2.1.197 (Claude Code)
```

Root `.env` was not present in this worktree (git-ignored, per-worktree); built a minimal equivalent (`PROJECT`, `NODE_NAME`, `NET`, bus URL, etc.) purely to run `config`/`build` — no real deploy attempted, no `up -d`, no real Anthropic credential exercised. Also live-verified `orion.schemas.registry.resolve("SelfStudyEnrichmentRequestV1")` resolves and round-trips a real Pydantic instance.

## Review findings fixed

- Finding: `evidence.py`'s `load_graph_nodes_for_clusters()` used plain substring containment (`cluster in haystack`), so a cluster like `services/orion-hub` matched unrelated nodes such as `services/orion-hub-analytics/...` or any node whose free-text label field happened to contain the substring anywhere — undermining the "every claim traceable to a specific evidence item" design goal.
  - Fix: added `_cluster_matches_field()` requiring a path-segment boundary (`/`, string start/end, or a non-alphanumeric separator) on both sides of the match.
  - Evidence: new regression test `test_load_graph_nodes_for_clusters_does_not_match_similarly_prefixed_cluster` (`services/orion-self-study-enrichment/tests/test_evidence.py`) — passes; would have failed against the pre-fix implementation.
- Finding: `scripts/self_study_enrichment_hook.py`'s hook-side rate-limit belt check (`_rate_limit_ok()`) persisted an incremented daily counter *before* `git_churn_delta()` ran. If that call raised (corrupted/shallow clone, `prev_sha` unreachable after a rebase/force-push), `main()`'s outer `except` would swallow it and exit 0 having already spent a rate-limit slot on nothing — repeated failures could silently exhaust the day's publish ceiling with zero real events ever published.
  - Fix: reordered so `git_churn_delta()` (fallible, no persistent side effect) runs first; `_rate_limit_ok()` (has a persistent side effect) only runs once a real payload is known to exist.
  - Evidence: `scripts/test_self_study_enrichment_hook.py` still passes (8/8); ordering change is structural and covered indirectly by the existing rate-limit tests plus the real-git-repo delta test exercising the new code path.

## Restart required

No restart required for existing services — this patch adds a new service and a new hook fragment; nothing existing changes behavior until deployed.

To deploy the new service once merged:

```bash
python scripts/sync_local_env_from_example.py
# then fill in services/orion-self-study-enrichment/.env's ANTHROPIC_API_KEY
scripts/safe_docker_build.sh orion-self-study-enrichment build
scripts/safe_docker_build.sh orion-self-study-enrichment up -d
```

To activate the git-hook trigger on a given checkout:

```bash
scripts/install_git_safety_hooks.sh .
```

## Risks / concerns

- Severity: low
  - Concern: `self_study.py`'s Layer 2 (`induce_self_concepts()`) does not read this service's cache yet — the producer/consumer pair for enrichment generation is complete and tested, but there is no live consumer wiring the generated prose into a self-study concept's `description` field.
  - Mitigation: explicitly disclosed as intended scope for this patch (per the task spec: "ship the producer side solid rather than rush a shallow consumer"). Follow-up patch: read `SELF_STUDY_ENRICHMENT_CACHE_DIR` entries keyed by evidence-prompt content hash and surface them as an additional, clearly-labeled (non-authoritative, `induced`-trust-tier) field on the relevant concept.
- Severity: low
  - Concern: producer service name `orion-git-hooks` in `channels.yaml` is a documented pseudo-producer (a host-side script, not a containerized service) — no existing repo convention covers this case.
  - Mitigation: disclosed here; low risk since `channels.yaml`'s `producer_services` list is documentation/contract metadata, not runtime-enforced against actual service identities anywhere found in this repo.
- Severity: low
  - Concern: no eval harness for this service (see Evals run above).
  - Mitigation: follow-up once the consumer side exists, per above.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1574

---

## Corrected 2026-08-12: auth mistake found after merge, fixed same day

**What was wrong.** The version of this service that merged as PR #1574
authenticated its `claude -p` subprocess via a service-local
`ANTHROPIC_API_KEY` (`app/settings.py`, `app/main.py`, `.env_example`,
`docker-compose.yml` all carried it; see the original "Env/config changes"
and "Credential isolation" sections above, left unedited for the record).
That is a real, direct mistake, not a style nit: the entire stated point of
wiring in a `claude -p` subprocess for this capability -- across the whole
design conversation that led to this PR -- was to reuse the operator's
**already-logged-in Claude Code CLI session** (the same subscription-based
auth this repo's own dev sessions run under), not to open a second,
separate pay-per-token Anthropic API billing relationship under a brand-new
key. Juniper flagged this directly and was upset it shipped this way;
disclosing it here plainly rather than quietly rewriting the PR's history,
per this repo's own honesty norms (CLAUDE.md "Runtime truth beats config
truth").

By the time this was caught, PR #1574 had already merged and its branch
had been deleted (GitHub does not allow pushing new commits to a closed/
merged PR), so the fix landed as a same-day follow-up branch/PR
(`fix/self-study-enrichment-claude-auth`) rather than as additional commits
on the original branch.

**What changed in the fix.**

- Removed `ANTHROPIC_API_KEY` entirely: no field in `app/settings.py`, no
  env injection in `app/main.py`'s `handle_request_payload`, no key in
  `.env_example` or `docker-compose.yml`. No dual-mode, no fallback.
- `app/main.py` now sets `CLAUDE_CONFIG_DIR` in the subprocess env
  (defaulting to `SELF_STUDY_ENRICHMENT_CLAUDE_CONFIG_DIR`, container-side
  `/root/.claude` since the Dockerfile runs as root with no `USER`
  directive) and defensively `env.pop("ANTHROPIC_API_KEY", None)`s any
  value that might otherwise leak in via `dict(os.environ)`.
  `CLAUDE_CONFIG_DIR` is Claude Code's own real env var for relocating
  where it resolves `.credentials.json` (default `~/.claude` if unset --
  confirmed against Claude Code's own documented behavior, not guessed).
- `docker-compose.yml` bind-mounts **only** the host's real
  `~/.claude/.credentials.json` file -- read-only, a single-file mount, not
  the whole `~/.claude` directory (which also holds unrelated things:
  `history.jsonl`, `stats-cache.json`, `gh-pr-status-cache.json`, session
  transcripts, plugins, etc, none of which this service needs or should
  see). The host path (`SELF_STUDY_ENRICHMENT_CLAUDE_CREDENTIALS_HOST_PATH`)
  is now a required (`${VAR:?err}`-guarded) compose variable -- an unset
  path fails the compose invocation loudly instead of silently resolving
  to an empty/default bind-mount source for a credential this sensitive
  (a real code-review finding, fixed in the same patch -- see below).
- This service never writes to, logs, or echoes any part of the mounted
  credentials file -- confirmed by inspection of every place `env`/the
  mount path is touched in `app/main.py` and `app/claude_runner.py`; only
  `claude`'s own subprocess reads it, never this service's own code.
- README's "Credential isolation" section rewritten in place with an
  explicit "Corrected 2026-08-12" note (not silently edited) explaining
  what was wrong and what the real mechanism is now.
- New regression tests in `services/orion-self-study-enrichment/tests/test_main.py`:
  - `test_claude_subprocess_env_has_no_api_key_and_sets_claude_config_dir`
    -- simulates an `ANTHROPIC_API_KEY` leaked into the *container's own*
    `os.environ` and proves it is stripped from the subprocess env before
    `claude -p` is spawned, and that `CLAUDE_CONFIG_DIR` is set correctly.
  - `test_settings_has_no_anthropic_api_key_field` -- schema-level guard
    that no `ANTHROPIC_API_KEY` field exists on `Settings`.
  - `test_docker_compose_has_no_anthropic_api_key_and_requires_credentials_host_path`
    -- deterministic (CLAUDE.md sec 4) text-level check on the actual
    `docker-compose.yml` runtime wiring: no `ANTHROPIC_API_KEY` reference
    anywhere, `CLAUDE_CONFIG_DIR` present, and the credentials host path
    uses the fail-fast `${VAR:?err}` form.

**Env/config changes (fix patch).**

- Removed keys: `ANTHROPIC_API_KEY` (fully, from `.env_example`,
  `settings.py`, `docker-compose.yml`).
- Added keys: `SELF_STUDY_ENRICHMENT_CLAUDE_CREDENTIALS_HOST_PATH`
  (operator-specific host path to their real `.credentials.json`, no safe
  default baked into code -- deliberately required, not defaulted),
  `SELF_STUDY_ENRICHMENT_CLAUDE_CONFIG_DIR` (container-side path, defaults
  to `/root/.claude`).
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`:
  ran; no existing local `.env` for this service was found in the primary
  checkout (never deployed there), so there was nothing to sync -- an
  operator deploying this for real must set
  `SELF_STUDY_ENRICHMENT_CLAUDE_CREDENTIALS_HOST_PATH` themselves before
  `docker compose config`/`build`/`up` will succeed (enforced by the new
  `${VAR:?err}` guard).
- Skipped keys requiring operator action:
  `SELF_STUDY_ENRICHMENT_CLAUDE_CREDENTIALS_HOST_PATH` -- operator must
  point this at their own real `~/.claude/.credentials.json` (or wherever
  their `CLAUDE_CONFIG_DIR` override points, if they use one) before
  deploying.

**Tests run (fix patch).**

```text
$ .venv/bin/python -m pytest services/orion-self-study-enrichment/tests -q
25 passed in 2.74s
```

**Docker/build/smoke checks (fix patch).**

```text
$ SELF_STUDY_ENRICHMENT_CLAUDE_CREDENTIALS_HOST_PATH=/home/athena/.claude/.credentials.json \
  SELF_STUDY_ENRICHMENT_REPO_HOST_PATH=/mnt/scripts/Orion-Sapienform \
  PROJECT=test NET=test-net \
  docker compose -f services/orion-self-study-enrichment/docker-compose.yml config
# validated cleanly: CLAUDE_CONFIG_DIR=/root/.claude in environment:,
# single-file read-only bind mount of .credentials.json -> /root/.claude/.credentials.json
# in volumes:, zero ANTHROPIC_API_KEY references anywhere in the rendered config.

$ PROJECT=test NET=test-net SELF_STUDY_ENRICHMENT_REPO_HOST_PATH=/mnt/scripts/Orion-Sapienform \
  docker compose -f services/orion-self-study-enrichment/docker-compose.yml config
# (SELF_STUDY_ENRICHMENT_CLAUDE_CREDENTIALS_HOST_PATH deliberately omitted)
# error while interpolating services.self-study-enrichment.volumes.[]:
#   required variable SELF_STUDY_ENRICHMENT_CLAUDE_CREDENTIALS_HOST_PATH is
#   missing a value: SELF_STUDY_ENRICHMENT_CLAUDE_CREDENTIALS_HOST_PATH must
#   be set -- see .env_example
# confirms the fail-fast guard works live, not just in a unit test.
```

Real `docker compose build`/`up` against this service was not re-run in
this fix patch (no code path changed that affects the image build itself --
only env/settings/compose wiring and the credential mount, all covered by
`config` above plus the unit tests).

**Review findings fixed (this fix patch).**

- Finding (must-fix): the credentials host-path bind mount had no
  fail-fast validation -- an unset
  `SELF_STUDY_ENRICHMENT_CLAUDE_CREDENTIALS_HOST_PATH` could resolve to an
  empty/undefined bind-mount source, an unsafe failure mode for a
  credential this sensitive.
  - Fix: changed to compose's `${VAR:?err}` required-variable form in
    `docker-compose.yml`'s volumes: block.
  - Evidence: live `docker compose config` run above with the var
    deliberately omitted -- fails loudly with the exact configured error
    message, confirmed live, not just asserted in a test.
- Finding (should-fix): `SELF_STUDY_ENRICHMENT_CLAUDE_CONFIG_DIR` defaulted
  to a specific operator's home directory (`/home/athena/.claude`) baked
  into `app/settings.py`'s own `Field(default=...)`, not just the
  `.env_example` (which is expected to be operator-specific) -- would
  silently point at a nonexistent path for any other operator/deployment.
  - Fix: changed the container-side default to `/root/.claude` in
    `settings.py` and `docker-compose.yml` (the container actually runs as
    root -- no `USER` directive in the Dockerfile -- so this is the real
    generic default, not another hardcoded username).
  - Evidence: `app/settings.py:41`, `docker-compose.yml`'s `CLAUDE_CONFIG_DIR`
    and volume-target interpolations; `.env_example` and README's example
    paths updated to match, with an explicit comment explaining why the
    container-side default differs from the host-side example path.
- Finding (should-fix): the `${SELF_STUDY_ENRICHMENT_CLAUDE_CONFIG_DIR:-/root/.claude}`
  interpolation was duplicated between the `environment:` and `volumes:`
  blocks in `docker-compose.yml` with no cross-reference -- a future edit
  to one without the other would silently break the mount (the `claude`
  binary would look in a different directory than where the credential is
  actually mounted, with no error, just an auth failure).
  - Fix: added an explicit comment at the `environment:` line pointing at
    the `volumes:` line and stating they must stay identical.
  - Evidence: `docker-compose.yml` lines around `CLAUDE_CONFIG_DIR=`.
- Finding (should-fix): no deterministic test exercised the actual
  `docker-compose.yml` runtime wiring (only Python-level `Settings`/
  subprocess-env tests existed).
  - Fix: added
    `test_docker_compose_has_no_anthropic_api_key_and_requires_credentials_host_path`,
    a plain text-level assertion against the real compose file (CLAUDE.md
    sec 4 -- deterministic check, not model judgment).
  - Evidence: `services/orion-self-study-enrichment/tests/test_main.py`;
    full suite passes (25/25).
- Finding (nit): the schema-level regression test's docstring implied
  broader coverage ("must not exist anywhere in this service's config
  surface") than it actually proves (it only proves the field is gone from
  `Settings`, not that a stray env var is inert).
  - Fix: narrowed the docstring to state its real coverage boundary and
    point at the subprocess-env test as the actual runtime guard.
  - Evidence: `test_settings_has_no_anthropic_api_key_field`'s updated
    docstring.

**Restart required.** No restart of any existing running service --
PR #1574's original merge never had this service deployed live (Docker
build/smoke in the original PR report was a one-off validation run, not a
persistent deployment). To deploy the corrected version:

```bash
python scripts/sync_local_env_from_example.py
# then set services/orion-self-study-enrichment/.env's
# SELF_STUDY_ENRICHMENT_CLAUDE_CREDENTIALS_HOST_PATH to the real host path
# of ~/.claude/.credentials.json (or wherever CLAUDE_CONFIG_DIR points, if
# overridden) -- no API key to set anymore.
scripts/safe_docker_build.sh orion-self-study-enrichment build
scripts/safe_docker_build.sh orion-self-study-enrichment up -d
```

**Risks / concerns (fix patch).**

- Severity: low
  - Concern: the fixed service was never actually run end-to-end against a
    real, live-mounted `.credentials.json` in this session (only
    `docker compose config` was validated, plus unit tests with a mocked
    subprocess) -- the real `claude -p` process authenticating
    successfully via the mounted file is not directly observed here.
  - Mitigation: `CLAUDE_CONFIG_DIR`'s behavior (relocating where Claude
    Code resolves `.credentials.json`) is Claude Code's own documented
    mechanism, not a novel assumption; the mount/env wiring itself is
    confirmed live via `docker compose config`. A first real deploy should
    include one live `claude -p` smoke run and check its exit code/output,
    not just that the process starts.
- Severity: low
  - Concern: mounting a real, subscription-tied login credential into a
    container is a larger blast radius than the rejected API-key design
    would have been if that container is ever compromised or its image
    published somewhere unintended.
  - Mitigation: mount is read-only, single-file, narrowly scoped (not the
    whole `~/.claude` directory); this is the explicit tradeoff Juniper
    asked for (reuse the real session, not a separate scoped key) and is
    documented plainly in the README's "Credential isolation" section
    rather than hidden.

**PR link (fix):** see below.
