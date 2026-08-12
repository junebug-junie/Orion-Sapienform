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
