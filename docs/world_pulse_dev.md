# World Pulse Dev Bring-Up

World Pulse is an internal Orion service and is not intended to be directly exposed to browsers.
In Tailscale deployments, expose World Pulse only through the Hub/proxy path (`/api/world-pulse/*` or `/world-pulse/*`).
External/browser access should go through Hub or a Tailscale path-based reverse proxy, not direct `http://127.0.0.1:8118`.

## Manual dry-run bring-up

1. Create venv and install deps:

```bash
python3 -m venv .venv-world-pulse
. .venv-world-pulse/bin/activate
python -m pip install -r services/orion-world-pulse/requirements.txt
```

2. Start World Pulse (safe mode, approved-source ingest enabled):

```bash
WORLD_PULSE_ENABLED=true \
WORLD_PULSE_DRY_RUN=true \
WORLD_PULSE_FETCH_ENABLED=true \
WORLD_PULSE_EMAIL_ENABLED=false \
WORLD_PULSE_EMAIL_DRY_RUN=true \
WORLD_PULSE_GRAPH_ENABLED=false \
WORLD_PULSE_GRAPH_DRY_RUN=true \
WORLD_PULSE_STANCE_ENABLED=false \
ACTIONS_WORLD_PULSE_ENABLED=false \
uvicorn app.main:app --app-dir services/orion-world-pulse --host 0.0.0.0 --port 8118
```

3. Start/restart Hub with host-reachable internal World Pulse URL:

```bash
WORLD_PULSE_BASE_URL=http://127.0.0.1:8118
```

Hub remains the browser-facing surface. Keep frontend calls relative (for example, `/api/world-pulse/latest`).
World Pulse must stay internal-only and should not be mounted as a direct public endpoint.

4. Verify:

```bash
curl http://127.0.0.1:8118/healthz
curl http://127.0.0.1:8080/api/world-pulse/healthz
python scripts/world_pulse_integration_smoke.py --fixtures --dry-run
python scripts/world_pulse_integration_smoke.py --dry-run --approved-sources
```

## Keep side effects disabled

```bash
WORLD_PULSE_EMAIL_ENABLED=false
WORLD_PULSE_GRAPH_ENABLED=false
WORLD_PULSE_STANCE_ENABLED=false
ACTIONS_WORLD_PULSE_ENABLED=false
```

## Bounded ingestion strategies

- `rss` / `atom`: pulls feed entries from the configured `url`.
- `sitemap`: reads sitemap URL lists with bounded child sitemap and URL caps.
- `html_section`: extracts anchor links from a single approved section page.
- `manual_urls`: uses explicitly curated URL lists only.
- Strategy fetch is always bounded by source `domains`, optional `allowed_path_prefixes`, and per-source limits.

## Per-source rollback and guardrails

- Disable an individual source by setting `enabled: false` in `config/world_pulse/sources.yaml`.
- Keep source sets approved-only (`enabled && approved`) and avoid adding broad crawlers.
- No recursive spidering is allowed; only configured strategy-scope fetch is allowed.

## Run status vs coverage status

- `run.status` reports pipeline execution health (`completed` / `partial` / `failed`).
- `digest.coverage_status` reports Daily World Pulse section completeness (`complete` / `partial` / `sparse` / `empty`).
- A run can be `completed` while coverage is `partial` or `sparse` when approved reachable sources do not cover all required/recommended sections.
- Current NASA/USGS-only ingest proves bounded ingestion and digest generation, but does not imply full politics/local/AI section coverage.

## Coverage contract in source config

- `required_sections` defines sections that should be covered for complete contract fulfillment.
- `recommended_sections` defines additional sections that improve completeness but do not block run success.
- Source-level `categories` drive section mapping; add approved bounded sources tagged to missing sections to improve coverage.

## Adding sources for missing sections

- Check `digest.section_coverage` and `run.metrics.missing_required_sections` / `missing_recommended_sections` after a dry-run.
- Add at most 1-2 approved sources per missing section; prefer stable RSS/Atom first.
- Set `strategy`, `domains`, and (when practical) `allowed_path_prefixes` to keep ingestion bounded.
- Keep `required: false` during expansion unless the source is operationally critical and stable.
- If a source fails in this environment, set `enabled: false` with a clear `notes` entry instead of deleting it.

## Source quality guidance

- Prefer official/public-interest sources, wire-style reporting, and stable metadata-rich feeds.
- Avoid paywalled feeds, social/comment sources, broad homepage scraping, and JS-rendered-only pages.
- Keep World Pulse strategy-scoped; it is not a general crawler and must not recursively spider links.

## Example approved source entry

```yaml
- source_id: npr_us_politics
  name: NPR Politics RSS
  type: rss
  strategy: rss
  url: https://feeds.npr.org/1014/rss.xml
  domains: ["feeds.npr.org", "npr.org"]
  categories: ["us_politics"]
  trust_tier: 2
  enabled: true
  approved: true
  required: false
  allowed_uses:
    digest: true
    claim_extraction: true
    graph_write: true
    stance_capsule: false
    prior_update_candidate: true
  politics_allowed: true
  requires_corroboration: true
  max_articles_per_day: 8
  notes: "Reachable RSS feed used for US politics coverage."
```

## Coverage status interpretation

- `complete`: all required and recommended sections are covered.
- `partial`: required sections covered, but one or more recommended sections missing.
- `sparse`: one or more required sections missing.
- `empty`: no section coverage from accepted articles.

Browser access must continue through Hub or a Tailscale path proxy; do not expose World Pulse directly.

## Evidence vs curated digest

- `articles_accepted` is the evidence layer and may be large.
- Situation tracking consolidates articles into topic/cluster updates before human digest rendering.
- Human-facing digest cards are curated and capped by `digest_policy` so Hub/email are readable.
- `accepted_article_count` can be much larger than `len(digest.items)` by design.

## Curation and consolidation policy

- `digest_policy.max_digest_items_total` (default `12`) caps total digest cards.
- `digest_policy.max_digest_items_per_section` (default `2`) limits section dominance.
- `digest_policy.min_digest_items_per_required_section` ensures required sections are represented when available.
- `situation_policy.max_situation_changes_per_run` caps change noise and favors topic-level consolidation.
- Section rollups summarize article counts, curated card counts, and coverage confidence by section.

## Tuning and inspection

- Tune policies in `config/world_pulse/sources.yaml` under `digest_policy` / `situation_policy`.
- Inspect `run.metrics.article_clusters`, `run.metrics.digest_items`, and `digest.section_rollups` to compare evidence volume vs curated output.
- Keep side effects disabled during iteration (`WORLD_PULSE_EMAIL_ENABLED=false`, `WORLD_PULSE_GRAPH_ENABLED=false`, `WORLD_PULSE_STANCE_ENABLED=false`, scheduler off).

## Deterministic topic clustering

- Clustering is deterministic and local: no embeddings, no vector database, and no LLM calls.
- Articles are normalized into topic terms, weak/source boilerplate tokens are removed, then assigned by token/entity similarity.
- Accepted articles remain full evidence; clusters are topic/situation candidates used for consolidated changes and curated digest cards.
- Singleton clusters are expected for genuinely distinct stories; a high singleton ratio can indicate either diverse source content or too-strict clustering thresholds.

## Clustering diagnostics

- `run.metrics.article_clusters`
- `run.metrics.singleton_cluster_count`
- `run.metrics.multi_article_cluster_count`
- `run.metrics.average_articles_per_cluster`
- `run.metrics.capped_situation_changes`

Tune `clustering_policy` in `config/world_pulse/sources.yaml`:
- `similarity_threshold`
- `max_articles_per_cluster`
- `min_strong_terms`
- `preserve_section_boundaries`

## Dev persistence mode (SQL + Hub messages)

Use this mode for manual dev runs where durable SQL rows and Hub message records are required, while keeping risky side effects off:

```bash
WORLD_PULSE_ENABLED=true
WORLD_PULSE_DRY_RUN=false
WORLD_PULSE_FETCH_ENABLED=true
WORLD_PULSE_SQL_ENABLED=true
WORLD_PULSE_HUB_MESSAGES_ENABLED=true
WORLD_PULSE_EMAIL_ENABLED=false
WORLD_PULSE_EMAIL_DRY_RUN=true
WORLD_PULSE_GRAPH_ENABLED=false
WORLD_PULSE_GRAPH_DRY_RUN=true
WORLD_PULSE_STANCE_ENABLED=false
ACTIONS_WORLD_PULSE_ENABLED=false
WORLD_PULSE_UI_FIXTURE_RUN_ENABLED=false
```

Run and verify through Hub proxy only:

```bash
curl -sS http://127.0.0.1:8080/api/world-pulse/healthz
curl -sS -X POST http://127.0.0.1:8080/api/world-pulse/run \
  -H "Content-Type: application/json" \
  -d '{"requested_by":"hub"}'
curl -sS http://127.0.0.1:8080/api/world-pulse/latest
curl -sS -X POST http://127.0.0.1:8080/api/world-pulse/api/world-pulse/runs/<run_id>/publish-hub-message
curl -sS -X POST http://127.0.0.1:8080/api/world-pulse/api/world-pulse/runs/<run_id>/publish-email
```

Expected safety behavior:
- Hub message publish: `published` on `hub.messages.create.v1` (messages rail).
- Email publish endpoint: `status=skipped` (preview rail remains disabled for sends).
- RDF write remains off (`WORLD_PULSE_GRAPH_ENABLED=false`), with dry-run setting available for explicit graph-plan inspection.
- Scheduler and stance remain off.

## SQL verification

For local dockerized Postgres, verify run rows directly:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c \
"select count(*) from world_pulse_run where run_id='<run_id>';"
```

Optional helper script:

```bash
python scripts/world_pulse_verify_persistence.py --run-id <run_id> --db-url <postgresql-url>
```

## Rollback flags

If you need to immediately disable durable side effects and return to safe dry mode:

```bash
WORLD_PULSE_DRY_RUN=true
WORLD_PULSE_SQL_ENABLED=false
WORLD_PULSE_HUB_MESSAGES_ENABLED=false
WORLD_PULSE_EMAIL_ENABLED=false
WORLD_PULSE_GRAPH_ENABLED=false
WORLD_PULSE_STANCE_ENABLED=false
ACTIONS_WORLD_PULSE_ENABLED=false
```

