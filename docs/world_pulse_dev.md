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
WORLD_PULSE_GRAPH_ENABLED=false
WORLD_PULSE_STANCE_ENABLED=false
ACTIONS_WORLD_PULSE_ENABLED=false
```

## Bounded ingestion strategies

- `rss` / `atom`: pulls feed entries from the configured `url`.
- `sitemap`: reads sitemap URL lists with bounded child sitemap and URL caps.
- `html_section`: extracts anchor links from a single approved section page.
- `manual_urls`: uses explicitly curated URL lists only.
- `api`: dispatched by API host. Only `export.arxiv.org` has an adapter (`app/services/ingest/arxiv_api_adapter.py`); an `api` source on any other host fetches nothing.

## arXiv research papers (`arxiv_research`)

Orion gets up to `max_articles_per_day` (default 4) new arXiv papers per daily run, read through the normal reading queue like any digest item.

- One newest-first export-API query per category in `urls` (cs.AI, cs.LG, cs.CL, q-bio.NC), picked round-robin so no category fills the day.
- Requests are spaced at least 3 s apart process-wide and carry an honest `orion-world-pulse` User-Agent, per arXiv's API terms. Neither is configurable downward.
- Deduped by bare arXiv id against every arXiv URL already in `world_pulse_read_seed` (abs, pdf, ar5iv, `vN` all count as one paper), read via `WORLD_PULSE_READ_HISTORY_DB_URL`. Blank or unreachable DSN logs `world_pulse_arxiv_dedupe_skipped`; Hub's exact-URL `already_read` gate is then the only guard.
- Seeds use `https://arxiv.org/abs/<id>` in the existing `ai_technology` section.
- Dedupe scope: only papers that became reading seeds. A paper shown in a digest but never seeded can be picked again; Hub's exact-URL `already_read` gate still stops a re-read.
- Coverage: arXiv articles do not count toward `ai_technology` coverage, so a tech-news outage still reads as missing and still triggers the curiosity gap-fill. Paper worth-reading/watching entries and situation changes are appended after the ordinary caps, so they displace no news entries.
- All four queries returning zero entries logs `world_pulse_arxiv_all_feeds_empty`; all four failing raises (recorded as `source_fetch_error:arxiv_research`).
- `digest_policy.reserved_source_slots.arxiv_research` gives papers their own digest cards outside `max_digest_items_total` and the per-section cap, so they never displace news coverage. `clustering_policy.singleton_source_ids` keeps each paper its own card.
- Preprints: trust tier 3, corroboration required, no graph or prior writes.
- Live check (read-only): `PYTHONPATH=.:services/orion-world-pulse python services/orion-world-pulse/evals/run_arxiv_source_eval.py --dsn <reading-queue DSN>` prints today's picks.
- Rollback: `enabled: false` on `arxiv_research` in `config/world_pulse/sources.yaml`.
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

## Known zero-source sections

- None as of 2026-10-13. `hardware_compute_gpu` had no passive source from 2026-07-13 and relied on the curiosity live-search gap-fill, which in practice fed the reader YouTube videos and listing pages (`tomshardware.com/news`, `wccftech.com/topic/hardware/`, `nvidianews.nvidia.com/`). Juniper approved replacing it with real feeds (`nvidia_newsroom`, `servethehome`); the gap-fill now only fires when those yield nothing, and it drops listing-page results.
- Do not re-add `nasa_news` (or any other off-topic source) to `hardware_compute_gpu`: NASA's near-daily cadence once marked the section `covered` for months with no GPU content.

## Source quality guidance

Every source must give Orion's reader a fresh, individual article it can actually read. Before enabling one, check all three and record the results in the source's `notes`:

1. `robots.txt` allows `Claude-User` for the article paths (the reading harness's WebFetch agent).
2. A real WebFetch through the harness returns article text, not a refusal, a 403, or a truncated shell.
3. The feed yields distinct, dated article links (not landing pages, not data files) with recent dates.

No user-agent spoofing or other block evasion. A site that refuses the reader is dropped, not routed around.

Runtime guards (`app/services/source_health.py`, `app/services/ingest/base.py`):

- Listing/landing/data-file URLs never become candidates (`orion/world_pulse_read/url_filters.url_looks_like_section_index`).
- `WORLD_PULSE_MAX_ITEM_AGE_DAYS` (default 3): items older than this by their own published date are dropped; undated items are dropped too, so `manual_urls` sources can no longer reach the digest.
- `WORLD_PULSE_STUCK_SOURCE_DIGESTS` (default 3): a source with no new article URL versus the previous real run for this many digests in a row gets a `source_stuck:<source_id>:<reason>:<n>` run warning (reason `same_urls`, `zero_fresh_items`, or `fetch_error`). Per-source counts are in `run.metrics.source_health` (`/api/world-pulse/runs`, `world_pulse_run.payload_json`), which is also where the next run reads its history from via `WORLD_PULSE_SOURCE_HEALTH_DB_URL`.
- Eval: `services/orion-world-pulse/evals/run_source_quality_eval.py --replay 10 --live --old-root <origin/main tree>`.

- Prefer official/public-interest sources, wire-style reporting, and stable metadata-rich feeds.
- Avoid paywalled feeds, social/comment sources, broad homepage scraping, and JS-rendered-only pages.
- Keep World Pulse strategy-scoped; it is not a general crawler and must not recursively spider links.

## Example approved source entry

```yaml
- source_id: pbs_newshour_politics
  name: PBS NewsHour Politics RSS
  type: rss
  strategy: rss
  url: https://www.pbs.org/newshour/feeds/rss/politics
  domains: ["www.pbs.org", "pbs.org"]
  allowed_path_prefixes: ["/newshour/"]
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
  max_articles_per_day: 6
  notes: "Verified 2026-10-13: robots.txt allows Claude-User; harness WebFetch returned full article text; feed 20 distinct dated items."
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
- Human-facing digest cards are curated and capped by `digest_policy` so Hub cards are readable.
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
- Keep side effects disabled during iteration (`WORLD_PULSE_GRAPH_ENABLED=false`, `WORLD_PULSE_STANCE_ENABLED=false`, scheduler off).

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
```

Expected safety behavior:
- Hub message publish: `published` on `hub.messages.create.v1` (messages rail).
- There is no World Pulse email path (retired 2026-09-30; the `publish-email` route, `WORLD_PULSE_EMAIL_*` keys, and `EmailWorldPulseRenderV1` were removed). The news digest reaches Juniper by email only through orion-actions' Journal Pass (`trigger_kind=world_pulse_digest`).
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
WORLD_PULSE_GRAPH_ENABLED=false
WORLD_PULSE_STANCE_ENABLED=false
ACTIONS_WORLD_PULSE_ENABLED=false
```

