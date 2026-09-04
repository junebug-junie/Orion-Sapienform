# orion-exo-exploration

Crawls three KSL classifieds category pages once a day, dedups by KSL's own
listing ID, scores tech/compute interest with a fixed keyword/price rule set,
keeps 14 days of history, and surfaces the results in a Hub tab.

"Exo" as in outside Orion's own systems -- this is Orion looking outward at
the secondhand-tech market on KSL, not a subsystem of Orion's cognition.

## Why this is not orion-world-pulse

`orion-world-pulse`'s `ArticleCandidate` schema (`trust_tier`, `region_scope`,
`discovered_via`) exists to turn world news into something Orion's cognition
can reason about. A classifieds listing is not news, has no trust tier worth
modeling, and is not meant to become a concept node -- it is meant to answer
one question for Juniper: "is there anything worth looking at on KSL today."
Extending world-pulse's schema to cover it would have made a domain-specific
field (`region_scope`) meaningless for every classifieds row, and made this
service's real key (KSL's own listing ID) invisible behind a schema built for
URLs. Checked and rejected as a host for this; see
`services/orion-world-pulse/app/services/ingest/base.py` for the fetch-helper
shape this service mirrors instead of importing.

## Why this is not Hub

Hub has no crawler, no scheduler for external fetches, and no retention
policy for third-party data -- it is a UI and proxy layer over other
services' APIs, and stays that way here too. This service owns the crawl,
the Postgres tables, the scoring, and the retention sweep; Hub's
`scripts/exo_exploration_routes.py` only proxies `GET /finds` and
`GET /crawl-runs` for the Hub tab, exactly like `curiosity_routes.py` and
`api_routes.py`'s world-pulse proxy already do for their own backing
services.

## Why keyword/price rules, not NER or an LLM call

Category pages already narrow the domain (every candidate came from an
Electronics/Computers/FREE KSL search), real titles are short and
keyword-dense (confirmed live 2026-09-04: "Asus Rog Nuc 2025 - RTX 5060 - 32
GB DDR5", "LOADED HP ELITEBOOK 13TH GEN I7 32GB 512GB WIN 11 W/FACTORY
WARRANTY"), and a fixed keyword list keeps every score inspectable --
`interest_reasons` is always the literal rule text that fired, per
AGENTS.md's "no empty-shell cognition." See `evals/run_interest_scoring_eval.py`
for the precision/recall check against real KSL titles.

## Contract

```
GET /finds?category=&min_interest=&status=   -> deduped current listings
GET /finds/{external_listing_id}              -> one listing
GET /crawl-runs                                -> recent crawl run history
GET /interest-rules                            -> the scoring rule set
GET /healthz /readyz
```

## Tables (Postgres `conjourney` DB, `exo_exploration_` prefix)

- `exo_exploration_listings_observed` -- append-only, one row per fetch.
- `exo_exploration_listings_current` -- one row per `external_listing_id`
  (KSL's own listing ID, parsed from the canonical `/listing/<id>` URL),
  upserted on every crawl. A renewal (same ID, new posted date) updates this
  row; it never inserts a second one.
- `exo_exploration_crawl_runs` -- one row per crawl attempt.
- `exo_exploration_interest_rules` -- the scoring rule set, seeded once with
  the 3 category URLs (informational) and a starter tech/compute keyword
  list. This is the extensibility seam for a future "add keywords from Hub"
  feature: a new row, not a schema change.

## Retention

Both `exo_exploration_listings_observed` and `exo_exploration_listings_current`
keep 14 days past a listing's `last_seen_at` (`EXO_EXPLORATION_RETENTION_DAYS`).
An hourly sweep loop (`app/crawl/daemon.py::retention_sweep_loop`) deletes
rows past `expires_at` -- deliberately more frequent than the daily crawl, so
an expired listing does not linger for up to a day.

## Crawl practice

- `robots.txt` fetched and checked against the 3 target category paths
  before this adapter was written (confirmed live 2026-09-04: no generic
  disallow; `ksl-crawler` is explicitly allowed; only `oodlebot` and
  `Clickagy Intelligence Bot v2` are blocked).
- Honest, identifying User-Agent (`EXO_EXPLORATION_USER_AGENT`).
- No login, cookies, or session -- anonymous public GETs only.
- A polite delay (`EXO_EXPLORATION_REQUEST_DELAY_SECONDS`) between every
  request, including per-listing detail fetches.
- A listing only gets a detail-page fetch (for its full description) after
  it already passed the keyword/price filter on its title alone --
  `EXO_EXPLORATION_MAX_DETAIL_FETCHES_PER_RUN` caps that per run.

## Operating

```bash
cd <worktree> && scripts/safe_docker_build.sh orion-exo-exploration up -d --build
curl localhost:8622/healthz
curl localhost:8622/finds
```
