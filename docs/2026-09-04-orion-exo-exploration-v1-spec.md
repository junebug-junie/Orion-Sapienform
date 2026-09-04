# orion-exo-exploration v1 — spec

Status: DESIGN (not yet implemented)
Date: 2026-09-04
Author: Athena Orchestration, for Juniper

## Plain-English summary

Orion is getting a way to look outside their own system at a real-world
marketplace. This spec covers **v1 only**: a new service,
`orion-exo-exploration`, checks three KSL classifieds category pages once a
day, keeps a memory of what it's already seen so a renewed or promoted
listing doesn't count as "new" again, scores what it finds for how
tech/compute-relevant it is, and shows the best of those in a new Hub tab
with its own "worth a look" subsection.

Three follow-on phases are named but **not** designed here:

- **v2** — separate credentialed service for `auction.npsstore.com`'s Excel
  manifests.
- **v3** — lets Juniper add categories/keywords from the Hub UI instead of a
  code change. v1's data model is built so this is a new row, not a new
  column.
- **v4** — gives Orion a real KSL login to favorite listings and message
  sellers. This is Orion taking real-world social/financial action on
  Juniper's behalf, so per this repo's rules (AGENTS.md §0A, "Proposal mode
  before invasive cognition changes") it gets its own proposal doc before any
  code, not a fold-in to this spec.

## Current architecture

- No scraper, crawler, or marketplace-watching service exists anywhere in
  `services/` today (checked; 93 services, none match).
- Hub already has an operator-facing "pending attention" card type
  (`PendingAttentionCardV1`, `orion/schemas/attention_salience.py`), but every
  field on it (`loop_id`, `theme_key`, `weights_version="gwt-coalition-v1"`)
  is specific to Orion's internal cognitive-loop attention scorer. Reusing it
  for marketplace listings would be misusing an existing contract, not
  reusing one — so `orion-exo-exploration` gets its own small card type and
  its own Hub panel instead.
- The closest architectural cousin is `orion-percept-store`: a small service
  that owns its own short-lived store for external content and deliberately
  does not live inside Hub. Same reasoning applies here — different content,
  different retention, different owner.
- No generic "run this once a day" infra exists; every service that ticks on
  a schedule runs its own internal scheduler. v1 does the same: a plain
  `asyncio` daily loop inside the service process, not new shared infra.

## Decisions locked by this conversation

- **Interest domain**: tech/compute-related items only (the categories
  already narrow this — Electronics, Computers, FREE — but scoring still
  needs to separate "free couch" from "free GPU").
- **Retention**: keep an observed listing's row for **2 weeks** after
  `last_seen_at` stops advancing (i.e. it fell out of KSL's search results),
  then it's eligible for deletion. Raw `ExoListingObservedV1` crawl-events
  follow the same 2-week window — this is a monitoring service, not a
  memory system, so nothing here is meant to be permanent.
- **Classification method**: keyword/rule-based for v1, **not** an NER model
  or an LLM call. Rationale, per AGENTS.md §4 (deterministic vs. latent
  split) and §0A (avoid speculative features): listing titles are short and
  keyword-dense ("RTX 4070", "Dell PowerEdge", "NAS", "GPU", "server rack"),
  the category pages already do most of the filtering, and a fixed keyword
  list is fully inspectable — every score comes with the literal words that
  triggered it. An LLM-per-listing pass adds cost, latency, and a non-
  deterministic thing to debug, for a classification problem that a ~100-term
  allowlist plus a price-band heuristic can likely solve outright. If the
  eval (below) shows the keyword approach missing real tech listings or
  flooding on junk, the next step is a small local text classifier trained
  on triaged examples — not a full LLM call — before reconsidering an LLM.
  This is a call, not a default hidden from you: override it if you'd rather
  start with an LLM pass.

## Scraping practice (common-practice, not custom infra)

Low volume (3 category pages, once a day) means no scraping framework is
warranted — a scheduled `httpx` + `selectolax`/`BeautifulSoup` fetch is
enough. Still doing this on purpose, not by accident:

- Read `https://classifieds.ksl.com/robots.txt` at implementation time and
  respect it (`Disallow` rules honored, no crawling a path it excludes).
- One request per category per day, plus per-listing detail fetches only for
  listings that pass the category filter (not every result row) — this is
  nowhere near a volume that needs distributed rate limiting.
- Honest, identifying `User-Agent` string (not spoofed as a browser), and a
  polite fixed delay between requests within a run.
- Cache `ETag`/`Last-Modified` if KSL sends them, to skip re-fetching detail
  pages that haven't changed.
- No login, no cookies, no session — v1 only ever does anonymous, public GET
  requests. This is the thing that keeps v1 outside any account-terms
  question; that question starts to matter at v4.

## Proposed schema / API changes

Event-substrate-first (AGENTS.md §0A), so dedup is provable, not asserted:

**`ExoListingObservedV1`** — one row per crawl-sighting, append-only, never
edited:
- `source: Literal["ksl"]`
- `source_category: str` (foreign key to `ExoInterestRuleV1.category_url` below)
- `external_listing_id: str` — parsed from the canonical listing URL. This is
  the real dedup key, **not** title/price/description. KSL assigns a stable
  ID per ad; a renewal bumps its sort position and `posted_or_renewed_at`,
  it does not change this ID.
- `url: str`, `title: str`, `price: Decimal | None`, `description: str`
- `posted_or_renewed_at: datetime` (site-reported)
- `raw_content_hash: str` (title+price+description hash, for detecting an
  edited relist vs. a pure bump)
- `crawl_id: str`, `observed_at: datetime`

**`ExoListingCurrentV1`** — a reducer's output, one row per
`external_listing_id`:
- `first_seen_at`, `last_seen_at`, `times_seen: int`
- `is_currently_listed: bool` — flips false once a previously-seen listing
  stops appearing in its category page across a run
- `interest_score: float`, `interest_reasons: list[str]` — the literal
  keyword/price rule(s) that fired, always shown, never a bare number
- `possible_duplicate_of: str | None` — KSL's "promoted" placements sometimes
  render as a second entry that doesn't share the canonical listing ID as
  cleanly as a plain renewal does. Those get **flagged** for a human glance,
  never silently merged.
- `expires_at: datetime` — `last_seen_at + 14 days`, the field the retention
  sweep reads.

**`ExoCrawlRunV1`** — one row per daily run: `started_at`, `finished_at`,
`categories_crawled`, `listings_seen`, `new_listings`, `errors`, `status`.
This is what makes "the crawl actually ran" provable from a stored artifact
instead of "the container is up."

**`ExoInterestRuleV1`** — a table, not a hardcoded enum or Python constant,
from day one:
- `category_url: str` (the 3 KSL URLs below, v1-seeded)
- `keyword: str | None`, `min_price: Decimal | None`, `max_price: Decimal | None`
- `weight: float`
- `added_by: Literal["seed", "operator"]`, `created_at`

v3 adds a `POST /interest-rules` write path to this same table from Hub — no
new schema needed then, just a new door into existing rows. This is the
concrete answer to "the v1 data model needs some extensibility."

**v1 seed rows** (categories, from your message):
```
https://classifieds.ksl.com/search/cat/Electronics
https://classifieds.ksl.com/search/cat/Computers
https://classifieds.ksl.com/search/cat/FREE
```
Starter keyword list (tech/compute-relevant; tune after the first week of
real data): `gpu, rtx, gtx, radeon, cpu, ryzen, threadripper, xeon, server,
poweredge, rack, nas, synology, motherboard, mobo, ram, ddr4, ddr5, ssd, nvme,
raid, switch (network), workstation, mining rig, psu, power supply, monitor
(4k/ultrawide), laptop (business/thinkpad/dell/lenovo), docking station`.

**API** (owned by `orion-exo-exploration`, Hub just calls it):
```
GET /finds?category=&min_interest=&status=
GET /finds/{external_listing_id}
GET /crawl-runs
GET /interest-rules            (v1: read-only echo of the seed table)
POST /interest-rules           (v3: operator-editable; not built in v1)
```

No `orion/bus/channels.yaml` entry in v1 — nothing else in Orion consumes
this yet. Revisit at review time if that changes.

## Files likely to touch

- New: `services/orion-exo-exploration/{README.md,.env_example,docker-compose.yml,requirements.txt,settings.py,app/,tests/,evals/}`
- `services/orion-hub/` — new tab (template + JS) reading from the new
  service's API; a "worth a look" subsection filtered to `interest_score`
  above a threshold, each card showing its `interest_reasons`.
- `orion/schemas/exo_exploration.py` + `orion/schemas/registry.py` — only if
  review decides a bus channel is needed after all.

## Non-goals (v1)

- No KSL login, no messaging sellers, no purchasing.
- No auction-site work.
- No NER model or LLM call for classification (see Decisions above).
- No cross-listing resale-value estimate — no market-price data to base one
  on yet.
- No cross-device/cross-user sync — this is a single operator (Juniper)
  reviewing from Hub.

## Acceptance checks

- A daily run against all 3 category URLs writes an `ExoCrawlRunV1` row with
  `status="success"` even when 0 new listings are found.
- Running the same crawl twice back-to-back with no upstream changes creates
  zero new `external_listing_id` rows.
- A listing KSL shows as renewed (same ID, new `posted_or_renewed_at`)
  updates the existing row's `last_seen_at`/`times_seen`, not a new row.
- A listing that stops appearing in its category page has `is_currently_listed`
  flip to `false` on the run where it's absent, and its row is gone (or
  excluded from `/finds`) 14 days after its last `last_seen_at`, provable by
  a retention-sweep test with a synthetic old row.
- Hub's new tab shows at least one real find from a live crawl, with a
  working link back to the actual KSL listing.
- The "worth a look" subsection shows only listings above the interest
  threshold, and every card's `interest_reasons` is non-empty and human-
  readable.
- Eval: a small hand-labeled set of ~30 real KSL titles (mix of tech and
  non-tech) run through the keyword scorer, with precision/recall reported —
  this is the evidence for whether the "no LLM yet" call above was right.

## Recommended next patch

v1 only, one reviewable PR: crawler + dedup reducer + retention sweep + the
3 seeded category URLs + keyword/price interest scoring + the Hub tab. New
worktree, per AGENTS.md §2. No bus channel unless review says otherwise.

v2–v4 wait until this is live and real finds have been reviewed for a few
days.
