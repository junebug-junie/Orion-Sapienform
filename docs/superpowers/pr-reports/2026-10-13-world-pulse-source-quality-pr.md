## Summary

- Orion's world-pulse reading was mostly nasa.gov. That wasn't curiosity. Most of the other feeds were broken. who.int re-served a January statement every day. usgs.gov served an index page and cisa.gov a JSON data file. BBC and NPR refuse the reader. Hardware had no feed at all, so it got YouTube videos and listing pages.
- Only individual article URLs reach the digest now. Listing, landing, and data-file URLs are rejected at ingest, and the curiosity gap-fill drops listing-page results.
- Freshness window: items older than `WORLD_PULSE_MAX_ITEM_AGE_DAYS` (default 3) are dropped, judged by the item's own published date. Undated items are dropped too. Atom/ISO dates and `dc:date` now parse.
- Stuck-source detector: a source can go `WORLD_PULSE_STUCK_SOURCE_DIGESTS` (default 3) digests in a row without a URL it hasn't served recently. When that happens it gets a `source_stuck:<id>:<reason>:<n>` run warning and a log warning. Per-source counts go into `run.metrics.source_health`.
- Sources: npr, bbc, who, cisa_kev, and usgs are disabled. 12 verified feeds were added. Digest curation is now breadth-first, so each section gets one card before any section gets two.

## Outcome moved

Fresh, readable reading seeds, measured with `evals/run_source_quality_eval.py`:

- **Replay of the last 10 real digests (live `world_pulse_read_seed`)**:

  | Section | Seeds | Distinct URLs | New readable URLs |
  |---|---|---|---|
  | us_politics (npr) | 20 | 14 | 0 |
  | global_politics (bbc) | 20 | 20 | 0 |
  | healthcare (who) | 20 | 2 | 2 |
  | security (cisa) | 10 | 1 | 0 |
  | science (usgs + nasa) | 20 | 8 | 7 |
  | local (gov) | 20 | 3 | 3 |
  | ai (huggingface) | 10 | 3 | 3 |
  | hardware (YouTube, listing pages) | 50 | 11 | 6 |

  The new stuck detector, replayed over those same digests, flags who.int, usgs.gov, and cisa.gov from digest 4. That is the earliest possible with K=3.
- **Live digest today, built twice through the real pipeline (dry run)**:
  - origin/main code and config: **1 of 12** seeds were fresh and readable (one NASA item). The rest were BBC/NPR (blocked), WHO items from 2025/2026-01, the USGS index page, the CISA JSON file, and stale governor and Hugging Face posts. Hardware was absent.
  - This branch: **12 of 12** were fresh and readable, across all 8 sections. Picks:
    - PBS NewsHour ×2 (US politics)
    - UN News ×2 (global politics)
    - Deseret ×2 (local politics)
    - Hugging Face (AI)
    - NASA (science)
    - STAT (health)
    - BleepingComputer ×2 (security)
    - ServeTheHome (hardware)

  The window dropped 901 stale feed items.

## Source verification (2026-10-13)

The check for every candidate:

- `robots.txt` allows `Claude-User`, checked with `urllib.robotparser` on the feed and on a sample article.
- A real WebFetch of an article through the harness returned article text.
- The feed (fetched with our own UA) gives distinct, dated article links.

No UA spoofing.

| Section | Kept / added | Dropped (why) |
|---|---|---|
| us_politics | PBS NewsHour Politics (~600w, 20/20 fresh), Roll Call (~1,100w, 7/10 fresh, some items are photo posts), CSM USA (~450w) | NPR (robots disallow Claude-User), AP (robots disallow, 403), Reuters (robots disallow, feed 404) |
| global_politics | PBS NewsHour World (~1,000w), UN News (~550w, 12/30 fresh), CSM World (~1,800w) | BBC (WebFetch refuses domain), DW (WebFetch refuses domain), France24 (robots disallow) |
| local_politics | Utah Governor (kept, readable, but no post in the last 3 days), Deseret Politics (~600w, 10/20 fresh) | Utah News Dispatch (WebFetch 403) |
| healthcare_mental_health | KFF Health News (~1,450w), STAT (~1,750w), ScienceDaily Mind & Brain (~650w) | WHO (newest item 2026-02-25), CDC newsroom (newest 09-22), NIH (403), MedicalXpress (thin, ~200w), The Conversation (robots disallow) |
| security | The Record (~400w), BleepingComputer (~330w) | CISA KEV (data file; cisa.gov feeds 403), Krebs (0 fresh in 3 days, not added) |
| science | NASA (kept), ScienceDaily Top Science (~1,400w) | USGS (index pages, no RSS), NOAA RSS (re-dates a 2020 Hurricane Isaias story as today), NSF (0 fresh) |
| hardware | NVIDIA Newsroom (~650w / blog ~2,100w), ServeTheHome (~650-1,400w) | Tom's Hardware (WebFetch returns a truncated shell, no body), TechPowerUp (robots disallow), Phoronix (WebFetch 403), HPCwire (WebFetch 403) |
| ai | Hugging Face (kept) | none |

## Current architecture

`config/world_pulse/sources.yaml` lists the RSS and manual sources. `pipeline.run_world_pulse` fetches each source, caps it to `max_articles_per_day` in feed order (no date check), clusters the articles, and curates up to 12 digest cards. `orion/world_pulse_read/seeds.py` turns each card's first article into a reading seed. Nothing compared one day's picks with the day before.

## Architecture touched

- orion-world-pulse ingest (`ingest/base.py`, `ingest/rss_adapter.py`), the pipeline fetch loop, digest curation, and curiosity findings.
- New `app/services/source_health.py`. It holds the freshness filter and the stuck detector. The detector does a fail-open read of the previous non-dry `world_pulse_run.payload_json -> run.metrics.source_health`.
- `orion/world_pulse_read/url_filters.py`: `/blog/<org>/<post>` is now treated as an article. 182 of 876 Hugging Face feed items were being rejected. This module is shared with the hub reader.
- No schema, bus, or API change. `run.metrics` is already `dict[str, Any]`.

## Files changed

- `config/world_pulse/sources.yaml`: 5 sources disabled with evidence, 12 added with verification notes, reuters/noaa notes re-checked.
- `services/orion-world-pulse/app/services/source_health.py`: new module (`filter_fresh`, `assess_source_health`, `load_previous_source_health`).
- `services/orion-world-pulse/app/services/pipeline.py`: wires the window and health metrics into run warnings.
- `services/orion-world-pulse/app/services/ingest/base.py`: ISO date parsing; listing/data URLs rejected.
- `services/orion-world-pulse/app/services/ingest/rss_adapter.py`: Atom `rel=alternate`, guid permalink, `dc:date`, RDF items, de-duplication.
- `services/orion-world-pulse/app/services/digest.py`: breadth-first curation.
- `services/orion-world-pulse/app/services/curiosity.py`: drops listing-page findings.
- `services/orion-world-pulse/app/settings.py`, `.env_example`, `scripts/sync_local_env_from_example.py`: 3 new keys.
- `orion/world_pulse_read/url_filters.py`: org-owned blog posts.
- `services/orion-world-pulse/tests/test_source_quality.py`: new tests. `conftest.py` isolates tests from the DB. 3 existing tests were updated for the new behaviour (dated fixtures, article-shaped slug).
- `services/orion-world-pulse/evals/run_source_quality_eval.py`: replay plus live old-vs-new eval.
- `docs/world_pulse_dev.md`: source verification rule, runtime guards, hardware section no longer zero-source.

## Schema / bus / API changes

- Added: `run.metrics.source_health`, `stuck_sources`, `source_health_history`, `max_item_age_days`, `stale_items_dropped`, `undated_items_dropped`, `future_items_dropped`, and run warning `source_stuck:<id>:<reason>:<n>`.
- Removed / renamed: none.
- Behavior changed: undated and >3-day-old items no longer reach the digest. `manual_urls` sources (undated) can no longer contribute.
- Compatibility: the first run after deploy has no history (`source_health_history=no_row`), so counting starts then. The first possible stuck flag is on the 4th daily run.

## Env/config changes

- Added keys: `WORLD_PULSE_MAX_ITEM_AGE_DAYS=3`, `WORLD_PULSE_STUCK_SOURCE_DIGESTS=3`, `WORLD_PULSE_SOURCE_HEALTH_DB_URL=postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney`.
- `.env_example` updated: yes. The compose file uses `env_file`, so no compose change was needed.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py orion-world-pulse`: yes. The primary checkout's `services/orion-world-pulse/.env` has all 3 keys.
- Skipped keys: none.

## Tests run

```text
pytest services/orion-world-pulse/tests                          124 passed
pytest tests/test_world_pulse_read_url_filters.py orion/world_pulse_read/tests  (with the above) 278 passed
pytest services/orion-hub/tests/test_world_pulse_read_*.py       235 passed, 20 skipped (no local Postgres)
scripts/check_env_template_parity.py                             PASS
scripts/check_service_env_compose_parity.py orion-world-pulse    N/A (env_file)
scripts/check_env_key_single_source.py                           OK
git diff --check                                                 clean
```

## Evals run

```text
evals/run_source_quality_eval.py --replay 10 --live --old-root <origin/main tree>   exit 0
  fresh readable seeds today: before=1 after=12 (all 8 sections)
```

Live history read: the `_PREVIOUS_SQL` query ran against live Postgres without error and returned no row, which is the pre-deploy state. The reviewer's EXPLAIN ANALYZE took 61ms.

## Docker/build/smoke checks

```text
Not built or deployed (per instruction). The reviewer confirmed the container resolves orion-athena-sql-db on app-net and that sqlalchemy/psycopg2 are already in requirements.
Live detector behaviour after deploy is UNVERIFIED until 3+ daily runs have landed.
```

## Review findings fixed

- Finding: an empty or failed run wiped the URL history, so a repeat URL afterwards counted as "new".
  - Fix: `seen_urls` (union of recent runs, cap 50) is carried through empty and failed runs.
  - Evidence: `test_an_empty_or_failed_run_does_not_reset_the_count`.
- Finding: a feed rotating between two stale URLs (A, B, A, B) was never flagged.
  - Fix: "new" is now checked against `seen_urls`.
  - Evidence: `test_rotating_between_the_same_stale_urls_goes_stuck`.
- Finding: the URL filter rejected org-owned Hugging Face posts (182/876).
  - Fix: `url_filters` treats `/blog/<org>/<post>` as an article. Taxonomy buckets are still treated as listings.
  - Evidence: the parametrized real-URL test; url_filters suites pass.
- Finding: a digest-disabled source would be falsely flagged as stuck.
  - Fix: such sources are not observed.
  - Evidence: `test_digest_disabled_source_is_not_observed`.
- Finding: the history status was ambiguous; the pool could wait 30s; future-dated items sorted to the top; the fail-open test passed for the wrong reason.
  - Fix: status is now `no_db_url`/`no_row`/`error`/`loaded`; `pool_timeout=5`; items dated more than a day ahead are dropped; the test now uses an unreachable Postgres URL.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-world-pulse up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: K counts runs, not days. A same-day re-run (scheduler restart) moves the counter forward.
  - Mitigation: the flag is a warning only. Follow-up: carry the count forward when `run.date` matches.
- Severity: low
  - Concern: Deseret's politics feed mixes in national stories (e.g. a California wealth tax piece) under `local_politics`.
  - Mitigation: they are still fresh, readable, Utah-desk articles. Revisit if this becomes noisy.
- Severity: low
  - Concern: `nasa_news` only admits `/news-release/` links, which is 1 of the feed's 10 items. This predates this patch and was left alone.
- Severity: low
  - Concern: hardware now comes from feeds, so the curiosity gap-fill for it fires only when those feeds come up empty. That is an intentional change, approved by Juniper.
- Severity: low
  - Concern: the arXiv source PR #2638 also edits `sources.yaml`, `pipeline.py`, `digest.py`, and `settings.py`. Whichever merges second needs a rebase.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2641

🤖 Generated with [Claude Code](https://claude.com/claude-code)
