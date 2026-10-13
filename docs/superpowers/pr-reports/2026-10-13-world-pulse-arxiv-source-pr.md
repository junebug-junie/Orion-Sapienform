## Summary

- Orion's daily world pulse now includes up to 4 new arXiv papers a day, picked from cs.AI, cs.LG, cs.CL and q-bio.NC (one from each, newest first), through the official arXiv export API.
- Each paper becomes a normal digest card in the existing `ai_technology` section, so Hub turns it into a `digest_item` reading seed on its `https://arxiv.org/abs/<id>` page: Stage 1, Stage 2, reading links and curiosity seeds all apply unchanged.
- Papers Orion already queued or read are skipped. Every arXiv URL shape in the reading queue (abs, pdf, ar5iv, `vN` suffix) is reduced to one bare id first, so the source never re-picks a paper and `already_read` skips stop recurring.
- Papers get their own 4 digest slots outside the 12-card news cap, so they do not push any news section out.
- arXiv's API terms are enforced in code: at least 3 s between requests (process-wide) and an honest `orion-world-pulse` User-Agent.
- New CI workflow runs the world-pulse test suite (none existed).

## Outcome moved

Before: every arXiv read was an ad-hoc `kind=reading` seed with no section (5 in the queue since 09-20); the feed had no arXiv source. After: 4 papers/day enter the feed with section `ai_technology`.

Live check today (real API, real reading queue, read-only, `evals/run_arxiv_source_eval.py`):

```text
already-seen arXiv ids in reading queue: 5
requests=4 min_gap_s=3.00 cap=4
- [cs.AI] 2610.13948  Can Language Models Play the Wikipedia Game?
- [cs.LG] 2610.13935  Hide and Seek: HIDE Concepts to SEEK explanations for black box models
- [cs.CL] 2610.13912  RSI-Anything: Unlocking Recursive Self-Improvement with World Models
- [q-bio.NC] 2610.13918  DyssomniaNN: Multi-Scale Spiking Neural Modeling of Human Sleep Disorders and Perceived Restedness
PASS
```

## Current architecture

- `services/orion-world-pulse` fetches sources from `config/world_pulse/sources.yaml` via strategy adapters (`rss`, `atom`, `sitemap`, `html_section`, `manual_urls`). Strategy `api` existed in the schema but the registry returned `[]` for it.
- Category comes from `source.categories[0]`; articles cluster into items; `curate_digest_items` keeps at most 12 cards, at most 2 per section. Live: both caps were binding every day (12 cards, ai_technology got 1, the Hugging Face blog).
- Hub (`orion/world_pulse_read/queue.py`) reads the last 5 digests and turns each digest item into a seed using the item's first article URL. `already_read` matching is exact-URL.

## Architecture touched

- New adapter `app/services/ingest/arxiv_api_adapter.py`; `registry.py` dispatches strategy `api` by API host (only `export.arxiv.org` has an adapter).
- `digest_policy.reserved_source_slots` (new, `digest.py` + `pipeline.py`): {source_id: n} cards outside the total and per-section caps. Reported `max_digest_items_total` includes them (16).
- `clustering_policy.singleton_source_ids` (new, `cluster.py`): a paper is never merged with another item. Without it, two papers with similar titles merge into one card and only one becomes a seed (test proves the merge happens by default).
- No schema, bus, or Hub change. Sections are a real contract (`DailyWorldPulseSectionsV1`, `extra="forbid"`, consumed by Hub/renderers), so no new section label was added; `ai_technology` is reused.

## Existing-mechanism check

- `read_evidence.same_site` already treats `export.arxiv.org` as the same site as `arxiv.org`; seed URL is the abs page (5 successful reads since 09-20).
- `url_looks_like_section_index` returns False for abs URLs; `normalize_source_url` leaves them unchanged (checked).
- Reused: `to_candidate` domain/path allowlist, `max_articles_per_day` as the daily cap, `user_agent` field, the existing lazy-engine pattern from `curiosity.py`.

## Files changed

- `services/orion-world-pulse/app/services/ingest/arxiv_api_adapter.py`: new. Atom parsing, id normalization, round-robin selection, reading-queue dedupe, request spacing.
- `services/orion-world-pulse/app/services/ingest/registry.py`: `api` strategy dispatch by host.
- `services/orion-world-pulse/app/services/digest.py`: `reserved_source_slots`.
- `services/orion-world-pulse/app/services/cluster.py`: `singleton_source_ids`.
- `services/orion-world-pulse/app/services/pipeline.py`: passes reserved slots; reported cap includes them.
- `services/orion-world-pulse/app/settings.py`, `.env_example`: `WORLD_PULSE_READ_HISTORY_DB_URL`.
- `config/world_pulse/sources.yaml`: `arxiv_research` source (enabled), reserved slots, singleton clustering.
- `scripts/sync_local_env_from_example.py`: `WORLD_PULSE_READ_HISTORY_` sync prefix (without it the key never lands; verified).
- `services/orion-world-pulse/tests/test_arxiv_source.py`, `tests/fixtures/sample_arxiv_atom.xml`: tests.
- `services/orion-world-pulse/evals/run_arxiv_source_eval.py`: live read-only eval.
- `services/orion-world-pulse/README.md` (new), `docs/world_pulse_dev.md`: docs.
- `.github/workflows/orion-world-pulse-tests.yml`: new CI gate for the service.

## Schema / bus / API changes

- Added: none (schema). New optional config keys inside existing free-form dicts: `digest_policy.reserved_source_slots`, `clustering_policy.singleton_source_ids`.
- Removed / Renamed: none.
- Behavior changed: strategy `api` now fetches for `export.arxiv.org` sources; other `api` sources still fetch nothing. Daily digest grows from <=12 to <=16 cards.
- Compatibility notes: digest payload shape unchanged; Hub needs no redeploy.

## Env/config changes

- Added keys: `WORLD_PULSE_READ_HISTORY_DB_URL` (orion-world-pulse), shipped filled with the conjourney DSN (same DB/host as `ORION_ACTION_OUTCOME_DB_URL`).
- Removed / Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes; verified `WORLD_PULSE_READ_HISTORY_DB_URL` present in `services/orion-world-pulse/.env` of the primary checkout.
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=.:services/orion-world-pulse pytest services/orion-world-pulse/tests -q   -> 115 passed
  (also in a clean venv from services/orion-world-pulse/requirements.txt + pytest-asyncio)
pytest tests/test_world_pulse_read_seeds.py tests/test_world_pulse_read_schemas.py   -> 8 passed
python3 scripts/check_env_template_parity.py                                           -> PASS
git diff --check                                                                       -> clean
Mutation: reserved slot 4 -> 0 in sources.yaml => pipeline seed test fails (0 paper seeds)
```

## Evals run

```text
services/orion-world-pulse/evals/run_arxiv_source_eval.py --dsn <live conjourney> -> PASS (output above)
```

## Docker/build/smoke checks

```text
docker build -f services/orion-world-pulse/Dockerfile .  -> ok
in-image: registry loads, arxiv_research is the only api source, User-Agent as above
scripts/safe_docker_build.sh orion-world-pulse build      -> refused in worktree: no root .env (expected; not deployed)
```

## Review findings fixed

Review ran in a subagent against `git diff origin/main...HEAD`. No must-fix; all should-fix items fixed.

- Finding: arXiv articles made `ai_technology` read as covered, which would hide a tech-news outage and suppress that section's curiosity gap-fill.
  - Fix: `_compute_coverage` excludes reserved-slot sources from articles, items and enabled sources.
  - Evidence: `test_arxiv_does_not_count_as_ai_technology_news_coverage` (status flips to covered without the exclusion); pipeline test asserts equal `articles_accepted` with and without arXiv.
- Finding: `metadata.query_category` could name the wrong feed for a cross-listed paper.
  - Fix: `select_papers_with_feed` returns the feed each pick was taken from.
  - Evidence: `test_query_category_names_the_feed_the_paper_was_taken_from`.
- Finding: dedupe SQL missed `ar5iv.org` URLs; host check matched `evilar5iv.org`.
  - Fix: SQL matches both hosts; host check is exact-or-subdomain.
  - Evidence: `test_seen_sql_covers_ar5iv_org_and_host_check_is_exact`.
- Finding: papers could take worth-reading/watching or situation-change slots on a busy day.
  - Fix: reserved-source clusters fill their own lists, appended after the ordinary caps.
  - Evidence: pipeline test compares news worth-reading/watching lists against a no-arXiv baseline.
- Nit: all-empty feeds looked like a quiet day. Fix: `world_pulse_arxiv_all_feeds_empty` warning.
- Nit (not changed): `allowed_uses.stance_capsule` is not enforced by `capsule.py` for any source (pre-existing). Papers stay out of the capsule today because a single tier-3 item scores ~0.6 < 0.65. Follow-up, not this PR.
- Nit (kept): separate `WORLD_PULSE_READ_HISTORY_DB_URL` rather than reusing `ORION_ACTION_OUTCOME_DB_URL`: different purpose, can be pointed at a read-only role without touching the curiosity gate.

Live dedupe proof (real API): marking today's cs.AI and q-bio.NC picks as seen via their pdf and ar5iv URL forms made the source pick 2610.13935 / 2610.13931 / 2610.13912 / 2610.13419 instead.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-world-pulse up -d --build
```

## Risks / concerns

- Severity: low. Concern: arXiv weekend/holiday gaps mean the newest listing barely changes; with dedupe the source then reaches deeper into the 25-per-category window and may return fewer than 4. Mitigation: that is the honest result; logged as `world_pulse_arxiv_selected ... picked=N`.
- Severity: low. Concern: if the reading-queue DB is unreachable, dedupe is skipped (logged `world_pulse_arxiv_dedupe_skipped`). A repeat abs URL is then caught by Hub's exact-URL `already_read` gate, so no re-read, but a slot is wasted. Mitigation: log line; same DB the service already uses.
- Severity: low. Concern: preprints are unreviewed. Mitigation: trust tier 3, corroboration required, no graph or prior writes.
- UNVERIFIED until deploy: the first live run producing 4 `ai_technology` seeds on arxiv.org/abs URLs. Check: `select url, section from world_pulse_read_seed where url like 'https://arxiv.org/abs/%' and kind='digest_item' order by created_at desc limit 4;` after the next 12:01 UTC run.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
