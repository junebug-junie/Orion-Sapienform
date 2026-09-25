# World-pulse: a read only counts if the source was actually fetched

**Base branch: `fix/world-pulse-wallet-refund` (PR #2335).** This PR stacks on
it -- both touch the same Hub world-pulse files. Merge #2335 first; GitHub then
retargets this PR to `main`.

## Summary

- Orion's world-pulse reader could mark an article "read" when the model never
  opened it. Now a Stage 1 read is only `done` if the harness's own tool trace
  shows a fetch of a non-listing page on that source's site that returned real
  content. Otherwise the seed is `failed` with `last_error=no_read_evidence`
  (or `:thin_fetch` / `:harness_unreported`, see below). The Stage 1 prompt now
  tells the reader to fetch the URL.
- The harness governor now reports which sources each turn actually fetched
  (`HarnessRunV1.source_fetches`, derived from raw WebFetch tool_use/tool_result
  pairs, never from what the model says). Hub puts it on the final frame as
  `harness_source_fetches`; Stage 1 stores the matching entries on the handoff
  as `read_evidence`.
- Stage 2 skips any handoff with no `read_evidence` before spending a Wallet B
  slot (this includes the live hollow read that is next in the Stage 2 queue).
- Old digest items no longer sit in the queue forever: every Stage 1 tick marks
  pending `digest_item` seeds older than
  `HUB_WORLD_PULSE_READ_DIGEST_ITEM_MAX_AGE_DAYS` (default 5) as `skipped` /
  `stale_digest_item`.
- The "is this a listing page?" filter now catches the roundup hub that
  produced the hollow read, topic/media pages, a national-news-release index
  and a JSON data feed -- and stops skipping real BBC articles (36 URLs).

## Outcome moved

- Failure mode closed: schema-valid, empty-substance Stage 1 reads (CLAUDE.md
  "No empty-shell cognition"). Live example: seed
  `finding:60d59b10-aaa3-4b3d-bd1d-4bf34266a73a:9b084fc0f1583da0`, claimed
  06:01:32 UTC, `done` 06:10:11, `harness_turn_trace.step_count=3` (init,
  assistant, result -- zero tool calls; grammar atoms say `tool=none`),
  `what_i_learned` begins "Metadata-only extraction; I did not fetch or read
  the article body this turn". It spent a Wallet A slot and counted as a read.
- Backlog drain: of the 142 pending seeds (read-only count, 06:42 and again
  07:15 UTC), **79 (56%)** are
  `digest_item` rows older than 5 days and are skipped on the first tick after
  deploy. Of the remaining 63, the new URL filter will skip 9 more as they
  come up (6 duplicate networkworld roundup rows, wccftech `/topic/hardware/`,
  silicondata `/media`, tomshardware `/gpus/news` -- the last was already
  caught). 54 stay readable (27 findings, 25 recent digest items, 2 readings).
- 31 pending BBC article seeds that the old filter would have silently
  discarded as `section_index_url` are now readable (most are older than 5
  days, so the stale sweep takes them instead -- recorded as stale, not as
  "not an article").

## Current architecture

- Stage 1 loop (`services/orion-hub/scripts/world_pulse_read_pipeline.py`):
  claim seed -> URL filter -> debit Wallet A -> unified turn (`reading_only`,
  WebFetch/WebSearch only) -> parse fenced JSON into
  `WorldPulseReadHandoffV1` -> Concept Atlas + journal -> `mark_seed_done`.
  The only "success" test was that the JSON validated.
- The governor (`orion/harness/runner.py`) already watched every raw FCC step
  with `ReadingReceiptTracker` (`orion/harness/reading_receipts.py`) and knew
  which fetches succeeded, but only used that to caveat reading-recommendation
  receipts. Nothing downstream could see it.
- Claim order `priority, attempts, created_at`: 39 priority-0
  finding/reading seeds (32 of them retries) ahead of 104 priority-10 digest
  items; ~5 new digest items/day vs 6 reads/day. Oldest pending 2026-09-07.

## Architecture touched

- Contract: `SourceFetchEvidenceV1` (new, nested), `HarnessRunV1.source_fetches`
  (optional, `None` = producer predates the field),
  `WorldPulseReadHandoffV1.read_evidence` (default `[]`).
- Producer: harness governor (`runner.py`, `app/bus_listener.py`).
- Transport: Hub final frame key `harness_source_fetches`
  (`orion/hub/turn_orchestrator.py::_success_frames`), only present when the
  governor reported.
- Consumers: Stage 1 gate (`orion/world_pulse_read/read_evidence.py`,
  pipeline `tick`/`_stage1_read`), Stage 2 skip.
- Queue: `skip_stale_digest_items`, `mark_stage2_skipped`
  (`orion/world_pulse_read/queue.py`).

## Files changed

- `orion/schemas/reading.py`: `SourceFetchEvidenceV1`.
- `orion/schemas/harness_finalize.py`: `HarnessRunV1.source_fetches`.
- `orion/schemas/world_pulse_read.py`: handoff `read_evidence`.
- `orion/harness/reading_receipts.py`: tracker records usable fetches (url, tool, chars).
- `orion/harness/runner.py`: carry `source_fetches` on `HarnessMotorResult`.
- `services/orion-harness-governor/app/bus_listener.py`: pass it into every motor-backed `HarnessRunV1`.
- `orion/hub/turn_orchestrator.py`: `harness_source_fetches` on the final frame.
- `orion/world_pulse_read/read_evidence.py` (new): same-site + content-floor policy, labels.
- `orion/world_pulse_read/retry.py`: `no_read_evidence:harness_unreported` is transient; plain `no_read_evidence` is not.
- `orion/world_pulse_read/queue.py`: stale sweep SQL, Stage 2 skip.
- `orion/world_pulse_read/url_filters.py`: wider listing/roundup/feed rules, item-parent rule.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py`: evidence gate, stale sweep each tick, `digest_item_max_age_days`.
- `services/orion-hub/scripts/world_pulse_read_stage2.py`: skip unread handoffs before debit.
- `services/orion-hub/app/settings.py`, `.env_example`, `scripts/main.py`, `README.md`: new key + docs.
- Tests/evals: see below.

## Schema / bus / API changes

- Added: `SourceFetchEvidenceV1 {url, tool_name, content_chars}`;
  `HarnessRunV1.source_fetches: list | None = None`;
  `WorldPulseReadHandoffV1.read_evidence: list = []`; final-frame key
  `harness_source_fetches`.
- Removed / renamed: none.
- Behavior changed: Stage 1 `done` requires evidence; Stage 2 skips unread
  handoffs; stale digest items are skipped; URL filter results changed for the
  URLs listed in `tests/test_world_pulse_read_url_filters.py`.
- Compatibility notes:
  - `HarnessRunV1` has default `extra` (ignore) and the nested model does too,
    so an old Hub ignores the new field and a new Hub reads `None` from an old
    governor. **Deploy the governor before (or with) Hub** -- a new Hub against
    an old governor fails every Stage 1 read with
    `no_read_evidence:harness_unreported` (transient: retried up to
    `max_attempts`, each retry a real turn and a real Wallet A charge).
  - `WorldPulseReadHandoffV1` is `extra="forbid"`: rolling Hub back after new
    rows are written would make the old Stage 2 reject those rows as
    `handoff_invalid`. Roll back only with Stage 2 paused, or accept that.
  - No DB migration: the stale sweep uses existing columns and the existing
    `(status, priority, created_at)` index.

## Env/config changes

- Added keys: `HUB_WORLD_PULSE_READ_DIGEST_ITEM_MAX_AGE_DAYS=5` (0 disables).
- Removed / renamed: none.
- `.env_example` updated: yes (`services/orion-hub/.env_example`).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`:
  yes -- key confirmed at `services/orion-hub/.env:751` (primary checkout,
  which is where the sync writes). `HUB_WORLD_PULSE_READ_` is in
  `SYNC_PREFIXES`. The script exits 2 on pre-existing drift in unrelated keys;
  not touched.
- skipped keys requiring operator action: none.

## Metric quality gate (read evidence)

1. Provenance: `ReadingReceiptTracker._observe_tool_result` -> a fetch tool
   (`WebFetch`, firecrawl scrape, context-mode fetch) whose tool_result is not
   `is_error` and passes `_usable_fetch_result`; `content_chars =
   len(body.strip())`.
2. Independence: not a transform of anything the gate already uses; the
   previous gate (JSON validates) is model-authored, this is tool-trace.
3. Theory anchor: a source was read in this turn iff the turn received that
   source's content from a fetch tool. Model prose ("I fetched...") is not
   evidence (live counter-example above).
4. Live data: the hollow row's trace has zero tool steps (step_count 3,
   grammar atoms `tool=none`); earlier `done` rows have 9-72 steps. Raw
   tool_result bodies are not persisted, so **the 200-char floor is
   UNVERIFIED against live WebFetch bodies**; every turn now logs
   `world_pulse_read_fetch_evidence ... chars=WebFetch:<n>,...` so it can be
   checked after deploy.
5. Existing mechanism: reused `ReadingReceiptTracker` (already computing
   `source_read` for recommendations) rather than a new parser.
6. Reversibility: optional fields with defaults; the gate is one check in the
   pipeline.

## Tests run

```text
pytest services/orion-hub/tests -k "world_pulse or reading"        -> 205 passed, 9 skipped
RUN_READING_POSTGRES=1 pytest <orion-reading-tests.yml file list>   -> 543 passed, 1 skipped
pytest orion/harness/tests orion/world_pulse_read/tests            -> 365 passed
(cd services/orion-harness-governor && pytest tests)                -> 54 passed
pytest tests/test_world_pulse_read_url_filters.py                   -> 23 passed
```

New tests fail on the pre-patch code (behavior files reverted, schemas kept):
14 failed / 4 passed across the new Stage 1, Stage 2, frame and tracker tests;
9 of the new URL-filter cases fail on the old filter; the governor
pass-through and real-Postgres stale-sweep tests fail too.

## Evals run

```text
pytest services/orion-hub/evals/test_reading_handoff_eval.py -q    -> 11 passed
```

Added `test_schema_valid_handoff_without_a_source_fetch_is_not_a_read`
(recorded live hollow response; no fetch / unreported / other-site fetch).
Existing attributed-learning cases now carry a recorded fetch, and the
empty/graph-output rejection cases assert they fail for their content, not
for a missing fetch.

## Docker/build/smoke checks

```text
Not run. Nothing deployed and no production data written (per task).
Live DB was queried read-only for the evidence above.
```

## Review findings fixed

Review ran in a subagent against `origin/fix/world-pulse-wallet-refund...HEAD`.
It confirmed the evidence path is real in production: `reading_only` FCC turns
run with `--tools WebFetch,WebSearch`, every stream-json line reaches
`ReadingReceiptTracker.observe`, and every governor `HarnessRunV1` path that can
yield a `final` frame carries `source_fetches`. No blockers.

- Finding: the 200-char floor measures WebFetch's answer to a reader-written
  prompt, and a cross-host `REDIRECT DETECTED` notice (not `is_error`) would
  pass it.
  - Fix: redirect notices no longer count as usable fetch content
    (`reading_receipts.py`); a same-site fetch under the floor gets its own
    label `no_read_evidence:thin_fetch` so the floor can be calibrated from real
    rows; the prompt asks WebFetch for the full text.
  - Evidence: `test_webfetch_redirect_notice_is_not_source_content`,
    `test_near_empty_fetch_result_is_not_evidence` (asserts `:thin_fetch`).
- Finding: the prompt never told the reader to fetch, and still said "if thin,
  return JSON anyway".
  - Fix: prompt now says fetch first, and that a turn with no successful fetch
    is discarded.
  - Evidence: `test_stage1_prompt_tells_the_reader_to_fetch`.
- Finding: same-site matching let a fetch of the site's homepage or `/news`
  count as reading the article.
  - Fix: evidence must be a non-listing page on the site
    (`source_page_candidates` reuses `url_looks_like_section_index`). Exact-path
    matching was rejected on purpose: live reads used YouTube oembed and the
    arXiv export API.
  - Evidence: `test_fetch_of_the_sites_homepage_is_not_evidence`,
    `test_sibling_endpoints_used_by_live_reads_count`.
- Finding: the stale sweep could skip a digest item that a finding/reading
  request was aliased onto, silently dropping that request (latent: 0 such
  rows live today).
  - Fix: `NOT EXISTS` a non-digest alias in the sweep; comment corrected
    (retry-pending rows are swept too).
  - Evidence: real-Postgres `test_stale_sweep_keeps_a_digest_item_a_request_is_aliased_to`.
- Finding: Stage 2 skip left `stage2_started` without a terminal event, and
  `reading_status` would report `stage1_completed` forever.
  - Fix: publish `stage2_failed` with `error=no_read_evidence`; map
    `done`/`skipped` to `skipped`.
  - Evidence: Stage 2 skip test asserts the last lifecycle event;
    `test_reading_status_reports_stage2_skip_as_skipped`.
- Finding: the new "latest + news word" rule caught real article slugs
  (`nvidia_latest_gpu_news`, `latest-updates-on-merger-with-x`).
  - Fix: rule only fires when the slug is listing words plus at most one topic
    word.
  - Evidence: `test_latest_news_article_slugs_are_not_roundups`.
- Finding: deploy-order / rollback hazards.
  - Fix: documented in README and below (governor first; pause Stage 2 before
    a Hub rollback).
- Not changed (nits): a table test over all five governor paths; emitting
  `harness_source_fetches` only on `reading_only` turns (it is harmless on
  chat frames). Pre-existing filter false positives the reviewer noted
  (`huggingface.co/blog/smollm3`, `anthropic.com/news/claude`,
  `.../releases/tag/v1.2`) are not introduced by this patch and are left alone.

## Restart required

Governor first, then Hub (from a worktree on the merged branch):

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium. Concern: 200-char content floor is uncalibrated against
  live WebFetch bodies. Mitigation: per-turn `world_pulse_read_fetch_evidence`
  log with char counts; a real read that trips it fails with a distinct
  `no_read_evidence` label and is visible on `/world-pulse-read/api/status`.
- Severity: low. Concern: same-site matching misses a source that redirects to
  a different domain (e.g. a `.co.uk` -> `.com` move) -- a real read would be
  failed. Mitigation: visible label; widen `same_site` if it shows up.
- Severity: low. Concern: deploy order (see Compatibility notes).
- Out of scope, seen live: the same URL is pending under several seeds
  (8x Rubin press release, 6x networkworld roundup, 7x one YouTube video) --
  cross-run findings are not being aliased. Not fixed here.

## PR link

(filled in after push)
