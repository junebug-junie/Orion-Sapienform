## Summary

- Every BBC and NPR world-pulse reading since 10-05 failed (`no_read_evidence`): 18 BBC + 17 NPR, 0 ever read. Each one burned a full reading turn and a Wallet A slot.
- Root cause is not the page, it is the fetch tool being refused. BBC: Claude Code's WebFetch refuses the domain before any request (`Claude Code is unable to fetch from www.bbc.co.uk`, also `www.bbc.com`, `feeds.bbci.co.uk`). NPR: `robots.txt` has `User-agent: Claude-User / Disallow: /` and the edge resets that agent, so WebFetch hangs to `timeout of 60000ms exceeded`.
- Both are deliberate publisher/tool policies. Per Juniper's rule (no fake UAs, no archive/lite-site routing), the fix marks these sites unreadable and skips the seed before Stage 1, with a reason naming the block, instead of spending a turn.
- New `orion/world_pulse_read/reader_blocks.py` (site list with the live refusal evidence). New BBC/NPR requests are recorded `skipped` on arrival (`enqueue_seeds`, any requester); rows queued before deploy are skipped at claim time unless a turn for them is already submitted. Regression tests per failing pattern, real-Postgres tests, and a replay eval over the real 10-05..13 seed outcomes.

## Outcome moved

Before (live `world_pulse_read_seed`, 10-05..13, non-YouTube `no_read_evidence`): bbc.co.uk 18, npr.org 17, nasa.gov 2 (`thin_fetch`).
After (replay eval on the same rows): 35/35 BBC/NPR seeds skipped before a turn (`skipped`, `last_error=source_blocks_reader:<site>:<why>`, no Wallet A debit); 0 of 16 recently-read sites caught. NASA is reachable (9 done since 10-01) and is left alone.

Live re-fetch of two real failed URLs, 2026-10-13, through the same WebFetch tool:
- `https://www.bbc.co.uk/news/articles/cmn06w5x9g25o?at_medium=RSS&at_campaign=rss` -> `Claude Code is unable to fetch from www.bbc.co.uk`
- `https://www.npr.org/2026/10/12/nx-s1-5998948/new-white-house-press-secretary-a-devoted-fan-of-trump-and-maga` -> `timeout of 60000ms exceeded`
From the FCC container: NPR with `-A 'Claude-User ...'` -> HTTP/2 INTERNAL_ERROR in 0.17s; same URL default curl agent -> 200 / 143 KB; BBC default agent -> 200 / 414 KB. The text exists; getting it would mean routing around the publishers' Claude block, which is out of bounds. No usable text is claimed for these sources.

## Current architecture

Stage 1 (`WorldPulseReadPipeline.tick`) claims a seed, validates the URL, skips section-index pages (world_pulse only), then runs a harness reading turn locked to WebFetch/WebSearch. Read evidence comes from the tool trace; a turn without a same-site fetch of >=200 chars is terminal `no_read_evidence`.

## Architecture touched

- Reading ingress (`orion/world_pulse_read/queue.py`): `enqueue_seeds` records a blocked URL as `skipped` with the block reason (`INSERT_SQL` now skips on a non-null arrival reason, not only on an alias). No queue slot, no tick; `reading_status` returns `skipped` + reason immediately.
- Hub Stage 1 pipeline: claim-time backstop for rows queued before deploy; yields to a seed with an unconsumed stage-1 durable turn (`has_pending_durable_turn`) so an in-flight result is never stranded.
- No schema, bus channel, or env change. Claim-time skip reuses `mark_seed_skipped` and the existing `reading.lifecycle.v1` `stage1_failed` event; the lifecycle has no "skipped" stage, so the `source_blocks_reader:` prefix on `error` marks it.

## Files changed

- `orion/world_pulse_read/reader_blocks.py`: site list + `reader_block_reason(url)`, with the live evidence.
- `orion/world_pulse_read/queue.py`: skip on arrival in `enqueue_seeds`; `INSERT_SQL` arrival-reason skip; `has_pending_durable_turn`.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py`: claim-time backstop before Stage 1.
- `services/orion-hub/tests/reading_queue_fakes.py`: fake mirrors the new `INSERT_SQL` rule.
- `services/orion-hub/tests/test_reading_postgres.py`: real-Postgres arrival skip for a Juniper request and a Curiosity request; durable-turn guard query.
- `orion/world_pulse_read/tests/test_reader_blocks.py`: live failing URLs per pattern, lookalike/negative hosts.
- `services/orion-hub/tests/test_world_pulse_read_pipeline.py`: pre-deploy BBC and NPR rows skipped with no read, no debit, terminal lifecycle; a blocked seed with a submitted turn is left to finish.
- `services/orion-hub/evals/test_reader_blocks_replay_eval.py` + `fixtures/reader_blocks_live_2026-10-13.json`: replay over real seed outcomes.

## Schema / bus / API changes

- Added: none. New `last_error` value `source_blocks_reader:<site>:<why>` on `status='skipped'` rows; tick return `skipped_reader_blocked`.
- Removed / Renamed: none.
- Behavior changed: BBC/NPR seeds no longer run a reading turn; `INSERT_SQL` skips any row given an arrival reason (previously only `ALREADY_READ` aliases carried one).
- Compatibility notes: already-failed rows untouched.

## Env/config changes

- None. No `.env_example` touched.

## Tests run

```text
pytest orion/world_pulse_read/tests services/orion-hub/evals/test_reader_blocks_replay_eval.py -q  -> 118 passed
pytest services/orion-hub/tests/test_world_pulse_read_{pipeline,stage2,queue}.py tests/test_world_pulse_read_url_filters.py services/orion-hub/evals/test_reading_handoff_eval.py -q -> 155 passed
RUN_READING_POSTGRES=1 pytest services/orion-hub/tests/test_reading_postgres.py -q -> 47 passed
Mutations: claim gate off -> 2 pipeline tests fail; durable guard off -> in-flight test fails; arrival skip off -> 2 Postgres tests fail.
pytest services/orion-hub/tests -q -> 3516 passed, 33 failed, all in files that do not touch reading (mind tab, recall canary, substrate effect, UI static); see CI for the main baseline.
```

## Evals run

```text
pytest services/orion-hub/evals/test_reader_blocks_replay_eval.py -q -> 2 passed (35/35 BBC/NPR skipped, 0/16 read sites caught)
```
This is a replay over a frozen fixture of real seed rows, not a live check; the live path is UNVERIFIED until Hub restarts.

## Docker/build/smoke checks

```text
Not deployed (per instructions). Live path UNVERIFIED until Hub restarts; check with:
SELECT last_error, count(*) FROM world_pulse_read_seed WHERE last_error LIKE 'source_blocks_reader%' GROUP BY 1;
```

## Review findings fixed

- Finding (should): the claim-time skip could strand a seed whose durable turn was already submitted (operator retry then refuses with `active_durable_binding`).
  - Fix: skip only when no unconsumed stage-1 `reading_durable_turn` exists (`has_pending_durable_turn`).
  - Evidence: `test_reader_blocked_seed_with_submitted_turn_is_not_stranded`; fails with the guard removed.
- Finding (should): checking only at claim time still spent a queue slot and a tick per blocked seed, and a requester saw `queued`.
  - Fix: skip on arrival in `enqueue_seeds`; claim-time stays as the backstop for already-queued rows.
  - Evidence: `test_reader_blocked_source_is_skipped_on_arrival_for_any_requester` (real Postgres, Juniper + Curiosity requesters, `reading_status` = `skipped` + reason, nothing claimable).
- Finding (nit): `stage1_failed` event vs `skipped` row status.
  - Fix: comment at the gate naming the `source_blocks_reader:` prefix as the marker; no schema change.
- Finding (nit): requester coverage untested beyond world_pulse.
  - Fix: covered by the Postgres arrival test above.
- Finding (nit): eval grouped by host only.
  - Fix: eval also asserts every BBC/NPR-family host in the fixture is caught.
- Finding (nit): the static list can go stale silently.
  - Not fixed in code: each entry carries its evidence; the skip count per site is queryable (below). Listed under risks.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: a static site list can go stale if BBC/NPR or the tool policy changes, and nothing re-checks it. Mitigation: each entry names its evidence (as of 2026-10-13); removing one is a one-line revert and nothing persisted depends on it. Spot check: `WebFetch` the site once; per-site skip counts via the SQL below.
- Severity: low. Concern: BBC/NPR stay in the world-pulse digest (RSS excerpts) but are never deep-read. That is the honest state; adding a readable source for the same beats is a separate call.
- YouTube skipped per Juniper. NASA's 2 `thin_fetch` are short digests from a reachable site, not a block.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
