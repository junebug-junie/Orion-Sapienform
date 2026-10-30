# fix(topic-foundry): stop retraining and re-enriching on every Hub restart

## Summary

Hub restarts 7 to 11 times a day during merges. Each restart sent topic-foundry a train request and an enrich request for each model. Topic-foundry did the same work again each time. This PR stops the repeats.

- A training request now gets the existing run when a run with the same spec completed in the last 24 hours. Before, only queued or running runs counted. (`fetch_run_by_spec_hash`, `repository.py`)
- Enrichment now reuses an earlier LLM answer when the segment text is the same. Before, every new run sent all of its segments to the LLM again, because each run makes new segment IDs. (new table `topic_foundry_enrichment_cache`)
- The enrich `limit` (200 per tick) now counts only new LLM calls. Reused answers are free. Self Atlas can now finish: it has 14,553 segments but only about 1,984 different texts.
- `POST /runs/train` accepts `"force": true` to train again on purpose.
- Hub comments that called the startup tick "cheap and idempotent by construction" now say what is true.

## Outcome moved

Replay of the real ticks from 2026-10-29 (`services/orion-topic-foundry/evals/replay_repeat_work.py`):

| Model | Train runs before | Train runs after | Segments enriched before (any source) | LLM calls after (steady state) |
|---|---|---|---|---|
| Chat model (`orion-hub-autonomous-v4`) | 6 | 1 | 1,229 | 46 |
| Self Atlas | 2 | 1 | 1,136 | 40 |
| AI Town | 7 | 7 | 0 | 0 |
| **Total** | **15** | **9** | **2,365** | **86** |

In the steady state, the enrichment work drops by about 96%. On 10-28/29, one enrichment took about 13 s (chat) and 8 s (Self Atlas) of wall-clock time. So the enrichment time drops from about 7 hours a day to about 16 minutes a day.

"Steady state" means the cache already holds the earlier runs. The first day after deploy starts with an empty cache, so it fills once: 407 chat texts and 1,984 Self Atlas texts (`--cold-cache`: 2,391 calls). The limit of 200 new calls per tick spreads that over about 10 ticks. Before this fix, Self Atlas needed 14,553 calls per run and never finished. Now it needs 1,984, one time.

The AI Town runs fail in under one second on every tick. That is a separate fault. Failed runs never dedup, so this PR does not change them.

## Current architecture

- Hub (`services/orion-hub/scripts/main.py`, topic-foundry scheduler) runs one tick at startup and one every 86,400 s. Each tick does three steps for each of chat, AI Town and Self Atlas: train POST, enrich POST on the latest completed run, ingest.
- Hub floors the training window to the UTC day, so all ticks in one day send the same spec hash.
- Topic-foundry deduped a train request only against queued or running runs. A completed run did not count. Spec hash `569e0f36f1` trained six times on 10-29.
- Enrichment picked segments with `enriched_at IS NULL` in one run. A new run has new segment rows, so every segment was new to it.

## Architecture touched

- `orion-topic-foundry`: train route dedup, enrichment loop, one new table, one new env key.
- `orion-hub`: comments and docstrings only. No behavior change in Hub.
- No bus channel or schema registry change.

## Files changed

- `services/orion-topic-foundry/app/storage/repository.py`: dedup query also matches a recent complete run; cache read/write functions; table created in `ensure_tables()`.
- `services/orion-topic-foundry/app/storage/ddl.py`: `topic_foundry_enrichment_cache` table.
- `services/orion-topic-foundry/app/routers/runs.py`: passes the dedup window (0 when `force`); logs `train_dedup_hit`.
- `services/orion-topic-foundry/app/models.py`: `RunTrainRequest.force`.
- `services/orion-topic-foundry/app/services/enrichment.py`: cache key, cache reuse, `limit` counts fresh calls only, `_enrich_segment` returns its source so a heuristic fallback is never cached, empty LLM answers are never cached, `enrichment_cache_hits` stat, `enrichment_pass` log line.
- `services/orion-topic-foundry/app/settings.py`, `.env_example`, `README.md`: new key and docs.
- `services/orion-topic-foundry/tests/test_repeat_work_dedup.py`: regression tests for both bugs.
- `services/orion-topic-foundry/tests/test_run_recovery.py`: stub follows the new `_enrich_segment` return shape.
- `services/orion-topic-foundry/evals/replay_repeat_work.py`: read-only replay of one real day.
- `services/orion-hub/scripts/main.py`, `scripts/concept_atlas_routes.py`, `app/settings.py`: comments now true.

## Schema / bus / API changes

- Added: table `topic_foundry_enrichment_cache (cache_key TEXT PK, enricher TEXT, enrichment JSONB, created_at TIMESTAMPTZ)`. Topic-foundry creates it at startup with `CREATE TABLE IF NOT EXISTS`, like its other tables. No manual migration file, so the post-pull migration gate does not apply.
- Added: optional `force: bool = false` on `POST /runs/train`.
- Removed: none.
- Renamed: none.
- Behavior changed: `POST /runs/train` can now return a run with `status: "complete"`. Hub already handles that (it logs it; `should_retry_startup_tick` treats any `triggered: true` as done). Enrich `limit` now counts new LLM calls, not all segments.
- Compatibility notes: additive. No bus payload change.

## Env/config changes

- Added keys: `TOPIC_FOUNDRY_TRAIN_DEDUP_COMPLETE_WINDOW_SECS=86400` (orion-topic-foundry). 0 turns the new check off.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-topic-foundry`: yes (`+TOPIC_FOUNDRY_TRAIN_DEDUP_COMPLETE_WINDOW_SECS`).
- skipped keys requiring operator action: none. The sync reported one older, unrelated divergence (`TOPIC_FOUNDRY_DRIFT_DAEMON` local `true`, example `false`); left as is.

## Tests run

```text
# inside the orion-topic-foundry image, worktree mounted read-only
python -m pytest -q tests
  172 passed, 5 failed
  The 5 failures also fail on main with the same image (pre-existing:
  test_chat_corpus_builder_stages, test_drift_bands x2, test_heartbeat_chassis x2).

# new tests against the OLD code (main), to prove they catch the bugs
python -m pytest -q test_repeat_work_dedup.py   -> 5 failed, 7 errors (all 12 original tests fail)

# Hub
PYTHONPATH=<worktree> .venv/bin/python -m pytest services/orion-hub/tests/test_topic_foundry_scheduler.py -q
  40 passed

scripts/check_env_template_parity.py      PASS
scripts/check_definition_drift.py --gate  PASS
scripts/check_env_key_single_source.py    OK
scripts/check_metric_lineage.py --gate    PASS
tests/test_no_personal_terms.py           7 passed
git diff --check                          clean
```

## Evals run

```text
python3 services/orion-topic-foundry/evals/replay_repeat_work.py --day 2026-10-29
  steady state: TOTAL train 15 -> 9, segments enriched 2365 -> LLM calls 86 (exit 0)
python3 ... --day 2026-10-29 --cold-cache
  first day after deploy: TOTAL train 15 -> 9, LLM calls 2391 (one-time fill; exit 0)
python3 ... --window-secs 72000
  TOTAL  train 15 -> 12  (a 20 h window misses Self Atlas: its same-spec reruns were 21 h apart)
```

Read-only live check of the new dedup SQL against Postgres: for the chat model's latest spec hash it returns run `36e08f3b` (complete 2026-10-30 02:35), so the next Hub restart today would get that run instead of training.

`max_tokens` for enrichment was left unset. Live output sizes over 7 days: median 815 characters, 99th percentile about 4,000, maximum 6,468. That looks safe for a cap near 4,096 tokens, but the model may also spend tokens on thinking, and I cannot see that from stored output. With about 96% fewer calls the gain is small, so I did not cap it.

## Docker/build/smoke checks

```text
No image build: no dependency or Dockerfile change. Tests ran in the existing
orion-topic-foundry image with the worktree mounted. Not deployed (per task).
Live path after deploy: UNVERIFIED.
```

## Review findings fixed

The review ran in a subagent. It found no blockers.

- Finding: `force=true` did not overwrite the cached answer (`ON CONFLICT DO NOTHING`), so the next normal run served the old answer again.
  - Fix: the insert is now `ON CONFLICT DO UPDATE`.
  - Evidence: `test_force_bypasses_the_cache_and_overwrites_it` runs a third, normal pass and checks that it gets the forced answer.
- Finding: an empty LLM answer (`{}`, which becomes "untitled") counted as an LLM result and would be cached forever for that text.
  - Fix: `_is_substantive()` caches only answers with a real title and some content.
  - Evidence: `test_empty_llm_answer_is_not_cached`.
- Finding: the test fake did not match the real SQL, and no test caught shared dicts between duplicate texts.
  - Fix: the fake matches the new overwrite SQL. A new test checks that duplicates get separate dicts.
  - Evidence: `test_duplicate_texts_get_separate_dicts`.
- Finding: the eval compared different things and only showed the best case.
  - Fix: the column now says "segments enriched (any source)". `--cold-cache` shows the first day after deploy. The SQL is built from the parsed date.
  - Evidence: both eval runs above.
- Finding (nit): the cache key names the bus route, not the model behind it. Training code changes do not reset the train dedup. The DDL comment was incomplete.
  - Fix: README "Repeat-work guards" documents both. The DDL comment lists every key input.
- Not changed (nit): Hub logs a dedup hit as `triggered=True status=complete`. This is harmless: the retry policy and the enrich step already handle it.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-topic-foundry up -d --build topic-foundry
```

Hub needs no restart for this change (comments only). Its next normal rebuild picks the comments up.

After deploy, check:

```bash
docker logs orion-athena-topic-foundry 2>&1 | grep -E "train_dedup_hit|enrichment_pass" | tail
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select count(*) from topic_foundry_enrichment_cache"
```

## Risks / concerns

- Severity: low. Concern: the first day after deploy still pays to fill the cache (about 2,391 calls). Mitigation: this happens one time only. The limit of 200 new calls per tick still applies.
- Severity: low. Concern: if the bus route starts serving a different model, cached answers from the old model are reused. Mitigation: bump `ENRICHMENT_CACHE_VERSION` (documented in the README).
- Severity: low. Concern: a code change to training (not the spec) no longer retrains the same day. Mitigation: send `"force": true`, or wait for the next UTC day.
- Severity: low. Concern: the cache is not cleared when `_finalize_enrichment` changes how it shapes output. Mitigation: bump `ENRICHMENT_CACHE_VERSION`. Prompt, taxonomy, text and route changes already change the key.
- Severity: info. Concern: AI Town training fails in under one second on every tick (7 failed runs on 10-29). Not addressed here.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
