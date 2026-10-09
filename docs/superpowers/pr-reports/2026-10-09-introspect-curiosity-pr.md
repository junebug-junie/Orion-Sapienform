# feat(introspect): `curiosity` tool — orion-introspect slice 3

## Summary

- Orion can look up their own past curiosity runs during a turn: recent runs, one run in full by id, runs found by meaning, and their open self-questions.
- orion-hub answers over the bus (`orion:introspect:curiosity:request`). It reuses the Curiosity tab's own run join (`curiosity_run_store` + `orion/curiosity/run_story.py`) inside read-only transactions.
- Labels are per item. A run with a write-up is `unsettled`, because the text is what Orion concluded at the time. Failed runs, and runs that finished without writing anything, are returned as short `record` items, so failures stay visible.
- Search by meaning uses a new `orion_curiosity` Chroma index the Hub builds itself. The floor is 0.65, calibrated on all 371 live write-ups.
- Fixes a Curiosity tab bug. Since 09-14 the tab listed reading, reverie-visual, compactor, episode-distill and daily-letter runs as "World question": about 350 of 915 runs in the 90-day view.
- Fixes a dreams search flaw shared with this slice. An empty search could report "nothing matched" before the index had actually stored the latest records.

## Outcome moved

When asked "what did you find out about X last week?", Orion can check instead of reconstructing. "Couldn't check" is never reported as "nothing there". The Curiosity tab now shows only curiosity runs.

## Current architecture

Slices 1–2 (#2381, #2384, #2537) shipped `reading_results` and `dreams`. Curiosity runs lived in Postgres and the graph, readable only through Hub UI endpoints. The spec named orion-substrate-runtime as the owner, but that service only stores candidate menus.

## Architecture touched

- orion-hub: new responder + index loop (`scripts/curiosity_introspect.py`, `scripts/curiosity_introspect_listener.py`), started in `main.py` beside the reading listener.
- `orion/curiosity/run_story.py`: admission rows of other workflows are dropped. Write-ups are also used as a source for finding runs, so the tab still lists write-up-only runs when the graph is down.
- `orion/introspect/`: `curiosity` tool, brief line, shared `confirmed_complete_as_of` rule (used by dreams too).
- Contracts: `CuriosityArguments`, `clip_json_text`, new request channel, orion-hub as producer on `orion:introspect:result:*` and `orion:vector:semantic:upsert`.

## Files changed

See `git diff --stat origin/main...HEAD` (36 files). Main ones:
- `services/orion-hub/scripts/curiosity_introspect*.py`, `curiosity_run_store.py`, `main.py`, `app/settings.py`, `.env_example`, `README.md`: responder, index, config, docs.
- `services/orion-hub/evals/run_curiosity_search_calibration.py`: similarity floor calibration.
- `orion/curiosity/run_story.py`: tab fix.
- `orion/schemas/introspect.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`, `orion/schema_skew_discovery.py`, `config/metrics/metric_definitions.lock.json`: contract.
- `orion/introspect/{tools,brief,semantic_index,transport}.py`, `scripts/smoke_introspect.py`: tool side.
- `services/orion-dream/app/introspect_listener.py`: shared index-complete rule.
- `.github/workflows/orion-reading-tests.yml`: Hub curiosity tests in CI.

## Schema / bus / API changes

- Added: `orion:introspect:curiosity:request` (orion-hub responder); `CuriosityArguments`; `curiosity` operation; orion-hub as producer on `orion:introspect:result:*` and `orion:vector:semantic:upsert`.
- Removed / renamed: none.
- Behavior changed: the Curiosity tab drops non-curiosity admission runs. Dream search treats the index as complete only after Chroma confirms every record, measured from 10 minutes before the pass started.
- Compatibility: additive; `HARNESS_FCC_INTROSPECT_ENABLED=0` removes the server.

## Env/config changes

- Added keys (orion-hub): `HUB_CURIOSITY_SEARCH_CHROMA_URL`, `_EMBED_URL`, `_COLLECTION`, `_MIN_SIMILARITY` (0.65), `_INDEX_INTERVAL_SEC`, `_INDEX_BATCH`. Shipped on.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, all 6 keys are in the primary checkout's `services/orion-hub/.env`.
- Skipped keys: none.

## Tests run

```text
orion-reading-tests.yml command list (fresh venv): 1050 passed, 64 skipped
orion/introspect/tests: 154 passed, 2 skipped (MCP tests pass in CI venv)
Hub curiosity tests: 271 passed, 3 skipped
Dream introspect: 53 passed; all dream tests 108 passed; schema-skew 18 passed
Gates: definition drift, metric lineage, bus reply channels, env template parity,
env key single-source, async routes, stdlib shadow: pass
Each fix's test was mutation-checked (fails with the fix reverted).
```

## Evals run

```text
run_curiosity_search_calibration.py on 371 live write-ups (read-only):
related questions best hit 0.745-0.874; unrelated 0.515-0.626 -> floor 0.65 (0.119 gap).
One ranking miss remains (a related run outranks the expected one, 0.793 vs 0.770).
Offline reading evals: 16 passed.
```

## Docker/build/smoke checks

```text
Read-only against live Postgres (no bus): recent mode 1.4 s, 499 runs; one run by id
serializes to 9,652 chars (budget 12,000); a reverie run id answers unknown with the graph absent.
Bus smoke: UNVERIFIED until Hub deploy (scripts/smoke_introspect.py --tool curiosity).
```

## Review findings fixed

- Finding: a found run with only a graph clock came back `ok=true, items=[]` when the graph was down (46 live write-ups).
  - Fix: the write-up timestamp is used as the clock (`clock_from: write_up`); not-found while the graph is down is unknown.
  - Evidence: listener tests with a fake graph reader, up and down.
- Finding: the `line` filter was applied after the 10-hit cut, which could return a false "nothing matched".
  - Fix: `line` is stored in the index metadata and filtered inside Chroma; it is hashed so existing docs re-upsert.
  - Evidence: 10 investigate hits + 1 self_inquiry test; live labels match the tab on all 353 shared runs.
- Finding: the index was marked complete once upserts were published, before they were stored (dreams too).
  - Fix: shared `confirmed_complete_as_of`: complete only when nothing was stale, minus a 10-minute margin.
  - Evidence: real `index_docs` against a fake Chroma that withholds upserts.
- Finding: 5 list items could serialize to 18k chars.
  - Fix: list text and short fields are clipped by serialized JSON length.
  - Evidence: 5-item worst case through the MCP server is 11,338 chars.
- Minor: a `since` older than 90 days is refused instead of silently undercounting. Self-sense and workflow-less admission rows are pinned as kept.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run from the primary checkout on main after merge.

## Risks / concerns

- Severity: low. Concern: search answers "unknown" until the `orion_curiosity` index is built (~37 passes, about 3 hours) plus one confirming pass. Mitigation: intended; recent and by-id modes work immediately.
- Severity: low. Concern: self-sense runs show as "completed, no write-up"; their answers in `self_sense_eval_log` are not surfaced. Mitigation: follow-up.
- Severity: low. Concern: with Hub graph credentials missing, unknown run ids answer "unknown" instead of "not found". Mitigation: deliberate, so absence is never claimed without checking.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
