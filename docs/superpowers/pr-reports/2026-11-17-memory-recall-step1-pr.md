# Memory recall step 1 + mesh-edge leak check

Fixes #3533
Fixes #3409

## Summary

- On every chat turn, Orion now looks through their lived memories (`episode_memory`, Juniper's stance messages, People-page facts) by meaning and by exact words. A local reranker scores the matches. At most 4 that clear a quality bar go into DURABLE MEMORY, each labelled with kind, record id and date. When nothing fits, nothing goes in.
- Superseded, rejected, expired and removed records are never served. Chroma is only an index: every hit is checked against Postgres, and the chosen ids are re-read right before serving. The index loop also deletes their copies from Chroma.
- orion-vector-host gains `POST /rerank`, a local cross-encoder (`BAAI/bge-reranker-v2-m3`) on the athena Tesla P4.
- The mesh-edge check (#3409): GitHub writes and email to anyone but Juniper are refused when they quote a core-page sentence or a stored-memory sentence verbatim. It covers Orion's `gh`/`curl` writes in FCC turns (a PreToolUse hook), Hub's PR comments, and non-Juniper email. If the check cannot run, the write is refused.
- Replay eval against Juniper's real messages. The bank is host-side only. The committed eval uses made-up fixtures.

## Outcome moved

The miss that started this was the 2026-11-14 chat turn, where Orion said they had no memory of an event that `episode_memory` held. In replay, that memory is now in the injected lines.

| | Before | After |
|---|---|---|
| Referring messages whose target lands in DURABLE MEMORY | 0 of 11 (no chat reader of these stores) | **9 of 11 (82%)**, including the 2026-11-14 miss |
| Small-talk messages that inject anything | n/a | 3 of 23 (13%) |
| Lines per message, last 150 real messages | 0 | 1.9 average |
| Lived-memory time per turn | 0 | p50 1.4 s, p90 2.5 s, max 3.2 s |

The lived-memory step starts at intake and runs while the other recall backends fetch. Those already take a 0.6-1.8 s median, so the added wall time is the part that sticks out past the fetch. That added wall time is **UNVERIFIED** live.

The two misses are both implicit references. Step 2 (#3534: query rewrite and link-following) targets those.

## Current architecture

- orion-recall's vector backend was amputated: `RECALL_ENABLE_VECTOR=false`, 0 hits in 16,168 recalls.
- Nothing on the chat recall path read `episode_memory`, stance crystallizations or `person_fact`.
- No reranker existed.
- No GitHub or email writer checked for core-page or memory text.

## Architecture touched

- **orion-recall**
  - `app/lived_memory.py`: the reader plus the index loop.
  - `app/worker.py`: the continuity phase appends the lived block. The task starts at intake. Pressure heuristics ignore lived items.
  - `app/main.py`: starts the reader and the index loop on the existing asyncpg pool.
- **orion-vector-host**: `app/reranker.py` and `POST /rerank`, on a GPU reservation for the P4, with a persistent HF cache volume.
- **Shared**
  - `orion/memory/lived_memory.py`: the serving SQL and the document shape.
  - `orion/memory/mesh_edge.py`: the guard.
  - `orion/fcc/mesh_edge_hook.py`: the FCC hook, installed on every turn by `orion/harness/fcc_motor.py`.
  - `orion/notify/transport.py`: the email check.
  - `orion/introspect/semantic_index.py`: `indexed_ids` and `delete_ids`.
- **orion-hub**: `scripts/session_roundup_loop.py`. The PR commenter refuses guarded text.
- **orion-harness-governor**: `psycopg2-binary` and `ORION_MESH_EDGE_PG_DSN`.

## Files changed

- `orion/memory/lived_memory.py`: which rows may be served, and one document per row.
- `services/orion-recall/app/lived_memory.py`:
  - dense + BM25 candidates (names weighted), live gating, rerank;
  - the bar, logit-unit ranking, the re-check, render;
  - the index loop with deletes.
- `services/orion-recall/app/{worker,main,settings}.py`, `docker-compose.yml`, `.env_example`, `README.md` §17: wiring and config.
- `services/orion-recall/scripts/backfill_lived_memory_index.py`: first backlog / rebuild (CLAUDE.md §14).
- `services/orion-recall/evals/run_lived_memory_replay_eval.py`, `evals/test_lived_memory_replay_eval.py`, `evals/fixtures/lived_memory_replay_fixture.json`: the replay eval and its fixture version.
- `services/orion-recall/tests/test_lived_memory.py`: unit and seam tests.
- `services/orion-vector-host/app/{reranker,main,settings}.py`, `docker-compose.yml`, `.env_example`, `README.md`, `tests/test_rerank_endpoint.py`: `/rerank`.
- `orion/memory/mesh_edge.py`, `orion/memory/tests/test_mesh_edge.py`: the guard.
- `orion/fcc/mesh_edge_hook.py`, `orion/fcc/tests/test_mesh_edge_hook.py`, `orion/harness/fcc_motor.py`: the FCC gh/curl hook.
- `services/orion-hub/scripts/session_roundup_loop.py`, `services/orion-hub/tests/test_session_roundup_loop.py`: the PR comment check.
- `orion/notify/transport.py`, `services/orion-notify/tests/test_email_mesh_edge.py`: the email check.
- `services/orion-harness-governor/{requirements.txt,.env_example,docker-compose.yml}`: the guard's DB access.
- `orion/bus/channels.yaml`: orion-recall added as a producer of `orion:vector:semantic:upsert`.
- `orion/memory/vector_collection_policy.py`: the `orion_lived_memory` policy.
- `.github/workflows/lived-memory-recall-tests.yml`: CI for all of the above.

## Schema / bus / API changes

- **Added**
  - HTTP `POST /rerank` on orion-vector-host:
    - request `{"query", "documents"}`, at most 64 documents;
    - reply `{"model", "scores"}`;
    - 503 when no model is loaded, or when 2 requests are already in flight.
  - Chroma collection `orion_lived_memory` (derived, rebuildable).
- **Bus**: orion-recall publishes `VectorUpsertV1` on the existing `orion:vector:semantic:upsert`. Catalog producer list updated. No new schema.
- **Behavior changed**
  - The continuity digest can carry a `[Lived memories ...]` block.
  - `recall_telemetry.selected_ids` now includes `episode_memory:<id>` / `stance:<id>` / `person_fact:<id>` items. That is the read log step 3 needs.
  - Every FCC turn's `--settings` now carries the mesh-edge PreToolUse hook (Bash only).
- **Compatibility**: none of the existing recall backends change. Lived recall is off with `RECALL_LIVED_MEMORY_ENABLED=false`.

## Env/config changes

- **Added keys, orion-recall**:
  - `RECALL_LIVED_MEMORY_ENABLED=true`
  - `_CHROMA_URL`
  - `_EMBED_URL`
  - `_RERANK_URL`
  - `_COLLECTION`
  - `_MIN_FIT=0.004`
  - `_MIN_QUERY_TOKENS=5`
  - `_MAX_LINES=4`
  - `_TIMEOUT_SEC=4.0`
  - `_INDEX_INTERVAL_SEC=300`
  - `_INDEX_BATCH=25`
- **Added keys, orion-vector-host**:
  - `VECTOR_HOST_RERANK_MODEL=BAAI/bge-reranker-v2-m3`
  - `VECTOR_HOST_RERANK_DEVICE=cuda`
  - `VECTOR_HOST_RERANK_GPU_DEVICE_ID=00000000:86:00.0`
- **Added keys, orion-harness-governor**: `ORION_MESH_EDGE_PG_DSN`
- `.env_example` updated: yes. The flag ships ON.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-recall orion-vector-host orion-harness-governor`. The three diverged keys from earlier drafts (`RECALL_LIVED_MEMORY_MIN_FIT`, `VECTOR_HOST_RERANK_MODEL`, `VECTOR_HOST_RERANK_DEVICE`) were set to the final values by hand.
- Skipped keys: none.

## Metric gate (the ranking score)

`score = logit + 1.0*recency + 0.5*importance + 0.5*is_correction`. The bar is on `fit = sigmoid(logit)`.

1. **Provenance**
   - `logit` comes from `services/orion-vector-host/app/reranker.py::_score`, the bge-reranker-v2-m3 cross-encoder output.
   - `recency = 0.5^(age/30 d)`, from `episode_memory.occurred_at` (else `created_at`), or `created_at` for stances and facts.
   - `importance` is `episode_memory.strength`, crystallization `salience`, or 0.9 for person facts.
   - `is_correction` is `supersedes_memory_id IS NOT NULL`.
2. **Independence**: the reranker reads text only. Recency reads time only. Importance is not independent of the memory's kind: `strength` is set by category.
3. **Theory**:
   - Cross-encoder relevance (monoBERT / BGE reranker).
   - The fit, recency and importance blend follows Generative Agents (Park et al. 2023).
   - Corrections outrank what they correct (Mem0-style update).
4. **Live data**
   - `strength` takes only 3 values (0.8 / 0.9 / 1.0). Stance salience is 1.0 on all 44 rows. Importance is close to degenerate, so it carries a small weight and only breaks near-ties.
   - The logits are uncalibrated: a real fit sits around -6 to +2. A first draft blended in sigmoid space, and recency outranked fit. Replay caught it (7/11), so the blend is now in logit units (9/11).
   - Recency can return to rest (old memories approach 0). Fit does too: small talk sits below the bar except in 3 of 23 cases.
5. **Existing mechanism**: recall's fusion has its own recency and backend weights for chat turns. There was no reranker and no lived-memory reader.
6. **Reversibility**: one flag. Config only, no schema. The Chroma collection is derived and rebuildable.

## Integration gate

This is a new reader on the existing chat recall path. It is not a new producer of Orion experience: no new family, workflow, intention adapter, Prime concern or organ bus channel. The index upserts ride the existing vector channel. `scripts/check_integration_seam.py`: OK.

## Tests run

```text
pytest services/orion-recall/tests/test_lived_memory.py services/orion-recall/evals/test_lived_memory_replay_eval.py   21 passed
pytest orion/memory/tests/test_mesh_edge.py orion/fcc/tests/test_mesh_edge_hook.py services/orion-notify/tests/test_email_mesh_edge.py tests/test_vector_collection_policy.py   passed
pytest orion/harness/tests/test_{task_anchor,session_resume,focus_anchor,fcc_reading_undeclared_tool}.py   102 passed
pytest services/orion-hub/tests/test_session_roundup_loop.py   28 passed, 20 skipped (PG)
pytest services/orion-vector-host/tests (inside the vector-host image, torch present)   19 passed
pytest services/orion-recall/tests   all lived tests pass; 2 failures pre-exist on main (#3569)
scripts/check_env_template_parity.py PASS; check_integration_seam.py OK; check_personal_terms.py clean
```

## Evals run

`services/orion-recall/evals/run_lived_memory_replay_eval.py` ran against the host-side bank `~/.orion/evals/lived_memory_recall_bank.json` (mode 600, ids only):

- 11 referring messages, each with its target record ids, labelled by keyword search independent of the retriever.
- Juniper's last 150 real Hub messages.
- Each message only sees records stored before it was sent.

| bar (fit) | targets hit | small talk injecting | lines / msg |
|---|---|---|---|
| 0.002 | 9/11 | 13% | 2.33 |
| **0.004 (shipped)** | **9/11** | **13%** | **1.92** |
| 0.006 | 8/11 | 13% | 1.65 |
| 0.01 | 8/11 | 13% | 1.35 |
| 0.02 | 5/11 | 9% | 0.93 |

Rerankers compared on the same candidates: v2-m3 9/11, bge-reranker-base 7/11, mxbai-rerank-base 7/11, ms-marco-MiniLM-L-12 6/11.

v2-m3 on CPU took p50 2.3-3.6 s, too slow for a chat turn. That is why it runs on the P4 (p50 1.0 s for the rerank).

A rare-word exception to the 5-word floor was tried. It raised small-talk injection to 30-48% and was removed.

## Docker/build/smoke checks

```text
docker build of the worktree vector-host image (temporary tag; removed after the evals): /rerank answered on CPU and on the P4
compose config: orion-recall OK
backfill: services/orion-recall/scripts/backfill_lived_memory_index.py -> /tmp/lived-memory-index-backfill/report.md
  verdict complete, 582/582 live records current in orion_lived_memory, 0 errors, read-only on Postgres
```

## Review findings fixed

A code-review subagent ran on the first commit.

- **Finding:** the gh hook missed line continuations, an absolute `gh` path, `curl --json` / `-d@`, stdin bodies, and several write subcommands.
  - **Fix:** every `gh` call counts as a write unless it is a known read. Continuations are unfolded. Redirect, `-d@` and `--json @` files are read. A body built at run time (`$(...)`, backticks, a variable, a pipe into `--body-file -`) is held.
  - **Evidence:** parametrized tests in `orion/fcc/tests/test_mesh_edge_hook.py`.
- **Finding:** apostrophes inside shell, JSON or HTML quoting escaped the verbatim match.
  - **Fix:** `normalize` decodes HTML entities, `\uXXXX` and `\n`, and shell `'\''`.
  - **Evidence:** `test_encoded_apostrophes_still_match`.
- **Finding:** a hook error let a write through.
  - **Fix:** writes fail closed. Non-GitHub calls still fail open.
  - **Evidence:** `test_a_hook_error_holds_a_write_but_not_other_calls`.
- **Finding:** retired records' text stayed in Chroma.
  - **Fix:** the index pass deletes docs whose record is no longer servable. Postgres keeps the rows. An empty live read deletes nothing.
  - **Evidence:** `test_index_pass_deletes_docs_whose_record_is_no_longer_servable`.
- **Finding:** the guard only covered text servable right now.
  - **Fix:** it now reads every page version and retired, rejected and removed memories.
  - **Evidence:** `test_guard_sql_covers_retired_rows_and_old_page_versions`. Live: 1,787 guarded sentences, 0 hits on CLAUDE.md.
- **Finding:** lived recall ran one step after another, adding its full time to every turn.
  - **Fix:** it starts at intake and is awaited at render under the remaining deadline. It is cancelled if never rendered.
- **Finding:** lived items fed the keyword pressure heuristics and the empty-bundle shadow trigger.
  - **Fix:** both now read the bundle without lived items.
- **Finding:** the committed eval did not test the shipped content-word floor.
  - **Fix:** the fixture eval now runs at `RECALL_LIVED_MEMORY_MIN_QUERY_TOKENS`.
- **Finding:** CI gaps.
  - **Fix:** the harness argv tests were added to the workflow. The rerank endpoint test needs torch, so it runs only in the vector-host image (stated above).
- **Finding:** the reranker re-downloaded on every recreate and loaded on the event loop.
  - **Fix:** persistent HF cache volume, loaded via `asyncio.to_thread`, 503 when busy.
- **Nits fixed:**
  - NaN logits never pass the bar.
  - `IS DISTINCT FROM 'rejected'`.
  - Recipients are compared with `parseaddr`.
  - Block messages name only the kind of source.
  - The lived timeout no longer sets `deadline_hit`.

## Restart required

Run from the clean primary `main` checkout after merge. The post-pull rebuild covers these. Manual order:

```bash
scripts/safe_docker_build.sh orion-vector-host up -d --build
scripts/safe_docker_build.sh orion-recall up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-notify up -d --build
```

The vector-host must go first: recall's reranker calls fail (and inject nothing) until `/rerank` exists. The first vector-host boot downloads bge-large and bge-reranker-v2-m3 into the new HF cache volume (a few minutes).

## Risks / concerns

- **Severity: medium. Concern: GPU sharing.** The reranker shares the athena P4 with orion-whisper-tts. It holds about 3.4 GB of 7.6 GB. Whisper/XTTS was idle (3 MiB) during the evals.
  - **Mitigation:** with no GPU visible, the reranker falls back to CPU. That is slow and often past the 4 s timeout, which means no lines, not wrong lines.
- **Severity: medium. Concern: live behavior is UNVERIFIED.** The live path is not proven: no deployed turn has yet shown the lines in the motor_boot prompt.
  - **Mitigation:** after deploy, check the `lived_memory_injected` log lines and `recall_telemetry.selected_ids`.
- **Severity: medium. Concern: the governor DSN is the postgres superuser.** It reaches FCC Bash through the env. This is not new reach (the `/repo` mount already exposes `.env` files), but it widens it.
  - **Mitigation:** #3567, move to `orion_readonly` with a GRANT. Needs Juniper.
- **Severity: low. Concern: hook blind spots.** The hook cannot see scripts that call gh internally, `gh pr create --fill`, or `git push` of files.
  - **Mitigation:** #3568.
- **Severity: low. Concern: email check is dormant.** No caller sends to anyone but Juniper today, so the email check never runs in production. It refuses whenever it cannot load: orion-notify has no Postgres driver, so mail to anyone else is refused.
- **Severity: low. Concern: an existing public PR quotes a memory.** PR #3341's body already quotes one short stored stance sentence verbatim (6 words, not health or family). This change does not edit old PRs.
- **Issues:** opened #3567, #3568, #3569. This PR closes #3533 and #3409.

## PR link

(see PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
