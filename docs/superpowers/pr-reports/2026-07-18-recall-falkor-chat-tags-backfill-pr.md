# Phase 3 of the recall RDF→Falkor cutover: historical chat-tag backfill

## Summary

- Populated FalkorDB's `orion_recall` graph with tag/entity/sentiment data for every historical chat turn predating the live Falkor writer (which only started working ~2026-07-18 after an unrelated 6-month observer-gate bug was fixed).
- Two-stage script: Stage A (host `.venv`, has `asyncpg` but not the spaCy transformer model) snapshots Postgres to `/tmp/backfill-recall-falkor-chat-tags/snapshot.json`; Stage B (runs inside the live `orion-athena-meta-tags` container via `docker cp`+`docker exec`, the only place `en_core_web_trf` is confirmed loadable) runs the real extraction pipeline and writes via the real production Cypher writer.
- **Not a copy of pre-existing Fuseki data.** Investigated first and confirmed there was almost nothing to copy — the same observer-gate bug that blocked the Falkor writer also blocked the Fuseki `chat_tagging` extraction for the same ~6-month window; Fuseki has only 204 enrichment records, ever. Re-ran the same extraction (spaCy NER + keyword sentiment heuristic) fresh against raw Postgres text instead, satisfying the Phase 0 spec's explicit "not a raw copy" requirement.
- Result: 1,708 turns written, 0 errors, ~3.6 minutes, real historical timestamps preserved (2025-10-19 → today), 174 distinct sessions linked, 1,027 turns with real extracted entities.

## Outcome moved

`orion_recall`'s `:ChatTurn` graph went from 3 nodes (today's live traffic only) to 1,712 — the historical population is now fully present, making Phase 4's read path (`fetch_falkor_chatturn_fragments`, PR #1192) actually useful instead of near-empty.

## Current architecture

Before this backfill: FalkorDB's `orion_recall` graph only had turns written by the live bus-driven pipeline going forward from the observer-gate fix (PR #1184, merged 2026-07-18). ~1,743 historical rows across `chat_history_log`/`social_room_turns` had no Falkor representation at all.

## Architecture touched

No service code changed. This is a one-time data operation plus two new utility scripts, committed for institutional memory/reusability (matching `scripts/backfill_phi_corpus.py`'s precedent — reuse the live production transform, don't reimplement it).

## Files changed

- `scripts/backfill_recall_falkor_chat_tags_snapshot.py` (new): host-side Postgres snapshot stage. Works around the same `scripts/platform/` stdlib-shadowing issue `backfill_phi_corpus.py` already documents.
- `scripts/backfill_recall_falkor_chat_tags_extract_and_write.py` (new): container-side extraction+write stage. Imports the real `app.falkor_recall_writer.write_chat_turn_tags_to_falkor` and the real loaded `app.main.nlp` spaCy pipeline directly — no logic duplicated.
- `services/orion-meta-tags/README.md`: "Not yet wired: historical backfill (Phase 3)" replaced with the real completion note and numbers.

## Schema / bus / API changes

None. Pure data population into the existing `orion_recall` graph schema (`:ChatTurn`, `:ChatSession`, `:Entity`, `HAS_TURN`, `MENTIONS_ENTITY` — same shape the live writer already produces).

## Env/config changes

None.

## Backfill protocol (AGENTS.md section 14)

- **Snapshot**: `/tmp/backfill-recall-falkor-chat-tags/snapshot.json` — 1,710 `chat_history_log` rows + 33 `social_room_turns` rows = 1,743 total, well under the 100k-row/100MB stop-and-ask threshold.
- **Progress log**: `/tmp/backfill-recall-falkor-chat-tags/progress.log` — percent/ETA/rate/error-count lines every 50 rows, full run output.
- **Report**: `/tmp/backfill-recall-falkor-chat-tags/report.md` — verdict, measurable outcome table, before/after examples, the social-turn dedup finding (see below), error count, "another pass needed?" (no).
- **Before/after**: `/tmp/backfill-recall-falkor-chat-tags/before_after.csv` — 5 representative rows including the oldest turn, both genuinely social-only turns, a dual-logged example, and a pre-existing live-written turn correctly skipped (not double-written).

## Real finding surfaced during this job (not a bug, worth documenting)

31 of the 33 `social_room_turns` rows are **dual-logged**: the same conversation also exists in `chat_history_log` under the identical `turn_id`/`correlation_id` (the hub relays social-room activity into the main chat log too). The script processes `chat_history_log` first, so those 31 turns landed as `source_kind="chat.history"` and were correctly skipped (not double-written) on the social pass — confirmed via a direct Postgres join showing the overlap, not assumed. This ordering is a net positive for current utility: `orion-recall`'s Phase 4 read path only recalls `chat.history` turns today, so this means those 31 real conversations are recallable now instead of invisible until social-turn read support exists (a later, unstarted phase). Only the 2 genuinely social-only rows show as `social.turn.stored.v1`.

## Tests run

No new tests — these are one-off utility scripts operating on live infrastructure state, not service code with an ongoing test surface. Correctness was verified against real data instead: direct `GRAPH.QUERY` spot-checks (a specific historical turn's entities/sentiment/timestamp matched expected values), aggregate counts reconciled exactly against independent Postgres counts (174 sessions in Falkor == 174 distinct non-null `session_id` values in `chat_history_log`, confirmed via `psql`), and the `written + skipped_already_present == rows_total` accounting checked out (1,708 + 35 = 1,743).

## Evals run

Not applicable.

## Docker/build/smoke checks

Ran live inside the already-running `orion-athena-meta-tags` container (no rebuild needed — the scripts were `docker cp`'d in and cleaned up afterward, not baked into the image). Confirmed the container was undisturbed by the job (no crash, no elevated error logs, live traffic during the run — e.g. the 3 pre-existing live-written turns and the ongoing organic turn count growth from 1,710→1,712 rows mid-run — was correctly handled by the idempotent skip logic).

## Review findings fixed

Not run through the 8-angle code-review skill — this is a one-off data-migration job, not a shipped service-code change reviewed against ongoing correctness/maintainability criteria. Verification here was against real data (see "Tests run" above), matching this repo's "runtime truth beats config truth" doctrine more directly than a static code review would for this kind of job.

## Restart required

None. No running service's code or config changed.

## Risks / concerns

- Severity: None identified.
- The job ran once, cleanly, with 0 errors, and is not re-runnable-by-accident-with-bad-effect: `write_chat_turn_tags_to_falkor` is MERGE-based (idempotent), and the extraction script's own pre-check additionally skips already-present `turn_id`s for accurate reporting.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1195
