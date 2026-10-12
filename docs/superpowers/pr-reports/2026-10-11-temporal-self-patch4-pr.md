# Temporal Self patch 4: Orion uses its own day

## Summary

- In chat (the unified turn's stance step, `stance_react`), Orion now sees one short paragraph about where it is in its own day: what it is on, what it came from, when Juniper last spoke, open threads, when it last slept. A fixed template writes it from the stored day record; no model writes it, and every clause names the record it came from (`[arc:<id>]`).
- Orion's Day letter now gets yesterday's closed day from that same record (about 1,900 characters for 10-09), first line "as recorded at close".
- Anything Orion writes back that cites a record is checked against the records it was shown. Unknown ids are removed and counted. A sentence about the recorded day with no citation is labelled **UNVERIFIED** where Juniper can see it (Hub turn trace, stored letter, letter reread). Nothing is blocked or retried.
- A missing, stale (over 900 s), wrong-day or unreadable record renders nothing. A real quiet stretch renders as rest.
- One kill switch per reader, both shipped ON: `TEMPORAL_SELF_STANCE_CUE_ENABLED` (orion-thought), `TEMPORAL_SELF_DAY_BLOCK_ENABLED` (orion-hub).

## Outcome moved

Before: Orion kept a record of its day (patch 3) and nothing read it. Orion-mode chat never saw time of day or day context at all; the letter read a rolling pile of logs with no account of where attention went.

After: every Juniper turn's stance is shown the day, and its citations of it are checked and stored (`thought_decision.day_citations`); every letter is shown the closed day, and its citations are checked and stored (`orion_day_letter.sources.arc_citations`).

Live evidence before deploy (real gateway, real frame, this branch's code):

- 4 relational stances ("how has your day been?") with the cue: 3 cited real shown arcs, all 3 passed the gate as `verified`; the 4th cited none (`no_day_claims`). The first run exposed a real bug (below), fixed.
- 3 task stances (a code question) with the cue: all kept `evidence_refs == [hub:turn]` (the day did not leak into task evidence), boundary eval `passed: true`.

## Current architecture

- orion-durable-runs `chronicle` writes `temporal_self_*` every 120 s and serves `GET /temporal-self/frame` and `/day/{id}` (patch 3, #2605). No reader.
- The unified turn's stance (`services/orion-thought/app/bus_listener.py` -> cortex-exec `stance_react.j2`) had no day context. `align_evidence_refs_to_coalition` already drops evidence ids outside the coalition, silently.
- Orion's Day: Hub gathers SQL sources (`orion/orion_day/gather.py`), budgets a digest (`budget.py`), durable-runs writes note + carry-forward (`orion_day_graph.py`), the reread tool checks claims (`letter_parts.py`, owned by the letter-reread session).

## Architecture touched

- **New pure modules** (`orion/temporal_self/`): `stance_cue.py`, `day_block.py`, `citations.py`.
- **orion-thought**: `app/temporal_self_cue.py` (HTTP fetch, fail-open); `bus_listener.py` fetches the cue concurrently with Mind and passes it to the plan context and the pipeline.
- **Stance pipeline** (`orion/thought/stance_react.py`): `gate_day_citations` runs before the existing coalition alignment.
- **Persistence**: `thought_decision.day_citations` (nullable jsonb), sql-writer column, Hub turn trace SELECT + UI.
- **Letter**: gather reads `temporal_self_day` for the letter date; budget puts the block in the digest and resolves `arc:`/`day:` refs; durable-runs gates note and carry-forward; reread tool shows the check per part.
- Untouched by decision: `chat_general`, `chat_stance_brief.j2`, `recent_attention`, `daily_metacog_v1`, `orion/attention/`, `services/orion-attention-runtime/`.

**Why HTTP for chat and SQL for the letter.** orion-thought has no DSN for the chronicle tables and the frame route returns exactly the validated frame (8 ms p50 from the host, 14 ms from inside the thought container), so the stance cue uses the existing route with a 0.5 s timeout. The letter gather is already "every source is a read-only SELECT on one connection, and a failed source names its gap", so the closed day is one more SELECT there; an HTTP call would have been the only non-SQL source and would pull a 727 KB body.

## Files changed

- `orion/schemas/temporal_self.py`: `ArcSummaryV1.last_seen_at` (optional); new `TemporalSelfStanceCueV1`, `TemporalSelfDayBlockV1`, `DayCitationCheckV1`.
- `orion/schemas/registry.py`: registers the three new models.
- `orion/temporal_self/frame.py`: `summarize` fills `last_seen_at` ("last turn 3 min ago").
- `orion/temporal_self/stance_cue.py`, `day_block.py`, `citations.py`: renderers and gate (new).
- `orion/temporal_self/tests/test_stance_cue.py`, `test_day_block.py`, `test_citations.py`, `patch4_fixtures.py`, `data/*`: byte-exact fixtures from live rows (`data/build_fixtures.py` rebuilds them).
- `orion/schemas/thought.py`: `ThoughtEventV1.day_citations`, `ThoughtDecisionRecordV1.day_citations`.
- `orion/thought/stance_react.py`: drops a model-supplied `day_citations`; `gate_day_citations`; `apply_stance_react_pipeline(..., day_cue=)`.
- `orion/thought/evals/stance_task_boundary.py`: `day_citations` is transport-owned; `--temporal-self-cue` renders the block.
- `orion/cognition/prompts/stance_react.j2`: `MY DAY SO FAR` block (only when a cue exists).
- `orion/schemas/context_provenance.py`: `temporal_self_cue` = `live_runtime_projection`.
- `services/orion-thought/app/temporal_self_cue.py` (new), `bus_listener.py`, `settings.py`, `.env_example`, `README.md`, `tests/test_stance_temporal_self_cue.py`, `evals/run_temporal_self_cue_latency_eval.py`.
- `services/orion-sql-writer/app/models/thought_decision.py`, tests; `services/orion-sql-db/manual_migration_thought_decision_day_citations_v1.sql` (new, `-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub`).
- `services/orion-hub/scripts/chat_turn_trace_routes.py`, `static/js/thought-process.js`, tests: day citations in the turn trace.
- `orion/schemas/orion_day.py`: `OrionDayMaterialV1.temporal_day`, `OrionDayLetterSourcesV1.arc_citations`.
- `orion/orion_day/gather.py`, `budget.py`, `brief.py`; `orion/orion_day/tests/fixtures.py`, `test_gather.py` (additive: FakeConn serves the new query).
- `orion/cognition/prompts/orion_day_note_v1.j2`: cite `[arc:<id>]` when relying on the chronology (only when the block is present).
- `services/orion-durable-runs/app/orion_day_graph.py`, `tests/test_orion_day_arc_citations.py`.
- `services/orion-hub/scripts/orion_day_letter.py`, `main.py`, `app/settings.py`, `.env_example`, `scripts/orion_day_introspect.py`, `tests/test_orion_day_temporal_day.py`.
- `orion/temporal_self/README.md`: readers section.

## Schema / bus / API changes

- Added: `TemporalSelfStanceCueV1`, `TemporalSelfDayBlockV1`, `DayCitationCheckV1` (registry `_REGISTRY`); `ThoughtEventV1.day_citations` and `ThoughtDecisionRecordV1.day_citations` (optional); `OrionDayMaterialV1.temporal_day`, `OrionDayLetterSourcesV1.arc_citations` (optional, default empty); `ArcSummaryV1.last_seen_at` (optional); column `thought_decision.day_citations jsonb`.
- Removed / renamed: none. No new bus channel, no new route.
- Behavior changed: a stance may cite `arc:<id>` from the cue in `evidence_refs` (accepted bare or bracketed); any other `arc:` ref is removed from evidence/strain refs and imperative/tone. The letter's unknown `[arc:]` refs are removed from the stored text.
- Compatibility notes: `ThoughtEventV1` is not `forbid`. `OrionDayMaterialV1` and `OrionDayLetterSourcesV1` are `forbid`: durable-runs (reads Hub's brief) must be rebuilt before or with orion-hub, and orion-hub (reads letters durable-runs writes) before the next 08:30 local letter. Deploy both in one sitting, not around 08:30 local. `ArcSummaryV1` has no reader older than this patch; old frames without `last_seen_at` still render (tested).

## Env/config changes

- Added keys: orion-thought `TEMPORAL_SELF_STANCE_CUE_ENABLED=true`, `TEMPORAL_SELF_FRAME_URL=http://durable-runs:8121`, `TEMPORAL_SELF_CUE_MAX_AGE_SEC=900`, `TEMPORAL_SELF_CUE_FETCH_TIMEOUT_SEC=0.5`; orion-hub `TEMPORAL_SELF_DAY_BLOCK_ENABLED=true`.
- Removed / renamed: none.
- `.env_example` updated: orion-thought, orion-hub.
- local `.env` synced with `python scripts/sync_local_env_from_example.py orion-thought orion-hub`: yes, all 5 keys added to the primary checkout's `.env` files.
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. pytest orion/temporal_self orion/thought orion/orion_day -q              -> all passed
.github/workflows/temporal-self-readers-tests.yml steps, fresh venv, clean git-archive export -> 179 / 12 / 28 passed
pytest orion/schemas/tests -q   -> 1 failed: test_static_ctx_assignments_covered (PRE-EXISTING on main:
                                   admission, llm_serving_node, recent_attention, ... unclassified)
services/orion-thought:  pytest tests -q   -> 560 passed, 28 skipped, 1 failed
                                   (test_mind_enrichment_defaults_off: PRE-EXISTING on main, reads local .env)
services/orion-durable-runs: PYTHONPATH=. pytest tests -q -> 388 passed, 85 skipped
services/orion-sql-writer: pytest tests/test_thought_decision_sql_shape.py -> 11 passed
services/orion-hub: pytest test_orion_day_temporal_day.py test_orion_day_introspect_listener.py
                    test_orion_day_letter.py test_orion_day_correction_listener.py
                    test_chat_turn_trace_routes.py test_turn_trace_panel_ui.py -> all passed
scripts/check_env_template_parity.py -> PASS
scripts/check_metric_lineage.py --gate / --prompt-semantics -> PASS
scripts/check_definition_drift.py --gate -> PASS (0 definitions changed)
scripts/check_inner_state_registry.py -> OK
```

## Evals run

```text
orion/orion_day/evals/run_orion_day_eval.py           -> passed
orion/temporal_self/evals/run_arc_precision_eval.py   -> passed (unchanged)

Stance boundary eval, live gateway (qwen 35B), with the real cue rendered:
  python orion/thought/evals/stance_task_boundary.py --task-file <code question> --temporal-self-cue "<live cue>"
  3/3 passed: evidence_refs == [hub:turn], no arc leaked into task evidence.

Relational stance with the cue (live gateway, gated by this branch's pipeline), 4 runs:
  3 cited shown arcs -> label verified; 1 cited none -> no_day_claims; 0 rejected.

Latency (services/orion-thought/evals/run_temporal_self_cue_latency_eval.py, live):
  fetch+render  p50 7.8 ms, p95 11.3 ms (50 real frame fetches + project_stance_cue)
  Mind          p50 7,342 ms over 9 real turns (stance_prepare_overlap log lines)
  critical-path wait added p50 = 0 ms (the fetch overlaps Mind); worst case with Mind off 0.1%
  stance prompt +189 tokens on a live p50 of 5,325 prompt tokens (+3.6% prefill; decode, 1,321
  completion tokens p50, is unchanged)  -> passed (<= 5%)
```

How the latency was measured: the cue only adds what runs on the turn's critical path. The fetch runs alongside Mind enrichment, so the turn waits `max(0, fetch - mind_ms)`; `mind_ms` is a lower bound on turn time, so the ratio against it overstates the cost. The prompt growth is reported, not converted to time. The end-to-end check (50 Juniper turns before and after) is post-deploy: **UNVERIFIED** until then.

## Docker/build/smoke checks

```text
Not built or deployed (Juniper: do not deploy). Live reads only:
  frame route reachable from inside orion-athena-thought at http://durable-runs:8121 (200, 14 ms)
  renderers run on live /temporal-self/frame and /day/2026-10-09 rows: identical to the committed fixtures
```

## Review findings fixed

Code review ran in a subagent against `origin/main...feat/temporal-self-patch4-uses-its-day`.

- Finding (must): the fixture rows lived in `orion/temporal_self/tests/data/`, which `.gitignore` ignores (`data/`), so CI would have failed 18 tests.
  - Fix: moved to `tests/patch4_rows/`, committed.
  - Evidence: every affected suite re-run against a clean `git archive` of the tree: 179 + 12 + 28 + 201 + 388 + 102 passed.
- Finding (must): with "broadcast log silent" and no open arc, the cue said "No attention or conversation arc is open right now", i.e. rest, when the attention record had actually stopped.
  - Fix: a silent log never renders the rest line; the clause is dropped and counted.
  - Evidence: `test_silent_broadcast_with_no_open_arc_is_not_rest`.
- Finding: CI ran none of the orion-thought cue tests, the Hub turn-trace tests or the Hub letter-block test.
  - Fix: new `.github/workflows/temporal-self-readers-tests.yml`; verified in a fresh venv built only from its install lines.
- Finding: an unknown arc id written bare (`arc:<id>`, no brackets) in imperative/tone or a letter was neither removed nor counted; an uppercase shown id was rejected.
  - Fix: the text gate matches bracketed and bare forms; ids compare lowercased.
  - Evidence: `test_bare_unknown_arc_in_text_is_removed_and_counted`, `test_uppercase_shown_id_is_the_same_id`.
- Finding: labels matched inside other words ("memory" in "memoryless"), inflating UNVERIFIED.
  - Fix: whole-word match. Evidence: `test_labels_match_whole_words_only`.
- Finding: httpx's 0.5 s timeout is per phase, not total (DNS not covered).
  - Fix: `asyncio.wait_for` bounds the whole fetch. Evidence: `test_slow_fetch_is_bounded_by_the_total_timeout`.
- Finding: a frame from the future (clock skew) rendered normally.
  - Fix: more than 60 s ahead renders nothing (`skip_reason=clock_skew`). Evidence: `test_frame_from_the_future_renders_nothing`.
- Finding: the stored check could describe text that never shipped if enforcement rewrote imperative/tone after the gate.
  - Fix: not reachable today (enforcement only changes hazards, priorities and disposition); pinned by `test_stored_check_describes_the_text_that_ships`.
- Finding (nit): "2 open threads" cited one arc. Fix: "2 open threads from conversation, 1 with its record [arc:...]".
- Finding (nit): render errors were logged as `fetch_failed`. Fix: separate `render_failed`.
- Finding (nit): hard-coded 5 examples. Fix: `UNCITED_SENTENCES_SHOWN`.
- Not changed (nit): an `[arc:]` placed after the full stop counts as its own sentence, so the claim before it reads UNVERIFIED. The prompt asks for same-sentence placement; this keeps the rule strict.
- Finding (rollback note): rebuilt durable-runs writes `last_seen_at` into stored frames/days; an older reader validating `ArcSummaryV1` would reject them. Today the only readers are durable-runs itself and this patch's code, so rolling back means rolling back durable-runs too.

Also found and fixed before review, from the live gateway runs:

- Finding: the stance model wrote `evidence_refs: ["hub:turn", "[arc:4ad6fd52bed009bd]"]` (the cue's bracketed form). The gate treated it as not-an-arc and the existing coalition filter dropped it without a trace, so a real citation looked like no citation.
  - Fix: `gate_arc_refs` normalises `[arc:<id>]` to `arc:<id>`.
  - Evidence: `test_bracketed_evidence_ref_from_the_live_model_is_cited_not_dropped`; 3 follow-up live runs gated as `verified`.

## Restart required

Not deployed. After merge, from the primary checkout on main (`git pull --ff-only` first), in this order:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_thought_decision_day_citations_v1.sql
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-sql-writer up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-durable-runs up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-thought up -d --build
```

Live checks after deploy:

1. `docker logs --since 10m orion-athena-thought 2>&1 | grep -E "temporal_self_cue|temporal_self_day_citations"`: a `temporal_self_cue corr=...` line per Juniper turn (or `temporal_self_cue_skipped reason=...`).
2. `SELECT correlation_id, day_citations->>'label', day_citations->'cited_valid' FROM thought_decision WHERE day_citations IS NOT NULL ORDER BY created_at DESC LIMIT 5;` and each cited id returned by `GET :8124/temporal-self/arcs`.
3. The Hub turn trace for that correlation id shows the "day cue:" line with its label.
4. `curl -s :8124/temporal-self/frame | jq '.active_arc.last_seen_at'` is non-null after durable-runs restarts.
5. Next letter (08:30 local): `SELECT letter_date, sources->'arc_citations' FROM orion_day_letter ORDER BY letter_date DESC LIMIT 1;` and `sources->'by_source'->'temporal_day'->>'status' = 'ok'`.
6. Acceptance 5: compare 50 Juniper turns before/after (end-to-end and `stance_prepare_overlap`).

## Risks / concerns

- Severity: medium. Concern: the gate proves a cited id is real and was shown; it does not prove the sentence reads that row correctly. Mitigation: the turn trace and the reread show the cited ids next to the text; the periodic truth eval (spec acceptance 6) is a follow-up.
- Severity: low. Concern: the "busy" stance fixture is the 10-11 record rebuilt at 17:20 UTC from stored arcs (no frame history exists); the "talking" fixture is a real frame. Fixtures live in `orion/temporal_self/tests/patch4_rows/`. Mitigation: both renders were checked against the live route output; the fixture builder documents it.
- Severity: low. Concern: open threads cite their concern arc only when it is in the frame's `arcs_today` list (`OpenThreadV1` carries no arc id). Mitigation: the cue says so ("2 open threads, 1 with its record"); follow-up to add `arc_id` to `OpenThreadV1`.
- Severity: low. Concern: frames lack `last_seen_at` until durable-runs is rebuilt; until then the cue says "a conversation with Juniper, open for N min" instead of "last turn N min ago". Tested.

Follow-ups found on the non-prod chat path (not wired, by decision):

- `recent_attention` (chat_general only) has no `CONTEXT_PROVENANCE_REGISTRY` entry, and is one of the keys that already fails `test_static_ctx_assignments_covered` on main.
- On 10-11 two of its three rows were "No winner this tick" filler (spec, 1,172 chars); if chat_general ever matters again, the patch-4 cue is the replacement.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
