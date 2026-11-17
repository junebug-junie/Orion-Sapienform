## Summary

- When Orion distills a new memory that corrects an older one, the older one is now retired: it is marked `superseded`, the correction points at it, and one event row records why. Every reader that only shows active memories (lived-memory recall, the `page:juniper` fold, situation, reverie seed) then stops showing the old one.
- A local-model check decides each pair, on the distiller's own GPU hold. Two strict questions must agree: "does B correct A?" (a sure yes) and then "can A and B both be true, each at its own date?" (a no). Anything else links nothing. A wrong retirement hides a true memory, so the check leans hard toward "no".
- Never retired by the model: memories Juniper confirmed, memories she is still being asked about, to-dos, and Orion's own views (a changed view is growth, not an error). A correction that is still waiting for Juniper's answer cannot retire anything.
- A re-runnable pass, `scripts/memory_supersede_pass.py`, does the same for existing rows: `dry-run` (writes nothing to the database), `apply` (the reviewed pairs only), `undo` (one pair). It follows the backfill protocol: snapshot, progress log, report, before/after CSV, all private under `/tmp/memory-supersede-pass/`.
- The pass also writes an ids-only manifest Orion can read, at `/mnt/orion-fcc/notes/memory_supersede_pass_manifest.json`.

## Outcome moved

Before: corrections sat next to the errors they corrected, and recall could serve both. After: write-time supersede for new memories, plus a reviewed pass for old ones. The dry run over the live store found **1 pair** (from 740 candidate checks). It is NOT applied. Juniper reviews it first.

The two contradictions the spec names are not proposed, and this is correct by the check's definition:
- The trip rows say Juniper was staying in a city for a work trip. They were true at their date. The home correction ("she lives elsewhere, not there") does not make them false. The model said so on both trip rows, in both questions. The wrong "home" claim the spec describes was never stored as its own `episode_memory` row; only its correction was.
- The pet-species correction has no wrong row to retire either: no stored memory calls the pet a cat. The two pet rows near it are about surgery and recovery, and they are still waiting for Juniper's confirmation answer, so they are protected anyway.

## Current architecture

`episode_memory.supersedes_memory_id` existed, and only the confirmation loop (Juniper's "revised" answer) wrote it. The distiller (`orion/memory/episode/store.py`) never linked a correction. The one-time pass from #3533 never ran. Live: 5 superseded rows, all from the confirmation loop.

## Architecture touched

- `orion/memory/episode/supersede.py` (new): candidates (the lived-memory serving SQL plus `orion_lived_memory` vectors, then a shared rare word or rare referent), the two-question check, `apply_link`, `undo_link`.
- `services/orion-durable-runs/app/episode_distill_graph.py`: new `supersede_check` node between `distill` and `persist`, on the same hold, with its own 300 s budget. A slow, failed or recalled check links nothing. It never spends an attempt and never re-distills. An operator control still stops the run.
- `orion/memory/episode/store.py`: `persist_episode` applies the links in the same transaction as the memories.
- `services/orion-durable-runs/app/admission_runtime.py`, `runner.py`: candidate planning; the check calls carry `purpose=memory_supersede_check` and temperature 0.0.

## Files changed

- `orion/memory/episode/supersede.py`: linking logic, writes, undo.
- `orion/memory/episode/store.py`: apply links in the persist transaction; `superseded` count.
- `services/orion-durable-runs/app/episode_distill_graph.py`: `supersede_check` node.
- `services/orion-durable-runs/app/admission_runtime.py`: `_plan_supersede`.
- `services/orion-durable-runs/app/runner.py`: purpose/temperature passthrough.
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`: 5 new keys.
- `scripts/memory_supersede_pass.py`: dry-run / apply / undo / manifest.
- `scripts/sql/2026-11-17_grant_orion_readonly_supersede_events.sql`: optional read grant so Orion can read the event trail.
- `scripts/sync_local_env_from_example.py`: `MEMORY_SUPERSEDE_` sync prefix.
- `orion/memory/README.md`: concept row.
- Tests: `orion/memory/episode/tests/test_supersede.py`, `test_supersede_pg.py`, `test_supersede_pass_script.py`, `test_store_pg.py`; `services/orion-durable-runs/tests/test_episode_distill_graph.py`.
- `.github/workflows/orion-memory-episode-tests.yml`: path filter for the pass script.

## Schema / bus / API changes

- Added: `episode_memory_event` ops `superseded` and `supersede_undone` (no DDL; the column is free text). Event id for a handled pair: `uuid5(SUPERSEDE_NAMESPACE, "superseded|<old_id>|<new_id>")`.
- Removed: none. Renamed: none.
- Behavior changed: the distiller can now retire older memories.
- Compatibility notes: no migration. Distill checkpoints saved before this patch have no `supersede` key and persist with no links.

## Env/config changes

- Added keys (orion-durable-runs): `MEMORY_SUPERSEDE_ENABLED=true`, `MEMORY_SUPERSEDE_CHROMA_URL`, `MEMORY_SUPERSEDE_EMBED_URL`, `MEMORY_SUPERSEDE_COLLECTION=orion_lived_memory`, `MEMORY_SUPERSEDE_MAX_CHECKS=12`.
- Removed keys: none. Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (all 5 added).
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/memory/episode/tests (with a throwaway Postgres 16)   339 passed
pytest orion/memory                                                2435 passed, 33 skipped (pg, no DSN in that run)
pytest durable-runs episode_distill graph/reconcile/referent       all passed (incl. slow-check, failing-check, hold tests)
pytest durable-runs -k "settings or admission_runtime or episode or core_page or runner"   94 passed, 15 skipped
scripts/check_env_template_parity.py    PASS
scripts/check_integration_seam.py       OK
scripts/check_personal_terms.py         no personal terms in tracked files
git diff --check                        clean
```

## Evals run

```text
Live dry run (read-only, backfill protocol), 2026-11-17:
  active rows snapshotted 488; candidate pairs 740; model-call errors 0
  first question sure-yes then vetoed by the second question: 5
  proposed pairs: 1   (statements only in /tmp/memory-supersede-pass/, mode 600)
Hand controls on the gateway (made-up statements):
  "pet is a cat" -> "pet is a dog, not a cat"            linked
  "has four GPUs" -> "corrected: eight, not four"          linked
  "in a city for a meeting" -> "lives in a small town"    not linked
  "lives in a lake-front hotel" -> "the hotel was a trip; lives elsewhere"   not linked (conservative miss)
```

## Docker/build/smoke checks

```text
Not deployed (per task). No image build run. Restart needed later for the write-time path.
```

## Review findings fixed

- Finding: the check ran inside the distill node's time budget; a slow check could cancel the distill answer and spend an attempt.
  - Fix: its own `supersede_check` node with its own budget; every failure links nothing.
  - Evidence: `test_a_slow_supersede_check_times_out_without_costing_an_attempt`.
- Finding: a lost hold or operator pause during checks was counted as a refused check and the loop kept calling.
  - Fix: the first failed model call stops the whole plan.
  - Evidence: `test_a_failed_call_stops_the_whole_plan`.
- Finding: undo could reverse a later retirement made another way, and cleared a pointer it did not set.
  - Fix: undo only when the row's latest status event is this pair's `superseded`; clear the pointer only when this link set it.
  - Evidence: `test_undo_never_reverses_a_later_retirement_and_keeps_a_pointer_it_did_not_set`.
- Finding: rows still awaiting Juniper's answer could be retired; unsettled corrections could retire rows.
  - Fix: old row must not be pending/unconfirmed/rejected; correction must be `auto` or `confirmed`.
  - Evidence: `test_open_asks_and_pending_corrections_are_never_applied`, unit test.
- Finding: same-episode siblings could retire each other on a re-distill.
  - Fix: skip pairs from one episode.
- Finding: an undone pair could be re-proposed and reported as applied.
  - Fix: undone pairs are excluded from candidates and logged `undone_never_reapplied` by apply.
- Finding: row locks taken in two statements; check calls indistinguishable in gateway telemetry; error text in a log.
  - Fix: one `ORDER BY memory_id FOR UPDATE`; `purpose=memory_supersede_check`, temperature 0.0; log exception type only.
- From live data: one proposed pair was a changed opinion, not a fact.
  - Fix: Orion's views are never the retired side.

## Restart required

Not deployed. When Juniper wants write-time supersede live, from the clean primary main checkout after merge:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Apply the reviewed pass pairs (after Juniper reads `/tmp/memory-supersede-pass/pairs.md`):

```bash
.venv/bin/python scripts/memory_supersede_pass.py apply --pairs /tmp/memory-supersede-pass/pairs.jsonl --yes
```

Undo one pair:

```bash
.venv/bin/python scripts/memory_supersede_pass.py undo --old <OLD_ID> --new <NEW_ID>
```

Optional, so Orion can read the event trail directly (needs #3593's grant for `episode_memory` too):

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "GRANT SELECT ON public.episode_memory_event TO orion_readonly;"
```

## Risks / concerns

- Severity: medium. Concern: a model "yes" that is wrong hides a true memory. Mitigation: two agreeing answers, protected states, undo command, every retirement logged with its reason.
- Severity: low. Concern: the check is conservative and misses some real corrections (see the hotel control). Mitigation: acceptable by design; both rows stay visible, as before.
- Severity: low. Concern: up to 12 short model calls per distilled episode on the same hold. Mitigation: `MEMORY_SUPERSEDE_MAX_CHECKS`, 300 s node budget, kill switch `MEMORY_SUPERSEDE_ENABLED=false`.
- Severity: low. Concern: `orion_readonly` cannot read `episode_memory` yet (the #3593 grant is not applied live). Until then Orion verifies through the manifest file. UNVERIFIED that Orion's FCC turn can read `/mnt/orion-fcc/notes/` (the Hub mounts `/mnt/orion-fcc` whole).
- Issues opened: none. Issues commented: #3537 (run the pass as its last step). Closes: #3587 on merge (its live "done when" still needs the apply).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3617

🤖 Generated with [Claude Code](https://claude.com/claude-code)
