# PR report: log retrieval events from real chat-time recall

Implements Item 3 of `docs/superpowers/specs/2026-07-13-recall-epistemic-honesty-and-observability-spec.md` — named in that spec as the single highest-priority item, since every other claim about live chat recall (including the companion reinforcement/decay spec's own acceptance checks) depends on this data existing.

## Summary

- `services/orion-recall/app/collectors/active_packet.py::fetch_active_packet_fragments()` — the function real chat turns actually call for PCR active-packet retrieval — never wrote to `memory_crystallization_retrieval_events`. Only Hub's `/api/memory/active-packet` route did, and real chat doesn't call that route.
- Reuses `insert_retrieval_event()` (`orion/memory/crystallization/repository.py`), unmodified — same table, same shared conceptual event, not a new schema.
- Gated on `packet.crystallization_refs` being non-empty — no empty-shell log rows for retrievals that found nothing.
- Wrapped in try/except with a warning log on failure — a broken audit write can never break a real chat turn or drop its recall results.
- Live-verified against real Postgres: a real row landed with a `created_at` timestamp inside the actual call window, confirmed by direct query, not by trusting the function's return value.

## Outcome moved

Real chat-time recall now produces an inspectable artifact. Before this patch, there was no way — none — to verify any claim about what Orion actually recalled during a live conversation; only synthetic smoke scripts hitting Hub's manual API route could be observed.

## Current architecture (before this patch)

Two recall paths exist: Hub's `/api/memory/active-packet` route (used by manual/API callers, already logs) and `orion-recall`'s `fetch_active_packet_fragments()` (used by real `chat_general` PCR turns, logged nothing). They call the same underlying `retrieve_active_packet()` but only one side had observability.

## Architecture touched

Single collector function in `orion-recall`. No schema, bus, or contract changes — reuses an existing table and an existing insert function verbatim.

## Files changed

- `services/orion-recall/app/collectors/active_packet.py`: import `insert_retrieval_event`, call it after `retrieve_active_packet()` returns with non-empty refs, defensive `getattr()` field mapping (see below), try/except around the write.
- `services/orion-recall/tests/test_active_packet_retrieval_event_logging.py` (new): 3 tests — logs when refs present, doesn't log when refs empty, a write failure doesn't propagate and break fragment return.

## Field mapping — what's real vs. what's genuinely unavailable

- `query=fragment` — real.
- `task_type` — real, already computed locally from `query.task_hints`.
- `session_id=getattr(query, "session_id", None)` — real, present on `RecallQueryV1`.
- `project_id=getattr(query, "project_id", None)` — **`None`, and this is correct, not lazy**: `RecallQueryV1` (`orion/core/contracts/recall.py`) has no `project_id` field at all. Verified field-by-field before defaulting, not guessed.
- `crystallization_ids=packet.crystallization_refs` — real.
- `card_refs=packet.card_refs` — genuinely empty in practice at this call site (confirmed live: `'card_refs': []`), because this collector's own call to `retrieve_active_packet()` never passes `card_refs=`/`active_cards=` the way Hub's route does — reads the real attribute, which happens to be empty, not hardcoded to `[]`.
- `trace=packet.retrieval_trace` — real, confirmed populated live (`rails`, `strategy`, `chroma_hits`, `graphiti_refs`, etc.).

`getattr()` used defensively throughout (not direct attribute access) because two pre-existing tests in this file construct fake packet objects with only an `items` attribute — direct access would have broken them.

## Schema / bus / API changes

None. Reuses `memory_crystallization_retrieval_events` and `insert_retrieval_event()` exactly as they already exist.

## Env/config changes

None.

## Tests run

```text
$ source venv/bin/activate && python -m pytest services/orion-recall/tests -q
3 failed, 95 passed, 7 warnings in 8.82s
```

The 3 failures are pre-existing and unrelated — independently confirmed by running the identical suite against clean `origin/main` (no this-branch changes applied at all):

```text
$ git checkout origin/main -- . && python -m pytest services/orion-recall/tests -q
3 failed, 92 passed, 7 warnings in 8.81s
```

Same 3 test names fail on unmodified `main`. 92 → 95 passed is exactly the 3 new tests this patch adds. No regression. (Both runs independently re-executed by the orchestrator, not taken from the implementing agent's report alone.)

## Evals run

No eval harness applies to a single collector function's logging side-effect. The live-DB verification below is the standard this session has held every claim to.

## Docker/build/smoke checks

```text
# Live verification against real Postgres (orion-athena-sql-db, localhost:55432), real data
# (98 real active crystallizations), independently re-queried by the orchestrator:

$ psql -c "select retrieval_event_id, session_id, task_type, array_length(crystallization_ids,1), created_at
           from memory_crystallization_retrieval_events
           where retrieval_event_id='503a0668-0b28-4afd-ae33-2b48a2742b8d';"

 503a0668-0b28-4afd-ae33-2b48a2742b8d | live-check-session-1 | chat_general | 98 | 2026-07-13 15:52:21.729951+00
```

Row confirmed to exist exactly as the implementing agent reported — queried independently, not trusted from the report. Left in place: this is an append-only audit table by design, the row is itself the runtime-truth artifact this whole session has been chasing, not test pollution to clean up.

## Review findings fixed

None material. Orchestrator independently read the full diff, independently re-ran the test suite (both on this branch and on clean `main` to isolate pre-existing failures), and independently re-queried the live database rather than trusting the implementing agent's report alone.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-recall/.env \
  -f services/orion-recall/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Low — `card_refs` is empty at this call site by construction (this collector doesn't pass card data into `retrieve_active_packet()` the way Hub's route does). Not a defect of this patch — it accurately logs what's actually available — but worth knowing if `card_refs` in this table's rows are ever analyzed and appear systematically empty for chat-originated rows vs. populated for Hub-originated ones.
- Severity: Low — `project_id` is always `None` for chat-originated rows since `RecallQueryV1` has no such field. Same as above: accurate, not a defect, worth knowing for any future analysis of this table.

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/orion-recall-retrieval-event-logging?expand=1`
