# Orion Memory Consolidation

Subscribes to `orion:memory:turn:persisted` (sql-writer post-commit outbox), classifies each chat turn via LLM gateway quick lane logprobs, patches `chat_history_log.spark_meta`, tracks consolidation windows, and on boundary closure runs a **deterministic consolidation gate** (default) or legacy graph suggest.

## Consolidation output modes

| `MEMORY_CONSOLIDATION_OUTPUT` | Behavior |
|-------------------------------|----------|
| `crystallization_propose` (default) | Run `consolidation_memory_gate`; **skip** low-signal windows (`consolidation_status=skipped`) or **propose** `MemoryCrystallizationV1` for governor review |
| `graph_draft` | Legacy path: LLM `memory_graph_suggest` + pending graph draft insert (manual bridge only) |
| `skip_only` | Run gate for traceability; always mark window skipped — no crystallization or graph draft |

Gate thresholds: `MEMORY_CONSOLIDATION_MIN_NOVELTY` (default `0.35`), `MEMORY_CONSOLIDATION_MIN_SIGNIFICANCE` (default `0.40`). Both floors require the window not be all low-info-social (`is_low_info_social` on every turn's prompt and response) as corroboration — a bare novelty/significance float alone is not sufficient, since a noisy classifier score on a short greeting-only turn was previously enough to crystallize it (`repair_signal` and `substantive_shift` are unaffected; they already require an independent shift-kind classification).

Grammar repair evidence (read-only): `MEMORY_CONSOLIDATION_FETCH_GRAMMAR_EVIDENCE=true` queries `grammar_events` by `hub.chat:{NODE_NAME}:{correlation_id}` trace. Optional override DSN: `MEMORY_CONSOLIDATION_GRAMMAR_DSN`.

**Note:** Proposed crystallization IDs are stored in `memory_consolidation_windows.draft_id` until a dedicated `crystallization_id` column migration lands.

## Cross-window concept-relation resolution (off by default)

Same-window duplicate detection (`orion.memory.crystallization.detection.detect_duplicates`) requires `scope_overlap`, which is structurally always `False` across two different consolidation windows — every crystallization gets a unique per-window `scope`, so two windows can never share one. `orion.memory.crystallization.concept_relation` adds a second, cross-window path: vector-similarity candidate retrieval (`candidate_retrieval.fetch_similar_candidates`, not scope-gated) followed by one bounded, structured-output LLM call that judges `same` / `refines` / `contradicts` / `unrelated` against the nearest existing active crystallizations of the same `kind`.

Dispatch is deliberately conservative: `same` reinforces the existing target (identical mechanism to the same-window path). `refines` and `contradicts` only attach a typed link to the *new* candidate's own `links` (persisted to `memory_crystallization_links` on insert, same as any other crystallization) — they never mutate or supersede the existing target's status. That stays a human decision via the existing `/api/memory/crystallizations/{id}/links` and supersede endpoints. Every decisive branch (`same`/`refines`/`contradicts`) stamps `provenance.concept_relation` (relation, target id, confidence) on the affected row for audit, independent of which branch acted.

**Decision log + belief-revision digest.** Every real LLM decision — including `unrelated` and sub-floor `contradicts`/`refines` that the dispatch above discards — is written to `memory_concept_relation_decisions` (`orion.memory.crystallization.repository.insert_concept_relation_decision`, guarded to never raise). `scripts/concept_relation_digest.py` (repo root, run on demand or via cron — not a live service loop) reads undigested rows and reports call volume / relation distribution / near-miss counts under `CONCEPT_RELATION_CONFIDENCE_FLOOR`, then creates one `reflection`-kind crystallization per real belief-revision decision (`same`/`refines`/`contradicts` that cleared the floor) — a structured, deterministic trace of Orion revising its own beliefs, not just a link nobody reads back.

Ships flag-gated off; flipping the flag alone does not activate anything without also configuring the embed/chroma hosts (both default to empty string):

| Env | Default | Purpose |
|-----|---------|---------|
| `CONCEPT_RELATION_RESOLUTION_ENABLED` | `false` | Master flag for this path |
| `CONCEPT_RELATION_CONFIDENCE_FLOOR` | `0.6` | Minimum LLM decision confidence to act; below this, falls through to the normal formation-policy path unchanged |
| `CONCEPT_RELATION_CANDIDATE_LIMIT` | `5` | Max vector-similar candidates fetched and sent to the LLM prompt |
| `CONCEPT_RELATION_TIMEOUT_SEC` | `8.0` | RPC timeout for the relation-judgment call |
| `CRYSTALLIZER_EMBED_HOST_URL` | *(empty)* | Embedding HTTP endpoint for candidate retrieval — must be set for this feature to do anything |
| `CRYSTALLIZER_EMBED_TIMEOUT_MS` | `8000` | Embed call timeout |
| `CHROMA_HOST` / `CHROMA_PORT` | *(empty)* / `8000` | Chroma vector store for candidate retrieval — must be set alongside the embed host |
| `CRYSTALLIZER_VECTOR_COLLECTION` | `orion_memory_crystallizations` | Chroma collection name (matches Hub's projection collection) |

### Scheduled maintenance (Athena cron)

`scripts/concept_relation_digest.py` is a standalone script, not a live service loop (see above) -- something external has to run it. Install on the host that runs the memory-consolidation stack (`crontab -e` as the operating user):

```cron
# Concept-relation decision digest -- turns memory_concept_relation_decisions rows
# into reflection crystallizations. Idempotent (only acts on digested=false rows),
# safe to run frequently. Requires POSTGRES_URI in the shell environment or a
# sourced .env; see services/orion-hub/.env.
*/30 * * * * cd /mnt/scripts/Orion-Sapienform && POSTGRES_URI=$(grep -m1 '^POSTGRES_URI=' services/orion-hub/.env | cut -d= -f2-) make concept-relation-digest >> /mnt/scripts/Orion-Sapienform/logs/orion-concept-relation-digest.log 2>&1
```

**If this cron entry dies, is dropped after a host migration, or the job starts failing silently, nothing else will notice on its own.** `make check-concept-relation-digest-liveness` is the fail-safe: it queries the real backlog (oldest undigested decision's age, not a heartbeat file that can go stale independently of the thing it claims to represent) and exits non-zero if it exceeds `MAX_AGE_HOURS` (default 3h -- generous headroom over the 30-minute cadence above). Run it by hand any time you suspect the digest stopped running:

```bash
POSTGRES_URI=... make check-concept-relation-digest-liveness
# or, to tighten/loosen the threshold:
POSTGRES_URI=... make check-concept-relation-digest-liveness MAX_AGE_HOURS=1
```

A clean exit (0) with "no undigested decisions pending" means the loop is closing. A STALE failure means: check `crontab -l` for the entry above, check `logs/orion-concept-relation-digest.log` for errors, and if this is a fresh host or post-migration box, re-add the crontab line (it does not persist itself -- see "Recreate ops after a fresh host" below).

### Recreate ops after a fresh host, lost crontab, or disaster recovery

Use this checklist any time this service (or its cron dependency) is missing after a host swap:

1. **Confirm the digest cron entry is actually installed:** `crontab -l | grep concept_relation_digest`. If empty, paste the block from "Scheduled maintenance" above.
2. **Confirm `logs/orion-concept-relation-digest.log` exists** (create the `logs/` dir if this is a fresh checkout: `mkdir -p /mnt/scripts/Orion-Sapienform/logs`).
3. **Run the liveness check once by hand** (`make check-concept-relation-digest-liveness`) to confirm the new cron entry is actually firing, rather than waiting up to 3 hours to find out.
4. **Do not assume `CONCEPT_RELATION_RESOLUTION_ENABLED=true` implies the digest is scheduled** -- they are two independent things that must both be true for the loop documented above to actually close. This exact "flag on, dependency not wired" pattern has already caused silent no-ops twice in this repo (`CONCEPT_RELATION_RESOLUTION_ENABLED` itself missing its embed/chroma hosts on first activation, and `RECALL_GRAPHITI_IN_CHAT` missing its adapter URL) -- checking both halves explicitly, every time, is cheaper than re-discovering this.

## Drive-history reflection synthesis (manual/on-demand only -- NOT cron'd)

`scripts/drive_history_reflection_synthesis.py` (repo root) reads Orion's own real,
persisted drive-activation history and synthesizes ONE `reflection`-kind
`MemoryCrystallizationV1` observing a long-horizon pattern -- e.g. "continuity has
been the dominant drive in most audited ticks this week." Source data: `DriveAuditV1`
(`orion/core/schemas/drives.py`) is computed on every DriveEngine tick and persisted,
append-only, by `services/orion-sql-writer` to the Postgres `drive_audits` table
(the old Fuseki `drives` graph froze on 2026-06-19 and was removed as both a write
and read path on 2026-07-15) -- unlike the "latest value only" stores this repo
already has for the same signal (`LocalProfileStore` / `autonomy_state_v2`'s
single-row UPSERT), this table is a genuine historical time-series, one row per
tick, including per-tick `drive_pressures` / `active_drives` JSONB.

**Architecture is deliberately split into a deterministic reducer stage and a
narrow LLM-phrasing stage** (event -> schema -> trace -> reducer -> projection ->
LLM phrasing -> crystallization, per this repo's event-substrate-first mandate) --
the LLM is never shown raw per-tick rows:

1. Fetch real `DriveAuditV1` ticks from Postgres `drive_audits` (bounded by
   `--max-events`, most recent first, same DSN as the crystallization write path).
2. `reduce_drive_history()` -- a **pure, unit-tested Python function** (same bar as
   `orion/spark/concept_induction/drive_tension.py`: synthetic-input/known-output
   tests, zero LLM involvement) -- aggregates them into dominant-drive counts/shares,
   a per-day breakdown, active-drive frequency, and mean pressures.
3. `build_fact_sheet()` renders that aggregation into a small numbered list of
   already-computed, already-verified fact strings (real dates, real counts, real
   percentages, real timestamps).
4. The LLM (bus RPC, same pattern as `concept_relation.py::resolve_concept_relation()`)
   receives ONLY the fact sheet and is asked to phrase ONE narrative sentence,
   citing specific facts by number.
5. `parse_and_validate_narrative()` enforces the grounding guardrail: every cited
   fact's real literal tokens (drive name, date, percentage, or timestamp) must
   appear verbatim in the narrative text, or the run is rejected -- an LLM cannot
   pass validation by inventing plausible-sounding but fake specifics.
6. Only then is a `reflection` crystallization written, with evidence refs citing
   both the aggregation object and the real cited `DriveAuditV1` artifacts, so a
   human reviewer can verify the narrative against what was actually computed.

**Guardrail: refuses to synthesize on thin data.** Below `MIN_EVENTS=5` real ticks
or `MIN_DISTINCT_DAYS=2` distinct calendar days in the queried window, the reducer
marks the aggregation insufficient and the script exits cleanly reporting exactly
why -- no LLM call is made, nothing is written. `MIN_DISTINCT_DAYS` exists because
DriveEngine ticks several times a minute in a single session, so event *count*
alone can be satisfied entirely within one sitting; requiring real day-spread is
what distinguishes an actual long-horizon pattern from a burst.

Run:

```bash
POSTGRES_URI=postgresql://user:pass@host:port/db python scripts/drive_history_reflection_synthesis.py
python scripts/drive_history_reflection_synthesis.py --postgres-uri postgresql://... --since-days 14
python scripts/drive_history_reflection_synthesis.py --json
```

| Env / flag | Default | Purpose |
|---|---|---|
| `--postgres-uri` / `$POSTGRES_URI` | *(required)* | Same convention as `concept_relation_digest.py` |
| `--subject` | `orion` | `DriveAuditV1.subjectKey` to read |
| `--since-days` | `30` | Window size; real coverage found is always reported, never assumed |
| `--max-events` | `500` | Cap on raw ticks fetched from `drive_audits` (most recent first) |
| `--redis` / `$ORION_BUS_URL` | `redis://localhost:6379/0` | Bus URL for the LLM gateway RPC |
| `--llm-route` | `metacog` | Gateway route for the narrative-phrasing call |

**This is NOT scheduled or cron'd in this patch, unlike `concept_relation_digest.py`
above.** It is explicitly a manual/on-demand tool: its output (a narrative claim
about Orion's own long-horizon tendencies) needs human review before anyone should
trust it as a recurring automated process. Run it by hand, read the generated
`summary` text critically, and decide separately whether a cron entry is warranted
once the grounding guardrail and prompt have been proven out on real data.

## Channels

| Direction | Channel |
|-----------|---------|
| In | `orion:memory:turn:persisted` |
| Out | `orion:chat:history:spark_meta:patch` |
| Out (threshold) | `orion:signals:memory_consolidation` (`signal.memory_consolidation.turn_change`) |
| Out (propose) | `orion:memory:crystallization:proposed` (`memory.crystallization.proposed.v1`) |

## Turn change appraisal

Each persisted turn (after the first in a window) gets a logprob-calibrated `turn_change_appraisal` patch on `spark_meta`: novelty score, shift kind, confidence, and baseline mode (`prior_turn` or `session_window` fallback). The first turn in a window uses `turn_change_status=skipped` (no baseline, no LLM call). High-confidence novel turns also emit `OrionSignalV1` on `orion:signals:memory_consolidation`.

| Env | Default | Purpose |
|-----|---------|---------|
| `TURN_CHANGE_CLASSIFY_ROUTE` | `metacog` | Gateway route for classify RPC (`metacog` or `quick`) |
| `TURN_CHANGE_CONFIDENCE_MARGIN` | `0.15` | Re-appraise vs session window when novelty margin is below this |
| `TURN_CHANGE_SUBSTRATE_THRESHOLD` | `0.65` | Minimum novelty to emit substrate signal |
| `TURN_CHANGE_WINDOW_TURNS` | `3` | Prior turns in session-window baseline |
| `CHANNEL_SIGNALS_PREFIX` | `orion:signals` | Bus prefix for organ signal publish |

## Bring-up

```bash
docker compose --env-file ../../.env --env-file ../orion-bus/.env -f docker-compose.yml up -d --build
```

Apply Postgres migration: `services/orion-sql-db/manual_migration_memory_consolidation_v1.sql`

## Smoke

```bash
PYTHONPATH=. python scripts/smoke_memory_consolidation_pipeline.py
bash scripts/smoke_memory_consolidation_gate.sh
```

Gate smoke runs deterministic unit tests (greeting skip + substantive propose). Pipeline smoke requires live stack (bus, sql-writer, postgres, llm-gateway, cortex).
