# Cost trace: what happens if the disabled autonomy/spark RDF graphs get turned back on

## Context

Follow-up to `docs/superpowers/specs/2026-07-11-drive-engine-concept-induction-deactivation-design.md`
and same-day investigation into why `orion-athena-fuseki`'s on-disk store reached 611GB.
That investigation found four graphs accounting for ~88M of the store's triples, all with
zero retention:

| Graph | Artifacts | Triples | Status as of 2026-07-14 |
| :--- | ---: | ---: | :--- |
| `autonomy/drives` | 445,506 | 37,452,939 | actively growing (~3,800/day) |
| `autonomy/identity` | 444,940 | 34,742,343 | stopped 2026-06-19 |
| `autonomy/goals` | 220,287 | 10,596,354 | stopped 2026-06-19 |
| `spark/concept-profile` | 12,914 | 5,611,466 | growth likely stopped 2026-07-12 |

All four are currently write-only: `AUTONOMY_GRAPH_BACKEND=disabled` (flipped in `e9b233e9`,
2026-06-19, "load/stability-motivated" per that commit's message) means nothing on the live
chat path reads `autonomy/identity` or `autonomy/goals` today, and
`CONCEPT_PROFILE_REPOSITORY_BACKEND=local` has never once been set to `graph` since the
concept-profile feature shipped (confirmed via `git log` on that key) — so `spark/concept-profile`
has never had a live reader at all. `autonomy/drives` alone kept receiving writes past
2026-06-19 because `RDF_SKIP_KINDS` in the live `.env` omits `memory.drives.audit.v1`, most
likely by omission rather than intent.

**The question this doc answers**: if someone flips these backends back on (restoring the
live chat-path reads that existed before 2026-06-19), what actually happens downstream? This
was measured empirically against the live store, not estimated from code alone — every timing
below is a real query run against the real graphs on 2026-07-14, with Fuseki given generous
headroom (`FUSEKI_JVM_XMX=96g`, container limit 110g, confirmed idle at ~18% memory
utilization during these tests, so none of these numbers are inflated by memory pressure).

## Where the read path actually lives

`services/orion-cortex-exec/app/chat_stance.py:2050` calls `build_autonomy_repository()` on
what is architecturally a per-chat-turn path (confirmed via `chat_stance.py:2713` threading
the result into `ctx["chat_autonomy_repository_status"]`, consumed by
`services/orion-cortex-exec/app/router.py:576,615`). When `AUTONOMY_GRAPH_BACKEND` resolves
to anything other than disabled, `build_autonomy_repository()` returns a
`GraphAutonomyRepository` (`orion/autonomy/repository.py:232`) instead of the local/file-based
fallback — meaning every turn fires live SPARQL against Fuseki instead of a cheap local read.

**`orion/autonomy/repository.py` has zero caching of any kind** — no `lru_cache`, no TTL, no
memoization anywhere in the file (confirmed by grep). Every call is a fresh live query.

## Per-graph findings

### `autonomy/drives` — `_fetch_drive_audit` (`repository.py:372-456`)

Query shape: `ORDER BY DESC(?created_at) ... LIMIT 1` — find the single newest `DriveAudit`
artifact for a subject/model_layer/entity_id.

```
trial 0: 8.00s
trial 1: 7.09s
trial 2: 5.41s
```

(Run unfiltered by subject as a worst-case proxy; a real subject filter would not meaningfully
help, since `ORDER BY DESC` still has to consider every matching row to find the max — see
identity below, which *is* subject-filtered and is slower, not faster.)

**Verdict**: 5-8 seconds per call, uncached, on every chat turn, just for this one field.

### `autonomy/identity` — `_fetch_identity` (`repository.py:342-370`)

Query shape: same `ORDER BY DESC(?created_at) ... LIMIT 1` pattern, filtered by real
`subjectKey="orion"`, `modelLayerKey="self-model"`, `entityId="self:orion"` (sampled live from
the store — these are real, in-use keys, not synthetic).

```
identity (latest, subject=orion): 22.94s
```

**Verdict**: worse than drives' unfiltered test, despite being filtered to one subject. The
`ORDER BY` still forces the engine to consider the full matching set before it can find the
max — subject filtering narrows *which* rows qualify but doesn't avoid the sort.

### `autonomy/goals` — `_fetch_active_goals` (`repository.py:458-546`)

Query shape: more sophisticated than the other two — an inner `GROUP BY ?drive_origin` with
`MAX(?priority)`, then joins back to fetch full rows, filtered to active (non-archived/
superseded/completed) proposals, same real subject key.

```
goals (active, subject=orion): 16.39s
```

**Verdict**: still far too slow for a per-turn call, despite being the most query-optimized of
the three ("Aggregate in SPARQL... so Fuseki does not scan/sort every ProposedGoal" per the
code's own comment at `repository.py:459-460` — a real, intentional optimization attempt that
still isn't enough at this data volume).

### `spark/concept-profile` — `build_latest_profile_query` (`orion/spark/concept_induction/graph_query.py:117-149`)

Query shape: **not** `ORDER BY LIMIT 1` — uses `FILTER NOT EXISTS` as an anti-join: for every
candidate profile row, checks whether any *other* profile exists with a higher
revision/timestamp for the same subject, and only keeps rows where none does.

```
concept-profile (latest revision, subject=orion): TIMED OUT at 60s
```

**Verdict**: this is the worst of the four. `FILTER NOT EXISTS` at this shape is effectively a
per-candidate scan against the rest of the subject's rows — it didn't even complete in 60
seconds against 12,914 artifacts (the smallest of the four graphs by artifact count), which
tells you the anti-join pattern itself is the dominant cost, not raw data volume. This query
shape would need to be redesigned (e.g., a maintained "current revision pointer" instead of a
computed anti-join) before it could ever be safe to run per-turn, independent of retention.

## What this means

**Turning any of these back on today, as configured, would very likely reproduce (or worsen)
the original Fuseki memory-pressure incident this investigation started from** — not as a
one-time cost, but as a sustained, uncached, multi-second-per-field cost on every single chat
turn, live, forever. The June 19 rollback (`e9b233e9`) was empirically correct to do, even
though its stated motivation ("load/stability-motivated") wasn't backed by a specific
measurement at the time — this doc is that measurement, three weeks later.

**Retention alone would not be sufficient to safely re-enable any of these**, independent of
how much history gets pruned:

- `drives`/`identity`/`goals` all use `ORDER BY DESC ... LIMIT N` or an aggregate-then-join
  shape. At `drives`' current write rate (~1 artifact every ~23 seconds), even a 30-day
  retention window is still ~130K records to sort through per query — likely still
  multi-second. The query pattern needs to change (e.g., a maintained "latest artifact per
  subject" pointer/index, updated at write time instead of computed at read time) before
  retention volume is the limiting factor rather than query shape.
- `concept-profile`'s `FILTER NOT EXISTS` anti-join is the most fundamentally unscalable shape
  of the four — it timed out against the *smallest* graph by artifact count, meaning volume
  reduction alone won't fix it either.

**Recommendation, if re-enabling any of these is ever wanted**: treat "reduce stored volume"
(retention/pruning — a separate, already-scoped effort) and "make the read query fast enough
for a live per-turn path" (query/index redesign) as two independent problems. Solving only the
first does not make re-enabling safe.

## Test methodology notes

- All timings measured via `docker exec orion-athena-substrate-runtime python3 -c "..."`,
  using `requests` directly against `SUBSTRATE_GRAPH_QUERY_URL` with real `SUBSTRATE_GRAPH_USER`/
  `SUBSTRATE_GRAPH_PASS` credentials already present in that container's environment — no
  synthetic data, no mocked backend.
- Fuseki was given a one-time, deliberate memory bump for this investigation
  (`FUSEKI_JVM_XMX` 36g→96g, `FUSEKI_JVM_XMS` 8g→16g, `FUSEKI_MEM_LIMIT` 40g→110g, restart
  required) specifically so that timing results here reflect real query cost, not artificial
  memory starvation. Confirmed idle utilization ~18% during these tests. Whether to revert this
  bump afterward is a separate, still-open operational decision — not addressed in this doc.
- Real subject/model_layer/entity_id keys (`orion` / `self-model` / `self:orion`) were sampled
  live from the `autonomy/drives` graph rather than guessed, to ensure filtered-query timings
  reflect actual production data shape.
