# Self-study wiring map (2026-09-03)

Date: 2026-09-03
Status: **current-state map** (read of the live code paths; GraphDB named-graph
contents and post-deploy analysis journal rate are **UNVERIFIED**)
Branch intent: documentation only. No schema, bus, env, or runtime change.

This is a map of what already exists, not a proposal to grow it. Two different
things share the name "self-study." Mixing them is the usual way this system
gets misread.

---

## Arsonist summary

Self-study is Orion looking at themselves: the repo, their own recent logs, and
what just changed. It is not one service. The only live write is the journal.
The RDF graph write was retired. There is no dedicated reducer. Chat-stance
still *asks* GraphDB for self-study concepts, but the named-graph env is blank
by default, so that pull is almost certainly a no-op.

Do not add a fifth "self-study" producer that stamps `source_kind='self_study'`
without an allowlist review. Thought's visual chain already had to lock out a
sibling writer after a live privacy leak.

---

## Current architecture

### Two systems, one name

**Self-model (inspect / induce / reflect / retrieve).** "What is Orion made
of." Lives in `services/orion-cortex-exec/app/self_study.py`. Verbs:
`self_repo_inspect`, `self_concept_induce`, `self_concept_reflect`,
`self_retrieve`. Policy in `self_study_policy.py`.

**Telemetry analysis.** "What just changed in my own logs." Lives in
`self_study_analysis.py`, verb `skills.self_study.analyze.v1`. One action
shape, four Postgres sources. Wired to autonomous dispatch as
`analyze_self_study_source`.

Both can write `journal_entries` with `source_kind='self_study'`. From SQL they
look related. From producers they are not.

A third writer, Hub curiosity investigation, also stamps
`source_kind='self_study'`. Thought refuses those rows on purpose.

### Layer 1 — inspect

`self_repo_inspect` walks the repo and builds an authoritative snapshot:

- service directories under `services/`
- `orion/` packages
- `orion/bus/channels.yaml`
- verb YAML + `@verb` decorators
- schema registry keys
- a small hardcoded env / touchpoint list (journal, recall, state)

Trust tier: **authoritative**. Journal write: a summary snapshot, explicitly
*not* treated as storage of record. Graph write: skipped (`channel_retired`).

### Layer 2 — induce

`self_concept_induce` clusters those facts into named concepts. Five hardcoded
clusters still exist (runtime boundary, journaling surface, recall surface,
self-study service cluster, bus write topology). Additive 2026-08 sources:

- graphify communities from `graphify-out/graph.json`
- structural-mass deltas from an in-process cache plus a durable JSONL volume
  (`SELF_STUDY_STRUCTURAL_MASS_HISTORY_PATH`)
- semantic-enrichment cache from the enrichment service volume
  (`SELF_STUDY_ENRICHMENT_CACHE_MOUNT_DIR`)

The enrichment service README still says induce does not read that cache.
**That README is stale.** `_semantic_enrichment_concepts()` in `self_study.py`
does read it.

Trust tier: **induced**. Graph write: skipped. No journal of its own.

### Layer 3 — reflect

`self_concept_reflect` turns induced concepts into findings (tension, blind
spot, growth area, and so on). Trust tier: **reflective**. Journal write: yes.
Graph write: skipped.

`self_review` workflow (`orion/cognition/workflows/registry.py`) is user- and
autonomy-invocable and runs `self_concept_reflect` twice. Historically this
workflow was `autonomous_invocable=True` with **zero invocations** until the
analysis verb was separately wired to dispatch
(`docs/superpowers/pr-reports/2026-08-25-analysis-to-self-study-journal-pr.md`).
Do not assume `self_review` is the live autonomous path.

### Retrieve

`self_retrieve` is how other code *asks*. Order:

1. GraphDB named graphs, if `SELF_STUDY_NAMED_GRAPH` is set
2. Fall back to re-running inspect / induce / reflect in process

Journal prose is **not** queried as truth (`storage_surface=journal` is
`not_queried`). Default `SELF_STUDY_NAMED_GRAPH=` in
`services/orion-cortex-exec/.env_example` is blank.

Policy (`self_study_policy.py`) is a closed list. Only:

- `legacy.plan` (mode-capped: factual / conceptual / reflective)
- `actions.respond_to_juniper_collapse_mirror.v1` (reflective)

Unknown consumers get `enabled=False`. Opt-in is `metadata.self_study.enabled`
(or the same key under options). Off unless someone turns it on.

### Analysis verb (separate)

`skills.self_study.analyze.v1` reads four already-stored tables. It does not
invent a metric, does not feed field pressure or proposal scoring, and writes
a journal entry only when a disclosed notability rule fires.

| Source | Table | What the rows actually measure |
|---|---|---|
| `concept_induction` | `memory_crystallizations` | induced memory concepts, kind + keep/reject |
| `vision_events` | `vision_events` | recognised vision events, confidence / salience |
| `affective_state` | `juniper_affective_state_log` | message volume / swear frequency — **not** emotion |
| `cocreation_signals` | `substrate_codebase_delta_log` | git / PR / graph-delta scores |

Rules: `producer_stalled`, `observation_gap`, `volume_shift`, `new_category`,
`lost_category`, `mean_shift`. Per-source cooldown 6h. Hard ceiling 16
entries/day across all four. Quiet (`skipped_not_notable`) is a correct
outcome.

Dispatch: proposal template `analyze_self_study_source` (base_priority 0.34,
read_only) → execution route of the same name → the verb. The route does **not**
pin `skill_args.source`; the verb picks whichever source has gone longest
without being analysed.

### Enrichment service (commit-triggered, not a cognition loop)

`orion-self-study-enrichment` is a thin producer of "what is this cluster for"
prose.

Trigger: `scripts/git_hooks/post-commit` →
`scripts/self_study_enrichment_hook.py`. Only if the commit touches
`services/`, `orion/bus/channels.yaml`, `orion/schemas/`, or
`orion/cognition/verbs/`. Publishes `SelfStudyEnrichmentRequestV1` on
`orion:self_study:enrichment:requested`.

On event: graphify nodes + git delta + nearby README → one `claude -p` call
(capped 8/day) → disk cache on a Docker volume. Auth is
`CLAUDE_CODE_OAUTH_TOKEN` (long-lived setup-token). Recreate the container
after rotating; `restart` does not re-interpolate Compose env.

### Write path (the only live one)

```
inspect / reflect / analysis / curiosity
        → orion:journal:write  (journal.entry.write.v1)
        → orion-sql-writer
        → journal_entries
```

Channel catalog: producers listed as `orion-actions`, `orion-cortex-exec`,
`orion-cortex-orch`. Hub curiosity also publishes here; that catalog line is
stale relative to Hub.

Graph writeback (`orion:rdf:enqueue` → named graphs `orion:self`,
`orion:self:induced`, `orion:self:reflective`) is a permanent skip. The RDF
writer is gone. `orion:self` was independently confirmed empty of triples
(2026-07-23, graph-compression verification).

### Who consumes it

**Live, and real:**

- **orion-sql-writer** — persists journal writes. Durable sink.
- **orion-thought visual chain** —
  `store.load_latest_self_study_reflection()`. Allowlists only the four
  analysis `source_ref` prefixes (`concept_induction:`, `vision_events:`,
  `affective_state:`, `cocreation_signals:`). Uses `starts_with()`, not SQL
  `LIKE`, because underscores are wildcards. A live curiosity reflection
  quoting sensitive personal content is why this is an allowlist, not
  "any `source_kind='self_study'` row."
- **Proposal / execution dispatch** — can fire the analysis verb.
- **Opt-in plan / metacog** — can inject retrieved self-study into a prompt.

**Wired, likely empty:**

- Chat stance (`chat_stance.py`) and `orion/cognition/projection_builder.py`
  register producer `self_study` → `map_self_study_to_substrate` (SPARQL
  against GraphDB named graphs, tier `graphdb_durable`). Default named-graph
  env is blank, and graph writes are retired, so this adapter skips unless
  someone set the env and old triples still exist. **UNVERIFIED** whether any
  live named graph still has content.

**Same journal tag, different beast:**

- Hub `curiosity_investigation.py` journals as `source_kind='self_study'`
  with `source_ref` prefix `curiosity:`. Thought's visual chain excludes it.

### Reducers

There is **no** dedicated self-study reducer, projection table, or cursor.

Closest reducer-*shaped* pieces:

- `self_study_analysis.py` `SourceWindow` — reduces a SQL window to counts /
  categories / timestamps, then maybe writes a journal row.
- Generic journal persistence in `orion-sql-writer`.
- Substrate unification producer `self_study` — a pull adapter, not an event
  reducer.

---

## Missing questions

1. Does GraphDB still hold any `orion:self*` triples? If not, delete or
   disable the chat-stance / projection-builder producer so it stops looking
   like a live self-model input.
2. Has `skills.self_study.analyze.v1` journaled since deploy, or is it sitting
   at `skipped_not_notable` / starving in the proposal arena?
3. Does `self_review` ever fire in production, or is inspect/induce/reflect
   still operator-manual plus harness?
4. Should Hub be listed as a `orion:journal:write` producer (curiosity), or
   should curiosity stop sharing `source_kind='self_study'`?
5. Is the enrichment cache actually populated on the cortex-exec mount, or is
   Layer 2's semantic-enrichment branch always returning `[]`?

---

## Proposed schema / API changes

None in this patch. This document does not add, remove, or rename a channel,
schema, verb, or env key.

If a follow-up heals catalog drift, the likely surfaces are:

- `orion/bus/channels.yaml` `orion:journal:write` producer list (add Hub, or
  document why curiosity is off-catalog)
- `services/orion-self-study-enrichment/README.md` "Fast-follow" section
  (induce **does** read the cache)

---

## Files likely to touch

This map (docs only):

- `docs/superpowers/specs/2026-09-03-self-study-wiring-map.md` (this file)
- `docs/superpowers/pr-reports/2026-09-03-self-study-wiring-map-pr.md`

Code already in the map (do not edit in this patch):

- `services/orion-cortex-exec/app/self_study.py`
- `services/orion-cortex-exec/app/self_study_policy.py`
- `services/orion-cortex-exec/app/self_study_analysis.py`
- `services/orion-cortex-exec/app/verb_adapters.py`
- `services/orion-self-study-enrichment/`
- `orion/substrate/relational/adapters/self_study.py`
- `orion/cognition/projection_builder.py`
- `services/orion-cortex-exec/app/chat_stance.py`
- `services/orion-thought/app/store.py` / `visual_chain.py`
- `services/orion-hub/scripts/curiosity_investigation.py`
- `config/proposals/proposal_policy.v1.yaml`
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`

---

## Non-goals

- Re-enabling RDF / GraphDB writeback
- Wiring retrieve into more consumers
- Adding a self-study reducer or projection table
- Changing analysis notability bars
- Expanding Thought's `source_ref` allowlist
- Fixing the enrichment README in this patch (called out; not edited)
- Live GraphDB / journal verification (called out as UNVERIFIED)

---

## Acceptance checks

- A reader who has never opened `self_study.py` can say what inspect, induce,
  reflect, retrieve, analysis, and enrichment each **do**, and which of them
  write.
- The two homonyms (self-model vs analysis) and the third journal stamp
  (curiosity) are named as separate producers.
- Graph writeback is described as retired, not as a live dual-write.
- "No dedicated reducer" is explicit.
- UNVERIFIED claims are labeled (GraphDB contents, analysis post-deploy rate).
- No new metric, channel, or schema in this changeset.

---

## Recommended next patch

Pick one, in this order, after a live check — do not implement from this map
alone:

1. **Prove or kill the GraphDB adapter.** Query the named graphs. If empty,
   stop registering `producer_id="self_study"` in chat stance / projection
   builder, or document it as a dead pull.
2. **Count analysis journal rows since deploy.** If zero, the dispatch wiring
   is the gap again, not the analysis code.
3. **Stale README:** one-paragraph fix in
   `services/orion-self-study-enrichment/README.md` stating induce *does* read
   the cache.

---

## Trust tiers (quick)

| Mode | Allowed tiers |
|---|---|
| factual | authoritative |
| conceptual | authoritative, induced |
| reflective | authoritative, induced, reflective |

Retrieve never upcasts. Policy may *downgrade* a requested mode; it never
widens past the consumer's max.
)
