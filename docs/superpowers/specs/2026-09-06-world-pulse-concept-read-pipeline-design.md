# World-pulse → Concept Atlas read pipeline — design

- Date: 2026-09-06
- Status: DESIGN (approved in brainstorm; not implemented)
- Owner: Juniper / Orion
- Related: `docs/superpowers/specs/2026-07-07-world-pulse-curiosity-followups-design.md`,
  `docs/superpowers/specs/2026-07-15-concept-atlas-graph-pipeline-design.md`,
  Curiosity Atlas loop (`services/orion-hub/scripts/curiosity_investigation.py`)

## Arsonist summary

World pulse already saves news (digest items + “Orion went looking” findings) in
Postgres, but Concept Atlas stays chat-heavy and Orion never really *reads* those
articles into the mesh. This design adds a **sibling agent/FCC pipeline** with
**two wallets** (Stage 1 heavy read, Stage 2 seeded curiosity-style chew) that
**never debit Curiosity Atlas**. Stage 1 and Stage 2 form a loop: after a full
curiosity pass, interesting threads can call Stage 1 tools again and keep hopping.
Outputs land in Concept Atlas (with world-pulse provenance), curiosity priors, and
an inspectable journal/trace. Backfill reuses the same seed queue from historical
digests.

## Current architecture

- **World pulse** (`services/orion-world-pulse`) builds digests with mechanical
  cluster “summaries” (not LLM article reads). Curiosity findings
  (`CuriosityFindingV1` on `DailyWorldPulseV1.curiosity_followups`) are scraped
  blurbs from gap-fill fetches — live in every recent digest (53/53 checked
  2026-09-06) but never become concept nodes.
- **Concept Atlas** is mostly chat → topic-foundry → Falkor substrate. Provenance
  already exists (`SubstrateProvenanceV1`: `source_kind`, `producer`,
  `evidence_refs`); topic-foundry filters on `producer`. No path from
  `CuriosityFindingV1` → `ConceptNodeV1` (graphify confirmed).
- **Curiosity Atlas** (`CuriosityInvestigation`) is an agent/FCC Hub loop with its
  own daily cap + cooldown writing `:Prior` nodes on `orion_worldview`.
- **Concept induction** consumes `world.pulse.run.result.v1` for metabolism /
  episode-journal reuse of followups — not for Concept Atlas materialization.

## Locked decisions (brainstorm)

| Decision | Choice |
|---|---|
| Seed order | Findings first, then digest items |
| Compute | Agent / FCC for all heavy lifting |
| Stage 1 budget (Wallet A) | ~6 / day |
| Stage 2 budget (Wallet B) | ~6 / day (own wallet) |
| Curiosity Atlas budget | **Never charged** by this pipeline |
| Hop policy | Open; no topic cage; multi-hop while interesting |
| Stage 1 ↔ Stage 2 | Loop: Stage 2 may re-invoke Stage 1 tools |
| Landings | Concept Atlas + curiosity priors + journal/trace |
| Backfill | Same queue; paced by live wallets; dedupe by stable `seed_id` |

## Design

### Queue & budgets

```text
World Pulse / backfill
        │
        ▼
 Seed queue (findings → digest items)
        │
        ▼
 ┌──────────────── Wallet A (~6/day) ────────────────┐
 │  Stage 1: heavy read / fetch / open web tools     │
 │  → handoff artifact                               │
 └───────────────────────┬───────────────────────────┘
                         │ seed
                         ▼
 ┌──────────────── Wallet B (~6/day) ────────────────┐
 │  Stage 2: seeded curiosity pass · priors · hops   │
 │  · multi-hop while interesting                     │
 │  · may call Stage 1 tools (debits A, not B)        │
 └───────────────────────┬───────────────────────────┘
                         │
         Concept Atlas · priors · journal/trace
```

Curiosity Atlas loop remains a separate process with its own cap; it may later
*observe* priors this pipeline wrote, but this pipeline does not spend Atlas slots.

### Stage 1 — heavy read (Wallet A)

Per turn:

1. Dequeue one seed: `url`, `title`, `section`, `run_id`, `kind` (`finding` |
   `digest_item`).
2. Agent FCC fetches/reads the page; may open-web-search (no topic cage) while in
   Stage 1.
3. Emit a **handoff artifact** Stage 2 can start from:

| Field | Purpose |
|---|---|
| `seed_ref` | World-pulse linkage (`run_id`, `url`, `kind`) |
| `what_i_learned` | Orion’s prose take from the read |
| `candidate_priors` | Draft claims for worldview priors |
| `concept_candidates` | Labels / link hints for Concept Atlas |
| `open_threads` | Threads that might need another heavy fetch |
| `trace_id` | Correlates Stage 1 → Stage 2 → hops |

4. Debit Wallet A once per Stage 1 turn (including Stage-2-driven re-entry).

Stage 1 does **not** run the full prior/hop pass.

### Stage 2 — seeded curiosity pass (Wallet B)

Per turn:

1. Start from a Stage 1 handoff (not a blank prior menu).
2. Agent FCC runs curiosity-style work: form/test/revise priors on
   `orion_worldview`, mesh links, hop while interesting (no topic cage).
3. May re-invoke Stage 1 tools mid-pass when a thread needs another heavy
   read/search → **debits Wallet A**.
4. Multi-hop inside Stage 2 debits Wallet B until the pass ends or budgets/safety
   ceiling stop it.
5. Stop conditions: nothing interesting left; Wallet A or B exhausted; hard
   safety ceiling on tool round-trips per pass (exact number in implementation
   plan — default sketch: small single-digit, operator-tunable).
6. Never touches Curiosity Atlas daily cap or cooldown.

### Landings

**Concept Atlas (Falkor substrate)**

- Materialize nodes/edges from Stage 1 candidates and Stage 2 mesh links.
- Provenance (required):
  - `source_kind`: `world_pulse.read` | `world_pulse.curiosity_hop`
  - `producer`: `world_pulse_read_pipeline`
  - `evidence_refs`: article URL + `run_id` + `trace_id`
- Hub filter “from world pulse” via `producer` / `source_kind` (same seam as
  topic-foundry’s producer filter).

**Curiosity priors (`orion_worldview`)**

- Stage 2 forms/tests/revises `:Prior` nodes from the handoff.
- Atlas open-ended loop may see them later without paying for this pipeline.

**Journal / trace**

- Inspectable chain per seed: Stage 1 → Stage 2 → Stage 1 re-entries → hops.
- Enough to answer “what did Orion learn from this article?” without log diving.

Raw scrape dumps do **not** enter the atlas — only distilled candidates + edges.

### Backfill

Live counts at design time (2026-09-06, `conjourney.world_pulse_digest`):

- 53 digests (all with ≥1 curiosity followup)
- ~260 finding-articles
- ~636 digest items

Procedure:

1. Scan digests chronologically (default: all retained since post-2026-07-23 DB
   rebuild).
2. Enqueue with stable `seed_id` for idempotent dedupe.
3. Drain via live Wallet A/B pace — no FCC burst.
4. Findings first; digest items after (operator may stop at findings-only).

Backfill does not invent summaries; Stage 1 still performs the real read.

### Implementation shape (approach)

**Sibling Hub loop(s)** mirroring `CuriosityInvestigation` lifecycle (tick,
cooldown, daily cap), not a new microservice:

- Shared helpers where cheap (FCC spawn, prior write path, journal).
- Distinct config keys / counters for Wallet A and Wallet B.
- Seed queue durable enough to survive Hub restart (Postgres or Redis — choose
  in plan; prefer Postgres if Hub already has the pool).

Bus/schema: only if a new durable job envelope is required; prefer reusing
existing world-pulse SQL as the source of truth for seeds.

## Files likely to touch (implementation)

- `services/orion-hub/scripts/` — new loop module(s), routes/status, wiring in
  `main.py`
- `orion/substrate/adapters/` — world-pulse → `ConceptNodeV1` / edges mapper
- `orion/curiosity/` — seeded prior formation reuse (thin; no Atlas budget)
- `services/orion-hub/tests/` + evals for handoff + provenance
- Operator backfill script under `scripts/` or Hub scripts
- Env: Wallet A/B caps, enable flags (`.env_example` + sync)

## Non-goals

- Debiting or merging with Curiosity Atlas daily cap
- LLM summarization inside the world-pulse *digest builder* (Stage 1 owns reading)
- Dumping every RSS article body into Falkor without distillation
- Replacing topic-foundry chat ingest
- Unbounded web crawl (budgets + safety ceiling bound the loop)
- Keyword-triggered emotional / conversational mode changes

## Acceptance checks

1. A seeded finding run produces Concept Atlas nodes with
   `producer=world_pulse_read_pipeline` and `evidence_refs` containing the URL.
2. Atlas UI (or Cypher probe) can distinguish world-pulse nodes from chat /
   topic-foundry nodes.
3. Wallet A and Wallet B counters increment independently; Curiosity Atlas
   daily counter does **not** move during a pipeline run.
4. Stage 2 can re-enter Stage 1 tools; that re-entry increments Wallet A only.
5. Trace/journal links Stage 1 → Stage 2 for one seed (`trace_id`).
6. Backfill enqueues historical findings without duplicate Stage 1 work for the
   same `seed_id`.
7. Focused tests cover handoff schema + provenance mapping; eval or smoke shows
   one live (or fixture) article path end-to-end **UNVERIFIED until implemented**.

## Risks / concerns

- **Severity: medium — FCC cost.** Two wallets × agent turns + loops can spend
  hard; caps and safety ceiling are load-bearing.
- **Severity: medium — loop runaway.** Stage 2 → Stage 1 → Stage 2 needs an
  explicit round-trip ceiling per seed/`trace_id`.
- **Severity: low — prior quality.** Open hops may write noisy priors; promotion
  / review seams already exist — do not invent a new keyword taxonomy.
- **Severity: low — collision.** Other Hub agents touch
  `curiosity_investigation.py` / FCC spawn; keep the sibling loop file-bounded.

## Recommended next patch

1. Implementation plan (`docs/superpowers/plans/…`) with thin vertical slice:
   seed queue + Wallet A read → handoff → Concept Atlas write with provenance.
2. Follow-on: Stage 2 seeded pass + loop-back + Wallet B.
3. Follow-on: backfill script + Hub status surface.

## Ambiguities deferred to plan (not blockers)

- Exact Stage 2 daily default if operator overrides (locked brainstorm default: 6)
- Durable queue backend (Postgres vs Redis)
- Numeric tool round-trip ceiling per `trace_id`
