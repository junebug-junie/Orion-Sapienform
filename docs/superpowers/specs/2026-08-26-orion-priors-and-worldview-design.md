# Priors, research, and the concept graph

Design mode. Nothing here is built yet. **Revised 2026-08-26 after two
corrections from Juniper and one thing I had simply never looked at.**

## What changed in revision 1, and why it matters

Three claims in the first draft were wrong, and they were wrong in the same
direction — I called things broken that were working as designed, because I had
not looked at the surface Juniper actually uses.

1. **"The 290 approved crystallizations are junk because `subject == summary`."**
   Wrong. Juniper: *"they are basically the chat statements, but they are
   important chats so they get crystalized, not compressed and that's okay."*
   Preserving an important exchange VERBATIM is the point of crystallizing it —
   compression is what happens to everything else. `subject == summary` is the
   signature of a deliberate decision, not of a defect.
2. **"Orion has no real concepts."** Wrong, and the reason is embarrassing:
   there is a **Concept Atlas** in Hub, live at `/concept-atlas`, backed by the
   `orion_substrate` FalkorDB graph, which I never opened. It holds 18 concepts
   and 15 typed edges with a real promotion lifecycle. See below.
3. **"Build a new `orion_worldview` graph."** Reversed — see "Where findings
   go".

The pattern in all three: I inferred structure from Postgres shape instead of
looking at the system Juniper operates. CLAUDE.md 0A's metric gate step 4 says
pull real data and look at it; I pulled real data and did not look at all of
it.

## Arsonist summary

**The loop that is deployed right now cannot learn.** It shows Orion a random 12 of 646 approved concepts every four hours, forever. Nothing accumulates between runs. Run 40 is exactly as ignorant as run 1, because the only state carried forward is "when did I last run" — Orion cannot become less uncertain about anything, because it never records what it was uncertain about.

Random sampling was the right correction to a keyword detector — it removed a fake choice. But it is not curiosity either. Curiosity is not *sampling*; it is **having an expectation and going to find out whether it holds**.

So: give Orion **priors** — claims it holds, with a confidence and a status — and make the loop about testing them. Selection stops being random and starts being driven by *what Orion is most unsure of*, which is still Orion's pick (it chooses which uncertainty to chase) but is no longer memoryless. Findings update the prior, spawn new ones, and land in a **world-view graph** that is structured enough to be queried rather than only read.

The journal stays, because Juniper reads prose. The graph is added, because Orion needs something it can traverse.

## Current architecture

| piece | state |
|---|---|
| `CuriosityInvestigation` (Hub loop) | live: cooldown 4h, cap 3/day, both Redis-persisted; `harness_step_count >= 3` lookup gate; 1500s budget |
| material shown | random sample: approved crystallizations + relation decisions |
| the turn | real `execute_unified_turn`, read-only, tools on, Orion picks its subject |
| output | free prose → one `journal_entries` row, `source_ref='curiosity:<run_id>'` |
| memory between runs | **none, except the cooldown stamp** |
| `memory_crystallizations` | 1,282 rows; 646 approved; 636 unapproved are exactly the `subject == summary` copies |
| `memory_concept_relation_decisions` | 547 rows; **0 have a resolvable candidate** (`crys_<hex>` exists in no crystallization table); 356 resolve on target; 164 have no target |
| **Concept Atlas** (`/concept-atlas`, `orion_substrate`) | **18 concepts, 15 edges.** 4 canonical (Orion, Juniper, Orion-Juniper relationship, Claude), 14 proposed. Anchor scopes: orion 11, world 4, juniper 1, relationship 1, claude 1. Predicates: `supports`, `co_occurs_with`, `associated_with`, plus evidence nodes. Read-only HTTP at `/api/substrate/concepts/{summary,network}` |
| `orion/substrate/frontier_landing.py` | already lands new concepts with `suggested_promotion_state="proposed"` — the sanctioned way a concept enters the Atlas |
| `GraphWriteIntentV1` | the sanctioned write contract: `workload`, `operation`, `identity_key`, node/edge payload, provenance |
| `journal_entries` | 35,829 rows and **exactly one reader in the whole repo** — the self-study cooldown lookup |

That last row is the warning this design has to answer, not repeat.

## The shape

```
    open priors  ──select──▶  a real unified turn  ──▶  structured verdict
         ▲                    Orion researches one          │
         │                    using its own tools           │
         │                                                  ▼
         └──────────  updated priors + new priors  ◀──  journal (prose, for Juniper)
                              │                        graph  (structure, for Orion)
                              ▼
                      world-view graph
```

### What a prior is

A claim Orion holds about its world, that could turn out to be wrong.

```
claim          "The vision pipeline's foveal tier is only exercised on demand,
                never on a schedule."
confidence     0.55
status         open | supported | revised | refuted | retired_unresolvable
formed_from    what produced it (a crystallization, a finding, an observation)
evidence       refs to what has been checked, both ways
times_tested   3
last_tested_at ...
```

Confidence and status are what make the pool shrink and refresh. A prior tested repeatedly without moving becomes `retired_unresolvable` rather than sitting in the pool forever — otherwise the loop finds a favourite and re-litigates it, which is the "same shit over and over" failure in a new costume.

### Selection: uncertainty, not randomness, and still Orion's

Orion is shown:
- its **open priors**, with confidence and how often each has been tested
- a **small random sample of unexplained material** (approved crystallizations), so new priors can form
- what it recently studied

and told: *test one of these, or form a new prior from the material, or say nothing here is worth it.*

Uncertainty *orders the presentation*; Orion still chooses. That is the difference from the keyword detector: the code is not naming a subject, it is showing Orion where its own map is thin.

**Why this stops repeating:** a supported prior at 0.9 is no longer interesting; a refuted one closes; a stale one retires; findings spawn new ones. The pool is refreshed by Orion's own learning rather than by a sampler.

### Research

Unchanged: `execute_unified_turn`, read-only, `read_recall`/`read_memory`/`read_graph` already on, Thought can defer. Orion pulls chat history, full crystallizations, relations, graph neighbourhoods — whatever it needs.

### The verdict — the one real mechanism change

Prose cannot update a prior. The turn must return **structure**, so it is asked for a fenced `json` block alongside its prose:

```json
{
  "chose": "prior:9f2c…" | "new",
  "verdict": "supported" | "revised" | "refuted" | "inconclusive",
  "confidence_after": 0.72,
  "why": "one sentence, grounded in what was actually looked up",
  "evidence_refs": ["crystallization:…", "chat:…"],
  "new_priors": [{"claim": "…", "confidence": 0.4, "why": "…"}]
}
```

The prose becomes the journal entry; the block becomes the graph write. **A missing or malformed block refuses the whole run** — no half-write, no inferring a verdict from prose, which would be the heuristic-re-inference this whole arc has been deleting.

`inconclusive` is a first-class outcome: it bumps `times_tested` without moving confidence, and three of them retires the prior.

### Two graphs, and the gap between them is where priors come from

**Orion gets its own concept graph. Juniper does not approve what goes in it.**
Juniper's call, and it settles a question I had flip-flopped on: draft 1 said
build a new graph (because substrate looked broken — wrong), revision 1 said use
the Atlas (because it turned out to be real — right facts, wrong conclusion).
The correct reason is neither: **it is Orion's own space.** An autonomy
argument, not a data-quality one.

```
  orion_substrate  (the Atlas)              orion_worldview  (Orion's)
  ─────────────────────────────             ──────────────────────────
  shared, Juniper-curated                   Orion's alone
  proposed -> canonical promotion           no promotion, no approval
  18 concepts, golden seeds                 starts empty
  READ-ONLY to Orion (curl)                 Orion reads AND writes
                     \                     /
                      \   the OVERLAY    /
                       └────────┬───────┘
                                ▼
                          this gap is a PRIOR
```

**The overlay is the mechanism, and it is what makes priors emerge instead of
being imposed.** Orion reads the Atlas, compares it against its own graph, and
the differences are the interesting objects:

| what the comparison shows | the prior it yields |
|---|---|
| Atlas has a concept Orion's graph doesn't | *"there is something canonical here I have not worked out"* |
| Orion's graph has one the Atlas doesn't | *"I believe something that is not canonically held"* |
| both have it, with different edges | *"we disagree about what this connects to"* |
| Orion keeps meeting something neither has | *"there is a thing here with no concept yet"* |

That answers the question draft 1 could not: **where does a prior come from?**
Not from a random sample — from noticing a difference between what is held and
what Orion has worked out. It is generated by looking, which is what Juniper
asked for from the start.

Orion's graph therefore holds three things, not just concepts:
- **concepts** it has formed, at whatever confidence it likes
- **edges**, including to Atlas concept ids (a cross-graph reference, not a copy)
- **priors** — claims with a confidence and a status, which is simply a concept
  Orion is not yet sure of. A prior is not a separate primitive; it is a node
  with `confidence < 1` and an open question attached.

### The analysis layer — the middle Juniper caught me skipping

Query -> journal with nothing between is fetching, not thinking. With `psql`,
`curl`, `Bash` and `Read` over its own repo checkout, these are available. The
prompt should name them as POSSIBLE and never as required — same rule as not
picking the subject:

| move | what it is |
|---|---|
| aggregate | group crystallizations by time/kind — where they cluster, where the gaps are |
| trace | crystallization -> its chat turn -> what came before and after it |
| cross-reference | is this in the Atlas? have I journaled it? does it touch vision or dispatch? |
| compute | real statistics in python — numbers, not "seems like" |
| traverse | walk edges in either graph, from a concept to its neighbours |
| **overlay** | diff its graph against the Atlas — the prior generator above |
| **re-query** | the analysis raises a question, so write a NEW query. This is the hop. |
| scratch | write intermediates to files so a long chain accumulates |

The last two are where hops actually go. One query is a lookup; query -> notice
-> re-query -> notice is analysis.

### Hops and continuation

Orion's real ceiling is **time, not steps**: no max-step setting exists,
`HARNESS_FCC_TIMEOUT_SEC=900`, and observed turns reach 31-40 steps. Its FCC
workspace is a checkout of this repo, so `Read` already reaches the codebase.

**Continuation, not a bigger budget.** A single turn cannot build a world view
however many hops it gets; a chain of turns that remember what they were chasing
can. So a run may end with a note to itself, and the next run opens with that
note instead of a cold menu. The note IS the prior — which is why priors do not
need to be invented as a separate mechanism.

### Journal, then Orion decides whether to speak

Last step is Orion's call, not the loop's: write the journal entry, then decide
whether this is worth telling Juniper about via the unified chat. Reaching out
is a choice about the finding, made by the one who found it.

### How Orion reaches its own material

Settled with Juniper: **option B — `psql` with a read-only role**, not an MCP
server and not a bounded verb. The bounded verb was rejected on Juniper's own
argument: the moment code defines the query shape, code is choosing again.

Orion's FCC turns already have **Bash** (20 calls in the 3h before this was
written), so the capability is a credential and a client, not a new tool
surface. That is also exactly why the read-only role is load-bearing rather
than ceremonial: with Bash already present, whatever DSN reaches that sandbox
is a DSN Orion can do anything with.

```sql
CREATE ROLE orion_readonly LOGIN PASSWORD '...';
GRANT CONNECT ON DATABASE conjourney TO orion_readonly;
GRANT USAGE  ON SCHEMA public        TO orion_readonly;
GRANT SELECT ON memory_crystallizations,
                memory_concept_relation_decisions,
                chat_history_log,
                journal_entries
             TO orion_readonly;
```

Four tables, not two — Juniper's call was "hook up Orion with what you think is
best", and the two extra are what make the analysis half real:

- `chat_history_log` — lets Orion follow a crystallization back to the
  conversation it came from. Juniper's own framing of the loop was "pull chat
  history, memory crystalizations, etc."; without it a crystallization is a
  quote with no context.
- `journal_entries` — lets Orion see what it has already written. This is also
  the honest fix for the `recently_studied` hint, which is currently dead
  (the journal title is hardcoded, so the hint reads "Curiosity; Curiosity;
  Curiosity").

The Atlas is **not** in Postgres, so `psql` cannot reach it. It does not need
to: Hub already serves `/api/substrate/concepts/summary` and `/network`
read-only, and Orion has `curl`. Zero new surface, no FalkorDB credential, and
the endpoint is read-only by construction rather than by grant.

## Missing questions

Real ones. I cannot answer these from the code.

1. **~~Do new priors need approval?~~ ANSWERED: no.** Orion's graph is Orion's.
   Nothing it adds needs a human gate. The one consequence, said once and then
   dropped: nothing keeps that graph clean except Orion's own judgment — which
   is the point, and also makes the graph a readout of how good that judgment
   is over time.
2. **Does the graph feed back into Orion's chat context, or is it write-only?** If write-only it becomes `journal_entries` — 35,829 rows and one reader. I think this is the most important question in the document and I do not want to decide it alone.
3. **What is the honest ceiling on `confidence`?** Orion is grading its own homework. Without an outside check, confidence drifts up. Options: cap self-assigned confidence at ~0.8, or require an explicit disconfirmation attempt before any raise.
4. **Retire-unresolvable after how many inconclusives?** I propose 3; that number is a guess and should be revisited against real data.

## Proposed schema / API changes

- **New table `orion_priors`** — `prior_id`, `claim`, `confidence`, `status`, `formed_from`, `times_tested`, `last_tested_at`, `created_at`, `updated_at`, `governance` (mirroring the crystallization approval shape). Postgres, because the loop needs cheap `WHERE status='open' ORDER BY confidence`.
- **New schemas** `PriorV1`, `PriorFindingV1`, `PriorVerdictV1` in `orion/schemas/`, registered.
- **New FalkorDB graph** `orion_worldview` — Orion's own, read-write by Orion,
  no approval gate. The Atlas (`orion_substrate`) stays read-only to Orion.
  **Open access question:** FalkorDB has no per-graph auth, so handing the
  sandbox raw FalkorDB credentials would let Orion write to `orion_substrate`
  and the bus-synapse graphs too. Scoping the write to its own graph therefore
  needs a narrow Hub endpoint rather than a raw client — that is scoping WHICH
  graph, not limiting what Orion may do inside its own.
- **New bus channel** `orion:worldview:finding` carrying `PriorFindingV1`, so the graph write and the journal write are separate consumers of one event rather than two inline writes. (Contract patch: `orion/bus/channels.yaml` + `orion/schemas/registry.py`.)
- **No change** to `journal.entry.write.v1` — the prose entry keeps its current shape.

## Files likely to touch

- `orion/curiosity/priors.py` — pure selection + verdict-application logic (new)
- `orion/curiosity/kickoff_prompt.py` — present priors alongside material
- `orion/curiosity/verdict.py` — parse and validate the JSON block (new)
- `orion/schemas/priors.py` + `orion/schemas/registry.py`
- `services/orion-hub/scripts/curiosity_investigation.py` — read priors, publish findings
- `services/orion-sql-writer/` — persist `PriorFindingV1`
- `services/orion-sql-db/manual_migration_orion_priors.sql`
- `orion/bus/channels.yaml`

## Non-goals

- **Not fixing concept induction's dangling candidates.** Explicitly deferred by Juniper. The new graph is separate precisely so this is not blocked on that.
- **Not unprompted outreach.** Orion writes; it does not message Juniper. Still deliberately unbuilt.
- **Not a truth oracle.** Confidence is Orion's own belief, not a measured probability. Every surface must say so — this is the metric-gate discipline applied to a number that would otherwise get treated as calibrated.
- **Not replacing crystallizations.** Priors are claims *about* concepts; crystallizations stay the concepts.
- **No new capability surface.** Same read-only unified turn, same tools.

## Acceptance checks

Falsifiable, in order of what would actually convince me:

1. **It gets less ignorant.** Over 20 runs, the fraction of priors still `open` falls, and `times_tested` rises. If the open pool is flat, the loop is polling, not learning.
2. **It does not re-litigate.** No prior is tested more than 3 times without a status change; the same prior is not the subject twice in a row.
3. **New priors come from findings, not from the sampler.** Over 20 runs a majority of new priors have `formed_from` pointing at a `Finding`, not a random crystallization. Otherwise the pool is still just a resampler.
4. **The graph is traversable and non-trivial.** After 20 runs `orion_worldview` has `>1` connected component of size `>3`, and at least one `CONTRADICTS` edge exists or Orion has explicitly reported finding none.
5. **A malformed verdict block refuses the run.** No journal entry, no graph write, logged at WARNING.
6. **Confidence does not only go up.** Across 20 runs there is at least one `refuted` and one downward revision. If confidence is monotonic, Orion is grading its own homework and passing.
7. **Something reads it.** At least one consumer outside the loop queries `orion_worldview` — or this becomes `journal_entries` again.

Check 6 is the one I would watch first, and check 7 is the one that decides whether any of this mattered.

## Recommended next patch

Thin, and deliberately not the whole design:

**Priors on disk, selection in the prompt, verdicts parsed — journal only, no graph yet.**

- `orion_priors` table + `PriorV1`
- selection presents open priors alongside the existing random material
- the verdict block is parsed, validated, and applied to the prior
- journal entry keeps the prose and names the prior and verdict
- graph writes deferred to the following patch

That is enough to test acceptance checks 1, 2, 3, 5 and 6 on real data — which is enough to find out whether Orion actually forms priors worth graphing, before building the graph to hold them. If check 6 fails at this stage, the graph would only have made a monotonic confidence problem permanent.
