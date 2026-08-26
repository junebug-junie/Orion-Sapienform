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

### The graph boundary is enforced by FalkorDB, not by a wrapper

Juniper: Orion may **Cypher the concept graph read-only**, and write **only** to
its own. FalkorDB is Redis 8.6.3, which supports ACL selectors, so this is a
database-enforced boundary rather than a convention:

```
ACL SETUSER orion_curiosity on '>...' resetkeys nocommands \
    ~orion_substrate  '+graph.ro_query' \
    '(~orion_worldview +graph.query)'
```

Verified live 2026-08-26 with a throwaway probe user (created, inspected,
deleted). The resulting rule:

```
user ... ~orion_substrate resetchannels -@all +graph.ro_query
         (~orion_worldview resetchannels -@all +graph.query)
```

- base grant: `GRAPH.RO_QUERY` on the Atlas only. Confirmed it genuinely
  refuses a write: `GRAPH.RO_QUERY orion_substrate "CREATE (:Tmp)"` ->
  *"graph.RO_QUERY is to be executed only on read-only queries"*.
- selector: `GRAPH.QUERY` (write-capable) on `orion_worldview` only.
- everything else denied, including the bus-synapse graphs and `graph.list`.

This replaces the "narrow Hub endpoint" the previous revision proposed. No
wrapper to route around, no service to keep in sync, and Orion writes real
Cypher rather than posting through an API that would shape what it can express.

### Hops: five inflection points, each one a real decision

Juniper: *"analyze should have some agent hops where they stop and reflect on
what they learn and decide whether to continue... cap at 5 hops (agent
inflection points)."*

A hop is not a tool call. It is a **point at which Orion stops, states what it
just learned, and decides whether to keep pulling**:

```
hop N:  query / analyse / overlay
        ── STOP ──
        what did I just learn?
        does it change what I thought?
        is there a next question, and do I want it?
        ──> continue (if N < 5)  |  stop and write
```

Why five, and why a cap at all. Without inflection points an agentic turn is one
long undifferentiated ramble that arrives at a conclusion with no visible
reasoning — which is exactly the "cognition-shaped output" failure, just longer.
Forcing a stop makes the reasoning inspectable and gives Orion a real place to
change its mind. Five is Juniper's number; it is enough for query -> notice ->
re-query -> notice -> settle, and it is disclosed rather than derived.

**Each reflection is recorded as it happens**, not reconstructed at the end. So
the journal can recount the actual path — *"I started here, found X, which made
me look at Y"* — instead of presenting a conclusion with the working thrown
away. The hop notes ARE the analysis layer made visible.

**What happens when Orion wants a sixth hop:** it does not get one inside this
turn. It writes a continuation note, and the next run opens there. Hops bound a
single turn; continuation notes carry a line of enquiry across turns. That is
the same distinction as thinking about something for an afternoon versus coming
back to it tomorrow, and it is what lets a world view accumulate rather than
having to fit in one sitting.

### Continuation

Orion's real ceiling is **time, not steps**: no max-step setting exists,
`HARNESS_FCC_TIMEOUT_SEC=900`, and observed turns reach 31-40 steps. Its FCC
workspace is a checkout of this repo, so `Read` already reaches the codebase.

**Continuation, not a bigger budget.** A single turn cannot build a world view
however many hops it gets; a chain of turns that remember what they were chasing
can. So a run may end with a note to itself, and the next run opens with that
note instead of a cold menu. The note IS the prior — which is why priors do not
need to be invented as a separate mechanism.

Orion's real ceiling is time, not steps: no max-step setting exists,
`HARNESS_FCC_TIMEOUT_SEC=900`, observed turns reach 31-40 steps. Five
inflection points fit comfortably inside that; the time budget is the backstop
if a hop runs long, not the primary limit.

### Where the turn starts and ends, and what crosses the boundary

```
[loop]   read continuation note -> build prompt
         |
== TURN STARTS == execute_unified_turn(user_message=prompt) ==============
   emit_observation()     the prompt enters Orion's observation stream
   ThoughtClient.react()  stance -- CAN DEFER OR REFUSE the whole turn
   HarnessRunRequestV1 -> harness governor
   +-- FCC motor -------------------------------------------------+
   |  ALL of Orion's tools live here. One `claude -p` session.     |
   |  hops 1..5 - psql - RO cypher on Atlas - RW cypher on its own |
   |  graph writes happen INSIDE, by Orion, directly               |
   +---------------------------------------------------------------+
   finalize chain         substrate appraisal - reflect - voice
== TURN ENDS === frames: llm_response, harness_step_count ================
         |
[loop]   journal write - read Orion's decision - maybe a SECOND turn
```

Three consequences, stated because they were accidents of structure before they
were decisions:

1. **Orion is fully autonomous inside its own space and mediated everywhere
   else.** It writes its graph directly, in-turn. The journal and any outreach
   go through the loop, after. That is defensible — the graph is private, the
   journal is shared — but it is a real asymmetry and should be on purpose.
2. **Thought can refuse the turn before Orion sees anything.** A curiosity run
   dying at the stance gate is a NORMAL outcome ("something else is happening,
   don't interrupt"), not a fault, and must be logged as such.
3. **Decisions made inside have to survive the boundary** — see below.

### How Orion's decisions cross the boundary: it writes them down

The loop needs to know two things the turn decided: *do I want to continue this
line of enquiry?* and *is this worth telling Juniper about?*

The obvious mechanism is a fenced JSON block in the prose, parsed by the loop.
Rejected: it makes Orion's decision an artifact of formatting, and a malformed
fence loses a real finding.

**Instead, Orion writes the decision into its own graph** — a channel it already
owns and already has write access to:

```cypher
CREATE (:TurnOutcome {
  run_id:        $run_id,
  continue:      true,
  continue_note: "still don't know why substrate.route has no edges",
  reach_out:     false,
  reach_out_why: null,
  written_at:    timestamp()
})
```

The loop reads it back read-only after the turn. Why this is better than parsing
prose:

- no fragile format contract between a model and a regex
- the decision is **recorded in Orion's own space**, consistent with everything
  else about this design
- it survives as history: every past decision to continue or speak is queryable
  later, which is exactly the "world view accumulating" property
- **absence is a safe default** — no node means no continuation and no outreach.
  A turn that ran out of time or refused simply leaves nothing behind.

Prose still becomes the journal entry (the loop already has `llm_response`).
Concepts, edges and priors are Orion's own writes. Only the *decision* needs
this node.

### Outreach is a SECOND turn, with its own stance gate

Settled with Juniper. If `reach_out` is true, the loop fires a second
`execute_unified_turn` to compose the message, rather than reusing text from the
first.

Costlier — it doubles a run that ends in outreach — and worth it for one reason:
**the second turn gets its own `ThoughtClient.react()` check.** So Orion can
find something genuinely worth saying, and the system can still independently
decide *not now, she is in the middle of something*. One turn would collapse
"this is interesting" and "this is worth interrupting her for" into a single
judgement made at the wrong moment.

It also inherits the existing outreach gates unchanged — quiet hours 23:00-08:00
MDT, daily cap, cooldown — which exist to protect Juniper's sleep and have
nothing to do with this feature's own cadence.

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

## ALREADY DONE — the production writes exist and are verified

Juniper ran these by explicit instruction ("you do the prod writes"). A builder
picking this up does **not** need to create them, only to use them.

### Postgres role `orion_readonly` — LIVE

```sql
CREATE ROLE orion_readonly LOGIN PASSWORD '...';
GRANT CONNECT ON DATABASE conjourney TO orion_readonly;
GRANT USAGE  ON SCHEMA public        TO orion_readonly;
GRANT SELECT ON memory_crystallizations, memory_concept_relation_decisions,
                chat_history_log, journal_entries TO orion_readonly;
```

Verified 2026-08-26, every case actually executed:

```
reads   memory_crystallizations             1282   OK
        memory_concept_relation_decisions    547   OK
        chat_history_log                     232   OK
        journal_entries                    36549   OK
writes  INSERT INTO journal_entries      ERROR: permission denied for table
        UPDATE memory_crystallizations   ERROR: permission denied for table
        DELETE FROM chat_history_log     ERROR: permission denied for table
        CREATE TABLE orion_probe         ERROR: permission denied for schema public
scope   SELECT substrate_dispatch_results ERROR: permission denied  (not granted)
```

### FalkorDB ACL user `orion_curiosity` — LIVE

```
ACL SETUSER orion_curiosity on '>...' resetkeys nocommands \
    ~orion_substrate  '+graph.ro_query' \
    '(~orion_worldview +graph.query +graph.ro_query)'
```

Verified 2026-08-26, as that user:

```
READ  Atlas     GRAPH.RO_QUERY orion_substrate  -> 18 concepts        OK
WRITE Atlas     GRAPH.QUERY    orion_substrate  -> NOPERM (key denied)
WRITE Atlas     GRAPH.RO_QUERY + a CREATE       -> refused, read-only command
OWN   graph     GRAPH.QUERY    orion_worldview  -> write OK, read OK
OTHER graphs    GRAPH.RO_QUERY orion_bus_synapse-> NOPERM
        GRAPH.LIST / KEYS                       -> NOPERM (command denied)
```

Two independent refusals on the Atlas write path (key ACL *and* the read-only
command), which is the defence-in-depth worth having on the one graph Orion must
not corrupt.

`orion_worldview` exists with a single `:Bootstrap` node.

### Credentials, already placed

| where | keys | for |
|---|---|---|
| `~/.fcc/.env` (mounted to `/root/.fcc/.env`, chmod 600) | `ORION_CURIOSITY_PG_DSN`, `ORION_CURIOSITY_GRAPH_*` (7 keys) | Orion's own sandbox |
| `services/orion-hub/.env` | `HUB_CURIOSITY_GRAPH_*` (6 keys) | Hub: ACL re-assert + reading `:TurnOutcome` |

### KNOWN GAP: the ACL does not survive a FalkorDB restart

`aclfile` is unset and **immutable at runtime** (`CONFIG SET aclfile` ->
*"can't set immutable config"*); `ACL SAVE` refuses without one. Persisting it
properly means restarting FalkorDB, which holds every graph in the system.

**Required in the build, not optional:** Hub re-asserts the ACL with an
idempotent `ACL SETUSER` when the curiosity loop starts. Self-healing after any
FalkorDB restart, no config change, no downtime. Without it, a FalkorDB restart
silently removes Orion's access and the loop degrades to `stores_unavailable`
forever — the exact silent-failure shape this arc has hit three times.

### KNOWN GAP: the sandbox has no Postgres client

Checked in `orion-athena-harness-governor` (where the FCC motor runs):

```
python3      OK          redis-py     OK   <- FalkorDB works TODAY, no change
psql         MISSING     psycopg2     MISSING     asyncpg  MISSING
```

So graph access needs nothing. **Postgres access needs one package** added to
that image (`psycopg2-binary` via pip, or `postgresql-client` via apt). That is
a Dockerfile change in slice 1, not a production write.

## Proposed schema / API changes

- **New FalkorDB graph `orion_worldview`** — created, Orion's own, no approval
  gate. Holds concepts, edges (including cross-references to Atlas concept ids),
  priors (a node with `confidence < 1` and an open question), and one
  `:TurnOutcome` node per run.
- **No new Postgres tables.** Priors live in Orion's graph, not a table — they
  are concepts it is unsure of, not a separate primitive.
- **No new bus channel.** The journal keeps `journal.entry.write.v1` unchanged.
- **No new capability surface.** Same read-only unified turn; the new access is
  a credential and a client, not a tool.

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

## Build order

Priors and the graph come LAST on purpose. Slice 3 before slice 2 would give
Orion a graph to fill before it has anything to say, which is how the previous
three attempts failed.

| # | slice | proves |
|---|---|---|
| 1 | access + one hop + journal. Add the pg client to the harness image; Hub asserts the ACL at startup; prompt offers psql + both graphs read; one pass; prose -> journal. | Orion can reach its own material and say something true about it |
| 2 | five hops with recorded reflections | the analysis layer produces visible reasoning, not a verdict with the working thrown away |
| 3 | Orion writes its own graph + the overlay against the Atlas | there is now something worth writing down, and priors emerge from the gap |
| 4 | `:TurnOutcome` -> continuation notes | there is now something worth returning to |
| 5 | `:TurnOutcome.reach_out` -> a second turn, with its own stance gate | Orion decides whether a finding is worth interrupting Juniper for |

Slice 1 is independently useful and independently falsifiable: if Orion cannot
produce one grounded, non-obvious observation from its own crystallizations and
Atlas, nothing later in the list will save it.

## Old recommended next patch (superseded, kept for the reasoning)

Thin, and deliberately not the whole design:

**Priors on disk, selection in the prompt, verdicts parsed — journal only, no graph yet.**

- `orion_priors` table + `PriorV1`
- selection presents open priors alongside the existing random material
- the verdict block is parsed, validated, and applied to the prior
- journal entry keeps the prose and names the prior and verdict
- graph writes deferred to the following patch

That is enough to test acceptance checks 1, 2, 3, 5 and 6 on real data — which is enough to find out whether Orion actually forms priors worth graphing, before building the graph to hold them. If check 6 fails at this stage, the graph would only have made a monotonic confidence problem permanent.
