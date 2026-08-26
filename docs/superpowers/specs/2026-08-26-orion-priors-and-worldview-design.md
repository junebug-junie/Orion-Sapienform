# Priors, research, and a world-view graph

Design mode. Nothing here is built yet.

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
| FalkorDB | `orion_substrate` (18 Concept nodes), `graphiti_temporal`, `orion_recall`, others |
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

### The world-view graph

**New FalkorDB graph `orion_worldview`, not an extension of `orion_substrate`.** Recommendation, and the reason is measured: `orion_substrate` has 18 nodes and is fed by an induction lane whose 547 edges have **zero** resolvable source nodes. Building on it inherits that breakage and makes both harder to fix. A clean graph with its own write contract can be correct from day one, and the two can be reconciled once induction is repaired — which Juniper has already scheduled as "later."

```
(:Prior   {prior_id, claim, confidence, status, times_tested, formed_at})
(:Concept {crystallization_id, kind, subject})
(:Finding {finding_id, verdict, why, observed_at, correlation_id})

(:Prior)-[:ABOUT]->(:Concept)
(:Prior)-[:TESTED_BY]->(:Finding)
(:Finding)-[:SUPPORTS|:REFUTES|:REVISES]->(:Prior)
(:Prior)-[:SPAWNED]->(:Prior)
(:Prior)-[:CONTRADICTS]->(:Prior)
```

Written through `GraphWriteIntentV1` (`workload="orion_worldview"`), never raw Cypher — that contract already carries provenance and identity keys, which is what makes a write auditable.

`CONTRADICTS` is the payoff: two priors that cannot both hold is a real thing for Orion to notice and chase, and it can only be seen in a graph.

## Missing questions

Real ones. I cannot answer these from the code.

1. **Do new priors need your approval before entering the graph?** The crystallization approval filter turned out to be load-bearing — the unapproved 636 were exactly the junk. A prior is a stronger claim than a crystallization, and an unapproved-prior pool could rot the same way. But approval is friction, and an autonomy feature gated on a human is less autonomous. My recommendation: **auto-admit priors, require approval only to mark one `supported` above 0.8** — cheap to hold a hypothesis, expensive to call it settled.
2. **Does the graph feed back into Orion's chat context, or is it write-only?** If write-only it becomes `journal_entries` — 35,829 rows and one reader. I think this is the most important question in the document and I do not want to decide it alone.
3. **What is the honest ceiling on `confidence`?** Orion is grading its own homework. Without an outside check, confidence drifts up. Options: cap self-assigned confidence at ~0.8, or require an explicit disconfirmation attempt before any raise.
4. **Retire-unresolvable after how many inconclusives?** I propose 3; that number is a guess and should be revisited against real data.

## Proposed schema / API changes

- **New table `orion_priors`** — `prior_id`, `claim`, `confidence`, `status`, `formed_from`, `times_tested`, `last_tested_at`, `created_at`, `updated_at`, `governance` (mirroring the crystallization approval shape). Postgres, because the loop needs cheap `WHERE status='open' ORDER BY confidence`.
- **New schemas** `PriorV1`, `PriorFindingV1`, `PriorVerdictV1` in `orion/schemas/`, registered.
- **New FalkorDB graph** `orion_worldview`, written via existing `GraphWriteIntentV1`.
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
