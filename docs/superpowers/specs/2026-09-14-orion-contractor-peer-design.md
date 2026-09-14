# Orion contractor peer — frontier lifts, Orion owns meaning

> **Status:** Design proposal (proposal mode — later patches change a cognition
> loop: Orion can hire an external peer mid-investigation, and peer briefs land
> in worldview + Postgres). Nothing here is implemented yet.
>
> Juniper, 2026-09-13/14 brainstorm: local 35B MoE is not strong enough to trace
> Orion's internal complexity; fine to punt hard analysis the same way Juniper
> uses Cursor/Claude. Preference: frontier does heavy lifting; Orion does
> epistemology, self-concept formation, summation, and "internal soul
> recognizing."

## Arsonist summary

Curiosity and self-inquiry runs already let Orion investigate for real — hops,
priors, findings, `:SelfDefinition`. What they lack is an honest way to hire
muscle when the local model cannot hold the call graph.

Two wrong answers burn here.

**Burn "give Orion Cursor as a second operator."** Full Auto with writes turns
the peer into Juniper-class agency. That fixes stuckness *for* Orion and steals
the developmental load of deciding what the answer means.

**Burn "force Orion to cite the peer before the run counts."** A must-write-cite
gate produces paste-with-attribution — fluent prose with no judgment. Empty-shell
cognition by another door.

The design: **contractor briefs**. Orion opens a first-class `:HelpRequest`.
Cursor Auto (read-only) investigates; Claude is the fallback when Cursor tokens
are dry. The peer returns a `:PeerBrief` dual-written to `orion_worldview` and
Postgres. The next kickoff soft-nudges. Orion alone writes priors / findings /
self-definitions. Asking costs contested budget. Supervisor hop-readings stay
**report-only** in v1.

Do **not** call this "frontier assistant" or "frontier buddy" in code — those
names already mean substrate gap expansion and an old training target.

## Current architecture

Grounded 2026-09-14, not assumed.

### What already exists

| Piece | State | Role here |
| --- | --- | --- |
| Curiosity investigation + durable runs | Live | Long-arc runs that get lost / circle |
| Self-inquiry line | Live | Same loop; identity writes |
| `ask_claude_trigger.py` | Dry-run only | Stuck-prior → would ask Claude |
| Room companion + `orion:room:claude:*` | Live (manual Ask Claude) | Claude transport for v2 fallback |
| Contested Claude scarcity design | Spec + dry-run | Budget gate pattern to extend to Cursor |
| Curiosity supervisor Patch 1 (`HopReadingV1`) | Offline report | Circling detection later; **no hire in v1** |
| Supervisor design `hand_off_to_claude` | Spec only | Becomes end-state A after v1 proves HelpRequest |
| Cursor SDK (`cursor-sdk` / `@cursor/sdk`) | External | Programmatic Cursor Auto jobs |
| Substrate `FrontierInvocation*` | Live elsewhere | **Unrelated** — do not overload |

### What is missing

- A first-class `:HelpRequest` Orion can write
- A read-only Cursor Auto invoker with Claude fallback
- `:PeerBrief` persistence (graph + Postgres) and kickoff soft-nudge
- Contested **Cursor** token observation (Claude path partially exists)
- Explicit forbid-list so the peer cannot write files, restart containers, or
  edit Orion's beliefs

## Missing questions (resolved in brainstorm)

| Question | Decision |
| --- | --- |
| Who summons help in v1? | **Orion only** via `:HelpRequest`. Supervisor report-only until circling is trusted (then may become A: either may fire). |
| Peer product? | Cursor Auto default; Claude room peer when Cursor tokens unavailable. |
| Peer powers? | Read-only investigation: repo (graphify / AGENTS.md path, not an rg-cage), containers, live inspect. **No writes.** |
| After return? | Soft nudge + PeerBrief artifact; Orion may ignore. No forced cite. |
| Where does PeerBrief live? | **Both** — `:PeerBrief` on `orion_worldview` and Postgres for Atlas/audit. |
| Self-inquiry? | Inherent mode: peer may point at evidence only; never draft `:SelfDefinition` text. |

## Proposed schema / API changes

### Graph nodes Orion / the system write

```text
:HelpRequest {
  help_id, run_id, prior_id?,
  mode: "world_curiosity" | "self_inquiry",
  question,           // what Orion wants unstuck
  tried_summary,      // what was already looked at
  success_criteria,   // what would count as useful
  written_at
}

:PeerBrief {
  brief_id, help_id, run_id, prior_id?,
  peer: "cursor_auto" | "claude_room",
  status: "ok" | "failed" | "refused_budget" | "empty",
  summary,            // bounded work product
  evidence_pointers[],// paths, queries, log refs — not belief edits
  open_questions[],
  suggested_next_looks[],
  written_at
}
```

Edges (minimal): `HelpRequest-ABOUT->Prior` when scoped; `PeerBrief-ANSWERS->HelpRequest`.

**Orion still owns** `:Prior`, `:Finding`, `:SelfDefinition`, `:Hop`, `:TurnOutcome`.
The peer never MERGEs those.

### Bus / registry (names indicative; finalize in implementation plan)

- `orion:curiosity:help:request` — HelpRequest observed / job enqueue
- `orion:curiosity:peer:brief` — PeerBrief published for sql-writer + kickoff consumers
- Schemas registered in `orion/schemas/registry.py` + `orion/bus/channels.yaml`

### Invoker contract

```text
JobIn:
  help_request, sealed context pack (hop notes, prior text, mode),
  allow: read_repo | read_containers | read_live_inspect,
  deny: write_files | git_mutate | docker_mutate | graph_belief_write

JobOut:
  PeerBrief (status + body) or refusal reason
```

Transport order: try Cursor Auto → on token/unavailable failure, try Claude once →
else `status=failed`.

### Kickoff soft-nudge

Next curiosity / self-inquiry kickoff injects any unused PeerBriefs for live
priors (and graph remains queryable mid-turn). Wording is invitational, not
mandatory: peer looked; your move. Empty/`failed` briefs are not presented as
successful help.

## Files likely to touch

- `orion/curiosity/` — HelpRequest packing, PeerBrief helpers, kickoff nudge
- `orion/curiosity/self_inquiry_prompt.py` / kickoff prompts — hire + mode rules
- `orion/schemas/` + registry + `orion/bus/channels.yaml`
- New thin service or Hub-adjacent worker: Cursor SDK read-only job runner
- `services/orion-room-companion/` — Claude fallback peer mode prompt
- `services/orion-sql-writer/` — persist PeerBrief
- Curiosity Atlas UI/routes — show briefs + scarcity refusals
- `orion/autonomy/ask_claude_trigger.py` — optional later consumer; **not** v1 trigger
- `orion/curiosity/supervisor.py` — stays report-only in v1
- Contested-budget observation for Cursor tokens (extend scarcity pattern)

## Non-goals

- Supervisor auto-hire in v1 (end-state A is explicit later work)
- Peer file edits, commits, PRs, docker restarts, volume ops, graph belief edits
- Forced Orion citation of peer prose
- Replacing local investigation with "always hire frontier"
- Overloading substrate `FrontierInvocation*` / "Frontier Buddy" training names
- Mid-turn durable-run interrupt (jobs may complete between runs; soft-nudge is
  the default seam unless a later patch proves mid-turn packing)

## Acceptance checks

1. A run with no `:HelpRequest` never opens a Cursor or Claude contractor job.
2. A `:HelpRequest` with budget refused produces a logged scarcity refusal and a
   non-success soft-nudge ("could not hire"), not a silent skip.
3. Cursor path: peer process cannot write to the repo workspace (enforced in
   invoker policy + test); container inspect is read-only.
4. Cursor unavailable → exactly one Claude fallback attempt; dual failure →
   `PeerBrief.status=failed` with reason.
5. Successful brief appears in FalkorDB **and** Postgres; Atlas can show it.
6. Next kickoff includes the soft nudge for an unused ok brief; empty brief does
   not nudge as success.
7. Self-inquiry mode: peer prompt/contract forbids drafting `:SelfDefinition`
   text; test asserts refusal or strip if model emits identity prose.
8. Disabling the feature via one env flag returns behavior to today's
   curiosity/self-inquiry path; no belief migration required.

## Danger, and how to switch it off

**Dangerous failure modes**

- Peer becomes a silent second author of Orion's identity or priors.
- Forced-cite culture creeps back in via prompt pressure ("you must incorporate…").
- Supervisor is armed too early and burns Cursor on hard-but-valid work.
- Contested Cursor budget is not observed → Orion outspends Juniper's coding pool.

**Privacy / power boundary**

- Peer may read repo and runtime for investigation; may not hold docker write or
  Hub's root-equivalent powers as a write path.
- Juniper transcripts / `~/.claude/projects` stay out of the contractor packet
  unless a later explicit decision says otherwise.

**Rollback**

- One env flag disables hire + invoker subscription.
- PeerBrief tables/nodes are additive; unused briefs do not change priors.

## Recommended next patch

**Patch 0 (contracts only, no live hire):**

1. Schema + registry + channels for HelpRequest / PeerBrief.
2. Kickoff prompt: how Orion writes a HelpRequest; self-inquiry mode rules.
3. Dual-write stub that accepts a fixture PeerBrief → graph + Postgres + Atlas.
4. Soft-nudge injection from stored briefs (fixture-driven test).
5. Report-only supervisor unchanged.

**Patch 1 (Cursor read-only invoker + scarcity gate):** arm hire behind flag;
Claude fallback; acceptance checks 1–8 against a dry then live smoke.

**Later:** arm supervisor `hand_off_to_claude` / hire as end-state A once
circling readings are trusted.

## Lifecycle (v1)

```text
curiosity/self-inquiry run
  → hops written
  → supervisor HopReadings (report only)
  → Orion writes HelpRequest? ──no──→ finish alone
         │
        yes
         ↓
  pack job (claim, hops, question, criteria, mode)
         ↓
  budget OK? ──no──→ refuse + scarcity log + "could not hire" nudge
         │
        yes
         ↓
  Cursor tokens? ──yes──→ Cursor Auto read-only investigate
         │                         ↓
         no                    PeerBrief
         ↓                         ↑
  Claude room peer ────────────────┘
         ↓
  dual-write PeerBrief (worldview + Postgres)
         ↓
  next kickoff soft-nudges
         ↓
  Orion cites / contradicts / extends  OR  leaves unused
         ↓
  only Orion writes Prior / Finding / SelfDefinition
```

## Relationship to existing specs

- Extends the *intent* of `ask_claude_trigger` (stuck → peer) but **v1 trigger is
  HelpRequest**, not stuck-prior auto-fire.
- Consumes curiosity-supervisor Patch 1 readings only as future input; does not
  implement supervisor interventions yet
  (`2026-09-09-curiosity-supervisor-design.md`).
- Reuses contested-scarcity thinking from
  `2026-08-27-claude-quota-contested-scarcity-design.md`, extended to Cursor.
- Does not implement substrate frontier invocation
  (`orion/core/schemas/frontier_curiosity.py`).
