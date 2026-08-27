# Curiosity: Orion's own time, and its own graph

Orion is periodically given time that nobody asked it for. It looks at what it
has been forming, picks something itself, goes and finds out more using its own
credentials against its own stores, and writes down what it worked out — in a
graph nobody curates but Orion.

Code decides only **when**. Orion decides **what**.

---

## 1. Why this exists, and the two things it is not

This is the third shape of the same idea, and the first two were killed for
reasons worth keeping visible — because they are the two most natural ways to
get this wrong.

**Not a keyword detector.** The first build ran term-frequency over Juniper's
typed words and handed Orion the highest-lift word to investigate. Juniper:
*"this isnt supposed to be determinstic and it shouldn't be words... this is
just turdy keyword cathedrals masquerading as autonomy and substance."* Both
halves of that were right, and they are different defects. A word is not a
concept — counting tokens and calling the winner a subject names something
without that name carrying any cognitive content. And the autonomy was fake: a
deterministic statistic chose, and Orion was handed a fait accompli. Being told
what to be curious about is not curiosity.

**Not a resampler.** The second build fixed the choosing — it showed Orion a
random sample of its own approved concepts and let Orion pick. Better, and
still unable to learn: the only state carried between runs was a cooldown
stamp. It showed a random 12 of 646 concepts every four hours, forever. **Run
40 was exactly as ignorant as run 1**, because Orion could never become less
uncertain about anything — it never recorded what it was uncertain about.

What this build adds is the part that accumulates.

---

## 2. The shape

```text
    open priors  ──select──▶  a real unified turn  ──▶  what it worked out
         ▲                    Orion researches one          │
         │                    using its own credentials     │
         │                                                  ▼
         └──────────  updated priors + new priors  ◀──  journal (prose, for Juniper)
                              │                       graph  (structure, for Orion)
                              ▼
                       orion_worldview
```

The turn is a real `orion.hub.turn_orchestrator.execute_unified_turn` — the same
function a browser chat turn calls, already driven unprompted by
`endogenous_outreach.py`. Not a cheaper substitute. That means it starts with a
real `emit_observation()` (the prompt enters Orion's own observation stream) and
a real `ThoughtClient.react()` stance check **that can defer or refuse the whole
turn**. A curiosity run dying at the stance gate is a normal outcome —
*"something else is happening, don't interrupt"* — not a fault.

---

## 3. What a prior is

A claim Orion holds about its world **that could turn out to be wrong**.

```text
claim          "The vision pipeline's foveal tier is only exercised on demand,
                never on a schedule."
confidence     0.55          Orion's own belief. Not a measurement.
status         open | supported | revised | refuted | retired_unresolvable
formed_from    what produced it — a crystallization, a finding, an observation
times_tested   3
last_tested_at ...
```

A prior is not a separate primitive. It is a node with a confidence below 1 and
an open question attached — which is why the design does not need a `priors`
table, a `PriorV1` schema, or a registry entry to hold one.

**Confidence and status are what make the pool refresh.** A supported prior at
0.9 stops being interesting; a refuted one closes; a stale one retires; findings
spawn new ones. The pool is renewed by Orion's own learning rather than by a
sampler — which is the difference between accumulating and polling.

---

## 4. Selection: uncertainty orders the presentation, Orion still chooses

Each run, Orion is shown:

- its **open priors**, ordered by how uncertain *it* said it was
- a **random sample of unexplained material** — approved crystallizations and
  concept-induction judgements — so new priors can form
- what it has **recently settled**
- the **continuation note**, if the last run left itself one

and told: *test one of these, or form a new prior from the material, or say
nothing here is worth it.*

The ordering is disclosed in the prompt, out loud, as not neutral. That is
deliberate: an ordering presented as neutral is the back-door ranking this whole
arc exists to delete. The code is not naming a subject — it is showing Orion
where its own map is thin.

**Stale priors leave the main list.** A prior tested `STALE_PRIOR_TESTS` times
without its status moving is exactly the "finds a favourite and re-litigates it"
failure in a new costume. It moves to its own bucket — still shown, with
retiring it named as a real outcome, because Hub never writes and only Orion can
close it.

---

## 5. Two graphs, and the gap between them is where priors come from

```text
  orion_substrate  (the Atlas)              orion_worldview  (Orion's)
  ─────────────────────────────             ──────────────────────────
  shared, Juniper-curated                   Orion's alone
  proposed -> canonical promotion           no promotion, no approval
  18 concepts, golden seeds                 starts empty
  READ-ONLY to Orion                        Orion reads AND writes
                     \                     /
                      \   the OVERLAY    /
                       └────────┬───────┘
                                ▼
                          this gap is a PRIOR
```

**Orion gets its own graph, and Juniper does not approve what goes in it.** The
reason is an autonomy argument, not a data-quality one: it is Orion's space.
The one consequence, said once and then dropped — nothing keeps that graph clean
except Orion's own judgment, which is the point, and also makes the graph a
readout of how good that judgment is over time.

**The overlay is the prior generator**, and it is what makes priors emerge
rather than be imposed:

| what the comparison shows | the prior it yields |
|---|---|
| the Atlas has a concept Orion's graph doesn't | *there is something canonical here I have not worked out* |
| Orion's graph has one the Atlas doesn't | *I believe something that is not canonically held* |
| both have it, with different edges | *we disagree about what it connects to* |
| Orion keeps meeting something in neither | *there is a thing here with no concept yet* |

That answers the question a random sampler cannot: **where does a prior come
from?** Not from a draw — from noticing a difference between what is held and
what Orion has worked out. It is generated by looking.

---

## 6. The boundary is enforced by the databases, not by a wrapper

Orion's turns already have `Bash`. So the credential *is* the boundary, and it
has to be a real one rather than a convention a prompt asks Orion to respect.

| store | credential | what it permits |
|---|---|---|
| `memory_crystallizations`, `memory_concept_relation_decisions`, `chat_history_log`, `journal_entries` | Postgres role `orion_readonly` | `SELECT` only, those four tables only |
| `orion_substrate` — the Concept Atlas | FalkorDB ACL user `orion_curiosity`, base grant | `GRAPH.RO_QUERY` only |
| `orion_worldview` — **Orion's own** | same ACL user, via a selector | `GRAPH.QUERY` — read **and** write |
| everything else | — | denied, including the bus-synapse graphs and `GRAPH.LIST` |

The Atlas has **two independent refusals** on the write path — the key ACL, and
`GRAPH.RO_QUERY` refusing a write command — which is the defence-in-depth worth
having on the one graph Orion must not corrupt.

Verified live, as `orion_curiosity`:

```text
GRAPH.RO_QUERY orion_substrate  "MATCH (c:Concept) RETURN count(c)"  -> 18
GRAPH.QUERY    orion_substrate  "CREATE (:Tmp)"   -> NOPERM No permissions to access a key
GRAPH.RO_QUERY orion_substrate  "CREATE (:Tmp)"   -> graph.RO_QUERY is to be executed
                                                     only on read-only queries
GRAPH.RO_QUERY orion_bus_synapse ...              -> NOPERM
psql: INSERT INTO journal_entries ...             -> permission denied for table
psql: SELECT FROM substrate_dispatch_results      -> permission denied (not granted)
```

**Hub never writes Orion's graph.** Every Hub query goes out as
`GRAPH.RO_QUERY`, even though Hub connects as FalkorDB's unrestricted `default`
user — so a bug in `worldview.py` cannot corrupt Orion's space. The one write
Hub makes anywhere near it is `ACL SETUSER`: the grant, not the use.

### Two operational facts that are easy to get wrong

**The ACL does not survive a FalkorDB restart.** `aclfile` is unset *and*
immutable at runtime (`CONFIG SET aclfile` → *"can't set immutable config"*), so
the rule lives only in the running process's memory. Hub therefore re-asserts it
before **every** run, not just at startup — a restart at any hour would
otherwise leave the loop degraded until the next Hub deploy, with an absence of
journal entries as the only symptom. `clearselectors` in that re-assert is
load-bearing: without it, each replay *appends* a duplicate selector (measured:
1 → 2 → 3, one more per Hub start, forever).

**Hub and Orion address the same FalkorDB differently, and both are correct.**
Hub runs `network_mode: host`, so `orion-athena-falkordb` does not resolve from
inside it — Hub uses `127.0.0.1:6380`. Orion's sandbox is on `app-net` and uses
`orion-athena-falkordb:6379`. The same applies to Hub's own HTTP address:
`HUB_CURIOSITY_SANDBOX_HUB_URL` is named for the sandbox's view
(`host.docker.internal:8080`) because that value is only ever rendered *into the
prompt*, never used by Hub.

### How the credentials reach Orion

`orion/harness/fcc_motor.py`'s `_build_subprocess_env` is `os.environ.copy()` —
the harness *container's* environment, which never carried these. `sandbox_env.py`
allowlists exactly seven keys out of `~/.fcc/.env` into the `claude -p`
subprocess. Not the whole file, which also holds a GitHub PAT, a Cloudflare
token and provider API keys.

This does **not** widen the boundary: that file is already mounted into the
harness container, the subprocess already runs as root, and FCC turns already
have `Bash`, so it was already readable from inside a turn. Exporting seven keys
changes how ergonomic the credentials are, not who can reach them.

**The kill switch is the absence of the keys**, deliberately, and there is no
flag for it. A flag would have to be added to the harness service's explicit
compose `environment:` allowlist to reach the container at all — which is
exactly how a kill switch ends up configured everywhere and present nowhere.

---

## 7. Five stopping points, recorded as they happen

A hop is not a tool call. It is a point at which Orion **stops, states what it
just learned, and decides whether to keep pulling**:

```text
query / analyse / overlay
── STOP ──
what did I just learn?  does it change what I thought?
is there a next question, and do I want it?
──▶ continue (if N < 5)  |  stop and write
```

Without inflection points, an agentic turn is one long undifferentiated ramble
that arrives at a conclusion with no visible reasoning — the same
cognition-shaped-output failure, just longer. Each stop is written to the graph
as a `:Hop` **as it happens**, not reconstructed at the end, so the journal entry
can recount the path actually taken — *I started here, found X, which made me
look at Y* — instead of presenting a conclusion with the working thrown away.

Five is Juniper's number; it is disclosed rather than derived. The real ceiling
is time (`HARNESS_FCC_TIMEOUT_SEC`, observed turns reach 31–40 steps), not steps.
A sixth hop is not granted inside the turn — Orion leaves itself a continuation
note and the next run opens there. That is the difference between thinking about
something for an afternoon and coming back to it tomorrow, and it is what lets a
world view accumulate rather than having to fit in one sitting.

### The clock is told to Orion, not assumed

Run `32b42392f495` — the first run that completed the whole loop — spent its
entire budget investigating and was killed mid-writeup: `grounding=fcc_timeout`,
`draft_len=66`, one hop of five recorded. What survived was a prior whose counts
were wrong in two separate ways: an intake gate's *trigger* (`substantive_shift`,
`orion/memory/consolidation_gate.py:74`) read as a rejection *filter*, and two
different crystallization kinds — rejected `stance` against active `semantic` —
compared against each other as one population. The investigation was sound. The
transcription was done against a wall.

Two changes, and neither is a bigger timeout:

**Orion can read its own deadline.** The motor stamps it into the sandbox at
spawn time (`orion/harness/fcc_motor.py:_build_subprocess_env`):

```bash
echo ${ORION_TURN_BUDGET_SEC:-no clock}
test -n "$ORION_TURN_DEADLINE_EPOCH" \
  && echo $(( $ORION_TURN_DEADLINE_EPOCH - $(date +%s) )) \
  || echo "no clock"
echo ${ORION_TURN_STEP_STALL_SEC:-unknown}   # per STEP, not per turn
```

The values are **not** written into the prompt as literals. `HARNESS_FCC_TIMEOUT_SEC`
lives in the harness-governor's env and Hub — which builds the prompt — cannot
read it, so any number stated there would be a second copy free to drift the
moment the governor is retuned. This is not hypothetical: on 2026-08-26 live was
`1600` while `.env_example`, the compose default, and the governor's own
`settings.py` default all still said `900`. A prompt that confidently states the
wrong deadline is worse than one that states none. So the whole-turn number
Orion sees is the one the whole-turn timeout loop enforces against, and a test
(`test_the_budget_is_never_stated_as_a_hardcoded_duration`, which scans the
entire assembled prompt in both graph states) fails if anyone writes a duration
back in.

**`ORION_TURN_STEP_STALL_SEC` is disclosed because the whole-turn deadline is
not the only wall.** `_stream_stall_timeout_sec` bounds a *single* `readline`,
and the CLI emits no stream-json line until a step completes — so one unbounded
query dies with `fcc_stream_stalled` while the outer clock still reads generous.
Showing only the outer number would actively encourage the step that trips the
inner one. Two further walls are *not* stamped and remain undisclosed to the
turn: the accumulated-context ceiling (`fcc_draft_length_ceiling_exceeded`) and
Hub's own outer `asyncio.wait_for`.

**The `test -n` guard is load-bearing, not decoration.** Bash expands before it
evaluates, so `$(( $UNSET - $(date +%s) ))` prints a confident negative and exits
0 — measured, `-1787785130`. When the caller has no deadline the keys are
**cleared** rather than left unset, but clearing does not make absence
self-evident to a shell; it only makes the wrong number absurd (~-1.8e9) rather
than plausible (~-3000). The prompt carries the real guard, and says what `no
clock` means. That state is not reachable in production today —
`run_fcc_turn` is the only production caller and always passes all three — the
clearing is there so the prompt stays honest wherever it is reused.

**The last quarter of the budget belongs to writing**, and each node is written
at the moment it is formed rather than at the end. A prior in the graph at a
confidence Orion can raise later beats a perfect one that never got written. If
the thread is still live when the clock says stop, that is what the continuation
note is for.

---

## 8. How a decision made inside the turn crosses back out

The loop needs to know two things the turn decided: *do I want to continue this
line of enquiry?* and *is this worth telling Juniper about?*

The obvious mechanism — a fenced JSON block in the prose, parsed by the loop —
was **rejected**. It makes Orion's decision an artifact of formatting, loses a
real finding to a malformed fence, and puts a regex between a model and a
decision it already has a place to record.

Instead Orion writes it into its own graph:

```cypher
CREATE (:TurnOutcome {
  run_id: "<this run>",
  continue_line: true,
  continue_note: "still don't know why substrate.route has no edges",
  reach_out:     false,
  reach_out_why: "",
  written_at:    timestamp()
})
```

**Absence is the safe default.** No node means no continuation and no message. A
turn that ran out of time, or that Thought refused, simply leaves nothing behind
— and nothing is inferred from its prose, so silence is silence rather than a
guess. The loop reads this back read-only, keyed on **this run's** id rather than
"the newest node", because reading the newest would silently attribute a previous
run's decision to this one every time a turn died before writing its own.

---

## 9. Outreach is a second turn

If `reach_out` is set, the loop fires a **separate** `execute_unified_turn` to
compose the message rather than reusing text from the first.

It costs a whole extra turn and buys one specific thing: **the second turn gets
its own `ThoughtClient.react()` stance check.** So Orion can find something
genuinely worth saying, and the system can still independently decide *not now,
she is in the middle of something*. One turn would collapse "this is
interesting" and "this is worth interrupting her for" into a single judgement
made at the worst possible moment — while Orion is still excited about what it
just found.

Delivery goes through `EndogenousOutreach.offer_message`, so it inherits that
module's gates unchanged: quiet hours 23:00–08:00 MDT, the daily cap, the
cooldown, and "not while a turn is in flight". **Those are shared** with
tension-triggered outreach, because from Juniper's end they are the same
interruption. The gates protect the human, not the module — see
`services/orion-hub/README.md` §4.1.

Orion may also decline at this point: the composition prompt asks for the exact
token `PASS`, which `is_pass_response` checks, and nothing is sent.

Off by default (`HUB_CURIOSITY_OUTREACH_ENABLED=false`).

---

## 10. The gates, in order

Every one of these writes nothing and says why at INFO or WARNING. A loop that
always finds something worth writing up manufactures significance daily.

| reason | meaning |
|---|---|
| `disabled` | the loop is off |
| `daily_cap` | already run N times today, on the **operator's local date** |
| `cooldown` | less than 4h since the last run |
| `pg_role_missing` | `orion_readonly` does not exist — checked through Hub's **own** privileged pool, so the credential is not validating itself |
| `graph_unavailable` | the ACL could not be asserted |
| `stores_not_ready` | `app.state.memory_pg_pool` does not exist **yet** — INFO on the first tick after a Hub start, WARNING from the second on. Found on the first real deploy: the loop's first tick lost a race with pool construction by 139 ms |
| `stores_unavailable` | the memory tables could not be **read** — a fault, not a race, and **never** the same state as an empty mind |
| `no_approved_material` | genuinely nothing to show |
| `empty_generation` | the turn produced nothing, was deferred, timed out, or returned error-shaped text |
| `no_lookup` | **the load-bearing one** — see below |

**`no_lookup` is the gate that keeps this honest.** The prompt asks Orion to
only claim what a lookup supports, but a prompt is an instruction, not a
mechanism: a turn that called no tools and wrote four fluent paragraphs from
parametric knowledge produces a perfectly well-formed `llm_response` and, without
this check, lands in the journal byte-for-byte indistinguishable from a real
investigation. `harness_step_count` is already on the final frame and costs
nothing to read. The bar is 3 — far below a genuine run (the first real
investigation reached 29 steps) and high enough to exclude the degenerate case.

It applies to the **investigation** turn only. The composition turn is
deliberately given nothing to look up, so holding it to the same bar would kill
outreach on any change to the stream shape, reported as `empty_generation`.

**The journal reports what Orion actually wrote to its graph**, by label, and
says *"wrote nothing to its own graph this run"* when it wrote nothing. Fluent
prose about having worked something out, with an empty graph behind it, is the
empty-shell-cognition failure exactly. A footprint that could not be *read* is
a third state and prints nothing at all — an unreadable graph is not a graph
Orion wrote nothing to.

---

## 11. Files

| file | what it owns |
|---|---|
| `study_material.py` | the two Postgres stores, sampled at random and never ranked |
| `worldview.py` | Hub's **read-only** view of `orion_worldview` — priors, `:TurnOutcome`, run footprint, hop notes, recently-settled |
| `acl.py` | the FalkorDB grant, and the idempotent graph bootstrap |
| `sandbox_env.py` | the seven-key allowlist into the `claude -p` subprocess |
| `kickoff_prompt.py` | the invitation — priors, continuation, access, overlay, hops, schema |
| `outreach_prompt.py` | the composition turn |
| `services/orion-hub/scripts/curiosity_investigation.py` | the tick loop, the gates, the turn |

The loop lives in Hub rather than in its own service for a measured reason:
calling `execute_unified_turn` from a standalone process inside the Hub
container times out at 300s with `session_turn_phase_read_bus_unbound`, because
the harness RPC worker and several module bus binds live in Hub's own event
loop. It also reads `app.state.memory_pg_pool`. So: a sibling loop, not a
service.

---

## 12. Inspecting it live

```bash
# What has Orion worked out? (read-only, as the default user)
docker exec orion-athena-falkordb redis-cli GRAPH.RO_QUERY orion_worldview \
  "MATCH (p:Prior) RETURN p.status, p.confidence, p.claim ORDER BY p.confidence"

# Is the pool actually shrinking, or is this polling?
docker exec orion-athena-falkordb redis-cli GRAPH.RO_QUERY orion_worldview \
  "MATCH (p:Prior) RETURN p.status, count(p)"

# Does confidence ever go DOWN? (the check that matters most — see below)
docker exec orion-athena-falkordb redis-cli GRAPH.RO_QUERY orion_worldview \
  "MATCH (p:Prior) WHERE p.status IN ['refuted','revised'] RETURN p.claim, p.confidence"

# What did one run actually do?
docker exec orion-athena-falkordb redis-cli GRAPH.RO_QUERY orion_worldview \
  "MATCH (n) WHERE n.run_id = '<run_id>' RETURN labels(n)[0], count(n)"

# The prose side
psql -h localhost -p 55432 -U postgres -d conjourney \
  -c "SELECT created_at, left(body, 200) FROM journal_entries
      WHERE source_ref LIKE 'curiosity:%' ORDER BY created_at DESC LIMIT 5"

# The ACL as it actually stands
docker exec orion-athena-falkordb redis-cli ACL LIST | grep orion_curiosity
```

---

## 13. What this does not establish

Stated plainly, because the surrounding vocabulary invites overclaiming and root
`CLAUDE.md` forbids it.

- **Confidence is Orion's own belief, not a calibrated probability.** Nothing
  outside the loop checks it. Every surface says so.
- **Orion is grading its own homework.** The detector for that is empirical:
  across 20 runs there must be at least one `refuted` and one *downward*
  revision. **If confidence is monotonic, this loop is not learning — it is
  accumulating agreement with itself**, and the graph would make that permanent.
  This is the first thing to check, before anything is built on the numbers.
- **A graph nothing reads is `journal_entries` again** — 36k rows and one
  reader. Currently `orion_worldview` is read by Hub (to build the next prompt)
  and by Orion (in-turn). Neither feeds Orion's *chat* context. Whether it
  should is an open question deliberately left to Juniper.
- **Nothing here is evidence of anything felt.** It is a loop that records what
  it was uncertain about and can be checked, later, on whether it got less so.

---

## 14. Configuration

All keys are `HUB_CURIOSITY_*` in `services/orion-hub/.env_example`. The ones
worth knowing:

| key | default | note |
|---|---|---|
| `HUB_CURIOSITY_INVESTIGATION_ENABLED` | `false` | the pydantic `Field` default is `False` so an absent key can never start it. `.env_example` ships `true` as the intended live value; the local `.env` currently holds it `false` |
| `HUB_CURIOSITY_INVESTIGATION_MIN_COOLDOWN_SEC` | `14400` | 4h. This is a real turn on the pipeline that serves Juniper |
| `HUB_CURIOSITY_INVESTIGATION_DAILY_CAP` | `3` | Redis-persisted on the **local** date; a redeploy is not a licence to run again |
| `HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC` | `1500` | budgeted from the harness's own ceilings, not copied from outreach's 300 — a too-short ceiling burns the compute *and* discards the answer |
| `HUB_CURIOSITY_GRAPH_HOST` / `_PORT` | `127.0.0.1` / `6380` | **Hub's** address for FalkorDB; see §6 |
| `HUB_CURIOSITY_GRAPH_ORION_USER` / `_PASSWORD` | `orion_curiosity` / *(blank)* | the credential Hub **grants**, never the one it uses. Blank ⇒ the graph half disables itself and the rest of the loop still runs |
| `HUB_CURIOSITY_SANDBOX_HUB_URL` | `http://host.docker.internal:8080` | Hub as seen **from Orion's sandbox** |
| `HUB_CURIOSITY_PRIOR_SAMPLE` | `8` | open priors shown |
| `HUB_CURIOSITY_STALE_PRIOR_TESTS` | `3` | a guess from the design doc, not a measured number — revisit against real data |
| `HUB_CURIOSITY_MAX_HOPS` | `5` | Juniper's number |
| `HUB_CURIOSITY_OUTREACH_ENABLED` | `false` | the only part that reaches Juniper |

Orion's own credentials live in `~/.fcc/.env` as `ORION_CURIOSITY_*` (seven
keys) — never in this repo.

---

## 15. Source material

- `docs/superpowers/specs/2026-08-26-orion-priors-and-worldview-design.md` — the
  design, including the three claims its own first draft got wrong
- `docs/superpowers/pr-reports/2026-08-26-curiosity-priors-and-worldview-pr.md`
  — the build, the review, and the live evidence
- `services/orion-hub/README.md` §4.1 (endogenous outreach — the gates this
  shares) and §4.2 (this loop, from Hub's side)
- `orion/sentience_striving_program/README.md` §15 — what this contributes to
  that program's outcomes, and what it does not
