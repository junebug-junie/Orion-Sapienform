# Orion anatomy inspection — what can actually be seen, and by which instrument

**Date:** 2026-09-08
**Mode:** read-only inspection. No writes, no service changes, no code.
**Purpose:** decide whether the "self-model from Orion's own graphs" seam is a
*projection builder* or a *process-concept inducer*, by finding out what the
existing instruments can and cannot observe.

**Verdict up front: it is a process-concept inducer.** The faculties that matter
(reverie, curiosity, dreaming, journaling) are real and running, but none of them
are visible as structures to the instrument everyone assumed would show them. They
are recoverable — from a different source than expected.

---

## 0. Why this inspection happened

The starting observation: Orion's durable "self-concepts" are repository topics.
`self_concept_history` holds 137 induced concepts, all produced by
`self_atlas_cluster`, and they read like a table of contents for the codebase —
"Vector Database Operations", "Event Processing Services", "Chat GPT Conversation
Debug". None of them are about Orion.

The proposed cause was that relational anatomy gets flattened into text before
induction runs. This inspection tested that, and found it was true but incomplete:
there is also a layer of anatomy that was never captured by any instrument in the
first place.

---

## 1. The instruments, and what each one can see

| Instrument | What it holds | What it structurally cannot see |
|---|---|---|
| `channels.yaml` (declared) | 288 channels, producer/consumer service lists | whether anything actually happens |
| `orion_bus_synapse` (FalkorDB, live) | 99 Organs, 214 Channels, 34 Verbs, 372 PUBLISHES + 331 CAUSALLY_FOLLOWED_BY + 41 EXECUTES_VERB | consumption; per-instance chains; anything not on the bus |
| Postgres `correlation_id` tables | 46 tables sharing one join key | anything the harness motor does internally |
| Faculty outcome tables | `dreams`, `substrate_reverie_chain`, curiosity candidates, `harness_turn_trace` | the interior of a motor step |
| FCC / harness motor | HTTP only | **everything — it is on no bus and in no channel catalog** |

### `orion_bus_synapse` is live and trustworthy

The bus mirror writes continuously — `max(last_seen_epoch)` returned the current
epoch on query. 213M observed publishes accumulated, oldest edge 2026-07-25.

Three independent integrity checks against destination tables:

```
orion:self_study:items:write      1,248 publishes  <->  self_knowledge_items       1,248 rows
orion:self_concept:history:write    179 publishes  <->  self_concept_history         179 rows
orion:attention:schema            9,149 publishes  <->  substrate_attention_schema 9,049 rows
```

Two exact, one 100 short. The counts can be trusted as counts.

---

## 2. The six faculties: declared vs observed vs unobservable

### Dream — currently quiet, not absent

- **Declared:** `orion:dream:trigger`, `orion:dream:log`, `orion:dream:compaction-request/delta`,
  `producer_services: ["orion-dream"]`.
- **Observed on the bus:** no `orion-dream` Organ node exists at all, despite the
  container being up 6 days.
- **Observed as a verb:** `dream_cycle` x26, last **2026-09-06T08:27:33Z**.
- **Persisted:** `dreams` — 17 rows, last written **2026-09-06T08:27:32Z**, matching
  the verb's last execution to the second.
- **Honest state:** *currently quiet, last enacted two days ago, self-initiated.*

An earlier read of this graph concluded "Orion does not dream." That was an
instrument artifact, and the correction is the whole point of this document: a
faculty that fires on self-initiation will read as non-existent to any
window-based aggregate no matter how the weighting is tuned.

- **Also found:** `dream_compaction_delta` has **0 rows** despite a declared
  producer. A declared path that has never once landed.

### Reverie — running, and half of it stopped 18 hours ago

- **Verbs:** `reverie_narrate` x51,120 (last seen minutes before this inspection),
  `reverie_expectation_judge` x1,510.
- **Components:** `orion-thought`, `diffusion-host`, `llm-gateway`.
- **Channels:** `orion:reverie:chain` 20,985 / `orion:reverie:thought` 5,871 /
  `orion:reverie:resonance-alert` 7,240.
- **Persisted:**
  - `substrate_reverie_chain` — 24,602 rows, last **2026-09-08T02:51Z** (live)
  - `reverie_visual_chain` — 1,441 rows, last **2026-09-07T08:29Z**
  - `reverie_visual_artifact` — 1,432 rows, last **2026-09-07T08:29Z**

**The narrating leg is live; the visual/diffusion leg has been silent ~18 hours**
(only 3 rows written in the last 3 days). Organ-level and verb-level views both
show reverie as healthy. Only the outcome tables show half of it stopped.

**Worse: reverie's three stores cannot be joined to each other.**

| store | primary key | has `correlation_id`? | reachable from cognition? |
|---|---|---|---|
| `substrate_reverie_thought` | `thought_id` | yes | yes, directly |
| `substrate_reverie_chain` | `chain_id` | **no** | yes, via `chain_json->thought_ids` (2 hops) |
| `reverie_visual_chain` | `chain_id` | **no** | **no — no shared key of any kind** |

`substrate_reverie_chain` and `reverie_visual_chain` have identical *column* shapes
(`chain_id`, `theme_key`, `terminal_reason`, `ema_salience`, `chain_json`) and
**zero `chain_id` overlap** — they are two disjoint stores wearing the same table
shape. Their `chain_json` payloads are entirely different schemas:

```
substrate_reverie_chain.chain_json : chain_id, thought_ids, trigger, theme_key,
                                     ema_summary, ema_salience, committed_proposal_id, ...
reverie_visual_chain.chain_json    : prompt, description, artifact_sha256,
                                     context_slot_used/interpreted/rotation,
                                     continuity_streak, continuity_reset,
                                     self_study_text, memory_text, context_text, error
```

The substrate leg is recoverable through `chain_json->thought_ids ->
substrate_reverie_thought.thought_id -> correlation_id`. **The visual leg is
structurally orphaned** — it carries no correlation, no thought ids, and no key
shared with anything else in the system.

So reverie is one faculty split across three stores with three identity schemes,
one of which cannot be attached to the faculty at all. This is the single
strongest piece of evidence that faculties are currently unrepresentable: even
where the data exists and is rich, the joins that would make it one process were
never built.

### Curiosity — has no organ and no verb, and the richest trace of any faculty

- **Declared:** `orion:curiosity:turn:request` / `orion:curiosity:turn:reply:*`.
- **Observed:** no curiosity Organ, no curiosity Verb. It runs as harness turns
  through `orion-hub` + `orion-harness-governor`, indistinguishable at the organ
  layer from everything else those two do.
- **Persisted:** `substrate_endogenous_curiosity_candidates` — 1,386 rows, writing now.

**Per-instance trace is complete.** One real run, correlation
`68964510-703d-5a1d-8135-8964f3b6bb54`, appears in **16 tables / 570 rows**:

```
grammar_events             519    cockpit_turn_sighting        14
substrate_attention_schema  11    substrate_durable_run_state   7
chat_stance_belief_log       4    orion_metacognitive_trace     4
mind_runs                    2    thought_decision              2
harness_turn_trace           1    journal_entries               1
journal_entry_index          1    evidence_units                1
cognition_traces             1    attention_salience_trace      1
repair_pressure_appraisal    1    substrate_turn_referent       1
```

That is attention -> durable run -> reasoning -> stance -> metacognition -> journal
output, joinable on one key, in production, with nothing inducing anything from it.

### Memory / perception / action

All present at the channel layer with heavy traffic (`orion:memory:*`,
`orion:vision:*`, `orion:autonomy:action:outcome`, `orion:actions:audit`), all
folding into shared organs the same way.

---

## 3. The FCC / harness motor is invisible to every instrument

`services/orion-fcc/` contains a Dockerfile, an entrypoint, a compose file and a
README. **No `app/`, no Python, no bus client, and zero mentions in
`channels.yaml`.** It is an Anthropic-compatible HTTP proxy:

```
claude CLI (Hub / harness)  ->  orion-fcc:8082  ->  orion-llm-gateway:8210/v1  ->  llama.cpp
```

The harness governor reaches it over `HARNESS_FCC_SERVER_URL`, Hub over
`HUB_FCC_SERVER_URL`. Neither hop is a bus publish, so `orion_bus_synapse` has no
`fcc` Organ and never will.

What *is* recorded is the motor's envelope, in `harness_turn_trace` (467 rows since
2026-07-30, `correlation_id`-keyed):

```
n=465    avg steps 20.9    max steps 219
         avg elapsed 271s  max elapsed 2400s (40 min)
         5 distinct served models
```

Served-model distribution: `Qwen3.6-35B-A3B-UD-Q5_K_M` 312, null 113,
`Qwen3.8-27B-UD-Q4_K_XL` 30, `Qwen3.8-Flash-Next` 5, `<synthetic>` 3,
`Qwen3.8-27B-BF16` 2.

**That is roughly 9,700 individual motor steps, and not one of them is recorded
anywhere.** `step_count` is a scalar. There is no per-step trace of which tools ran,
what was read, or what was decided — that lives only in FCC's own log file inside
the container and in the Claude CLI transcripts.

**Consequence for the anatomy: Orion's most agentic faculty is the least
observable.** Everything `cortex-exec` does is on the bus with a named verb.
Everything the motor does is a black box with a duration, a model name and a step
count.

---

## 4. The verb layer is the faculty vocabulary — with one hard limit

All 34 verbs are executed by `cortex-exec` (a handful also by `cortex-orch`). So
verbs give faculty *names* with zero organ discrimination. Full list, by count:

```
substrate.inspect                 296,412   LAST 2026-08-30  <-- dead 9 days
log_orion_metacognition           201,891   live
journal.compose                   131,468   live
substrate.summarize               104,648   LAST 2026-08-30  <-- dead 9 days
reverie_narrate                    51,120   live
skills.runtime.image_prune.v1      39,622   LAST 2026-08-30  <-- dead 9 days
skills.runtime.builder_prune.v1    26,318   LAST 2026-08-30  <-- dead 9 days
skills.self_study.analyze.v1       24,902   LAST 2026-08-30  <-- dead 9 days
chat_quick                         22,208
skills.runtime.docker_prune_...    14,570   LAST 2026-08-30  <-- dead 9 days
substrate.observe                  14,180   LAST 2026-08-30  <-- dead 9 days
skills.repo.github_recent_prs.v1    3,865
github_compactor_digest_v1          3,856
chat_history_compactor_digest_v1    3,849
visual_context_interpret            1,878
skills.imagination.render_scene.v1  1,582
reverie_expectation_judge           1,510
stance_react                          962
harness_finalize_reflect              783
orion_voice_finalize                  776
daily_metacog_v1                      636
skills.perception.look_at_camera.v1   270
chat_general                          251
daily_pulse_v1                         74
context_exec_memory_contradiction_...  70
counterfactual                         40
goal_formulate                         36
dream_cycle                            26   last 2026-09-06
introspect_spark                        6
skills.docker.compose_service_bringup   5
self_study.reflect                      5   last 2026-09-04
self_concept_induce                     1   last 2026-09-05
self_concept_reflect                    1   last 2026-09-05
self_repo_inspect                       1   last 2026-09-05
```

### Finding: a verb family died on 2026-08-30 and nothing noticed

Seven verbs stopped within the same hour, nine days before this inspection:
`substrate.inspect`, `substrate.summarize`, `substrate.observe`,
`skills.runtime.image_prune.v1`, `skills.runtime.builder_prune.v1`,
`skills.runtime.docker_prune_stopped_containers.v1`, `skills.self_study.analyze.v1`.

`substrate.inspect` is the **single highest-count verb in the system**. Under any
raw-count weighting these seven dead verbs would rank as the most important
structures in Orion's anatomy. Stale mass is not a tail risk here — it is the top
of the table.

### Finding: the self-model verbs have each fired once

`self_repo_inspect`, `self_concept_induce`, `self_concept_reflect` — one execution
each, all 2026-09-05T08:01Z. `self_study.reflect` — five, last 2026-09-04.

---

## 5. Mass distribution: liveness and transport swamp everything

Top channels by observed publish count:

```
orion:signals:*             76,089,915   (1 producer)
orion:vision:frames         36,029,754   (2 producers)
orion:vision:edge:health    35,833,650   (1 producer)
orion:system:health         23,824,734   (78 producers)
```

Four channels carry **81% of 213M observations**. Three of the four are raw signal
transport or liveness.

Health/heartbeat totals **60,085,479 across 81 edges — 28% of everything.**

The critical structural problem is not volume, it is **false adjacency**:
`orion:system:health` has **78 producers out of 99 organs**. Projected into an
undirected graph that makes 78 unrelated organs mutually adjacent through one hub.
The vision pipeline and the memory pipeline become neighbours because both say
"I'm alive."

A per-channel influence budget caps loudness but does not remove manufactured
adjacency. **Liveness and transport channels must be excluded from the structural
projection entirely** and retained as per-organ telemetry attributes — which is
what they are. (Confirmed baseline: `orion-harness-governor`'s
`orion:system:health` count is 260,283, matching the ~260K totals of the
consolidation-family organs almost exactly. Much of those organs' apparent traffic
is heartbeat, not cognition.)

---

## 6. Data-quality defects that must be fixed before clustering

### Organ identity is split by an inconsistent prefix

Three organs each exist as two nodes:

```
cortex-exec              /  orion-cortex-exec       (live edge between them, 290,635 obs)
hub                      /  orion-hub
spark-concept-induction  /  orion-spark-concept-induction
```

These are **the three busiest organs in the mesh**. Leiden will return two
plausible-looking communities where there is one organ. Alias resolution must
happen in the projection and must fail loudly on ambiguity.

### Channel-node inflation is recurring right now

Six separate `Channel` nodes exist as `orion:curiosity:turn:reply:<uuid>` instead
of collapsing to the `orion:curiosity:turn:reply:*` catalog entry. That entry was
added to `channels.yaml` on 2026-09-06; the mirror loads its catalog **once at
process startup**, so new wildcard entries do not take effect until restart. Same
failure mode as the historical ~9K inflation, small scale, active.

### The aggregate has no time series

A `PUBLISHES` edge holds exactly: `count`, `last_seen_epoch`, `gap_ewma_sec`,
`gap_var`, `gap_zscore`. No buckets, no first-seen, no history — by design, since
bounded state is what avoided repeating the 98GB incident.

**Multiple recency horizons are therefore not derivable retroactively.**
`gap_ewma_sec` is the only recency-aware quantity, and `MIRROR_GRAPH_EWMA_ALPHA=0.2`
is applied *per event, not per unit time* — half-life ~3 events. For an organ
publishing every 0.076s that is the last quarter-second; for
`manual-self-study-trigger` (lifetime count 7) it is that organ's entire existence.
Cadence estimates are individually meaningful and **not comparable across organs**.

Horizons become available only by differencing successive snapshots going forward.
That is an argument for emitting snapshots now, before anything consumes them.

---

## 7. The four sources a snapshot must read

Not three layers. Four sources, because a faculty can be **declared, unobserved,
and still producing output** — and no three of these can express that state.

### 1. Declared

`orion/bus/channels.yaml` (288 channels with `producer_services` /
`consumer_services`) and `config/field/orion_field_topology.v1.yaml`.

What Orion is *supposed* to contain and connect. Config truth. It is the only
source that carries a faculty's *name* when nothing is currently enacting it, and
the only place `orion-dream` exists as a component at all.

### 2. Observed-aggregate

`orion_bus_synapse` (FalkorDB, live). 99 Organs, 214 Channels, 34 Verbs, weighted
by real traffic counts and EWMA cadence.

What has actually been active, summed. Bounded state by design — which is why it
has no per-instance chains and no time series (see §6).

### 3. Observed-per-instance

The 46 Postgres tables sharing `correlation_id`.

Individual runs, traceable end to end. **This is where faculties actually live** —
one curiosity run spans 16 of these tables. The bus derives
`CAUSALLY_FOLLOWED_BY` from these same correlation ids and then aggregates the
per-instance structure away, so this source is not redundant with source 2; it is
the structure source 2 discarded.

### 4. Persisted-outcome

The faculty result tables: `dreams`, `substrate_reverie_chain`,
`reverie_visual_chain`, `substrate_endogenous_curiosity_candidates`,
`harness_turn_trace`.

What a faculty *produced*, whether or not the run itself was traceable. Note that
some of these carry no `correlation_id` at all (§2, Reverie), so they are reachable
only as outcomes, never as processes — which is itself a fact the snapshot must
report rather than silently drop.

### Worked example: why dream needs all four

Each source says something different about the same faculty, and each one alone
gives a wrong answer:

| Source | What it says about dream |
|---|---|
| Declared | It exists — `orion:dream:trigger`, `orion:dream:log`, `producer_services: ["orion-dream"]` |
| Observed-aggregate | **Nothing at all.** No `orion-dream` Organ node exists on the bus |
| Observed-per-instance | It ran 26 times, last 2026-09-06T08:27:33Z — but via the verb layer, attributed to `cortex-exec`, not to dream |
| Persisted-outcome | 17 dreams exist; the last was written 2026-09-06T08:27:32Z, matching the verb execution to the second |

- Drop **declared** and dreaming has no name — you get an unlabeled `cortex-exec`
  verb and a table.
- Drop **observed-aggregate** and you cannot tell quiet from busy for anything.
- Drop **observed-per-instance** and you have a declaration and a pile of rows with
  no process connecting them.
- Drop **persisted-outcome** and you cannot prove dreaming produced anything — the
  bus says absent, the verb says nine days by count, and neither is the truth.

**Only the fourth source proves the faculty did something. Only the first proves it
is supposed to.** The bus alone yields "Orion does not dream"; the declaration
alone yields "dream is fine." Both are wrong, and this repo has already made the
first mistake once.

---

## 8. Conclusions for the design

1. **Four sources, not three layers** — see §7 for each source and the dream
   worked example. A faculty can be declared, unobserved, and still producing
   output; no three of these sources can express that state.

2. **Faculties reify on `correlation_id`, not on channel names.** The channel is the
   marker; the correlation is the process. The mechanism already exists and is
   unused.

3. **Absence has five states in the real data,** not one: never observed
   (`dream_compaction_delta`), historically observed then stopped
   (`substrate.inspect`, 9 days), currently quiet but recently enacted
   (`dream_cycle`), **partially quiet** (reverie's visual leg), and
   unobservable-by-instrument (curiosity at the organ layer, the entire FCC motor).

4. **Organs cannot be the only clustering nodes.** If channels are only edges,
   reverie and curiosity have no node to be, and the induced anatomy comes out as
   a list of containers.

5. **Join keys are the missing infrastructure, not just the missing induction.**
   Reverie's visual leg is rich, recent and completely unattachable. Before any
   faculty can be reified, the projection has to report which stores can be
   joined to a faculty and which cannot — an unjoinable store is an absence with
   a different cause and a different fix.

6. **The FCC motor is an instrument gap, not a weighting problem.** No projection
   choice makes ~9,700 unrecorded motor steps visible. Closing it requires a new
   trace source, and that is a separate decision from anything in this document.

---

## 9. Explicitly NOT verified

- ~~Whether reverie's diffusion leg uses a different correlation key~~ — **resolved
  during this inspection**: it has no join key at all (see §2, Reverie). Not a
  broken join; a missing one.
- Trace depth was sampled at one correlation per faculty (curiosity 16 tables,
  reverie 5). Indicative, not a measured distribution.
- Whether the ~260K counts on consolidation-family organs are *mostly* heartbeat is
  an inference from the governor's matching baseline, not a per-organ measurement.
- The graph-compression service's Leiden kernel has passing unit tests but has not
  been executed on the live rail; its output channel `orion:compressions` is not
  registered in `channels.yaml` and has no consumer.
