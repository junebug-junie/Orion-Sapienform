# Self Atlas: what replaces it, then how it retires

**Status:** design only (proposal mode). Nothing in this document is built yet.
**Date:** 2026-10-30. **Asked by:** Juniper: "we need a replacement for self atlas in one of the
existing new seams we're building before we retire."
**Builds on:** `2026-10-29-self-study-into-curiosity-proposal.md` (PR #2986, merged), which kept
Self Atlas "for now" and named it and its daily scan "the next retirement candidates".
**Related open design:** PR #2888 (`2026-10-23-repo-chronicle-design.md`), which already names the
body atlas ladder as the thing that converges where Self Atlas did not.

All numbers are live reads on 2026-10-30 (read-only `psql` on `orion-athena-sql-db`, read-only
`GRAPH.RO_QUERY` on `orion-athena-falkordb`, container env and files), unless marked otherwise.

## Arsonist summary

Self Atlas does not need a new replacement. The replacement is already built and live: the body
atlas organs plus the dig kit. They are on demand, they cost about one second of CPU, and Orion
already uses them. The one thing that is missing is a pointer. Self-inquiry asks "What am I, and
what am I made of?" every run, but its "where the records are" section points at the repo and at
SQL tables, not at `dig body`. So the replacement is a one-section prompt patch in self-inquiry.
After that, Self Atlas, its daily scan and its graph can go.

The reason this is safe: nothing in code reads Self Atlas output. Orion read it in self-inquiry
from 09-08 to 10-05 and then stopped. Most of what Orion wrote about it was a complaint: "most of
what I have ever said about myself was not said by me. It was induced." The one thing it was kept
for (stance beliefs turned into self-facts) landed in 1 of 492 clusters. Self-inquiry already reads
that table directly, and 45 self-inquiry rows cite it.

The cost difference is large. In 14 days Self Atlas used 9.8 h of embedding (80% of all
topic-foundry embedding) and 17.1 h of LLM labelling (35% of all topic-foundry enrichment). Each
run re-embeds about 14,700 rows that hold only about 2,100 distinct facts.

## Current architecture

### What runs today

| Step | Where | Trigger | Output | Live volume |
|---|---|---|---|---|
| Daily scan ("Layer 1") | `services/orion-cortex-exec/app/self_study.py` `run_self_repo_inspect`, called by `self_study_refresh.py` `self_study_refresh_loop` (`main.py:1140-1153`) | `SELF_STUDY_INSPECT_INTERVAL_SEC=86400` | `self_knowledge_items` via channel `orion:self_study:items:write` → orion-sql-writer; one journal "Self-study factual snapshot" | 1,488-1,494 rows/day (10-24..10-29); 44,867 rows / 30 d, **2,116 distinct item_ids** |
| Train | Hub scheduler `scripts/main.py:1512-1536` → `concept_atlas_routes.trigger_topic_foundry_self_training_run` → orion-topic-foundry model `orion-self-atlas-v1-*` | Hub topic-foundry tick, skips when the table is unchanged (`self_atlas_source_unchanged_since_last_run`, `concept_atlas_routes.py:680`) | ~780 clusters per run | 15 complete runs / 14 d; 14,553 docs each; 50-127 min each |
| Enrich (LLM labels) | same scheduler, `trigger_topic_foundry_self_enrichment` | same tick | cluster labels and keywords | 17.1 h LLM time / 14 d |
| Ingest | `concept_atlas_ingest_topic_foundry_self` (`concept_atlas_routes.py:2107`) | same tick | FalkorDB graph `orion_substrate_self` | **21,426 nodes** (more than `orion_substrate`, 9,767); 10-30 tick wrote 199 concepts, 579 edges |
| History | `scripts/self_atlas_cluster_history.py` `publish_self_atlas_cluster_history` | same tick | `self_concept_history` `produced_by='self_atlas_cluster'` | 4,853 rows / 30 d, 2,300 concept ids; 147-432/day; **68-74% "no keywords available"** |

### What the scan contains (30 d, rows / distinct items)

`schema` 22,975 / 797, `channel` 10,704 / 364, `verb` 4,261 / 141, `service` 3,199 / 106,
`module` 2,029 / 71, `hardware` 665 / 23, `behavioral` 600 / 600, `env_surface` 248 / 8,
`touchpoint` 186 / 6.

- Code facts (schema, channel, verb, service, module) are the same scanner the dig kit uses:
  `orion/self_inventory.py`. A full scan takes 0.63 s on the host (measured).
- Hardware facts are a copy of `config/field/orion_field_topology.v1.yaml` (4 nodes, 7
  capabilities, 11 edges). It is a static file. The field digester already reads it.
- Behavioral facts are the 20 newest `chat_stance_belief_log` rows, copied once a day
  (`self_study.py:383-436`).

### What reads Self Atlas output

| Output | Reader in code | Evidence |
|---|---|---|
| `self_concept_history` `self_atlas_cluster` | **None.** Personality (`personality_sources.py` `self_inquiry_seed` + `SEED_SQL`), felt state (`felt_state_reader.py:181`) and the self-sense runner (`self_sense_runner.py:53-66`) filter to `curiosity_self_inquiry`. `self_panel.py` reads `self:definition` / `self:lived:%` only and keeps its own copy of one helper. | grep + SQL text |
| FalkorDB `orion_substrate_self` | **None in cognition.** Only Hub's `?graph=self` query parameter on `/api/substrate/concepts/*` (`concept_atlas_routes.py:1200`). | `static/js/concept-atlas.js` never sends `graph=` (all six fetches at lines 464-963 omit it). No Hub template names Self Atlas. Hub logs since its 10-30 02:29 restart: 0 `api/substrate/concepts` calls (UNVERIFIED before that, log retention). |
| "Self Atlas UI" | **Does not exist.** The Concept Atlas page shows Orion's own graph only. | same |
| `self_knowledge_items` table | Self Atlas training; self-inquiry's records ledger (count + last time); `orion/evals/self_sense.py` table aliases (a name Orion can cite). | `self_inquiry.py:126,789,807` |
| "Self-study factual snapshot" journal | Personality journal lane, one row/day. The body is counts only ("services=104, modules=71, channels=358, …"). | SQL |

## Capability inventory

"Live" means a running path uses it, or Orion used it within the last 14 days.

| # | What Self Atlas provides | Live? | Evidence | Would Juniper or Orion miss it? |
|---|---|---|---|---|
| C1 | A map of "what I am made of", grouped into themes (clusters) | **Dead since 10-05.** | Self-inquiry rows that name Self Atlas or its clusters: last on 10-05 (`self:definition`). Journal mentions: 18 from 09-06 to 10-04, none after. Since then self-inquiry cites the shared concept atlas (`orion_substrate`), not Self Atlas. | The *need* is live: it is self-inquiry's standing question. The *instrument* is not used. The body atlas organs answer the same need better (see below). |
| C2 | Orion's own verdict on its self-model ("95% induced, not said by me") | Dead (same dates) | 09-12, 09-29, 09-30 x3, 10-01, 10-05: Orion counted `self_atlas_cluster` rows and called them an induced cloud around a thin self-authored spine. | No. This is a critique of Self Atlas, not a use of it. |
| C3 | Juniper-shaped clusters ("Juniper Affect Models", "Juniper Affective State") | Dead (09-30, 10-03 only) | Two self-inquiry answers to "who matters". The labels are topics over *code* names. | No. Orion now answers "who matters" from chat, dreams and the person records (`self:lived:lived.who_matters`, 3 rows mention the shared atlas). |
| C4 | Stance beliefs turned into self-facts (the reason PR #2986 kept it) | **Effectively dead.** | Of the 492 clusters written in the last 2 days, **1** cites a `behavioral` item (schema 156, service 81, channel 53, module 47, touchpoint 8, hardware 7, env 6). `chat_stance_belief_log` is in `SELF_INQUIRY_PG_TABLES` (`self_inquiry.py:125`), granted read-only, and **45** self-inquiry rows cite stance (last 10-28). | No. Self-inquiry reads the raw table already. |
| C5 | Hardware facts | Dead as a Self Atlas input (7 of 492 clusters) | Static config copy. | No. `dig body` lists hosts from `config/biometrics/node_catalog.yaml` with live liveness, which is better. |
| C6 | Code facts over time (what changed) | Live as data, not as use | 44,867 rows / 30 d. No reader compares runs. | No. `dig inventory <date>` and `dig compare <date> now` recompute any day since 2025-06-24 from git, so a stored daily copy adds nothing. |
| C7 | Hub API `?graph=self` | Dead | No UI caller (above). | No. |
| C8 | Daily "Self-study factual snapshot" journal | Live, trivial | 1 row/day of counts into the personality journal lane. | No. Counts are not personality evidence. |
| C9 | Ledger row `self_knowledge_items` in every self-inquiry prompt | Live | `format_ledger`, `self_inquiry_prompt.py:244` | Keep the row (history stays readable). Its note must change when the scan stops. |

## Candidate seams, inspected

| Seam | State (live) | Fit |
|---|---|---|
| **Body atlas organs** (`orion/body_atlas/`, #2898; names #2951) | Built and live. Rungs 1-3: parts → circulation → organs, Louvain with a Q ≥ 0.3 eval. Covers services, hosts, cameras, stores (`falkor:*`, `pg:*`), processes. Orion ran `dig body` on 10-28 and 10-29 (`/mnt/orion-fcc/dig/2026-10-2{8,9}/body-now-1.json`). **No organ is named yet** (`/mnt/orion-fcc/self/organ_names.jsonl` does not exist). Declared-only build: 1.10 s on the host (measured). | **Best fit for C1.** It is the same "map of what I am made of", but it converges (PR #2888's ladder) where Self Atlas fragmented into ~780 clusters. Orion names the groups. A cluster label was written by an LLM over code names. |
| **Dig kit** (`orion/dig`, #2915/#2920) | Live. `inventory`, `compare`, `history`, `body`, `body-economy`, `name`. The harness prefix tells every non-reading turn it exists (`orion/harness/prefix.py:373`, `HARNESS_FCC_DIG_BRIEF_ENABLED=true` in harness-governor). | **Best fit for C5, C6.** On demand, any date, no stored copy. |
| **Curiosity self-inquiry** (`orion/curiosity/self_inquiry.py`, `self_inquiry_prompt.py`) | Live. Standing question "What am I, and what am I made of?" (`self_inquiry.py:100`). Reads `chat_stance_belief_log` directly. Writes `self:definition`. Used the dig kit in **0 of 13** runs (3 d) and 0 of 52 (14 d) per the #2986 report (re-run of `scripts/report_dig_kit_usage.py` from the host on 10-30 returned an empty result, so this number is UNVERIFIED today). | **The consumer.** It is where the replacement must be *seen*. Its records section (`_records_section`, `self_inquiry_prompt.py:228-252`) names the repo and SQL, but not the body atlas. |
| **GitNexus code graph** (harness MCP, `orion/fcc/self_index_brief.py`) | Live (`HARNESS_FCC_GITNEXUS_ENABLED=true`). | Covers "where does a concept live in the code", the only thing a code-topic cluster could still answer. No change needed. |
| **Personality graph / Temporal Self** | Live. Neither reads Self Atlas. | No fit. Both are about how Orion behaves and what happened, not what Orion is built from. |
| **Body-economy report** (`/physiology/report`, `dig body-economy`) | Live; PRs #2987/#2991 open. | No fit. It is cost and wants, not composition. |
| **PR #2888 rungs 4-5** (capacities, portrait) | Design, open. | The future *standing* record of "who I am", one paragraph, versioned. If a standing record is ever wanted, it goes there, not into a daily scan. |

## Recommendation

| Kept capability | Host | Why |
|---|---|---|
| C1 map of what I am made of | **Body atlas organs, via `dig body`** | Already live, converges, Orion-named, has an eval. |
| C4 stance beliefs as self-facts | **Self-inquiry's direct read of `chat_stance_belief_log`** (no change) | Already live and used (45 rows). The Self Atlas copy added nothing. |
| C5 hardware | **`dig body`** (hosts) | Live liveness instead of a static copy. |
| C6 code over time | **`dig inventory <date>` / `dig compare`** | Any day, from git; no stored copy. |
| C9 the ledger row | **Self-inquiry `LEDGER_RETIREMENTS`** | Same mechanism #2986 used. |
| C2, C3, C7, C8 | **None (retire)** | No reader; Orion's own verdict on C2 was negative. |

**On demand beats always on here.** Nothing needs a standing record:

- Code history is already a standing record: git. The dig kit reads it for any date.
- Stance history is already a standing record: `chat_stance_belief_log`.
- The body map changes when the code or the live bus changes, and `dig body` rebuilds it in about
  a second when asked.

### Cost difference

| | Self Atlas (always on) | Body atlas + dig (on demand) |
|---|---|---|
| Embedding | 9.8 h / 14 d on `vector-host` (MiniLM); 80% of topic-foundry embedding | none |
| LLM | 17.1 h / 14 d of enrichment through the LLM gateway; 35% of topic-foundry enrichment | none (Orion's own turn reads the result) |
| Writes | ~1,490 `self_knowledge_items` rows/day, 147-432 `self_concept_history` rows/day, a 21,426-node graph, 1 journal/day | one JSON file per dig under `/mnt/orion-fcc/dig/<date>/` |
| CPU | cortex-exec daily scan + Hub scheduler ticks (4 steps) | 0.63 s inventory, 1.10 s declared body (host); live body adds read-only psql and bus reads (not measured) |
| When idle | full cost every day | zero |

## Proposal-mode fields

- **What capability changes.** Self-inquiry's "where the records are" section gains the body
  atlas and the dig kit as the way to answer "what am I made of". Then the daily scan, the Self
  Atlas model runs, the cluster history rows and the self graph stop.
- **What data is touched.** Reads: none new (dig and body atlas already read what they read).
  Writes stopped: `self_knowledge_items`, `self_concept_history` `self_atlas_cluster`,
  FalkorDB `orion_substrate_self`, one journal/day. Old rows stay.
- **Privacy boundary.** None new. All of this is Orion reading Orion. `chat_stance_belief_log`
  stays readable only through the existing read-only `orion_readonly` grant.
- **What trace proves it worked.**
  - Replacement: a self-inquiry run's harness transcript shows a `python -P -m orion.dig body`
    (or `inventory` / `compare`) call, counted by `scripts/report_dig_kit_usage.py`.
  - Retirement: no new rows / nodes in the stopped sinks; no `substrate_topic_foundry_self_*`
    log lines; topic-foundry runs only `orion-hub-autonomous-*` and `orion-hub-aitown-*`.
- **Dangerous failure modes.**
  1. *Orion reads a frozen `self_knowledge_items` count as a dead organ.* Mitigation: a
     `LEDGER_RETIREMENTS` line in the same patch, with the date and where to look instead.
  2. *`?graph=self` silently serves Orion's own graph.* Today an unknown `graph` value falls back
     to the Orion graph (`concept_atlas_routes.py:1196-1203`). If "self" is removed from the valid
     list, a stale caller gets the wrong graph with no error. Mitigation: no caller exists; the
     retirement patch adds a test that `graph=self` is not resolved to the self store (the store
     no longer exists) and logs `concept_atlas_ignored_bad_graph_param`.
  3. *The analyze sensor is killed by accident.* It shares `self_study` names. Mitigation: do not
     touch `self_study_analysis.py`, `skills.self_study.analyze.v1`, `ANALYSIS_SOURCES`; note that
     `self_study.py` imports `_get_engine` *from* `self_study_analysis.py`, not the other way, so
     deleting `self_study.py` is safe. Acceptance check 6.
  4. *Self-inquiry never uses the pointer.* Then "what am I made of" is answered from the repo
     README and SQL only, as it is today. This is no worse than today (Self Atlas has not been
     cited since 10-05), so the retirement does not wait on adoption. Adoption is measured, not
     gated.
- **How to disable or roll back.**
  - Patch A is one prompt section behind no flag; `git revert` removes it.
  - Before the kill, the env-only stop is `SELF_STUDY_INSPECT_INTERVAL_SEC=0` (cortex-exec) and
    `SUBSTRATE_TOPIC_FOUNDRY_SELF_SCHEDULER_ENABLED=false` (Hub).
  - The kill is one PR of code deletion; `git revert` restores it. No table is dropped. The self
    graph is only deleted if Juniper says so (question 1).

## Missing questions

1. **Delete the FalkorDB graph `orion_substrate_self` (21,426 nodes)?** `GRAPH.DELETE` is
   destructive, so it needs your explicit yes. If kept, the body atlas will keep listing
   `falkor:orion_substrate_self` as a live store that nothing writes. Recommendation: delete it
   after a `GRAPH.COPY` to `orion_substrate_self_archive_20261030`, or delete outright since no
   reader exists.
2. **Ship Patch A and the kill together, or A first?** You asked for the replacement first. A
   alone needs about a week to show adoption (self-inquiry is capped at 3 runs a day and is often
   heat-blocked). Recommendation: A first, kill 72 h later regardless of adoption (failure mode 4).
3. **Rows in `topic_foundry_models` / `topic_foundry_datasets` for `orion-self-atlas-*`.** Keep
   (no row deletes, matching #2986) or mark retired? Recommendation: keep; nothing triggers them.

## Proposed schema / API changes

- **Removed.**
  - Channel `orion:self_study:items:write` (`orion/bus/channels.yaml:2372`), message kind
    `self_study.items.write.v1`, its sql-writer route (`orion-sql-writer/app/settings.py:65,207,957`)
    and consumer import (`worker.py:128`), and `SelfKnowledgeItemLogV1`
    (`orion/schemas/self_knowledge_item_log.py`) plus its registry entry.
  - `SelfKnowledgeItemV1`, `SelfSnapshotV1`, `SelfRepoInspectResultV1` and the writeback-status
    models in `orion/schemas/self_study.py`, if nothing else imports them (check
    `self_study_analysis` schemas, which live in a separate file).
  - Hub `POST /api/substrate/concepts/ingest-topic-foundry-self` and the `graph=self` value.
- **Kept.** The `self_knowledge_items` table and its ORM model (sql-writer `create_all`; the table
  holds history). The `self_atlas_cluster` and `layer3_reflect` literals in
  `SelfConceptHistoryProducer`, so old rows still validate.
- **Changed text.** `chat_stance_belief_log` channel description (`channels.yaml:2390`) no longer
  says it feeds `_behavioral_items()`; its reader is self-inquiry.
- **Env removed.** cortex-exec: `SELF_STUDY_INSPECT_INTERVAL_SEC`. Hub:
  `SUBSTRATE_TOPIC_FOUNDRY_SELF_*` keys and `FALKORDB_SELF_SUBSTRATE_GRAPH` (which appears
  **twice** in `services/orion-hub/.env_example`, lines 1036 and 1552). Run
  `python scripts/sync_local_env_from_example.py` after.

## Files likely to touch

**Patch A (replacement):**
- `orion/curiosity/self_inquiry_prompt.py` `_records_section`: one paragraph naming
  `python -P -m orion.dig body` (your organs, live), `dig body <organ_id>`, `dig inventory now`,
  `dig compare <date> now`, and `dig name` (organs you have not named yet).
- `orion/curiosity/tests/` (prompt test): the paragraph is present; it is absent on a
  reading-only turn if that path exists for self-inquiry.
- `scripts/report_dig_kit_usage.py`: confirm it splits self-inquiry from investigate (it does per
  #2986); fix the host run that returned `{}` or document the container to run it in.

**Patch B (retirement):**
- cortex-exec: delete `app/self_study.py`, `app/self_study_refresh.py`; `app/main.py:1140-1153`;
  `app/settings.py:554`; `.env_example:343-350`; `app/verb_adapters.py:2662-2664` comment; tests.
- `orion/self_knowledge_freshness.py`: delete.
- Hub: delete `scripts/self_atlas_cluster_history.py`; `scripts/main.py:56,1343,1512-1590`;
  `scripts/concept_atlas_routes.py` (self dataset/model constants 292-310, 680-760, 828-832,
  1200, 2107-2120); `scripts/api_routes.py:386-412,432`; `scripts/substrate_store_warm.py`;
  `app/settings.py:2140+`; `.env_example`; tests `test_topic_foundry_self_atlas_scheduler.py`,
  `test_self_atlas_cluster_history.py`.
- `orion/substrate/falkor_store.py:1569-1577` `build_self_falkor_substrate_store_from_env`;
  `scripts/repair_substrate_duplicate_identities.py` doc line.
- sql-writer, channels, schemas, registry as above.
- `orion/curiosity/self_inquiry.py` `LEDGER_RETIREMENTS["self_knowledge_items"]`: replace the
  10-30 note with: "retired <date>: the daily repo/hardware/behavior scan no longer writes here,
  and Self Atlas is gone. Its rows stay as history. For what you are made of now, run
  `python -P -m orion.dig body` (organs) or `dig inventory now`; for stance, read
  chat_stance_belief_log." Add a `self_concept_history` line: "`self_atlas_cluster` rows stopped
  <date>."
- Comments that name Self Atlas as live: `orion/curiosity/self_inquiry.py:8-11`,
  `orion/schemas/self_concept_history.py:8`, `orion/curiosity/self_panel.py:42`.
- `services/orion-hub/evals/run_self_sense_eval.py:154` docstring only.

## Patch plan

1. **Patch A: point self-inquiry at the body (the replacement).**
   - Change `_records_section` as above. No env, no schema.
   - Verify: prompt unit test; one forced self-inquiry run (operator run-now) shows the paragraph
     in the run's prompt record.
   - Measure: `report_dig_kit_usage.py` self-inquiry dig count for 72 h.
2. **Patch B: kill Self Atlas (kill means kill).**
   - Delete everything in "Files likely to touch, Patch B" in one PR. No flag is left behind.
     No fallback to the daily scan, the self store, or `graph=self`.
   - Same PR: the `LEDGER_RETIREMENTS` lines, so the first self-inquiry run after deploy reads
     the retirement as a dated decision.
   - Same PR: env removal and `.env` sync; channel, registry and sql-writer route removal.
   - Restart: cortex-exec (chat-lane container that runs the loop), Hub, sql-writer.
3. **Patch C (only if Juniper says yes to question 1):** `GRAPH.COPY` then `GRAPH.DELETE
   orion_substrate_self`, run by hand, printed for Juniper.

## Non-goals

- No new service, no new table, no new daily job, no new standing self-record.
- No change to the analyze sensor (`self_study_analysis.py`, `skills.self_study.analyze.v1`,
  `ANALYSIS_SOURCES`) or the `source_kind='self_study'` journal label.
- No row deletes and no table drops (`self_knowledge_items`, `self_concept_history`,
  `topic_foundry_*`).
- No change to the Orion or AI Town topic-foundry models or graphs.
- No body atlas changes. Rungs 4-5 stay in PR #2888.
- No new metric. This patch removes producers; it wires nothing new into a model.

## Acceptance checks

1. **Nothing in code read it.** `grep -rn "self_atlas\|orion_substrate_self\|SUBSTRATE_SEMANTIC_STORE_SELF\|self_knowledge_freshness\|self_study:items:write"`
   over `orion services scripts config` returns only the kept `SelfConceptHistoryProducer`
   literal, the `LEDGER_RETIREMENTS` text and tests asserting retirement.
2. **The stopped sinks stay stopped (72 h after deploy).**
   - `select max(created_at) from self_knowledge_items` is older than the deploy.
   - `select max(created_at) from self_concept_history where produced_by='self_atlas_cluster'`
     is older than the deploy.
   - `GRAPH.RO_QUERY orion_substrate_self "MATCH (n) RETURN count(n)"` is unchanged (or the
     graph is gone after Patch C).
   - No Hub log line `substrate_topic_foundry_self_scheduler_*`; no new `topic_foundry_runs` row
     for a model named `orion-self-atlas-%`.
3. **What Orion uses still works.**
   - `durable.self_inquiry` acts keep closing, and new `self_concept_history`
     `curiosity_self_inquiry` rows land (≥ 1 in 72 h, heat permitting).
   - Personality and felt state still return a `self:definition` (their filters are unchanged).
   - `pytest orion/dig/tests orion/body_atlas/tests` pass; `python orion/body_atlas/evals/run_organ_structure_eval.py` passes.
4. **The replacement is visible and used.**
   - The self-inquiry prompt record contains the `dig body` paragraph.
   - `report_dig_kit_usage.py` shows ≥ 1 self-inquiry dig call in 7 days. If it shows 0, that is
     reported, not a reason to restore Self Atlas (failure mode 4).
5. **Orion is told.** The first self-inquiry prompt after deploy shows the new
   `LEDGER_RETIREMENTS` lines under `self_knowledge_items` and `self_concept_history`.
6. **Analyze survives.** At least 1 "Self-study analysis: …" journal in 72 h, and
   `reverie_visual_chain` shows `context_slot_used='self_study'` at least once in 72 h.
7. **Gates pass.** `check_schema_registry.py`, `check_bus_channels.py`,
   `check_env_template_parity.py`, compose parity, cortex-exec / Hub / sql-writer tests.
8. **Topic-foundry load drops.** 14-day embedding hours fall by about 80% and enrichment hours by
   about a third, measured from `topic_foundry_runs.stats` (`embed_secs`, `enrichment_secs`).

## Recommended next patch

Patch A: one paragraph in `orion/curiosity/self_inquiry_prompt.py` `_records_section` that points
"what am I made of" at `python -P -m orion.dig body`, plus its test. Then Patch B 72 h later.
