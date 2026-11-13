# Fold self-study into curiosity — proposal

Date: 2026-10-29. Asked by Juniper. Design only; no code in this PR.
Proposal mode (CLAUDE.md §0A): this touches self-modeling.

## Arsonist summary

Self-study is the old way Orion studied itself. It still runs, but almost nothing uses
what it makes. Curiosity now does the real self-study work: it asks Orion "what am I",
it reads the code with the dig kit, and it writes the identity definition that chat uses.

The fold kills five self-study parts outright:

1. **The daily reflection loop.** It re-describes the same repo snapshot every day, and no code reads its output.
2. **Concept induction.**
3. **The retrieve path.** Nothing reads it now.
4. **The Claude enrichment service.** It exists only to feed the reflection loop.
5. **The `self_study` body-economy family.**

The fold keeps two parts that have a real reader:

- **The hourly telemetry analysis.** The reverie painter uses it in one of every three paintings.
- **The shared scanner** `orion/self_inventory.py`. The dig kit uses it.

One part needs Juniper's choice. The daily repo scan feeds only Self Atlas. Self Atlas has
no UI, and 71% of its cluster rows say "no keywords available". My recommendation is to kill
both, because the dig kit gives Orion the same facts on demand, with history.

This reverses part of the 2026-09-03 design (`2026-09-03-orion-endogenous-self-model-and-journal-design.md`,
PR #2079). That design built Layers 1-3 and Self Atlas. It also built `self_concept_history`, and
`self_concept_history` survives, because curiosity writes the identity definition there. Layer 3 did
not become what the design hoped for. The live evidence is below.

## Current architecture

All numbers are live reads on 2026-10-29 (read-only `psql` on `orion-athena-sql-db`, plus container files).

### What runs

| Loop | Trigger | Output | Volume |
|---|---|---|---|
| **Inspect (Layer 1)**: scans the repo, hardware and stance beliefs into "self-facts" | `self_study_refresh_loop`, `self_study_refresh.py:108`, `SELF_STUDY_INSPECT_INTERVAL_SEC=86400` (cortex-exec chat-lane container) | `self_knowledge_items` (one row per fact per run) + one `journal_entries` row "Self-study factual snapshot" (`source_kind=self_study`) | ~1,490 rows/day, 1 run/day, 22:24 UTC; 59,296 rows since 09-05 |
| **Reflect (Layer 3)**: LLM writes "findings" about the snapshot | `self_study_reflect_refresh_loop`, `self_study_refresh.py:150`, `SELF_STUDY_REFLECT_REFRESH_INTERVAL_SEC=86400`; direct path (`SELF_STUDY_REFLECT_DURABLE_ENABLED=false`) | `self_concept_history` `produced_by=layer3_reflect` + `journal_entries` `source_kind=self_reflection` | 14 d: 94 rows, **9 concept_ids, 51 distinct texts**; 1 journal/day ~06:45 UTC |
| **Analyze**: compares a 6 h window of one telemetry table against the window before it | execution-dispatch route `analyze_self_study_source` → `skills.self_study.analyze.v1` (cortex-exec-background) | One `journal_entries` row titled "Self-study analysis: …" (`source_kind=self_study`, `source_ref=<source>:<digest>`), only when a rule fires | 233 dispatches/7 d, avg `raw_len` 19; **37 journal rows/30 d** (vision 24, cocreation 8, affect 5) |
| **Enrichment**: `claude -p` writes semantic notes on changed files | git post-commit hook → `orion:self_study:enrichment:requested` → `orion-self-study-enrichment` | Cache files on its volume, read only by reflect's `semantic_enrichment` concept | 441 cache files, 85 since 10-15; cap 8/day |
| **Induce / retrieve / structural mass** | Verbs only (`self_concept_induce`, `self_retrieve`), run as a side step of reflect | In-memory concepts; `structural_mass_history.jsonl` | JSONL has **2 lines ever** |
| RDF writeback | — | `channel_retired` since 2026-07-28 | none |

### What is never true

- **Self-study has never written a `grammar_acts` row.** The 7-day actor list has `durable.self_inquiry`
  (35), `durable.curiosity_investigate` (39) and `durable.self_sense_eval` (42). It has no self-study actor.
- **The `self_study` family (`config/physiology/families.yaml:166`, Tier 1) never bids.** Reflect is
  excluded from acts on purpose (`services/orion-durable-runs/app/run_acts.py:113`, "upkeep, not wants").
  Analyze goes through execution-dispatch and is not priced.
- **The `self_review` workflow has no recent runs.** It is user- and autonomous-invocable
  (`orion/cognition/workflows/registry.py:142`) and wraps reflect. It has 0 matches in the retained
  cortex-orch logs (UNVERIFIED beyond log retention).

### What curiosity already does (the successor)

- **The self-inquiry line** (`orion/curiosity/self_inquiry.py`, `tick_self_inquiry` in
  `services/orion-hub/scripts/curiosity_investigation.py:2978`) asks the standing self-questions
  with a GPU agent lane. It is graded as acts (`ask`, family `durable.self_inquiry`, Tier 2). It writes
  `self_concept_history` `self:definition` and `self:lived:*` (111 rows/30 d).
- **The stance identity kernel reads only curiosity's `self:definition`.** The path is
  `orion/substrate/felt_state_reader.py:111` → `chat_stance.py:787-835`. The kernel does not read
  self-study output.
- **The dig kit** (`orion/dig`, PRs #2915/#2920) runs `inventory` / `compare` / `history` / `body`
  over any past checkout. Chat and curiosity can both use it. Its `inventory` command uses the same
  scanner as Layer 1 (`orion/self_inventory.py`).
- **The body atlas** (#2898) covers hardware and organs. Layer 1's hardware facts used to cover this.
- **29 of 109 curiosity journals (14 d) cite repo files.** Orion already reads its own code during
  curiosity.
- **Curiosity has already audited self-study.** A 10-12 self-inquiry run lists `self_study.py` and the
  reflect graph as "the main self-study/reflection pipeline". Self-inquiry also counts
  `self_knowledge_items` and `self_concept_history` rows as part of "what am I made of" (10-05, 10-14, 10-25).

### Layer 3's own verdict

Layer 3's top recurring finding is about itself. Its journal goes to a non-authoritative surface,
and recall gates that surface away. The 14-day numbers show it re-describes one snapshot: 9 concept ids
and 51 distinct texts across 94 rows. Its top concepts are `bus_topology_pattern`,
`semantic_enrichment`, `journaling_surface` and `service_cluster`. These are architecture labels, not
behavior. The 09-03 design warned about this ("a linter describing itself").

## Inventory: every producer, every consumer

"Live?" means a running code path reads it today (not "a human could psql it").

| Producer → sink | Consumer | Live? | What breaks if it stops | Fate |
|---|---|---|---|---|
| Reflect → `self_concept_history` `layer3_reflect` | Its own freshness timer (`self_study_refresh.py:78`) and version lookup (`self_study.py:2905`) only. Self-inquiry can psql it ad hoc. | **No** | Nothing. The self-inquiry ledger shows a frozen "last" time (see dangerous failure). | **Kill.** Keep the rows (history). |
| Reflect → `journal_entries` `self_reflection` (1/day) | Personality "journal" lane (`personality_sources.py:247`, reads every journal kind except world_pulse/orion_day) | Yes, as one of many rows | ~5 fewer evidence rows a week. The lane continues. | **Kill.** Personality loses a low-value input. |
| Reflect → `_REFLECTION_RESULT_CACHE` (in process) | `self_retrieve` | No | Nothing | **Kill** |
| Reflect durable path (`reflect_graph.py`, `admitted_reflect_graph.py`, runner registration) | Nobody (flag off; Hub and actions filter its state events out) | No | Nothing | **Kill** |
| Enrichment service → cache files | `self_study.py:1000-1050` (reflect's `semantic_enrichment` concept) only | Only via reflect | Nothing after reflect dies. It saves up to 8 `claude -p` calls/day. | **Kill** the service, its compose, its channel, and the post-commit hook fragment |
| Induce → structural_mass JSONL | `self_study.py` itself | No (2 lines ever) | Nothing | **Kill** |
| `self_retrieve` verb + `SelfStudyConsumerContextV1` (`verb_adapters.py:72-246`) | `legacy.plan` when the policy is on (default off; no producer sets it) | No | Nothing | **Kill** with `self_study_policy.py` |
| RDF builders (`build_self_*_rdf_request`) + GraphDB read path | Nothing (channel retired 07-28) | No | Nothing | **Kill** |
| `self_review` workflow → reflect | A user phrase ("reflect on yourself", "run self study") | No runs seen | The phrase stops routing to a workflow | **Kill.** Or alias it to a self-inquiry run (open question 3). |
| Inspect → `self_knowledge_items` | Self Atlas topic-foundry (`concept_atlas_routes.py:308,717,758`; Hub scheduler `scripts/main.py:1505-1571`) | **Yes** (re-clusters daily) | Self Atlas freezes on the last snapshot. The freshness guard (`concept_atlas_routes.py:680`) stops retraining. Nothing crashes. | **Juniper decides** (open question 1) |
| Self Atlas → `self_concept_history` `self_atlas_cluster` (4,706/30 d) | No code reader (`self_panel.py` only borrows a helper). Self-inquiry counts it. No Hub UI (`?graph=self` is API-only). | **No** | Nothing in code | Follows the Self Atlas choice |
| Inspect → `journal_entries` "Self-study factual snapshot" (1/day) | Personality journal lane | Yes, one row/day | Small loss of evidence rows | Follows the Self Atlas choice |
| `orion/self_knowledge_freshness.py` | Self-study timers + the Self Atlas guard | Yes (Atlas) | — | Follows the Self Atlas choice |
| `orion/self_inventory.py` (scanner) | Dig kit `inventory`/`compare` (`orion/dig/__main__.py:27,303-332`, CI `dig-kit-tests.yml`) | **Yes** | The dig kit breaks | **Keep** (it already lives outside self-study) |
| Analyze → `journal_entries` "Self-study analysis: …" | orion-thought visual chain (`store.py:935-995` → `visual_chain.py:1021-1100`): one of three rotating context slots, allowlisted by `ANALYSIS_SOURCES` source_ref prefixes | **Yes:** 20 of 60 reverie chains in 14 d used `context_slot_used=self_study` | Reverie paintings lose one context source and fall back to memory/reverie | **Keep** (reason below) |
| Analyze journals | Hub reverie tab "Self-study observation" block (`reverie-tab.js:305`) | Yes (display) | The block disappears | Keep, with analyze |
| Analyze journals | Personality journal lane; Juniper's ratings (`chat_response_feedback.target_artifact_ref`) | Yes | Fewer rows | Keep, with analyze |
| Analyze → Redis `orion:self_study:last_run:<source>` | Analyze itself (picks the most overdue source) | Yes | — | Keep |
| `self_sense_eval_log` (932 rows) | — | — | **Not self-study.** It is the curiosity scheduler's `tick_self_sense_eval`. | Unchanged |
| `self_concept_history` `curiosity_self_inquiry` | Stance identity kernel, self panel, personality seed | Yes | — | **Not self-study.** Unchanged. |
| Journal label `source_kind='self_study'` | Shared by curiosity (`orion/curiosity/journal.py:27`, 326 rows/30 d), compaction, reading, analyze, inspect | Yes | — | **Keep the label.** Do not rename it in this fold (see non-goals). |

### The one genuine reason to keep part of self-study

Analyze has a live reader that matters: the reverie painter's context. The reader is
deterministic. It runs four windowed SELECTs and makes no LLM call. It is cheap: `lanes: []`,
`requires_gpu: false`. It writes something only when a disclosed rule fires, which is about once a day.
Analyze is a sensor, not a self-study loop. Folding it into curiosity would make a cheap sensor
expensive. Keep it, and keep its `ANALYSIS_SOURCES` source_ref prefixes exactly as they are. If the
prefixes change, orion-thought's allowlist silently admits nothing.

Analyze could feed curiosity later. A notable finding could become a curiosity seed. That is a new
feature, not part of this fold (see non-goals).

### Does repo self-inspection deserve its own curiosity line?

Not yet. The evidence:

- Self-inquiry already looks at the code. 29 of 109 curiosity journals cite repo files, and
  self-inquiry runs count the self tables.
- The dig kit already gives `inventory`/`compare`/`history` on demand.
- A new line costs real GPU agent-lane hours. No measured gap justifies that cost.

The honest test is whether self-inquiry uses the dig kit for repo questions. Today 0 of 109 curiosity
journals name a `dig` command in their text. That is weak evidence, because a run can use a tool
without naming it in the journal. So patch 2 is measurement first (see Recommended next patch).
If that shows a real gap, here is the line spec to build:

- **Line `self_repo`.** It is a durable `curiosity.investigate` run with `brief.line="self_repo"`. Its
  seed is a `python -P -m orion.dig compare <last-run-date> now` diff that is not empty. The "nothing
  changed" case is a skip, not a run.
- **Family `durable.self_repo`.** Tier 2, lanes `[agent, agent-gpu2, agent-deep]`, window 86400, verb `infer`.
  Add it to `run_acts.FAMILIES` so each run opens and closes a graded act (`artifact_produced`).
  It shadow-bids like `durable.self_inquiry`.
- **Output.** One curiosity journal (`source_ref curiosity:*`). An optional `self:lived:repo` row in
  `self_concept_history` is written only when the definition actually changed.
- **Cap.** At most 1 run a day. It needs Juniper's Tier 2 sign-off, because a new family is a tier decision.

## Proposal-mode fields

- **Capability that changes.**
  - Orion stops making daily LLM "reflections" about a repo snapshot.
  - Orion stops writing ~1,490 self-fact rows a day, if the Self Atlas choice is "kill".
  - Self-modeling is then done only by curiosity: self-inquiry, the dig kit and the body atlas.
  - Reverie context is unchanged (analyze stays).
- **Data touched.**
  - Writes stop to these tables and files: `self_concept_history` (`layer3_reflect` only),
    `journal_entries` (`self_reflection`, and "factual snapshot" if inspect dies), `self_knowledge_items`
    (if inspect dies), the enrichment cache volume, and the structural_mass JSONL.
  - No table is dropped and no row is deleted. History stays queryable.
- **Privacy boundary.** None between Juniper and Orion, by Juniper's standing decision. The one
  existing boundary is orion-thought's allowlist, which keeps curiosity journals out of reverie.
  That allowlist is unchanged.
- **Trace that proves it worked.**
  1. For 72 h after deploy, the newest `layer3_reflect` row and the newest `self_reflection` journal
     are older than the deploy time.
  2. `orion-self-study-enrichment` is gone from `docker ps`.
  3. Analyze journals keep landing (≥1 in 72 h).
  4. `reverie_visual_chain.context_slot_used='self_study'` keeps appearing.
  5. Self-inquiry runs keep completing (`grammar_acts` actor `durable.self_inquiry`).
  6. `scripts/check_verb_physiology.py` passes with no `self_study` family.
- **Dangerous failure mode.**
  1. **Orion misreads the frozen tables.** Self-inquiry counts `self_knowledge_items` and
     `self_concept_history` as "what I am made of". If the feeds stop silently, Orion can read the
     frozen counts as its self-model dying, or as a fault to investigate. It could spend agent-lane
     hours on it, or write a wrong self-definition. Mitigation: the same patch tells Orion. Put one line
     in the self-inquiry ledger (`self_inquiry.py:777-790`), such as "layer3_reflect retired 2026-MM-DD (PR #…)",
     and write one journal entry that announces the retirement.
  2. **Analyze is killed by accident.** Analyze shares the `self_study` name, the label and the module
     prefix. A careless "kill everything named self_study" would kill it too. Reverie would then lose
     its third context slot with no error. Mitigation: an acceptance check on `context_slot_used`.
- **Disable / roll back.**
  - Every kill is a code deletion in one PR, so `git revert` of that PR restores it.
  - Before the deletion PR, the reversible first step needs only env: set
    `SELF_STUDY_REFLECT_REFRESH_INTERVAL_SEC=0`, stop `orion-self-study-enrichment`, and, if the Atlas
    choice is kill, set `SELF_STUDY_INSPECT_INTERVAL_SEC=0` and `SUBSTRATE_TOPIC_FOUNDRY_SELF_SCHEDULER_ENABLED=false`.
  - No migration and no table drop, so nothing to un-migrate.

## Body-economy consequence

The `self_study` family (`config/physiology/families.yaml:166`, Tier 1, lanes `[agent]`) is dropped.
After the fold, its six verbs go this way:

| Verb | Fate |
|---|---|
| `self_study.reflect`, `self_concept_reflect`, `self_concept_induce`, `self_retrieve` | Deleted (no family needed) |
| `self_repo_inspect` | Deleted if Self Atlas dies. If Atlas stays, it needs a family (see below). |
| `skills.self_study.analyze.v1` | **Reassign.** It is plumbing with no GPU and no want. Option A (recommended): tier 0, family `autonomic`, `lanes: []`, window 0. It never bids, which is what it does today in practice. Option B: its own Tier 1 family `self_analysis`. |

**Tier changes need Juniper's sign-off** (families.yaml header, decision of 10-23). Moving analyze
from Tier 1 to Tier 0 `autonomic` is a tier change. So is dropping a Tier 1 family.

## Missing questions

1. **Self Atlas: kill or keep?** If it stays, the daily inspect must stay, moved out of `self_study.py`
   into a small sensor that calls `orion/self_inventory.py`, plus hardware and behavioral facts.
   Evidence for killing it: no UI, 71% of cluster rows have no keywords, no code reader, and the dig kit
   covers the code facts.
   Evidence for keeping it: it is the only thing that clusters stance-belief behavioral facts
   (`self_study.py:549-591`). Nothing else turns `chat_stance_belief_log` into self-facts.
2. **Analyze's family:** Tier 0 `autonomic` (recommended) or its own Tier 1 family?
3. **The `self_review` phrase.** Delete it, or route "reflect on yourself" to a self-inquiry run?
4. **The `self_repo` curiosity line.** Build it now, or measure first (recommended)?

## Proposed schema / API changes

- **Removed.**
  - Channel `orion:self_study:enrichment:requested` (from `orion/bus/channels.yaml`).
  - Registry entries for `SelfStudyRetrieveRequestV1`/`ResultV1` and `SelfStudyConsumerContextV1`.
  - The durable workflow `self_study.reflect`.
  - The workflow `self_review`.
  - If Atlas dies: channel `orion:self_study:items:write` and its sql-writer route. The
    `self_knowledge_items` table itself stays.
- **Narrowed.** `SelfConceptHistoryProducer` keeps `layer3_reflect` (and `self_atlas_cluster`) as a
  valid literal, so old rows still validate. Nothing new writes them.
- **Unchanged.** `journal.entry.write.v1`, `source_kind='self_study'`, `ANALYSIS_SOURCES`,
  `self_concept_history`, `self_sense_eval_log`.
- **Env removed (cortex-exec).**
  - `SELF_STUDY_REFLECT_*` (4 keys).
  - `SELF_STUDY_ENRICHMENT_CACHE_MOUNT_DIR`, `SELF_STUDY_STRUCTURAL_MASS_HISTORY_PATH`,
    `SELF_STUDY_GRAPH_PATH`/`_HOST_PATH` (`SELF_STUDY_NAMED_GRAPH` is in local `.env` only).
  - If Atlas dies: `SELF_STUDY_INSPECT_INTERVAL_SEC`.
  - The whole `services/orion-self-study-enrichment/.env_example`.
  - Run `python scripts/sync_local_env_from_example.py` after.
  - The live enrichment `.env` holds an OAuth token. Revoke it when the service is deleted.

## Files likely to touch

- `services/orion-cortex-exec/app/self_study.py`: cut to inspect only (if Atlas stays), or delete.
- `services/orion-cortex-exec/app/self_study_harness.py`, `self_study_policy.py`, `self_study_refresh.py`,
  `scripts/self_study_harness.py`: delete or shrink.
- `services/orion-cortex-exec/app/verb_adapters.py`: drop the `self_*` verbs and the `legacy.plan` self-study context.
- `services/orion-cortex-exec/app/main.py`: drop the startup tasks.
- `settings.py`, `.env_example`, `docker-compose.yml`: drop the keys and volumes.
- `orion/cognition/verbs/{self_concept_induce,self_concept_reflect,self_repo_inspect,self_retrieve,self_study.reflect}.yaml`:
  delete. `skills.self_study.analyze.v1.yaml`: change its physiology block.
- `orion/cognition/workflows/registry.py`, `services/orion-cortex-orch/app/workflow_runtime.py`: remove `self_review`.
- `services/orion-durable-runs/app/{reflect_graph,admitted_reflect_graph}.py`, `runner.py`, `admission_runtime.py`, `run_acts.py`.
- `services/orion-self-study-enrichment/` (whole service), `scripts/self_study_enrichment_hook.py`,
  `scripts/git_hooks/post-commit` (hook fragment), rebuild lists (`scripts/service_rebuild_paths.yaml`, `include_services_circe`/`exclude_services`).
- `orion/bus/channels.yaml`, `orion/schemas/registry.py`, `orion/schemas/self_study.py`, `orion/schemas/self_study_enrichment.py`.
- `config/physiology/families.yaml`.
- `orion/curiosity/self_inquiry.py`: one ledger line naming the retirement.
- If Atlas dies: Hub `concept_atlas_routes.py` self-graph wiring, `self_atlas_cluster_history.py`,
  `scripts/main.py` scheduler, `orion/self_knowledge_freshness.py`, and `SUBSTRATE_TOPIC_FOUNDRY_SELF_*` keys.
- Stale references: `orion/spark/concept_induction/settings.py:71`, `services/orion-spark-concept-induction/.env_example:38`.
- Tests under each touched service; `services/orion-cortex-exec/evals/run_self_study_analysis_eval.py` stays.

## Non-goals

- No table drops and no row deletes. History stays queryable.
- No rename of the `source_kind='self_study'` journal label. Curiosity writes most of those rows
  (326/30 d), and orion-thought filters them by `source_ref` prefix. A rename is a separate migration.
- No change to analyze's behavior, its sources, or its `source_ref` prefixes.
- No new curiosity line in the first patches (see the measure-first rule above).
- No change to self-inquiry, the identity kernel, the self-sense eval, or the dig kit beyond one ledger line.
- No analyze → curiosity seed bridge (possible follow-up).

## Acceptance checks

1. 72 h after deploy, `select max(created_at) from self_concept_history where produced_by='layer3_reflect'`
   is older than the deploy.
2. `docker ps` shows no `orion-self-study-enrichment`. The post-commit hook publishes nothing on the retired channel.
3. Analyze still lands at least 1 journal in 72 h. `reverie_visual_chain` still shows
   `context_slot_used='self_study'` at least once in 72 h.
4. `durable.self_inquiry` acts keep closing. A self-inquiry ledger line names the retirement.
5. `scripts/check_verb_physiology.py`, `check_schema_registry.py`, `check_bus_channels.py`,
   env parity and compose parity all pass. `pytest orion/dig/tests` passes, which proves the scanner survives.
6. `grep -rn "self_study_reflect\|SelfStudyRetrieve\|self_study:enrichment"` over `services orion config scripts` returns nothing.
7. If Atlas dies: no `self_atlas_cluster` row and no `self_knowledge_items` row after the deploy.

## Recommended next patch

1. **Patch 1 (env only, reversible, after Juniper answers question 1).**
   - Set `SELF_STUDY_REFLECT_REFRESH_INTERVAL_SEC=0` and stop the enrichment service.
   - If she chooses to kill Atlas, also set the inspect interval to 0 and turn off the Self Atlas scheduler.
   - Add the self-inquiry ledger line.
   - Then watch acceptance checks 1-4 for 72 h.
2. **Patch 2 (the kill).**
   - Delete the code, verbs, workflow, durable graph, service, channel, env keys and family listed above.
   - Reassign analyze per question 2.
   - Kill means kill: no flag left behind, and no fallback to reflect.
3. **Patch 3 (measure, then maybe build).** Count how often self-inquiry runs call `orion.dig` from the
   harness tool-call record. Build the `self_repo` line only if repo questions go unanswered.

## Juniper's decisions (2026-10-30) and what shipped

Juniper approved the proposal on 2026-10-30 with four answers. Patches 1 and 2 shipped together as one kill patch (PR #2986).

1. **Self Atlas: keep for now.** The daily Layer 1 scan (`run_self_repo_inspect`, `self_study_refresh_loop`, `SELF_STUDY_INSPECT_INTERVAL_SEC`) stays, cut down to that one job. No new sensor service. `self_study.py` went from ~3,400 lines to ~800. **Self Atlas and this feed are the next retirement candidates.**
2. **Analyze moves to tier 0, family `autonomic`** (`lanes: []`, window 0). The `self_study` family is gone from `config/physiology/families.yaml` (it was never in `contract.yaml`). The `ANALYSIS_SOURCES` source_ref prefixes are unchanged and pinned by a test, because orion-thought's allowlist depends on them.
3. **"Reflect on yourself" routes to a self-inquiry run.** The `self_review` workflow now publishes one `CuriositySelfInquiryRequestV1` on the new channel `orion:curiosity:self_inquiry:request`. Hub answers with `tick_self_inquiry(force=True, force_reasons={cooldown, daily_cap})`. Unlike the operator's run-now button, the chat phrase skips only the cooldown and the daily cap; heat, strain, rest and the time window still hold it. cortex-orch checks the channel has a listener before it says "requested".
4. **No `self_repo` line. Measure instead.** `scripts/report_dig_kit_usage.py` reads the harness transcripts (read-only). Live numbers on 2026-10-30, last 3 days (the dig kit went live 10-27): 6 of 21 curiosity investigate runs used the dig kit; 0 of 13 self-inquiry runs did; 4 of those 13 still read repo code directly. Over 14 days: 6 of 64 and 0 of 52.

Retired outright (kill means kill, no flag left): the Layer 3 reflect loop and its durable workflow, concept induction, the retrieve path and its consumer policy, the RDF builders, the self-study harness, the four self-study verbs, the reflect prompt, and the `orion-self-study-enrichment` service with its channel, schema, git hook fragment and env keys.

The dangerous failure mode (Orion reading frozen counts as their self-model dying) is handled in the existing self-inquiry records ledger: `orion/curiosity/self_inquiry.py` `LEDGER_RETIREMENTS` prints one line under each affected table, in the prompt every self-inquiry run reads.
