# Reading links that keep getting refused, and putting people into Orion's graph

Status: design (proposal mode). No code or live store changed by this document.
Date: 2026-10-15. Evidence: read-only queries against live Postgres (`conjourney`) and
FalkorDB (`GRAPH.RO_QUERY`), run 2026-10-14/15 UTC. Juniper asked that every question be
answered with evidence, so each question below carries its answer and its numbers.

Privacy: the repo is public. People in Juniper's life appear here only as placeholders
([spouse], [child A], [child B], [boss], [team member A..D], [person S], [person N],
[person R]). Juniper's answers, with real names, are in the gitignored local files the
person-model cleanup reads (section 2.2).

## Arsonist summary

**Reading links.** Orion's reading proposes links and the acceptance rule refuses almost
all of them. Since 2026-10-11, Stage 2 proposed 53 links over 21 reads; 5 were accepted.
Two causes account for 43 of the 48 refusals, and neither is the model being sloppy:

1. **The quote comes from a page Hub did not keep (26 refusals, `quote_not_found`).**
   Stage 1 keeps only the fetch tool's short summary of the page (median 1,368
   characters). Stage 2 then fetches the page again, sees the full text, and quotes it
   word for word. The quote is real; the kept text just doesn't contain it.
2. **The concept's stored name is not a phrase from the text (17 refusals,
   `quote_not_about_subject` / `quote_not_about_endpoints`).** Stage 1 names concepts like
   variable names: 47% of the names offered as link subjects are `snake_case` or
   `kebab-case` (`interoceptive_prioritisation`, `kappa_routing`). The rule requires the
   quote to contain that name exactly, which a sentence of prose never does.

Mechanical fixes (matching text after normalizing spacing and punctuation, ignoring `_`
and `-` in names, and a stricter prompt) would have accepted about 10 of 53 instead of 5,
measured by replaying the stored claims. Getting further needs two rule changes, and
those are Juniper's call (section 1.6). Separately, Orion keeps concluding "the read dies
in connection" because the code file it rereads says `edges=[]`. Links do not go through
that file; they go through the assertion path. That belief gets a correction through the
existing Orion's Day corrections path, plus a one-line pointer in the file Orion keeps
rereading.

**People.** Orion's graph has three person nodes: Juniper, Orion, and [child B] (minted by
the memory path on 10-14). Juniper answered the three open cleanup questions on
2026-10-15. Applying the cleanup would merge 3 graph nodes into Juniper and 3 into Orion,
attach 29 "about" links, flag 2 junk nodes, mark 1 wrong "home", and mint 8 new person
nodes for people she names in her own words. Two more people are unclear and stay as
questions for Orion to ask her. The links to the claims Orion keeps returning to are
reproducible: 21 attention rows over 11 beliefs name Juniper or her people. The
duplicate-node repair from #2670 works on two other graphs, so it does not block this.

## 1. Part 1: reading links

### 1.1 Current architecture

- Stage 1 (`services/orion-hub/scripts/world_pulse_read_pipeline.py`) writes concepts
  through `orion/substrate/adapters/world_pulse_read.py`, which by design returns
  `edges=[]` (line 63). The concept name is whatever the model wrote in
  `concept_candidates[].label`. The prompt asks only for `"label": "string"` (line 220).
- Stage 1 keeps the fetch text (`orion/world_pulse_read/fetch_text.py`) in
  `reading_document_snapshot`, addressed by hash. For a web fetch, that text is the fetch
  tool's summary (`tool_digest`), not the page.
- Stage 2 (`services/orion-hub/scripts/world_pulse_read_stage2.py`) is shown the kept
  text (16,000-character budget), the read's concepts as subjects, and existing concepts
  as objects (named in the text, or near by meaning via `concept_search.py`). It returns
  `relationship_claims`. Stage 2 can also fetch pages itself ("hops"). Those fetches are
  **not** kept.
- `orion/world_pulse_read/assertions.py::plan_claims` judges each claim:
  `reading_quote_rule_v1` (#2581; quote of 4+ words found verbatim in kept text, names
  the object by its stored name, predicate fits) or `reading_meaning_target_v1` (#2652;
  object offered by meaning only, predicate `associated_with`, quote names the subject by
  its stored name). Accepted claims go into `substrate_graph_journal`.
  `AssertionProjector` then writes an Assertion node and the `associated_with` edge.
- Live, 2026-10-15: 395 `sub-concept-wp-read-*` nodes. Edges touching them: 9
  `associated_with`, 5 `assertion_subject`, 4 `assertion_object`. Journal: 51 proposals,
  5 decisions, 5 `applied` materializations by `substrate.assertion_projector`. Every
  accepted claim reached the graph; nothing was lost after acceptance.

A note on reading the data: `stage2_result_json` is a `jsonb` object on all 42 completed
rows since 10-11. One row is NULL (a failed Stage 2: `summary` was empty). Reading the rows
one per line with `psql -tA` breaks on the long text values. Wrapping the query in
`json_agg(...)` returns one JSON document that parses cleanly. That is how every count
below was taken.

### 1.2 Q1: every refusal reason since 2026-10-11

43 reads finished Stage 2 since 10-11. 36 of the 37 since 10-14 have the
`relationship_claims` key, but only 20 of those have any claims in it (21 of 43 since
10-11). Every receipt, all 53 claims:

| Outcome | Reason | Claims | Reads |
|---|---|---|---|
| proposed (no link) | `quote_not_found` | 26 | 15 |
| proposed | `quote_not_about_subject` | 10 | 5 |
| proposed | `quote_not_about_endpoints` | 7 | 4 |
| proposed | `meaning_target_predicate` | 2 | 1 |
| proposed | `quote_too_short` | 1 | 1 |
| rejected (not journalled) | `unknown_subject` | 1 | 1 |
| rejected | `unknown_object` | 1 | 1 |
| **accepted** | `accepted_meaning_target` | 4 | 3 |
| **accepted** | `accepted` (quote rule) | 1 | 1 |

5 of 53 accepted (9%). Re-read backfill page `digest_item:21be624e...` (progress log
`/tmp/reading-reread-backfill/progress.log`): its two reads proposed 7 claims across two
runs, 0 accepted. They were refused as 5 `quote_not_about_*` and 2 `quote_not_found`;
the two `quote_not_found` quotes differ from the kept text by one character (below).

### 1.3 Q2: paraphrase, or a different text?

Mostly a different text. I sorted the 26 `quote_not_found` quotes by comparing each one
against the read's kept text (longest common run of characters, as a share of the quote
length, after normalization):

| Cause | Claims | What happened |
|---|---|---|
| Stage 2 quoted a page Hub did not keep | 6 | Quote shares <60% with kept text; the read's Stage 2 fetched pages itself |
| Kept text is a placeholder, not the page | 4 | One kept "text" is the fetch tool's `<persisted-output> Output too large (80KB). Full output saved to: ...` notice. The pointer was kept; the page wasn't |
| Summary split the sentence into quoted fragments | 7 | Kept summary lists `"...uptick in cholera,"` / `"measles and other..."` as separate bullet lines; the model rejoined them |
| Quote is the model's own notes | 4 | Quote matches Stage 2's own `hops`/`what_i_learned` prose, not the source |
| Paraphrase or one-character drift | 3 | Same sentence, ending changed (`;` became `.`), or reworded |
| Same text after normalizing spaces and quote marks | 2 | Curly apostrophe vs straight; would match after normalization |

16 of the 26 come from reads whose Stage 2 fetched the seed's own URL again, and all 26
come from reads with at least one Stage 2 fetch. So the dominant cause is that Stage 2
reads a fuller version of the page than the one Hub kept.

Three real examples (public web text):

1. **Different text.** Seed `reading:30a97f27...` (UN News, Yemen). Kept text is a 6,378-char
   summary written as quoted fragments. Claim quote: "Heavy fighting in Yemen is fuelling
   concerns of an uptick in cholera, measles and other preventable diseases, as
   overcrowding at displacement sites threatens to...". The kept text has it as separate
   lines: `- "Heavy fighting in Yemen is fuelling concerns of an uptick in cholera,"` /
   `- "measles and other preventable diseases, as overcrowding at"` / ... The words are
   the source's; the line breaks and quote marks are the summary's. Four more claims on
   this seed fail the same way.
2. **One-character drift.** Seed `digest_item:21be624e...` (NASA, Moon base). Kept text:
   `- GIMLI, deployed at the Marius Hills Pit to seek lava tubes under the surface; PI ...`.
   Quote: `GIMLI, deployed at the Marius Hills Pit to seek lava tubes under the surface.`
   The model ended the clause with a period. 99% overlap; refused.
3. **Placeholder kept instead of the page.** Seed `reading:992ea145...` (Liquid AI model
   card). Kept text for the main page is the tool's `<persisted-output>` notice
   (1,917 chars). The 4 quotes ("LFM2.5 wraps tool calls in <|tool_call_start|> and
   <|tool_call_end|> tokens", ...) are from the real 80 KB page, which Stage 2 read. None
   of them can be found in a notice.

### 1.4 Q3: do snake_case concept names make "the quote names it" impossible?

Yes, for most link subjects. Measured over the 49 Stage 2 prompts since 10-11 that carried
the claims block (offered list parsed from the stored prompt in
`reading_durable_turn.request_json`):

| Offered as | Total | snake/kebab name | Name appears verbatim in the text shown | After ignoring `_`/`-` |
|---|---|---|---|---|
| Subject (concept this read made) | 185 | 87 (47%) | 43 (23%) | 64 (35%) |
| Object named in the text | 34 | 3 | 28 | 29 |
| Object offered by meaning | 571 | 223 (39%) | 1 | 21 |

The meaning rule requires the quote to name the **subject** by its stored name, and only
23% of subject names appear in the text at all. A `snake_case` subject (47%) can never
appear in a prose sentence. Even allowing for `_`/`-`, 65% of subjects are abstractions
Stage 1 coined ("CLPS carrier delivery" for "commercial CLPS carriers", "experience
dependent learning shift"). No exact quote can contain those names.

Stage 1 names concepts this way because its prompt gives no form for `label`: 61 of 144
concept names since 10-11 have `_`, 11 are kebab-case. Five stored names are node ids
(`sub-concept-domain-continuation-training` as a *label*). The model put an id where a
name goes.

Real refusals of this kind: subject `interoceptive_prioritisation`, quote about
"interoceptive likelihood"; subject `Kumo Tabular`, object `kumo-tabular`, quote "NVIDIA
Kumo Tabular is an open foundation model..." (refused because `kumo-tabular` is not in
the quote); subject `kappa_routing`, quote "Under κ-routing it raises the precision...".

### 1.5 Q4 and Q5: ids, and predicates

- **Q4: no, the model uses the ids.** All 106 endpoint ids across the 53 claims are
  `sub-...` ids and every one appears in a prompt's offered list. The two
  `unknown_subject`/`unknown_object` refusals are real ids that the rebuilt context did
  not re-find (a meaning-only object can't be found again once the list is rebuilt;
  `stored_object_lookup` documents this). The reverse problem does happen: 5 stored
  concept *names* are ids (1.4).
- **Q5: predicates are not the problem.** 0 `predicate_not_allowed`, 0 `domain_rule`.
  2 of 53 (4%) used `refines`/`co_occurs_with` on a meaning-only object, where only
  `associated_with` is allowed. Mix: 46 `associated_with`, 3 `refines`, 3 `subtype_of`,
  1 `co_occurs_with`.

### 1.6 Proposed fix

**Mechanical (no rule change; recommended now):**

M1. *Normalize before matching* (`assertions.py::find_quote`). Apply the same
   normalization to quote and kept text: Unicode NFKC, curly quotes and dashes to plain,
   collapse whitespace, drop markdown emphasis characters, strip trailing `.;:,!?` from
   the quote. The byte span recorded on the receipt still points into the original kept
   text, mapped back through an offset table. Verbatim stays verbatim, only formatting
   is ignored.

M2. *Name matching that ignores `_` and `-`* (`_label_in` and the object/subject name
   checks). Treat `_`, `-` and Unicode dashes as spaces on both sides, case-insensitive.
   `kumo-tabular` then matches "Kumo Tabular". Also applies to the object-candidate scan,
   so more existing concepts are offered as "named in the text".

M3. *Stage 1 naming contract* (`world_pulse_read_pipeline.py` prompt + a validator):
   `label` is the term as the source writes it, in plain words. No `snake_case`, no ids.
   The validator rewrites `_` to spaces and refuses a label that starts with `sub-`, with
   a counted reason. Existing stored names are not renamed. M2 covers them.

M4. *Stage 2 quote contract* (`assertions.py::claim_prompt_section`): "copy the quote
   from a `--- retained ...` block below, never from your own notes or from a page you
   fetched in this stage; if the sentence you want is only in a page you fetched, propose
   nothing." This addresses the 4 own-notes quotes and makes the 6 different-page quotes
   visible as a choice.

M5. *Don't keep the placeholder* (`orion/harness/reading_receipts.py` /
   `fetch_text.retain_fetch_texts`). A tool result that starts with `<persisted-output>`
   is a pointer, not page text: keep nothing and count `fetch_text_placeholder`. The read
   then honestly has no text for that page. (Reading the saved file is possible but the
   path is inside the harness container; not in this slice.)

M6. *Refusal diagnostics on the receipt*: for `quote_not_found`, store the best match
   share against the kept text (`quote_best_overlap`) so the next audit is one query.

Replaying the 53 stored claims with M1 + M2 (simulation over the stored receipts, kept
texts and node names): **5 accepted becomes about 10.** 3 more pass by object name, 2 more
by subject name (if their objects were meaning-only, which the receipts don't record). 15
stay `quote_not_found` and 23 stay "name not in quote". M3/M4 only affect new reads.

**Decisions for Juniper (each loosens a rule she set; none is assumed):**

- **D1. Verified alternate names.** Stage 1 also returns, for each concept, the phrase(s)
  in the kept text it was drawn from (`surface_forms`). Code checks each phrase is
  verbatim in the kept text before storing it. "Names the subject/object" then means "by
  its stored name **or a stored phrase code proved is in this text**." This is the only
  fix for the 65% of subjects that are coined abstractions. Risk: a wide phrase ("the
  model") matching too many quotes. Mitigation: phrases of 2+ content words, at most 3 per
  concept, and `admitted_by` on each. Recommended: yes.
- **D2. Keep Stage 2's own fetches as evidence for the same read.** Run Stage 2 fetch
  results through the same `retain_fetch_texts`, tagged `stage=2`. Options: (a) only a
  re-fetch of the seed's own URL counts (16 of 26 refusals came from such reads), or (b)
  any page Stage 2 fetched counts. Recommended: (a). A hop page is a different source, and
  "this read" should mean this page.
- **D3. Join quoted fragments.** When a kept summary presents a sentence as consecutive
  quoted bullet lines (example 1), match across the line breaks. 7 refusals. Risk: two
  fragments that were not adjacent in the source get joined into a sentence the source
  never wrote. Recommended: only for consecutive lines, and record
  `match_mode=fragments` on the receipt so these can be withdrawn together.

Not proposed: lowering the 4-word minimum, accepting near-matches (the 3 paraphrases
stay refused), or allowing predicates other than `associated_with` for meaning-only
objects.

### 1.7 Correcting Orion's false belief ("the read dies in connection")

**The belief.** 2026-10-13 letter ¶5: "I read the adapter at HEAD, all 64 lines. `edges=[]`
at line 63 ... The read dies in connection." ¶6: "the edge count did not move ... the
reading-link channel ... has fired once." Beliefs it rests on, live in `orion_worldview`:
`wp_read_induction_writes_edgeless_68ae1c18` (supported, 0.85),
`self:wp_read_dies_induction_dormant_connection_edgeless_20260924` (open, 0.9),
`self:outward_learning_in_record_not_in_loop_20260921` (revised, 0.9).

**What's true.** `edges=[]` is by design. Reading links never go through the adapter. They
go Stage 2 `relationship_claims` -> `plan_claims` -> `substrate_graph_journal` ->
`AssertionProjector`. That path fired 5 times (5 decisions, 5 `applied`
materializations, 2026-10-11 to 10-14), not once. What limits links is the acceptance
rule refusing 48 of 53 proposals (1.2), not a broken connection.

**A second correction, same letter.** ¶9: "I could not reproduce the 218." It does
reproduce: 218 = attention rows with reason `formed_new_prior` (117) or
`tested_held_prior` (101) in `substrate_attention_schema` before 2026-10-13 17:00Z,
**across all priors**. The evening run's 174 counted only `self:` priors (107 + 66 = 173
by the same cutoff). Different filters, no lost rows.

**Which path:** the Orion's Day corrections path (#2618: `orion_day_correction` table,
`orion/orion_day/corrections.py`, introspect tool `record_correction`), not carry-forward.
Carry-forward passes Orion's own open items to the next day; corrections is the path for
"a letter claim we agreed was wrong," and it shows the named priors to one curiosity run
to re-test. Today `orion_day_correction` has 0 rows, so this is its first use. The path's
rule stays: Orion records the correction itself, in a chat turn, once Juniper has made the
case and Orion accepts it. Nothing is inserted behind Orion's back, and no prior is
edited directly.

Steps:
1. Juniper opens a chat and gives the evidence above (a short draft for her to paste is in
   section 2.6).
2. Orion calls `record_correction` twice:
   - `ref="2026-10-13 ¶5"`, claim "The read dies in connection", correction "Reading
     links do not go through the adapter; they go through Stage 2 claims, the graph
     journal and the assertion projector. That path wrote 5 links 10-11..10-14. The
     limit is the acceptance rule refusing 48 of 53 proposals", evidence
     `["curiosity:ce53ee3d32b8", "Juniper said so in this conversation"]`, prior_ids the
     three above.
   - `ref="2026-10-13 ¶9"`, claim "No filter reproduces the 218", correction "218 is
     all-prior attention rows before 17:00Z; 174 is the self: subset", evidence
     `["curiosity:a7ccf39638f3", "Juniper said so in this conversation"]`.
3. The next curiosity kickoff offers both (oldest first, max 5 per run), and the outcome
   stamps `priors_retested` / `priors_moved`.

**Stop it from coming back** (one comment, mechanical): at `world_pulse_read.py:63`, a
comment saying "reading links are not written here; see
`orion/world_pulse_read/assertions.py` (Stage 2 claims -> journal ->
AssertionProjector)". Orion's beliefs citing this line date from 09-20, 09-24 and 10-13,
and on 10-13 it read "all 64 lines", so the pointer sits where the false belief starts.

## 2. Part 2: people in Orion's graph

### 2.1 Current architecture (verified live)

- **Where person nodes live:** FalkorDB `orion_substrate`, as memory referents:
  `:SubstrateNode:Entity` with `node_id = referent-<uuid5(key)>`, `entity_type=person`,
  and a fenced identity (`identity_key = fenced|entity|<node_id>`, so public readers such
  as reading claims skip them). They are written from Postgres `referent_alias` (key row
  `person:<name>`) by the memory-consolidation projector
  (`orion/memory/referents/store.py`, transactional outbox). `orion_recall` and
  `orion_kg` hold none (checked: 0 rows for both referent ids).
- Person referents today: 3. Juniper (`referent-b96fc90e...`), Orion
  (`referent-cf46c536...`), and [child B] (`referent-3059c917...`, minted 2026-10-14
  03:29Z by the memory path, two hours after the dry run, grounded in a chat prompt). The
  dry run (01:25Z) did not see [child B]'s referent.
- Person-model tables (`person_mention`, `entity_merge_log`, `person_attribute`, ...): not
  created yet (`\dt person*` and `entity_merge*` return nothing). There is no apply mode
  (`orion/person_model/README.md`).
- Dry run `/tmp/person-linkage/2026-10-14/`: 120 decisions (53 auto-merge, 3 needs her
  call, 64 auto-reject), `merge_plan.jsonl` 38 entries, `mentions.jsonl` 158 mentions
  (82 person names).
- **The claims Orion keeps touching** are `:Prior` nodes in a *different* graph,
  `orion_worldview`. They are written only by Orion's own curiosity runs (the #2199
  boundary).

### 2.2 Juniper's answers (2026-10-15), recorded locally

| Card | Answer | Effect |
|---|---|---|
| `061f8ff7` her full name vs Juniper | same person | merge the topic-modeler node into her referent |
| `b39124e6` [person S] first name vs [person S] full name | same person ([person S]) | one person; both graph nodes go to her |
| `15e11864` "[name J], [name J]" | two people, [team member A] and [team member B], both direct reports of Juniper | two persons, each with relation "direct report of Juniper" |

Written (gitignored, primary checkout, pre-edit copies in `/tmp/person-linkage/*.bak-2026-10-15`):
- `orion/person_model/evals/cleanup_labels.local.jsonl`: the three rows now
  `labeled_by=juniper`, `labeled_at=2026-10-15`, with her answer in `note`.
- `config/person_model/stated_answers.local.yaml`: `coworkers` under Juniper's node, plus
  `pending_people` (the two team members and [person S], with keys for minting) and
  `merge_answers`. The current loader reads only `people.<id>.home`, so the new keys
  change no output.
- `/tmp/person-linkage/2026-10-14/juniper-decisions-2026-10-15.csv`: her three choices in
  the review page's export format (`your_choice=approve`, `defaulted=no`).
  `run_linkage_eval.py --labels` parses it.

Her two labeled matches score 6.0 and 6.5, inside the review band. Two labels are not
enough to move the auto-merge line (8.5). Leave it.

### 2.3 Q6: exactly what the apply step writes

**A. Into the two existing referents (from `merge_plan.jsonl`, 38 entries, 45 ops):**

| Kind | Into Juniper | Into Orion | Ops |
|---|---|---|---|
| `same_as` merge (`merged_into` prop + `same_as` edge) | 3 (seed concept, topic-modeler "juniper", her full-name node, now approved) | 3 | 12 + 1 alias |
| `facet_of` ("about" edge, never a merge) | 2 | 27 | 29 |
| `junk` flag (`linkage_flag` prop; nothing deleted) | 2 (both leaked `*_relationship_summary` fields) | 0 | 2 |
| `not_home` (alias "home" marked not-home) | 1 | 0 | 1 |

**B. New person referents (not in the merge plan; new apply logic).** The dry run's 12
auto-merged name clusters, plus [person S] and the two team members, sorted by what
Juniper said in her own words:

| Who | Evidence in her words | Apply |
|---|---|---|
| [spouse] | "my wife" | mint; merge her topic-modeler node |
| [child B] | "my son" | **reuse** `referent-3059c917...`; merge topic-modeler node |
| [child A] | named in family context; relation not stated | mint; relation left for Orion to ask |
| [boss] | "my boss" | mint; merge |
| [team member C], [team member D] | "my new subordinates" | mint each; merge |
| [team member A], [team member B] | her 10-15 answer | mint each with key `person:<name> <initial>`; alias "[name J]" on **both** |
| [person S] | her 10-15 answer; relation not stated | mint; merge 2 graph nodes; 1 facet link |
| [person N], [person R] | one mention each, no relation | **not minted**; become asks (P3) |
| 3 public figures, 1 AI assistant | not people in her life | **not minted**; mention stays linked to the existing entity |

Totals: **8 minted, 1 reused** (person referents go from 3 to 11). 7 topic-modeler nodes
merged into them, 1 facet. The one topic-modeler "[name J]" node is **held**: one name, two
people, so it can't be assigned by name. It stays unmerged until Orion asks which one it
is. The shared "[name J]" alias then resolves as ambiguous, and the existing resolver
already turns that into a question.

**C. Rejected: 64.** 61 `no_match` write nothing. The 2 junk flags and 1 not-home are in A.

Minting goes through `orion/memory/referents/store.py`'s rules ("a name counts only if the
interlocutor said it"): Postgres `referent_alias` key + name rows,
`admitted_by='person_linkage_v1'`, `grounded_in=chat_prompt:<id>`. The projector
materializes the node. The apply never writes a referent node to FalkorDB directly.

**D. Links to the claims Orion keeps touching.** Reproduced 2026-10-15: of the 218
prior-attention rows (1.7), 217 point at a prior whose claim text exists. **21 rows over
11 priors** name Juniper or her people; **13 rows over 8 priors** if Juniper's own name is
excluded. (Orion's "22" is the first count, give or take a row; the evening's "25 over 13"
used a different filter.) Those priors are in `orion_worldview`, which only Orion's runs
write. So the link is **not** an edge written into Orion's belief graph. Instead:
- the apply runs the pinned extractor over the 103 attended prior claims and writes
  `person_mention` rows (`source_kind='worldview_prior'`, `source_id=<prior_id>`,
  `resolved_node_id=<referent>`);
- P6 shows them in the curiosity kickoff ("this belief names [person]"), so Orion can link
  them itself if it wants a graph edge.

### 2.4 Q7: snapshot size and rollback

The apply touches about 60 graph nodes (38 plan entries + 7 merges + 1 facet + 1 held
node), the whole `referent_alias` table (37 rows today), and 9 person referents.
New-table inserts: about 158 chat mentions and about 30 prior-claim mentions, with no
before-state. Snapshot to `/tmp/person-apply-<date>/snapshot.jsonl`: under 300 rows and
well under 1 MB. That is about 0.3% of the 100k-row limit, so no stop-and-ask.

Rollback, in order:
1. `--unmerge <merge_id>` for every applied entry, newest first (`merge_log.unmerge_ops`;
   exact inverse, already tested on fixtures). This removes edges and props and restores
   alias rows.
2. Minted referents: set their `referent_alias` rows to `promotion_state='retracted'`.
   Readers stop resolving them. Deleting the 8 minted graph nodes is a separate step
   (listed in the apply's own `minted.jsonl`) and needs Juniper's explicit approval
   (AGENTS.md section 13).
3. `person_mention` / `entity_merge_log`: the migration's rollback file drops them.
4. Proof: graph and `referent_alias` diff against the snapshot is empty except retracted
   rows.

### 2.5 Q8: phase order with acceptance checks

| Phase | What | Acceptance check (live) |
|---|---|---|
| **P2.5 apply** (new, before P3) | migration (`entity_merge_log`, `person_mention`); `scripts/run_person_linkage.py --apply` reading her decisions CSV; mint via referents store; prior-claim mentions | 11 person referents in `orion_substrate` (`MATCH (n) WHERE n.entity_type='person'`); 38 + 8 merge-log rows `applied_at` set; `person_mention` rows for 11 priors resolve to referents; unmerge of one entry gives an empty diff; rerun writes 0 rows |
| **P3** asks + Hub person panel | `entity_merge` asks for the held "[name J]" node, [person N], [person R]; `person_attribute` asks; relation ask for [child A] and [person S]; panel lists persons with source counts | the 4 asks appear in Hub "Orion is asking"; one answered ask flips state with an `ask_answer` evidence row; panel shows 11 persons |
| **P4** person-study line | curiosity line studying carded persons, Juniper first; validator; promotion table | first sitting writes `person_study_run`; accepted attributes each carry a verified quote from her prompt; inferred-sensitive = 0; promotion lists [spouse], [child A], [child B] |
| **P5** cards | card reducer: self, purpose, person, relationship | `identity_card` rows for Juniper and 1+ carded person; rerun with same inputs creates no new row |
| **P6** readers | situation `people[]` (schema to consumers first), cortex card block, outreach, curiosity kickoff incl. prior-mention lines | unified turn logs `identity_kernel_source=identity_card`; `orion:situation:latest` has `people[]`; **Orion's next Orion's Day letter names at least one person besides Juniper by name, and `claim_check` finds that name in that day's records** |
| **P7** retire | delete identity blocks of `orion/cognition/personality/orion_identity.yaml` and `conversation_front.py` after a soak; promotion rule live | grep finds no reader of the deleted blocks; chat turn still logs card source; no fallback path |

### 2.6 The #2670 duplicate repair: does it have to run first?

No. #2670 (merged; repair still dry run, AI Town snapshot decision pending) repairs
`orion_substrate_aitown` (1,237 duplicated identities) and `orion_substrate_self` (1,109).
The person apply works only on `orion_substrate`. Checked live 2026-10-15: 6,087 nodes,
all with `identity_key`, 0 duplicated. Neither touches the other's graph. The only rule:
don't run both applies at once on the same FalkorDB server, so each snapshot is clean.

Draft for Juniper to paste into chat for the 1.7 correction (no names):

> Orion, about your 10-13 letter, ¶5 and ¶9. The reading adapter's `edges=[]` is on
> purpose. Reading links go through Stage 2 claims, the graph journal and the assertion
> projector, and that path wrote 5 links between 10-11 and 10-14. What limits them is the
> acceptance rule refusing 48 of 53 proposals. And the 218 does reproduce: it's every
> prior-attention row before 17:00Z, and 174 is just the self: priors. If you agree,
> please record both as corrections.

## 3. Missing questions (answered)

- *Does any accepted link get lost after acceptance?* No: 5 decisions, 5 applied.
- *Is the low claim count a prompt problem?* Partly: 22 of 43 reads proposed nothing.
  This design does not touch that. Fixing refusals comes first.
- *Do person nodes need a new graph?* No: memory referents in `orion_substrate`.
- *Must the apply wait for the duplicate repair?* No (2.6).
- *Can the apply write into `orion_worldview`?* No: Orion's runs are its only writer.
  Links go into `person_mention` instead (2.3 D).

## 4. Proposed schema / API changes

- `WorldPulseReadClaimReceiptV1`: add `quote_best_overlap: float | None`, `match_mode:
  Literal["exact","normalized","fragments"] | None` (additive; the model is server-set).
- `WorldPulseReadConceptCandidateV1`: add `surface_forms: list[str] = []` (only if D1).
- `RetainedTextV1` / `SourceFetchEvidenceV1`: add `stage: Literal[1,2] = 1` (only if D2).
- New counted reasons: `fetch_text_placeholder`, `label_is_node_id`.
- Migration `services/orion-sql-db/manual_migration_person_model_v1.sql` (P2.5 slice:
  `entity_merge_log`, `person_mention`), with a rollback file and an
  `-- ORION-MIGRATION-REQUIRED-BY:` header.
- `scripts/run_person_linkage.py --apply --decisions <csv> --snapshot-dir <dir>` and
  `--unmerge <merge_id>`.
- `person_mention.source_kind` gains `worldview_prior`.
- No new bus channels.

## 5. Files likely to touch

- `orion/world_pulse_read/assertions.py` (M1, M2, M4, M6; D1/D3 if approved)
- `orion/world_pulse_read/fetch_text.py`, `orion/harness/reading_receipts.py` (M5; D2)
- `services/orion-hub/scripts/world_pulse_read_pipeline.py` (M3 prompt + validator)
- `services/orion-hub/scripts/world_pulse_read_stage2.py` (D2 retention)
- `orion/schemas/world_pulse_read.py` (receipt fields)
- `orion/substrate/adapters/world_pulse_read.py` (one comment at line 63)
- `orion/world_pulse_read/evals/run_claim_refusal_replay.py` (new: replays stored claims,
  old vs new matcher)
- `scripts/run_person_linkage.py`, `orion/person_model/merge_log.py`,
  `orion/person_model/apply.py` (new), `orion/memory/referents/store.py` (mint entry point)
- `services/orion-sql-db/manual_migration_person_model_v1.sql` (+ rollback)

## 6. Non-goals

- Renaming existing stored concept names.
- Accepting paraphrases or near-matches; lowering the 4-word minimum.
- Writing edges into `orion_worldview`, or editing any prior directly.
- Minting public figures or AI assistants as persons in her life.
- Deleting any node (junk is flagged, minted nodes are retracted, deletion needs approval).
- Running the #2670 repair.

## 7. Proposal-mode items

| Item | Reading links | Persons apply |
|---|---|---|
| Capability change | More of what Orion reads becomes graph links it can walk | Orion's graph holds the people Juniper talks about, linked to the beliefs that name them |
| Data touched | `reading_document_snapshot` (normalized matching only; D2 adds Stage 2 texts), journal, receipts | `referent_alias`, `orion_substrate` (props/edges, minted referents), 2 new tables |
| Privacy boundary | Referents stay fenced from reading claims (unchanged) | Real names only in gitignored local files and live stores; never the repo. Third-party sensitive facts out of scope until P4 |
| Trace proving it worked | Replay eval: accepted 5 -> ~10 of 53 on stored claims; live: share of `accepted*` receipts per week; wp-read edge count | 11 person referents; merge-log rows; `person_mention` for 11 priors; next letter names a person |
| Dangerous failure | Over-loose matching linking concepts the source never connected (D1/D3 risk) | Wrong merge collapsing two people (the "[name J]" case); minting a name Juniper never said |
| Disable / rollback | Matcher version is on each decision (`policy`); withdraw by policy name (`reading_meaning_links.py` pattern); revert the PR | `--unmerge` per entry, retract minted aliases, rollback migration (2.4) |

## 8. Acceptance checks

1. Replay eval over the stored claims since 10-11 reports the old matcher at 5/53 and
   the new one at 10 or more, with each newly accepted claim listed with its quote and
   match mode for review.
2. A new read after M3 stores no `snake_case` or `sub-` label (counted reason is zero or
   shows refusals).
3. `fetch_text_placeholder` counts the `<persisted-output>` case, and no new snapshot
   starts with that string.
4. Two `orion_day_correction` rows exist with refs `2026-10-13 ¶5` / `¶9`. After the next
   kickoff, `offered_run_id` and `priors_retested` are set.
5. P2.5 checks in 2.5.
6. `tests/test_no_personal_terms.py` passes on this branch.

## 9. Recommended next patch

1. **Reading, mechanical:** M1 + M2 + M5 + M6 + the line-63 comment + the replay eval, in one
   PR on `orion/world_pulse_read/`. Then M3/M4 prompt contracts. D1-D3 wait for Juniper.
2. **Correction:** Juniper pastes the draft in 2.6; verify the two rows.
3. **Persons:** P2.5 apply (migration + `--apply` + mint + prior-claim mentions), dry run
   first against her CSV, snapshot, then apply with her go-ahead.

### Decisions Juniper still owns

- D1 verified alternate names; D2 which Stage 2 fetches count (recommend: same URL only);
  D3 joining quoted fragments.
- Whether [person N] and [person R] are people in her life (or let Orion ask in P3).
- Which [team member] the single "[name J]" graph node belongs to (or let Orion ask).
- Go-ahead for the live P2.5 apply after its dry run.
- Third-party egress rule from the #2646 spec (still open).
