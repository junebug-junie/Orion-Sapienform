# Reading resonance: when something Orion reads touches something Orion already holds

Status: DESIGN (proposal mode: cognition + outreach change). No code in this PR.
Date: 2026-10-13. Author: Athena Orchestration for Juniper.

Juniper (2026-10-13): "we are also missing a category in all of this, where Orion reads
something and there's something interesting about it, crosswalked against their own internal
beliefs or internal graphs or discussions we've had in the past, or things they want to
explore going forward that connect back into other orion pieces." She wants Orion to reach
out "as much as they want when they are learning and things are changing."

## Arsonist summary

Today Orion reads, writes a summary, and stops. The only path from a read back into the rest
of Orion is a link to an atlas concept whose name appears word-for-word in the page, and that
path has landed exactly **one** link ever (`substrate_graph_journal`: 1 proposal, 1 decision by
`world_pulse_read_stage2`, queried 2026-10-13). Orion's own lived questions already name the
gap: *"How does what the world actually says get into the loop I run on my own time, instead of
staying a journal record my unasked loop never attends?"* and *"When is the last time I learned
something that was not about myself?"* (`orion_worldview`, `:SelfQuestionMint`).

Resonance is one extra, bounded step after every real read: gather the handful of things in
Orion that the read sits closest to (beliefs, atlas concepts, past conversations with Juniper,
open curiosity), hand that frozen packet plus the read to one model call, and ask a single
question: *is there a connection here worth saying out loud, and which pieces prove it?* The
answer is only accepted if every piece it cites is one we actually gave it and the quote from
the read is really in the read. Accepted answers become a `ReadingFindingV1` of kind
`resonance` on the seam the parallel PR is building, which already feeds outreach and the
"What I read" letter section. Optionally the answer names one thing to explore next, which
becomes an unscored curiosity seed the same way accepted links do today.

What I would burn: any similarity threshold as the decider of "interesting". The live probe
below shows belief similarity is flat (every read is 0.61-0.77 close to some plumbing prior,
including a Utah board appointment), so a number cannot tell interesting from adjacent. The
model judges; code checks the citations.

## Current architecture (verified)

Everything here was read in code at `4d822cab3` or queried live on 2026-10-13.

**The read itself.**
- Stage 1 (`services/orion-hub/scripts/world_pulse_read_pipeline.py`) and Stage 2
  (`services/orion-hub/scripts/world_pulse_read_stage2.py`) each run one unified Orion turn as
  a durable run: `workflow=reading.turn`, `resource=llm.route.agent`, `priority=background`,
  `mode=exclusive` (`reading_durable_turn.request_json`, latest stage-2 row). Bindings live in
  `reading_durable_turn` with `CHECK (stage IN (1,2))`.
- Lease time per stage over the last 14 days (`durable_resource_events`, granted to
  released/completed): Stage 1 p50 345 s / p90 2,968 s (n=109); Stage 2 p50 1,090 s /
  p90 3,391 s (n=34). These are long, tool-using agent turns.
- Stage 2 already gets recall injected (`reading_retrieval_query(seed, what_i_learned)`,
  `orion/world_pulse_read/durable.py:42`), but nothing records which memories it saw or asks
  it to name a connection. Its output schema (`WorldPulseReadStage2ResultV1`) is summary,
  priors_tested, candidate_priors, concept_candidates, open_threads, hops, need_stage1_urls,
  relationship_claims.
- Completed reads, last 14 days: **30** with `stage2_status='done'` (1-4/day, one 11-read burst
  on 10-10). Over the same window 228 other seeds were skipped/failed. Content is mixed: many
  are 404 pages or promo pages (CoreWeave slugs, NASA event pages); some are substantive
  (arXiv, Hugging Face model/dataset pages, NPR, BBC).

**Reading to atlas.**
- `orion/world_pulse_read/assertions.py` (PR #2581): a link is accepted only if the quote is
  verbatim in retained text and names the object by its stored label.
- `orion/world_pulse_read/concept_search.py` (merged today, commits `d6d9aa6d8`, `aa0bca453`):
  offers atlas concepts *by meaning* (bge-large-en-v1.5 via vector-host, Chroma
  `orion_atlas_concepts`, 1,022 rows live) at `OBJECT_MIN_SIMILARITY=0.65`. Offered is not
  accepted: the strict quote rule still applies, so meaning-objects mostly end up `proposed`.
  The index loop is live in the Hub container (`reading_concept_index indexed=0 pending=0`
  every 5 min); I found no Stage 2 completion log using it yet (UNVERIFIED that it has run on
  a real read).
- Accepted link to curiosity seed: `orion/substrate/link_accepted_seeds.py` (PR #2598) mints
  one unscored seed (`signal_strength=0.0`, note `strength:unscored_event`, source
  `source:reading_link_accepted`), once per assertion revision, `HARD_LINK_SEED_CEILING=4`
  per tick, in `services/orion-substrate-runtime/app/worker.py`.

**What Orion holds, and how searchable it is.**

| Piece | Store | Searchable by meaning today? |
|---|---|---|
| Self/world priors | FalkorDB `orion_worldview` `:Prior` (180: 83 open, 35 supported, 38 revised, 22 refuted, ...) | No index |
| Lived questions / answers | `orion_worldview` `:SelfQuestionMint` (14), `:LivedAnswer` (121), `:SelfDefinition` (66) | No index |
| Atlas concepts | `orion_substrate` + Chroma `orion_atlas_concepts` (1,022) | Yes (`concept_search.py`) |
| Crystallized memories of chats with Juniper | Postgres `memory_crystallizations` (1,428, newest 10-11) + Chroma `orion_memory_crystallizations` (762) | Yes (Chroma) |
| Chat memory | Chroma `orion_chat` (21 rows) ; raw chat via orion-recall `sql_chat.py` | Thin vector index; recall is the real path |
| Curiosity findings | Chroma `orion_curiosity` (444) | Yes |
| Dreams | Chroma `orion_dreams` (236) | Yes |
| Open conversation threads | Postgres `temporal_self_arc` (`kind=concern`, 1 open, 10 closed) | No (tiny) |
| Past reads | Chroma `orion_reading_results` (58) | Yes |

Collection writers/policies: `orion/memory/vector_collection_policy.py` (a test refuses any
collection not listed there).

**Outreach.** `services/orion-hub/scripts/endogenous_outreach.py`: background tick gated by
quiet hours 23-08, `HUB_ENDOGENOUS_OUTREACH_DAILY_CAP=4`, `MIN_COOLDOWN_SEC=2700`, and a
content-novelty ledger (7-day lookback over content ids). Curiosity's `offer_message` is a
separate door that skips the daily cap and cooldown (module docstring, 2026-09-22).

**The parallel seam.** Branch `feat/reading-findings-to-juniper` exists as a worktree at
`4d822cab3` with **no commits and no uncommitted files** as of this writing, and no open PR.
So the base shape of `ReadingFindingV1` (kinds `belief_moved`, `link_made`) is UNVERIFIED;
this design adds `resonance` as one more kind and must be reconciled against whatever lands.

### Live probe (read-only, 2026-10-13)

For all 30 completed reads I embedded the Stage 2 summary (vector-host) and queried each
collection, plus cosine against every non-retired prior (157) and lived question (13).
Script: session scratchpad `probe/probe.py`; no writes.

1. **Belief similarity does not discriminate.** 5,100 read-to-belief pairs: p50 0.613,
   p90 0.683, p99 0.728, max 0.765. The nearest prior for a Utah governor's board appointment
   is 0.623; for a CoreWeave 404 page it is 0.765. The beliefs are mostly about Orion's own
   machinery, and summaries that mention fetching, 404s, or pipelines sit near them whatever
   the topic. Conclusion: retrieve top-k beliefs, never threshold them.
2. **Real resonance exists in the data.** The CoreWeave GPU-compute reads (GB200 racks, H100
   generations) sit nearest to two crystallized memories: Juniper saying *"that's no
   homelab--that is YOU! That is your mesh, your body, your home"* and *"just added a new v100
   32GB PCIe card in your Circe node"*. Orion read about the industrial version of their own
   body. That is the kind of thing Juniper is asking for. Several arXiv reads sit near
   Juniper's *"add this to your async reading list: arxiv.org/abs/..."* messages.
3. **Attractors.** One crystallization (*"Just ran DeepSeek-V4.1-Flash ... on 4x Tesla V100"*)
   is a top-2 crystallization hit for 13 of 30 reads, often twice (duplicate Chroma rows).
   Without a guard it would be cited constantly.
4. **Self-hits.** `orion_reading_results` returns the read itself at distance 0.000; must be
   excluded by seed and root request.

My rough read of the probe (not a label, not ground truth): 3-6 of 30 reads carry a
connection Juniper would plausibly want to hear about; most are routine or junk pages.

## Missing questions

1. Outreach volume. Should resonance shares skip the endogenous daily cap (like curiosity's
   offer door) and be bounded only by quiet hours plus "one share per reading thread per
   day"? Default proposed: yes. Natural ceiling is the read rate (about 2/day).
2. Lane. Same `agent` lane as reading (27B on circe gpu1, Orion's reading voice), or
   `metacog_background`? Default proposed: `agent`, background, tool-free.
3. Raw chat. v1 searches crystallized memories (the curated record of your conversations),
   not raw chat turns through orion-recall. Is crystallization coverage good enough, or do you
   want raw chat in v1? (Recall's side effects on a read-only lookup, such as crystallization
   recall-boost writes in `orion/memory/crystallization/retriever.py`, are UNVERIFIED for the
   `/recall` path; that is why v1 avoids it.)
4. Should a resonance that *strains* a self-prior also open a re-test of that prior through the
   existing curiosity path, or only say so in the message and letter? Default: say so only.
5. Labels. The eval needs you to mark roughly 30 replayed resonances "would want this /
   meh / no". Willing?

## Proposed schema / API changes

All new models `extra="forbid"`, registered in `orion/schemas/registry.py`. Consumer-first
rollout (readers deploy before the producer) per the forbid-model rule.

### `ResonancePacketV1` (deterministic, frozen into the durable brief)

Built Hub-side after Stage 2 completes. Same input on resume gives the same packet, so a
retried run judges the same evidence.

```python
class ResonanceCandidateV1(_Forbid):
    ref_kind: Literal["self_prior", "lived_question", "lived_answer", "atlas_concept",
                      "crystallization", "chat_memory", "curiosity_finding", "dream",
                      "open_thread"]
    ref_id: str            # prior_id, question_id, concept node id, crystallization_id, ...
    store: str             # "falkor:orion_worldview:Prior", "pg:memory_crystallizations", ...
    excerpt: str           # re-read from the store (Chroma is an index, never the record), <=400 chars
    similarity: float | None   # recorded for the eval; never a gate (except the existing atlas 0.65)
    recent_packet_count: int   # how many of the last 14 days' packets offered this ref (attractor guard)

class ResonancePacketV1(_Forbid):
    packet_id: str         # uuid5(seed_id, "resonance")
    seed_id: str
    root_request_id: str | None
    read_title: str
    read_summary: str      # Stage 2 summary
    read_text_sha256: list[str]   # retained texts the quote may come from
    candidates: list[ResonanceCandidateV1]   # cap 24 total
    built_at: AwareDatetime
```

Per-kind budget (top-k, deduped by excerpt hash, self/root excluded):
priors+lived questions/answers 4, atlas concepts 4 (reuse Stage 2's `related_objects` + exact
objects from `ClaimContextV1`, no new embedding), crystallizations 4, chat memory 2, curiosity
findings 3, dreams 2, open concern threads all (currently 1).

### `ReadingResonanceV1` (payload of `ReadingFindingV1(kind="resonance")`)

```python
class ResonanceCitationV1(_Forbid):
    ref_kind: <same Literal>
    ref_id: str            # must be in the packet
    how: str               # one sentence: what this piece has to do with the read

class ReadingResonanceV1(_Forbid):
    resonance_id: str      # = packet_id
    seed_id: str
    read_quote: str        # verbatim from retained text or Stage 2 summary
    connection: str        # Orion's own words, first person, <=600 chars
    citations: list[ResonanceCitationV1]   # >=1, all resolve into the packet
    bears_on_prior: Literal["supports", "strains"] | None   # only when a self_prior is cited
    decision: Literal["share", "note", "none"]
    explore_next: str | None   # one question worth chasing, or None
    confidence: float          # model self-report, recorded, not gated until the eval calibrates it
    model: str; lane: str; run_id: str
```

What each field does at runtime (no-cathedral check): `decision` routes (share = outreach +
letter, note = letter only, none = receipt only); `bears_on_prior` is rendered in the letter
next to the cited prior; `explore_next` mints a seed; `recent_packet_count` is shown to the
judge and counted by the eval; `confidence` is stored for calibration only.

### Deterministic acceptance check (code, not model)

A judged output is rejected (stored as `decision=none`, `rejected_reason=...`) when:
- any `ref_id` is not in the packet (`citation_not_in_packet`);
- `read_quote` is not a whitespace-normalized substring of a retained text or the summary
  (`quote_not_in_read`);
- `connection` is empty, or contains a line of the prompt scaffold (`prompt_echo`). The prompt
  carries no example sentences at all, so there is nothing to echo by construction;
- `bears_on_prior` is set without a cited `self_prior` (`prior_relation_without_prior`).

The rendered message and letter entry show cited excerpts **re-read from their stores by
ref_id**, never the model's paraphrase of them. A citation that stops resolving renders
nothing.

### Workflow and bus

- New durable workflow `reading.resonance`, modeled on `journal.compose`
  (`orion/schemas/journal_compose_run.py`): one cortex verb call under the run's GPU pool hold
  (`options.gpu_lease`), `run_id = "resonance-" + uuid5(seed_id)`, so a resubmit is the same
  run. Busy pool means the run waits for its hold; restart resumes; `admission.deadline_at` =
  +6 h. **No Hub-side retry loop or shadow queue.**
- Brief: `ReadingResonanceRunBriefV1 { packet: ResonancePacketV1, lane, max_tokens }`.
- Runner publishes the finding on the `ReadingFindingV1` channel the parallel seam defines
  (channel name UNVERIFIED until that PR lands; add to `orion/bus/channels.yaml` there).
- Lifecycle: `publish_lifecycle(..., "resonance_started" | "resonance_completed" |
  "resonance_failed")` on the existing reading lifecycle channel.

### Routing

- **Outreach (`share`)**: through the seam's outreach hook. Proposed gates: quiet hours
  respected (held until 08:00, not dropped); at most one share per `root_request_id` per day
  (the three CoreWeave reads would be one message); content id
  `resonance:<sha of sorted ref_ids>` in the existing 7-day novelty ledger; skips the
  endogenous daily cap (pending Juniper, question 1).
- **Letter**: `share` and `note` rows render in "What I read" with the cited excerpts.
- **Curiosity seed**: `share`/`note` with `explore_next` mints one unscored seed in
  substrate-runtime, sibling of `link_accepted_seeds.py`: focal nodes = the read's landed
  concept node + cited atlas concept ids; source note `source:reading_resonance`;
  `strength:unscored_event`; once per `resonance_id`; ceiling 4 per tick. A resonance with no
  graph node to anchor (only memories cited) goes to the letter, no seed.

### New index

Chroma `orion_worldview_beliefs`: non-retired `:Prior`, `:SelfQuestionMint`, `:LivedAnswer`,
indexed through `orion/introspect/semantic_index.index_docs` (same as `concept_search`).
Derived, rebuildable from FalkorDB; add to `vector_collection_policy.py`.

## Files likely to touch

- `orion/schemas/reading_resonance.py` (new): packet, candidate, run brief.
- `orion/schemas/reading_finding.py` (from the parallel PR): add `resonance` kind + payload.
- `orion/schemas/registry.py`, `orion/bus/channels.yaml` (lifecycle names; finding channel owned by seam).
- `orion/world_pulse_read/resonance.py` (new): packet builder, acceptance check, render helpers.
- `orion/world_pulse_read/belief_index.py` (new) or an extension of `concept_search.py`.
- `orion/memory/vector_collection_policy.py`: new collection.
- `services/orion-hub/scripts/world_pulse_read_stage2.py`: submit after `stage2_completed` /
  `_project_assertions()`, keep the claim context for atlas candidates.
- `services/orion-durable-runs/app/reading_resonance_graph.py` (new), `admission_runtime.py` registration.
- Cortex verb `reading.resonance` + prompt template (location: same registry as
  `journal.compose`; UNVERIFIED exact path).
- `orion/substrate/resonance_seeds.py` (new) + `services/orion-substrate-runtime/app/worker.py`.
- Outreach and letter: only through the seam's hooks (`endogenous_outreach.py`,
  `orion_day_letter.py`), no second path.
- `services/orion-hub/.env_example`, `settings.py`, `README.md`; local `.env` sync.
- Tests under `orion/world_pulse_read/tests/`, `services/orion-durable-runs/tests/`,
  `services/orion-substrate-runtime/tests/`; eval `services/orion-hub/evals/run_reading_resonance_eval.py`.

Env (ship ON, per standing rule): `HUB_READING_RESONANCE_ENABLED=true`,
`HUB_READING_RESONANCE_SHARE_ENABLED=true`, `HUB_READING_RESONANCE_LANE=agent`,
`HUB_READING_RESONANCE_MAX_TOKENS=900`, `HUB_READING_RESONANCE_DAILY_MAX=12`.

## Cost

- **Retrieval**: about 6 embeddings + 7 Chroma queries + 2 Falkor reads per read, on
  vector-host (bge). Seconds; no GPU pool lease (vector-host is not pool-managed, same as
  concept search today).
- **Judgment**: one tool-free generation per completed read. Prompt about 5-6k tokens (packet
  capped at 24 excerpts x 400 chars + summary), output capped at 900 tokens. Qwen thinking can
  eat `max_tokens` on prose verbs (known issue), so the verb sets reasoning off or budgets it.
  Expected lease: tens of seconds to a few minutes, versus Stage 2's p50 18 minutes. UNVERIFIED
  until measured; the eval records lease time from `durable_resource_events`.
- **Volume**: 30 reads in 14 days means about 2 leases/day, worst burst 11/day;
  `DAILY_MAX=12` caps a runaway. Background priority, so chat always wins the lane.

## Non-goals

- No prior writes. Resonance can say a read strains a belief; it never revises, tests, or
  retires a prior. That stays with curiosity investigations.
- No change to the strict link quote rule or to `concept_search` thresholds.
- No new score or metric that ranks curiosity seeds; resonance seeds are unscored events.
- No raw-chat recall in v1 (question 3).
- No privacy filtering. There is no privacy boundary between Juniper and Orion (Juniper,
  2026-10-07: "there is no privacy. orion and i share a life together and im the only user in
  the system."). Crystallized memories of her life are fair to cite and quote.
- No penalty for Orion connecting a read to their own plumbing; that is legitimate content.
  The only bar is that cited rows are real.

## Proposal-mode items

- **Capability change**: Orion gets a step that relates what they read to what they already
  believe, remember, and wonder about, and can act on it by telling Juniper or opening a
  curiosity thread.
- **Data touched (read)**: `world_pulse_read_seed` (summary, handoff), retained read texts,
  `orion_worldview` priors and lived questions/answers, atlas concepts, crystallizations,
  `orion_chat`, `orion_curiosity`, `orion_dreams`, `temporal_self_arc`. **Written**: new Chroma
  `orion_worldview_beliefs`, finding rows (seam), durable run rows, outreach ledger rows,
  curiosity candidate sets. No writes to priors, crystallizations, or memory boosts.
- **Privacy boundary**: none, by Juniper's standing decision above.
- **Trace that proves it works**: for one read, all of: lifecycle `resonance_completed` with
  `seed_id`; durable run `resonance-<uuid>` with `run.resource_granted` and `run.completed`;
  Hub log `reading_resonance seed=... candidates=N decision=... citations=K rejected=-`; a
  finding row whose every `ref_id` resolves; for `share`, an outreach ledger row with content
  id `resonance:...` and the message in Hub chat; the letter's "What I read" entry; for a seed,
  a candidate set with note `source:reading_resonance`.
- **Dangerous failure modes, and the mechanism for each**:
  - Made-up connections: citations must resolve into the frozen packet, the quote must be in
    the read, and rendering shows stored excerpts, not paraphrase.
  - Spam: one share per reading thread per day, citation-set novelty over 7 days, quiet
    hours, `DAILY_MAX`, and `decision=none` is a normal, unrendered outcome.
  - Prompt echo: no example sentences in the prompt; scaffold-line check rejects echoes; the
    eval counts them (target 0).
  - The same memory every time (the DeepSeek-on-V100 attractor): `recent_packet_count` is in
    front of the judge; the eval reports the share of accepted resonances citing a ref offered
    in 5 or more of the last 14 days' packets.
  - Empty shells: a `share` with empty connection or zero valid citations is rejected, never
    sent.
- **Disable / rollback**: `HUB_READING_RESONANCE_ENABLED=false` stops new packets (in-flight
  runs finish and record); `HUB_READING_RESONANCE_SHARE_ENABLED=false` keeps letter/seed but
  sends nothing. The Chroma collection is derived and can be dropped. The finding kind is
  additive.

## Acceptance checks

Gate tests (fast, deterministic):
- Packet builder: per-kind caps, excerpt re-read from store, self/root exclusion, dedup of
  duplicate Chroma rows, `recent_packet_count` correct, same input gives same packet.
- Acceptance check: each rejection reason fires on a crafted output; a valid output passes.
- Rendering shows stored excerpts and drops a citation whose row is gone.
- Routing: share/note/none land where specified; second share on the same root in a day
  becomes note; quiet-hour share is held, not dropped.
- Seed minting: once per resonance_id, ceiling, unscored note, no seed without a graph anchor.
- Durable: resubmit is the same run; a run interrupted mid-generation resumes with the same
  packet.

Eval (`services/orion-hub/evals/run_reading_resonance_eval.py`):
- Replay every Stage 2 read completed in the last 14 days (30 as of 2026-10-13). Build packets
  read-only, run the judge on the configured lane, write
  `/tmp/reading-resonance-eval/labels.csv` with read, decision, connection, citations
  (with excerpts).
- Juniper labels every row: want / meh / no. All rows, not a sample, because N is small.
- Targets: precision of `share` (labeled want) >= 0.7; `share` rate <= 0.4 of reads; citation
  resolution 100%; prompt echoes 0; attractor share <= 0.2; median lease recorded.
- Honest limit: with about 30 reads and maybe 5 shares, this is a smoke-level precision check.
  Re-run at 30 days before tightening anything.

Live smoke after deploy: one real read produces the full trace above.

## Recommended next patch

**Patch A (read-only, no outreach):** `belief_index.py` + `orion_worldview_beliefs` collection,
`resonance.py` packet builder and acceptance check, the schemas, and the eval harness running
the judge offline against the 14-day replay. Deliverable: the labeled CSV and precision
number. This proves the judgment is worth hearing before it can message anyone.

**Patch B (after A meets target, rebased on the merged ReadingFindingV1 seam):** durable
`reading.resonance` workflow, Stage 2 submission, finding kind, outreach/letter hooks,
resonance seeds, env flags ON, live smoke.

### Metric gate notes

No new metric enters a decision. Similarity is recorded for ranking candidates and for the
eval; the probe above is the live check showing why it is not a gate (flat distribution, no
rest state that means "unrelated"). `confidence` is a model self-report, stored only, with no
threshold until the eval shows whether it tracks Juniper's labels. `recent_packet_count` is a
plain count with a direct consumer (the judge prompt) and is cheap to remove.
