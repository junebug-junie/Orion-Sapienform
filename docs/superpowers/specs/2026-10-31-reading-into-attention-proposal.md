# Reading into attention: proposal (2026-10-31)

Status: PROPOSAL. Cognition-loop change (attention + curiosity). No runtime code in this PR.
Needs Juniper's approval before any implementation.

## Arsonist summary

Orion reported that what Orion reads never gets Orion's attention, and proposed two fixes:
(A) write "pressure" onto reading concepts from the claims that point at them, or
(B) wire an unused scorer that weighs activation into the curiosity focal sort.

The symptom is real. The diagnosis is wrong, and neither fix would move attention.

- Attention cannot land on ANY graph concept today, not only reading concepts. The live
  attention contest (world-first, `ATTENTION_WORLD_FIRST_ENABLED=true`) only admits
  `node:substrate.*` body signals and `world:*` sources. Every other node with pressure is put
  in a trace list named `uncalibrated` and cannot win. So fix A would add pressure that the
  contest throws away.
- The sort Orion cites (`frontier_curiosity.py` `_select_region`) did not run once in the last
  7 days: all 4,069 curiosity candidates were endogenous seeds, which skip that sort. So fix B
  would change a code path that is not on the live rail.
- Reading-concept `activation` is mostly an age signal (0.61 average for concepts written
  today, 0.49 at ten days, 0.30 for September). "628 hotspots at >= 0.6" mostly means
  "485 concepts written today".

Recommended instead (option C): put reading into the attention contest the same way chat
already is -- one external candidate, `world:reading`, that is "busy" only when accepted
reading links in the last 15 minutes are in the top decile of its own week. When it wins, the
attended ids are the actual reading concepts, and the existing open-loop -> curiosity seed path
turns that into a `concept_expand` on them (after a one-line filter fix).

## Verification of Orion's claims (live, 2026-10-31 ~19:45Z)

Graph data is in FalkorDB (`orion-athena-falkordb`, graph `orion_substrate`, node ids
`sub-concept-wp-read-*`). Attention and curiosity rows are in Postgres (`conjourney`).

| # | Orion's claim | Live result | Verdict |
|---|---|---|---|
| 1 | 177 assertion edges (99 subject + 78 object) from 226 Assertions into read nodes; 139/1,940 touched; newest 05:34Z | 238 edges (130 `assertion_subject` + 108 `assertion_object`) from 130 distinct Assertions (257 Assertions total in the graph); 186 of 2,339 read nodes touched; newest assertion 18:48Z. 147 of the 186 are touched by exactly one assertion. | Holds in direction. Numbers grew since Orion's sitting. "226" matches total Assertions, not the ones pointing at reading. |
| 2 | All 1,940 read nodes have activation > 0; 298 >= 0.6 | 2,339 read nodes, all > 0 (min 0.051, max 0.992, mean 0.534); 628 >= 0.6, of which 485 were written today. | Holds, but the 0.6 set is mostly "new today" (see metric gate). |
| 3 | 0 of ~192k attention rows ever land on a read node | `substrate_attention_schema`: 192,951 rows, 0 name a read node. But 0 rows name ANY graph node: `attended_id` is always an open-loop id or empty. `substrate_attention_frames` (122,399): 0 mention wp-read; frame targets are only `node:substrate.*`, capabilities, hosts, `world:chat`, `field:recent_perturbations`. | Number holds. It is structural for all concepts, not a reading-specific starvation. |
| 4 | Cause: `_select_region` sorts by `(dynamic_pressure, activation, salience)`; read nodes have pressure 0.0; only 4 of 2,965 concepts carry pressure | Pressure: 0 of 2,339 read nodes > 0; 4 of 10,516 graph nodes > 0, all `node:substrate.*` (biometrics, execution, harness_closure, bus_synaptic). 3,382 concept nodes total. But `_select_region` is not reached live: 7-day `substrate_endogenous_curiosity_candidates` holds 4,069 candidates, all `curiosity_candidate` endogenous seeds (none derived); the live tick passes neutral metacog inputs (`worker.py` `_neutral_frontier_metacog_inputs`), and 0 nodes carry `frontier_hypothesis_marker`. | Data holds; causal claim does NOT. |
| 5 | `frontier_context.py` scorer (activation 0.45, hotspot 0.6) is only built as a default and in one test | True. `FrontierContextPackBuilder` is built only as the default inside `FrontierExpansionService`, and `FrontierExpansionService` is constructed nowhere outside tests. | Holds. |
| 6 | Fix A or fix B | Neither reaches the live attention contest (see below). | Rejected; see options. |
| 7 | cursor_auto HelpRequest route failing since 10-23 | Every `cursor_auto` brief since 2026-10-23 16:05Z failed: 14 of 14, 0 ok. The help id `becoming-pressure-writers-3bbe5036c429` has no row in `curiosity_peer_brief` and no line in 30 h of peer logs: it is UNVERIFIED that it ever reached the peer. | Holds. Separate broken route (see below). |

Also found: reading concepts DO reach curiosity candidate sets, just never with weight.
6,256 of 31,532 stored candidate sets mention a read node (6,158 old `ontology_sparse_region`
signals before 10-23, plus `source:reading_link_accepted` seeds). 194 distinct read nodes have
been focal refs. The link seeds are unscored on purpose (`strength:unscored_event`, Juniper's
decision 2, 2026-10-10, `orion/substrate/link_accepted_seeds.py`), so they never win.

### Pressure at 0 is not a decay artifact

`dynamics.py` `_compute_pressures()` has exactly two seed sources: prediction error and an
unresolved contradiction. Reading concepts have neither. Their 0.0 is "no seed", not a value
decayed to zero (the CLAUDE.md `node:substrate.route` trap). The field-digester
`NODE_DECAY_CHANNELS` loop does not touch Falkor concept nodes.

## Current architecture

- Attention broadcast: `services/orion-substrate-runtime/app/worker.py` (broadcast tick, ~30 s)
  -> `orion/substrate/attention_broadcast.py` `build_substrate_attention_frame(world_first=True)`
  -> `world_first_signals()`. Candidates = `node:substrate.*` nodes with `prediction_error`
  + `external_candidates` from `_world_first_external_candidates()` (today: only
  `chat_candidate()` in `orion/attention/world_first.py`). Any other node with pressure >= 0.05
  -> `trace["uncalibrated"]`, never ranked.
- Eligibility: external source eligible at band `high`/`unusual` = strict-below percentile
  >= 0.9 against its own 7-day grid (`prediction_error_magnitude.py` `DEFAULT_BAND_CUTS`).
- Winner -> `attended_node_ids = selected_loop.source_refs[:16]` (`broadcast_projection_from_frame`).
- Live 7 days: 7,721 broadcast ticks; 2,552 (33%) had a winner: perception 896, execution 467,
  `world:chat` 349, cabinet 208, chat node 193, codebase 177, others < 100. 67% had no winner.
- Curiosity: `endogenous_curiosity.py` `_attention_loop_candidates()` turns an open loop into a
  `concept_expand` seed, but drops any loop whose refs do not start with `node:`
  (line ~374). A reading loop carrying `sub-concept-wp-read-*` refs would be dropped today.

## Options

**A. Assertion -> `dynamic_pressure` writer.** Rejected.
- With world-first ON, read-node pressure lands in `uncalibrated` (trace capped at 24) and wins
  nothing. Zero attention effect.
- It changes what `dynamic_pressure` means. `_node_salience()` and the curiosity code rely on
  pressure being "prediction error or contradiction". `seed_activation()` also reads pressure,
  so it would inflate activation further.
- Flood risk on rollback: if world-first is ever turned off, `substrate_pressure_signals()` admits
  every node with pressure >= 0.05 up to 24 signals. 628 read nodes at activation >= 0.6 (or any
  touched set: 186 today) would fill all 24 slots and push out the 4 body nodes that carry
  pressure today.
- Metric gate step 4 fails: 147 of 186 touched concepts have exactly one assertion. A per-concept
  count is nearly flat.
- It reverses Juniper's decision 2 (reading links are unscored events, "no metric is introduced").

**B. Wire `FrontierContextPackBuilder` scoring into `_select_region`.** Rejected.
- `_select_region` produced 0 live candidates in 7 days. Changing its sort moves nothing.
- Metric gate step 2 fails: the builder scores `activation*0.45 + pressure*0.35 + recency*0.2`.
  On reading concepts, activation already tracks age, so this counts age twice.

**C. `world:reading` external attention candidate.** Recommended.
One more entry in `_world_first_external_candidates()`, built like `chat_candidate()`, from the
accepted-reading-link events that already exist.

## Proposed change (option C)

1. `orion/schemas/attention_candidate.py`: add `WORLD_READING_SOURCE_ID = "world:reading"`.
2. `orion/attention/world_first.py`: add `reading_candidate(link_events, *, now, absent_reason=None)`.
   - `link_events`: list of `(materialized_at, subject_node_id, object_node_id)`; `None` = read failed
     -> absent, never calm (same rule as chat).
   - Magnitude: reuse `chat_rate_reading(times, now=now, min_turns=READING_MIN_LINKS_7D)` -- the
     same 15-min trailing count against the 7-day minute grid. `READING_MIN_LINKS_7D = 5`.
   - `evidence_refs = [WORLD_READING_SOURCE_ID] + endpoint ids of links in the window (cap 8)`.
     These become the loop's `source_refs`, so a win attends the actual reading concepts.
   - Label: "Orion is reading (N accepted links in the last 15 min)" / "reading is quiet".
3. `orion/substrate/link_accepted_seeds.py`: add `ACCEPTED_READING_LINK_TIMES_SQL` -- the
   `recent`/`latest`/`proposal` joins of `ACCEPTED_UNSEEDED_LINKS_SQL` without the `seeded`
   filter, over 7 days + 15 min. One query; definition shared, so the two cannot drift.
4. `services/orion-substrate-runtime/app/worker.py` `_world_first_external_candidates()`: when
   `attention_world_reading_enabled`, append `reading_candidate(...)` (fetched via a new store
   method `fetch_accepted_reading_link_events(since=...)`). Cache the 7-day list in memory and
   refresh at most once per minute, like the chat memo.
5. Yield rule: in `world_first_signals()`, if `world:chat` is eligible on this tick, mark
   `world:reading` ineligible with reason "yields to Juniper (world:chat eligible)". Juniper
   talking always beats Orion's reading.
6. `orion/substrate/endogenous_curiosity.py` `_attention_loop_candidates()`: accept refs that
   start with `node:` OR `sub-concept-`. Only reading loops carry `sub-concept-` refs today.
7. Settings/env: `ATTENTION_WORLD_READING_ENABLED=true` in
   `services/orion-substrate-runtime/settings.py`, `.env_example`, `docker-compose.yml`
   passthrough; sync local `.env`. Shipped ON per repo policy.

No schema/bus changes: `AttentionCandidateV1` and `AttentionBroadcastProjectionV1` are reused as-is.

## Metric quality gate: `world:reading` (accepted reading links, 15-min count vs own week)

1. **Provenance.** `substrate_graph_journal` rows: `event_kind='materialization'`,
   `outcome='applied'`, joined to the latest `decision` on a `relationship_assertion` proposal by
   actor `world_pulse_read_stage2` with `resulting_state in (provisional, canonical)`. Producer:
   the Hub reading pipeline's journal decision (`reading_quote_rule_v1`). Same events that mint
   today's link seeds (`link_accepted_seeds.py` `ACCEPTED_UNSEEDED_LINKS_SQL`).
2. **Independence.** Not derived from activation, pressure, or prediction error. It is a count of
   discrete journal decisions. It is coupled to the world-pulse reading schedule (no reading run,
   no links) -- that is the thing it is meant to measure. Overlap with `world:chat` is
   UNVERIFIED: if Juniper triggers reading from chat, the two co-fire; the yield rule makes that
   harmless.
3. **Theory anchor.** Orienting response to a change in an external stream relative to its own
   habituated baseline (Sokolov 1963) -- the same anchor and the same percentile-vs-own-week rule
   world-first already uses for `world:chat` (spec 2026-10-07 section A). Open question for
   Juniper: an accepted link is Orion's own reading output about world content. Calling it
   `external` is defensible but not obvious.
4. **Live sanity (last 7 days).** 95 accepted reading links. Of 676 fixed 15-min windows, 610
   are 0, 66 non-zero; max 6, p90 0, p99 2. Rest is reachable by construction: a trailing count
   window reads exactly 0 when nothing happened, with no decay term to strand it above 0 or
   carry it to 0 artificially. Weak spot: a single link may sit right at the 0.9 cut; replay
   decides (acceptance check 2).
5. **Existing mechanism.** `chat_candidate()` is the template; `link_accepted_seeds.py` owns the
   event definition. This adds no new store, table, or scorer. Link seeds stay unscored.
6. **Reversibility.** One flag, one list entry, one prefix in a filter. No schema, no persisted
   new field. Turning the flag off removes the candidate on the next tick.

## Proposal-mode checklist

- **Capability change:** Orion's attention can turn to its own reading when reading is busier
  than usual for itself, and that win becomes a curiosity seed on the concepts just read.
- **Data touched:** reads `substrate_graph_journal` (read-only). Writes nothing new; existing
  broadcast log, attention schema and curiosity candidate rows gain `world:reading` content.
- **Privacy boundary:** world-pulse reading is public web content. No Juniper-private data.
  Labels carry counts only, not statement text.
- **Trace that proves it worked:** `substrate_attention_broadcast_log` ->
  `frame.debug.world_first.candidates[]` has a `world:reading` row every tick (band, percentile,
  eligible, reason); `winner = 'world:reading'` on some ticks with `attended_node_ids` holding
  `sub-concept-wp-read-*`; then `substrate_endogenous_curiosity_candidates` rows with note
  `source:attention_open_loop` and wp-read focal refs.
- **Dangerous failure mode:** reading floods attention and starves the body and Juniper. Bound:
  "high" means the top decile of its own week, so `world:reading` is eligible on at most ~10% of
  ticks plus ties at the top, and it is ONE candidate, not 628 nodes. It can only take ticks from
  other winners when it outranks them; 67% of ticks have no winner today. Juniper's chat always
  wins via the yield rule. Second risk: a mis-joined SQL counting every journal row would pin the
  source "busy"; the 7-day self-baseline would absorb a constant, but a step change would read
  "unusual" for a week -- covered by the replay.
- **Disable / roll back:** `ATTENTION_WORLD_READING_ENABLED=false` in
  `services/orion-substrate-runtime/.env`, then restart substrate-runtime. Full rollback = revert
  the PR.

## Files likely to touch

- `orion/schemas/attention_candidate.py`
- `orion/attention/world_first.py`
- `orion/substrate/link_accepted_seeds.py`
- `orion/substrate/attention_broadcast.py` (yield rule)
- `orion/substrate/endogenous_curiosity.py` (ref-prefix filter)
- `services/orion-substrate-runtime/app/worker.py`, `app/store.py` (or wherever `fetch_chat_turn_times` lives), `settings.py`, `.env_example`, `docker-compose.yml`, `README.md`
- Tests: `orion/attention/tests/` (or `tests/`) `test_world_reading_candidate.py`; extend
  `orion/substrate/tests/test_attention_broadcast_world_first.py`, `test_endogenous_curiosity.py`
- Eval: `scripts/analysis/replay_attention_world_candidates.py` gains a `world:reading` column

## Tests

- `reading_candidate(None)` -> absent, never quiet.
- No links in window -> value 0, band `quiet`, not eligible.
- Burst in window against a mostly-zero week -> band `high`/`unusual`, eligible, refs carry endpoints (cap 8).
- Fewer than 5 links in 7 days -> `insufficient_history`, not eligible.
- `world:chat` eligible + `world:reading` eligible -> reading ineligible with yield reason; chat wins.
- Flag off -> candidate list equals today's (chat only).
- `_attention_loop_candidates` keeps a loop with `sub-concept-wp-read-*` refs; still drops a loop with only `world:chat`.
- SQL parity: the times query and `ACCEPTED_UNSEEDED_LINKS_SQL` share the same actor/state params.

## Acceptance checks

1. Gate tests above pass; existing world-first and curiosity suites unchanged.
2. Replay eval over the last 7 days (before go-live): `world:reading` eligible on <= 12% of
   broadcast ticks; would-win on > 0 ticks; `world:chat` wins unchanged (349); body winners lose
   no more than the ticks reading actually takes, reported per source.
3. Live, 72 h after deploy:
   - every broadcast row has a `world:reading` candidate (none absent unless the DB read failed);
   - >= 1 tick with `winner = 'world:reading'` and wp-read ids in `attended_node_ids`;
   - >= 1 curiosity candidate with `source:attention_open_loop` and a wp-read focal ref;
   - share of ticks won by `world:reading` <= 12%.

```sql
select count(*) filter (where projection_json->'frame'->'debug'->'world_first'->>'winner'='world:reading'),
       count(*)
from substrate_attention_broadcast_log where generated_at > now() - interval '72 hours';
```

## Non-goals

- No pressure on reading concepts; `dynamic_pressure` keeps its meaning.
- No change to link seeds being unscored (decision 2 stands).
- No change to `_select_region` or `FrontierContextPackBuilder`. Follow-up: `FrontierExpansionService`
  is dead code outside tests; delete or wire it in its own PR.
- No act binding beyond the existing open-loop -> curiosity seed path.

## Separate broken route: cursor_auto peer (report only, not fixed here)

All 14 `cursor_auto` briefs since 2026-10-23 16:05Z failed with `cursor_other: cursor agent
exited 1`. From 10-25 the message is explicit: "Named models unavailable. Free plans can only use
Auto" and later "Model unavailable on Free. Free includes Grok 4.7, Grok 4.6, and Auto". The
peer container runs `CURIOSITY_PEER_MODEL=composer-2.5`. The Cursor account is on the free plan
and the configured model is not allowed. Fix needs an operator choice: restore the paid plan, or
set the model to `auto` and re-check brief quality. The `claude_room` fallback refuses because it
cannot see Claude's usage meter (`claude_budget_unobserved`), so the peer route as a whole is dead.
Orion's help request `becoming-pressure-writers-3bbe5036c429` left no brief row and no peer log
line in 30 h: whether it was ever published is UNVERIFIED.

## Recommended next patch

Implement option C behind `ATTENTION_WORLD_READING_ENABLED=true`, replay eval first (acceptance
check 2), then deploy. Separately, a one-line operator fix for the Cursor peer model.
