# Orion Memory Consolidation

Subscribes to `orion:memory:turn:persisted` (sql-writer post-commit outbox), classifies each chat turn via LLM gateway quick lane logprobs, patches `chat_history_log.spark_meta`, tracks consolidation windows, and on boundary closure runs a **deterministic consolidation gate** (default) or legacy graph suggest.

## Consolidation output modes

| `MEMORY_CONSOLIDATION_OUTPUT` | Behavior |
|-------------------------------|----------|
| `crystallization_propose` (default) | Run `consolidation_memory_gate`; **skip** low-signal windows (`consolidation_status=skipped`) or **propose** `MemoryCrystallizationV1` for governor review |
| `graph_draft` | Legacy path: LLM `memory_graph_suggest` + pending graph draft insert (manual bridge only) |
| `skip_only` | Run gate for traceability; always mark window skipped — no crystallization or graph draft |

Gate thresholds: `MEMORY_CONSOLIDATION_MIN_NOVELTY` (default `0.35`), `MEMORY_CONSOLIDATION_MIN_SIGNIFICANCE` (default `0.40`).

**Junk check runs first (memory redesign Stage 0A).** Before any other rule, the gate judges each turn's *user prompt alone* (`orion/memory/intake_junk.py`): greetings/filler ("sup", "ty!", "you back?"; never anything with a negation, a non-English-script letter, or a question about a real topic) and Hub skill commands (aliases from the real workflow registry, `orion/cognition/workflows/registry.py`: "Run github compactor.", "Do a journal pass.", "Compact the last 24 hours of chat into a memory digest."; a command followed by real content, like "Do a journal pass about my labs", is a memory) are junk. A window whose prompts are all junk is skipped (`low_info_social` / `hub_command`), even with a repair signal or a high novelty score. Orion's reply is no longer part of this judgment: it is almost never small talk, so "prompt AND reply are low-info" admitted every greeting. A kept window's row summary is its last *non-junk* prompt, so ["Headed to Austin…", "sup yo"] is saved as the Austin line, not "sup yo".

`repair_signal` now means real repair pressure: the Hub only emits the grammar atom at or above the repair contract's `concrete_bias` level (0.45, `orion.substrate.appraisal.contract.REPAIR_SIGNAL_LEVEL_FLOOR`), not whenever an appraisal ran (~96% of turns before).

Replay eval over 30 days of real chat windows (fixture redacted: only junk and spec-quoted lines are verbatim):

```bash
python services/orion-memory-consolidation/evals/run_intake_gate_replay_eval.py            # replay fixture
python services/orion-memory-consolidation/evals/run_intake_gate_replay_eval.py --refresh  # re-capture, read-only
pytest services/orion-memory-consolidation/evals -q
```

Grammar repair evidence (read-only): `MEMORY_CONSOLIDATION_FETCH_GRAMMAR_EVIDENCE=true` queries `grammar_events` by `hub.chat:{NODE_NAME}:{correlation_id}` trace. Optional override DSN: `MEMORY_CONSOLIDATION_GRAMMAR_DSN`.

**Note:** Proposed crystallization IDs are stored in `memory_consolidation_windows.draft_id` until a dedicated `crystallization_id` column migration lands.

## Cross-window concept-relation resolution (on by default)

Same-window duplicate detection (`orion.memory.crystallization.detection.detect_duplicates`) requires `scope_overlap`, which is structurally always `False` across two different consolidation windows — every crystallization gets a unique per-window `scope`, so two windows can never share one. `orion.memory.crystallization.concept_relation` adds a second, cross-window path: vector-similarity candidate retrieval (`candidate_retrieval.fetch_similar_candidates`, not scope-gated) followed by one bounded, structured-output LLM call that judges `same` / `refines` / `contradicts` / `unrelated` against the nearest existing active crystallizations of the same `kind`.

Dispatch is deliberately conservative: `same` reinforces the existing target (identical mechanism to the same-window path). `refines` and `contradicts` only attach a typed link to the *new* candidate's own `links` (persisted to `memory_crystallization_links` on insert, same as any other crystallization) — they never mutate or supersede the existing target's status. That stays a human decision via the existing `/api/memory/crystallizations/{id}/links` and supersede endpoints. Every decisive branch (`same`/`refines`/`contradicts`) stamps `provenance.concept_relation` (relation, target id, confidence) on the affected row for audit, independent of which branch acted.

**Decision log + belief-revision digest.** Every real LLM decision — including `unrelated` and sub-floor `contradicts`/`refines` that the dispatch above discards — is written to `memory_concept_relation_decisions` (`orion.memory.crystallization.repository.insert_concept_relation_decision`, guarded to never raise). `scripts/concept_relation_digest.py` (repo root, run on demand or via cron — not a live service loop) reads undigested rows and reports call volume / relation distribution / near-miss counts under `CONCEPT_RELATION_CONFIDENCE_FLOOR`, and marks them digested. **As of 2026-08-20 it no longer also writes a `reflection`-kind crystallization per decision** — that mirrored `memory_concept_relation_decisions` (already the real, durable, queryable trace) into `memory_crystallizations` a second time, auto-approved and bypassing manual review, and had grown to 356/620 (57%) of all active crystallizations before it was removed. `memory_concept_relation_decisions` remains the belief-revision trace; query it directly rather than expecting crystallizations for it.

Ships on (2026-10-10). Defaults point the embed host and Chroma at the athena containers on `app-net`:

| Env | Default | Purpose |
|-----|---------|---------|
| `CONCEPT_RELATION_RESOLUTION_ENABLED` | `true` | Master flag for this path |
| `CONCEPT_RELATION_CONFIDENCE_FLOOR` | `0.6` | Minimum LLM decision confidence to act; below this, falls through to the normal formation-policy path unchanged |
| `CONCEPT_RELATION_CANDIDATE_LIMIT` | `5` | Max vector-similar candidates fetched and sent to the LLM prompt |
| `CONCEPT_RELATION_TIMEOUT_SEC` | `8.0` | RPC timeout for the relation-judgment call |
| `CRYSTALLIZER_EMBED_HOST_URL` | `http://orion-athena-vector-host:8320/embedding` | Embedding HTTP endpoint for candidate retrieval and for projecting new crystallizations into Chroma |
| `CRYSTALLIZER_EMBED_TIMEOUT_MS` | `8000` | Embed call timeout |
| `CHROMA_HOST` / `CHROMA_PORT` | `orion-athena-vector-db` / `8000` | Chroma vector store for candidate retrieval |
| `CRYSTALLIZER_VECTOR_COLLECTION` | `orion_memory_crystallizations` | Chroma collection name (matches Hub's projection collection) |

**Readiness (boot + every 10 min).** If resolution is enabled but either host is empty or unreachable, or the collection holds fewer documents than `CONCEPT_RELATION_CANDIDATE_LIMIT`, the service logs `concept_relation_resolution_degraded` (WARNING) and `/health` returns `"degraded": true` with reasons under `concept_relation.problems` and the probe time in `concept_relation.checked_at`. This replaced a silent no-op that let the writer produce zero decisions from 2026-09-07 to 2026-10-10.

### Scheduled maintenance (Athena cron)

`scripts/concept_relation_digest.py` is a standalone script, not a live service loop (see above) -- something external has to run it. Install on the host that runs the memory-consolidation stack (`crontab -e` as the operating user):

```cron
# Concept-relation decision digest -- reports on memory_concept_relation_decisions rows
# and marks them digested (no longer writes crystallizations, see above). Idempotent
# (only acts on digested=false rows), safe to run frequently. Requires POSTGRES_URI in
# the shell environment or a sourced .env; see services/orion-hub/.env.
*/30 * * * * cd /mnt/scripts/Orion-Sapienform && POSTGRES_URI=$(grep -m1 '^POSTGRES_URI=' services/orion-hub/.env | cut -d= -f2-) make concept-relation-digest >> /mnt/scripts/Orion-Sapienform/logs/orion-concept-relation-digest.log 2>&1
```

**If this cron entry dies, is dropped after a host migration, or the job starts failing silently, nothing else will notice on its own.** `make check-concept-relation-digest-liveness` is the fail-safe: it queries the real backlog (oldest undigested decision's age, not a heartbeat file that can go stale independently of the thing it claims to represent) and exits non-zero if it exceeds `MAX_AGE_HOURS` (default 3h -- generous headroom over the 30-minute cadence above). Run it by hand any time you suspect the digest stopped running:

```bash
POSTGRES_URI=... make check-concept-relation-digest-liveness
# or, to tighten/loosen the threshold:
POSTGRES_URI=... make check-concept-relation-digest-liveness MAX_AGE_HOURS=1
```

A clean exit (0) with "no undigested decisions pending" means the loop is closing. A STALE failure means: check `crontab -l` for the entry above, check `logs/orion-concept-relation-digest.log` for errors, and if this is a fresh host or post-migration box, re-add the crontab line (it does not persist itself -- see "Recreate ops after a fresh host" below).

### Recreate ops after a fresh host, lost crontab, or disaster recovery

Use this checklist any time this service (or its cron dependency) is missing after a host swap:

1. **Confirm the digest cron entry is actually installed:** `crontab -l | grep concept_relation_digest`. If empty, paste the block from "Scheduled maintenance" above.
2. **Confirm `logs/orion-concept-relation-digest.log` exists** (create the `logs/` dir if this is a fresh checkout: `mkdir -p /mnt/scripts/Orion-Sapienform/logs`).
3. **Run the liveness check once by hand** (`make check-concept-relation-digest-liveness`) to confirm the new cron entry is actually firing, rather than waiting up to 3 hours to find out.
4. **Do not assume `CONCEPT_RELATION_RESOLUTION_ENABLED=true` implies the digest is scheduled** -- they are two independent things that must both be true for the loop documented above to actually close. This exact "flag on, dependency not wired" pattern has already caused silent no-ops twice in this repo (`CONCEPT_RELATION_RESOLUTION_ENABLED` itself missing its embed/chroma hosts on first activation, and `RECALL_GRAPHITI_IN_CHAT` missing its adapter URL) -- checking both halves explicitly, every time, is cheaper than re-discovering this.

## Drive-history reflection synthesis (manual/on-demand only -- NOT cron'd)

**Source data is frozen as of 2026-07-30.** `DriveEngine` (the producer this
section describes) was deleted outright (`chore/delete-orion-drives`, PR #1486),
following through on `orion/sentience_striving_program/README.md` §8's halt.
Postgres `drive_audits` no longer receives new rows -- its retention-prune job
was disabled specifically so this now-finite history isn't deleted. This script
still runs against whatever ticks accumulated before the deletion; it will never
observe a pattern more recent than 2026-07-30, and `MIN_DISTINCT_DAYS`'s
burst-vs-pattern distinction (below) no longer matters going forward since no new
ticks will ever arrive to burst.

`scripts/drive_history_reflection_synthesis.py` (repo root) reads Orion's own real,
persisted drive-activation history and synthesizes ONE `reflection`-kind
`MemoryCrystallizationV1` observing a long-horizon pattern -- e.g. "continuity has
been the dominant drive in most audited ticks this week." Source data: `DriveAuditV1`
(`orion/core/schemas/drives.py`) was computed on every DriveEngine tick and persisted,
append-only, by `services/orion-sql-writer` to the Postgres `drive_audits` table
(the old Fuseki `drives` graph froze on 2026-06-19 and was removed as both a write
and read path on 2026-07-15) -- unlike the "latest value only" stores this repo
already has for the same signal (`LocalProfileStore` / `autonomy_state_v2`'s
single-row UPSERT), this table is a genuine historical time-series, one row per
tick, including per-tick `drive_pressures` / `active_drives` JSONB.

**Architecture is deliberately split into a deterministic reducer stage and a
narrow LLM-phrasing stage** (event -> schema -> trace -> reducer -> projection ->
LLM phrasing -> crystallization, per this repo's event-substrate-first mandate) --
the LLM is never shown raw per-tick rows:

1. Fetch real `DriveAuditV1` ticks from Postgres `drive_audits` (bounded by
   `--max-events`, most recent first, same DSN as the crystallization write path).
2. `reduce_drive_history()` -- a **pure, unit-tested Python function** (same bar as
   `orion/spark/concept_induction/drive_tension.py`: synthetic-input/known-output
   tests, zero LLM involvement) -- aggregates them into dominant-drive counts/shares,
   a per-day breakdown, active-drive frequency, and mean pressures.
3. `build_fact_sheet()` renders that aggregation into a small numbered list of
   already-computed, already-verified fact strings (real dates, real counts, real
   percentages, real timestamps).
4. The LLM (bus RPC, same pattern as `concept_relation.py::resolve_concept_relation()`)
   receives ONLY the fact sheet and is asked to phrase ONE narrative sentence,
   citing specific facts by number.
5. `parse_and_validate_narrative()` enforces the grounding guardrail: every cited
   fact's real literal tokens (drive name, date, percentage, or timestamp) must
   appear verbatim in the narrative text, or the run is rejected -- an LLM cannot
   pass validation by inventing plausible-sounding but fake specifics.
6. Only then is a `reflection` crystallization written, with evidence refs citing
   both the aggregation object and the real cited `DriveAuditV1` artifacts, so a
   human reviewer can verify the narrative against what was actually computed.

**Guardrail: refuses to synthesize on thin data.** Below `MIN_EVENTS=5` real ticks
or `MIN_DISTINCT_DAYS=2` distinct calendar days in the queried window, the reducer
marks the aggregation insufficient and the script exits cleanly reporting exactly
why -- no LLM call is made, nothing is written. `MIN_DISTINCT_DAYS` exists because
DriveEngine ticks several times a minute in a single session, so event *count*
alone can be satisfied entirely within one sitting; requiring real day-spread is
what distinguishes an actual long-horizon pattern from a burst.

Run:

```bash
POSTGRES_URI=postgresql://user:pass@host:port/db python scripts/drive_history_reflection_synthesis.py
python scripts/drive_history_reflection_synthesis.py --postgres-uri postgresql://... --since-days 14
python scripts/drive_history_reflection_synthesis.py --json
```

| Env / flag | Default | Purpose |
|---|---|---|
| `--postgres-uri` / `$POSTGRES_URI` | *(required)* | Same convention as `concept_relation_digest.py` |
| `--subject` | `orion` | `DriveAuditV1.subjectKey` to read |
| `--since-days` | `30` | Window size; real coverage found is always reported, never assumed |
| `--max-events` | `500` | Cap on raw ticks fetched from `drive_audits` (most recent first) |
| `--redis` / `$ORION_BUS_URL` | `redis://localhost:6379/0` | Bus URL for the LLM gateway RPC |
| `--llm-route` | `metacog` | Gateway route for the narrative-phrasing call |

**This is NOT scheduled or cron'd in this patch, unlike `concept_relation_digest.py`
above.** It is explicitly a manual/on-demand tool: its output (a narrative claim
about Orion's own long-horizon tendencies) needs human review before anyone should
trust it as a recurring automated process. Run it by hand, read the generated
`summary` text critically, and decide separately whether a cron entry is warranted
once the grounding guardrail and prompt have been proven out on real data.

## Channels

| Direction | Channel |
|-----------|---------|
| In | `orion:memory:turn:persisted` |
| In (confirmation loop) | `orion:attention:loop_outcome` (`attention.loop.outcome.v1`, `memory-confirm-*` loops only) |
| Out | `orion:chat:history:spark_meta:patch` |
| Out (threshold) | `orion:signals:memory_consolidation` (`signal.memory_consolidation.turn_change`) |
| Out (propose) | `orion:memory:crystallization:proposed` (`memory.crystallization.proposed.v1`) |
| Out (shadow) | `orion:memory:episode:closed` (`memory.episode.closed.v1`) |

## Episode boundary (Stage 1, shadow)

Spec: `docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md`.

- **One classification per turn (Fix 2).** sql-writer publishes each turn twice. A turn already in a window is not classified again (`WindowStore.find_windowed_turn`); the second pass used to compare the turn with itself and overwrite `chat_history_log`'s boundary score with a meaningless low value.
- **Wall clock on the turn (Fix 1).** The Hub stamps `spark_meta.conversation_phase = {phase_change, delta_user_seconds, crossed_day, source}`. With `MEMORY_LEGACY_BOUNDARY_USE_PHASE=false` (default) the live window rule and the classify prompt do not see it, so live closing is unchanged.
- **Rule 3 in shadow.** Each direct-conversation turn is also placed into `memory_episode_shadow`: long_gap / next_day / stale_thread split; resumed_thread splits only with a judge score >= `MEMORY_BOUNDARY_OVERRIDE_THRESHOLD`; same_breath / short_pause never split; no phase falls back to the `MEMORY_WINDOW_FALLBACK_GAP_SEC` gap. The boundary turn opens the next episode. Each turn records both the legacy and the Rule 3 decision. A closed episode publishes `memory.episode.closed.v1` with `close_lag_sec`; an episode of only workflow commands closes as `skipped/command_only`. AI Town is excluded.
- **Close audit.** Each live window records `close_reason` (`legacy:...`) and `boundary_score_at_close`.
- **Replay eval:** `python services/orion-memory-consolidation/evals/run_episode_boundary_replay_eval.py [--refresh]`.

| Env | Default | Purpose |
|-----|---------|---------|
| `MEMORY_EPISODE_SHADOW_ENABLED` | `true` | Kill switch for the Rule 3 shadow tracker and its close event |
| `CHANNEL_MEMORY_EPISODE_CLOSED` | `orion:memory:episode:closed` | Close event channel |
| `MEMORY_LEGACY_BOUNDARY_USE_PHASE` | `false` | Let the live window rule and classify prompt read the phase stamp (changes live windows) |

## Referent projector (memory Stage 2, 2026-10-06)

`app/referent_projector.py`, a 30 s loop (`MEMORY_REFERENT_PROJECTOR_ENABLED`,
`MEMORY_REFERENT_PROJECTOR_TICK_SEC`, `FALKORDB_URI`, `FALKORDB_SUBSTRATE_GRAPH`,
`SUBSTRATE_ASSERTION_REQUIRED_READERS`). It copies what the referent step wrote into Postgres
into Orion's one graph, as producer `memory.referents`. It never hydrates the whole graph: at
start it loads only its own nodes. Rebuild ONLY with `scripts/rebuild_referent_graph.py`
(truncating `referent_projection` by hand re-projects nodes and evidence but not accepted
assertions). Concepts it owns (the rest are in `orion/memory/referents/README.md` and
`orion/substrate/README.md`):

| Concept | What it means in plain English | Producer | Consumer | Test |
|---|---|---|---|---|
| Readiness gate | Writes NOTHING until every substrate reader advertises it can read the new shapes; meanwhile logs `referent_projector_waiting reason=readers_not_ready missing=[...]` and `/health` shows `referent_projector.state=waiting` with the missing readers. | `orion/substrate/reader_capability.py`: each reader service calls `advertise_at_startup()` at boot, which re-writes its key every 10 min with a 30 min expiry, so a dead or rolled-back reader closes the gate again | `ReferentProjector.run_once`, `/health` | `tests/test_referent_projector_pg.py::test_projector_writes_nothing_until_every_reader_is_ready`, `test_health_reports_the_missing_readers` |
| Referent node in Falkor (Entity, or Concept for `concept:` keys; producer `memory.referents`) | The thing itself, with its usable names shown on it. Fenced: never merged by label or embedding. | this projector | neighborhood reads; `AssertionProjector` endpoint check | `tests/test_referent_projector_pg.py` |
| Memory Evidence node (`episode_memory:<id>`) + `observed_in` provenance edge | "This thing is mentioned in that memory", with when Orion learned it (`valid_from`) and when the memory stopped being active (`valid_to`). The text stays in Postgres. | this projector | `AssertionProjector` (a claim's evidence must exist); every walk/region read refuses it | projector test (6 edges, all `provenance`; superseding closes them) |
| Label-collision question | Our "circe" meets topic-foundry's "circe": Orion asks whether they are the same; our node stays walkable; never a merge. | this projector (first projection only) | identity questions in the daily report | projector test |
| `referent_projection` ledger | What the projector last wrote per node/memory, so it rewrites only what changed. | this projector | this projector; `rebuild()` | projector test (second tick writes nothing), discipline eval (rebuild into an empty graph = same ids, no new journal rows) |

## Memory confirmation loop ("Orion is asking", shadow)

Spec: `docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md` sections 3 and 5, pulled forward from Stage 3 (Juniper, 2026-10-06). Code: `orion/memory/episode/confirmation.py`, wired here by `app/confirmation_loop.py`.

When the shadow distiller stores a high-stakes memory, Orion asks Juniper about it once, in the Hub's "Orion is asking" panel, and her answer changes the shadow memory. Nothing outside the shadow `episode_memory*` tables and the card she sees changes.

- **Open.** A ticker (`MEMORY_CONFIRMATION_TICK_SEC`) gives each `stakes=high`, `pending_confirmation`, unasked memory an `orion_ask` card (`source_kind=memory_confirmation`, `source_ref=memory-confirm-<memory_id>`, expires in 7 days) and sets `confirmation_loop_id`. At most **5** cards (memory + open-question kinds) are open at once, and at most `MEMORY_CONFIRMATION_DAILY_CAP` (3) new cards open per local day; the rest wait, oldest first. A memory whose statement is word-for-word one Juniper already rejected is not asked (`confirm_skipped` event; exact match only). Trace: `episode_memory_event` `confirm_asked`.
- **Wording** is deterministic: the memory's statement, quoted, framed by where it came from and why Orion is asking. A memory from an internal channel (reverie, dream, curiosity, journal, topic model) is always "This came from my own ..., not from anything you told me", whatever its voice.
- **Apply.** Juniper's Confirm / Revise / Reject arrives as `AttentionLoopOutcomeV1` on `CHANNEL_ATTENTION_LOOP_OUTCOME` and, every tick, from the `attention_loop_outcome` table (any outcome no event carries yet), so a lost publish delays an answer but never drops it. Confirm: `confirmed`, voice -> `worked_out_together`, reinforced (strength +0.2, half-life x2 capped at 365 d). Revise: a new `confirmed` / `worked_out_together` memory in her words supersedes the original (kept as `corrected` / `superseded`); its only evidence is her note (`confirmation_revise`), and none of the old quotes are copied. The note must be at least 6 words and differ from the current wording (Reject is the button for "drop it"). Reject: `rejected` / status `rejected`; excluding rejected memories from recall lands with the Stage 2 recall PR (PR F), since nothing reads `episode_memory` for recall yet. First answer wins; replays are no-ops; one failing outcome is logged and counted (`apply_failed`) and never stops the tick.
- **Reap.** A `pending_confirmation` memory whose card was closed without a usable answer (no outcome row, or only an invalid one) gets its loop id cleared and is asked again with a fresh card (`ask_reaped` event, logged as a warning).
- **Expire.** An unanswered card after 7 days: card `expired`, memory `unconfirmed`. Never a yes, no outcome row, never re-asked by the panel.
- **Not built: answering in chat.** Deferred (see the PR report): it needs a model judgment on the turn after an asked card, and the panel path is complete without it.

| Concept | Plain-English meaning | Producer | Consumer | Test |
|---|---|---|---|---|
| `stakes=high` | Orion should check with Juniper before keeping this as settled | distiller + `validate.resolve_stakes` | `confirmation.open_cards` (only high memories get a card) | `test_high_stakes_memory_gets_one_card_and_low_gets_none` |
| `stakes_reason=health` | About her or her family's health | distiller (v3 rubric) | card line "It's about health, so I'd rather check than assume." | `test_every_high_stakes_category_has_its_own_card_wording` |
| `stakes_reason=family_relationships` | About her family and close relationships | distiller | card line "...the people close to you, so I want to get it right." | same |
| `stakes_reason=juniper_feelings` | About how she was feeling | distiller | card line "...I don't want to put words in your mouth." | same |
| `stakes_reason=identity_conclusion_about_juniper` | Orion's read on who she is, beyond her words | distiller | card line "It's my read on who you are..." + closer "Is that fair, and should I keep it?" | same + `test_direction_and_identity_cards_close_with_their_own_question` |
| `stakes_reason=orion_machinery` | Orion's conclusion about how it works | distiller | card line "...you can check it better than I can." | same |
| `stakes_reason=orion_asks_direction` | Orion needs her direction | distiller / validator (`asks_direction`) | card line "I need your direction on this one." + closer "Is that the right direction?" | same |
| `stakes_reason=orion_relationship` | About the two of them | distiller | card line "It's about us, so I don't want to decide it alone." | same |
| `stakes_reason=unjudged` (or none on a high row) | High, but no category was given | validator | card line "I couldn't tell how personal this is, so I'm checking first." | `test_uncategorized_high_stakes_says_so` |
| `stakes_reason=none` | Low stakes | distiller | never asked (no card) | `test_high_stakes_memory_gets_one_card_and_low_gets_none` |
| `confirmation_state=pending_confirmation` | Waiting to be asked or answered | validator | `open_cards`, `expire_cards`, daily report flag | `test_confirmation_pg.py` |
| `confirmation_loop_id` | Which card asks about this memory | `open_cards` | `expire_cards` join, Hub `memory_statement` lookup, `apply_outcome` | `test_high_stakes_memory_gets_one_card_and_low_gets_none` |
| `confirmation_state=unconfirmed` | Asked, no answer in 7 days; not a yes | `expire_cards` | daily report flag; `apply_outcome` still accepts a later answer | `test_seven_day_expiry_marks_unconfirmed_never_confirmed_and_frees_the_slot` |
| `confirmation_state=confirmed` | Juniper said yes | `apply_outcome` | voice relabel, reinforcement, daily report flag | `test_confirm_relabels_voice_reinforces_and_records_the_outcome` |
| `confirmation_state=corrected` | Juniper reworded it; superseded | `apply_outcome` | supersede chain, daily report flag | `test_revise_supersedes_the_original_and_confirms_her_wording` |
| `confirmation_state=rejected` | Juniper said no | `apply_outcome` | do-not-remint check in `open_cards`, daily report flag (recall exclusion: PR F) | `test_reject_marks_rejected_and_an_exact_repeat_is_never_asked` |
| `attention_loop_outcome` (`memory-confirm-*`) | Her answer, the one resolution record | Hub `POST /api/asks/{id}/resolve` | `handle_loop_outcome` (bus) + `pending_outcomes` (table) | `test_ask_resolve_pg.py`, `test_catch_up_applies_an_outcome_whose_bus_event_was_lost` |

| Env | Default | Purpose |
|-----|---------|---------|
| `MEMORY_CONFIRMATION_LOOP_ENABLED` | `true` | Kill switch: stops opening, expiring and applying |
| `MEMORY_CONFIRMATION_TICK_SEC` | `60` | Ticker period (expire, catch up, reap, open) |
| `MEMORY_CONFIRMATION_DAILY_CAP` | `3` | New cards per local day, on top of the 5-open cap (0 = none) |
| `MEMORY_CONFIRMATION_TZ` | `America/Denver` | Whose midnight starts the daily cap's day |
| `CHANNEL_ATTENTION_LOOP_OUTCOME` | `orion:attention:loop_outcome` | Outcome event channel |

Needs `services/orion-sql-db/manual_migration_memory_confirmation_v1.sql` (indexes) after the episode-memory, walkway-camera and attention-loop-outcome migrations. Eval: `python services/orion-memory-consolidation/evals/run_memory_confirmation_replay_eval.py --scratch-dsn <throwaway admin DSN> [--live-dsn <read-only>]` (counts only).

## Turn change appraisal

Each persisted turn (after the first in a window) gets a logprob-calibrated `turn_change_appraisal` patch on `spark_meta`: novelty score, shift kind, confidence, and baseline mode (`prior_turn` or `session_window` fallback). The first turn in a window uses `turn_change_status=skipped` (no baseline, no LLM call). High-confidence novel turns also emit `OrionSignalV1` on `orion:signals:memory_consolidation`.

| Env | Default | Purpose |
|-----|---------|---------|
| `TURN_CHANGE_CLASSIFY_ROUTE` | `metacog_background` | Gateway route for classify RPC (`metacog_background`, `metacog`, or `quick`); background yields to live Mind metacog traffic |
| `TURN_CHANGE_CONFIDENCE_MARGIN` | `0.15` | Re-appraise vs session window when novelty margin is below this |
| `TURN_CHANGE_SUBSTRATE_THRESHOLD` | `0.65` | Minimum novelty to emit substrate signal |
| `TURN_CHANGE_WINDOW_TURNS` | `3` | Prior turns in session-window baseline |
| `CHANNEL_SIGNALS_PREFIX` | `orion:signals` | Bus prefix for organ signal publish |

## Bring-up

```bash
docker compose --env-file ../../.env --env-file ../orion-bus/.env -f docker-compose.yml up -d --build
```

Apply Postgres migrations: `services/orion-sql-db/manual_migration_memory_consolidation_v1.sql`, then `services/orion-sql-db/manual_migration_memory_episode_v1.sql` (shadow episodes + close audit; the service is fail-open without it).

## Smoke

```bash
PYTHONPATH=. python scripts/smoke_memory_consolidation_pipeline.py
bash scripts/smoke_memory_consolidation_gate.sh
```

Gate smoke runs deterministic unit tests (greeting skip + substantive propose). Pipeline smoke requires live stack (bus, sql-writer, postgres, llm-gateway, cortex).
