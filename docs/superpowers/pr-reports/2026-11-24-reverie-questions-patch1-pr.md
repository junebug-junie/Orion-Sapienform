## Summary

Reveries that ask, Patch 1 (spec `docs/superpowers/specs/2026-11-18-reverie-questions-design.md`, approved by Juniper; "hit it" 2026-11-24).

- A reverie now runs only in a **quiet stretch**: Orion's arousal reads `idle` and nothing has won attention for 10 minutes. At most one reverie per quiet stretch and 8 a day.
- Each reverie starts from **one piece of real material** that code picks: a named person's walkway routine that did not happen (clean vs unrecognised, never "did not come"), a self-bet calibration gap (n ≥ 20, gap ≥ 0.15, beyond 2 standard errors), or an old memory beside the newest one (centered cosine, ranks 5–25, live memories only).
- It leaves **one question** for curiosity, or says why there is none. Code refuses a ref the seed did not carry, a near-duplicate of an open question, and machinery words. One `reverie_question` row per reverie.
- An accepted question goes to the **curiosity focus slot**: origin forced to self, never replaces a pending focus, at most 3 a day. Investigate offer rows record `constants.reverie_focus = "v1"` from enable time; the three-arm draw and `select_priors` are untouched (#3183).
- The chronicle reads the table as the new `reverie_question` experience; reverie keeps reading the now and recording its cites (#3499).
- **Retired, kill means kill:** the coalition-winner trigger, the 4-step chain, refractory, EMA, no-coalition streak, resonance monitor and page, room-camera expectations and their judge, the `reverie_narrate` / `reverie_expectation_judge` verbs, the semantic-lift concern cards, the dream's REM compaction of reverie chains and its applier, and the `reverie.thought` / `reverie.chain` / `compaction-request` / `compaction-delta` / `resonance-alert` channels. The old tables stay (no DROP).

## Outcome moved

Reverie stops narrating gauges (88% of 1,337 thoughts in 7 days) and starts producing questions curiosity can dig. Before: about 190 reverie rows a day, 0 cites by any organ, 0 focus rows ever. After: at most 8 rows a day, each grounded in a resolvable seed, up to 3 a day reaching the focus slot. Live outcome is UNVERIFIED until deployed (see Risks).

## Current architecture

`services/orion-thought/app/chain.py` ran a reverie chain every 90 s whenever the attention broadcast had a winner: up to 4 `reverie_narrate` calls on the brain lane, a theme refractory store, an EMA, a resonance tripwire with orion-notify pages, room-camera expectations scored by a judge verb, and publishes on 4 bus channels. Consumers read `substrate_reverie_thought` / `_chain` directly (day letter, situation brief, chat-stance glimpse, Hub reverie tab and observability panel, dream REM compaction, self-monitor catalog).

## Architecture touched

- orion-thought: new poll (`app/reverie.py`) + pure gate/seeds/picker (`app/reverie_seeds.py`) + `reverie_question` store; chain/resonance/health monitor/grounding/broadcast/vision readers deleted.
- Contracts: `ReverieSeedV1`, `ReverieQuestionV1` (registered); `TemporalSourceKind.reverie_question`; `IntrospectToolBindingV1.invocation_context` gains `reverie`; migration `manual_migration_reverie_question.sql`.
- Hub: `queue_curiosity_focus` accepts `reverie`; offer-row constants; reverie tab and observability panel read `reverie_question`.
- orion-durable-runs: chronicle reads `reverie_question`; retired readers removed (replay-only adapters in `orion/temporal_self/history_sources.py` for frozen eval days).
- Consumers moved: day letter (`orion/orion_day/*`, letter template, Hub mailer/introspect), situation brief (`orion/situational/reverie_reader.py`), chat-stance glimpse (`felt_state_reader` lane `latest_reverie_question` + `chat_stance.py`), Prime's frame (sees reverie questions through the now), Hub reverie tab + observability.
- orion-dream: REM compaction, compaction store, applier and consent ask removed.
- cortex-exec: `reverie_narrate` dropped from the background-lane verb set; `reverie_question` routes on the agent lane by its own `llm_route`.

## Files changed

- `orion/schemas/reverie.py`, `orion/schemas/registry.py`: the two contracts.
- `services/orion-sql-db/manual_migration_reverie_question.sql`: additive table + 3 indexes, `REQUIRED-BY: orion-thought`.
- `orion/cognition/verbs/reverie_question.yaml`, `orion/cognition/prompts/reverie_question.j2`: the verb (agent lane, `recall_enabled: false`).
- `services/orion-thought/app/{reverie,reverie_seeds,store,settings,main}.py`: the new organ; retired modules deleted.
- `services/orion-thought/tests/test_reverie_questions.py`, `evals/test_reverie_question_check_eval.py`: gate tests and the refusal eval.
- `orion/reverie/questions.py`: one SQL + line renderer every consumer shares.
- `services/orion-hub/scripts/{curiosity_focus_listener,curiosity_investigation,main,reverie_routes,substrate_observability_routes}.py`, `static/js/reverie-tab.js`, `templates/index.html`: focus caller, switch constant, reverie surfaces.
- `orion/temporal_self/{sources,history_sources,now,arcs}.py`, `services/orion-durable-runs/app/temporal_self_sources.py`, `config/integration/organs.yaml`: experience contract.
- `orion/orion_day/*`, `orion/schemas/orion_day.py`, `orion/situational/*`, `orion/substrate/felt_state_reader.py`, `services/orion-cortex-exec/app/{chat_stance,llm_lane,substrate_felt_state_reader}.py`: consumers.
- `orion/bus/channels.yaml`, `config/physiology/contract.yaml`, `config/metrics/metric_definitions.lock.json`, `orion/metrics/semantics.py`: contract registries re-locked.
- `services/orion-dream/*`: REM compaction retirement.
- `scripts/smoke_reverie_seeds.py`: live read-only smoke.
- `scripts/sql/2026-11-18_grant_orion_readonly_reverie_question_{trace,table}.sql`: headers (trace grant APPLIED live 2026-11-18 after Juniper approved; table grant approved, applied after the migration).

## Schema / bus / API changes

- Added: `ReverieSeedV1`, `ReverieQuestionV1`, table `reverie_question`, `TemporalSourceKind` `reverie_question`, `invocation_context` `reverie`, `GET /api/reverie/questions/recent`, `OrionDayMaterialV1.reverie_questions`, felt-state lane `latest_reverie_question`.
- Removed: channels `orion:reverie:thought`, `orion:reverie:chain`, `orion:dream:compaction-request`, `orion:dream:compaction-delta`, `orion:reverie:resonance-alert`; verbs `reverie_narrate`, `reverie_expectation_judge`; `GET /api/reverie/text/recent`; dream `/dreams/rem-preview`.
- Behavior changed: the reverie trigger (quiet gate, not a coalition winner); `ORION_REVERIE_INTERVAL_SEC` is now the gate's poll (60 s).
- Compatibility: `reverie_chain` / `expectation_verdict` stay in `TemporalSourceKind` so history rows parse; old schema models stay registered for history. Additive literals on forbid models: **deploy orion-hub, orion-durable-runs' now readers (orion-thought, orion-hub, orion-mind, orion-prediction) before orion-thought / orion-durable-runs emit the new values.** `OrionDayMaterialV1.reverie_questions`: deploy orion-hub and orion-durable-runs together, outside the letter's 08:30 window.

## Env/config changes

- Added keys (orion-thought): `ORION_REVERIE_QUIET_MIN_CALM_SEC=600`, `ORION_REVERIE_MAX_PER_DAY=8`, `ORION_REVERIE_FOCUS_MAX_PER_DAY=3`, `ORION_REVERIE_FAR_MEMORY_BAND=5:25`, `ORION_REVERIE_THREAD_LAPSE_DAYS=14`, `ORION_REVERIE_FAR_MEMORY_CHROMA_URL`, `ORION_REVERIE_TIMEZONE`. (orion-hub): `ORION_REVERIE_FOCUS_MAX_PER_DAY=3`.
- Changed: `ORION_REVERIE_INTERVAL_SEC=60` (was 90), `ORION_REVERIE_ENABLED` defaults on.
- Removed (orion-thought): `ORION_REVERIE_MIN_SALIENCE`, `CHANNEL_REVERIE_THOUGHT`, semantic-lift, metacog-background, perception, street, expectation, chain, refractory, no-coalition, chain-arousal, ground-episodes, compaction-request, resonance keys, `ORION_ATTENTION_SALIENCE_V2_ENABLED`, `CHANNEL_ATTENTION_SALIENCE_TRACE`, `SELF_READINGS_REVERIE_ENABLED`, recent-arcs keys, `ORION_VISION_EVENTS_LEGACY_CUTOFF`, `ORION_CAMERA_DESCRIPTIONS`. (orion-dream): REM and compaction-apply keys.
- `.env_example` updated: yes. Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (8 keys added); `ORION_REVERIE_INTERVAL_SEC` set 90 → 60 by hand (the sync reported it diverged). Retired keys left in local `.env` are inert.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-thought  pytest tests evals                     592+ passed, 31 skipped
services/orion-hub      focus, reverie routes, observability, orion_day, outreach, offer decisions   622 passed
services/orion-durable-runs pytest tests (PYTHONPATH=repo)     1074 passed, 156 skipped
services/orion-dream    pytest tests                           316 passed, 9 skipped
services/orion-sql-writer pytest tests                         1334 passed, 250 skipped
services/orion-cortex-exec touched + situation + felt-state     263 passed, 1 failed (test_stale_row_not_injected: fails identically on main 043cb52c7, pre-existing)
services/orion-recall test_recall_bounded_retrieval            42 passed
services/orion-actions walkway tests                           46 passed
orion/{reverie,orion_day,situational,prime,temporal_self,integration/evals,self_monitor,curiosity,schemas,physiology,dig} + tests/ + scripts/analysis/tests   all passed
```

Spec acceptance checks covered as gate tests: 2 (gate: idle+calm, engaged+calm, idle with a winner, short calm, plus stretch/cap/lane refusals), 3 (each seed from a fixture), 4 (bad ref, near-duplicate, machinery words), 5 (cap holds, never replaces a pending focus, origin forced self: `services/orion-hub/tests/test_curiosity_focus.py`), 6 (superseded memory never served), 8 (retired producers gone; integration seam, bus channels, schema registry gates).

## Evals run

```text
services/orion-thought/evals/test_reverie_question_check_eval.py   12/12 labeled answers classified right (accuracy 1.00)
orion/integration/evals                                            11 passed (now-ablation fixtures regenerated for the new prompt)
```

## Docker/build/smoke checks

```text
python3 scripts/smoke_reverie_seeds.py   (live DB via the published port, Chroma at :8500; read-only)
[absence]    6 candidates; evidence=unrecognised; ref vision_percept_expectation:a1ea8bc8-... -> row status=missed
[lost_bet]   11 candidates; turn.self.more_steps_than_usual 0.1-0.2: stated 0.10, happened 34%, n=338; ref -> 338 settled bets
[far_memory] anchor episode_memory (2026-11-23) + partner episode_memory (2026-07-24) at rank 13, centered sim 0.20; both refs -> live record
exit 0
```

Static gates (`.github/workflows/orion-static-gates.yml`, every step run locally): green after re-locking metric definitions, registering `app/reverie.py` in the body-economy contract and dropping the retired prompt site. No Docker build was run (no image change beyond source).

## Review findings fixed

Code review not run by this build; the orchestrator runs it.

## Restart required

Apply the migration first (the post-pull rebuild applies it automatically; it is additive), then the table grant by hand:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < scripts/sql/2026-11-18_grant_orion_readonly_reverie_question_table.sql
```

Deploy order: orion-hub, orion-mind, orion-prediction, orion-cortex-exec, orion-dream, then orion-durable-runs and orion-thought (the post-pull rebuild from clean main does this).

## Risks / concerns

- Severity: medium. Concern: the agent-lane call from orion-thought has never run live; if the gateway refuses `agent` without a pool hold, every reverie is a lane refusal (logged `reverie_skipped reason=llm_unavailable` / `lane_refused_stretch`, no rows). Mitigation: Orion checks `lane_refused_stretch` live; UNVERIFIED.
- Severity: medium. Concern: the visual (painting) chain still reads the latest narrated reverie interpretation as a context seed (`services/orion-thought/app/store.py load_latest_reverie_interpretation`). Painting was out of scope, so that read is left on the old table; with no new rows it will age out and the slot goes empty. Mitigation: ticket #3813.
- Severity: low. Concern: the reverie-scope `AttentionSalienceTraceV1` emit stopped with the narration tick; chat-scope traces continue. Readers that split by scope see no new reverie rows.
- Severity: low. Concern: `config/physiology/consent.yaml` still carries Juniper's standing consent for dream compaction downscale; its only bidder (the applier) is retired. Left as granted; removing a grant is her call.
- Severity: low. Concern: far-memory anchor deviates from the spec (newest kept episode memory, not "the most salient item in the now", which has no vector). Documented in `reverie_seeds.pick_anchor`.
- Severity: low. Concern: the near-duplicate matcher is the reverie organ's own word-set Jaccard ≥ 0.6 (the inquiry manager's embedding matcher is design-only); a ref used in the last 14 days is "spent", so a standing gap is not re-asked daily (spec relied on the duplicate refusal for that).
- Issues: closes #3414 (resonance monitor retired). Opened: #3813 (painting context seed).

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
