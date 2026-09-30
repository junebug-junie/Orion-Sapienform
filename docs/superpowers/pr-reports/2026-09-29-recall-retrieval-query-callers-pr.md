# feat(recall): callers send what to search for (retrieval_query, context_only)

**Stacked on PR #2416 (`feat/recall-bounded-retrieval`). This PR must merge and deploy after #2416.**

## Summary

This is Phase 3 of the recall retrieval design (`docs/superpowers/specs/2026-09-29-recall-retrieval-query-architecture-design.md`), with Juniper's 2026-09-29 answers. #2416 taught recall to accept "what to search for" separately from the turn text. This PR makes the callers actually say it.

- **Self-inquiry and curiosity turns recall about their standing question**, not the 30k-character instruction prompt:
  - self-inquiry uses the picked question;
  - an urgent run uses the seed's question;
  - a world-curiosity run uses its continuation note.
  The question rides the durable run brief, so it survives a Hub restart. If a run has none, the turn falls back to the Mind appraisal (the old in-memory dict), then to nothing.
- **World-pulse reading recalls about `"<source title> — <stage-1 claim>"`**, capped at 1000 characters. Stage 1 has no claim yet, so it searches the title alone. The text is stored on the reading brief, alongside the prompt.
- **stance_react's second recall (phase 3) reuses exactly the search text its first recall (phase 0+1) used.** It does not re-derive it.
- **The reverie verbs send `mode="context_only"`** instead of an empty search. They get only the "what is going on" feeds.
- **Every cortex-exec recall now sends `deadline_ms`**, set to the RPC wait it actually uses, so recall can stop at 80% of it (#2416).

## Outcome moved

The failure this fixes: `stance_react` recall latency tracks query length (corr 0.98). The 72-second recall was a 30,663-character prompt that recall had to guess a topic from.

After this PR plus #2416:
- self-initiated turns hand recall a question under 1,000 characters;
- reverie stops pretending to search;
- recall knows the caller's deadline.

This also changes **what** Orion remembers during reflection and reading. That change is the point of Juniper's answers, and it is a cognition change, not only a speed change.

## Current architecture

Before this patch:
- Every cortex-exec recall used the last user message as the search text (`executor.py` `run_recall_step`).
- For self-initiated turns, that was the stance message: the Mind appraisal when an in-memory dict hit, otherwise the whole prompt.
- Reading turns never passed an appraisal.
- stance_react recalled twice on the same text.
- The reverie verbs sent empty queries because the router enables recall by default.

## Architecture touched

- **Contracts** (all additive, optional):
  - `StanceReactRequestV1.retrieval_query`
  - `CuriosityRunBriefV1.retrieval_query`
  - `CuriosityTurnRequestV1.retrieval_query`
  - `ReadingRunBriefV1.retrieval_query`
- **orion-hub**: `execute_unified_turn(retrieval_query=...)`, the curiosity loop, and the reading pipeline and listener.
- **orion-thought**: the stance listener copies the value to `ctx["retrieval_query"]`.
- **orion-durable-runs**: the runner forwards the brief's value onto the Hub turn request.
- **orion-cortex-exec**: `run_recall_step` (every recall path goes through it: router pre-recall, inline step, PCR 0+1, PCR 3, stance grounding, supervisor), `pcr_chat_memory`, and the router's verb recall mode.
- **orion-cortex-orch** (shared `orion/cognition`): `build_recall_query_v1` and Mind prefetch.
- **Verb YAML**: new `recall_mode:` key, read by `plan_loader` into plan metadata `recall_mode_default`.

## Files changed

- `orion/cognition/recall_query.py`: shared helpers `cap_retrieval_query`, `retrieval_query_from_ctx` and `recall_mode_from_cfg`. The builder prefers `ctx["retrieval_query"]` and accepts `deadline_ms`.
- `orion/cognition/recall_prefetch.py`: sends `deadline_ms`. It no longer bails out when only a retrieval query exists.
- `orion/cognition/plan_loader.py`, `orion/cognition/verbs/reverie_narrate.yaml`, `reverie_expectation_judge.yaml`: add `recall_mode: context_only`.
- `services/orion-cortex-exec/app/executor.py`: `run_recall_step` sends `retrieval_query`, `deadline_ms` and `mode`. `fragment` stays the turn text.
- `services/orion-cortex-exec/app/pcr_chat_memory.py`: phase 0+1 records its search text, and phase 3 reuses it.
- `services/orion-cortex-exec/app/router.py`: fills `recall_cfg["mode"]` from the verb default unless the caller set one.
- `orion/schemas/thought.py`, `services/orion-thought/app/bus_listener.py`: stance contract field and ctx hand-off.
- `orion/hub/turn_orchestrator.py`: new `retrieval_query` parameter, sent on the stance request.
- `orion/schemas/durable_run.py`, `services/orion-durable-runs/app/graph.py`: durable carriage for curiosity runs.
- `services/orion-hub/scripts/curiosity_investigation.py`: standing question per line, carried on the brief and on in-process turn requests, with the appraisal as fallback.
- `orion/schemas/reading_turn.py`, `orion/world_pulse_read/durable.py`, `services/orion-hub/scripts/world_pulse_read_pipeline.py`, `world_pulse_read_stage2.py`, `reading_turn_listener.py`: `reading_retrieval_query(seed, claim)` and its carriage.
- Tests (new):
  - `services/orion-cortex-exec/tests/test_recall_retrieval_query_callers.py`
  - `services/orion-thought/tests/test_stance_context_retrieval_query.py`
  - `services/orion-hub/tests/test_curiosity_recall_retrieval_query.py`
  - `services/orion-durable-runs/tests/test_turn_retrieval_query.py`
  - `orion/cognition/tests/test_recall_query_retrieval_query.py`
  - `orion/world_pulse_read/tests/test_reading_retrieval_query.py`
- Tests (appended): the stance request tests in `test_turn_orchestrator_utterance_origin.py`, and the reading tests in `test_world_pulse_read_pipeline.py` and `test_world_pulse_read_stage2.py`.
- Test stubs: the shared `_generate` stub in `test_curiosity_investigation.py` and the reading `held_model_boundary` fixtures now accept the new keyword.

## Schema / bus / API changes

- Added: `retrieval_query: str | None` (max 1000) on:
  - `StanceReactRequestV1` (not forbid);
  - `CuriosityRunBriefV1`, `CuriosityTurnRequestV1` and `ReadingRunBriefV1` (all `extra="forbid"`).
- Added: the verb YAML key `recall_mode`, and the plan metadata key `recall_mode_default`.
- Removed / renamed: none.
- Behavior changed:
  - The recall search text for self-initiated turns and reading turns.
  - stance phase 3 search text (it now reuses phase 0+1's).
  - The reverie recall mode.
  - `deadline_ms` is now always sent by cortex-exec, and by orch prefetch.
- No new channels. Registry entries point at the same classes, so no registry edit was needed.
- **Compatibility and deploy order** (consumer first):
  1. **orion-recall** (#2416). It must accept the `retrieval_query`, `deadline_ms` and `mode` fields on `RecallQueryV1`, which is a forbid model. Otherwise every cortex-exec recall is rejected.
  2. **orion-durable-runs**. The brief and turn-request models are forbid. An old runner rejects a curiosity brief that carries the field, and Hub falls back to running the turn in-process. An old runner also rejects any reading request, because `poll_turn` dumps without `exclude_none`. That surfaces as `ReadingPending` and is retried next tick, not lost.
  3. **orion-thought**, **orion-cortex-exec** and **orion-cortex-orch**. The reverie `context_only` default needs both thought (which builds the plan) and cortex-exec (whose router reads it).
  4. **orion-hub** last.

## Env/config changes

- Added / removed / renamed keys: none in this PR's own changes.
- `.env_example` updated: no. The merged base (#2416, ef65e239f) changed `services/orion-recall/.env_example`.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, run after the merge. The primary checkout's `services/orion-recall/.env` now has:
  - `RECALL_MAX_QUERY_CHARS`
  - `RECALL_MAX_SUB_QUERIES`
  - `RECALL_DEADLINE_MS_DEFAULT`
  - `RECALL_FETCH_CONCURRENCY`
  - `RECALL_BROWSE_SHORTCUT_ENABLED`
- Skipped keys requiring operator action: none.

## Tests run

Baseline runs came from a separate detached checkout of the base, so they were never measured against a tree being edited.

Every run below was made after merging base commit ef65e239f. "Before" means the base branch with none of this PR's changes.

The test results come down to three points:
- **No new failures anywhere.**
- Every service's passing count went up by exactly the number of tests this PR adds.
- Every remaining failure also fails on the base branch.

```text
orion-cortex-exec  (the whole suite cannot load in one run on the base branch: "Verb already registered: legacy.plan";
                    so it was run one test file at a time, 149 files)
  before: 6 files failing: chat_kids_story_plumbing, chat_quick_plumbing, cognition_trace_verb_runtime,
          main_autonomy_graph_probe, situation_prompt_integration, story_weave_smoke
  after:  the same 6 files failing, nothing else; new test_recall_retrieval_query_callers.py: 14 passed
orion-thought      before: 1 failed, 492 passed   after: 1 failed, 497 passed
                   (same pre-existing failure: test_settings_mind_enrichment::test_mind_enrichment_defaults_off)
orion-durable-runs before: 188 passed, 64 skipped   after: 191 passed, 64 skipped
orion-hub          before: 38 failed, 3119 passed   after: 38 failed, 3131 passed
                   (the same 38 tests fail in both; none newly fails and none newly passes. One of the 38 is
                    test_turn_orchestrator_utterance_origin::test_execute_unified_turn_uses_mind_appraisal_text_for_stance_not_harness,
                    which fails on the base too because Mind work-shape text is added to the start of the prompt)
orion/cognition/tests (skipping 3 files that also fail to load on the base branch: test_packs, test_planner,
                       test_reflect_recall_integration)
                   before: 84 passed   after: 90 passed
orion/world_pulse_read/tests  before: 92 passed   after: 98 passed
orion/autonomy/tests/test_policy_act.py + tests/test_recall_bounded_retrieval_contract.py: 43 passed
```

Every static gate in `.github/workflows/orion-static-gates.yml` was run locally and passed:
- env-sync tests;
- the schema registry;
- substrate ladder liveness and schema-skew discovery;
- the grammar producer catalog;
- field topology;
- substrate requests;
- `check_metric_lineage --gate` and `check_definition_drift --gate` (both PASS, both re-run after the merge; nothing needed re-locking);
- inner-state registry;
- stdlib shadowing;
- the two graphify tests, with no graphify output committed;
- hostname references;
- relative mounts;
- claude.json mounts;
- journal dispatch;
- schedule collisions (report-only);
- sentience instruments (static);
- system_health producers;
- control-surface parity;
- async routes;
- chat-route poachers;
- the Hub `node --test` suite.

## Evals run

No eval in this PR. #2416 ships the recall latency and quality eval over stored `recall_telemetry` queries (`services/orion-recall/evals/`). That eval measures recall given a query. What this PR changes is which query each caller sends, and the only honest measure of that is live `recall_telemetry` after deploy:
- `retrieval_query_source='caller'` rows for `stance_react`;
- `query_chars` of 1000 or less;
- `mode='context_only'` rows for the reverie verbs.

This is an eval gap, proposed as a follow-up: a telemetry check that no `stance_react` row has `query_chars` over 1000 when `retrieval_query_source='caller'`.

## Docker/build/smoke checks

Not run. The task instructions say not to deploy or restart anything. Live behavior is **UNVERIFIED**.

## Review findings fixed

- Finding (found by this PR's own test before review): phase 3 re-derived the search text whenever phase 0+1 had none. `None` meant "derive from ctx" to `run_recall_step`, so a ctx rewrite between the phases changed phase 3's search.
  - Fix: `pcr_retrieval_query` returns `""` to mean "explicitly none", and `run_recall_step` sends that as `None`.
  - Evidence: `test_pcr_phase3_reuses_none_when_phase01_had_no_caller_query` failed before the fix and passes after it.
- Code-review subagent: not run by this agent (the orchestrator runs it).

## Restart required

Do not run this until #2416 is merged and deployed. Order matters (see Compatibility).

```bash
scripts/safe_docker_build.sh orion-recall up -d --build        # PR #2416
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- **Severity: medium.** A fresh world-curiosity kickoff (no continuation note) has no standing question. Orion has not picked one yet. That turn falls back to the Mind appraisal, which is mostly boilerplate ("Investigation claim: not yet chosen."). This is the same text recall searched before whenever the in-memory dict hit, and #2416's intake guard bounds it. It is still a weak search. A better answer needs a decision on what a question-less kickoff should recall about.
- **Severity: medium.** Rolling back only orion-durable-runs or only orion-hub after deploy leaves stored rows the old code rejects:
  - checkpointed curiosity briefs;
  - reading `request_json` rows, which contain `"retrieval_query"` (even when it is `null`, because they are stored without `exclude_none`).
  Roll back hub first, and let in-flight reading and curiosity runs drain before rolling back the runner.
- **Severity: low.** The search text changes what Orion remembers during self-inquiry, curiosity and reading. Juniper approved this change. It is still a cognition change, and there is no quality eval for it yet (see Evals).
- **Severity: low.** Reading seeds that are pinned documents and have no title search only the claim at stage 2, and nothing at stage 1 (recall then condenses the prompt), because an opaque document reference is not a useful search.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
