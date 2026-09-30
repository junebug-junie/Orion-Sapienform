# feat(recall): callers send what to search for (retrieval_query, context_only)

**Stacked on PR #2416 (`feat/recall-bounded-retrieval`). This PR must merge and deploy after #2416.**

## Summary

This is Phase 3 of the recall retrieval design (`docs/superpowers/specs/2026-09-29-recall-retrieval-query-architecture-design.md`), with Juniper's 2026-09-29 answers. #2416 taught recall to accept "what to search for" separately from the turn text. This PR makes the callers actually say it.

- **Self-inquiry and curiosity turns recall about their standing question**, not the 30k-character instruction prompt:
  - self-inquiry uses the picked question;
  - an urgent run uses the seed's question;
  - a world-curiosity run uses its continuation note.
  The question rides the durable run brief, so it survives a Hub restart. A run with no standing question (a fresh world-curiosity kickoff) sends none, and recall condenses the prompt itself. It never sends the kickoff's boilerplate appraisal as if it were a question.
- **Outreach turns recall about what they are going to say**:
  - curiosity outreach composition uses the finding it is composing about;
  - endogenous outreach uses its talkable content (open priors, curiosity summaries, daydream).
  Both are capped at 1000 characters, and both send nothing when there is nothing to talk about.
- **World-pulse reading recalls about `"<source title> — <stage-1 claim>"`**, capped at 1000 characters. Stage 1 has no claim yet, so it searches the title alone. The text is stored on the reading brief, alongside the prompt.
- **stance_react's second recall (phase 3) reuses exactly the search text its first recall (phase 0+1) used.** It does not re-derive it.
- **The reverie verbs send `mode="context_only"`** instead of an empty search. They get only the "what is going on" feeds. The setting lives under its own key (`recall_cfg["query_mode"]`), because `recall_cfg["mode"]` already means something else.
- **The search text never enters the stance LLM's prompt.** It rides the stance request to cortex-exec's recall calls only.
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
- **orion-hub**: `execute_unified_turn(retrieval_query=...)`, the curiosity loop (including outreach composition), endogenous outreach, and the reading pipeline and listener.
- **orion-thought**: the stance listener copies the value to `ctx["retrieval_query"]`.
- **orion-durable-runs**: the runner forwards the brief's value onto the Hub turn request.
- **orion-cortex-exec**: `run_recall_step` (every recall path goes through it: router pre-recall, inline step, PCR 0+1, PCR 3, stance grounding, supervisor), `pcr_chat_memory`, and the router's verb query mode.
- **orion-cortex-orch** (shared `orion/cognition`): `build_recall_query_v1` and Mind prefetch.
- **Verb YAML**: new `recall_query_mode:` key. `plan_loader` reads it into plan metadata `recall_query_mode_default`, and the router copies that into `recall_cfg["query_mode"]`. It deliberately does not use `recall_cfg["mode"]`, which is `RecallDirective.mode` ("hybrid"/"deep"/"graph").

## Files changed

- `orion/cognition/recall_query.py`: shared helpers `cap_retrieval_query`, `retrieval_query_from_ctx`, `recall_query_mode_from_cfg` and `RECALL_QUERY_MODE_KEY`. The builder prefers `ctx["retrieval_query"]` and accepts `deadline_ms`.
- `orion/cognition/recall_prefetch.py`: sends `deadline_ms`. It no longer bails out when only a retrieval query exists, or for `context_only` with no text (same rule as the builder).
- `orion/cognition/plan_loader.py`, `orion/cognition/verbs/reverie_narrate.yaml`, `reverie_expectation_judge.yaml`: add `recall_query_mode: context_only`.
- `services/orion-cortex-exec/app/executor.py`: `run_recall_step` sends `retrieval_query`, `deadline_ms` and `mode`. `fragment` stays the turn text.
- `services/orion-cortex-exec/app/pcr_chat_memory.py`: phase 0+1 records its search text, and phase 3 reuses it.
- `services/orion-cortex-exec/app/router.py`: fills `recall_cfg["query_mode"]` from the verb default unless the caller set one.
- `orion/schemas/thought.py`, `services/orion-thought/app/bus_listener.py`: stance contract field and ctx hand-off.
- `orion/hub/turn_orchestrator.py`: new `retrieval_query` parameter, sent on the stance request. It is not copied into `stance_inputs`, because the stance prompt template renders those.
- `orion/schemas/durable_run.py`, `services/orion-durable-runs/app/graph.py`: durable carriage for curiosity runs. The forbid models drop the key from every dump when it is unset (see Compatibility).
- `services/orion-hub/scripts/curiosity_investigation.py`: standing question per line, carried on the brief and on in-process turn requests (no appraisal fallback). The outreach composition turn searches its finding.
- `services/orion-hub/scripts/endogenous_outreach.py`: `outreach_retrieval_query(ctx)`, threaded through `_generate` into both lane attempts.
- `orion/schemas/reading_turn.py`, `orion/world_pulse_read/durable.py`, `services/orion-hub/scripts/world_pulse_read_pipeline.py`, `world_pulse_read_stage2.py`, `reading_turn_listener.py`: `reading_retrieval_query(seed, claim)` and its carriage.
- Tests (new):
  - `services/orion-cortex-exec/tests/test_recall_retrieval_query_callers.py`
  - `services/orion-thought/tests/test_stance_context_retrieval_query.py`
  - `services/orion-hub/tests/test_curiosity_recall_retrieval_query.py`
  - `services/orion-durable-runs/tests/test_turn_retrieval_query.py`
  - `orion/cognition/tests/test_recall_query_retrieval_query.py`
  - `orion/world_pulse_read/tests/test_reading_retrieval_query.py`
- Tests (appended): the stance request tests in `test_turn_orchestrator_utterance_origin.py`, and the reading tests in `test_world_pulse_read_pipeline.py` and `test_world_pulse_read_stage2.py`.
- Test stubs: these stand-in functions now accept the new keyword:
  - the shared `_generate` stub in `test_curiosity_investigation.py`;
  - the reading `held_model_boundary` fixtures;
  - the six `_generate` stubs in `test_endogenous_outreach.py`;
  - the three stubs in `services/orion-hub/evals/test_reading_handoff_eval.py`.

## Schema / bus / API changes

- Added: `retrieval_query: str | None` (max 1000) on:
  - `StanceReactRequestV1` (not forbid);
  - `CuriosityRunBriefV1`, `CuriosityTurnRequestV1` and `ReadingRunBriefV1` (all `extra="forbid"`).
- Added: the verb YAML key `recall_query_mode`, the plan metadata key `recall_query_mode_default`, and the exec `recall_cfg` key `query_mode`.
- Removed / renamed: none.
- Behavior changed:
  - The recall search text for self-initiated turns and reading turns.
  - stance phase 3 search text (it now reuses phase 0+1's).
  - The reverie recall mode.
  - `deadline_ms` is now always sent by cortex-exec, and by orch prefetch.
- No new channels. Registry entries point at the same classes, so no registry edit was needed.
- **Compatibility and deploy order.** Checked in both directions for every new field. The dangerous direction is a new producer talking to an old consumer.
  - **Unset fields are absent from every dump.** `ReadingRunBriefV1`, `CuriosityRunBriefV1` and `CuriosityTurnRequestV1` all reject unknown keys, even ones set to null. Each now drops `retrieval_query` from every dump when it is unset (the same rule `orion/schemas/reading.py` already uses). That covers all four dump sites:
    - the runner's reading RPC to Hub;
    - Hub's stored `reading_durable_turn.request_json`;
    - Hub's POST to the runner;
    - the runner's checkpointed curiosity brief.
    Before this fix, a new runner sent `"retrieval_query": null` in every reading turn, and the old Hub rejected all of them. That was the review BLOCKER.
  - **A set value still needs a new consumer.** Hub always sets it for self-inquiry, urgent runs, and every reading turn with a title or URL. So consumers deploy first:
    1. **orion-recall** (#2416). This is a hard requirement. `RecallQueryV1` rejects unknown keys, and new cortex-exec always sends `deadline_ms` and `mode`. A new cortex-exec (or orch) against an old recall fails every recall.
    2. **orion-durable-runs**. Before Hub. An old runner rejects a brief that carries a set query: a curiosity kickoff then falls back to an in-process turn, and a reading turn is retried as `ReadingPending`. A new runner against an old Hub is now safe, because the old Hub never sets the field and the runner omits it when unset.
    3. **orion-thought**, **orion-cortex-exec** and **orion-cortex-orch**.
       - `StanceReactRequestV1` accepts unknown keys, so an old thought just ignores the field (recall then condenses the prompt).
       - Reverie's `context_only` default needs both thought (which builds the plan metadata) and cortex-exec (whose router reads it). If either is old, reverie keeps today's empty-query behavior, which is harmless.
    4. **orion-hub** last.
  - **Rollback.** An unset field is invisible to old code. A run that already stored a set value (a checkpointed curiosity brief, or a reading `request_json`) is rejected by an older runner or Hub. Roll back hub first, and let in-flight runs finish before rolling back durable-runs.

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

Every run below was made after merging base commit ef65e239f, and re-run after the review fixes. "Before" means the base branch with none of this PR's changes.

The test results come down to three points:
- **No new failures anywhere.**
- Every service's passing count went up by exactly the number of tests this PR adds.
- Every remaining failure also fails on the base branch.

```text
Final numbers, after the review fixes (fd62194aa..a50486c9d):
orion-cortex-exec  (the whole suite cannot load in one run on the base branch: "Verb already registered: legacy.plan";
                    so it was run one test file at a time, all 148 files)
  before: 6 files failing: chat_kids_story_plumbing, chat_quick_plumbing, cognition_trace_verb_runtime,
          main_autonomy_graph_probe, situation_prompt_integration, story_weave_smoke
  after:  the same 6 files failing, nothing else; test_recall_retrieval_query_callers.py: 15 passed
orion-thought (tests + evals)  before: 1 failed, 492 passed   after: 1 failed, 502 passed
                   (same pre-existing failure: test_settings_mind_enrichment::test_mind_enrichment_defaults_off)
orion-durable-runs before: 188 passed, 64 skipped   after: 194 passed, 64 skipped
orion-hub tests    before: 38 failed, 3119 passed   after: 38 failed, 3138 passed
                   (the same 38 tests fail in both; none newly fails. One of the 38 is
                    test_turn_orchestrator_utterance_origin::test_execute_unified_turn_uses_mind_appraisal_text_for_stance_not_harness,
                    which fails on the base too because Mind work-shape text is added to the start of the prompt)
orion-hub evals    31 passed, 3 skipped (9 had been failing since 1c516cca3; fixed in a50486c9d)
orion/cognition/tests + orion/world_pulse_read/tests + test_policy_act + the #2416 contract test:
                   233 passed (in orion/cognition/tests, 3 files that fail to load on the base branch too were
                   skipped: test_packs, test_planner, test_reflect_recall_integration)
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

From the code review of this PR (a separate review run):

- **BLOCKER. Reading turns would break between the durable-runs deploy and the hub deploy.** The runner's reading RPC (`runner.py` `_run_reading_turn`) dumped the whole request, so an unset `brief.retrieval_query` went to a not-yet-upgraded Hub as `null`. The old Hub's `ReadingTurnRequestV1` rejects unknown keys, so every reading turn would fail while the runner held a GPU lease for up to about 2.5 hours.
  - Fix (`d30dfeb37`): the three forbid models drop `retrieval_query` from every dump when it is unset. This follows the existing `orion/schemas/reading.py` pattern. I audited every dump of a new field in this PR; see Compatibility, which now covers the new-producer-to-old-consumer direction as well.
  - Evidence: `services/orion-durable-runs/tests/test_turn_retrieval_query.py::test_new_runner_reading_turn_validates_on_an_old_hub` drives the real `DurableRunner._run_reading_turn` and validates what it sends against a copy of the old Hub model that has no `retrieval_query` field. It fails without the fix. So does `test_unset_query_is_absent_from_every_dump_of_every_forbid_model`, which covers the stored JSON, the POST and the checkpoint. Both pass with it.
- **SHOULD. The standing question leaked into the stance LLM prompt.** The Hub copied it into `stance_inputs`, and `stance_react.j2` renders every `stance_inputs` key as "additional context". That was an unreviewed change to a cognition prompt.
  - Fix (`fc49600b3`): the query rides only `StanceReactRequestV1.retrieval_query`, and orion-thought puts it on the top-level exec ctx, never in `stance_inputs`. I also fixed the misleading "cockpit trace" comment.
  - Evidence: `test_stance_context_retrieval_query.py::test_retrieval_query_never_reaches_the_stance_prompt` renders the real `stance_react.j2` from the real thought ctx and asserts the query text is absent. I checked that the template does render the text when it is in `stance_inputs`. The Hub test asserts `stance_inputs` has no `retrieval_query`.
- **SHOULD. Outreach paths were uncovered, and the kickoff query was mislabelled.**
  - Uncovered: curiosity outreach composition and endogenous outreach (both lane attempts) sent the whole self-authored prompt as the search text. Fix (`78480bb27`): they now send the finding being composed about, or the talkable content (open priors, curiosity summaries, daydream). The text is whitespace-collapsed and capped at 1000 characters, or omitted when there is nothing to talk about.
  - Mislabelled: `_generate` fell back to the Mind appraisal. At kickoff that is `build_investigation_subject` boilerplate, and it would have been recorded as `retrieval_query_source='caller'`. Fix (same commit): the caller query is now the durable standing question or nothing. The appraisal still goes to Mind and stance unchanged.
  - Evidence:
    - `test_curiosity_investigation.py::test_outreach_composition_recalls_about_the_finding_not_the_prompt` and `::test_outreach_composition_with_no_finding_sends_no_query`;
    - `test_endogenous_outreach.py`: `test_outreach_retrieval_query_is_the_talkable_content`, `..._none_without_talkable_content`, `test_outreach_tick_sends_its_subject_not_the_prompt`, and `test_real_generate_threads_retrieval_query_into_both_lane_attempts` (real `_generate`, agent lane fails, chat fallback carries the same query);
    - `test_curiosity_recall_retrieval_query.py::test_turn_without_a_standing_question_sends_none_not_the_boilerplate_appraisal` and `::test_fresh_investigation_kickoff_sends_no_caller_query_end_to_end`.
- **NIT. The new mode collided with the existing `recall_cfg["mode"]`.** That key already means `RecallDirective.mode`. cortex-orch always sends `"hybrid"`, which silently suppressed reverie's `context_only` default.
  - Fix (`5bcd123cd`): the new mode moved to its own key: `recall_cfg["query_mode"]`, plan metadata `recall_query_mode_default`, verb YAML `recall_query_mode`.
  - Evidence: `test_recall_retrieval_query_callers.py::test_orch_routed_recall_directive_keeps_the_verb_context_only_default` runs the real router with the reverie plan and `RecallDirective().model_dump()` (`mode="hybrid"`), and asserts that every recall sent is `context_only`. `test_router_caller_query_mode_beats_verb_default` now uses a realistic caller key. The cognition test asserts `recall_cfg["mode"]` is never read as the query mode.
- **NIT. Prefetch and the builder disagreed on `context_only`.** Prefetch refused a textless `context_only` request that the builder accepted.
  - Fix (`2a2347506`): prefetch allows it too.
  - Evidence: `test_recall_query_retrieval_query.py::test_prefetch_and_builder_agree_context_only_needs_no_text` (prefetch sends `mode=context_only`), and `::test_prefetch_still_refuses_an_empty_retrieve_query`.
- **Found while re-running evals for the review.** Since `1c516cca3`, 9 of the 11 tests in `services/orion-hub/evals/test_reading_handoff_eval.py` failed with a `TypeError`: their stand-in `_generate` did not accept the reading turn's new keyword. I had not run the hub evals in the first round.
  - Fix (`a50486c9d`): the stubs accept it.
  - Evidence: `services/orion-hub/evals`: 31 passed, 3 skipped.

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

- **Severity: medium.** A fresh world-curiosity kickoff (no continuation note) has no standing question, because Orion has not picked one yet. It sends none, and recall condenses the prompt using #2416's intake guard. That is bounded, but it is still a weak search. A better answer needs a decision on what a question-less kickoff should recall about.
- **Severity: medium.** Rollback: stored rows with a set query are rejected by older code. These are checkpointed curiosity briefs and reading `request_json` rows. (Rows with an unset query no longer carry the key.) Roll back hub first, and let in-flight reading and curiosity runs finish before rolling back the runner.
- **Severity: low.** The endogenous outreach search text joins every talkable item. When there are several priors and summaries, recall gets a multi-topic query (capped at 1000 characters). That is the honest subject of the turn, but it may retrieve less sharply than a single topic would.
- **Severity: low.** The search text changes what Orion remembers during self-inquiry, curiosity and reading. Juniper approved this change. It is still a cognition change, and there is no quality eval for it yet (see Evals).
- **Severity: low.** Reading seeds that are pinned documents and have no title search only the claim at stage 2, and nothing at stage 1 (recall then condenses the prompt), because an opaque document reference is not a useful search.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2423

🤖 Generated with [Claude Code](https://claude.com/claude-code)
