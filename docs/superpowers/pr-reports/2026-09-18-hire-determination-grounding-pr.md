# Hire determination grounding — PR report

Branch: `feat/hire-determination-impl`

## Summary

- Mind requests now carry who spoke (`utterance_origin`: Juniper vs Orion) plus a short situation line, so curiosity kickoff is not read as if Juniper typed it.
- Thought coloring forwards what the turn is trying to do (`user_intent`) for both origins, and only Orion-origin turns may carry soft work-shape labels (`expected_depth`, `cross_cutting`, `foresight_note`). Hire-frame fields stay blocked on Juniper chat.
- Curiosity Mind appraises a short investigation subject (claim + continue note), not the full kickoff / HelpRequest Cypher teach. The motor still gets the full prompt.
- Kickoff teach asks Orion to write `:InvestigationRole` (`local_crawl` or `hire_cursor`) early; HelpRequest remains the hire ticket after a short local look. Python never stamps the role or the hire.
- Empty `tried_summary` HelpRequests are skipped at enqueue when hops already exist for the run.

## Outcome moved

The hire-vs-local-crawl choice is grounded in origin-aware Mind framing, advisory work-shape labels, and an Orion-authored role node — not “optional / stuck” teach, not attention winners, not Python auto-hire. Flag-off still omits teach and enqueue.

Live thought-event / graph / Hub-chat proof is **UNVERIFIED**. Do not treat this PR as runtime evidence.

## Current architecture

Before this patch:

- Curiosity passed the full kickoff as the Mind “user” text. Origin was unmarked.
- `select_mind_coloring` dropped `user_intent`. Soft depth / breadth / foresight labels did not exist.
- Hire teach said HelpRequest was optional, only if stuck, not as a default. Missing HelpRequest could mean “chose local” or “never considered.”
- Empty `tried_summary` still published when hops existed.

## Architecture touched

- `orion-mind` — origin-conditional stance handoff; optional soft labels on `ChatStanceBrief`
- `orion-thought` — origin prose on Mind requests; guarded coloring allow-list
- `orion-hub` — Hub chat stamps `juniper`; curiosity stamps `orion` + `mind_appraisal_text`
- `orion/curiosity` — subject builder, role teach, RO `:InvestigationRole` reader, empty-`tried_summary` enqueue skip
- Spec + plan under `docs/superpowers/`

## Files changed

- `orion/mind/v1.py`: optional `utterance_origin` on `MindRunRequestV1`
- `orion/schemas/chat_stance.py`: optional `expected_depth` / `cross_cutting` / `foresight_note`
- `orion/hub/turn_orchestrator.py`: origin + appraisal kwargs into stance/Mind; harness keeps full prompt
- `services/orion-thought/app/mind_enrichment.py`: origin note; base vs Orion work-shape coloring keys
- `services/orion-thought/app/bus_listener.py`: pass origin into coloring selector
- `services/orion-mind/app/stance_handoff.py` + `engine.py`: Orion-only work-shape instruction + coerce
- `services/orion-hub/scripts/api_routes.py`: Hub chat `utterance_origin="juniper"`
- `services/orion-hub/scripts/curiosity_investigation.py`: `orion` origin + subject-sized appraisal
- `orion/curiosity/investigation_subject.py`: short subject builder
- `orion/curiosity/kickoff_prompt.py` / `self_inquiry_prompt.py`: role split teach; drop stuck-only frame
- `orion/curiosity/worldview.py`: RO `:InvestigationRole` read / latest-wins
- `orion/curiosity/peer_briefs.py`: skip enqueue when `tried_summary` blank and hops exist
- Tests under thought, mind, hub, curiosity, and peer kickoff/acceptance
- `docs/superpowers/specs/2026-09-15-orion-hire-determination-grounding-design.md`: status → implementing; live proof UNVERIFIED
- `docs/superpowers/plans/2026-09-18-hire-determination-grounding.md`: implementation plan (carried onto branch)

`orion/curiosity/README.md` was left alone: it does not document HelpRequest teach.

## Schema / bus / API changes

- Added: `MindRunRequestV1.utterance_origin` (`juniper` | `orion` | unset)
- Added: optional `ChatStanceBrief` work-shape fields (existing registry import; no new kind)
- Added: graph label `InvestigationRole` (Orion-authored MERGE in teach text only; no registry schema)
- Added: coloring keys `user_intent`, `uncertainty_summary`; Orion-only `expected_depth`, `cross_cutting`, `foresight_note`
- Removed: none
- Renamed: `_help_request_section` → `_role_and_help_section` (alias kept)
- Behavior changed: stance/Mind use `mind_appraisal_text` when set; enqueue skips empty `tried_summary` if hop_count > 0
- Compatibility notes: unset origin and blank appraisal keep previous behavior. `CuriosityTurnRequestV1` is **unchanged** (`extra=forbid`) — durable turns after Hub restart still fall back to full kickoff (see follow-ups)

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: n/a
- skipped keys requiring operator action: none

`HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED=false` still omits role/HelpRequest teach and enqueue.

## Tests run

```text
Worktree: /mnt/scripts/Orion-Sapienform-hire-determination-impl
Thought and hub suites run separately (app package collision).

PYTHONPATH=. pytest \
  services/orion-thought/tests/test_mind_utterance_origin.py \
  services/orion-thought/tests/test_mind_coloring_selector.py \
  services/orion-thought/tests/test_mind_light_snapshot.py -q
# 25 passed, 18 warnings in 2.69s

PYTHONPATH=. pytest \
  tests/test_curiosity_peer_kickoff.py \
  tests/test_curiosity_peer_patch0_acceptance.py \
  orion/curiosity/tests/test_investigation_subject.py \
  tests/test_investigation_role_worldview.py -q
# 21 passed in 0.30s

PYTHONPATH=. pytest \
  services/orion-hub/tests/test_curiosity_help_request_enqueue.py \
  services/orion-hub/tests/test_turn_orchestrator_utterance_origin.py -q
# 11 passed, 35 warnings in 8.05s

PYTHONPATH=. pytest \
  services/orion-mind/tests/test_stance_handoff_soft_labels.py \
  services/orion-mind/tests/test_mind_llm_pipeline.py -q
# 31 passed, 18 warnings in 2.61s

python scripts/check_env_template_parity.py
# env template parity: PASS (88 service(s) compared)
# (unrelated local .env missing-key WARNs; this patch added no keys)

git diff --check
# clean

scripts/check_schema_registry.py / scripts/check_bus_channels.py
# not present in this repo (Makefile already notes agent-check chain is incomplete)
```

Warnings are pre-existing pydantic `model_*` protected-namespace noise, not from this patch.

## Evals run

```text
PYTHONPATH=.:services/orion-thought pytest \
  services/orion-thought/evals/test_mind_enrichment_eval.py -q
# 3 passed, 18 warnings in 2.27s

No hire-determination quality eval (live thought-event inspect). That
acceptance remains UNVERIFIED until thought + hub + mind are deployed
and an operator inspects one Hub chat turn and one curiosity kickoff.
```

## Docker/build/smoke checks

```text
Live smoke: UNVERIFIED. This session did not deploy thought / mind / hub
and did not inspect a live thought.event.v1, Hub chat prefix, or
InvestigationRole graph row. No correlation ids.

Do not claim runtime proof from unit tests.
```

## Review findings fixed

Per-task SDD reviews on Tasks 1–5 found no Critical/Important blockers. Material items already in the commits:

- Finding: unused `ValidationError` import in `stance_handoff.py`
  - Fix: removed in Task 2
  - Evidence: `6aa63fda6`
- Finding: combined thought+mind pytest in one process rebinds `app`
  - Fix: suites run separately (pre-existing isolation)
  - Evidence: Task 6 gate commands above

## Known follow-ups (not in this PR)

- Persist `mind_appraisal_text` on `CuriosityTurnRequestV1` so durable curiosity turns after a Hub restart still appraise the subject instead of the full kickoff. In-process same-Hub callbacks work today; the RPC model is `extra=forbid`.
- Optional test nits from earlier reviews: invalid-origin drop fixture; origin+broadcast `situation_compact` merge fixture; symmetric Juniper origin-prose assert; duplicate mind handoff tests; thought eval not extended for the new coloring keys; stale test name.

## Restart required

```bash
# From this worktree (not the shared checkout), after merge or for live verify:
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Acceptance smoke after restart (still UNVERIFIED until run):

1. One Hub Juniper chat turn: inspect thought/prefix — no `expected_depth` steering.
2. One curiosity durable kickoff: inspect Mind/thought for claim-shaped `user_intent` / subject, not “write Cypher like this.” Record correlation id.
3. If Orion writes `:InvestigationRole` + optional HelpRequest, confirm graph reads distinguish `local_crawl` / missing / HelpRequest.

## Risks / concerns

- Severity: should
- Concern: Live acceptance is UNVERIFIED. Code existing is not proof the live path moved.
- Mitigation: restart commands + smoke checklist above; do not merge on “tests passed” as if it were a thought-event inspect.
- Severity: should
- Concern: After Hub restart, durable curiosity turns lose the in-memory subject map and Mind falls back to the full kickoff until `CuriosityTurnRequestV1` carries `mind_appraisal_text`.
- Mitigation: documented follow-up; same-process durable callbacks are fine until restart.
- Severity: low
- Concern: Optional Hub post-run roles-vs-helps accounting line was skipped. Footprint `wrote=` will show `InvestigationRole` if Orion writes it.
- Mitigation: RO reader is ready; add a log if operators need the split.

## PR link

(filled after `gh pr create`)
