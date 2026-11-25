## Summary

Integration patch 7b: two more organs read Orion's now and record what they used. Part of #3429 (epic #3434).

- **Paintings read the now.** Before a waking painting's one model step (the interpret call that turns a context clause into a paintable scene), it reads `GET /temporal-self/now?organ=painting`. The items go into that call's prompt with their `[experience:<id>]` tags. Tags the model writes are taken out before the diffusion prompt and posted to `POST /temporal-self/now/use` once the plan is the one that will be painted.
- **Memory reads the now.** The reminiscence hop about the present (`where_i_am_now`) reads `GET /temporal-self/now?organ=memory`, shows the items, and posts the reply's tags after the hop's save receipt is proven.
- **Cameras stay exempt, with the real reason.** No camera model step should cite: a look is a dig command inside a chat or curiosity turn (that turn's organ reads the now), perception describes frames, and camera asks are opened by rules. A cite there would be another organ's cite under the camera's name.
- **Dreams already read** through the nightly dream.day run. The sleep cycle and dream.carry do not; that is #3972.
- Kill switches ship on: `ORION_PAINTING_NOW_ENABLED`, `REMINISCENCE_NOW_ENABLED`.

## Outcome moved

Organs with a checked read-back in the registry: 5 of 16 before (reverie, dream, curiosity, play, prime), 7 of 16 after (+ painting, memory). `scripts/check_integration_seam.py` passes.

The spec's done-when ("closed organs through checked read-back reach at least 12 of 15") is a live count of organs whose experiences another organ cited in a day. Live baseline before this patch (`scripts/report_integration_coverage.py`, 2026-11-24T11:15Z..11-25T11:15Z): **5 of 16 closed** (curiosity, camera, body, prediction, play). After deploy: UNVERIFIED. 12 is not reachable by this patch alone: reverie's reads are never closed (#3971), and 7 organs have no planned reader (#3457).

## Current architecture

Readers before: Prime (`orion-mind`), reverie (`orion-thought` temporal_self_cue.py), curiosity (Hub curiosity_now.py), play (Hub retro_commentary.py), dream.day (`orion-durable-runs` dream_day_store.py). Painting, camera and memory had `exempt_read: "not yet migrated; integration patch 7"`.

## Architecture touched

- orion-thought: `app/visual_chain.py` (compute_visual_plan reads, `close_plan_now` posts), `app/visual_steps.py` (prepare posts only after its own freeze won), `app/temporal_self_cue.py` (painting read/strip/close helpers on the existing shared reader), `orion/cognition/prompts/visual_context_interpret.j2` (optional now block).
- orion-durable-runs: new `app/reminiscence_now.py`; `app/reminiscence_graph.py` (hop wiring); `app/admission_runtime.py` (dep).
- `config/integration/organs.yaml`: painting and memory `reads_now`; camera exemption reason; dream comment.

## Files changed

- `services/orion-thought/app/visual_chain.py`, `visual_steps.py`, `temporal_self_cue.py`, `settings.py`, `.env_example`: painting read-back.
- `orion/cognition/prompts/visual_context_interpret.j2`: the now block and tag rule.
- `services/orion-durable-runs/app/reminiscence_now.py` (new), `reminiscence_graph.py`, `admission_runtime.py`, `settings.py`, `.env_example`, `docker-compose.yml`: memory read-back.
- `config/integration/organs.yaml`: registry.
- Tests: `services/orion-thought/tests/test_painting_now.py` (new), `test_visual_steps.py`, `services/orion-durable-runs/tests/test_reminiscence_graph.py`, `tests/test_check_integration_seam.py` (planted-exemption test now uses camera).

## Schema / bus / API changes

- Added: none (no new schema, kind, channel or route). New reader organs `painting` and `memory` use the existing endpoints.
- Behavior changed: `VisualPlan` (frozen in `stage_json.plan`) and chain_json gain `now_shown_id`, `now_cites`; old frozen plans load (missing -> None). Reminiscence step records gain `now_shown_id`, `now_cites`.
- Compatibility: the interpret call gets one new optional context key `orion_now`.

## Env/config changes

- Added keys: `ORION_PAINTING_NOW_ENABLED=true`, `ORION_PAINTING_NOW_FETCH_TIMEOUT_SEC=2.0` (orion-thought); `REMINISCENCE_NOW_ENABLED=true` (orion-durable-runs, also in compose).
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py` (`--all-keys orion-thought` for the thought keys): yes.
- skipped keys: none.

## Tests run

```text
services/orion-thought: pytest tests -> 593 passed, 31 skipped
services/orion-durable-runs: PYTHONPATH=<repo> pytest tests -> 1117 passed, 157 skipped
tests/test_check_integration_seam.py tests/test_report_integration_coverage.py -> pass
tests/test_organ_names_partition.py -> 12 passed; orion/dig/tests/test_organs.py -> 25 passed
tests/test_check_verb_physiology.py, orion-cortex-exec test_system_only_llm_messages.py -> 41 passed
python scripts/check_integration_seam.py --require-base -> OK (16 organs)
python scripts/check_env_template_parity.py -> PASS
```

## Evals run

```text
ORION_LIVE_DSN=... python scripts/report_integration_coverage.py  (read-only, live baseline above)
```
No replay eval of cite quality yet; the spec's weekly ablation eval (hollow cites) covers every reader and is not part of this patch.

## Docker/build/smoke checks

```text
Not run (no deploy asked). Live cites: UNVERIFIED until orion-thought and orion-durable-runs are rebuilt.
```

## Review findings fixed

- Finding: a painting's cites were posted before the plan froze, so a concurrent prepare that lost the freeze (or a crash before freeze) left cites for a picture never painted.
  - Fix: compute_visual_plan keeps the cites on the plan; `prepare_step` posts only when the frozen plan carries its own shown id; the legacy run-once posts right after computing.
  - Evidence: `test_prepare_that_wins_the_freeze_closes_its_read`, `test_prepare_that_loses_the_freeze_posts_nothing`.
- Finding: a sentence that was only tags still posted cites while the raw slot text was painted.
  - Fix: closed with no cites. Evidence: `test_a_sentence_that_was_only_tags_keeps_no_cite`.
- Finding: malformed tags could reach the 77-token diffusion prompt.
  - Fix: loose strip pattern plus bare-tag strip. Evidence: `test_malformed_tags_never_reach_the_diffusion_prompt` (5 shapes).
- Finding: memory posted cites before the receipt check, so a failed attempt stored cites for an unproven hop.
  - Fix: cites extracted first, posted after verification; a failed receipt closes with none. Evidence: `test_a_reply_with_a_tag_but_no_proven_receipt_closes_with_no_cites`.
- Nit: memory read/post not bounded as a whole -> `asyncio.wait_for`. Nit: memory record kept invented ids -> shown ids only. Nit: camera comment wording -> fixed.
- Not taken: reuse dream_day_store's client for memory. The integration gate checks that the named `reads_now` module itself calls the routes, so memory keeps its own small module (as curiosity and dream.day each do).

## Restart required

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```
(From the clean primary main checkout after merge; the post-pull rebuild does this.)

## Risks / concerns

- Severity: low. Concern: the painting's interpret call now has a little more prompt; a model could chase the now instead of SOURCE. Mitigation: prompt keeps SOURCE as what the scene depicts; kill switch.
- Severity: low. Concern: memory reads once a day only, so it adds at most one reader run per day. Mitigation: none needed.
- Issues opened: #3971 (reverie reads never closed: 228 reads, 0 closed in 3 days, live), #3972 (dream cycle/carry read-back). Closed: none. Part of #3429.

## PR link

(see PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
