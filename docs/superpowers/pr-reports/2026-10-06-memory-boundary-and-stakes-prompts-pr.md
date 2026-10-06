# Memory redesign: define "boundary" for the judge, and give the distiller Juniper's stakes rubric

## Summary

- The judge that decides whether a conversation has ended was asked "BOUNDARY: YES or NO", but nobody told it what a boundary is. It said yes to almost everything, so one evening's chat was cut into several episodes. The prompt now defines it in Juniper's terms: yes means the thread ended or moved to something unrelated; no means a pause, a follow-up, an elaboration, a reaction, a return to the same subject, or small talk inside the thread. (`orion/memory/turn_change_classify.py`, `BOUNDARY_DEFINITION`)
- The distiller (the step that turns an episode into memories) marked every memory low-stakes, including Juniper's sister and her being scared her AC had broken Orion. The prompt now has Juniper's rubric as definitions with examples, and every memory must name its category in `stakes_reason` (`none` when low). Prompt version `memory_episode_distill.v3`.
- The validator never reads a memory's words to judge stakes. It only checks that `stakes`, `stakes_reason` and `asks_direction` are present and agree. When they disagree it moves the memory toward high (ask Juniper), never toward low, and logs why.
- Two evals on the live model lanes, scores and counts only: a 30-day boundary re-score, and a stakes before/after on 10 real episodes.

## Outcome moved

**Boundaries (30 days, 106 of Juniper's turns, live `metacog_background` lane, both prompts on the same turns):**

| | before | after |
|---|---|---|
| mean boundary score | 0.90 | 0.31 |
| turns scoring at least 0.92 | 69.5% | 28.6% |
| `resumed_thread` turns at least 0.92 (the only place the score decides anything under Rule 3) | 18 of 26 | 4 of 26 |
| episodes under Rule 3 | 52 (2.17 per active day) | 38 (1.58 per active day) |
| Chicago session, 2026-10-05 02:55-04:20 UTC (12 turns) | 2 episodes | **1 episode**; every follow-up scores 0.00 |
| Austin morning, 2026-09-28 | 3 episodes | 2 (the split is at the "Run github compactor" command turn) |

The score now depends on the content. It is high after long gaps (long_gap: mean 0.71) and on topic shifts (0.41), and low on stance changes inside a thread (0.03) and on short pauses (0.17). Before, every one of those groups averaged above 0.85.

**Stakes (live `memory_distill` lane, 27B, one run per prompt per episode):**

| | high-stakes memories |
|---|---|
| Deployed shadow distiller, stored in `episode_memory` (4 live episodes) | 0 of 31 |
| Old prompt (v2) re-run today on the same lane (9 episodes) | 0 of 55 proposed |
| New prompt (v3), after validation (10 episodes) | **15 of 69** |

v3 categories: 7 Juniper's feelings, 3 Orion and Juniper's relationship, 2 family/relationships, 2 Orion asking for direction, 1 conclusion about who Juniper is. The two named misses are now high: the sister memories are `family_relationships`, and "scared she busted my inference" is `juniper_feelings`. The consistency check changed one memory, from low to high (`asks_direction` was true).

## Current architecture

- **Boundary.** Each chat turn gets one classify call: four lines, NOVEL / SHIFT / MEMORY / BOUNDARY. The BOUNDARY score is the YES-vs-NO probability taken from the token logprobs. Episode Rule 3 (`services/orion-memory-consolidation/app/boundary.py::rule3_boundary`):
  - under about 20 minutes (`same_breath`, `short_pause`): never a boundary;
  - `resumed_thread`: a boundary only when the score is at least 0.92;
  - `long_gap`, `next_day`, `stale_thread`: always a boundary;
  - no phase: falls back to a 5400 s gap.
- **Stakes.** The distiller prompt said "low by default" and listed high cases in one sentence. The schema enum still had `safety_location` and a single `orion_self_conclusion`. The validator copied the distiller's stakes unchecked. Live result: 25 of 25 stored memories low (31 of 31 by the time of this run).

## Architecture touched

- Shared library only: `orion/memory`, `orion/schemas`, `orion/cognition/prompts`. No bus channel, no env key, no SQL change.
- Runtime seams: the memory-consolidation classify call and the brief's `prompt_version`; durable-runs distill render and validate.

## Files changed

- `orion/memory/turn_change_classify.py`: `BOUNDARY_DEFINITION`, placed in the turn-change prompt before the answer lines. The output format and parsing are unchanged.
- `orion/cognition/prompts/memory_episode_distill.j2`: the stakes rubric, with one definition and two or three short examples per category. `stakes_reason` is required on every memory. The shape example now shows one high memory and two low ones.
- `orion/schemas/memory_episode.py`: the new `StakesReason` categories plus `none`, `HIGH_STAKES_REASONS`, and prompt v3. `DistilledMemoryV1.stakes_reason` is now a plain string, so an unknown category cannot silently drop the whole memory at parse time. The validator checks the value instead.
- `orion/memory/episode/validate.py`: `resolve_stakes()`, which checks presence and consistency only.
- `orion/memory/episode/tests/*`, `tests/test_turn_change_classify.py`: tests for prompt rendering, parsing and stakes consistency. They also check that every schema category is defined in the prompt.
- `services/orion-memory-consolidation/evals/run_boundary_prompt_rescore_eval.py` (+ test): the new 30-day before/after re-score.
- `services/orion-memory-consolidation/evals/run_episode_distill_eval.py`: adds `--live` (episodes the deployed distiller already ran, with their stored stakes as the "before") and `--template` (re-runs an older prompt on the same lane). It also reports stakes by category.
- `services/orion-memory-consolidation/evals/results/2026-10-06-*.json`: scores and counts only, no text.
- `docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md`: a short note that the 2026-10-06 decision supersedes the old stakes floor.

## Schema / bus / API changes

- Added: `StakesReason` values `family_relationships`, `juniper_feelings`, `orion_machinery`, `orion_asks_direction`, `orion_relationship`, `none`.
- Removed: `family`, `relationship`, `safety_location`, `orion_self_conclusion`. No stored row uses any of them (every `episode_memory.stakes_reason` is NULL).
- Behavior changed: low memories now store `stakes_reason='none'`. A memory with no category, or with an unknown one, is stored high with a NULL reason and a `stakes_raised` / `stakes_reason_missing` event.
- Compatibility: not a bus contract. `DistilledMemoryV1` is only the distiller's parsed output. `EpisodeDistillBriefV1.prompt_version` now defaults to v3.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/memory/episode/tests tests/test_turn_change_classify.py services/orion-memory-consolidation/tests services/orion-memory-consolidation/evals -q
  459 passed, 12 skipped
(services/orion-durable-runs) pytest tests/test_episode_distill_graph.py tests/test_episode_distill_reconcile.py -q
  13 passed, 1 skipped
Static gates from .github/workflows/orion-static-gates.yml (every python gate, incl. check_definition_drift.py --gate,
check_metric_lineage.py --gate): all PASS. No re-lock: no metric definition changed. Hub JS node tests not run (no JS touched).
```

## Evals run

```text
python services/orion-memory-consolidation/evals/run_boundary_prompt_rescore_eval.py \
  --summary-json services/orion-memory-consolidation/evals/results/2026-10-06-boundary-prompt-rescore-summary.json
  106 turns, 105 scored per prompt, 0 errors (numbers above)

python services/orion-memory-consolidation/evals/run_episode_distill_eval.py --route memory_distill --compare-route "" \
  --repeat 1 --live [--template <v2 from git>]
  -> results/2026-10-06-distill-stakes-before-after-summary.json (numbers above)
```

How the boundary wording was chosen, stated plainly. The first wording ("YES only when ... CURRENT does not depend on BASELINE to be understood") overcorrected. It pushed 102 of 105 scores below 0.2, real topic switches included: a heating-and-cooling (HVAC) chat followed 37 minutes later by lab results scored 0.00. With that wording Rule 3 could never split on the judge. I then tried two more phrasings of the same definition on the 29 gap turns. I read each pair myself, locally and without committing any text. The one shipped ("the user has moved on from the BASELINE thread") catches the clear switches (lab results, mowing the lawn, a new question after "off to bed"). It has one false yes (a reply that elaborates on "meow") and one miss (a disk-space thread followed by a heavy family update). These labels are my own judgment on about 26 turns, and the wording was picked on the same data it is scored on. Treat the "after" numbers as optimistic until a fresh month confirms them.

## Docker/build/smoke checks

```text
No build or deploy (task said do not deploy). Evals called the live gateway over the bus, read-only.
```

## Review findings fixed

- No review subagent, as instructed for this task.

## Restart required

Run after merge, from the primary checkout on main (one line):

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-memory-consolidation up -d --build && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Rebuild both services together. The validator and the prompt ship in the same image. But a distill run already checkpointed with the v2 prompt, and finished by the new code, would get all-high stakes, because v2 never names a category. That direction is safe (it only adds confirmations), but it is noisy.

## Risks / concerns

- **Medium. The boundary wording was tuned on the data it is measured on.** 26 decisive turns, my own labels, three wordings. Mitigation: re-run `run_boundary_prompt_rescore_eval.py` after a few weeks of new turns.
- **Medium. Single runs.** The two "before" passes differed by one resumed_thread split (17 vs 18), so run-to-run noise exists. The stakes numbers are one run per prompt per episode.
- **Low. The Chicago phase in this replay differs from the live stamp.** The replay recomputes phase from gaps between turns, so the 03:51 turn (a 17-minute gap) is `short_pause` here, while the live Hub stamped it `resumed_thread`. Either way the new score at 03:51 is 0.00, so the session stays one episode.
- **Low. Borderline stakes calls remain.** For example, in the Chicago episode, her height and finding economy seats "brutal" stayed low. The rubric is the model's judgment, by design. The validator does not second-guess it.
- **Not fixed here, observed during the eval.** From at least 03:30 until 03:50 UTC today (start not pinned), the GPU pool's cooling shed (`rule=cooling shed=cabinet_rising`) left every metacog and quick classify request unserved. The live service logged `memory_classify_degraded` for those turns.

## PR link

(see the PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
