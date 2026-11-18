# feat(dream): carried dreams — text → picture → text → picture → text → picture

## Summary

- A completed sleep now ends in a **carried dream**: six hops alternating words and pictures,
  run as one durable run (`dream.carry`).
  - **Text hop:** Orion writes a passage and a picture prompt.
  - **Picture hop:** Orion paints the prompt, looks at the painting, and the next text hop dreams on
    from what was *seen*.
- **Picture hops reuse the existing painting run** (`reverie.visual`), so they keep its heat retries,
  GPU holds, crash recovery and cleanup.
  - A new dream mode paints the prompt verbatim.
  - Dream pictures never count toward, continue or show up in Orion's waking paintings.
- **Heat is expected.** A picture that fails on heat gets a fresh painting run, up to 3 per hop.
  A carry that runs out of time finishes **partial** with the hops it made and says where it
  stopped.
- **No sleep is left without a dream.**
  - A carry that can't be submitted falls back to the one-paragraph story (#2565).
  - So does a carry that made zero hops.
- **Hub:** the Dream tab shows each carried dream as a strip: passage, picture, "Orion saw:", ….
- **Off switch:** `DREAM_CARRY_ENABLED=false` returns to the one-paragraph story; carries already
  running still finish.

## Outcome moved

- **Before:** one paragraph per sleep, from one model call.
- **After:** a 6-hop word/picture dream per sleep, with every hop checkpointed and visible in Hub.
  Measured on real material: the sleep fixture `dc-981127ddbddf` and 3 real painting captions
  (eval below).

## Current architecture

- **Story dream (#2565):** sleep → `dream.trigger` → cortex-orch `dream_cycle` verb → `dreams`.
- **Paintings:** `reverie.visual` durable run (prepare → hold → generate → caption), executed by
  orion-thought.
  - It is tied to the waking baseline and continuity.
  - It allows one open attempt at a time, with a 600 s cooldown.
  - Live, last 3 days: 15 produced, 26 ended `retry_window_expired` after `thermal_refused`.

## Architecture touched

| Service | Change |
|---|---|
| orion-dream | Text-hop and finish step responder; carry submit through cortex-orch; story fallback; `POST /dreams/carry/run` |
| orion-durable-runs | `dream.carry` graph; child painting runs with heat-type resubmit; `dream_hop` forwarded to painting steps; caption in the painting run's detail |
| orion-thought | Dream hop mode: verbatim prompt; no baseline, continuity, chain row, artifact row or receipt; caption returned |
| orion-hub | `/api/dream/carries`, `/api/dream/carry/image/{sha}` (only shas a carry names), Dream tab strip |
| orion-cortex-orch | Test only: the durable ingress already accepts any workflow |
| contracts | `orion/schemas/dream_carry.py`, `DreamHopImageV1`, `"dream.carry"` workflow, carry step channels, metric definition re-lock |

## Files changed

See `git diff --stat origin/main...HEAD`. The main ones:

- **Contract:**
  - `orion/schemas/dream_carry.py` (new)
  - `orion/schemas/reverie_visual_run.py`, `orion/schemas/durable_run.py`, `orion/schemas/registry.py`
  - `orion/bus/channels.yaml`, `config/metrics/metric_definitions.lock.json`
- **orion-durable-runs:**
  - `app/dream_carry_graph.py` (new), `app/admission_runtime.py`, `app/reverie_visual_graph.py`,
    `app/runner.py`, `app/settings.py`
  - `orion/durable_runs/registry_store.py` (`terminal_detail`)
- **orion-dream:**
  - `app/carry.py`, `app/carry_listener.py`, `app/carry_submit.py` (new)
  - `app/main.py`, `app/story.py`, `app/llm.py`, `app/settings.py`
- **orion-thought:** `app/visual_steps.py`, `app/bus_listener.py`
- **orion-hub:** `scripts/dream_routes.py`, `static/js/dream-tab.js`, `templates/index.html`,
  `evals/dream_browser.cjs`
- **Docs:**
  - `docs/superpowers/specs/2026-10-10-dream-carry-through-design.md` (approved, updated to as-built)
  - the service READMEs

## Schema / bus / API changes

- **Added:**
  - `DreamCarryBriefV1`, `DreamCarryHopV1`, `DreamCarryStepRequestV1/ResultV1`, `DreamHopImageV1`
  - `ReverieVisualRunBriefV1.dream_hop`, `ReverieVisualStepRequestV1.dream_hop`,
    `ReverieVisualStepResultV1.caption`
  - `"dream.carry"` in `DurableWorkflowV1`
  - channels `orion:dream:carry:step:request` and `orion:dream:carry:step:reply:*`
  - orion-dream as producer of `orion:dream:log` and `orion:cortex:request`, and consumer of
    `orion:cortex:result*`
- **Removed / renamed:** none.
- **Behavior changed:** with the flag on, a completed sleep submits `dream.carry` instead of
  publishing `dream.trigger`; the trigger becomes the fallback.
- **Compatibility:**
  - Waking painting step requests omit `dream_hop` entirely, and thought replies with
    `exclude_none`, so waking traffic is byte-identical to before.
  - Additive Literal/forbid fields still need the consumer-first restart order below.

## Env/config changes

- **Added:**
  - orion-dream: `DREAM_CARRY_ENABLED=true`, `DREAM_CARRY_DEADLINE_SEC=14400`,
    `CHANNEL_CORTEX_REQUEST`, `CHANNEL_DREAM_LOG`
  - orion-durable-runs: `DREAM_CARRY_FINISH_GRACE_SEC=1800`, `DREAM_CARRY_CHILD_MAX_ATTEMPTS=3`,
    `DREAM_CARRY_CHILD_MIN_WINDOW_SEC=900`, `DREAM_CARRY_CHILD_WINDOW_SEC=2400`, `DREAM_CARRY_CHILD_RETRY_GAP_SEC=900`
- **Removed / renamed:** none.
- **`.env_example` updated:** yes, each service, plus compose and settings.
- **Local `.env` synced:** yes, `sync_local_env_from_example.py --all-keys <service>`, run from the
  branch's copy of the script so it reads the branch templates.
  - Durable-runs flagged `POSTGRES_URI` and `DURABLE_RUNS_GRAPH_HOST` as diverged; both are
    host-specific and were left alone.

## Tests run

```text
orion-dream tests+evals                                   191 passed, 1 skipped
orion-durable-runs (PYTHONPATH=repo)                      350 passed, 74 skipped
  with a throwaway postgres (builder run)                 423 passed, 1 skipped
  incl. a real carry driven end to end through _drive: three child paintings,
  three LLM holds all released, hops_made=6
orion-thought                                             525 passed, 1 failed
  the failure is pre-existing on main: test_settings_mind_enrichment default URL
orion-cortex-orch                                         222 passed, 31 failed
  same set fails on main (32 there), none new
hub dream CI set (test_dream_routes + test_dream_hypotheses)  30 passed
tests/test_dream_trigger_contract.py                      3 passed
scripts/check_definition_drift.py --gate                  PASS (re-locked for the new channels)
scripts/check_metric_lineage.py --gate                    PASS
git diff --check                                          clean
Mutation checks: every key fix in each service was reverted one at a time, and each revert
makes its tests fail. Covered:
  verbatim dream prompt; no waking chain row; caption on replay; exclude_none reply;
  gpu_lease forward; 45-word clip; caption in the continue prompt;
  zero-hop story fallback; responder always on; resubmit before fallback;
  heat-type child resubmit: retryable, attempt cap, min window, unique dispatch ids,
  cancel current child, backoff;
  child window cap; replacement gap; first-child window; cancel on partial; terminal keeps sha.
```

## Evals run

```text
services/orion-dream/evals/test_dream_carry_eval.py
  A full 6-hop carry built from the real sleep dc-981127ddbddf, with the real prompt builders and
  parser and 3 real captions orion-thought wrote for Orion's paintings.
  Asserts:
    - every later prompt carries the previous caption and passage;
    - the opening prompt carries every replayed item and no hypotheses;
    - the fragments alternate text and image, ending on the last caption.
  The model is fake: this proves the hand-offs, not that the model follows the picture.
node services/orion-hub/evals/dream_browser.cjs (real Hub template, fixture APIs, Chrome 131)
  15 checks passed, including the carried dream strip, an escaped injected caption and the
  carry empty state. Screenshot reviewed.
```

## Docker/build/smoke checks

```text
Live smoke: UNVERIFIED until deploy. After deploy, POST /dreams/carry/run once and check:
  substrate_durable_run_state: dream.carry completed, hops_made=6
  three child reverie.visual runs whose attempts record stage_json.dream_hop, and no new
    reverie_visual_chain rows
  a dreams row with profile dream.carry and 6 fragments; the Hub Dream tab strip
```

## Review findings fixed

Each service part was reviewed by its own builder's subagent. A cross-service review ran over the
merged branch and found no high-severity problems.

- **Medium: a carry that failed before its first hop left the sleep with no dream.**
  - Fix: zero hops still calls finish, and orion-dream publishes the one-paragraph story
    (`story-fallback:<trigger_id>`), once per run.
  - Evidence: tests in durable-runs and orion-dream; mutation-checked.
- **Medium: dream pictures could hold thought's single painting slot for hours when hot.**
  - Fix: dream child window of 2400 s; at least 900 s before a replacement child, so a waking
    painting can claim the slot in between.
  - Evidence: window and gap tests.
- **Low: a first child was submitted with seconds left.**
  - Fix: the min-window check also applies to the first child.
- **Low: a completed-partial carry left its child running.**
  - Fix: cancel the current child on any carry terminal.
- **Low: a replayed produced child could lose its picture.**
  - Fix: terminal branches of the painting run keep `artifact_sha256` and `caption`.
- **Low: a submit timeout after admission gave two dreams.**
  - Fix: resubmit once (the run id dedupes) before falling back to the story.
- **Low: the kill switch also killed carries in flight.**
  - Fix: the step responder always runs; the flag gates only new carries.
- **Low: 60 words can overflow CLIP's 77 tokens.**
  - Fix: prompts are clipped to 45 words.
- **Low: heat ended the dream at the first hot picture** (26 of 41 live paintings ended on heat).
  - Fix: heat-type misses get a fresh child, up to 3 per hop, inside the carry window.
- **Note: the spec described things that were not built.**
  - Fix: the spec was rewritten as built (no table, child runs, the dream flag on the painting
    run).
- **CI: the always-on carry responder created a Postgres engine at startup.** That broke an
  existing lifespan test on a runner without psycopg.
  - Fix: the engine is built on first lookup; the lifespan test stubs the responder.
  - Evidence: `test_building_the_carry_responder_needs_no_database_driver`; all 21 CI checks green.
- **Not fixed:**
  - L8: one checkpoint per 30 s child poll. The runtime has no way to wait without writing one;
    it costs storage only.
  - See risks for the rest.

## Restart required

From the primary checkout on main after merge, consumers first, orion-dream last:

```bash
git pull --ff-only && for s in orion-sql-writer orion-hub orion-actions orion-thought orion-durable-runs orion-cortex-orch orion-dream; do scripts/safe_docker_build.sh $s up -d --build || break; done
```

## Risks / concerns

- Severity: medium
  - Concern: heat. Each carry adds up to 3 paintings, and the cabinet already refuses most
    paintings.
  - Mitigation: dream children share the painter's one-at-a-time slot and 600 s cooldown, wait
    900 s before a replacement, and use 40-minute windows. A hot night gives a partial carry,
    never a lost one.
- Severity: low
  - Concern: duplicates after a restart in a narrow window.
    - A finish replayed after orion-dream restarts between publishing and replying can write a
      second dreams row, or a second fallback story.
    - The guard is in memory, plus an unindexed dreams lookup for the carry itself.
  - Mitigation: rare. A unique index on the audit dream_id in sql-writer would close it.
- Severity: low
  - Concern: deploy order. If orion-dream starts carries before durable-runs knows `dream.carry`,
    the run is misrouted.
  - Mitigation: durable-runs now registers it, and the restart command puts orion-dream last.
- Severity: low
  - Concern: real model output is untested. The eval proves the hand-offs, not dream quality.
  - Mitigation: start one by hand after deploy and read it.
- Severity: note
  - Concern: about a third of real sleep material is bare reverie ids (`open-loop-…`), which give
    hop 0 little to work with.
  - Mitigation: resolving them to their loop titles is the next small patch.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2586

🤖 Generated with [Claude Code](https://claude.com/claude-code)
