## Summary

- Juniper's rule (2026-11-12): world-pulse reading turns never send LLM work to the fast lane or the metacog lane.
- Reading turns are durable runs. Every one holds a GPU seat (`ReadingTurnRequestV1.gpu_lease` is required). Two kinds of call ignored that seat: the pre-turn repair check and the three Mind phases.
- One rule now covers both: a turn that holds a seat runs all of its LLM work on that seat (`orion.llm.resource_lease.held_call_route`: route `agent` plus the hold as `options.gpu_lease`). Juniper's own turns have no hold, so they keep `quick_turn` and `metacog_turn`.
- The live gate also found that the repair check has been hollow on every lane since about 10-23. It never read the real per-token scores, so every reading was the fixed text fallback (confidence 0.65 on 100% of `repair_pressure_appraisal_log` rows since 10-23). The parser now reads both reply shapes the gateway returns.
- Three more scoring bugs sat on that dead path and are fixed. (1) Answers were matched to kinds by order, so a stray "No" before the lines shifted every score onto the wrong kind; answers now match by character position, and any mismatch falls back for the whole reply. (2) A rare lowercase ` no` overwrote the real ` NO` score; spellings of one answer now add up, and confidence is the gap between the two summed answers. (3) A very sure answer whose opposite fell out of the top-8 list dropped the reply to the fallback; the missing answer is now bounded by the lowest listed score.

## Outcome moved

- World-pulse checks and Mind phases no longer queue for a 4,096-token slot they cannot fit. In the 24 h before this patch, 41 stage-2 checks and 40 stage-2 Mind calls waited until they were cancelled, 6 checks overflowed the context after a 300 s wait, and 8 Mind calls ran out of budget on a borrowed seat.
- The repair check returns real, varying scores instead of one fixed constant. This applies on every lane, Juniper's fast lane too (see Risks).

## Current architecture

- The Hub (`orion/hub/turn_orchestrator.py`, `execute_unified_turn`) sends the pre-turn check before the stance. The check request (`PreTurnAppraisalRequestV1`) had no hold field. cortex-exec (`app/pre_turn_appraisal.py`) always used `REPAIR_PRESSURE_PROBE_ROUTE=quick_turn` (fast class: Qwen3-8B, 4,096 tokens per slot).
- orion-thought builds the Mind request (`build_light_mind_request`) from the stance request. The stance request already carried the hold, but the Mind request did not. orion-mind ran every phase on `MIND_*_MODEL_ROUTE=metacog`.
- Stance, motor and finalize judge already ran under the hold (`options.gpu_lease`, attached by the gateway).

### Inventory: world-pulse LLM work on the small lanes (24 h to 2026-11-12 ~08:00Z)

Source: `llm_call_usage` and `gpu_pool_leases`, joined to `cockpit_turn_sighting.raw->>'session_id'` by correlation id. 91 stage-1 turns and 70 stage-2 turns. 160 of 161 had hold-child leases. The one exception was a stage-1 turn whose stance timed out before any held call.

| Call site | Caller, route | Stage 1 | Stage 2 |
|---|---|---|---|
| Pre-turn repair check | cortex-exec, `quick_turn` (fast class) | 90 served on fast, 1 cancelled | 11 served on fast; 6 context overflow (300 s wait); 41 cancelled, never seated; 18 spilled to agent/agent-gpu2 (avg wait 68 s / 12 s) |
| Mind phases (semantic, frontier judge, stance handoff) | orion-mind, `metacog` | 100 served on metacog, 2 on fast, 1 cancelled | 11 served on metacog; 40 cancelled, never seated; 11 spilled to agent; 8 on agent-gpu2 `caller_budget_exhausted` |
| Stance react, finalize judge, motor | cortex-exec, Anthropic-shaped motor, route `agent` + hold | all on agent seats | all on agent seats |

No other world-pulse call site touched the fast or metacog lane.

## Architecture touched

- `orion/llm/resource_lease.py`: new `held_call_route(gpu_lease, default_route)`. This is the one rule. A held call returns `("agent", hold_json)`. An unheld call returns `(default_route, None)`. There is no fallback from a refused attach back to the small lane.
- Pre-turn check: the Hub passes the turn's hold into the request. cortex-exec routes the probe with the helper and logs `route=` and `held=` on every result and failure line.
- Mind: orion-thought copies the stance request's hold onto `MindRunRequestV1.gpu_lease`. orion-mind routes all three phases with the helper and sends the hold on `options.gpu_lease` through `MindLLMRequestContext`.
- Probe parser: `_normalize_logprob_content` reads `raw.choices[0].logprobs.content` (Strata, OpenAI-compatible) and llama.cpp's flat `raw.completion_probabilities`. `_yes_no_logprobs` takes the log-sum-exp over the spellings of each answer.

## Files changed

- `orion/llm/resource_lease.py`: `held_call_route`.
- `orion/schemas/pre_turn_appraisal.py`: optional `gpu_lease` on `PreTurnAppraisalRequestV1`. The serializer omits it when None.
- `orion/hub/turn_orchestrator.py`: the turn's hold is resolved before the check and passed into `_run_pre_turn_appraisal`.
- `services/orion-cortex-exec/app/pre_turn_appraisal.py`: the probe uses the helper. The hold goes on the gateway options. Logs carry the route and the held flag.
- `orion/mind/v1.py`: optional `gpu_lease` on `MindRunRequestV1`.
- `services/orion-thought/app/mind_enrichment.py`: copies the hold.
- `services/orion-mind/app/engine.py`, `llm_context.py`, `llm_client.py`: route every phase onto the hold and send it.
- `orion/substrate/appraisal/paradigms/repair_pressure_v2.py`: reads both reply shapes; `_aligned_answer_entries` matches each answer to its own token by character position (replaces positional `_answer_token_entries`); all kinds fall back together.
- `orion/substrate/appraisal/probe/logprob_runner.py`: answers summed over spellings; margin = gap between the summed answers (replaces raw top-1/top-2 `_top1_margin`); a missing answer is bounded by the lowest listed logprob (`features.bounded`).
- Tests: `tests/test_held_turn_lane_rule.py` (new), `services/orion-cortex-exec/tests/test_pre_turn_appraisal_rpc.py`, `services/orion-hub/tests/test_turn_orchestrator_ws_frames.py`, `services/orion-mind/tests/test_mind_turn_route.py`, `services/orion-thought/tests/test_mind_utterance_origin.py`.
- `.github/workflows/held-turn-lane-rule-tests.yml` (new): runs the rule's tests per service. The Hub link runs in `orion-hub-full-suite`.
- `services/orion-cortex-exec/README.md`, `services/orion-mind/README.md`: document the rule.

## Schema / bus / API changes

- Added: `PreTurnAppraisalRequestV1.gpu_lease: GpuLeaseRefV1 | None = None`. The model is `extra="forbid"`. When the field is None the key is left out of the dump, so Juniper's requests stay byte-identical.
- Added: `MindRunRequestV1.gpu_lease: GpuLeaseRefV1 | None = None`. The model ignores unknown fields.
- Removed / renamed: none.
- Behavior changed: a held turn's probe and Mind phases use route `agent` plus `options.gpu_lease`. No channel or registry change (no new schema id).
- Compatibility (deploy order):
  1. **cortex-exec before Hub.** A new Hub sends `gpu_lease` on held turns. An old cortex-exec rejects that key (`extra="forbid"`), so held-turn checks would fail until cortex-exec is rebuilt. Juniper's turns are safe in either order.
  2. orion-mind and orion-thought can deploy in either order. Nothing changes until both are rebuilt.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
tests/test_held_turn_lane_rule.py, test_repair_pressure_v2_paradigm.py, test_logprob_probe_runner.py,
  test_pre_turn_appraisal_schemas.py                                  40 passed
  (the parser tests fail with the parser fixes reverted: confirmed twice)
orion/llm + orion/gpu_pool/tests + root pre-turn tests                615 passed
services/orion-mind/tests                                             250 passed
services/orion-thought/tests                                          849 passed, 32 skipped
services/orion-cortex-exec: test_pre_turn_appraisal_rpc.py,
  test_pre_turn_appraisal_concurrency.py, test_repair_pressure_speech_wiring.py, and 10 more
  related files run one by one                                        all passed
services/orion-hub/tests/test_turn_orchestrator_ws_frames.py          55 passed
Static gates: chat-route poachers, metric lineage, definition drift,
  env key single source, sentience instruments                        PASS
```

When the whole cortex-exec `tests/` folder runs in one process, collection fails with `Verb already registered: legacy...` in 14 files. This patch does not touch verb registration. CI runs that folder as separate file lists.

## Evals run

```text
orion/substrate/evals/repair_pressure_v2_eval.py                      4 passed

Live gate (the eval for this change): real repair-pressure probes, each live reply scored by
BOTH the deployed parser and this branch's final parser (staged into the running cortex-exec
container; same reply, two parsers).

World-pulse threads on the agent seats (route agent), 10 threads, 2.2k-39k chars:
  deployed parser: 10/10 text fallback (fixed scores)
  this branch:      0/10 fallback; levels 0.11 0.12 0.12 0.14 0.14 0.16 0.22 0.24 0.28 0.32
                    confidences 0.22-0.84; repair contract fired 0 before, 0 after
Earlier gate rounds, same parser family, by seat (llm_call_usage.pool_role/model):
  agent-gpu2 (Bonsai-27B, llama.cpp flat shape): 8 probes, all logprob_probe_v2
    corr e904a5ae-8bc2-4f07-9fb1-872d2419f4db, 82c4c9f4-68d3-403f-a50c-7cd12fd27964, ...
    (the 39k thread ran twice: identical level both times)
  agent / gpu1 (Qwen3.8-Flash-Next on Strata, OpenAI choices shape): 4 probes, all logprob_probe_v2
    corr c7ea6042-5693-45b7-81d0-d64ab1cce283, 92685de1-122e-4e18-a6eb-bdd244c65610, ...
  Latency 3-30 s per probe (13.8k-token prompt: 12 s gpu1, 30 s gpu2 uncached, 3 s cached).

Juniper's own recent turns on quick_turn (route unchanged), 25 turns:
  deployed parser: 25/25 text fallback
  this branch:      0/25 fallback; levels 0.00-0.42 (median 0.057); repair contract fired
                    0 before, 0 after
  (each probed with the single user message, not the Hub's full 8-message window)
```

The gate probes did not attach to a live hold. That would have taken a running reading's slot. The attach path is the one stance and judge calls already use under every reading hold (159 held stance calls served for reading turns in the same 24 h).

## Docker/build/smoke checks

```text
No image build run. The change is Python only. No dependency, port or compose change.
Live gate above ran the new parser inside the running orion-athena-cortex-exec container.
```

## Review findings fixed

Review ran in a subagent on commit a4b5037db.

- Finding: answers were mapped to kinds by position. A preamble "No" token or reordered lines put scores on the wrong kind (reproduced: trust_rupture's YES landed on coherence_gap). Too few answer tokens silently dropped kinds.
  - Fix: `_aligned_answer_entries` re-parses the lines from the token text and matches each YES/NO to the token whose span covers it. Any mismatch (glued token, missing kind, duplicate) falls back for the whole reply, never a mix.
  - Evidence: `test_a_preamble_no_token_does_not_shift_scores_onto_other_kinds`, `test_reordered_lines_score_their_own_kind`, `test_an_answer_glued_to_other_text_falls_back_for_the_whole_reply` (all fail on the first commit's parser).
- Finding: the score summed spellings, but the confidence margin still compared raw top-1 and top-2 tokens. ` YES`/`YES`/` NO` gave score 0.993 with confidence 0.03, and `min()` spread that to the whole slice.
  - Fix: margin = |log P(YES) - log P(NO)| over the summed answers. Same value as before when each answer has one spelling.
  - Evidence: `test_two_spellings_of_one_answer_are_a_confident_answer`; the existing `test_score_kind_from_answer_token_uses_margin_as_confidence` still passes unchanged.
- Finding: the parser fix changes Juniper's turns too, not only held turns.
  - Fix: measured, not split. Same live replies scored by both parsers (Evals run): Juniper 25/25 fallback before, 0/25 after; the repair contract fired 0 times before and 0 after. Stated as a risk below.
  - Evidence: Evals run.
  - Found while measuring: 6 of 25 of Juniper's replies still fell back, because a very sure answer's opposite was outside the top 8. Fix: bound it by the lowest listed logprob. Test: `test_an_answer_missing_from_top_k_is_bounded_not_dropped`. Re-run: 0/25 fallback.
- Finding: Mind phase timing on the agent seat is unverified live.
  - Fix: stated as UNVERIFIED with the measured pre-patch numbers (Risks) and a post-deploy check (Restart required).
- Finding: deploy order (cortex-exec before Hub) is not enforced by tooling.
  - Fix: stated in Schema/bus and Restart required.
- Finding (nit): the Mind exception record named the configured small-lane routes for held runs.
  - Fix: routes go through `held_call_route`. Test: `test_held_run_exception_record_names_the_seat_route`.
- Finding (nit): `held_lease` came from the semantic call only.
  - Fix: one `held_call_route(req.gpu_lease, "")` for the lease, one loop for the three routes.
- Finding: the Hub link ran only in `orion-hub-full-suite`.
  - Fix: a `hub` job in `held-turn-lane-rule-tests.yml` runs the three Hub hold tests.

## Restart required

Deploy order matters. Run from the clean primary main checkout after merge (post-pull rebuild does this):

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Check after deploy (24 h): world-pulse leases on fast or metacog should be 0.

```sql
SELECT holder, work_class, coalesce(role,'none'), count(*) FROM gpu_pool_leases
WHERE created_at > now() - interval '24 hours' AND holder IN ('cortex-exec','orion-mind')
  AND work_class IN ('fast','metacog') AND hold_lease_id IS NULL
  AND turn_correlation_id IN (SELECT turn_correlation_id FROM gpu_pool_leases
                              WHERE hold_lease_id IS NOT NULL AND created_at > now() - interval '24 hours')
GROUP BY 1,2,3;
```

Also check on one stage-2 reading turn: cortex-exec logs `pre_turn_appraisal_paradigm_result ... route=agent held=True`; orion-mind phase telemetry shows `route=agent` and no phase timeout; `repair_pressure_appraisal_log.confidence` is no longer pinned at 0.65. Mind timing under a gpu2 hold is UNVERIFIED until this runs.

## Risks / concerns

- Severity: medium. Concern: Juniper's own repair readings change. The parser fix is not limited to held turns. Her fast-lane probes stop returning the fixed fallback (always 0.087 per kind at 0.65 confidence) and return real scores. The `repair_concrete` contract gate (`level >= 0.75 and confidence >= 0.60`) and the equilibrium metacog gate now see real numbers for the first time since about 10-23. Mitigation: this is the probe's designed behaviour. On 25 of her real recent turns the contract fired 0 times before and 0 after (levels 0.00-0.42). Confidence now spreads 0.03-1.0 instead of a flat 0.65, so the equilibrium gate's trigger floor sees different numbers. Watch `repair_pressure_appraisal_log` for a week.
- Severity: medium. Concern: Mind phases on a gpu2 hold may be slow. Before this patch, 8 Mind calls that spilled onto agent-gpu2 ran out of their 60 s budget (frontier judge runs with thinking on). Under the hold the call is seated at once, but Bonsai generation speed is unchanged. Mitigation: a timed-out phase is recorded as a fail-open Mind run, never as fabricated output. If gpu2 Mind runs keep timing out, the follow-up is to measure them by role and decide on thinking or the token budget there. That is a cognition change, so it is not in this patch.
- Severity: low. Concern: two models score reading checks. Qwen3.8-Flash-Next (gpu1) and Bonsai-27B (gpu2) gave close levels on the same 39k thread (0.116 vs 0.114), but different per-kind scores. Juniper's checks use Qwen3-8B. Equilibrium reads all of them the same way. Mitigation: `llm_call_usage.model` joins by correlation id, so readings can be split by model later.
- Severity: low. Concern: a held probe that cannot attach (hold recalled, stale generation) fails the check with the gateway's reason. It does not retry on `quick_turn`. This is the intended behaviour under Juniper's rule.

## PR link

(added after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
