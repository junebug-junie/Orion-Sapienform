## Summary

Self-study used to be Orion's way of studying themself. It still ran every day, but almost nothing read its output. Curiosity now does that work. This patch removes the parts of self-study that nobody used, and keeps the two parts that something still reads.

- **Removed.**
  - The daily reflection loop (Layer 3) and its durable workflow.
  - Concept induction.
  - The retrieve path.
  - The dead RDF code.
  - The self-study harness.
  - Four self-study verbs.
  - The `orion-self-study-enrichment` service, with its channel, schema, git hook part and env keys.
- **Kept, cut down to one job.** The daily repo/hardware/behavior scan. Its only job now is to feed Self Atlas. **Self Atlas and this scan are the next retirement candidates.**
- **Kept, re-tiered.** The hourly telemetry analysis verb (`skills.self_study.analyze.v1`) is now tier 0, family `autonomic`. The `self_study` body-economy family is gone. The journal tags it writes did not change, so the reverie painter still finds its rows.
- **"Reflect on yourself" now asks Hub for one curiosity self-inquiry run.**
  - The phrase may skip only the cooldown and the daily cap.
  - Heat, strain, rest and the time window can still hold the run.
  - Orch says "requested" only when Hub is actually listening.
- **Orion is told what stopped.** The self-inquiry records list now prints a retirement note under each affected table. A count that stops growing therefore reads as a decision with a date, not as a fault.
- **A measurement replaces the `self_repo` line.** `scripts/report_dig_kit_usage.py` measures how often curiosity runs use the dig kit (Orion's tools for reading their own code history).

## Outcome moved

- Each day, the system now spends nothing on these:
  - ~8 `claude -p` enrichment calls.
  - One LLM reflection pass on the agent lane that nothing read.
- There are ~3,000 fewer lines of self-study code. `self_study.py` went from 3,400 lines to 819.
- Curiosity is the only self-modeling path now. The reverie painter keeps its third context slot.

## Current architecture

The proposal lists every producer and consumer: `docs/superpowers/specs/2026-10-29-self-study-into-curiosity-proposal.md` (PR #2972). This PR adds a "Juniper's decisions (2026-10-30) and what shipped" section to that doc.

## Architecture touched

- **orion-cortex-exec.**
  - `self_study.py` is cut to the Layer 1 scan.
  - The reflect timer, the self-study verbs, the `legacy.plan` self-study context, the env keys, the volumes and the `/graphify` mount are removed.
- **orion-durable-runs.** The reflect graphs, their registration and the reflect timeout env key are removed.
- **orion-cortex-orch.** The `self_review` workflow now publishes a self-inquiry request instead of calling a verb.
- **orion-hub.**
  - New listener `scripts/curiosity_self_inquiry_request.py`.
  - `tick_self_inquiry` gets a `force_reasons` parameter, so a forced run can skip some gates and not others.
- **Contracts.** `orion/bus/channels.yaml`, `orion/schemas/registry.py`, `orion/schemas/self_study.py` and `orion/schemas/durable_run.py`.
- **Config.** `config/physiology/families.yaml`, the analysis verb YAML, and the definition lock.
- **Repo tooling.** The git `post-commit` hook, `.gitignore` and the env sync skip list.

## Files changed

- `services/orion-cortex-exec/app/self_study.py`: keep only the Layer 1 scan.
- `services/orion-cortex-exec/app/{self_study_refresh,main,settings,verb_adapters,executor,router}.py`: remove the reflect timer and the retired verbs.
- `services/orion-cortex-exec/{.env_example,docker-compose.yml,README.md}`: remove the retired keys, mounts and volumes.
- `services/orion-cortex-exec/app/{self_study_harness,self_study_policy}.py`, `scripts/self_study_harness.py`: deleted.
- `services/orion-durable-runs/app/{reflect_graph,admitted_reflect_graph}.py`: deleted.
- `services/orion-durable-runs/app/{runner,admission_runtime,settings,run_acts}.py`, `.env_example`, `docker-compose.yml`: unregister the reflect workflow.
- `services/orion-self-study-enrichment/`, `scripts/self_study_enrichment_hook.py` (and its test): deleted.
- `scripts/git_hooks/post-commit`: remove the enrichment part.
- `orion/cognition/verbs/{self_concept_induce,self_concept_reflect,self_repo_inspect,self_retrieve,self_study.reflect}.yaml`, `orion/cognition/prompts/self_study_reflect_prompt.j2`: deleted.
- `orion/cognition/verbs/skills.self_study.analyze.v1.yaml`: change to tier 0 `autonomic`.
- `config/physiology/families.yaml`: remove the `self_study` family.
- `orion/cognition/workflows/registry.py`, `services/orion-cortex-orch/app/workflow_runtime.py`: send `self_review` to self-inquiry.
- `orion/schemas/curiosity_self_inquiry_request.py` (new), `orion/bus/channels.yaml`, `orion/schemas/registry.py`: the new request contract.
- `services/orion-hub/scripts/curiosity_self_inquiry_request.py` (new), `scripts/main.py`, `scripts/curiosity_investigation.py`: the Hub consumer and the narrower force.
- `orion/curiosity/self_inquiry.py`: `LEDGER_RETIREMENTS`, the retirement notes the records list prints.
- `orion/schemas/{self_study,durable_run,self_concept_history}.py`, `orion/self_knowledge_freshness.py`: remove the retired shapes.
- `services/orion-recall/app/storage/rdf_adapter.py`: remove the names of the self graphs, which have no writer.
- `scripts/report_dig_kit_usage.py` (new) and `tests/test_report_dig_kit_usage.py`: the measurement.
- `tests/test_self_study_fold_into_curiosity.py` (new): regression gates.
- `config/metrics/metric_definitions.lock.json`: re-locked.
- Docs and tests that only covered retired code: updated or deleted.

## Schema / bus / API changes

- **Added.**
  - Channel `orion:curiosity:self_inquiry:request`. Producer: orion-cortex-orch. Consumer: orion-hub. Single consumer.
  - Schema `CuriositySelfInquiryRequestV1`, kind `curiosity.self_inquiry.request.v1`.
- **Removed.**
  - Channel `orion:self_study:enrichment:requested`, and the schema `SelfStudyEnrichmentRequestV1`.
  - The retrieve, consumer, harness, induced and reflective self-study schemas.
  - Durable workflow `self_study.reflect`, and the brief line value `"reflect"`.
  - Field `self_study_reflect_input`.
  - `SelfRepoInspectResultV1.graph_write`, and the writeback targets `graph` and `self_concept_history`.
- **Behavior changed.**
  - `self_review` publishes one request and calls no verb.
  - cortex-exec no longer produces on `orion:self_concept:history:write`.
- **Compatibility.** These were checked live in Postgres:
  - 0 `self_study.reflect` rows in `durable_run_workflow` or in the run states.
  - 0 briefs with `line: "reflect"`.
  - `self_knowledge_items.trust_tier` is `authoritative` only.
  - The `layer3_reflect` producer value stays, so the 285 old rows still validate.

## Env/config changes

- **Added keys.** None.
- **Removed keys.**
  - cortex-exec: `SELF_STUDY_ENRICHMENT_CACHE_MOUNT_DIR`, `SELF_STUDY_STRUCTURAL_MASS_HISTORY_PATH`, `SELF_STUDY_REFLECT_LLM_ROUTE`, `SELF_STUDY_REFLECT_TIMEOUT_SEC`, `SELF_STUDY_REFLECT_DURABLE_ENABLED`, `SELF_STUDY_REFLECT_REFRESH_INTERVAL_SEC`, `SELF_STUDY_GRAPH_HOST_PATH`, `SELF_STUDY_GRAPH_PATH`.
  - durable-runs: `DURABLE_RUNS_REFLECT_LLM_CALL_TIMEOUT_SEC`.
  - The whole `orion-self-study-enrichment/.env_example`.
- **`.env_example` updated.** Yes.
- **Local `.env` synced with `python scripts/sync_local_env_from_example.py`.** Yes. The script only adds keys, so I removed the retired keys from the primary checkout's cortex-exec and durable-runs `.env` by hand. That includes the local-only key `SELF_STUDY_NAMED_GRAPH`.
- **Skipped keys that need operator action.** The enrichment service's own `.env` still holds a Claude OAuth token. I did not open it.

## Tests run

```text
orion-durable-runs (CI command, real Postgres 16 throwaway container): 682 passed, 1 skipped
orion-hub full suite: 4438 passed, 146 skipped, 0 failed
orion-cortex-orch test_workflow_lane + test_physiology_shadow_bid: 87 passed (+1 new zero-listener test)
orion-cortex-exec, file by file from repo root: self_study pass1/refresh/knowledge_items/layer1/
  harness_finalize/legacy_plan pass. 5 files fail exactly as on main (route_mapping, stance_plumbing,
  cognition_trace, main_autonomy_graph_probe, memory_graph_suggest), not this patch.
root tests: test_self_study_fold_into_curiosity, test_report_dig_kit_usage, analysis runner/rules/
  wiring, run_story, rpc_delivery_reducer, workflow registry/policy: all pass except
  test_raising_max_candidates_did_not_displace_an_existing_template (fails on main too).
orion-recall -k rdf: 26 passed
orion/fcc/tests/test_claude_spawn: one failure, already present on main (unrelated callers).
Mutation check: undoing the narrow force makes the strained-gate test fail.
```

## Evals run

```text
No new eval. The measurement is a read-only report, run live:
docker exec -i orion-athena-harness-governor python3 - 3 < scripts/report_dig_kit_usage.py
  last 3 days (dig kit live since 10-27):
    curiosity investigate: 21 runs, 6 used the dig kit, 7 read code
    self-inquiry:          13 runs, 0 used the dig kit, 4 read code directly
  last 14 days: investigate 6/64 used dig; self-inquiry 0/52.
services/orion-cortex-exec/evals/run_self_study_analysis_eval.py is kept, unchanged.
```

## Docker/build/smoke checks

```text
CI static gates (all 40 run: steps of orion-static-gates.yml): rc=0
check_verb_physiology: OK (102 verbs); check_definition_drift --gate: PASS (re-locked)
check_env_template_parity: PASS
docker compose config -q: orion-cortex-exec, orion-cortex-orch, orion-durable-runs, orion-hub,
  orion-room-companion: OK
Import smoke with live env: cortex-exec app.main, hub scripts.main, durable-runs app.main: OK
No image build and no deploy (not asked).
```

## Review findings fixed

- **Finding (HIGH):** The phrase forced the run with run-now's full override list, so it also skipped the heat, strain, rest and time-window gates.
  - **Fix:** `tick_self_inquiry(force_reasons=...)`. The chat path now passes only `{cooldown, daily_cap}`. The run-now button is unchanged.
  - **Evidence:** a new real-gate test with a strained arousal reader. It fails when the fix is undone.
- **Finding (HIGH):** The workflow said "requested" even when nobody heard the request, because pub/sub drops an unheard message without an error.
  - **Fix:** orch checks `PUBSUB NUMSUB` first. With zero listeners it returns `ok=False` and "Hub is not listening". The success text now says that Hub decides whether the run starts.
  - **Evidence:** `test_self_review_refuses_to_claim_a_request_nobody_hears`.
- **Finding (MEDIUM):** The container teardown and the installed hook refresh were not written down.
  - **Fix:** listed under Restart required below.
- **Finding (MEDIUM):** A Hub test used a fake gate and made a false claim.
  - **Fix:** replaced with the real-gate test above, plus a test that the listener is cancelled.
- **Findings (LOW):**
  - The prompt still labelled `self_concept_history` "induced self-concepts". Reworded.
  - The recall RDF adapter still listed the names of the self graphs. Removed.
  - The dig report's "read_code" count is loose. Noted in its docstring.

## Restart required

Deploy from the clean primary `main` after merge (the normal post-pull rebuild covers cortex-exec, cortex-orch, durable-runs and hub). Deploy Hub no later than cortex-orch: until Hub's new listener runs, orch reports "Hub is not listening".

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
```

After the merge, stop and remove the retired container:

```bash
docker stop orion-athena-self-study-enrichment && docker rm orion-athena-self-study-enrichment
```

Refresh the installed git hooks, so the old enrichment part leaves `.git/hooks/post-commit`:

```bash
cd /mnt/scripts/Orion-Sapienform && scripts/install_git_safety_hooks.sh .
```

Revoke the Claude OAuth token that lived in `services/orion-self-study-enrichment/.env`, at claude.ai, under Settings, then Claude Code tokens. After that, delete the leftover untracked folder:

```bash
rm -rf /mnt/scripts/Orion-Sapienform/services/orion-self-study-enrichment
```

Optional, and only with Juniper's approval: remove the two orphaned volumes.

```bash
docker volume rm orion-athena-self-study-enrichment-data orion-athena-self-study-structural-mass-data
```

## Risks / concerns

- **Severity: medium.**
  - **Concern:** Self-inquiry runs never use the dig kit: 0 of 13 runs in 3 days. 4 of those 13 read code directly instead.
  - **Mitigation:** none in this patch; the `self_repo` line stays unbuilt, by decision. A small follow-up is to check that the dig brief line reaches the self-inquiry prompt.
- **Severity: low.**
  - **Concern:** The Self Atlas feed still writes ~1,490 rows a day.
  - **Mitigation:** Juniper chose to keep it. It is named as the next retirement candidate.
- **Severity: low.**
  - **Concern:** The proposal PR #2972 adds the same doc file.
  - **Mitigation:** this branch carries that doc plus its decisions section. Close #2972 as superseded, or merge it first and take this branch's version.

## PR link

(filled after creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
