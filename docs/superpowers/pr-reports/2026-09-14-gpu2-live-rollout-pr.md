# GPU2 live rollout — completed

The full physical sequence completed: real 1200-second queue wait, automatic
GPU2 borrowing, fenced FCC execution, persisted result, ownership release and
automatic diffusion restoration. The run required the runtime fixes and retries
documented below. All activation and safety flags remain live. No runtime
blocker remains. GitHub requires one approving review before PR #2218 can merge.

The merged GPU2 controller and diffusion drain API are now deployed on Circe.
The first physical round-trip exposed a deployment error: Hub uses host
networking, so durable-runs cannot resolve `orion-athena-hub` through bridge DNS.
Use Athena's Tailscale address for cabinet readings in settings, Compose, the
operator template and the actual production env. This restores real thermal
eligibility without changing any activation or safety gate.

The architecture document now correctly states that GPU2 control uses the
internal service/tailnet boundary without bearer tokens; GPU1 retains its
existing authentication and callers cannot supply arbitrary Docker arguments.

## Deployment and runtime evidence

- Source baseline on Athena and Circe: `e642932d3`, including merges #2215/#2216.
- Local envs synchronized on both hosts, including Circe's six missing GPU2
  keys and Athena's explicit lane policy. All requested enable flags remain
  true; both shadow flags are false; widening remains 1200 seconds.
- Installed missing additive `reverie_visual_attempt` and
  `visual_baseline_checkpoint` schema; confirmed all admission, lease, capacity
  and GPU2 tables exist in the configured production database.
- Real baseline production: chain `385a652b-8d03-477d-90ea-570f08e897d8`,
  831679-byte image, SHA256
  `582019276046ed14be359f92837f1765873cafe5f8d51a2b61a52ceb902f2000`,
  produced `2026-09-14T02:23:26.108336Z`. This cleared overdue baseline debt.
- Agent capability audit through Gateway produced `{"ok":true}` under an
  enforced JSON schema. Live model is
  `/models/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf`, context 131072, vision false.
- Manual durable intent `gpu2:1:agent-burst` at `02:23:43.279919Z` drained and
  stopped diffusion, loaded burst on GPU2 and reached controller readiness in
  49.37 seconds. Gateway subsequently advertised the exact model/context.
- Unleased burst call returned HTTP 503,
  `agent_burst_requires_durable_capacity_lease`.
- Durable intent `gpu2:2:diffusion` at `02:24:55.762490Z` restored diffusion
  after zero leases/permits and an idle upstream slot. Controller transition
  25.82 seconds, diffusion cold start 14.46 seconds; `/ready` returned true.
- Initial contention probe `gpu2-live-20260914T022620` was submitted at
  `02:26:21.287170Z` behind real streamed Gateway inference on agent. Its wait
  held no durable graph worker. The ordinary request reached Gateway's real
  900-second deadline; the run correctly took the newly free preferred lane at
  `02:41:21.037049Z`. This is not a passing automatic GPU2 acceptance.
- A fresh test uses a temporary HTTP Gateway instance of the same production
  image and shared capacity authority with a 3600-second request budget. It
  does not consume bus requests or alter production activation/request flags.
  Temporary contention gateway was removed after the burst grant.

## FCC compatibility correction

The first preferred-lane run exposed a real native Anthropic failure: Claude
SessionStart hooks append a `system` message after the user message, causing
the Qwen template to return HTTP 500 (`System message must be at the beginning`).
Gateway now hoists those blocks into the top-level Anthropic system field,
preserving content, cache metadata and conversational/tool order. A captured
real Claude request reproduced the failure; the normalized request rendered
successfully against the same live model. The automatic run below is the
end-to-end inference evaluation. Gateway has no separate periodic eval harness.

## Review findings fixed

- Finding: malformed hook context could throw after acquiring a permit, before
  entering its cleanup block.
  - Fix: normalize and validate before capacity acquisition; reject unsupported
    system content with HTTP 400.
  - Evidence: both malformed-content regressions assert no capacity acquisition.

## Checks and review

- 21 Gateway Anthropic tests pass, including the captured hook-message shape,
  string/list system content, cache metadata, tool order and malformed content.
- 15 focused elastic API/policy tests and the cabinet deployment-contract
  regression pass. Regression checks the host-network topology and agreement
  of all cabinet defaults on a tailnet address. Reachability was verified by a
  live GET from inside the deployed durable-runs container.
- All affected services built and deployed through `safe_docker_build.sh` in
  linked worktrees; primary ignored envs were synchronized.
- Independent code review: no material findings. Added the recommended
  cabinet-address contract regression.
- Sanitized detailed runtime snapshots are retained under
  `/tmp/gpu2-live-evidence` on Athena. They contain no bearer credentials.

World Pulse's reading queue is not integrated with this admission seam.

## Deployed identities

All rebuilt services use source baseline `e642932d3`; durable-runs adds
`e41318fff`, Gateway adds `fefa6e782`, governor adds `3bf2a697e`, and
Cortex Orchestrator adds `e41318fff`. The existing GPU1 agent container
was not recreated. Image digests below identify the actual containers.

| Host | Container | Image SHA256 |
| --- | --- | --- |
| athena | `orion-athena-thought` | `2e11f1d73c85f30d09a12c4180f8e3096e8a46d741be8e861330d07e18bd2e9a` |
| athena | `orion-athena-feedback-runtime` | `a1c1351cab8531eb290e93f70ddc948a0dff1dea0d4458ccfd0bcf7bb8d51ef3` |
| athena | `orion-athena-execution-dispatch-runtime` | `643e05a418fbb4b4a2eb5a856333f24d52ddf56ac137b04e6d1ee4319a2d4a01` |
| athena | `orion-athena-proposal-runtime` | `65e6cd910c7c2638f4b5363f8eb4ffa37b0befb1c28b7d2a273f922d154555a9` |
| athena | `orion-athena-durable-runs` | `bd485723ebc44fd57598a118a4f2e6c763fcf06cc8dc60eaab689df16730ecd3` |
| athena | `orion-llm-gateway` | `c58a0d3a6c5bf7814d07363defa7f55b9f9cd8caed96f98dc5d309a70fe6d5f7` |
| athena | `orion-athena-hub` | `f607e29cec3561a0ea4de67d55588ca36814590ce96b900514d62e67fd26ccb6` |
| athena | `orion-athena-harness-governor` | `bc866cd89da8a654e8ad7cd69de39e9adde9e1c9c932f862e3bffcf5aef3ae3f` |
| athena | `orion-athena-cortex-exec-spark` | `67acd57d0ad32beaaeae1a5f0a564f9c45d491e4eb9e3a0933b45421dfd47382` |
| athena | `orion-athena-cortex-exec-chat` | `b65c0df725861537bdf38c60e846639e2a6a3a2b828bc23d97065f88320566a4` |
| athena | `orion-athena-cortex-exec` | `1653bef8d40854aa51486d5ec1daa6b211da63f90d84790038ad959c624e37ed` |
| athena | `orion-athena-cortex-exec-background` | `ae50861fdd08e70c8c16e69e2636dd6288d97885a4e5f0bfceb2b8d6247029e7` |
| athena | `orion-athena-cortex-orch` | `f398762a7d7888f2128d2cf66be27548acc4c209f9f80b6e807588464f8b4905` |
| circe | `orion-circe-diffusion-host` | `b90400846444d114f00e1bfc34a344fb5ab388d276af340c792cea8df4bc3ad1` |
| circe | `orion-circe-gpu-lane-controller` | `3bf6da7e35e9224c0f28594156139a5c225a1cc1f524022603be622386e68b7c` |
| circe | `orion-circe-atlas-llamacpp-agent` | `684e9fea45b78201fcb0be21027a59dcecb2978a8080172061667e997d620f1b` |
| circe | `orion-circe-atlas-llamacpp-agent-burst` | `f8e2a0793aceb9f99ff8617fbb0d1a1a701ed5501d3c3c5062f50c56bdace912` |

## Effective production configuration

These values were read from running containers after recreation:

```text
GPU2_ENABLED=true
DURABLE_RUNS_ADMISSION_ENABLED=true
DURABLE_RUNS_CAPACITY_ENABLED=true
DURABLE_RUNS_ADMISSION_SHADOW=false
DURABLE_RUNS_WIDENING_ENABLED=true
DURABLE_RUNS_WIDENING_AFTER_SEC=1200
DURABLE_RUNS_ELASTIC_ENABLED=true
DURABLE_RUNS_ELASTIC_SHADOW=false
DURABLE_RUNS_ELASTIC_ASSIGNMENTS=true
DURABLE_RUNS_ELASTIC_RESTORATION=true
DURABLE_RUNS_ELASTIC_THERMAL_ENABLED=true
ORION_VISUAL_ELASTIC_STATUS_ENABLED=true
HUB_CURIOSITY_ELASTIC_ACTIVATION_ENABLED=true
```

Parsed durable lane policy:

```json
{
  "agent": {
    "capabilities": {
      "structured_output": true
    }
  },
  "agent-burst": {
    "compatible_with": [
      "agent"
    ],
    "activatable": true,
    "activation_model": "/models/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf",
    "capabilities": {
      "structured_output": true
    },
    "quality_drop": 0,
    "switching_cost_seconds": 120
  }
}
```

## Physical mapping

| Physical GPU | Owner | Host port | Profile / context |
| --- | --- | --- | --- |
| Circe GPU1, V100-SXM2 32GB | Existing agent | 8015 | agent-flex / 131072; unchanged |
| Circe GPU2, Tesla PG500-216 32GB | Diffusion normally; burst during borrowing | 8014 diffusion; 8016 burst | `qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex` / 131072 |

Controller uses port 8090. The exact cached model is
`/mnt/telemetry/llm-cache/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf` (17,559,178,144 bytes).
Both controller and diffusion have one Python worker. Burst remains normally
stopped and controller-owned. No GPU1 affect/agent topology or authentication changed.

## Automatic acceptance sequence

Run `gpu2-live-20260914T024447` was submitted at `02:48:43.505774Z`.
The real preferred-lane stream held permit `436ccda000254761a467ce6e8bc15712`
until the burst grant; it emitted 28,570 SSE chunks before test-owned cancellation.
No timestamps were backdated and no manual assignment was made.

- Automatic intent `gpu2:3:agent-burst`: `03:08:44.107113Z`, **1200.601339
  seconds** after submission.
- 123 queue/activation observations all had no lease and no active durable
  graph worker. The broker resumed the run only after physical readiness.
- Controller reached burst readiness at `03:09:05.786Z`, measured transition
  15.79 seconds. Gateway discovery briefly lagged; idempotent reconciliation
  retried the same operation without consuming an inference attempt.
- Fenced lease `f3578ea169e945458a902066cb125d25`, generation 2, granted
  `03:09:11.021939Z` for `llm.route.agent-burst` at Circe:8016.
- First execution permit `5cf3745cef944c5fb184cc4a82997fb0`; correlation
  `1c4c74e1-c53a-5f66-a43e-18634fbecaba`. The preparatory inference returned
  2671 completion tokens with `finish_reason=stop` on the exact expected model.
- Native FCC `/v1/messages` requests returned HTTP 200 through the same fenced
  burst route; real tool use followed, with no API errors in the FCC session.
- Visual probe at `03:09:15.180868Z` returned `deferred_resource` and
  `artifact_persisted=false`. The prior success timestamp
  `02:23:26.108336Z` and image hash were identical before and after.

Thermal suppression is covered by the existing activation-policy regression;
live cabinet observations remained fresh and eligible during this acceptance.
No thermal, baseline, assignment, restoration or widening gate was bypassed.

## FCC progress accounting correction

The first burst FCC attempt performed real tool reads but hit a false draft
ceiling at `03:17:12Z`: `524402 >= 524288` characters. Claude emits a
`system/thinking_tokens` counter for individual tokens. The generic fallback
counted its serialized UUID/session metadata as content and published each
counter as a grammar step, also bloating the finalizer input. A tiny real probe
produced 21 counters totaling 4257 metadata characters despite completing normally.

The motor now excludes exactly those progress counters from steps and accounting;
actual assistant thinking text counts against the unchanged ceiling. Review
found no material issue. All 51 focused motor/context-budget tests pass, including
2500 progress ticks with a valid small result and oversized real thinking that
still triggers the ceiling. The failed attempt is not counted as successful FCC
acceptance; its normal durable retry retains the assigned burst lane.

The corrected live motor probe completed with a real `Read` tool receipt,
11 recorded steps and exit code 0 in 74.762 seconds. Its answer accurately
described the inspected README; session `584a27e7-7275-4d10-86ee-6a46ef0cf899`.

The failed automatic attempt released ownership. Protected restoration intent
`gpu2:4:diffusion` at `03:28:01.175818Z` completed at `03:28:25.603059Z`,
transition 24.37 seconds and cold start 12.63 seconds. `/ready` reported
`ready=true`, `draining=false`, `model_loaded=true`, `load_error=null`.
The run remained durable with no lease/worker during the configured 600-second
diffusion residency interval, retaining its original burst assignment for retry.

## Resource-event registry correction

Live resource-event publication raised `Unknown schema_id: ResourceEventV1`.
The model existed in the kind registry but was missing from the lookup used by
real bus envelope validation. The existing global parity test reproduced this
and also caught the related missing `DurableRunReceiptV1` lookup. Both aliases
now resolve; a service regression exercises the actual resolver, and the global
parity test is now a static CI step. Two global registry tests and the dedicated
resource-event regression pass.

The retained retry automatically requested `gpu2:5:agent-burst` at
`03:38:29.039122Z`. An immediate `rpc:ConnectionError` released its lease and
was retried by the existing policy. The following attempt reached the corrected
FCC motor with correlation `f0e8a62e-ce7e-5b70-be56-e3218a150685`.

## Completed FCC and persisted run result

The successful attempt used lease `df15c9bc314445c18122d802526acc4f`,
generation 4, granted at `03:40:00.813551Z` on the retained burst route.
FCC session `57b1c972-dce0-4628-bcc2-6d07745e1632` completed its answer at
`03:48:30Z`, after ten successful tool receipts and 37 grammar events. There
were no API errors in this attempt. Finalization reported `finalize_failed=false`
and `response_repair_skipped reason=aligned`: the original motor answer was
retained, not substituted with repair text.

The durable run completed at `03:49:24.643272Z`, with **3 attempts** including
the two failures documented above. Journal identity:
`curiosity-self-inquiry:gpu2-live-20260914T024447`. Stored finding: 3403
characters; SHA256 `7c8def94567798fb7737d29f663aa886673d0934ea1fceeb14e84d7859344eea`.

The answer contains a substantive source/eval audit. It also repeats deployment
claims from an older report (including its old 404), so its rollout assessment
is stale. This is a recorded eval limitation, not evidence of current runtime
state. Current deployment claims in this report come from physical/container,
HTTP, database and bus observations instead.

After registry deployment, a real subscriber captured 55 resource events for
this run, including its lease release and terminal completion. The production
resource-event outbox reached **zero unpublished rows**. Cortex Orchestrator,
the durable receipt consumer, was also rebuilt with the lookup correction.

## Final automatic restoration and ownership evidence

Baseline debt became urgent at its original due time. The persisted event
requested `gpu2:6:diffusion` at `03:53:26.608673Z` with
`reason=visual_baseline_urgent`. This is live evidence that preserved baseline
debt affects ownership; no timer or safety flag was changed for the test.

Restoration completed at `03:53:51.189758Z`: transition **24.54 seconds**,
diffusion cold start **12.86 seconds**. Controller reports active diffusion,
burst container exited, no error, generation 6. Diffusion `/ready` reports
`ready=true`, `draining=false`, `model_loaded=true`, `load_error=null`.
The authority reports **zero active leases and zero active permits**.

All 17 burst permits in the automatic run are released and carried a lease.
The successful attempt used these nine permit identities:

```text
f93f7e07c5a143c5a6b0971a07a07fbd
8894321f395e47b2b684fce20a83ee0b
677d94024c5e43ccbd3d9403e2ff9242
30f9e50c9a60423da6f0952dbb9b7d6e
4aad0e49f0cb4d7b9203a4e37a3353ec
34e97062277c475a94bd04f2f22a6e61
99543a0d73824dd7975839bc6db750cb
2bd4f529fba44b79bbb742f233c63155
f28554f7d0bf4c51b0e4a4a108763370
```

The original automatic drain acknowledged at `03:08:50.316140Z`; idle status
followed at `03:08:50.319166Z` and application shutdown completed at
`03:08:50.662764Z`. No generation was interrupted. Actual measured transitions
fit the conservative declared budgets, so the budgets were not reduced.

Final physical snapshots, complete polling history, FCC/run result, sanitized
image/config inventory, bus delivery and permit records are retained in
`/tmp/gpu2-live-evidence` on Athena. The temporary contention Gateway was removed.
Athena primary checkout is clean; Circe’s three unrelated hardware-log files
remain intact. No env was committed and no GPU2 credential was introduced.

Final independent evidence review found no material misclaims. Final env parity
passes for all 10 affected Athena service templates and all 3 Circe templates.
Seven CI jobs pass on the code revision, including static registry parity,
isolated-Postgres admission/fairness checks, Gateway transport, SQL writer,
reading and browser smoke. No merge conflicts remain.
