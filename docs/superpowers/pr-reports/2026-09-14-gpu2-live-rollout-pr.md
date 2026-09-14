# GPU2 live rollout and cabinet endpoint correction

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
`1ab72f626` and Gateway adds `fefa6e782`. The existing GPU1 agent container
was not recreated. Image digests below identify the actual containers.

| Host | Container | Image SHA256 |
| --- | --- | --- |
| athena | `orion-athena-thought` | `2e11f1d73c85f30d09a12c4180f8e3096e8a46d741be8e861330d07e18bd2e9a` |
| athena | `orion-athena-feedback-runtime` | `a1c1351cab8531eb290e93f70ddc948a0dff1dea0d4458ccfd0bcf7bb8d51ef3` |
| athena | `orion-athena-execution-dispatch-runtime` | `643e05a418fbb4b4a2eb5a856333f24d52ddf56ac137b04e6d1ee4319a2d4a01` |
| athena | `orion-athena-proposal-runtime` | `65e6cd910c7c2638f4b5363f8eb4ffa37b0befb1c28b7d2a273f922d154555a9` |
| athena | `orion-athena-durable-runs` | `4e70d3c3afc88dc4348c4337d2720a855f245d17f93000641aa5c89f6493d267` |
| athena | `orion-llm-gateway` | `c58a0d3a6c5bf7814d07363defa7f55b9f9cd8caed96f98dc5d309a70fe6d5f7` |
| athena | `orion-athena-hub` | `f607e29cec3561a0ea4de67d55588ca36814590ce96b900514d62e67fd26ccb6` |
| athena | `orion-athena-harness-governor` | `ddf751a357a0a03cab8feb1059f20d86cce35253511b8eeec0ebb1dc72285084` |
| athena | `orion-athena-cortex-exec-spark` | `67acd57d0ad32beaaeae1a5f0a564f9c45d491e4eb9e3a0933b45421dfd47382` |
| athena | `orion-athena-cortex-exec-chat` | `b65c0df725861537bdf38c60e846639e2a6a3a2b828bc23d97065f88320566a4` |
| athena | `orion-athena-cortex-exec` | `1653bef8d40854aa51486d5ec1daa6b211da63f90d84790038ad959c624e37ed` |
| athena | `orion-athena-cortex-exec-background` | `ae50861fdd08e70c8c16e69e2636dd6288d97885a4e5f0bfceb2b8d6247029e7` |
| athena | `orion-athena-cortex-orch` | `e61c6e20a87640001b0b6b190bb99ee2c9f93396729a6f8fe21daf15644b47b6` |
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
