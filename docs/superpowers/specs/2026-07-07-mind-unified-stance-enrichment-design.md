# Mind → Unified Turn Stance Enrichment (advisory self/frontier coloring)

- Date: 2026-07-07
- Status: Design (approved framing; pending written-spec review)
- Owner: Juniper
- Service touched (primary): `services/orion-thought`
- Contract touched: none on the wire (internal verb-context only); optional artifact republish reuses `MindRunArtifactV1`

## Arsonist summary

Orion has two cognition pipelines that never talk. **Mind** (`services/orion-mind`, `POST /v1/mind/run`) produces a rich, evidence-backed, multi-phase stance synthesis, but today it only runs on the legacy **brain / `chat_general`** path (Hub → cortex-gateway → cortex-orch), where cortex-orch builds Mind's snapshot and can even let Mind *drive* the stance (`mind_skip_stance_synthesis`). The **unified turn** (Hub `turn_orchestrator` → `orion-thought` → `orion-harness-governor`) computes its stance cold via the `stance_react` verb and produces a comparatively lean `ThoughtEventV1` (`imperative` + `tone` + `stance_harness_slice` + `grounding_capsule`). Mind never runs there.

We want the unified turn to feel "more alive" by borrowing Mind's *self/attention* synthesis — **without** contradicting the association/grounding already present, and **without** forcing chat framing onto tooling/coding/agent turns.

The smallest useful seam: run Mind inside `orion-thought` before the `stance_react` call, select only the **mode-agnostic self/frontier subset** of Mind's output, and inject it as an **advisory** `mind_coloring` block into the `stance_react` verb context. `stance_react` stays the sole author of `ThoughtEventV1`, so it reconciles Mind's coloring against the association/grounding it already holds. Fail-open, flag-gated, cheap observability preserved.

## Current architecture

- Unified stance stage: `services/orion-thought`
  - `app/bus_listener.py`
    - `run_stance_react(request, *, bus, cortex_client)` → builds a `stance_react` plan request, executes via cortex-exec, parses `ThoughtEventV1`, runs `apply_stance_react_pipeline`, attaches `grounding_capsule`.
    - `build_stance_react_context(request)` → the context dict fed to the verb prompt (`user_message`, `stance_inputs`, `association`, `repair_bundle`, `coalition_projection`, `metadata`).
  - `app/settings.py` → flags + timeouts (`stance_react_timeout_sec`, bus channels). No Mind config today.
  - `requirements.txt` → no `httpx`.
- Verb + prompt (in shared `orion/` package, executed by `orion-cortex-exec`):
  - `orion/cognition/verbs/stance_react.yaml`
  - `orion/cognition/prompts/stance_react.j2` (consumes named context keys; explicitly "semantic inference — no keyword lists", "do not invent extra top-level keys").
- Contracts:
  - `orion/schemas/thought.py` → `ThoughtEventV1`, `StanceReactRequestV1` (has open `stance_inputs: dict`), `GroundingCapsuleV1`, `StanceHarnessSliceV1`.
  - `orion/mind/v1.py` + `orion/mind/synthesis_v1.py` → `MindRunRequestV1`, `MindRunResultV1`, `MindHandoffBriefV1`, `ActiveCognitiveFrontierV1`, `MindShadowSynthesisV1`, `MindStanceHandoffV1`.
  - `orion/schemas/mind/artifact.py` → `MindRunArtifactV1` (published today by cortex-orch on `orion:mind:artifact`, persisted to Postgres `mind_runs`, surfaced read-only via `services/orion-hub/scripts/mind_routes.py` `/api/mind/runs*`).
- Reference for "how cortex-orch feeds/consumes Mind": `services/orion-cortex-orch/app/mind_runtime.py`
  - `build_mind_run_request(...)` (snapshot facets: `cognitive_projection`, `recall_bundle`, `autonomy_compact`, `social_compact`, `situation_compact`, `identity_background`).
  - `merge_mind_brief_into_plan_metadata(...)` (the legacy "Mind can drive stance" path — **out of scope** to replicate here).
  - `call_orion_mind_http(...)`, `publish_mind_run_artifact(...)`.
  - Shared, importable builders: `orion/cognition/projection_builder.py`, `orion/cognition/recall_prefetch.py`.

### Live evidence (runtime truth)

Verified on a live brain turn `correlation_id=03d109e3-…`, `mind_run_id=377f5e45-…`, session `orion_journal`:

- Mind ran 3 LLM phases (`semantic_synthesis` → `active_frontier_judge` → `stance_handoff`), wall ~11.3s, `mind_quality=meaningful_synthesis`.
- cortex-orch published `orion:mind:artifact ok=True`; hub UI read it via `GET /api/mind/runs/377f5e45-…` (200).
- Other turns showed `mind_skipped reason=mind_enabled_not_true` — Mind is gated on a metadata flag.
- The unified-turn workers (`orion-thought` 7155, `orion-harness-governor` 7156) had **no** log lines for that correlation id — confirming Mind does not run in the unified path today.

## Goals

1. The unified turn's stance (`ThoughtEventV1.imperative` / `tone` / `stance_harness_slice`) is measurably shaped by Mind's self/attention synthesis when Mind produces `meaningful_synthesis`.
2. Enrichment is **mode-agnostic**: it must help general chat, and must not degrade or mis-frame tooling/coding/agent turns.
3. No contradiction: `stance_react` (with association + grounding) remains authoritative and reconciles Mind's advisory input.
4. The busy pre-fcc path stays robust: fail-open, tight timeout, flag-gated, off by default.
5. Preserve the cheap brain observability (Mind artifact/`mind_runs`/EKG) for unified turns.

## Non-goals

- **Mind driving/replacing the stance** (the legacy `mind_skip_stance_synthesis` behavior). Named as future work "Phase B"; not built here.
- Using Mind's **task-control** output (`task_mode`, `answer_strategy`, `allowed_verbs`, `route_kind`, `mode_binding`, `mode_suggestion`). Explicitly dropped.
- Running Mind at the Hub level or changing `StanceReactRequestV1` / any bus channel semantics.
- New UI/EKG work. (Artifact republish only, reusing existing surfaces.)
- Cold projection rebuild on the turn-critical path.
- Any keyword/phrase/regex detector for user state (forbidden by `conversational-behavior-anti-slop`).

## The choke point (anti-slop compliance)

This is conversational-stance behavior, so per `.cursor/rules/conversational-behavior-anti-slop.mdc`:

- **Choke point:** `build_stance_react_context()` in `services/orion-thought/app/bus_listener.py` (context assembly) → `orion/cognition/prompts/stance_react.j2` (consumption) → `apply_stance_react_pipeline()` in `orion/thought/stance_react.py` (post-parse enforcement).
- **Signals already inferred vs. destroyed:** the `stance_react` LLM already infers posture/frame from the whole turn. Nothing Python-side destroys it today; we are *adding* an advisory prior, not post-processing the output. `mind_coloring` is injected as prompt context and never rewrites `ThoughtEventV1` fields deterministically.
- **Wired through the pipeline:** schema/context change (new `mind_coloring` context key) + prompt change, not a prompt-only patch and not a banned-phrase list.
- **Structural test:** fixture that replays a `MindRunResultV1` through the coloring selector + `build_stance_react_context` and asserts (a) self/frontier fields present, (b) task-control fields absent, (c) prompt renders the block; plus a fail-open test.
- **No trigger lists:** `mind_coloring` selection is a fixed structural projection of Mind's schema fields — not `if "surgery" in user_message`.

## Architecture

```text
Hub turn_orchestrator (mode=orion)
  → RPC orion:thought:request (StanceReactRequestV1)
      orion-thought.run_stance_react:
        1. [flag on] build light Mind snapshot from request (association + grounding + user_message + recall-if-present)
        2. call orion-mind POST /v1/mind/run  (httpx, tight timeout, fail-open)
        3. select self/frontier coloring subset from MindRunResultV1  → mind_coloring
        4. build_stance_react_context(request, mind_coloring=...)     # advisory block
        5. execute stance_react verb  → ThoughtEventV1  (authoritative reconciler)
        6. [flag on] publish MindRunArtifactV1 → orion:mind:artifact  (cheap observability)
  → reply ThoughtEventV1 → Hub → RPC harness → fcc/claude → voice
```

### Component 1 — Mind snapshot builder (light)

New `services/orion-thought/app/mind_enrichment.py`:

- `build_light_mind_request(request: StanceReactRequestV1) -> MindRunRequestV1`
  - `snapshot_inputs.user_text` = `request.user_message`
  - `snapshot_inputs.messages_tail` = from association/stance_inputs if available (else empty)
  - `snapshot_inputs.facets`:
    - `recall_bundle` — only if already present on the request (do **not** trigger a fresh recall fan-out on the critical path in v1; a follow-up may add a bounded prefetch behind its own flag).
    - `identity_background` / `grounding` — from `association`/grounding capsule fields if present on the request.
    - `coalition` — compact from `association.broadcast` (attended nodes, open loops).
  - `policy` = `MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=<flag>, router_profile_id=<flag>)`.
  - Deliberately **no** `cognitive_projection` cold rebuild. If Mind returns low quality due to a thin snapshot, we fail-open (skip enrichment). This honors "no empty-shell cognition": we never inject an empty coloring.

### Component 2 — Mind HTTP client + fail-open

In `mind_enrichment.py`:

- `async def run_mind_for_thought(request, *, settings) -> MindRunResultV1 | None`
  - `httpx.AsyncClient` POST to `f"{ORION_MIND_BASE_URL}/v1/mind/run"`, timeout `ORION_THOUGHT_MIND_TIMEOUT_SEC`.
  - Any error/timeout/oversized body → log `mind_enrichment_failed corr=… reason=…` and return `None` (fail-open).
  - Bound response size (reuse the max-bytes guard pattern from cortex-orch).

### Component 3 — Coloring selector (the USE/DROP split)

`select_mind_coloring(result: MindRunResultV1) -> dict | None`:

- Return `None` (skip) unless `result.ok` **and** `result.brief.mind_quality == "meaningful_synthesis"`. (No empty-shell injection.)
- **USE** (mode-agnostic self/attention):
  - `attention_frontier`: from `result.brief.active_frontier.selected` → list of `{label, summary, score}` (bounded, e.g. top 3).
  - `reflective_themes`: from `shadow_synthesis.attention_focus` (bounded).
  - `curiosity_threads`: from `shadow_synthesis.curiosity_candidate` (bounded).
  - `self_relevance`, `identity_salience`, `juniper_relevance`: from `stance_payload` (self-relational only).
  - `mind_quality`, `mind_run_id`, `snapshot_hash`: provenance for the prompt + telemetry.
- **DROP** (task control — would contradict non-chat turns): `task_mode`, `answer_strategy`, `allowed_verbs`, `route_kind`, `mode_binding`, `mode_suggestion`, `response_priorities`, `response_hazards`, `conversation_frame`. These stay owned by `stance_react`.
- Bound total size (chars) to protect prompt budget; truncate lists deterministically.

### Component 4 — Prompt consumption (advisory, reconciled)

`orion/cognition/prompts/stance_react.j2` gains one block (rendered only when `mind_coloring` present):

```jinja
{% if mind_coloring %}
PRIOR SELF-SIGNAL (advisory — from Orion's Mind; reconcile, do not obey)
- This is Orion's own background self/attention read, computed before this turn's routing.
- Use it to color imperative/tone and the felt layer so Orion sounds like an ongoing presence.
- It is NOT task instruction. If it conflicts with the user_message, association, grounding,
  or the actual task (technical/agent/coding), those WIN. Never let it force chat framing.
- attention_frontier: {{ mind_coloring.attention_frontier }}
- reflective_themes: {{ mind_coloring.reflective_themes }}
- curiosity_threads: {{ mind_coloring.curiosity_threads }}
- self_relevance: {{ mind_coloring.self_relevance }}
- identity_salience: {{ mind_coloring.identity_salience }}
{% endif %}
```

- Reinforces "do not invent extra top-level keys" — the model must still emit valid `ThoughtEventV1`.
- No change to required output fields; enrichment is purely input-side.

### Component 5 — Wiring in `run_stance_react`

- Add optional `mind_coloring` param threaded from `run_stance_react` into `build_stance_react_context`.
- `build_stance_react_context(request, *, mind_coloring=None)` adds `"mind_coloring": mind_coloring` to the context when present.
- `run_stance_react`: when `ORION_THOUGHT_MIND_ENRICHMENT_ENABLED`, call `run_mind_for_thought` → `select_mind_coloring`; pass result through. On `None`, behavior is byte-for-byte identical to today.

### Component 6 — Cheap observability

- When enrichment ran (even if coloring was skipped), optionally publish `MindRunArtifactV1` to `orion:mind:artifact` from `orion-thought` (reuse `orion/schemas/mind/artifact.py`), so the existing state-journaler → `mind_runs` → hub `/api/mind/runs*` EKG lights up for unified turns.
- Gate behind `ORION_THOUGHT_MIND_ARTIFACT_PUBLISH_ENABLED` (default follows the enrichment flag).
- `request_summary_jsonb.mode = "orion"` so unified-turn runs are distinguishable from brain runs.

## Data flow / reconciliation guarantee

The only authoritative producer of `ThoughtEventV1` remains the `stance_react` LLM call. Mind enters strictly as **advisory prompt context** with explicit "reconcile, do not obey; existing inputs win" instructions. There is no deterministic Python step that overwrites stance fields from Mind. This is what structurally prevents contradiction and chat-mode leakage on tooling/coding/agent turns.

## Error handling

- Mind unconfigured / flag off → no-op, today's behavior.
- Mind timeout/error/oversized → fail-open, log `mind_enrichment_failed`, continue.
- Mind low quality (`!= meaningful_synthesis`) → skip coloring (no empty-shell injection); optionally still publish artifact for observability.
- Coloring oversized → deterministic truncation, never drop the whole turn.
- Artifact publish failure → log-and-continue; must never fail the stance stage.

## Config / env (service: `orion-thought`)

Add to `services/orion-thought/.env_example` (+ `settings.py`, `docker-compose.yml`), then run `python scripts/sync_local_env_from_example.py`:

- `ORION_THOUGHT_MIND_ENRICHMENT_ENABLED` (bool, default `false`)
- `ORION_MIND_BASE_URL` (default `http://orion-mind:6611`)
- `ORION_THOUGHT_MIND_TIMEOUT_SEC` (float, default `15.0`)
- `ORION_THOUGHT_MIND_WALL_MS` (int, default `12000`)
- `ORION_THOUGHT_MIND_ROUTER_PROFILE` (str, default `default`)
- `ORION_THOUGHT_MIND_MAX_RESPONSE_BYTES` (int, default matches cortex-orch)
- `ORION_THOUGHT_MIND_ARTIFACT_PUBLISH_ENABLED` (bool, default `false`)
- `ORION_THOUGHT_MIND_COLORING_MAX_ITEMS` (int, default `3`)

`ORION_BUS_URL` stays `redis://100.92.216.81:6379/0` (already set).

## Dependencies

- Add `httpx>=0.27` to `services/orion-thought/requirements.txt` (+ Dockerfile rebuild). Mind is HTTP-only.

## Files likely to touch

- `services/orion-thought/app/mind_enrichment.py` — new: snapshot, HTTP call, coloring selector, artifact publish.
- `services/orion-thought/app/bus_listener.py` — thread `mind_coloring` into `run_stance_react` / `build_stance_react_context`; optional artifact publish.
- `services/orion-thought/app/settings.py` — new flags/URLs/timeouts.
- `services/orion-thought/.env_example` (+ local `.env` via sync script), `docker-compose.yml`, `requirements.txt`, `README.md`.
- `orion/cognition/prompts/stance_react.j2` — advisory `mind_coloring` block.
- `services/orion-thought/tests/` — new tests (selector, fail-open, prompt-render, artifact summary mode).
- `services/orion-thought/evals/` — small enrichment eval (see below) or note the gap.

No changes to `orion/bus/channels.yaml` or `orion/schemas/registry.py` (no new event shape; `MindRunArtifactV1` and `orion:mind:artifact` already registered).

## Testing

### Gate tests (deterministic, fast)

- `test_mind_coloring_selector.py`
  - meaningful synthesis → coloring includes `attention_frontier`, `reflective_themes`, `curiosity_threads`, `self_relevance`, `identity_salience`.
  - coloring **excludes** every DROP key (`task_mode`, `answer_strategy`, `allowed_verbs`, `route_kind`, `mode_binding`, `mode_suggestion`, `response_priorities`, `response_hazards`, `conversation_frame`).
  - non-meaningful / not-ok result → `None`.
  - oversized lists → deterministic truncation to `MAX_ITEMS`.
- `test_stance_context_mind_coloring.py`
  - `build_stance_react_context(request, mind_coloring=...)` puts the block in context; absent when `None` (context byte-identical to baseline).
- `test_stance_prompt_renders_coloring.py`
  - render `stance_react.j2` with/without `mind_coloring`; assert the advisory block + "existing inputs WIN" language appear only when present; no new required output keys introduced.
- `test_mind_enrichment_fail_open.py`
  - Mind client raises/times out → `run_stance_react` returns the same `ThoughtEventV1` as with enrichment disabled (monkeypatched cortex client).
- `test_mind_artifact_mode_tag.py`
  - published `MindRunArtifactV1.request_summary_jsonb.mode == "orion"`.

### Eval (periodic)

- `evals/run_mind_enrichment_eval.py`: replay N fixed transcripts (mix of relational, technical/coding, agent-tool) through the stance stage with enrichment on vs off; assert:
  - technical/coding/agent turns keep their `task_mode`/`conversation_frame` (no chat-mode leakage) — the anti-contradiction guarantee.
  - relational turns show increased self/curiosity signal in `imperative`/`tone` (aliveness).
  - If a full eval harness is infeasible in this pass, ship the fixtures + selector-level assertions and record the harness gap in the PR.

## Rollout

1. Land flag-off. Gate tests green.
2. Rebuild `orion-thought` (httpx). Smoke `GET :7155/health`.
3. Enable `ORION_THOUGHT_MIND_ENRICHMENT_ENABLED=true` + artifact publish on Athena; send unified-turn (`mode=orion`) turns; verify:
   - `orion-thought` logs a Mind call + coloring decision for the correlation id.
   - `orion:mind:artifact` / `mind_runs` shows an `orion`-mode run; EKG surfaces it.
   - Latency delta on the thought stage is within budget; fail-open verified by pointing `ORION_MIND_BASE_URL` at a dead port.
4. Compare stance quality on a small transcript set before/after.

Restart (operator; not run by agent):

```bash
docker compose --env-file .env --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml up -d --build
```

## Acceptance checks

- Flag off → `orion-thought` behavior and `ThoughtEventV1` output unchanged (regression test proves it).
- Flag on, meaningful Mind → `mind_coloring` present in verb context and rendered in prompt; `ThoughtEventV1` still valid.
- Task-control fields never cross from Mind into the unified stance (selector test).
- Technical/coding/agent turns do not get chat-framed (eval).
- Mind down/slow → turn still completes (fail-open test + dead-port smoke).
- Unified-turn Mind runs visible in `mind_runs` with `mode="orion"`.

## Risks / concerns

- **Latency (medium):** an extra ~11s Mind call on the busy pre-fcc path. Mitigation: tight timeout + fail-open + reduced policy; a future flag can run Mind concurrently or prefetch. Measure in rollout step 3.
- **Snapshot thinness (medium):** without cortex-orch's cold projection rebuild, Mind may less often reach `meaningful_synthesis` from the light snapshot, reducing how often enrichment fires. Acceptable for v1 (fail-open to today's behavior); a follow-up can add a bounded recall/projection prefetch behind its own flag.
- **Prompt-budget (low):** coloring bounded + truncated.
- **Double artifact (low):** brain path already publishes artifacts; unified adds its own with `mode="orion"`. Distinguishable, no dedup needed.

## Recommended next patch

Write the implementation plan (writing-plans) as an ordered, test-first slice:
1. Selector + snapshot (pure, unit-tested) → 2. HTTP client + fail-open → 3. context/prompt wiring → 4. settings/env/deps/docker → 5. artifact publish → 6. eval/smoke.
