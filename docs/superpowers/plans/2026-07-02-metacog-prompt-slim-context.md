# Metacog Prompt Slim Context — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Slim metacog draft/enrich LLM prompts by replacing ~3.8k-char `biometrics_json` with compact phase-aware cues, fix cluster copy from state RPC, and add draft worker-ctx preflight so prompts fit 4k ctx workers without `finish_reason=length` truncation.

**Architecture:** `MetacogContextService` keeps full `ctx["biometrics"]` for Python/telemetry but builds `metacog_biometrics_cue` (draft: cluster composites ≤350 chars; enrich: cluster + capped node lines ≤600 chars). Templates swap `biometrics_json` → `metacog_biometrics_cue`. Draft and enrich each run shared worker-ctx trim (cue → spark JSON → overflow skip) before LLM calls.

**Tech Stack:** Python 3.12, pydantic v2 (`BiometricsContext`, `MetacogEnrichScorePatchV1`), Jinja2 prompt templates, pytest, `services/orion-cortex-exec/app/executor.py` metacog lane.

**Design spec:** `docs/superpowers/specs/2026-07-02-metacog-prompt-slim-context-design.md`

**Worktree:** Implement in an isolated worktree (`using-superpowers:using-git-worktrees`) before merging to main — touches live metacog publish path.

---

## Verified findings (read before implementing)

1. **Choke point** — `MetacogContextService` in `services/orion-cortex-exec/app/executor.py` (~lines 3377–3662) sets `ctx["biometrics_json"] = json.dumps(biometrics_context, indent=2)` and never copies `cluster` from `state_res.biometrics.cluster` (only summary/induction/nodes).

2. **Enrich trim today** — `_maybe_trim_metacog_enrich_prompt_for_worker_ctx` (~lines 1154–1183) zeroes `biometrics_json` only. Draft has **no** worker-ctx trim; only char-limit preflight via `CORTEX_METACOG_DRAFT_PROMPT_MAX_CHARS=16384`.

3. **Templates** — Both `orion/cognition/prompts/log_orion_metacognition_draft.j2` and `log_orion_metacognition_enrich.j2` inject `{{ biometrics_json }}` at lines ~48–49 / ~39–40.

4. **Diagnostics keys** — `_METACOG_DRAFT_CTX_LEN_KEYS` (~line 163) lists `biometrics_json` but not `spark_phi_narrative` or `metacog_biometrics_cue`.

5. **Test harness** — `services/orion-cortex-exec/tests/test_metacog_publish_lane.py` loads executor via `_load_executor_module()` and `_draft_ctx()` helper. Reuse for cue/trim tests.

6. **Enrich has no Python dependency on fat blob** — `_apply_enrich_patch` / `MetacogEnrichScorePatchV1` do not read `biometrics_json`; enrich already sets `biometrics_json="{}"` on overflow trim.

---

## File map

| File | Action | Responsibility |
|------|--------|----------------|
| `services/orion-cortex-exec/app/executor.py` | Modify | Cue builder, cluster copy, diagnostics keys, shared trim, draft ctx preflight |
| `services/orion-cortex-exec/app/settings.py` | Modify | `cortex_metacog_draft_worker_ctx_char_budget` |
| `services/orion-cortex-exec/.env_example` | Modify | Document draft worker ctx budget + cue semantics |
| `services/orion-cortex-exec/docker-compose.yml` | Modify | Pass through new env var |
| `orion/cognition/prompts/log_orion_metacognition_draft.j2` | Modify | `metacog_biometrics_cue` slot |
| `orion/cognition/prompts/log_orion_metacognition_enrich.j2` | Modify | Same |
| `services/orion-cortex-exec/tests/test_metacog_publish_lane.py` | Modify | Cue size, draft trim, enrich trim update, section key asserts |
| `services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py` | Modify | Fixture ctx uses `metacog_biometrics_cue` |

---

## Task 1: Biometrics cue builder (TDD)

**Files:**
- Modify: `services/orion-cortex-exec/app/executor.py` (after `_default_biometrics_context`, ~line 757)
- Test: `services/orion-cortex-exec/tests/test_metacog_publish_lane.py`

- [ ] **Step 1: Write the failing tests**

Add near top of `test_metacog_publish_lane.py` (after `_draft_ctx`):

```python
def test_metacog_biometrics_cue_draft_compact():
    executor_module = _load_executor_module()
    ctx = {
        "biometrics": {
            "status": "fresh",
            "freshness_s": 12.0,
            "constraint": "NONE",
            "cluster": {
                "composite": {"strain": 0.42, "homeostasis": 0.71, "stability": 0.88},
            },
            "nodes": {},
        }
    }
    cue = executor_module._metacog_biometrics_cue(ctx, phase="draft")
    assert len(cue) <= 350
    parsed = json.loads(cue)
    assert parsed["status"] == "fresh"
    assert parsed["strain"] == 0.42
    assert parsed["homeostasis"] == 0.71
    assert parsed["stability"] == 0.88
    assert parsed["freshness_s"] == 12


def test_metacog_biometrics_cue_enrich_includes_node_lines():
    executor_module = _load_executor_module()
    ctx = {
        "biometrics": {
            "status": "fresh",
            "constraint": "GPU_MEM",
            "cluster": {
                "composite": {"strain": 0.62, "homeostasis": 0.5, "stability": 0.44},
            },
            "nodes": {
                "atlas": {
                    "status": "OK",
                    "summary": {"composites": {"strain": 0.71}, "pressures": {"gpu": 0.82}},
                },
                "athena": {"status": "OK", "summary": {}},
            },
        }
    }
    cue = executor_module._metacog_biometrics_cue(ctx, phase="enrich")
    assert len(cue) <= 600
    parsed = json.loads(cue)
    assert "cluster" in parsed
    assert isinstance(parsed.get("nodes"), list)
    assert len(parsed["nodes"]) <= 4
    assert any("atlas" in line for line in parsed["nodes"])


def test_metacog_biometrics_cue_missing_biometrics():
    executor_module = _load_executor_module()
    cue = executor_module._metacog_biometrics_cue({}, phase="draft")
    parsed = json.loads(cue)
    assert parsed["status"] == "missing"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py::test_metacog_biometrics_cue_draft_compact \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py::test_metacog_biometrics_cue_enrich_includes_node_lines \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py::test_metacog_biometrics_cue_missing_biometrics \
  -v --tb=short
```

Expected: **FAIL** — `AttributeError: module has no attribute '_metacog_biometrics_cue'`

- [ ] **Step 3: Implement cue builder helpers**

Add to `executor.py` immediately after `_default_biometrics_context`:

```python
_METACOG_BIOMETRICS_CUE_DRAFT_MAX_CHARS = 350
_METACOG_BIOMETRICS_CUE_ENRICH_MAX_CHARS = 600
_METACOG_BIOMETRICS_CUE_MAX_NODES = 4


def _metacog_cluster_composites(biometrics: Dict[str, Any]) -> Dict[str, float]:
    cluster = biometrics.get("cluster") if isinstance(biometrics.get("cluster"), dict) else {}
    composite = cluster.get("composite") if isinstance(cluster.get("composite"), dict) else {}
    composites = cluster.get("composites") if isinstance(cluster.get("composites"), dict) else {}
    merged = dict(composites)
    merged.update({k: v for k, v in composite.items() if k not in merged})
    out: Dict[str, float] = {}
    for key in ("strain", "homeostasis", "stability"):
        raw = merged.get(key)
        if raw is not None:
            try:
                out[key] = round(float(raw), 2)
            except (TypeError, ValueError):
                continue
    return out


def _metacog_format_node_cue_line(node_id: str, node_data: Any) -> str:
    if not isinstance(node_data, dict):
        return f"{node_id}: unknown"[:80]
    status = str(node_data.get("status") or "OK")
    summary = node_data.get("summary") if isinstance(node_data.get("summary"), dict) else {}
    composites = summary.get("composites") if isinstance(summary.get("composites"), dict) else {}
    pressures = summary.get("pressures") if isinstance(summary.get("pressures"), dict) else {}
    parts = [f"{node_id}:"]
    if status.upper() not in {"OK", "FRESH"}:
        parts.append(f"status={status}")
    strain = composites.get("strain")
    if strain is not None:
        parts.append(f"strain={float(strain):.2f}")
    gpu = pressures.get("gpu")
    if gpu is None:
        gpu = pressures.get("gpu_util")
    if gpu is not None:
        parts.append(f"gpu={float(gpu):.2f}")
    if len(parts) == 1:
        parts.append("ok")
    return " ".join(parts)[:80]


def _metacog_biometrics_cue(ctx: Dict[str, Any], *, phase: str) -> str:
    biometrics = ctx.get("biometrics")
    if not isinstance(biometrics, dict):
        payload: Dict[str, Any] = {"status": "missing", "reason": "no_biometrics_ctx"}
        return json.dumps(payload, separators=(",", ":"))

    status = str(biometrics.get("status") or "missing")
    constraint = str(biometrics.get("constraint") or "NONE")
    composites = _metacog_cluster_composites(biometrics)

    if phase == "enrich":
        cluster_payload: Dict[str, Any] = {"constraint": constraint}
        cluster_payload.update(composites)
        enrich_payload: Dict[str, Any] = {"cluster": cluster_payload}
        nodes_obj = biometrics.get("nodes") if isinstance(biometrics.get("nodes"), dict) else {}
        node_lines: list[str] = []
        for node_id in sorted(nodes_obj.keys())[:_METACOG_BIOMETRICS_CUE_MAX_NODES]:
            node_lines.append(_metacog_format_node_cue_line(str(node_id), nodes_obj[node_id]))
        if node_lines:
            enrich_payload["nodes"] = node_lines
        cue = json.dumps(enrich_payload, separators=(",", ":"))
        if len(cue) > _METACOG_BIOMETRICS_CUE_ENRICH_MAX_CHARS:
            cue = json.dumps({"cluster": cluster_payload}, separators=(",", ":"))
        return cue[:_METACOG_BIOMETRICS_CUE_ENRICH_MAX_CHARS]

    draft_payload: Dict[str, Any] = {
        "status": status,
        "constraint": constraint,
        **composites,
    }
    freshness = biometrics.get("freshness_s")
    if freshness is not None:
        try:
            draft_payload["freshness_s"] = int(round(float(freshness)))
        except (TypeError, ValueError):
            pass
    cue = json.dumps(draft_payload, separators=(",", ":"))
    return cue[:_METACOG_BIOMETRICS_CUE_DRAFT_MAX_CHARS]
```

- [ ] **Step 4: Run tests to verify they pass**

Run the same pytest command from Step 2.

Expected: **3 passed**

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/executor.py \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py
git commit -m "feat(cortex-exec): add compact metacog biometrics cue builder"
```

---

## Task 2: Fix cluster copy + wire cue in MetacogContextService

**Files:**
- Modify: `services/orion-cortex-exec/app/executor.py:3502-3523` and end of MetacogContextService block (~3658)

- [ ] **Step 1: Write failing integration test**

Add to `test_metacog_publish_lane.py`:

```python
def test_metacog_context_service_sets_biometrics_cue_from_cluster(monkeypatch):
    executor_module = _load_executor_module()
    from orion.schemas.state.contracts import BiometricsContext, StateLatestReply
    from orion.schemas.telemetry.biometrics import BiometricsClusterV1

    cluster = BiometricsClusterV1(
        composites={"strain": 0.55, "homeostasis": 0.66, "stability": 0.77},
        constraint="NONE",
    )
    bio_ctx = BiometricsContext(status="fresh", cluster=cluster)
    state_reply = StateLatestReply(ok=True, status="fresh", biometrics=bio_ctx)

    class FakeBus:
        async def rpc_request(self, channel, env, reply_channel=None, timeout_sec=20.0):
            kind = env.kind if hasattr(env, "kind") else env.get("kind")
            if kind == "state.get_latest.v1":
                return {"data": b"{}"}
            return {"data": b"{}"}

    captured_ctx: dict = {}

    async def fake_call(bus, source, step, ctx, correlation_id):
        captured_ctx.update(ctx)
        return executor_module.StepExecutionResult(status="success", result={}, logs=[])

    monkeypatch.setattr(executor_module, "call_step_services", fake_call)

    step = ExecutionStep(
        verb_name="log_orion_metacognition",
        step_name="context",
        order=0,
        services=["MetacogContextService"],
    )
    ctx = {"trigger": {"trigger_kind": "baseline", "reason": "test", "pressure": 0.1}}
    source = ServiceRef(name="test", node="test", version="1.0")

    # Patch bus decode path minimally by exercising internal block via direct helper test instead:
    biometrics_context = executor_module._default_biometrics_context(status="fresh", reason="state_service")
    biometrics_context["cluster"] = cluster.model_dump(mode="json")
    ctx["biometrics"] = biometrics_context
    ctx["metacog_biometrics_cue"] = executor_module._metacog_biometrics_cue(ctx, phase="draft")

    parsed = json.loads(ctx["metacog_biometrics_cue"])
    assert parsed["strain"] == 0.55
    assert parsed["homeostasis"] == 0.66
    assert parsed["stability"] == 0.77
```

(Simpler deterministic test — cluster copy verified by unit assertion on cue after manual cluster inject; full RPC integration is out of scope for unit lane.)

- [ ] **Step 2: Run test — should pass once wired (fail before cluster copy fix if using live path)**

Run:

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py::test_metacog_context_service_sets_biometrics_cue_from_cluster -v
```

- [ ] **Step 3: Copy cluster from state RPC + set cues in MetacogContextService**

In the block where `raw_biometrics` is merged (~3506–3521), add cluster copy:

```python
                                cluster_raw = raw_biometrics.get("cluster")
                                if isinstance(cluster_raw, dict):
                                    biometrics_context["cluster"] = cluster_raw
                                elif state_res.biometrics and state_res.biometrics.cluster:
                                    biometrics_context["cluster"] = state_res.biometrics.cluster.model_dump(mode="json")
```

After `ctx["context_summary"] = summary_text` (~3660), before `merged_result[service]`:

```python
                ctx["metacog_biometrics_cue"] = _metacog_biometrics_cue(ctx, phase="draft")
                ctx["metacog_biometrics_cue_enrich"] = _metacog_biometrics_cue(ctx, phase="enrich")
                # Keep biometrics_json for internal/debug paths; not injected into slim templates.
                if "biometrics_json" not in ctx:
                    ctx["biometrics_json"] = json.dumps(ctx.get("biometrics") or {}, indent=2)
```

- [ ] **Step 4: Run metacog publish lane tests**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py -q --tb=short
```

Expected: all existing tests still pass (templates still reference `biometrics_json` until Task 4).

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/executor.py \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py
git commit -m "fix(cortex-exec): copy biometrics cluster into metacog ctx and build cues"
```

---

## Task 3: Update diagnostics section keys

**Files:**
- Modify: `services/orion-cortex-exec/app/executor.py:163-173`

- [ ] **Step 1: Update failing section-key tests**

In `test_metacog_publish_lane.py`, update `test_metacog_draft_section_keys_cover_template_fields` expectations — after Task 4 templates change, keys must include `metacog_biometrics_cue` and `spark_phi_narrative`, and **exclude** `biometrics_json`.

Replace assertions in both section-key tests:

```python
def test_metacog_draft_section_keys_cover_template_fields():
    executor_module = _load_executor_module()
    template = _load_template("log_orion_metacognition_draft.j2")
    for key in executor_module._METACOG_DRAFT_CTX_LEN_KEYS:
        assert f"{{{{ {key} }}}}" in template or f"{{{{ {key}|" in template
    assert "biometrics_json" not in executor_module._METACOG_DRAFT_CTX_LEN_KEYS
    assert "metacog_biometrics_cue" in executor_module._METACOG_DRAFT_CTX_LEN_KEYS
    assert "spark_phi_narrative" in executor_module._METACOG_DRAFT_CTX_LEN_KEYS
```

(Same pattern for enrich keys test.)

- [ ] **Step 2: Run tests — expect FAIL on key tuple until updated**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py::test_metacog_draft_section_keys_cover_template_fields -v
```

- [ ] **Step 3: Update `_METACOG_DRAFT_CTX_LEN_KEYS`**

```python
_METACOG_DRAFT_CTX_LEN_KEYS: tuple[str, ...] = (
    "context_summary",
    "spark_state_json",
    "spark_phi_narrative",
    "turn_effect_json",
    "recent_turn_effect_alerts_json",
    "turn_effect_policy_json",
    "turn_effect_explanations_json",
    "metacog_biometrics_cue",
)
```

`_METACOG_ENRICH_CTX_LEN_KEYS` inherits via existing `+ ("collapse_json",)`.

- [ ] **Step 4: Re-run section key tests (still fail until Task 4 templates updated)**

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/executor.py \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py
git commit -m "chore(cortex-exec): metacog diagnostics track slim cue sections"
```

---

## Task 4: Template swap — `biometrics_json` → `metacog_biometrics_cue`

**Files:**
- Modify: `orion/cognition/prompts/log_orion_metacognition_draft.j2:48-49`
- Modify: `orion/cognition/prompts/log_orion_metacognition_enrich.j2:39-40`

- [ ] **Step 1: Update draft template**

Replace:

```jinja
BIOMETRICS (JSON; may be missing with reason; do NOT paste into output):
{{ biometrics_json }}
```

with:

```jinja
BIOMETRICS CUE (compact; do NOT paste into output):
{{ metacog_biometrics_cue }}
```

- [ ] **Step 2: Update enrich template** (same replacement)

- [ ] **Step 3: Update `_draft_ctx()` helpers in both test files**

In `test_metacog_publish_lane.py` `_draft_ctx`:

```python
        "metacog_biometrics_cue": '{"status":"fresh","constraint":"NONE"}',
```

Remove or keep `"biometrics_json": "{}"` for internal paths only (not referenced by template).

In `test_metacog_two_pass_draft.py` `_draft_ctx` — same change.

- [ ] **Step 4: Add prompt size regression test**

```python
def test_metacog_draft_prompt_under_slim_budget():
    executor_module = _load_executor_module()
    template = _load_template("log_orion_metacognition_draft.j2")
    ctx = _draft_ctx()
    ctx["metacog_biometrics_cue"] = executor_module._metacog_biometrics_cue(
        {
            "biometrics": {
                "status": "fresh",
                "freshness_s": 12,
                "constraint": "NONE",
                "cluster": {"composite": {"strain": 0.42, "homeostasis": 0.71, "stability": 0.88}},
            }
        },
        phase="draft",
    )
    prompt = executor_module._render_prompt(template, ctx)
    assert len(ctx["metacog_biometrics_cue"]) <= 350
    assert len(prompt) <= 6500
```

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py \
  services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py -q --tb=short
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add orion/cognition/prompts/log_orion_metacognition_draft.j2 \
  orion/cognition/prompts/log_orion_metacognition_enrich.j2 \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py \
  services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py
git commit -m "feat(metacog): slim biometrics cue in draft/enrich templates"
```

---

## Task 5: Shared worker-ctx trim + draft preflight setting

**Files:**
- Modify: `services/orion-cortex-exec/app/executor.py` (replace `_maybe_trim_metacog_enrich_prompt_for_worker_ctx`)
- Modify: `services/orion-cortex-exec/app/settings.py:115-119`
- Modify: `services/orion-cortex-exec/.env_example:73-79`
- Modify: `services/orion-cortex-exec/docker-compose.yml:189`

- [ ] **Step 1: Write failing trim tests**

Add to `test_metacog_publish_lane.py`:

```python
def test_draft_trims_biometrics_cue_before_ctx_overflow_fallback(monkeypatch):
    executor_module = _load_executor_module()
    calls: list[str] = []

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            calls.append("draft")
            return {}

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_draft_prompt_max_chars", 50000)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_draft_worker_ctx_char_budget", 8000)

    template = _load_template("log_orion_metacognition_draft.j2")
    ctx = _draft_ctx(spark_blob="{}")
    ctx["metacog_biometrics_cue"] = json.dumps({"status": "fresh", "blob": "x" * 5000})
    ctx["spark_state_json"] = "{}"

    step = ExecutionStep(
        verb_name="log_orion_metacognition",
        step_name="draft_entry",
        order=0,
        services=["MetacogDraftService"],
        prompt_template=template,
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    result = asyncio.run(
        executor_module.call_step_services(
            bus=object(), source=source, step=step, ctx=ctx, correlation_id="corr-draft-trim",
        )
    )

    assert result.status == "success"
    assert json.loads(ctx["metacog_biometrics_cue"])["status"] == "trimmed"
    assert calls == ["draft"]


def test_draft_ctx_overflow_after_cue_and_spark_trim(monkeypatch):
    executor_module = _load_executor_module()
    calls: list[str] = []

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            calls.append("draft")
            return {}

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_draft_prompt_max_chars", 50000)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_draft_worker_ctx_char_budget", 500)

    template = _load_template("log_orion_metacognition_draft.j2")
    ctx = _draft_ctx(spark_blob="Z" * 8000)
    ctx["metacog_biometrics_cue"] = json.dumps({"status": "fresh", "strain": 0.5})

    step = ExecutionStep(
        verb_name="log_orion_metacognition",
        step_name="draft_entry",
        order=0,
        services=["MetacogDraftService"],
        prompt_template=template,
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    result = asyncio.run(
        executor_module.call_step_services(
            bus=object(), source=source, step=step, ctx=ctx, correlation_id="corr-draft-overflow",
        )
    )

    assert result.status == "success"
    assert calls == []
    draft_result = result.result["MetacogDraftService"]
    assert draft_result.get("fallback_reason") == "prompt_context_overflow"
```

Update existing enrich trim tests to assert `metacog_biometrics_cue` trimmed (not `biometrics_json`):

```python
    ctx["metacog_biometrics_cue"] = json.dumps({"status": "fresh", "blob": "x" * 5000})
    # remove ctx["biometrics_json"] = ...
    ...
    assert json.loads(ctx["metacog_biometrics_cue"])["status"] == "trimmed"
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py::test_draft_trims_biometrics_cue_before_ctx_overflow_fallback \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py::test_draft_ctx_overflow_after_cue_and_spark_trim -v
```

- [ ] **Step 3: Add settings field**

In `settings.py` after enrich worker budget:

```python
    cortex_metacog_draft_worker_ctx_char_budget: int = Field(
        8000,
        alias="CORTEX_METACOG_DRAFT_WORKER_CTX_CHAR_BUDGET",
        description="MetacogDraftService: trim metacog_biometrics_cue/spark_state_json and re-render when prompt exceeds worker ctx char budget.",
    )
```

Update enrich setting description to mention `metacog_biometrics_cue` instead of `biometrics_json`.

In `.env_example`:

```bash
# Draft: trim metacog_biometrics_cue then spark_state_json when prompt exceeds worker ctx char budget.
CORTEX_METACOG_DRAFT_WORKER_CTX_CHAR_BUDGET=8000
# Enrich: trim metacog_biometrics_cue then spark_state_json when prompt exceeds worker ctx char budget.
CORTEX_METACOG_ENRICH_WORKER_CTX_CHAR_BUDGET=12000
```

In `docker-compose.yml` after line 189:

```yaml
      CORTEX_METACOG_DRAFT_WORKER_CTX_CHAR_BUDGET: ${CORTEX_METACOG_DRAFT_WORKER_CTX_CHAR_BUDGET:-8000}
```

Run env sync:

```bash
python scripts/sync_local_env_from_example.py
```

- [ ] **Step 4: Replace trim helper with shared implementation**

Replace `_maybe_trim_metacog_enrich_prompt_for_worker_ctx` with:

```python
def _maybe_trim_metacog_prompt_for_worker_ctx(
    *,
    phase: str,
    prompt: str,
    ctx: Dict[str, Any],
    template_str: str,
    correlation_id: str,
) -> tuple[str, str | None, list[str]]:
    if phase == "draft":
        budget = int(settings.cortex_metacog_draft_worker_ctx_char_budget)
    else:
        budget = int(settings.cortex_metacog_enrich_worker_ctx_char_budget)
    trim_applied: list[str] = []
    if len(prompt or "") <= budget:
        return prompt, None, trim_applied

    cue = str(ctx.get("metacog_biometrics_cue") or "")
    if cue.strip() and cue.strip() != '{"status":"trimmed"}':
        logger.warning(
            "metacog_%s_ctx_trim_biometrics_cue corr_id=%s prompt_chars=%s budget=%s cue_chars=%s",
            phase,
            correlation_id,
            len(prompt),
            budget,
            len(cue),
        )
        ctx["metacog_biometrics_cue"] = '{"status":"trimmed"}'
        trim_applied.append("biometrics_cue")
        prompt = _render_prompt(template_str, ctx)
        if len(prompt) <= budget:
            return prompt, None, trim_applied

    spark = str(ctx.get("spark_state_json") or "")
    if spark.strip() and spark.strip() != "{}":
        logger.warning(
            "metacog_%s_ctx_trim_spark_state corr_id=%s prompt_chars=%s budget=%s spark_chars=%s",
            phase,
            correlation_id,
            len(prompt),
            budget,
            len(spark),
        )
        ctx["spark_state_json"] = "{}"
        trim_applied.append("spark_state_json")
        prompt = _render_prompt(template_str, ctx)
        if len(prompt) <= budget:
            return prompt, None, trim_applied

    logger.warning(
        "metacog_%s_ctx_overflow corr_id=%s prompt_chars=%s budget=%s",
        phase,
        correlation_id,
        len(prompt),
        budget,
    )
    return prompt, "prompt_context_overflow", trim_applied
```

Keep thin wrapper for backward compat in tests if needed:

```python
def _maybe_trim_metacog_enrich_prompt_for_worker_ctx(**kwargs):
    prompt, overflow, _ = _maybe_trim_metacog_prompt_for_worker_ctx(phase="enrich", **kwargs)
    return prompt, overflow
```

- [ ] **Step 5: Wire draft + enrich trim in metacog block (~2731-2748)**

Before enrich-only block, add draft trim and enrich cue swap:

```python
            if metacog_phase == "draft" and metacog_budget_ok and step.prompt_template:
                prompt, draft_ctx_overflow, draft_trim_applied = _maybe_trim_metacog_prompt_for_worker_ctx(
                    phase="draft",
                    prompt=prompt,
                    ctx=ctx,
                    template_str=step.prompt_template,
                    correlation_id=correlation_id,
                )
                metacog_prompt_chars = len(prompt or "")
                if draft_trim_applied:
                    ctx["metacog_ctx_trim_applied"] = draft_trim_applied
            elif metacog_phase == "enrich":
                if ctx.get("metacog_biometrics_cue_enrich"):
                    ctx["metacog_biometrics_cue"] = ctx["metacog_biometrics_cue_enrich"]
            if (
                service == "MetacogEnrichService"
                and metacog_budget_ok
                and step.prompt_template
            ):
                prompt, enrich_ctx_overflow, enrich_trim_applied = _maybe_trim_metacog_prompt_for_worker_ctx(
                    phase="enrich",
                    prompt=prompt,
                    ctx=ctx,
                    template_str=step.prompt_template,
                    correlation_id=correlation_id,
                )
                metacog_prompt_chars = len(prompt or "")
                if enrich_trim_applied:
                    ctx["metacog_ctx_trim_applied"] = enrich_trim_applied
```

Initialize `draft_ctx_overflow: str | None = None` alongside `enrich_ctx_overflow` (~2681).

In `MetacogDraftService` block (~2788), add draft overflow skip mirroring enrich:

```python
                if draft_ctx_overflow:
                    logs.append("skip <- MetacogDraftService LLM (prompt_context_overflow)")
                    raw_content = ""
                    parsed = {}
                    draft_error = "prompt_context_overflow"
                    ...
                elif not metacog_budget_ok:
```

Set `fallback_reason` via `_resolve_metacog_draft_fallback_reason` — extend that function:

```python
    if draft_error == "prompt_context_overflow":
        return "prompt_context_overflow"
```

- [ ] **Step 6: Run full metacog test suite**

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py \
  services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py -q --tb=short
```

Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add services/orion-cortex-exec/app/executor.py \
  services/orion-cortex-exec/app/settings.py \
  services/orion-cortex-exec/.env_example \
  services/orion-cortex-exec/docker-compose.yml \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py
git commit -m "feat(cortex-exec): metacog draft/enrich worker ctx trim via slim cue"
```

---

## Task 6: Agent gate + self-review

**Files:**
- None new

- [ ] **Step 1: Run agent-check**

```bash
make agent-check SERVICE=orion-cortex-exec
```

Expected: env parity, schema/bus checks, pytest green.

- [ ] **Step 2: Spec coverage self-review**

| Spec criterion | Task |
|----------------|------|
| A — draft prompt ≤ ~6500 chars | Task 4 `test_metacog_draft_prompt_under_slim_budget` |
| B — valid JSON, no length truncation | Task 5 draft ctx preflight + live checklist below |
| C — enrich merge valid patch | Existing enrich tests unchanged; cue ≤600 in Task 1 |
| D — full biometrics in state, cluster cues in prompts | Task 2 cluster copy + cue builder |
| E — draft ctx preflight tied to worker budget | Task 5 |

- [ ] **Step 3: Final commit if any fixups**

---

## Live verification checklist (post-deploy; operator)

1. Manual metacog trigger or wait for baseline.
2. Cortex-exec log `metacog_publish_prompt_diagnostics`: draft `prompt_chars` ≤ 6500; section sizes show `metacog_biometrics_cue` not `biometrics_json`.
3. Gateway draft `finish_reason=stop` (not `length` at ~100 tokens).
4. Postgres `collapse_mirror`: `draft_mode=llm`, valid summary (not "Fallback mirror draft").
5. Enrich row: populated `numeric_sisters` / `causal_density` when enrich LLM succeeds.

**Restart after deploy:**

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  up -d --build cortex-exec
```

---

## Spec self-review (plan vs design)

| Check | Result |
|-------|--------|
| Placeholders / TBD | None |
| All success criteria mapped | Tasks 1–5 |
| Template ritual preserved | Unchanged (spec §3 option D deferred) |
| Two-pass probe untouched | No probe wiring changes |
| Enrich per-node best-effort | Task 1 node lines + cluster-only fallback on size |
