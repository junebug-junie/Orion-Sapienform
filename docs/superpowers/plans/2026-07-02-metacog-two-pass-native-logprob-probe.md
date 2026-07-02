# Metacog Two-Pass Native Logprob Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore reliable metacog collapse-mirror JSON drafts while keeping native-aligned `llm_uncertainty` telemetry by splitting draft into pass 1 (chat + `json_object`) and pass 2 (short native completion probe).

**Architecture:** Pass 1 uses existing `MetacogDraftService` chat path with `response_format: json_object` and **no** logprob flags. Pass 2 runs only after successful parse/validate, using a minimal probe prompt built from pass-1 patch fields. Gateway `_should_use_native_llamacpp_completion` gains a guard: never detour when `response_format` is set. Enrich stays single-pass chat; oversized enrich prompts trim `biometrics_json` once, then fail with `prompt_context_overflow` before the LLM call.

**Tech Stack:** Python 3.12, FastAPI services (`orion-llm-gateway`, `orion-cortex-exec`), pydantic v2 (`MetacogDraftTextPatchV1`), pytest, existing `LLMGatewayClient` bus RPC.

**Design spec:** `docs/superpowers/specs/2026-07-02-metacog-two-pass-native-logprob-probe-design.md`

**Worktree:** Implement in an isolated worktree (`using-superpowers:using-git-worktrees`) before merging to main — this touches live metacog publish path.

---

## Verified findings (read before implementing)

1. **Root cause choke point** — `services/orion-llm-gateway/app/llm_backend.py::_should_use_native_llamacpp_completion` (lines 717–725) returns true when `return_logprobs` + `logprob_probe_mode=native_completion` + gateway flags are on. It does **not** check `response_format`. `run_llm_chat` (line 1379) routes to native completion before chat.

2. **Cortex-exec draft wiring** — `services/orion-cortex-exec/app/executor.py` MetacogDraftService block (lines 2685–2773) adds logprob options to the **same** request as `response_format: json_object`. That combination triggered the native detour regression.

3. **Uncertainty attachment** — `attach_llm_uncertainty_to_collapse_payload` in `orion/schemas/collapse_mirror.py` already accepts `source=llamacpp_native_completion`. Draft block already reads `llm_res.meta["llm_uncertainty"]` (lines 2766–2773).

4. **Firebreak** — `test_firebreak.py` and `test_metacog_publish_lane.py::test_firebreak_skip_includes_fallback_reason_and_diagnostics` assert baseline + `draft_mode=fallback` skips publish. Must remain unchanged.

5. **Test harness pattern** — `services/orion-cortex-exec/tests/test_metacog_publish_lane.py` loads executor via `_load_executor_module()` and monkeypatches `LLMGatewayClient`. Reuse this for two-pass tests.

6. **Metacog profile ctx** — `config/llm_profiles.yaml` `llama3-8b-instruct-q4km-atlas-metacog` has `ctx_size: 4096`. Cortex-exec does not load profiles today; enrich preflight uses a new char-budget env (default 12000 ≈ 3000 prompt tokens reserved from 4k ctx).

---

## File map

| File | Action | Responsibility |
|------|--------|----------------|
| `services/orion-llm-gateway/app/llm_backend.py` | Modify | Block native detour when `response_format` present |
| `services/orion-llm-gateway/tests/test_llm_backend.py` | Modify | Regression test for guard |
| `services/orion-cortex-exec/app/executor.py` | Modify | Two-pass draft, probe helpers, enrich ctx trim |
| `services/orion-cortex-exec/app/settings.py` | Modify | Optional probe kill-switch + enrich ctx char budget |
| `services/orion-cortex-exec/.env_example` | Modify | Document two-pass semantics |
| `services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py` | Create | Pass 1/2 option split, probe failure non-fatal, uncertainty attach |
| `services/orion-cortex-exec/tests/test_metacog_publish_lane.py` | Modify | Enrich ctx-overflow trim test |
| `services/orion-cortex-exec/tests/test_collapse_llm_uncertainty_telemetry.py` | Modify | Optional native-source fixture assert |

---

## Task 1: Gateway — `response_format` blocks native completion detour

**Files:**
- Modify: `services/orion-llm-gateway/app/llm_backend.py:717-725`
- Modify: `services/orion-llm-gateway/tests/test_llm_backend.py`

- [ ] **Step 1: Write the failing test**

Add to `services/orion-llm-gateway/tests/test_llm_backend.py` inside `TestLLMBackend`:

```python
@patch("app.llm_backend._execute_llamacpp_native_completion")
@patch("app.llm_backend._execute_openai_chat")
def test_response_format_blocks_native_completion_detour(self, mock_openai, mock_native):
    mock_openai.return_value = {"text": "{}", "spark_meta": {}, "raw": {}}
    original = settings.llm_route_table_json
    try:
        settings.llm_route_table_json = (
            '{"chat":{"url":"http://llamacpp:8080","served_by":"atlas","backend":"llamacpp"}}'
        )
        _load_route_targets.cache_clear()
        with patch.object(settings, "llm_logprob_summary_enabled", True), patch.object(
            settings, "llm_logprob_native_completion_enabled", True
        ):
            run_llm_chat(
                ChatBody(
                    messages=[ChatMessage(role="user", content="hi")],
                    options={
                        "return_logprobs": True,
                        "logprob_probe_mode": "native_completion",
                        "response_format": {"type": "json_object"},
                    },
                )
            )
        mock_openai.assert_called_once()
        mock_native.assert_not_called()
    finally:
        settings.llm_route_table_json = original
        _load_route_targets.cache_clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=services/orion-llm-gateway ./orion_dev/bin/python -m pytest \
  services/orion-llm-gateway/tests/test_llm_backend.py::TestLLMBackend::test_response_format_blocks_native_completion_detour -v
```

Expected: **FAIL** — `mock_native.assert_not_called()` fails because native path is chosen today.

- [ ] **Step 3: Implement minimal guard**

In `services/orion-llm-gateway/app/llm_backend.py`, replace `_should_use_native_llamacpp_completion`:

```python
def _should_use_native_llamacpp_completion(body: ChatBody, backend: str) -> bool:
    opts = body.options or {}
    if opts.get("response_format"):
        return False
    return (
        backend == "llamacpp"
        and bool(opts.get("return_logprobs"))
        and bool(getattr(settings, "llm_logprob_summary_enabled", False))
        and bool(getattr(settings, "llm_logprob_native_completion_enabled", False))
        and str(opts.get("logprob_probe_mode") or "").strip().lower() == "native_completion"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=services/orion-llm-gateway ./orion_dev/bin/python -m pytest \
  services/orion-llm-gateway/tests/test_llm_backend.py -q --tb=short
```

Expected: **PASS** (all tests in file, including new guard and existing `test_run_llm_chat_routes_native_completion_when_opted_in`).

- [ ] **Step 5: Commit**

```bash
git add services/orion-llm-gateway/app/llm_backend.py \
  services/orion-llm-gateway/tests/test_llm_backend.py
git commit -m "fix(llm-gateway): keep json_object requests on chat completions path"
```

---

## Task 2: Cortex-exec — probe message helpers

**Files:**
- Modify: `services/orion-cortex-exec/app/executor.py` (near `_metacog_messages`, ~line 759)
- Create: `services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py`

- [ ] **Step 1: Write the failing tests**

Create `services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py`:

```python
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from orion.schemas.metacog_patches import MetacogDraftTextPatchV1

ROOT = Path(__file__).resolve().parents[3]
EXEC_ROOT = Path(__file__).resolve().parents[1]


def _load_executor_module():
    app_dir = EXEC_ROOT / "app"
    executor_path = app_dir / "executor.py"
    package_name = "orion_cortex_exec_two_pass"
    app_package_name = f"{package_name}.app"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(app_dir.parent)]
        sys.modules[package_name] = pkg
    if app_package_name not in sys.modules:
        pkg = types.ModuleType(app_package_name)
        pkg.__path__ = [str(app_dir)]
        sys.modules[app_package_name] = pkg
    spec = importlib.util.spec_from_file_location(f"{app_package_name}.executor", executor_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_metacog_uncertainty_probe_messages_use_patch_fields():
    executor_module = _load_executor_module()
    patch = MetacogDraftTextPatchV1(
        type="flow",
        emergent_entity="Atlas",
        summary="steady focus",
        resonance_signature="flow: Atlas | Δ:low | →maintain",
    )
    messages = executor_module._metacog_uncertainty_probe_messages(patch)
    assert messages[0]["role"] == "system"
    assert "resonance_signature" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "type=flow" in messages[1]["content"]
    assert "entity=Atlas" in messages[1]["content"]
    assert "reference_signature=" in messages[1]["content"]


def test_metacog_uncertainty_probe_messages_truncate_long_fields():
    executor_module = _load_executor_module()
    patch = MetacogDraftTextPatchV1(summary="x" * 800)
    messages = executor_module._metacog_uncertainty_probe_messages(patch)
    assert len(messages[0]["content"]) <= 512
    assert len(messages[1]["content"]) <= 512
    assert messages[1]["content"].endswith("...")


def test_should_run_metacog_uncertainty_probe_respects_settings(monkeypatch):
    executor_module = _load_executor_module()
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_return_logprobs", False)
    assert executor_module._should_run_metacog_uncertainty_probe() is False
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_return_logprobs", True)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_uncertainty_probe_enabled", False)
    assert executor_module._should_run_metacog_uncertainty_probe() is False
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_uncertainty_probe_enabled", True)
    assert executor_module._should_run_metacog_uncertainty_probe() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py -v --tb=short
```

Expected: **FAIL** — `AttributeError: module has no attribute '_metacog_uncertainty_probe_messages'`.

- [ ] **Step 3: Implement helpers**

Add to `services/orion-cortex-exec/app/executor.py` immediately after `_metacog_messages`:

```python
_METACOG_UNCERTAINTY_PROBE_MAX_CHARS = 512


def _truncate_metacog_probe_text(text: str, *, limit: int = _METACOG_UNCERTAINTY_PROBE_MAX_CHARS) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _metacog_uncertainty_probe_messages(patch: MetacogDraftTextPatchV1) -> List[Dict[str, Any]]:
    typ = _truncate_metacog_probe_text(patch.type or "unknown")
    entity = _truncate_metacog_probe_text(patch.emergent_entity or "unknown")
    summary = _truncate_metacog_probe_text(patch.summary or "")
    sig_hint = _truncate_metacog_probe_text(patch.resonance_signature or "")
    system = _truncate_metacog_probe_text(
        "You are a metacognition uncertainty probe. "
        "Output exactly one line: the resonance_signature only. "
        "No JSON, no markdown, no preamble."
    )
    user_parts = [f"type={typ} entity={entity} summary={summary}"]
    if sig_hint:
        user_parts.append(f"reference_signature={sig_hint}")
    user_parts.append('Format: "<type>: <entity> | Δ:<delta> | →<intent>"')
    user = _truncate_metacog_probe_text(" ".join(user_parts))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _should_run_metacog_uncertainty_probe() -> bool:
    if not getattr(settings, "cortex_metacog_return_logprobs", False):
        return False
    return bool(getattr(settings, "cortex_metacog_uncertainty_probe_enabled", True))
```

Ensure `MetacogDraftTextPatchV1` is already imported at top of `executor.py` (it is).

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py::test_metacog_uncertainty_probe_messages_use_patch_fields \
  services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py::test_metacog_uncertainty_probe_messages_truncate_long_fields \
  services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py::test_should_run_metacog_uncertainty_probe_respects_settings \
  -v
```

Expected: **PASS**

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/executor.py \
  services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py
git commit -m "feat(cortex-exec): add metacog uncertainty probe message helpers"
```

---

## Task 3: Cortex-exec — two-pass MetacogDraftService

**Files:**
- Modify: `services/orion-cortex-exec/app/executor.py:2685-2773`
- Modify: `services/orion-cortex-exec/app/settings.py`
- Modify: `services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py`

- [ ] **Step 1: Add settings field**

In `services/orion-cortex-exec/app/settings.py` after `cortex_metacog_logprob_probe_mode`:

```python
cortex_metacog_uncertainty_probe_enabled: bool = Field(
    True,
    alias="CORTEX_METACOG_UNCERTAINTY_PROBE_ENABLED",
    description="When CORTEX_METACOG_RETURN_LOGPROBS: run pass-2 native probe after successful draft parse.",
)
```

- [ ] **Step 2: Write failing integration tests**

Append to `services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py`:

```python
import asyncio
import json
from unittest.mock import MagicMock

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import ExecutionStep

_VALID_DRAFT_JSON = json.dumps(
    {
        "type": "flow",
        "emergent_entity": "Atlas",
        "summary": "steady",
        "mantra": "breathe",
        "resonance_signature": "flow: Atlas | Δ:low | →maintain",
    }
)


def _load_template(name: str) -> str:
    return (ROOT / "orion" / "cognition" / "prompts" / name).read_text(encoding="utf-8")


def _draft_ctx() -> dict:
    return {
        "trigger": {"trigger_kind": "baseline", "reason": "test", "pressure": 0.1, "zen_state": "zen"},
        "trigger_kind": "baseline",
        "context_summary": "unit test",
        "spark_state_json": "{}",
        "turn_effect_json": "{}",
        "recent_turn_effect_alerts_json": "[]",
        "turn_effect_policy_json": "{}",
        "turn_effect_explanations_json": "{}",
        "biometrics_json": "{}",
        "spark_phi_narrative": "",
    }


def test_metacog_draft_pass1_excludes_logprob_flags(monkeypatch):
    executor_module = _load_executor_module()
    captured: list[dict] = []

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            req = kwargs["req"]
            captured.append(dict(req.options or {}))
            if len(captured) == 1:
                return type("R", (), {"meta": {}, "choices": [{"message": {"content": _VALID_DRAFT_JSON}}]})()
            return type(
                "R",
                (),
                {"meta": {"llm_uncertainty": {"schema_version": "v1", "available": True, "source": "llamacpp_native_completion"}}, "choices": []},
            )()

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_return_logprobs", True)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_logprob_probe_mode", "native_completion")
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_uncertainty_probe_enabled", True)

    template = _load_template("log_orion_metacognition_draft.j2")
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
            bus=object(),
            source=source,
            step=step,
            ctx=_draft_ctx(),
            correlation_id="corr-two-pass-options",
        )
    )

    assert result.status == "success"
    assert len(captured) == 2
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert "return_logprobs" not in captured[0]
    assert "logprob_probe_mode" not in captured[0]
    assert captured[1]["return_logprobs"] is True
    assert captured[1]["logprob_probe_mode"] == "native_completion"
    assert captured[1]["max_tokens"] == 128
    assert "response_format" not in captured[1]
    telemetry = result.result["MetacogDraftService"]["entry"]["state_snapshot"]["telemetry"]
    assert telemetry["metacog_draft_mode"] == "llm"
    assert telemetry["llm_uncertainty"]["source"] == "llamacpp_native_completion"


def test_metacog_probe_failure_does_not_force_fallback(monkeypatch):
    executor_module = _load_executor_module()

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            req = kwargs["req"]
            opts = req.options or {}
            if opts.get("response_format"):
                return type("R", (), {"meta": {}, "choices": [{"message": {"content": _VALID_DRAFT_JSON}}]})()
            raise TimeoutError("probe timeout")

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_return_logprobs", True)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_logprob_probe_mode", "native_completion")
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_uncertainty_probe_enabled", True)

    template = _load_template("log_orion_metacognition_draft.j2")
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
            bus=object(),
            source=source,
            step=step,
            ctx=_draft_ctx(),
            correlation_id="corr-probe-fail",
        )
    )

    assert result.status == "success"
    telemetry = result.result["MetacogDraftService"]["entry"]["state_snapshot"]["telemetry"]
    assert telemetry["metacog_draft_mode"] == "llm"
    assert "llm_uncertainty" not in telemetry
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py::test_metacog_draft_pass1_excludes_logprob_flags \
  services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py::test_metacog_probe_failure_does_not_force_fallback \
  -v --tb=short
```

Expected: **FAIL** — pass 1 still includes logprob flags; only one LLM call.

- [ ] **Step 4: Implement two-pass draft**

In `MetacogDraftService` block, replace logprob wiring on `md_options` and add pass 2 after successful `patch_model` validation.

**Pass 1 options** — remove the block at lines 2692–2700; keep only:

```python
md_options: Dict[str, Any] = {
    "temperature": 0.8,
    "max_tokens": 1024,
    "response_format": {"type": "json_object"},
    "stream": False,
    **_md_lane,
}
```

**Pass 2** — after `patch_model = MetacogDraftTextPatchV1.model_validate(filtered)` succeeds and `draft_error`/`patch_error` are both None, insert before `base_entry = _fallback_metacog_draft(ctx)`:

```python
probe_unc: Dict[str, Any] | None = None
if (
    metacog_budget_ok
    and not draft_error
    and not patch_error
    and _should_run_metacog_uncertainty_probe()
):
    probe_mode = str(
        getattr(settings, "cortex_metacog_logprob_probe_mode", "") or ""
    ).strip()
    if probe_mode == "native_completion":
        probe_options: Dict[str, Any] = {
            "temperature": 0.8,
            "max_tokens": 128,
            "return_logprobs": True,
            "logprobs_top_k": 5,
            "logprob_summary_only": True,
            "logprob_probe_mode": "native_completion",
            "stream": False,
            **_md_lane,
        }
        probe_req = ChatRequestPayload(
            model=req_model,
            profile=ctx.get("profile_name") or settings.atlas_metacog_profile_name,
            messages=_metacog_uncertainty_probe_messages(patch_model),
            raw_user_text="metacog_uncertainty_probe",
            route="metacog",
            options=probe_options,
        )
        try:
            probe_res = await llm_client.chat(
                source=source,
                req=probe_req,
                correlation_id=correlation_id,
                reply_to=reply_channel,
                timeout_sec=effective_timeout,
            )
            if hasattr(probe_res, "meta") and isinstance(probe_res.meta, dict):
                maybe_unc = probe_res.meta.get("llm_uncertainty")
                if isinstance(maybe_unc, dict):
                    probe_unc = maybe_unc
        except Exception as probe_exc:
            logger.warning(
                "metacog_uncertainty_probe_failed corr_id=%s error=%s",
                correlation_id,
                probe_exc,
            )
```

**Uncertainty attach** — replace the existing `unc` block (lines 2766–2773) with:

```python
unc = probe_unc
if unc is None:
    md = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
    if isinstance(md.get("llm_uncertainty"), dict):
        unc = md["llm_uncertainty"]
if isinstance(unc, dict):
    attach_llm_uncertainty_to_collapse_payload(base_entry, unc)
```

Do **not** read pass-1 `llm_res.meta` for uncertainty (pass 1 has logprobs off).

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py -v --tb=short
```

Expected: **PASS** (all tests in file).

Also run firebreak regression:

```bash
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_firebreak.py \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py -q --tb=short
```

Expected: **PASS**

- [ ] **Step 6: Commit**

```bash
git add services/orion-cortex-exec/app/executor.py \
  services/orion-cortex-exec/app/settings.py \
  services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py
git commit -m "feat(cortex-exec): split metacog draft into content pass and native uncertainty probe"
```

---

## Task 4: Cortex-exec — enrich worker ctx preflight (biometrics trim)

**Files:**
- Modify: `services/orion-cortex-exec/app/executor.py` (near `_enforce_metacog_publish_prompt_budget`)
- Modify: `services/orion-cortex-exec/app/settings.py`
- Modify: `services/orion-cortex-exec/tests/test_metacog_publish_lane.py`

- [ ] **Step 1: Add settings field**

In `services/orion-cortex-exec/app/settings.py` after `cortex_metacog_enrich_prompt_max_chars`:

```python
cortex_metacog_enrich_worker_ctx_char_budget: int = Field(
    12000,
    alias="CORTEX_METACOG_ENRICH_WORKER_CTX_CHAR_BUDGET",
    description="MetacogEnrichService: trim biometrics_json and re-render when prompt exceeds worker ctx char budget.",
)
```

- [ ] **Step 2: Write failing test**

Append to `services/orion-cortex-exec/tests/test_metacog_publish_lane.py`:

```python
def test_enrich_trims_biometrics_before_ctx_overflow_fallback(monkeypatch):
    executor_module = _load_executor_module()
    calls: list[str] = []

    class FakeLLMClient:
        def __init__(self, bus):
            self.bus = bus

        async def chat(self, **kwargs):
            calls.append("enrich")
            return {}

    monkeypatch.setattr(executor_module, "LLMGatewayClient", FakeLLMClient)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_enrich_prompt_max_chars", 50000)
    monkeypatch.setattr(executor_module.settings, "cortex_metacog_enrich_worker_ctx_char_budget", 3000)

    draft_entry = CollapseMirrorEntryV2(
        event_id="evt-trim",
        id="evt-trim",
        trigger="dense",
        observer="orion",
        observer_state=["zen"],
        type="flow",
        emergent_entity="Test",
        summary="Test summary",
        mantra="Test mantra",
        field_resonance="Test resonance",
        resonance_signature="Test sig",
        source_service="metacog",
    ).model_dump(mode="json")
    draft_entry["state_snapshot"] = {"telemetry": {"metacog_draft_mode": "llm"}}

    template = _load_template("log_orion_metacognition_enrich.j2")
    ctx = _draft_ctx(spark_blob="Z" * 4000)
    ctx["biometrics_json"] = json.dumps({"hrv": "x" * 5000})
    ctx["collapse_entry"] = draft_entry
    ctx["collapse_json"] = json.dumps(draft_entry)

    step = ExecutionStep(
        verb_name="log_orion_metacognition",
        step_name="enrich_entry",
        order=1,
        services=["MetacogEnrichService"],
        prompt_template=template,
    )
    source = ServiceRef(name="test", node="test", version="1.0")

    result = asyncio.run(
        executor_module.call_step_services(
            bus=object(),
            source=source,
            step=step,
            ctx=ctx,
            correlation_id="corr-enrich-trim",
        )
    )

    assert result.status == "success"
    assert ctx["biometrics_json"] == "{}"
    enrich_result = result.result["MetacogEnrichService"]
    assert enrich_result["ok"] is True
    assert enrich_result.get("fallback_reason") != "prompt_context_overflow"
```

Add `import json` at top of file if missing.

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py::test_enrich_trims_biometrics_before_ctx_overflow_fallback -v --tb=short
```

Expected: **FAIL** — biometrics not trimmed; may hit `prompt_context_overflow` or LLM called with huge prompt.

- [ ] **Step 4: Implement enrich ctx preflight**

Add helper after `_enforce_metacog_publish_prompt_budget`:

```python
def _maybe_trim_metacog_enrich_prompt_for_worker_ctx(
    *,
    prompt: str,
    ctx: Dict[str, Any],
    template_str: str,
    correlation_id: str,
) -> tuple[str, str | None]:
    budget = int(settings.cortex_metacog_enrich_worker_ctx_char_budget)
    if len(prompt or "") <= budget:
        return prompt, None
    bio = str(ctx.get("biometrics_json") or "")
    if bio.strip() and bio.strip() != "{}":
        logger.warning(
            "metacog_enrich_ctx_trim_biometrics corr_id=%s prompt_chars=%s budget=%s bio_chars=%s",
            correlation_id,
            len(prompt),
            budget,
            len(bio),
        )
        ctx["biometrics_json"] = "{}"
        prompt = _render_prompt(template_str, ctx)
        if len(prompt) <= budget:
            return prompt, None
    logger.warning(
        "metacog_enrich_ctx_overflow corr_id=%s prompt_chars=%s budget=%s",
        correlation_id,
        len(prompt),
        budget,
    )
    return prompt, "prompt_context_overflow"
```

In `call_step_services`, after `prompt = _render_prompt(...)` and before metacog budget enforcement, for `MetacogEnrichService` only:

```python
enrich_ctx_overflow: str | None = None
if service == "MetacogEnrichService" and step.prompt_template:
    prompt, enrich_ctx_overflow = _maybe_trim_metacog_enrich_prompt_for_worker_ctx(
        prompt=prompt,
        ctx=ctx,
        template_str=step.prompt_template,
        correlation_id=correlation_id,
    )
```

In `MetacogEnrichService` block, before existing `if not metacog_budget_ok:` check, add:

```python
if enrich_ctx_overflow:
    logs.append("skip <- MetacogEnrichService LLM (prompt_context_overflow)")
    enrich_fallback_reason = enrich_ctx_overflow
    patch = {}
    metacog_budget_ok = False
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py -q --tb=short
```

Expected: **PASS**

- [ ] **Step 6: Commit**

```bash
git add services/orion-cortex-exec/app/executor.py \
  services/orion-cortex-exec/app/settings.py \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py
git commit -m "fix(cortex-exec): trim enrich biometrics when prompt exceeds worker ctx budget"
```

---

## Task 5: Env contract + local sync

**Files:**
- Modify: `services/orion-cortex-exec/.env_example`
- Modify: `services/orion-cortex-exec/docker-compose.yml` (if new env vars need passthrough)

- [ ] **Step 1: Update `.env_example`**

Replace the metacog logprob comment block (lines 68–76) with:

```bash
# Metacog draft: pass 1 = chat JSON (always). Pass 2 = native uncertainty probe when true.
CORTEX_METACOG_RETURN_LOGPROBS=true
CORTEX_METACOG_LOGPROB_PROBE_MODE=native_completion
# Kill-switch for pass 2 only (pass 1 unaffected):
CORTEX_METACOG_UNCERTAINTY_PROBE_ENABLED=true
# Enrich: trim biometrics_json once when rendered prompt exceeds worker ctx char budget.
CORTEX_METACOG_ENRICH_WORKER_CTX_CHAR_BUDGET=12000
```

- [ ] **Step 2: Add docker-compose passthrough** (if not already generic)

In `services/orion-cortex-exec/docker-compose.yml` environment section near existing `CORTEX_METACOG_*` keys, add:

```yaml
CORTEX_METACOG_UNCERTAINTY_PROBE_ENABLED: ${CORTEX_METACOG_UNCERTAINTY_PROBE_ENABLED:-true}
CORTEX_METACOG_ENRICH_WORKER_CTX_CHAR_BUDGET: ${CORTEX_METACOG_ENRICH_WORKER_CTX_CHAR_BUDGET:-12000}
```

- [ ] **Step 3: Sync local env**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
python scripts/sync_local_env_from_example.py
```

Expected: exit code **0**; `services/orion-cortex-exec/.env` gains new keys.

- [ ] **Step 4: Commit**

```bash
git add services/orion-cortex-exec/.env_example services/orion-cortex-exec/docker-compose.yml
git commit -m "docs(cortex-exec): document two-pass metacog logprob and enrich ctx budget env"
```

---

## Task 6: Compile + service-scoped test sweep

- [ ] **Step 1: Compile changed modules**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
./orion_dev/bin/python -m compileall services/orion-llm-gateway/app/llm_backend.py \
  services/orion-cortex-exec/app/executor.py services/orion-cortex-exec/app/settings.py
```

Expected: exit code **0**, no syntax errors.

- [ ] **Step 2: Run targeted pytest sweep**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=services/orion-llm-gateway ./orion_dev/bin/python -m pytest \
  services/orion-llm-gateway/tests/test_llm_backend.py -q --tb=short
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_metacog_two_pass_draft.py \
  services/orion-cortex-exec/tests/test_metacog_publish_lane.py \
  services/orion-cortex-exec/tests/test_firebreak.py \
  services/orion-cortex-exec/tests/test_collapse_llm_uncertainty_telemetry.py \
  -q --tb=short
```

Expected: all **PASS**.

---

## Live-stack acceptance (operator)

After deploy + container restart (`orion-cortex-exec`, `orion-llm-gateway`):

1. Trigger baseline metacog (or manual `log_orion_metacognition` with `trigger_kind=baseline`).
2. Gateway logs for same `corr_id`:
   - Pass 1: `llamacpp req ... /v1/chat/completions` (draft)
   - Pass 2: `native completion` with `n_predict<=128`
3. SQL / store: `collapse_mirror` row with `observer=orion`, `metacog_draft_mode=llm`, `llm_uncertainty.source=llamacpp_native_completion`.
4. Set `CORTEX_METACOG_RETURN_LOGPROBS=false` → single LLM call, publish still works.

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|------------------|------|
| Pass 1 chat + json_object, no logprobs | Task 3 |
| Pass 2 native probe, non-fatal failure | Task 2, Task 3 |
| Gateway guard on response_format | Task 1 |
| Enrich single-pass + biometrics trim | Task 4 |
| Env contract | Task 5 |
| Firebreak regression | Task 3 Step 5, Task 6 |
| Acceptance checks 1–5 | Task 6 + Live-stack section |

No placeholders remain. Type names (`MetacogDraftTextPatchV1`, `probe_unc`, `_should_run_metacog_uncertainty_probe`) are consistent across tasks.
