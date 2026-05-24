# LLM Uncertainty (Logprob Summary) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a canonical, summary-only `llm_uncertainty` object at the LLM Gateway boundary (OpenAI-compatible `/v1/chat/completions`), propagate it through `ChatResultPayload.meta` and chat `spark_meta`, then consume it in Mind phase telemetry and optional metacog triggers — framed as **language surface stability**, not truth confidence.

**Architecture:** Opt-in request options (`return_logprobs`, `logprobs_top_k`) cause the gateway to forward OpenAI `logprobs`/`top_logprobs` to llama.cpp, extract a compact summary from `choices[0].logprobs.content`, and attach it to `meta["llm_uncertainty"]`. Downstream organs read the summary only; full per-token distributions stay behind a debug flag. Mind enables logprobs only on `semantic_synthesis` initially.

**Tech Stack:** Python 3.12, Pydantic v2, httpx, unittest/pytest, Redis bus (`OrionBusAsync`), PostgreSQL JSONB (`spark_meta`).

**Design source:** GPT architecture discussion (2026-05-23) — gateway-first, summary-only, Mind-first consumer.

**Worktree (required — do not touch `feat/repair-pressure-v1`):**

```bash
cd /mnt/scripts/Orion-Sapienform
git worktree add .worktrees/feat-llm-uncertainty-v1 -b feat/llm-uncertainty-v1 origin/main
cd .worktrees/feat-llm-uncertainty-v1
```

**Isolation rules:**

- All commits on branch `feat/llm-uncertainty-v1` only.
- When updating `services/*/.env_example`, mirror keys into the matching `services/*/.env` on the machine (not committed).
- Do not merge or cherry-pick into other local branches except copying `.env` keys manually.
- After implementation: run code-reviewer subagent, fix findings, push PR from this branch only.

---

## Phase map

| Phase | Outcome | Primary services |
|-------|---------|------------------|
| **1** | Gateway extracts `llm_uncertainty`; `meta["llm_uncertainty"]` on bus replies | `orion-llm-gateway` |
| **2** | Chat history `spark_meta["llm_uncertainty"]` via cortex metadata → hub | `orion-cortex-exec`, `orion-hub`, `orion-sql-writer` |
| **3** | Mind `MindPhaseTelemetry.llm_uncertainty` for semantic synthesis | `orion-mind` |
| **4** | Advisory metacog trigger `llm_surface_instability` | `orion-mind`, bus → `orion-cortex-orch` |
| **5** | Collapse Mirror telemetry + journal index (optional, after signal proves useful) | `orion-cortex-exec`, journaler |

**Subagent rule:** One task = one commit. Do not start Phase 2 until Phase 1 gateway tests pass.

---

## Canonical `llm_uncertainty` shape (v1)

```python
{
    "schema_version": "v1",
    "source": "llamacpp_openai_chat",  # future: llamacpp_native_probe
    "available": True,
    "diagnostic_only": True,
    "confidence_semantics": "language_surface_stability_not_truth",
    "token_count_observed": 128,
    "mean_logprob": -0.71,
    "min_logprob": -4.9,
    "mean_top1_margin": 1.35,
    "low_margin_token_count": 9,
    "low_logprob_token_count": 6,
    "entropy_proxy_mean": 0.42,
    "unstable_span_count": 2,
}
```

Request options (via existing `ChatRequestPayload.options` — no schema migration):

```python
{
    "return_logprobs": True,
    "logprobs_top_k": 5,
    "logprob_summary_only": True,
    "logprob_probe_mode": "openai_chat",
}
```

Gateway env defaults (Phase 1 settings):

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_LOGPROB_SUMMARY_ENABLED` | `false` | Global allow (still requires per-request `return_logprobs`) |
| `LLM_LOGPROB_TOP_K_DEFAULT` | `5` | Default `top_logprobs` when enabled |
| `LLM_LOGPROB_LOW_MARGIN_THRESHOLD` | `0.5` | Margin below this counts as low-margin token |
| `LLM_LOGPROB_LOW_LOGPROB_THRESHOLD` | `-2.0` | logprob below this counts as low-logprob token |
| `LLM_LOGPROB_UNSTABLE_SPAN_MIN_LEN` | `3` | Consecutive low-margin run length for `unstable_span_count` |

Mind env (Phase 3):

| Variable | Default | Purpose |
|----------|---------|---------|
| `MIND_LLM_RETURN_LOGPROBS_SEMANTIC` | `false` | Opt-in logprobs for semantic_synthesis only |
| `MIND_LLM_UNCERTAINTY_METACOG_ENABLED` | `false` | Phase 4 advisory trigger |

---

## File structure

| Path | Responsibility |
|------|----------------|
| `services/orion-llm-gateway/app/llm_uncertainty.py` | **New** — extract + summarize logprobs from OpenAI response |
| `services/orion-llm-gateway/app/llm_backend.py` | Request passthrough + attach `llm_uncertainty` to return dict |
| `services/orion-llm-gateway/app/main.py` | Copy `llm_uncertainty` into `ChatResultPayload.meta` |
| `services/orion-llm-gateway/app/settings.py` | Threshold env knobs |
| `services/orion-llm-gateway/tests/test_llm_uncertainty.py` | **New** — unit tests for extractor |
| `services/orion-llm-gateway/tests/test_llm_backend.py` | Integration: payload includes logprobs when opted in |
| `services/orion-cortex-exec/app/router.py` | Forward `llm_uncertainty` into execution payload metadata |
| `services/orion-cortex-exec/tests/test_llm_uncertainty_metadata.py` | **New** — metadata passthrough |
| `services/orion-sql-writer/app/worker.py` | Merge `llm_uncertainty` from turn payload into `spark_meta` |
| `services/orion-sql-writer/tests/test_llm_uncertainty_spark_meta.py` | **New** |
| `services/orion-mind/app/llm_client.py` | Return `result.meta`; propagate to `request_json` meta |
| `services/orion-mind/app/phase_telemetry.py` | `llm_uncertainty` field |
| `services/orion-mind/app/synthesis.py` | Opt-in `return_logprobs` + set telemetry |
| `services/orion-mind/app/uncertainty_metacog.py` | **New** — advisory trigger evaluator |
| `services/orion-mind/app/settings.py` | Mind logprob flags |

---

# Phase 1 — Gateway-only, summary-only

### Task 1: Extractor module with failing tests

**Files:**
- Create: `services/orion-llm-gateway/app/llm_uncertainty.py`
- Test: `services/orion-llm-gateway/tests/test_llm_uncertainty.py`

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for OpenAI-shaped logprob summary extraction."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.llm_uncertainty import (  # noqa: E402
    extract_llm_uncertainty_from_openai_response,
    summarize_logprob_content,
)


def _sample_content():
    return [
        {"token": "The", "logprob": -0.1, "top_logprobs": [{"token": "The", "logprob": -0.1}, {"token": "A", "logprob": -2.5}]},
        {"token": " cat", "logprob": -0.3, "top_logprobs": [{"token": " cat", "logprob": -0.3}, {"token": " dog", "logprob": -1.8}]},
        {"token": " sat", "logprob": -3.5, "top_logprobs": [{"token": " sat", "logprob": -3.5}, {"token": " ran", "logprob": -3.2}]},
    ]


def test_summarize_logprob_content_computes_means_and_counts() -> None:
    summary = summarize_logprob_content(_sample_content())
    assert summary["available"] is True
    assert summary["token_count_observed"] == 3
    assert summary["mean_logprob"] == pytest.approx((-0.1 + -0.3 + -3.5) / 3, rel=1e-3)
    assert summary["min_logprob"] == pytest.approx(-3.5, rel=1e-3)
    assert summary["low_logprob_token_count"] >= 1
    assert summary["schema_version"] == "v1"


def test_extract_from_openai_response_reads_choices_logprobs() -> None:
    raw = {"choices": [{"logprobs": {"content": _sample_content()}}]}
    out = extract_llm_uncertainty_from_openai_response(raw, source="llamacpp_openai_chat")
    assert out is not None
    assert out["source"] == "llamacpp_openai_chat"
    assert out["available"] is True
    assert out["diagnostic_only"] is True


def test_extract_returns_none_when_no_logprobs() -> None:
    assert extract_llm_uncertainty_from_openai_response({"choices": [{}]}, source="x") is None
```

Add at top of test file: `import pytest`

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/orion-llm-gateway && PYTHONPATH=../.. python -m pytest tests/test_llm_uncertainty.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.llm_uncertainty'`

- [ ] **Step 3: Write minimal implementation**

Create `services/orion-llm-gateway/app/llm_uncertainty.py`:

```python
"""Summary-only language-surface stability metrics from OpenAI logprobs."""
from __future__ import annotations

import math
from typing import Any

from app.settings import settings

SCHEMA_VERSION = "v1"
CONFIDENCE_SEMANTICS = "language_surface_stability_not_truth"


def _token_logprob(entry: dict[str, Any]) -> float | None:
    lp = entry.get("logprob")
    if isinstance(lp, (int, float)):
        return float(lp)
    return None


def _top1_margin(entry: dict[str, Any]) -> float | None:
    tops = entry.get("top_logprobs")
    if not isinstance(tops, list) or len(tops) < 2:
        return None
    lps = [_token_logprob(t) for t in tops if isinstance(t, dict)]
    lps = [x for x in lps if x is not None]
    if len(lps) < 2:
        return None
    lps.sort(reverse=True)
    return lps[0] - lps[1]


def _entropy_proxy(entry: dict[str, Any]) -> float | None:
    tops = entry.get("top_logprobs")
    if not isinstance(tops, list) or not tops:
        return None
    lps = [_token_logprob(t) for t in tops if isinstance(t, dict)]
    lps = [x for x in lps if x is not None]
    if not lps:
        return None
    max_lp = max(lps)
    weights = [math.exp(lp - max_lp) for lp in lps]
    total = sum(weights)
    if total <= 0:
        return None
    probs = [w / total for w in weights]
    ent = -sum(p * math.log(p + 1e-12) for p in probs)
    return ent


def _count_unstable_spans(margins: list[float | None], *, min_len: int) -> int:
    run = 0
    count = 0
    threshold = float(getattr(settings, "llm_logprob_low_margin_threshold", 0.5))
    for m in margins:
        if m is not None and m < threshold:
            run += 1
            if run >= min_len:
                count += 1
        else:
            run = 0
    return count


def summarize_logprob_content(content: list[dict[str, Any]]) -> dict[str, Any]:
    logprobs: list[float] = []
    margins: list[float | None] = []
    entropies: list[float] = []
    low_margin = 0
    low_logprob = 0
    low_margin_threshold = float(getattr(settings, "llm_logprob_low_margin_threshold", 0.5))
    low_logprob_threshold = float(getattr(settings, "llm_logprob_low_logprob_threshold", -2.0))

    for entry in content:
        if not isinstance(entry, dict):
            continue
        lp = _token_logprob(entry)
        if lp is not None:
            logprobs.append(lp)
            if lp < low_logprob_threshold:
                low_logprob += 1
        margin = _top1_margin(entry)
        margins.append(margin)
        if margin is not None and margin < low_margin_threshold:
            low_margin += 1
        ent = _entropy_proxy(entry)
        if ent is not None:
            entropies.append(ent)

    if not logprobs:
        return {
            "schema_version": SCHEMA_VERSION,
            "available": False,
            "diagnostic_only": True,
            "confidence_semantics": CONFIDENCE_SEMANTICS,
            "token_count_observed": 0,
        }

    span_min = int(getattr(settings, "llm_logprob_unstable_span_min_len", 3))
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "diagnostic_only": True,
        "confidence_semantics": CONFIDENCE_SEMANTICS,
        "token_count_observed": len(logprobs),
        "mean_logprob": sum(logprobs) / len(logprobs),
        "min_logprob": min(logprobs),
        "mean_top1_margin": (sum(m for m in margins if m is not None) / max(1, sum(1 for m in margins if m is not None))),
        "low_margin_token_count": low_margin,
        "low_logprob_token_count": low_logprob,
        "entropy_proxy_mean": (sum(entropies) / len(entropies)) if entropies else None,
        "unstable_span_count": _count_unstable_spans(margins, min_len=span_min),
    }


def extract_llm_uncertainty_from_openai_response(
    raw_data: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any] | None:
    if not isinstance(raw_data, dict):
        return None
    choices = raw_data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    logprobs = first.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None
    summary = summarize_logprob_content(content)
    summary["source"] = source
    return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/orion-llm-gateway && PYTHONPATH=../.. python -m pytest tests/test_llm_uncertainty.py -v`

Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add services/orion-llm-gateway/app/llm_uncertainty.py services/orion-llm-gateway/tests/test_llm_uncertainty.py
git commit -m "feat(llm-gateway): add logprob summary extractor"
```

---

### Task 2: Gateway settings + env parity

**Files:**
- Modify: `services/orion-llm-gateway/app/settings.py`
- Modify: `services/orion-llm-gateway/.env_example`
- Modify: `services/orion-llm-gateway/.env` (local only, not committed)
- Modify: `services/orion-llm-gateway/docker-compose.yml`

- [ ] **Step 1: Add settings fields**

In `services/orion-llm-gateway/app/settings.py`, after `read_timeout_sec`:

```python
    llm_logprob_summary_enabled: bool = Field(False, alias="LLM_LOGPROB_SUMMARY_ENABLED")
    llm_logprob_top_k_default: int = Field(5, alias="LLM_LOGPROB_TOP_K_DEFAULT")
    llm_logprob_low_margin_threshold: float = Field(0.5, alias="LLM_LOGPROB_LOW_MARGIN_THRESHOLD")
    llm_logprob_low_logprob_threshold: float = Field(-2.0, alias="LLM_LOGPROB_LOW_LOGPROB_THRESHOLD")
    llm_logprob_unstable_span_min_len: int = Field(3, alias="LLM_LOGPROB_UNSTABLE_SPAN_MIN_LEN")
```

- [ ] **Step 2: Update `.env_example`**

Append to `services/orion-llm-gateway/.env_example`:

```bash
# Logprob summary (language surface stability; opt-in per request via return_logprobs)
LLM_LOGPROB_SUMMARY_ENABLED=false
LLM_LOGPROB_TOP_K_DEFAULT=5
LLM_LOGPROB_LOW_MARGIN_THRESHOLD=0.5
LLM_LOGPROB_LOW_LOGPROB_THRESHOLD=-2.0
LLM_LOGPROB_UNSTABLE_SPAN_MIN_LEN=3
```

Copy the same block into `services/orion-llm-gateway/.env`.

- [ ] **Step 3: Wire docker-compose**

In `services/orion-llm-gateway/docker-compose.yml` under `environment:`:

```yaml
      - LLM_LOGPROB_SUMMARY_ENABLED=${LLM_LOGPROB_SUMMARY_ENABLED:-false}
      - LLM_LOGPROB_TOP_K_DEFAULT=${LLM_LOGPROB_TOP_K_DEFAULT:-5}
      - LLM_LOGPROB_LOW_MARGIN_THRESHOLD=${LLM_LOGPROB_LOW_MARGIN_THRESHOLD:-0.5}
      - LLM_LOGPROB_LOW_LOGPROB_THRESHOLD=${LLM_LOGPROB_LOW_LOGPROB_THRESHOLD:-2.0}
      - LLM_LOGPROB_UNSTABLE_SPAN_MIN_LEN=${LLM_LOGPROB_UNSTABLE_SPAN_MIN_LEN:-3}
```

- [ ] **Step 4: Commit**

```bash
git add services/orion-llm-gateway/app/settings.py services/orion-llm-gateway/.env_example services/orion-llm-gateway/docker-compose.yml
git commit -m "chore(llm-gateway): add logprob summary env knobs"
```

---

### Task 3: Request passthrough + response attachment in `_execute_openai_chat`

**Files:**
- Modify: `services/orion-llm-gateway/app/llm_backend.py`
- Test: `services/orion-llm-gateway/tests/test_llm_backend.py`

- [ ] **Step 1: Write failing test**

Add to `TestLLMBackendExecution` in `test_llm_backend.py`:

```python
    @patch("app.llm_backend._common_http_client")
    def test_execute_openai_chat_forwards_logprobs_when_requested(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "choices": [{
                "message": {"content": "OK"},
                "logprobs": {
                    "content": [
                        {"token": "OK", "logprob": -0.2, "top_logprobs": [
                            {"token": "OK", "logprob": -0.2},
                            {"token": "NO", "logprob": -2.0},
                        ]},
                    ]
                },
            }]
        }
        body = ChatBody(
            messages=[ChatMessage(role="user", content="hi")],
            options={"return_logprobs": True, "logprobs_top_k": 3},
        )
        with patch.object(settings, "llm_logprob_summary_enabled", True):
            result = _execute_openai_chat(
                body=body,
                model="test-model",
                base_url="http://localhost",
                backend_name="llamacpp",
            )
        args, kwargs = mock_client.post.call_args
        payload = kwargs["json"]
        assert payload.get("logprobs") is True
        assert payload.get("top_logprobs") == 3
        assert isinstance(result.get("llm_uncertainty"), dict)
        assert result["llm_uncertainty"].get("available") is True
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `cd services/orion-llm-gateway && PYTHONPATH=../.. python -m pytest tests/test_llm_backend.py::TestLLMBackendExecution::test_execute_openai_chat_forwards_logprobs_when_requested -v`

Expected: FAIL (`logprobs` not in payload or no `llm_uncertainty` in result)

- [ ] **Step 3: Implement in `llm_backend.py`**

Add import at top:

```python
from app.llm_uncertainty import extract_llm_uncertainty_from_openai_response
```

After building `payload` dict (~line 852), before cleaning Nones:

```python
    return_logprobs = bool(opts.get("return_logprobs"))
    logprob_summary_enabled = bool(getattr(settings, "llm_logprob_summary_enabled", False))
    if return_logprobs and logprob_summary_enabled and backend_name in ("vllm", "llamacpp", "llama-cola"):
        top_k = int(opts.get("logprobs_top_k") or getattr(settings, "llm_logprob_top_k_default", 5))
        payload["logprobs"] = True
        payload["top_logprobs"] = max(1, min(top_k, 20))
```

After `raw_data = r.json()` and text extraction (~line 1007), before return:

```python
            llm_uncertainty = None
            if return_logprobs and logprob_summary_enabled:
                source_label = f"{backend_name}_openai_chat"
                llm_uncertainty = extract_llm_uncertainty_from_openai_response(
                    raw_data, source=source_label
                )
```

Add to return dict (~line 1036):

```python
                "llm_uncertainty": llm_uncertainty,
```

- [ ] **Step 4: Run test — expect PASS**

Run: `cd services/orion-llm-gateway && PYTHONPATH=../.. python -m pytest tests/test_llm_backend.py::TestLLMBackendExecution::test_execute_openai_chat_forwards_logprobs_when_requested -v`

- [ ] **Step 5: Commit**

```bash
git add services/orion-llm-gateway/app/llm_backend.py services/orion-llm-gateway/tests/test_llm_backend.py
git commit -m "feat(llm-gateway): forward logprobs and attach llm_uncertainty summary"
```

---

### Task 4: Expose in `handle_chat` → `ChatResultPayload.meta`

**Files:**
- Modify: `services/orion-llm-gateway/app/main.py`
- Test: `services/orion-llm-gateway/tests/test_handle_chat_meta.py` (new)

- [ ] **Step 1: Write failing test**

Create `services/orion-llm-gateway/tests/test_handle_chat_meta.py`:

```python
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef  # noqa: E402

from app.main import handle_chat  # noqa: E402


@pytest.mark.asyncio
async def test_handle_chat_meta_includes_llm_uncertainty():
    fake_result = {
        "text": "hi",
        "spark_meta": {},
        "raw": {"usage": {}},
        "llm_uncertainty": {"schema_version": "v1", "available": True, "mean_logprob": -0.5},
    }
    req = BaseEnvelope(
        kind="llm.chat.request",
        source=ServiceRef(name="test", node="n", version="0"),
        correlation_id="corr-1",
        payload=ChatRequestPayload(
            messages=[LLMMessage(role="user", content="ping")],
            route="quick",
        ).model_dump(mode="json"),
    )
    with patch("app.main.run_llm_chat", return_value=fake_result):
        out = await handle_chat(req, bus_handle=None)
    assert out.payload.meta is not None
    assert out.payload.meta.get("llm_uncertainty", {}).get("available") is True
```

- [ ] **Step 2: Run — expect FAIL**

Run: `cd services/orion-llm-gateway && PYTHONPATH=../.. python -m pytest tests/test_handle_chat_meta.py -v`

- [ ] **Step 3: Patch `main.py`**

After `structured_diag` merge into `meta` (~line 243):

```python
    llm_uncertainty = result.get("llm_uncertainty") if isinstance(result, dict) else None
    if isinstance(llm_uncertainty, dict):
        meta["llm_uncertainty"] = llm_uncertainty
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add services/orion-llm-gateway/app/main.py services/orion-llm-gateway/tests/test_handle_chat_meta.py
git commit -m "feat(llm-gateway): expose llm_uncertainty on ChatResultPayload.meta"
```

---

### Task 5: Phase 1 verification gate

- [ ] **Step 1: Run gateway test suite**

Run: `./scripts/test_service.sh orion-llm-gateway`

Expected: all tests PASS

- [ ] **Step 2: Manual smoke (optional, staging llama.cpp)**

```bash
curl -s http://<llamacpp-host>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Active-GGUF-Model","messages":[{"role":"user","content":"Say hi"}],"logprobs":true,"top_logprobs":3,"max_tokens":16}'
```

Confirm `choices[0].logprobs.content` exists before enabling in production routes.

---

# Phase 2 — Chat history `spark_meta` persistence

### Task 6: Cortex exec forwards gateway `meta.llm_uncertainty` into result metadata

**Files:**
- Modify: `services/orion-cortex-exec/app/router.py` (or executor step merge — locate where `LLMGatewayService` step result is normalized)
- Create: `services/orion-cortex-exec/tests/test_llm_uncertainty_metadata.py`

Hub already does `spark_meta = {..., **gateway_meta}` where `gateway_meta = resp.cortex_result.metadata`. So copy `llm_uncertainty` from gateway step payload `meta` into `PlanExecutionResult.metadata`.

- [ ] **Step 1: Find merge point**

Search: `grep -n "LLMGatewayService" services/orion-cortex-exec/app/executor.py | head`

In the function that builds `ctx["metadata"]` or exports `PlanExecutionResult.metadata`, after gateway response is parsed:

```python
    gw_meta = payload.get("meta") if isinstance(payload, dict) else None
    if isinstance(gw_meta, dict):
        unc = gw_meta.get("llm_uncertainty")
        if isinstance(unc, dict):
            md = ctx.setdefault("metadata", {})
            if isinstance(md, dict):
                md["llm_uncertainty"] = unc
```

Also ensure `LLMGatewayService` step `result` dict retains `meta` from bus decode (if stripped today, preserve it).

- [ ] **Step 2: Test**

```python
def test_plan_result_metadata_includes_llm_uncertainty():
    # Build minimal PlanExecutionResult via router export helper with fake step result:
    # LLMGatewayService: {"content": "x", "meta": {"llm_uncertainty": {"available": True}}}
    assert result.metadata["llm_uncertainty"]["available"] is True
```

- [ ] **Step 3: Commit**

```bash
git add services/orion-cortex-exec/
git commit -m "feat(cortex-exec): forward llm_uncertainty into execution metadata"
```

---

### Task 7: SQL writer merges `llm_uncertainty` into chat `spark_meta`

**Files:**
- Modify: `services/orion-sql-writer/app/worker.py` (ChatHistoryLogSQL block ~819)
- Create: `services/orion-sql-writer/tests/test_llm_uncertainty_spark_meta.py`

- [ ] **Step 1: Write failing test**

Test `_merge_spark_meta` path: payload with top-level `meta.llm_uncertainty` or `spark_meta.llm_uncertainty` ends up in filtered `spark_meta`.

- [ ] **Step 2: Implement**

In `ChatHistoryLogSQL` block after thinking_source merge:

```python
            meta_block = data.get("meta") if isinstance(data.get("meta"), dict) else {}
            unc = meta_block.get("llm_uncertainty")
            if not isinstance(unc, dict):
                sm = data.get("spark_meta") if isinstance(data.get("spark_meta"), dict) else {}
                unc = sm.get("llm_uncertainty")
            if isinstance(unc, dict):
                filtered_data["spark_meta"] = _merge_spark_meta(
                    filtered_data.get("spark_meta"), {"llm_uncertainty": unc}
                )
```

- [ ] **Step 3: Run tests**

Run: `./scripts/test_service.sh orion-sql-writer`

- [ ] **Step 4: Commit**

```bash
git add services/orion-sql-writer/
git commit -m "feat(sql-writer): persist llm_uncertainty in chat spark_meta"
```

---

# Phase 3 — Mind phase telemetry

### Task 8: `MindLLMClient` returns `result.meta`

**Files:**
- Modify: `services/orion-mind/app/llm_client.py`
- Modify: `services/orion-mind/tests/` (extend existing mind llm tests if present)

- [ ] **Step 1: Change `_bus_chat` return type**

```python
    ) -> tuple[str, dict[str, Any], str | None, dict[str, Any]]:
```

At return (~line 229):

```python
                result_meta = dict(result.meta or {})
                return str(result.content or result.text or ""), usage, result.model_used, result_meta
```

Update `request_json`:

```python
            content, usage, model_used, result_meta = self._bus_chat(...)
            meta["model_used"] = model_used
            meta["usage"] = usage
            if isinstance(result_meta, dict):
                unc = result_meta.get("llm_uncertainty")
                if isinstance(unc, dict):
                    meta["llm_uncertainty"] = unc
```

Update `_call` inner annotation and `FakeMindLLMClient` if tests break (return empty `{}` as fourth tuple element).

- [ ] **Step 2: Run mind tests**

Run: `./scripts/test_service.sh orion-mind`

- [ ] **Step 3: Commit**

```bash
git add services/orion-mind/app/llm_client.py
git commit -m "feat(mind): propagate llm_uncertainty from gateway meta"
```

---

### Task 9: `MindPhaseTelemetry.llm_uncertainty` + semantic synthesis opt-in

**Files:**
- Modify: `services/orion-mind/app/phase_telemetry.py`
- Modify: `services/orion-mind/app/synthesis.py`
- Modify: `services/orion-mind/app/settings.py`
- Modify: `services/orion-mind/.env_example` and `.env`

- [ ] **Step 1: Add dataclass field**

```python
    llm_uncertainty: dict[str, Any] | None = None
```

- [ ] **Step 2: Mind settings**

```python
    MIND_LLM_RETURN_LOGPROBS_SEMANTIC: bool = Field(default=False, alias="MIND_LLM_RETURN_LOGPROBS_SEMANTIC")
```

`.env_example`:

```bash
MIND_LLM_RETURN_LOGPROBS_SEMANTIC=false
```

- [ ] **Step 3: Opt-in in `run_semantic_synthesis`**

Before `client.request_json`:

```python
    options_extra: dict[str, Any] = {}
    if settings.MIND_LLM_RETURN_LOGPROBS_SEMANTIC:
        options_extra = {"return_logprobs": True, "logprobs_top_k": 5, "logprob_summary_only": True}
```

Pass via extending `MindLLMClient.request_json` with optional `extra_options: dict | None = None` merged into bus options, **or** set on context and read in `request_json` when `context.phase_name == "semantic_synthesis"`.

After `request_json` returns, set:

```python
    if isinstance(meta.get("llm_uncertainty"), dict):
        telemetry.llm_uncertainty = meta["llm_uncertainty"]
```

- [ ] **Step 4: Tests**

Unit test: when `MIND_LLM_RETURN_LOGPROBS_SEMANTIC=true` and fake client returns meta with `llm_uncertainty`, telemetry field is populated.

- [ ] **Step 5: Commit**

```bash
git add services/orion-mind/
git commit -m "feat(mind): attach llm_uncertainty to semantic synthesis telemetry"
```

---

# Phase 4 — Advisory metacog trigger

### Task 10: Instability evaluator + bus publish

**Files:**
- Create: `services/orion-mind/app/uncertainty_metacog.py`
- Modify: `services/orion-mind/app/synthesis.py` (call after telemetry built)
- Modify: `services/orion-mind/app/settings.py`

- [ ] **Step 1: Evaluator**

```python
def should_emit_llm_surface_instability(unc: dict[str, Any]) -> tuple[bool, str]:
    if not unc.get("available"):
        return False, "unavailable"
    tokens = int(unc.get("token_count_observed") or 0)
    if tokens <= 0:
        return False, "no_tokens"
    low_lp = int(unc.get("low_logprob_token_count") or 0)
    unstable = int(unc.get("unstable_span_count") or 0)
    margin = unc.get("mean_top1_margin")
    if unstable >= 1:
        return True, "unstable_span"
    if isinstance(margin, (int, float)) and margin < 0.75:
        return True, "low_mean_margin"
    if low_lp / tokens > 0.15:
        return True, "high_low_logprob_ratio"
    return False, "stable"
```

- [ ] **Step 2: Publish `orion.metacog.trigger.v1`**

Use existing bus pattern from `orion.schemas.telemetry.metacog_trigger.MetacogTriggerV1`:

```python
MetacogTriggerV1(
    trigger_kind="llm_surface_instability",
    reason="language_surface_unstable",
    pressure=min(1.0, low_lp / max(tokens, 1)),
    upstream={"llm_uncertainty": unc, "phase": "semantic_synthesis"},
)
```

Gate with `MIND_LLM_UNCERTAINTY_METACOG_ENABLED=false` by default.

- [ ] **Step 3: Test evaluator thresholds**

- [ ] **Step 4: Commit**

```bash
git add services/orion-mind/app/uncertainty_metacog.py services/orion-mind/app/synthesis.py
git commit -m "feat(mind): advisory metacog trigger for llm surface instability"
```

---

# Phase 5 — Collapse Mirror + journals (deferred until signal validated)

### Task 11: Collapse Mirror telemetry slice

**Files:**
- Modify: `services/orion-cortex-exec/app/executor.py` (collapse entry builder ~2342)

When `ctx["metadata"]["llm_uncertainty"]` exists, merge into `state_snapshot["telemetry"]`:

```python
telemetry["llm_uncertainty"] = {
    "mean_logprob": unc.get("mean_logprob"),
    "mean_top1_margin": unc.get("mean_top1_margin"),
    "unstable_span_count": unc.get("unstable_span_count"),
}
```

Set `epistemic_status` only if not already set — prefer `"model_observed"` when attaching diagnostic telemetry.

**Do not** add a new collapse mirror `type`; use existing `state_snapshot.telemetry` only.

- [ ] Commit separately after Phase 1–4 validated in staging.

### Task 12: Journal index metadata

Attach `llm_uncertainty` to journal evidence/index path (not journal `body`). Locate post-write index builder in journaler — add `llm_uncertainty` key to evidence unit metadata when present in upstream envelope.

---

## Self-review checklist

| Requirement | Task(s) |
|-------------|---------|
| Gateway OpenAI chat logprobs passthrough | 3 |
| Summary-only `llm_uncertainty` schema | 1 |
| `ChatResultPayload.meta` | 4 |
| No new SQL columns | 7 (JSONB only) |
| Mind phase telemetry | 8–9 |
| Metacog advisory trigger | 10 |
| Language-surface framing (`confidence_semantics`) | 1 |
| Collapse/journal deferred | 11–12 |
| Worktree isolation | header |
| `.env` sync on example updates | 2, 9 |

**Placeholder scan:** No TBD steps; each task includes concrete code paths and commands.

---

## Post-implementation: PR workflow

1. Run `./scripts/test_service.sh orion-llm-gateway`, `orion-mind`, `orion-sql-writer`, `orion-cortex-exec` (affected).
2. Invoke **code-reviewer** subagent on full diff vs `origin/main`.
3. Fix all reviewer issues; re-run tests.
4. Write PR report: `docs/superpowers/pr-reports/2026-05-23-llm-uncertainty-logprobs-pr.md`
5. Push and open PR:

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-llm-uncertainty-v1
git push -u origin feat/llm-uncertainty-v1
gh pr create --title "feat: LLM logprob summary (llm_uncertainty) at gateway + Mind telemetry" --body "$(cat <<'EOF'
## Summary
- Adds opt-in OpenAI logprob passthrough and summary-only `llm_uncertainty` at the LLM gateway.
- Propagates through chat `spark_meta` and Mind `MindPhaseTelemetry` for semantic synthesis.
- Optional advisory metacog trigger for language surface instability (default off).

## Test plan
- [ ] Gateway unit tests (`test_llm_uncertainty`, `test_llm_backend` logprobs case)
- [ ] Mind telemetry carries `llm_uncertainty` when `MIND_LLM_RETURN_LOGPROBS_SEMANTIC=true`
- [ ] Chat turn row in SQL has `spark_meta.llm_uncertainty` after hub chat with logprobs enabled
- [ ] Metacog trigger remains off unless `MIND_LLM_UNCERTAINTY_METACOG_ENABLED=true`

EOF
)"
```

---

## Execution handoff

**Plan saved to:** `docs/superpowers/plans/2026-05-23-llm-uncertainty-logprobs.md` (in worktree `.worktrees/feat-llm-uncertainty-v1`, branch `feat/llm-uncertainty-v1`).

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks.
2. **Inline Execution** — execute in this session with executing-plans checkpoints.

**Which approach?**
