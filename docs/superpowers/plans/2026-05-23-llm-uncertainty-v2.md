# LLM Uncertainty v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend PR #608 summary-only `llm_uncertainty` with native llama.cpp `/completion` aligned generation, queryable SQL scalars on `chat_history_log`, Collapse Mirror `state_snapshot.telemetry` attachment, and journal index persistence—without breaking the existing OpenAI `/v1/chat/completions` path.

**Architecture:** PR #608 already summarizes OpenAI-shaped `choices[0].logprobs.content` at the gateway. v2 adds an **opt-in alternate generation path** for `llamacpp`: `messages → POST /apply-template → prompt → POST /completion` with `n_probs`, then the same `_split_think_blocks()` and summary extractor so probabilities describe the **actual** assistant text. Downstream, SQL writer denormalizes summary fields into scalar columns while keeping `spark_meta.llm_uncertainty` as JSON source of truth; cortex-exec nests uncertainty under `state_snapshot.telemetry` for collapse mirrors; journal indexing carries uncertainty on `journal_entry_index` only (not `journal_entries`).

**Tech Stack:** Python 3.11+, FastAPI/httpx (orion-llm-gateway), SQLAlchemy + PostgreSQL (orion-sql-writer), Pydantic v2 schemas (`orion/journaler`, `orion/schemas/collapse_mirror`), pytest.

**Base branch:** `origin/main` at merge commit `e55bdabd` (PR #608 merged).

**Isolation (mandatory):**
- Worktree: `/mnt/scripts/Orion-Sapienform/.worktrees/feat-llm-uncertainty-v2`
- Branch: `feat/llm-uncertainty-v2` (tracks `origin/main`; does **not** touch `feat/repair-pressure-v1` or other active worktrees)
- One task = one commit
- When updating `.env_example`, also copy values into `.env` in the **worktree only** (never commit `.env`)
- Do not copy files to the main workspace checkout except `.env` sync if the user runs gateway locally from main path

---

## File map

| File | Responsibility |
|------|----------------|
| `services/orion-llm-gateway/app/llm_uncertainty.py` | Add native `/completion` prob normalization + extractor |
| `services/orion-llm-gateway/app/llm_backend.py` | Add `_execute_llamacpp_native_completion`; branch in `run_llm_chat` |
| `services/orion-llm-gateway/app/settings.py` | `LLM_LOGPROB_NATIVE_COMPLETION_*` settings |
| `services/orion-llm-gateway/.env_example`, `.env`, `docker-compose.yml` | Env parity |
| `services/orion-llm-gateway/tests/test_llm_uncertainty.py` | Native extractor unit tests |
| `services/orion-llm-gateway/tests/test_llm_backend.py` | Mocked apply-template + completion integration tests |
| `services/orion-sql-writer/app/main.py` | Idempotent `ALTER TABLE` for new columns |
| `services/orion-sql-writer/app/models/chat_history_log.py` | Scalar ORM columns |
| `services/orion-sql-writer/app/worker.py` | Extract scalars from `meta` / `spark_meta` on chat history write |
| `services/orion-sql-writer/tests/test_llm_uncertainty_spark_meta.py` | Extend with scalar column assertions |
| `orion/schemas/collapse_mirror.py` | `attach_llm_uncertainty_to_collapse_payload()` helper |
| `services/orion-cortex-exec/app/executor.py` | Wire helper after MetacogDraftService LLM (and any other collapse-from-LLM path found) |
| `services/orion-cortex-exec/tests/test_collapse_llm_uncertainty_telemetry.py` | **Create** — telemetry nesting tests |
| `orion/journaler/schemas.py` | Optional fields on `JournalEntryIndexV1` |
| `orion/journaler/indexing.py` | Populate index payload fields from `llm_uncertainty` |
| `services/orion-sql-writer/app/models/journal_entry_index.py` | JSONB + scalar columns |
| `services/orion-sql-writer/app/main.py` | `ALTER TABLE journal_entry_index` migrations |
| `services/orion-sql-writer/app/worker.py` | Pass `llm_uncertainty` into index builder |
| `services/orion-sql-writer/tests/test_journal_entry_indexing.py` | Index uncertainty tests |

---

## Prerequisites (already done for this plan author)

```bash
cd /mnt/scripts/Orion-Sapienform
git worktree add -b feat/llm-uncertainty-v2 .worktrees/feat-llm-uncertainty-v2 origin/main
cd .worktrees/feat-llm-uncertainty-v2
```

All implementation commands below assume `cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-llm-uncertainty-v2`.

---

### Task 1: Native llama.cpp completion uncertainty extractor

**Files:**
- Modify: `services/orion-llm-gateway/app/llm_uncertainty.py`
- Test: `services/orion-llm-gateway/tests/test_llm_uncertainty.py`

- [ ] **Step 1: Write the failing tests**

Append to `services/orion-llm-gateway/tests/test_llm_uncertainty.py`:

```python
from app.llm_uncertainty import (  # noqa: E402
    extract_llm_uncertainty_from_native_completion,
    native_completion_probs_to_logprob_content,
)


def _native_prob_token(token: str, logprob: float, alt_token: str, alt_logprob: float) -> dict:
    return {
        "token": token,
        "logprob": logprob,
        "top_logprobs": [
            {"token": token, "logprob": logprob},
            {"token": alt_token, "logprob": alt_logprob},
        ],
    }


def test_native_completion_probs_to_logprob_content_reads_probs_array() -> None:
    raw = {
        "content": "The cat sat",
        "probs": [
            _native_prob_token("The", -0.1, "A", -2.5),
            _native_prob_token(" cat", -0.3, " dog", -1.8),
        ],
    }
    content = native_completion_probs_to_logprob_content(raw)
    assert len(content) == 2
    assert content[0]["token"] == "The"


def test_native_completion_probs_to_logprob_content_reads_completion_probabilities() -> None:
    raw = {
        "completion_probabilities": [
            {
                "content": "Hi",
                "probs": [_native_prob_token("Hi", -0.2, "Hey", -1.5)],
            }
        ]
    }
    content = native_completion_probs_to_logprob_content(raw)
    assert len(content) == 1
    assert content[0]["logprob"] == pytest.approx(-0.2, rel=1e-3)


def test_extract_from_native_completion_sets_source() -> None:
    raw = {
        "content": "OK",
        "probs": [_native_prob_token("OK", -0.2, "NO", -2.0)],
    }
    out = extract_llm_uncertainty_from_native_completion(raw)
    assert out is not None
    assert out["source"] == "llamacpp_native_completion"
    assert out["available"] is True
    assert out["confidence_semantics"] == "language_surface_stability_not_truth"


def test_extract_from_native_completion_returns_none_without_probs() -> None:
    assert extract_llm_uncertainty_from_native_completion({"content": "x"}) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-llm-uncertainty-v2/services/orion-llm-gateway
pytest tests/test_llm_uncertainty.py::test_native_completion_probs_to_logprob_content_reads_probs_array -v
```

Expected: `ImportError` or `AttributeError` for `native_completion_probs_to_logprob_content`

- [ ] **Step 3: Implement native normalization + extractor**

Add to `services/orion-llm-gateway/app/llm_uncertainty.py` (after `summarize_logprob_content`):

```python
def native_completion_probs_to_logprob_content(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize llama.cpp /completion prob shapes into OpenAI logprobs.content entries."""
    if not isinstance(raw, dict):
        return []

    probs: list[Any] = []
    if isinstance(raw.get("probs"), list):
        probs = raw["probs"]
    elif isinstance(raw.get("completion_probabilities"), list) and raw["completion_probabilities"]:
        first = raw["completion_probabilities"][0]
        if isinstance(first, dict) and isinstance(first.get("probs"), list):
            probs = first["probs"]

    content: list[dict[str, Any]] = []
    for entry in probs:
        if not isinstance(entry, dict):
            continue
        token = entry.get("token")
        if token is None:
            continue
        lp = entry.get("logprob")
        if lp is None and entry.get("prob") is not None:
            # post_sampling_probs=true path — skip or convert; native path sets post_sampling_probs=false
            continue
        tops = entry.get("top_logprobs")
        if tops is None:
            tops = entry.get("top_probs")
        normalized_tops: list[dict[str, Any]] = []
        if isinstance(tops, list):
            for t in tops:
                if not isinstance(t, dict):
                    continue
                t_lp = t.get("logprob")
                if t_lp is None and t.get("prob") is not None:
                    continue
                normalized_tops.append({"token": t.get("token"), "logprob": t_lp})
        content.append({"token": token, "logprob": lp, "top_logprobs": normalized_tops})
    return content


def extract_llm_uncertainty_from_native_completion(
    raw_data: dict[str, Any],
    *,
    source: str = "llamacpp_native_completion",
) -> dict[str, Any] | None:
    if not isinstance(raw_data, dict):
        return None
    content = native_completion_probs_to_logprob_content(raw_data)
    if not content:
        return None
    summary = summarize_logprob_content(content)
    summary["source"] = source
    return summary
```

- [ ] **Step 4: Run native uncertainty tests**

Run:

```bash
pytest tests/test_llm_uncertainty.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-llm-gateway/app/llm_uncertainty.py services/orion-llm-gateway/tests/test_llm_uncertainty.py
git commit -m "feat(llm-gateway): extract llm_uncertainty from native llama.cpp completion"
```

---

### Task 2: Native llama.cpp aligned generation path

**Files:**
- Modify: `services/orion-llm-gateway/app/llm_backend.py`
- Modify: `services/orion-llm-gateway/app/settings.py`
- Modify: `services/orion-llm-gateway/.env_example`, `services/orion-llm-gateway/.env`, `services/orion-llm-gateway/docker-compose.yml`
- Test: `services/orion-llm-gateway/tests/test_llm_backend.py`

**Routing rule (critical):** When **all** are true:
- `backend_name == "llamacpp"`
- `bool(opts.get("return_logprobs"))`
- `bool(getattr(settings, "llm_logprob_summary_enabled", False))`
- `str(opts.get("logprob_probe_mode") or "").strip().lower() == "native_completion"`
- `bool(getattr(settings, "llm_logprob_native_completion_enabled", False))`

…call `_execute_llamacpp_native_completion` instead of `_execute_openai_chat`. This is the **primary** generation for that request—not a second sample.

Request options example:

```python
options = {
    "return_logprobs": True,
    "logprob_probe_mode": "native_completion",
    "logprobs_top_k": 5,
    "max_tokens": 256,
}
```

- [ ] **Step 1: Write failing integration tests**

Add to `services/orion-llm-gateway/tests/test_llm_backend.py` (match existing import/style):

```python
    @patch("app.llm_backend._common_http_client")
    def test_execute_llamacpp_native_completion_apply_template_then_completion(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client_factory.return_value.__enter__.return_value = mock_client

        apply_resp = MagicMock()
        apply_resp.raise_for_status = MagicMock()
        apply_resp.json.return_value = {"prompt": "<|user|>hi<|assistant|>"}

        completion_resp = MagicMock()
        completion_resp.raise_for_status = MagicMock()
        completion_resp.json.return_value = {
            "content": "OK",
            "probs": [
                {
                    "token": "OK",
                    "logprob": -0.2,
                    "top_logprobs": [
                        {"token": "OK", "logprob": -0.2},
                        {"token": "NO", "logprob": -2.0},
                    ],
                }
            ],
        }

        mock_client.post.side_effect = [apply_resp, completion_resp]

        body = ChatBody(
            messages=[ChatMessage(role="user", content="hi")],
            options={
                "return_logprobs": True,
                "logprob_probe_mode": "native_completion",
                "logprobs_top_k": 5,
                "max_tokens": 64,
            },
        )
        with patch.object(settings, "llm_logprob_summary_enabled", True), patch.object(
            settings, "llm_logprob_native_completion_enabled", True
        ):
            result = _execute_llamacpp_native_completion(
                body=body,
                model="test-model",
                base_url="http://llamacpp:8080",
                backend_name="llamacpp",
            )

        assert mock_client.post.call_count == 2
        apply_url = mock_client.post.call_args_list[0][0][0]
        completion_url = mock_client.post.call_args_list[1][0][0]
        assert apply_url.endswith("/apply-template")
        assert completion_url.endswith("/completion")
        completion_payload = mock_client.post.call_args_list[1].kwargs["json"]
        assert completion_payload["n_probs"] == 5
        assert completion_payload["post_sampling_probs"] is False
        assert result["text"] == "OK"
        assert result["llm_uncertainty"]["source"] == "llamacpp_native_completion"
        assert result["llm_uncertainty"]["available"] is True

    @patch("app.llm_backend._execute_llamacpp_native_completion")
    @patch("app.llm_backend._execute_openai_chat")
    def test_run_llm_chat_routes_native_completion_when_opted_in(
        self, mock_openai, mock_native, monkeypatch
    ):
        mock_native.return_value = {"text": "native", "spark_meta": {}, "raw": {}, "llm_uncertainty": {"available": True}}
        body = ChatBody(
            messages=[ChatMessage(role="user", content="hi")],
            options={"return_logprobs": True, "logprob_probe_mode": "native_completion"},
        )
        monkeypatch.setattr(settings, "llamacpp_url", "http://llamacpp:8080")
        monkeypatch.setattr(settings, "llm_logprob_summary_enabled", True)
        monkeypatch.setattr(settings, "llm_logprob_native_completion_enabled", True)
        monkeypatch.setattr(
            "app.llm_backend.resolve_route_target",
            lambda *a, **k: SimpleNamespace(backend="llamacpp", url="http://llamacpp:8080", served_by="atlas"),
        )
        from app.llm_backend import run_llm_chat

        run_llm_chat(body, model="m", route="chat")
        mock_native.assert_called_once()
        mock_openai.assert_not_called()
```

Import `SimpleNamespace` from `types` if not already present.

- [ ] **Step 2: Run tests to verify failure**

```bash
cd services/orion-llm-gateway
pytest tests/test_llm_backend.py::TestExecuteOpenAIChat::test_execute_llamacpp_native_completion_apply_template_then_completion -v
```

Expected: FAIL (`_execute_llamacpp_native_completion` not defined)

- [ ] **Step 3: Add settings**

In `services/orion-llm-gateway/app/settings.py` after existing `llm_logprob_unstable_span_min_len`:

```python
    llm_logprob_native_completion_enabled: bool = Field(
        False, alias="LLM_LOGPROB_NATIVE_COMPLETION_ENABLED"
    )
    llm_logprob_native_completion_max_tokens: int = Field(
        256, alias="LLM_LOGPROB_NATIVE_COMPLETION_MAX_TOKENS"
    )
```

In `.env_example` and `.env` (worktree only):

```bash
LLM_LOGPROB_NATIVE_COMPLETION_ENABLED=false
LLM_LOGPROB_NATIVE_COMPLETION_MAX_TOKENS=256
```

In `docker-compose.yml` `environment:` block next to other `LLM_LOGPROB_*`:

```yaml
      - LLM_LOGPROB_NATIVE_COMPLETION_ENABLED=${LLM_LOGPROB_NATIVE_COMPLETION_ENABLED:-false}
      - LLM_LOGPROB_NATIVE_COMPLETION_MAX_TOKENS=${LLM_LOGPROB_NATIVE_COMPLETION_MAX_TOKENS:-256}
```

- [ ] **Step 4: Implement `_execute_llamacpp_native_completion`**

In `llm_backend.py`, add import:

```python
from .llm_uncertainty import (
    extract_llm_uncertainty_from_native_completion,
    extract_llm_uncertainty_from_openai_response,
)
```

Add function (mirror `_execute_openai_chat` return dict keys: `text`, `spark_meta`, `spark_vector`, `raw`, `reasoning_content`, `reasoning_trace`, `inline_think_content`, `structured_output_diagnostics`, `llm_uncertainty`):

```python
def _execute_llamacpp_native_completion(
    body: ChatBody,
    model: str,
    base_url: str,
    backend_name: str,
    route: Optional[str] = None,
    served_by: Optional[str] = None,
) -> Dict[str, Any]:
    """Aligned generation via llama.cpp /apply-template + /completion with n_probs."""
    if not base_url:
        err = f"{backend_name} URL not configured"
        logger.error(f"[LLM-GW] {err}")
        return {"text": f"[Error: {err}]", "spark_meta": {}, "raw": {}}

    spark_meta = _spark_ingest_for_body(body)
    opts = body.options or {}
    apply_url = f"{base_url.rstrip('/')}/apply-template"
    completion_url = f"{base_url.rstrip('/')}/completion"

    top_k = int(opts.get("logprobs_top_k") or getattr(settings, "llm_logprob_top_k_default", 5))
    n_probs = max(1, min(top_k, 20))
    max_tokens = opts.get("max_tokens")
    if max_tokens is None:
        max_tokens = int(getattr(settings, "llm_logprob_native_completion_max_tokens", 256))
    max_tokens = int(max_tokens)

    apply_payload: Dict[str, Any] = {
        "model": model,
        "messages": _serialize_messages(body.messages or []),
        "temperature": opts.get("temperature"),
        "top_p": opts.get("top_p"),
    }
    ctk = opts.get("chat_template_kwargs")
    if isinstance(ctk, dict) and ctk:
        apply_payload["chat_template_kwargs"] = ctk
    apply_payload = {k: v for k, v in apply_payload.items() if v is not None}

    completion_payload: Dict[str, Any] = {
        "prompt": None,  # filled after apply-template
        "n_predict": max_tokens,
        "n_probs": n_probs,
        "stream": False,
        "post_sampling_probs": False,
        "temperature": opts.get("temperature"),
        "top_p": opts.get("top_p"),
        "stop": opts.get("stop"),
    }
    completion_payload = {k: v for k, v in completion_payload.items() if v is not None}

    try:
        with _common_http_client(body) as client:
            apply_resp = client.post(apply_url, json=apply_payload)
            if apply_resp.status_code == 404:
                return {
                    "text": f"[Error: {backend_name} /apply-template 404 at {apply_url}]",
                    "spark_meta": spark_meta,
                    "raw": {},
                }
            apply_resp.raise_for_status()
            apply_data = apply_resp.json()
            prompt = apply_data.get("prompt") if isinstance(apply_data, dict) else None
            if not isinstance(prompt, str) or not prompt.strip():
                return {
                    "text": f"[Error: {backend_name} /apply-template returned empty prompt]",
                    "spark_meta": spark_meta,
                    "raw": apply_data if isinstance(apply_data, dict) else {},
                }

            completion_payload["prompt"] = prompt
            r = client.post(completion_url, json=completion_payload)
            if r.status_code == 404:
                return {
                    "text": f"[Error: {backend_name} /completion 404 at {completion_url}]",
                    "spark_meta": spark_meta,
                    "raw": {},
                }
            r.raise_for_status()
            raw_data = r.json() if isinstance(r.json(), dict) else {}

        text = str(raw_data.get("content") or "")
        llm_uncertainty = None
        if bool(getattr(settings, "llm_logprob_summary_enabled", False)):
            llm_uncertainty = extract_llm_uncertainty_from_native_completion(raw_data)

        text, think_reasoning = _split_think_blocks(text)
        _spark_post_ingest_for_reply(body, spark_meta, text)
        spark_vector = None  # native completion has no OpenAI embedding block

        raw_out = dict(raw_data)
        if opts.get("logprob_summary_only", True) and isinstance(raw_out.get("probs"), list):
            raw_out = {k: v for k, v in raw_out.items() if k not in ("probs", "completion_probabilities")}

        return {
            "text": text,
            "spark_meta": spark_meta,
            "spark_vector": spark_vector,
            "raw": raw_out,
            "reasoning_content": None,
            "reasoning_trace": None,
            "inline_think_content": think_reasoning or None,
            "structured_output_diagnostics": None,
            "llm_uncertainty": llm_uncertainty,
        }
    except httpx.TimeoutException:
        logger.error("[LLM-GW] %s native completion TIMEOUT corr=%s", backend_name, body.trace_id)
        return {"text": f"[Error: {backend_name} timed out]", "spark_meta": spark_meta, "raw": {}}
    except Exception as e:
        logger.error(f"[LLM-GW] {backend_name} native completion error: {e}", exc_info=True)
        return {"text": f"[Error: {backend_name} failed: {str(e)}]", "spark_meta": spark_meta, "raw": {}}
```

In `run_llm_chat`, immediately before the final `result = _execute_openai_chat(...)` for `llamacpp` backend (~line 1306), insert:

```python
    opts = body.options or {}
    use_native = (
        backend == "llamacpp"
        and bool(opts.get("return_logprobs"))
        and bool(getattr(settings, "llm_logprob_summary_enabled", False))
        and bool(getattr(settings, "llm_logprob_native_completion_enabled", False))
        and str(opts.get("logprob_probe_mode") or "").strip().lower() == "native_completion"
    )
    if use_native:
        result = _execute_llamacpp_native_completion(
            body, model, base_url, backend, route=route, served_by=served_by
        )
        if isinstance(result, dict):
            result["backend"] = backend
            result["model"] = model
            result["route"] = route
            result["served_by"] = served_by
        return result
```

**Do not implement side-probe mode in v2.** If added later, tag with `source=llamacpp_native_completion_side_probe`, `alignment=independent_resample`, `diagnostic_only=true`.

- [ ] **Step 5: Run gateway tests**

```bash
cd services/orion-llm-gateway
pytest tests/test_llm_uncertainty.py tests/test_llm_backend.py -v
```

Expected: all PASS; existing OpenAI logprob tests unchanged

- [ ] **Step 6: Commit**

```bash
git add services/orion-llm-gateway/app/llm_backend.py services/orion-llm-gateway/app/settings.py \
  services/orion-llm-gateway/.env_example services/orion-llm-gateway/docker-compose.yml \
  services/orion-llm-gateway/tests/test_llm_backend.py
# .env is gitignored — still update locally
git commit -m "feat(llm-gateway): add native llama.cpp completion path for aligned logprobs"
```

---

### Task 3: SQL scalar columns on `chat_history_log`

**Files:**
- Modify: `services/orion-sql-writer/app/main.py`
- Modify: `services/orion-sql-writer/app/models/chat_history_log.py`
- Modify: `services/orion-sql-writer/app/worker.py`
- Test: `services/orion-sql-writer/tests/test_llm_uncertainty_spark_meta.py`

- [ ] **Step 1: Write failing scalar column test**

Append to `services/orion-sql-writer/tests/test_llm_uncertainty_spark_meta.py`:

```python
_UNC_FULL = {
    "schema_version": "v1",
    "available": True,
    "source": "llamacpp_native_completion",
    "mean_logprob": -0.74,
    "min_logprob": -3.5,
    "mean_top1_margin": 1.2,
    "low_margin_token_count": 2,
    "low_logprob_token_count": 1,
    "unstable_span_count": 2,
}


def _write_chat_history_row(monkeypatch, payload: dict):
    sess = _FakeSession()
    monkeypatch.setattr(worker, "get_session", lambda: sess)
    monkeypatch.setattr(worker, "remove_session", lambda: None)
    assert worker._write_row(worker.ChatHistoryLogSQL, payload) is True
    return sess.row


def test_chat_history_log_scalar_columns_from_meta_llm_uncertainty(monkeypatch) -> None:
    row = _write_chat_history_row(
        monkeypatch,
        {
            "correlation_id": "corr-scalar-unc",
            "prompt": "hello",
            "response": "world",
            "meta": {"llm_uncertainty": _UNC_FULL},
        },
    )
    assert row.llm_uncertainty_source == "llamacpp_native_completion"
    assert row.llm_mean_logprob == pytest.approx(-0.74, rel=1e-3)
    assert row.llm_min_logprob == pytest.approx(-3.5, rel=1e-3)
    assert row.llm_mean_top1_margin == pytest.approx(1.2, rel=1e-3)
    assert row.llm_low_margin_token_count == 2
    assert row.llm_low_logprob_token_count == 1
    assert row.llm_unstable_span_count == 2
    assert row.llm_uncertainty_available is True
    assert row.spark_meta["llm_uncertainty"] == _UNC_FULL
```

Add `import pytest` at top if missing.

- [ ] **Step 2: Run test to verify failure**

```bash
cd services/orion-sql-writer
pytest tests/test_llm_uncertainty_spark_meta.py::test_chat_history_log_scalar_columns_from_meta_llm_uncertainty -v
```

Expected: FAIL (no `llm_mean_logprob` on model/row)

- [ ] **Step 3: Migration + model + worker helper**

In `app/main.py` startup migrations (near other `chat_history_log` alters):

```python
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_uncertainty_source TEXT;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_mean_logprob DOUBLE PRECISION;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_min_logprob DOUBLE PRECISION;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_mean_top1_margin DOUBLE PRECISION;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_low_margin_token_count INTEGER;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_low_logprob_token_count INTEGER;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_unstable_span_count INTEGER;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_uncertainty_available BOOLEAN;"
            )
```

In `app/models/chat_history_log.py`:

```python
from sqlalchemy import Boolean, Column, Float, Integer, String, Text, DateTime
# ...
    llm_uncertainty_source = Column(String, nullable=True)
    llm_mean_logprob = Column(Float, nullable=True)
    llm_min_logprob = Column(Float, nullable=True)
    llm_mean_top1_margin = Column(Float, nullable=True)
    llm_low_margin_token_count = Column(Integer, nullable=True)
    llm_low_logprob_token_count = Column(Integer, nullable=True)
    llm_unstable_span_count = Column(Integer, nullable=True)
    llm_uncertainty_available = Column(Boolean, nullable=True)
```

In `app/worker.py` add helper near `_merge_spark_meta`:

```python
def _chat_history_llm_uncertainty_scalars(unc: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(unc, dict):
        return {}
    available = unc.get("available")
    return {
        "llm_uncertainty_source": unc.get("source"),
        "llm_mean_logprob": unc.get("mean_logprob"),
        "llm_min_logprob": unc.get("min_logprob"),
        "llm_mean_top1_margin": unc.get("mean_top1_margin"),
        "llm_low_margin_token_count": unc.get("low_margin_token_count"),
        "llm_low_logprob_token_count": unc.get("low_logprob_token_count"),
        "llm_unstable_span_count": unc.get("unstable_span_count"),
        "llm_uncertainty_available": available if isinstance(available, bool) else None,
    }
```

Inside `if sql_model_cls is ChatHistoryLogSQL:` block, after merging `spark_meta` with `llm_uncertainty`:

```python
            if isinstance(unc, dict):
                filtered_data.update(_chat_history_llm_uncertainty_scalars(unc))
```

- [ ] **Step 4: Run sql-writer tests**

```bash
pytest tests/test_llm_uncertainty_spark_meta.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-sql-writer/app/main.py services/orion-sql-writer/app/models/chat_history_log.py \
  services/orion-sql-writer/app/worker.py services/orion-sql-writer/tests/test_llm_uncertainty_spark_meta.py
git commit -m "feat(sql-writer): add chat_history_log llm uncertainty scalar columns"
```

---

### Task 4: Collapse Mirror `state_snapshot.telemetry` attachment

**Files:**
- Modify: `orion/schemas/collapse_mirror.py`
- Modify: `services/orion-cortex-exec/app/executor.py`
- Create: `services/orion-cortex-exec/tests/test_collapse_llm_uncertainty_telemetry.py`

- [ ] **Step 1: Write failing test**

Create `services/orion-cortex-exec/tests/test_collapse_llm_uncertainty_telemetry.py`:

```python
from orion.schemas.collapse_mirror import attach_llm_uncertainty_to_collapse_payload

_UNC = {
    "schema_version": "v1",
    "available": True,
    "source": "llamacpp_native_completion",
    "mean_logprob": -0.74,
    "mean_top1_margin": 1.2,
    "unstable_span_count": 2,
}


def test_attach_llm_uncertainty_preserves_existing_telemetry() -> None:
    payload = {
        "state_snapshot": {
            "telemetry": {
                "change_type_meta": {"flow": 0.9},
                "gpu_util": 0.42,
            }
        }
    }
    attach_llm_uncertainty_to_collapse_payload(payload, _UNC)
    telemetry = payload["state_snapshot"]["telemetry"]
    assert telemetry["gpu_util"] == 0.42
    assert telemetry["change_type_meta"] == {"flow": 0.9}
    assert telemetry["llm_uncertainty"] == _UNC
    assert telemetry["llm_uncertainty_semantics"] == "language_surface_stability_not_truth"


def test_attach_llm_uncertainty_noop_when_missing() -> None:
    payload = {"state_snapshot": {"telemetry": {"gpu_util": 1.0}}}
    attach_llm_uncertainty_to_collapse_payload(payload, None)
    assert "llm_uncertainty" not in payload["state_snapshot"]["telemetry"]
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd services/orion-cortex-exec
pytest tests/test_collapse_llm_uncertainty_telemetry.py -v
```

Expected: `ImportError` for `attach_llm_uncertainty_to_collapse_payload`

- [ ] **Step 3: Implement helper**

In `orion/schemas/collapse_mirror.py` (module level, after constants):

```python
def attach_llm_uncertainty_to_collapse_payload(
    payload: dict[str, Any],
    llm_uncertainty: dict[str, Any] | None,
) -> None:
    if not isinstance(payload, dict) or not isinstance(llm_uncertainty, dict):
        return
    state_snapshot = payload.setdefault("state_snapshot", {})
    if not isinstance(state_snapshot, dict):
        state_snapshot = {}
        payload["state_snapshot"] = state_snapshot
    telemetry = state_snapshot.setdefault("telemetry", {})
    if not isinstance(telemetry, dict):
        telemetry = {}
        state_snapshot["telemetry"] = telemetry
    telemetry["llm_uncertainty"] = llm_uncertainty
    telemetry["llm_uncertainty_semantics"] = "language_surface_stability_not_truth"
```

- [ ] **Step 4: Wire MetacogDraftService**

In `services/orion-cortex-exec/app/executor.py` import:

```python
from orion.schemas.collapse_mirror import (
    CollapseMirrorEntryV2,
    attach_llm_uncertainty_to_collapse_payload,
    find_collapse_entry,
    normalize_collapse_entry,
)
```

In `MetacogDraftService` block, after `base_entry = _fallback_metacog_draft(ctx).model_dump(mode="json")` and `_apply_metacog_system_fields`, before `_apply_draft_patch`:

```python
                    unc = None
                    md = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
                    if isinstance(md.get("llm_uncertainty"), dict):
                        unc = md["llm_uncertainty"]
                    elif hasattr(llm_res, "meta") and isinstance(llm_res.meta, dict):
                        unc = llm_res.meta.get("llm_uncertainty")
                    if isinstance(unc, dict):
                        attach_llm_uncertainty_to_collapse_payload(base_entry, unc)
```

Also call `_forward_llm_uncertainty_metadata` on the gateway-shaped dict if the LLM step returns payload with `meta`—ensure `llm_res.meta` is populated (gateway already sets `ChatResultPayload.meta`).

Search for other collapse mirror compositions from LLM outputs (e.g. `MetacogEnrichService` only patches scores—**do not** attach unrelated uncertainty there unless the same LLM call produced it).

- [ ] **Step 5: Run cortex-exec tests**

```bash
pytest tests/test_collapse_llm_uncertainty_telemetry.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add orion/schemas/collapse_mirror.py services/orion-cortex-exec/app/executor.py \
  services/orion-cortex-exec/tests/test_collapse_llm_uncertainty_telemetry.py
git commit -m "feat(cortex-exec): attach llm_uncertainty to collapse mirror telemetry"
```

---

### Task 5: Journal entry index attachment

**Files:**
- Modify: `orion/journaler/schemas.py`
- Modify: `orion/journaler/indexing.py`
- Modify: `services/orion-sql-writer/app/models/journal_entry_index.py`
- Modify: `services/orion-sql-writer/app/main.py`
- Modify: `services/orion-sql-writer/app/worker.py`
- Test: `services/orion-sql-writer/tests/test_journal_entry_indexing.py`

- [ ] **Step 1: Write failing journal index test**

Append to `services/orion-sql-writer/tests/test_journal_entry_indexing.py`:

```python
def test_index_payload_carries_llm_uncertainty_fields() -> None:
    unc = {
        "schema_version": "v1",
        "available": True,
        "source": "llamacpp_openai_chat",
        "mean_logprob": -0.5,
        "mean_top1_margin": 0.9,
        "unstable_span_count": 1,
    }
    payload = build_journal_entry_index_payload(
        _base_write(),
        stance_metadata={"llm_uncertainty": unc},
    )
    assert payload["llm_uncertainty"] == unc
    assert payload["llm_mean_logprob"] == pytest.approx(-0.5, rel=1e-3)
    assert payload["llm_mean_top1_margin"] == pytest.approx(0.9, rel=1e-3)
    assert payload["llm_unstable_span_count"] == 1
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd services/orion-sql-writer
pytest tests/test_journal_entry_indexing.py::test_index_payload_carries_llm_uncertainty_fields -v
```

Expected: FAIL (unknown fields on payload / schema)

- [ ] **Step 3: Schema + indexing + SQL**

`orion/journaler/schemas.py` on `JournalEntryIndexV1`:

```python
    llm_uncertainty: dict[str, Any] | None = None
    llm_mean_logprob: float | None = None
    llm_mean_top1_margin: float | None = None
    llm_unstable_span_count: int | None = None
```

`orion/journaler/indexing.py` — extend signature and payload build:

```python
def build_journal_entry_index_payload(
    write: JournalEntryWriteV1,
    *,
    trigger: JournalTriggerV1 | None = None,
    chat_stance: ChatStanceBrief | None = None,
    stance_metadata: dict[str, Any] | None = None,
    llm_uncertainty: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # ... existing code ...
    unc = llm_uncertainty
    if unc is None and isinstance(stance_metadata, dict):
        candidate = stance_metadata.get("llm_uncertainty")
        unc = candidate if isinstance(candidate, dict) else None
    payload["llm_uncertainty"] = unc
    payload["llm_mean_logprob"] = unc.get("mean_logprob") if isinstance(unc, dict) else None
    payload["llm_mean_top1_margin"] = unc.get("mean_top1_margin") if isinstance(unc, dict) else None
    payload["llm_unstable_span_count"] = unc.get("unstable_span_count") if isinstance(unc, dict) else None
    return payload
```

`services/orion-sql-writer/app/worker.py` — extend `_extract_journal_index_context`:

```python
    raw_unc = container.get("llm_uncertainty")
    if isinstance(raw_unc, dict):
        if stance_meta is None:
            stance_meta = {}
        stance_meta.setdefault("llm_uncertainty", raw_unc)
```

And when calling `build_journal_entry_index_payload`, pass through if top-level envelope has `llm_uncertainty`.

`journal_entry_index` model columns:

```python
from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
# ...
    llm_uncertainty = Column(JSONB, nullable=True)
    llm_mean_logprob = Column(Float, nullable=True)
    llm_mean_top1_margin = Column(Float, nullable=True)
    llm_unstable_span_count = Column(Integer, nullable=True)
```

`main.py` migrations after `journal_entry_index` CREATE:

```python
            conn.exec_driver_sql(
                "ALTER TABLE journal_entry_index ADD COLUMN IF NOT EXISTS llm_uncertainty JSONB;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE journal_entry_index ADD COLUMN IF NOT EXISTS llm_mean_logprob DOUBLE PRECISION;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE journal_entry_index ADD COLUMN IF NOT EXISTS llm_mean_top1_margin DOUBLE PRECISION;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE journal_entry_index ADD COLUMN IF NOT EXISTS llm_unstable_span_count INTEGER;"
            )
```

- [ ] **Step 4: Run journal + registry tests**

```bash
pytest tests/test_journal_entry_indexing.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/journaler/schemas.py orion/journaler/indexing.py \
  services/orion-sql-writer/app/models/journal_entry_index.py services/orion-sql-writer/app/main.py \
  services/orion-sql-writer/app/worker.py services/orion-sql-writer/tests/test_journal_entry_indexing.py
git commit -m "feat(journal): persist llm_uncertainty on journal_entry_index"
```

---

### Task 6: Verification, code review, PR

- [ ] **Step 1: Run targeted test suites**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-llm-uncertainty-v2

cd services/orion-llm-gateway && pytest tests/test_llm_uncertainty.py tests/test_llm_backend.py tests/test_handle_chat_meta.py -q
cd ../orion-sql-writer && pytest tests/test_llm_uncertainty_spark_meta.py tests/test_journal_entry_indexing.py -q
cd ../orion-cortex-exec && pytest tests/test_collapse_llm_uncertainty_telemetry.py tests/test_llm_uncertainty_metadata.py -q
cd ../../services/orion-mind && pytest tests/test_llm_uncertainty_telemetry.py tests/test_uncertainty_metacog.py -q
```

Expected: all PASS (PR #608 mind tests still pass unchanged)

- [ ] **Step 2: Code review subagent**

Use **superpowers:requesting-code-review** — dispatch code-reviewer subagent on `feat/llm-uncertainty-v2` vs `origin/main`. Fix every substantive issue in follow-up commits (one fix theme per commit).

- [ ] **Step 3: Write PR report**

Save to `docs/superpowers/pr-reports/2026-05-23-llm-uncertainty-v2-pr.md` covering:
- Summary + architecture diagram
- Opt-in flags table (gateway native, existing OpenAI path)
- Schema migrations
- Test evidence (commands + pass)
- Explicit note: native path is **aligned generation**, not side-probe

- [ ] **Step 4: Push and open PR**

```bash
git push -u origin feat/llm-uncertainty-v2
gh pr create --base main --head feat/llm-uncertainty-v2 --title "feat: LLM uncertainty v2 (native completion, SQL scalars, collapse + journal index)" --body "$(cat <<'EOF'
## Summary
- Native llama.cpp `/apply-template` + `/completion` aligned logprob path (`logprob_probe_mode=native_completion`)
- Queryable `chat_history_log` scalar columns; `spark_meta.llm_uncertainty` remains JSON source of truth
- Collapse Mirror nests `llm_uncertainty` under `state_snapshot.telemetry`
- Journal retrieval index stores `llm_uncertainty` JSON + summary scalars (not `journal_entries`)

## Test plan
- [ ] Gateway: `pytest services/orion-llm-gateway/tests/test_llm_uncertainty.py tests/test_llm_backend.py`
- [ ] SQL writer: `pytest services/orion-sql-writer/tests/test_llm_uncertainty_spark_meta.py tests/test_journal_entry_indexing.py`
- [ ] Cortex: `pytest services/orion-cortex-exec/tests/test_collapse_llm_uncertainty_telemetry.py`
- [ ] Mind regression: `pytest services/orion-mind/tests/test_llm_uncertainty_telemetry.py`

EOF
)"
```

---

## Self-review (plan author checklist)

| Requirement | Task |
|-------------|------|
| Native `/completion` n_probs, not detached side probe | Task 2 routing + explicit non-goals |
| Reuse `summarize_logprob_content`, source `llamacpp_native_completion` | Task 1 |
| OpenAI `/v1/chat/completions` path unchanged by default | Task 2 branch only when all flags set |
| `chat_history_log` scalar columns + JSON preserved | Task 3 |
| Collapse telemetry helper, no new Collapse SQL columns | Task 4 |
| Journal index only, not `journal_entries` | Task 5 |
| `language_surface_stability_not_truth` semantics | Tasks 1, 4 |
| No full token arrays in persistence by default | Task 2 strips `probs` from `raw` when `logprob_summary_only` |
| Worktree isolation | Prerequisites + Task 6 |
| `.env` + `.env_example` + docker-compose + settings | Tasks 2, 3 |
| One commit per task | Each task Step 5/6 |

**Side-probe tagging** documented for future only; not implemented in v2.

---

## Acceptance criteria

1. PR #608 tests still pass without modification (except additive new tests).
2. Gateway tests cover OpenAI chat (existing) and `native_completion` (new).
3. SQL writer tests prove scalar columns + `spark_meta.llm_uncertainty` JSON.
4. Collapse test proves `state_snapshot.telemetry.llm_uncertainty` + semantics key.
5. Journal index test proves JSON + scalar fields; `journal_entries` unchanged.
6. No full per-token logprob arrays in SQL or default `raw` payloads.
