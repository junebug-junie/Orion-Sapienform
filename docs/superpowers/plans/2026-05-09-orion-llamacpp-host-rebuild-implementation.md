# orion-llamacpp-host rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a pinned `orion-llamacpp-host` image with a documented launch-time **thinking policy** (argv + `--jinja` pairing for Qwen3-style off), fast pytest regression tests, and **live** Gates A/B scripts so “thinking off” is proven on a real GGUF—not argv alone.

**Architecture:** Add a small `thinking_policy` module that derives **effective** `reasoning_budget` and **whether `--jinja` is required** from `LlamaCppConfig` (precedence table below). `build_llama_server_cmd_and_env` in `app/main.py` consumes that result, keeps existing `--help` probing, emits deduplicated `--jinja`, logs one summary line, then extends bash validation for `/apply-template` + `/v1/chat/completions` with anchored assertions.

**Tech stack:** Python 3, Pydantic (`LlamaCppConfig`), pytest, bash/curl/jq, Docker (`ghcr.io/ggml-org/llama.cpp`), existing `scripts/test_service.sh` / repo venv per `AGENTS.md`.

**Design source:** `docs/2026-05-09-orion-llamacpp-host-rebuild-design.md`

---

## File map (what changes)

| File | Responsibility |
|------|----------------|
| `services/orion-llamacpp-host/app/thinking_policy.py` | **New.** Pure functions: parse `enable_thinking` from kwargs, compute effective `reasoning_budget`, `require_jinja_for_template_path` booleans, human-readable `thinking_intent` string for logs. |
| `services/orion-llamacpp-host/app/main.py` | Call policy helpers; emit `--jinja` when required; optionally inject effective `--reasoning-budget` before existing append logic; log `thinking_policy_applied` summary; avoid duplicate `--jinja`. |
| `services/orion-llamacpp-host/tests/test_profile_forwarding.py` | Extend expectations (`--jinja` for 8b balanced); add synthetic profile test for kwargs-only false + implicit budget; add test that explicit `reasoning_budget: 1` is not overwritten. |
| `services/orion-llamacpp-host/scripts/verify_qwen3_thinking_off_live.sh` | **New.** Gate A: `curl` `/apply-template` with kwargs false; Gate B: `curl` `/v1/chat/completions`; optional golden file compare. |
| `services/orion-llamacpp-host/scripts/goldens/README.md` | **New.** Documents how to refresh goldens when `LLAMACPP_IMAGE_TAG` or GGUF changes. |
| `services/orion-llamacpp-host/scripts/goldens/qwen3-8b-q4km_thinking_off_completion.json` | **New (optional).** Checked-in minimal JSON: `model`, `messages`, `max_tokens`, `temperature` + expected **substring checks** or full `choices[0].message.content` prefix—populated after first live capture on the chosen pin. |
| `services/orion-llamacpp-host/scripts/validate_llamacpp_upgrade.sh` | Source `verify_qwen3_thinking_off_live.sh` when `VERIFY_QWEN3_THINKING_OFF=1` and `MODEL_QWEN3_PATH` set, or document separate invocation. |
| `services/orion-llamacpp-host/README.md` | Pin parity with Dockerfile; Gates A/B; link to design doc; `LLAMACPP_IMAGE_TAG` build arg; record **observed** `llama-server --version` line for default pin after bump. |
| `services/orion-llamacpp-host/Dockerfile` | Confirm `ARG LLAMACPP_IMAGE_TAG=...` matches README; bump only after live gates pass on candidate tag. |
| `docs/2026-05-09-orion-llamacpp-host-rebuild-design.md` | Optional appendix: recorded `b` build from default image after pin is finalized. |

---

## Precedence table (launch-time, v1)

Implement exactly this in `thinking_policy.py` and test it.

1. If `cfg.reasoning_budget` is **not** `None` → **effective_budget** = `int(cfg.reasoning_budget)` (never auto-overwritten). **`require_jinja`** is true if that value is `0` **or** `chat_template_kwargs` is non-empty (kwargs emission uses the jinja template path).
2. Else if `chat_template_kwargs.enable_thinking` is **`False`** → **effective_budget** = `0` (integer) for emission purposes **only when** `--reasoning-budget` exists in `supported_flags`; if flag missing, effective_budget stays `None` (do not invent CLI args).
3. Else if `chat_template_kwargs.enable_thinking` is **`True`** → leave **effective_budget** `None` unless you later add a positive default (YAGNI: **do not** in v1).
4. Else (`enable_thinking` absent or `None`) → **effective_budget** = `None`.

**`--jinja` emission (dedupe):** After building the list of flags you intend to add, set `need_jinja = False`, then:

- If `--reasoning-format` will be emitted → `need_jinja = True` (existing behavior).
- If `--chat-template-kwargs` will be emitted → `need_jinja = True` (Jinja path for kwargs per upstream PR #13196 server examples).
- If effective or explicit `reasoning_budget` is emitted and value is **`0`** → `need_jinja = True` (PR #13771 discussion: Qwen3 + budget 0 needs jinja on server).

Emit a single `--jinja` before any flag that requires it (order: match current style—today `--jinja` appears immediately before `--reasoning-format`; for kwargs-only profiles, insert `--jinja` immediately before `--chat-template-kwargs` if you need deterministic ordering; `llama-server` generally accepts `--jinja` anywhere—verify once against `--help` if unsure).

---

### Task 1: Thinking policy module (pure logic + unit tests)

**Files:**

- Create: `services/orion-llamacpp-host/app/thinking_policy.py`
- Create: `services/orion-llamacpp-host/tests/test_thinking_policy.py`
- Modify: (none until Task 2)

- [ ] **Step 1: Create `thinking_policy.py`**

```python
# services/orion-llamacpp-host/app/thinking_policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from .profiles import LlamaCppConfig


def _enable_thinking_from_kwargs(kwargs: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not kwargs:
        return None
    if "enable_thinking" not in kwargs:
        return None
    return bool(kwargs["enable_thinking"])


@dataclass(frozen=True)
class ThinkingLaunchPolicy:
    """Launch-time only; per-request overrides are v2 (gateway)."""

    intent_label: str  # e.g. "off_kwargs", "on_kwargs", "explicit_budget_0", "default"
    effective_reasoning_budget: Optional[int]  # value to pass to append_flag if not None
    require_jinja: bool  # OR with reasoning-format path in main


def resolve_thinking_launch_policy(
    cfg: LlamaCppConfig,
    supported_flags: Optional[Set[str]],
) -> ThinkingLaunchPolicy:
    kwargs = cfg.chat_template_kwargs
    et = _enable_thinking_from_kwargs(kwargs)

    has_budget_flag = supported_flags is None or "--reasoning-budget" in supported_flags

    explicit_budget = cfg.reasoning_budget
    if explicit_budget is not None:
        b = int(explicit_budget)
        # Budget 0 needs jinja for Qwen3 (PR #13771 thread); any chat_template_kwargs emission uses jinja path.
        need_jinja = b == 0 or kwargs is not None
        return ThinkingLaunchPolicy(
            intent_label="explicit_budget",
            effective_reasoning_budget=b,
            require_jinja=need_jinja,
        )

    if et is False and has_budget_flag:
        return ThinkingLaunchPolicy(
            intent_label="off_kwargs_implicit_budget",
            effective_reasoning_budget=0,
            require_jinja=True,
        )

    if et is False and not has_budget_flag:
        return ThinkingLaunchPolicy(
            intent_label="off_kwargs_no_budget_flag",
            effective_reasoning_budget=None,
            require_jinja=kwargs is not None,  # kwargs still need jinja when emitted
        )

    if et is True:
        return ThinkingLaunchPolicy(
            intent_label="on_kwargs",
            effective_reasoning_budget=None,
            require_jinja=kwargs is not None,
        )

    return ThinkingLaunchPolicy(
        intent_label="default",
        effective_reasoning_budget=None,
        require_jinja=kwargs is not None,
    )
```

- [ ] **Step 2: Create failing tests**

```python
# services/orion-llamacpp-host/tests/test_thinking_policy.py
from __future__ import annotations

from app.profiles import LlamaCppConfig
from app.thinking_policy import resolve_thinking_launch_policy


def test_explicit_reasoning_budget_not_overwritten():
    cfg = LlamaCppConfig(
        chat_template_kwargs={"enable_thinking": False},
        reasoning_budget=8192,
    )
    pol = resolve_thinking_launch_policy(cfg, {"--reasoning-budget", "--chat-template-kwargs"})
    assert pol.effective_reasoning_budget == 8192
    assert pol.require_jinja is True  # kwargs present → jinja for template path


def test_explicit_reasoning_budget_nonzero_no_kwargs_no_jinja_from_policy():
    cfg = LlamaCppConfig(reasoning_budget=8192)
    pol = resolve_thinking_launch_policy(cfg, {"--reasoning-budget"})
    assert pol.effective_reasoning_budget == 8192
    assert pol.require_jinja is False


def test_implicit_budget_zero_when_kwargs_false():
    cfg = LlamaCppConfig(chat_template_kwargs={"enable_thinking": False})
    pol = resolve_thinking_launch_policy(cfg, {"--reasoning-budget", "--chat-template-kwargs"})
    assert pol.effective_reasoning_budget == 0
    assert pol.require_jinja is True


def test_no_implicit_budget_when_flag_missing():
    cfg = LlamaCppConfig(chat_template_kwargs={"enable_thinking": False})
    pol = resolve_thinking_launch_policy(cfg, {"--chat-template-kwargs"})
    assert pol.effective_reasoning_budget is None
    assert pol.require_jinja is True
```

- [ ] **Step 3: Run tests from repo root**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=services/orion-llamacpp-host ./venv/bin/python -m pytest services/orion-llamacpp-host/tests/test_thinking_policy.py -q --tb=short
```

Expected: **PASS** (module + tests added together is acceptable for greenfield helper; if you strictly want red-first, add tests before module with `import` expecting ImportError).

- [ ] **Step 4: Commit**

```bash
git add services/orion-llamacpp-host/app/thinking_policy.py services/orion-llamacpp-host/tests/test_thinking_policy.py
git commit -m "feat(llamacpp-host): add thinking launch policy helper"
```

---

### Task 2: Integrate policy into `build_llama_server_cmd_and_env`

**Files:**

- Modify: `services/orion-llamacpp-host/app/main.py` (imports + body of `build_llama_server_cmd_and_env` approx. lines 181–298)
- Modify: `services/orion-llamacpp-host/tests/test_profile_forwarding.py`

- [ ] **Step 1: Wire `main.py`**

1. `from .thinking_policy import resolve_thinking_launch_policy`
2. After `append_flag` is defined and `supported_flags` is known, call:

```python
    policy = resolve_thinking_launch_policy(cfg, supported_flags)
```

3. **Replace** the unconditional block:

```python
    if cfg.reasoning_budget is not None:
        append_flag("--reasoning-budget", str(int(cfg.reasoning_budget)))
```

with: if `policy.effective_reasoning_budget is not None`, append `--reasoning-budget` with `str(policy.effective_reasoning_budget)`.

4. **Before** emitting `chat_template_kwargs`, if `policy.require_jinja` or `cfg.reasoning_format is not None` (existing), ensure `--jinja` is emitted once:

```python
    def ensure_jinja() -> None:
        if "--jinja" not in cmd:
            append_flag("--jinja")

    if cfg.reasoning_format is not None:
        ensure_jinja()
    elif policy.require_jinja:
        ensure_jinja()
```

5. Keep existing `if cfg.reasoning_format is not None: append_flag("--jinja"); append_flag("--reasoning-format", ...)` but **refactor** so `ensure_jinja()` is not duplicated: e.g. if `reasoning_format` set, `ensure_jinja()` then `append_flag("--reasoning-format", ...)`.

6. Reorder so **`--jinja`** appears before **`--reasoning-format`** and before **`--chat-template-kwargs`** / **`--reasoning-budget`** as needed.

7. After argv is finalized, log:

```python
    logger.info(
        "thinking_launch_policy intent=%s effective_reasoning_budget=%s jinja_in_cmd=%s",
        policy.intent_label,
        policy.effective_reasoning_budget,
        "--jinja" in cmd,
    )
```

8. **ERROR log adjustment:** when `cfg.chat_template_kwargs` requests keys but `--chat-template-kwargs` missing—unchanged. When **`enable_thinking` is False** and `policy.effective_reasoning_budget is None` and `--reasoning-budget` not in supported_flags, add **WARNING**: thinking may stay on without upgrade.

- [ ] **Step 2: Update `test_qwen3_8b_balanced_profile_forwards_enable_thinking_false`**

After integration, the balanced profile must include **`--jinja`** before kwargs (assert `"--jinja" in cmd` and `cmd.index("--jinja") < cmd.index("--chat-template-kwargs")`).

- [ ] **Step 3: Add argv test for implicit budget**

New test building `LLMProfile` inline with `LlamaCppConfig(chat_template_kwargs={"enable_thinking": False})` only (no `reasoning_budget` field), same monkeypatch flags as 8b test, assert `--reasoning-budget` is `0` and `--jinja` present.

- [ ] **Step 4: Run service tests**

```bash
cd /mnt/scripts/Orion-Sapienform
./scripts/test_service.sh orion-llamacpp-host services/orion-llamacpp-host/tests/test_profile_forwarding.py services/orion-llamacpp-host/tests/test_thinking_policy.py -q --tb=short
```

Expected: **all PASS**.

- [ ] **Step 5: Commit**

```bash
git add services/orion-llamacpp-host/app/main.py services/orion-llamacpp-host/tests/test_profile_forwarding.py
git commit -m "feat(llamacpp-host): apply thinking policy to llama-server argv"
```

---

### Task 3: Dockerfile pin + README parity + version line

**Files:**

- Modify: `services/orion-llamacpp-host/Dockerfile`
- Modify: `services/orion-llamacpp-host/README.md`

- [ ] **Step 1: Candidate image**

Pick a `LLAMACPP_IMAGE_TAG` (e.g. keep `server-cuda-b8660` or newer **only after** Task 4 live gates pass on Atlas-like GPU). Document in README the **exact** `docker build` build-arg.

- [ ] **Step 2: Record version**

After a local `docker build`:

```bash
docker run --rm orion-llamacpp-host:0.1.0 /app/llama-server --version
```

Paste the full line into README under “Pinned binary”.

- [ ] **Step 3: README section “Verifying thinking off”**

Bullets: run pytest; run `verify_qwen3_thinking_off_live.sh`; env vars.

- [ ] **Step 4: Commit**

```bash
git add services/orion-llamacpp-host/Dockerfile services/orion-llamacpp-host/README.md
git commit -m "docs(llamacpp-host): align README pin and verification with Dockerfile"
```

---

### Task 4: Live verification script (Gates A + B)

**Files:**

- Create: `services/orion-llamacpp-host/scripts/verify_qwen3_thinking_off_live.sh`
- Create: `services/orion-llamacpp-host/scripts/goldens/README.md`
- Optionally create: `services/orion-llamacpp-host/scripts/goldens/qwen3-8b-q4km_thinking_off_completion.json`
- Modify: `services/orion-llamacpp-host/scripts/validate_llamacpp_upgrade.sh` (optional hook)

- [ ] **Step 1: Implement Gate A (apply-template)**

Use the **same user message** as in [PR #13196](https://github.com/ggml-org/llama.cpp/pull/13196) (`Give me a short introduction to large language models.`). Start `llama-server` in docker with:

- `-m` mounted GGUF
- `--jinja`
- `--reasoning-budget 0`
- `--chat-template-kwargs '{"enable_thinking":false}'` (or profile-equivalent)
- Minimal `--ctx-size` / `--n-gpu-layers` as in `validate_llamacpp_upgrade.sh`

`curl -fsS http://127.0.0.1:8080/apply-template -H 'Content-Type: application/json' -d '...'`

Parse JSON with `jq -e '.prompt != null'`. **Substring anchors:** assert `.prompt` contains `im_start` / `assistant` per **captured** golden on your pin (first run: `jq .prompt > scripts/goldens/qwen3_apply_template_prompt_snippet.txt` and commit a **short unique substring** from that file, not the whole prompt, to reduce template drift noise—document in `goldens/README.md`).

- [ ] **Step 2: Implement Gate B (chat completions)**

`curl` `POST /v1/chat/completions` with `model` = GGUF basename, `max_tokens` small (e.g. 64), fixed user message `Reply with exactly: LIVE-GATE-B-OK` and **low temperature**.

Assertions (pick one strategy and document in script header):

- **Preferred:** `jq` extracts `content` and **fails** if it matches `grep -E '<think|redacted_thinking'` anchored patterns from **your** golden capture README; **or**
- Check `content` contains `LIVE-GATE-B-OK` and does **not** contain a **stored** forbidden substring file `scripts/goldens/forbidden_thinking_markers.txt` (one pattern per line, from upstream issues/PR—populate after first failing model output review).

- [ ] **Step 3: Wire optional hook in `validate_llamacpp_upgrade.sh`**

```bash
if [[ "${VERIFY_QWEN3_THINKING_OFF:-}" == "1" ]]; then
  bash "$(dirname "$0")/verify_qwen3_thinking_off_live.sh"
fi
```

- [ ] **Step 4: Run live script on GPU host** (human / CI with GPU)

```bash
export MODEL_QWEN3_PATH=/abs/path/to/Qwen_Qwen3-8B-Q4_K_M.gguf
export IMAGE=orion-llamacpp-host:0.1.0
bash services/orion-llamacpp-host/scripts/verify_qwen3_thinking_off_live.sh
```

Expected: exit **0**.

- [ ] **Step 5: Commit**

```bash
git add services/orion-llamacpp-host/scripts/verify_qwen3_thinking_off_live.sh services/orion-llamacpp-host/scripts/goldens/ services/orion-llamacpp-host/scripts/validate_llamacpp_upgrade.sh
git commit -m "test(llamacpp-host): live gates for Qwen3 thinking off"
```

---

### Task 5: Design doc appendix (optional)

**Files:**

- Modify: `docs/2026-05-09-orion-llamacpp-host-rebuild-design.md`

- [ ] **Step 1:** Add section **Appendix: Observed binary** with `llama-server --version` output and image tag after Task 3/4 complete.

- [ ] **Step 2: Commit**

```bash
git add docs/2026-05-09-orion-llamacpp-host-rebuild-design.md
git commit -m "docs: record llamacpp pin version for thinking-off rebuild"
```

---

## Plan self-review

| Spec section | Task coverage |
|--------------|---------------|
| Pinned image + README | Task 3 |
| Thinking policy + jinja pairing | Tasks 1–2 |
| Version evidence | Tasks 3–5 |
| Fast pytest | Tasks 1–2 |
| Live Gates A/B | Task 4 |
| v2 per-request | Out of scope (no gateway tasks) |

**Placeholder scan:** No `TBD` strings in task bodies; golden paths require engineer to paste first capture—steps name the files to create.

**Type consistency:** `ThinkingLaunchPolicy` fields used consistently in `main.py` logging.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-09-orion-llamacpp-host-rebuild-implementation.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration (subagent-driven-development skill).

2. **Inline Execution** — run tasks in this session with checkpoints (executing-plans skill).

**Which approach do you want?**
