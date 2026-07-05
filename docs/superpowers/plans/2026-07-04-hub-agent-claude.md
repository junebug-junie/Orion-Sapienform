# Hub Agent Claude Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Hub mode `agent-claude` that runs one Claude Code harness turn per chat message, streams `stream-json` events live to the chat WebSocket, and routes inference through `fcc-server` → `orion-llm-gateway` without modifying the `free-claude-code` package.

**Architecture:** Hub subprocess bridge on Athena: `websocket_handler` branches on `mode=agent-claude` before cortex/context-exec, calls `prepare_agent_claude_input()` → `fcc_claude_bridge.run_turn()` which spawns `claude -p … --output-format stream-json` with `ANTHROPIC_BASE_URL` pointed at `HUB_FCC_SERVER_URL`. Parsed stdout lines become `claude_step` WS frames; final assistant text becomes `llm_response`. Model tier comes from FCC env key labels (`MODEL_HAIKU`, …) mapped to stable Claude tier ids.

**Tech Stack:** Python 3.10+, FastAPI WebSocket, asyncio subprocess, Pydantic v2 settings, vanilla JS (Hub UI), pytest with `PYTHONPATH=services/orion-hub:.` from repo root.

**Spec:** [`docs/superpowers/specs/2026-07-04-hub-agent-claude-design.md`](../specs/2026-07-04-hub-agent-claude-design.md)

**Branch:** `feat/hub-agent-claude` (worktree recommended: `../Orion-Sapienform-hub-agent-claude`)

---

## File map

| Path | Responsibility |
|------|----------------|
| `services/orion-hub/app/settings.py` | **Modify** — `HUB_AGENT_CLAUDE_*`, `HUB_FCC_*` env keys |
| `services/orion-hub/.env_example` | **Modify** — document agent-claude keys (default disabled) |
| `services/orion-hub/docker-compose.yml` | **Modify** — pass-through env vars only |
| `services/orion-hub/scripts/fcc_env_catalog.py` | **Create** — read `~/.fcc/.env`; expose set model labels + auth token |
| `services/orion-hub/scripts/fcc_model_mapping.py` | **Create** — label → Claude tier model id |
| `services/orion-hub/scripts/agent_claude_input.py` | **Create** — v1 pass-through; v2 slash hook seam |
| `services/orion-hub/scripts/fcc_claude_bridge.py` | **Create** — subprocess spawn, stream-json parse, cancel registry |
| `services/orion-hub/scripts/websocket_handler.py` | **Modify** — branch `agent-claude` before context-exec/cortex |
| `services/orion-hub/scripts/api_routes.py` | **Modify** — HTTP chat parity + `GET /api/fcc-model-labels` |
| `services/orion-hub/templates/index.html` | **Modify** — Mode option + FCC model dropdown (gated) |
| `services/orion-hub/static/js/app.js` | **Modify** — mode spec, payload, WS handler for `claude_step` |
| `services/orion-hub/static/js/agent-claude-trace.js` | **Create** — `appendLiveClaudeStep()` renderer |
| `services/orion-hub/tests/test_fcc_env_catalog.py` | **Create** |
| `services/orion-hub/tests/test_fcc_model_label_mapping.py` | **Create** |
| `services/orion-hub/tests/test_fcc_claude_bridge_parse.py` | **Create** |
| `services/orion-hub/tests/test_agent_claude_input.py` | **Create** |
| `services/orion-hub/tests/test_websocket_agent_claude_routing.py` | **Create** |
| `services/orion-hub/tests/test_fcc_model_labels_api.py` | **Create** |
| `services/orion-hub/scripts/verify_agent_claude_stream_live.py` | **Create** — operator live smoke |
| `services/orion-hub/README.md` | **Modify** — agent-claude mode docs |

**Non-goals:** FCC package changes, bus/cortex integration, slash commands (v2), session resume (v2), running `claude` inside Hub Docker by default.

---

### Task 1: Agent-claude settings

**Files:**
- Modify: `services/orion-hub/app/settings.py`
- Modify: `services/orion-hub/.env_example`
- Modify: `services/orion-hub/docker-compose.yml`

- [ ] **Step 1: Add settings fields**

In `services/orion-hub/app/settings.py`, after the `HUB_AGENT_REPL_ENABLED` block (~line 108):

```python
    # --- Hub Agent Claude (FCC harness in chat) ---
    HUB_AGENT_CLAUDE_ENABLED: bool = Field(
        default=False,
        alias="HUB_AGENT_CLAUDE_ENABLED",
    )
    HUB_FCC_ENV_PATH: str = Field(
        default="~/.fcc/.env",
        alias="HUB_FCC_ENV_PATH",
    )
    HUB_FCC_SERVER_URL: str = Field(
        default="http://127.0.0.1:8082",
        alias="HUB_FCC_SERVER_URL",
    )
    HUB_FCC_AUTH_TOKEN: str = Field(
        default="",
        alias="HUB_FCC_AUTH_TOKEN",
    )
    HUB_AGENT_CLAUDE_BIN: str = Field(
        default="claude",
        alias="HUB_AGENT_CLAUDE_BIN",
    )
    HUB_AGENT_CLAUDE_WORKSPACE: str = Field(
        default="/mnt/scripts/Orion-Sapienform",
        alias="HUB_AGENT_CLAUDE_WORKSPACE",
    )
    HUB_AGENT_CLAUDE_TIMEOUT_SEC: float = Field(
        default=900.0,
        alias="HUB_AGENT_CLAUDE_TIMEOUT_SEC",
    )
    HUB_AGENT_CLAUDE_MAX_CONCURRENT: int = Field(
        default=1,
        alias="HUB_AGENT_CLAUDE_MAX_CONCURRENT",
    )
```

- [ ] **Step 2: Update `.env_example`**

Append after the `HUB_AGENT_REPL_ENABLED` block:

```bash
# Hub Agent Claude — FCC harness in chat (host-network dev; disabled in Docker by default)
HUB_AGENT_CLAUDE_ENABLED=false
HUB_FCC_ENV_PATH=~/.fcc/.env
HUB_FCC_SERVER_URL=http://127.0.0.1:8082
HUB_FCC_AUTH_TOKEN=
HUB_AGENT_CLAUDE_BIN=claude
HUB_AGENT_CLAUDE_WORKSPACE=/mnt/scripts/Orion-Sapienform
HUB_AGENT_CLAUDE_TIMEOUT_SEC=900
HUB_AGENT_CLAUDE_MAX_CONCURRENT=1
```

- [ ] **Step 3: Update docker-compose environment block**

In `services/orion-hub/docker-compose.yml`, after `HUB_AGENT_REPL_ENABLED`:

```yaml
      - HUB_AGENT_CLAUDE_ENABLED=${HUB_AGENT_CLAUDE_ENABLED:-false}
      - HUB_FCC_ENV_PATH=${HUB_FCC_ENV_PATH:-~/.fcc/.env}
      - HUB_FCC_SERVER_URL=${HUB_FCC_SERVER_URL:-http://127.0.0.1:8082}
      - HUB_FCC_AUTH_TOKEN=${HUB_FCC_AUTH_TOKEN:-}
      - HUB_AGENT_CLAUDE_BIN=${HUB_AGENT_CLAUDE_BIN:-claude}
      - HUB_AGENT_CLAUDE_WORKSPACE=${HUB_AGENT_CLAUDE_WORKSPACE:-/mnt/scripts/Orion-Sapienform}
      - HUB_AGENT_CLAUDE_TIMEOUT_SEC=${HUB_AGENT_CLAUDE_TIMEOUT_SEC:-900}
      - HUB_AGENT_CLAUDE_MAX_CONCURRENT=${HUB_AGENT_CLAUDE_MAX_CONCURRENT:-1}
```

- [ ] **Step 4: Sync local env**

Run from repo root:

```bash
python scripts/sync_local_env_from_example.py
```

Expected: keys added to `services/orion-hub/.env` (not staged).

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/app/settings.py services/orion-hub/.env_example services/orion-hub/docker-compose.yml
git commit -m "feat(hub): add agent-claude FCC harness settings"
```

---

### Task 2: FCC env catalog

**Files:**
- Create: `services/orion-hub/scripts/fcc_env_catalog.py`
- Create: `services/orion-hub/tests/test_fcc_env_catalog.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-hub/tests/test_fcc_env_catalog.py`:

```python
"""FCC env catalog: model labels from ~/.fcc/.env fixture."""
from __future__ import annotations

from pathlib import Path

from scripts.fcc_env_catalog import (
    FCC_MODEL_ENV_KEYS,
    load_fcc_env,
    model_labels_from_env,
    resolve_auth_token,
)


def test_model_labels_only_for_set_keys(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MODEL=llamacpp/chat\n"
        "MODEL_OPUS=\n"
        "MODEL_HAIKU=llamacpp/quick\n"
        "ANTHROPIC_AUTH_TOKEN=fixture-token\n",
        encoding="utf-8",
    )
    env = load_fcc_env(env_file)
    labels = model_labels_from_env(env)
    assert labels == ["MODEL", "MODEL_HAIKU"]
    assert resolve_auth_token(env, override="") == "fixture-token"
    assert resolve_auth_token(env, override="override") == "override"


def test_model_labels_empty_when_no_keys(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=1\n", encoding="utf-8")
    assert model_labels_from_env(load_fcc_env(env_file)) == []


def test_catalog_keys_are_stable() -> None:
    assert "MODEL" in FCC_MODEL_ENV_KEYS
    assert "MODEL_HAIKU" in FCC_MODEL_ENV_KEYS
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_fcc_env_catalog.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fcc_env_catalog'`

- [ ] **Step 3: Write minimal implementation**

Create `services/orion-hub/scripts/fcc_env_catalog.py`:

```python
"""Read FCC operator env for Hub agent-claude model label catalog."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

FCC_MODEL_ENV_KEYS: tuple[str, ...] = (
    "MODEL",
    "MODEL_OPUS",
    "MODEL_SONNET",
    "MODEL_HAIKU",
)


def expand_env_path(raw: str) -> Path:
    return Path(os.path.expanduser(str(raw or "").strip() or "~/.fcc/.env"))


def load_fcc_env(path: Path | str) -> Dict[str, str]:
    p = Path(path)
    if not p.is_file():
        return {}
    out: Dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def model_labels_from_env(env: Dict[str, str]) -> List[str]:
    labels: List[str] = []
    for key in FCC_MODEL_ENV_KEYS:
        if str(env.get(key) or "").strip():
            labels.append(key)
    return labels


def resolve_auth_token(env: Dict[str, str], *, override: str = "") -> str:
    token = str(override or "").strip()
    if token:
        return token
    return str(env.get("ANTHROPIC_AUTH_TOKEN") or "").strip()


def catalog_from_settings(*, env_path: str, auth_override: str = "") -> dict:
    from scripts.settings import settings

    path = expand_env_path(env_path or settings.HUB_FCC_ENV_PATH)
    env = load_fcc_env(path)
    return {
        "enabled": bool(settings.HUB_AGENT_CLAUDE_ENABLED),
        "env_path": str(path),
        "labels": model_labels_from_env(env),
        "default_label": model_labels_from_env(env)[0] if model_labels_from_env(env) else None,
        "auth_token_configured": bool(resolve_auth_token(env, override=auth_override or settings.HUB_FCC_AUTH_TOKEN)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_fcc_env_catalog.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/fcc_env_catalog.py services/orion-hub/tests/test_fcc_env_catalog.py
git commit -m "feat(hub): add FCC env catalog for agent-claude model labels"
```

---

### Task 3: Model label → Claude tier mapping

**Files:**
- Create: `services/orion-hub/scripts/fcc_model_mapping.py`
- Create: `services/orion-hub/tests/test_fcc_model_label_mapping.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-hub/tests/test_fcc_model_label_mapping.py`:

```python
from __future__ import annotations

import pytest

from scripts.fcc_model_mapping import (
    DEFAULT_FCC_MODEL_LABEL,
    label_to_claude_model_id,
)


def test_label_to_tier_model_ids() -> None:
    assert label_to_claude_model_id("MODEL") == "claude-sonnet-4-20250514"
    assert label_to_claude_model_id("MODEL_OPUS") == "claude-opus-4-20250514"
    assert label_to_claude_model_id("MODEL_SONNET") == "claude-sonnet-4-20250514"
    assert label_to_claude_model_id("MODEL_HAIKU") == "claude-haiku-4-20250514"


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError, match="unknown fcc model label"):
        label_to_claude_model_id("MODEL_GHOST")


def test_default_label_is_model() -> None:
    assert DEFAULT_FCC_MODEL_LABEL == "MODEL"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_fcc_model_label_mapping.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

Create `services/orion-hub/scripts/fcc_model_mapping.py`:

```python
"""Map FCC env key labels to stable Claude tier model ids for claude CLI --model."""
from __future__ import annotations

DEFAULT_FCC_MODEL_LABEL = "MODEL"

_LABEL_TO_CLAUDE_MODEL: dict[str, str] = {
    "MODEL": "claude-sonnet-4-20250514",
    "MODEL_OPUS": "claude-opus-4-20250514",
    "MODEL_SONNET": "claude-sonnet-4-20250514",
    "MODEL_HAIKU": "claude-haiku-4-20250514",
}


def label_to_claude_model_id(label: str) -> str:
    key = str(label or "").strip()
    if key not in _LABEL_TO_CLAUDE_MODEL:
        raise ValueError(f"unknown fcc model label: {label!r}")
    return _LABEL_TO_CLAUDE_MODEL[key]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_fcc_model_label_mapping.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/fcc_model_mapping.py services/orion-hub/tests/test_fcc_model_label_mapping.py
git commit -m "feat(hub): map FCC model labels to Claude tier ids"
```

---

### Task 4: Agent-claude input seam (v1 pass-through)

**Files:**
- Create: `services/orion-hub/scripts/agent_claude_input.py`
- Create: `services/orion-hub/tests/test_agent_claude_input.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-hub/tests/test_agent_claude_input.py`:

```python
from __future__ import annotations

from scripts.agent_claude_input import TurnRequest, prepare_agent_claude_input


def test_v1_pass_through_prompt() -> None:
    result = prepare_agent_claude_input("  explain websocket_handler  ")
    assert isinstance(result, TurnRequest)
    assert result.prompt == "explain websocket_handler"


def test_v1_empty_becomes_empty_string() -> None:
    result = prepare_agent_claude_input("   ")
    assert result.prompt == ""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_agent_claude_input.py -v
```

Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Create `services/orion-hub/scripts/agent_claude_input.py`:

```python
"""Prepare Hub agent-claude turn input. v2 adds slash-command dispatch."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TurnRequest:
    prompt: str


def prepare_agent_claude_input(text: str) -> TurnRequest:
    return TurnRequest(prompt=str(text or "").strip())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_agent_claude_input.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/agent_claude_input.py services/orion-hub/tests/test_agent_claude_input.py
git commit -m "feat(hub): add agent-claude input pass-through seam"
```

---

### Task 5: Stream-json parse + final text extraction

**Files:**
- Create: `services/orion-hub/scripts/fcc_claude_bridge.py` (parse helpers only first)
- Create: `services/orion-hub/tests/test_fcc_claude_bridge_parse.py`
- Test fixture: `services/orion-hub/tests/fixtures/fcc_claude_stream.jsonl`

- [ ] **Step 1: Add fixture**

Create `services/orion-hub/tests/fixtures/fcc_claude_stream.jsonl`:

```text
{"type":"system","subtype":"init","session_id":"sess-abc"}
{"type":"assistant","message":{"content":[{"type":"text","text":"Checking files."}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read","input":{}}]}}
{"type":"result","result":"Here are the hub scripts.","session_id":"sess-abc","duration_ms":1200}
```

- [ ] **Step 2: Write the failing test**

Create `services/orion-hub/tests/test_fcc_claude_bridge_parse.py`:

```python
from __future__ import annotations

from pathlib import Path

from scripts.fcc_claude_bridge import (
    build_step_frame,
    extract_final_from_stream_event,
    parse_stream_json_line,
    parse_stream_json_lines,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "fcc_claude_stream.jsonl"


def test_parse_stream_json_line_skips_blank_and_raw_fallback() -> None:
    assert parse_stream_json_line("") is None
    assert parse_stream_json_line("{not json") == {"type": "raw", "content": "{not json"}


def test_parse_stream_json_lines_and_final_text() -> None:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    events = parse_stream_json_lines(lines)
    assert len(events) == 4
    assert events[0]["type"] == "system"
    final_text, session_id, duration_ms = extract_final_from_stream_event(events[-1], accumulated="ignored")
    assert final_text == "Here are the hub scripts."
    assert session_id == "sess-abc"
    assert duration_ms == 1200


def test_build_step_frame_shape() -> None:
    raw = {"type": "assistant", "message": {"content": []}}
    step = build_step_frame(raw)
    assert step == {"type": "assistant", "raw": raw}
```

- [ ] **Step 3: Run test to verify it fails**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_fcc_claude_bridge_parse.py -v
```

Expected: FAIL

- [ ] **Step 4: Write parse helpers in bridge module**

Create `services/orion-hub/scripts/fcc_claude_bridge.py` with parse section (subprocess added in Task 6):

```python
"""Spawn Claude Code harness turns for Hub agent-claude mode."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


def parse_stream_json_line(line: str) -> Optional[Dict[str, Any]]:
    stripped = str(line or "").strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {"type": "raw", "content": stripped}
    if not isinstance(parsed, dict):
        return {"type": "raw", "content": stripped}
    return parsed


def parse_stream_json_lines(lines: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in lines:
        parsed = parse_stream_json_line(line)
        if parsed is not None:
            out.append(parsed)
    return out


def build_step_frame(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": str(raw.get("type") or "unknown"), "raw": raw}


def _text_blocks_from_assistant(event: Dict[str, Any]) -> str:
    message = event.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def extract_final_from_stream_event(
    event: Dict[str, Any],
    *,
    accumulated: str,
) -> Tuple[str, Optional[str], Optional[int]]:
    etype = str(event.get("type") or "")
    session_id = event.get("session_id")
    duration_ms = event.get("duration_ms")
    dur = int(duration_ms) if isinstance(duration_ms, (int, float)) else None
    sid = str(session_id) if session_id else None

    if etype == "result":
        result = event.get("result")
        if isinstance(result, str) and result.strip():
            return result.strip(), sid, dur
        if isinstance(result, dict):
            text = str(result.get("result") or result.get("text") or "").strip()
            if text:
                return text, sid, dur

    assistant_text = _text_blocks_from_assistant(event)
    if assistant_text.strip():
        return assistant_text.strip(), sid, dur

    return accumulated, sid, dur
```

- [ ] **Step 5: Run test to verify it passes**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_fcc_claude_bridge_parse.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub/scripts/fcc_claude_bridge.py services/orion-hub/tests/test_fcc_claude_bridge_parse.py services/orion-hub/tests/fixtures/fcc_claude_stream.jsonl
git commit -m "feat(hub): add FCC Claude stream-json parse helpers"
```

---

### Task 6: FCC Claude bridge subprocess + cancel registry

**Files:**
- Modify: `services/orion-hub/scripts/fcc_claude_bridge.py`
- Create: `services/orion-hub/tests/test_fcc_claude_bridge_run.py`

- [ ] **Step 1: Write the failing test (mock subprocess)**

Create `services/orion-hub/tests/test_fcc_claude_bridge_run.py`:

```python
from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

from scripts import fcc_claude_bridge as bridge


class _FakeStream:
    def __init__(self, lines: List[bytes]) -> None:
        self._lines = list(lines)
        self._idx = 0

    async def readline(self) -> bytes:
        if self._idx >= len(self._lines):
            return b""
        line = self._lines[self._idx]
        self._idx += 1
        await asyncio.sleep(0)
        return line


class _FakeProc:
    def __init__(self, stdout_lines: List[str], returncode: int = 0) -> None:
        self.stdout = _FakeStream([ln.encode("utf-8") + b"\n" for ln in stdout_lines])
        self.returncode = returncode
        self._terminated = False

    def kill(self) -> None:
        self._terminated = True

    async def wait(self) -> int:
        return self.returncode


@pytest.mark.asyncio
async def test_run_turn_yields_steps_and_final(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        '{"type":"assistant","message":{"content":[{"type":"text","text":"Hi"}]}}',
        '{"type":"result","result":"Done.","session_id":"s1","duration_ms":50}',
    ]

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        return _FakeProc(lines)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(bridge, "_preflight_fcc_server", lambda *a, **k: None)

    events = []
    async for ev in bridge.run_turn(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-1",
        workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082",
        auth_token="tok",
        claude_bin="claude",
        timeout_sec=30.0,
    ):
        events.append(ev)

    kinds = [e["type"] for e in events]
    assert kinds.count("step") >= 1
    assert kinds[-1] == "final"
    assert events[-1]["llm_response"] == "Done."
    assert events[-1]["metadata"]["claude_session_id"] == "s1"


@pytest.mark.asyncio
async def test_cancel_turn_sigterms_active(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _FakeProc([], returncode=-15)

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        bridge._register_process("corr-x", proc)  # type: ignore[attr-defined]
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    assert await bridge.cancel_turn("corr-x") is True
    assert proc._terminated is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_fcc_claude_bridge_run.py -v
```

Expected: FAIL — `run_turn` / `cancel_turn` not defined

- [ ] **Step 3: Implement subprocess bridge**

Append to `services/orion-hub/scripts/fcc_claude_bridge.py`:

```python
import asyncio
import logging
import os
import time
import urllib.error
import urllib.request
from typing import AsyncIterator, Dict, Optional

from scripts.fcc_env_catalog import load_fcc_env, resolve_auth_token
from scripts.fcc_model_mapping import DEFAULT_FCC_MODEL_LABEL, label_to_claude_model_id

logger = logging.getLogger("orion-hub.fcc_claude_bridge")

_ACTIVE: Dict[str, asyncio.subprocess.Process] = {}


def _register_process(correlation_id: str, proc: asyncio.subprocess.Process) -> None:
    _ACTIVE[str(correlation_id)] = proc


def _unregister_process(correlation_id: str) -> None:
    _ACTIVE.pop(str(correlation_id), None)


def active_turns() -> list[dict]:
    return [{"correlation_id": cid} for cid in sorted(_ACTIVE.keys())]


def _preflight_fcc_server(url: str, *, timeout_sec: float = 3.0) -> None:
    health_url = str(url or "").rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_sec) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"fcc-server health returned {resp.status}")
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        raise RuntimeError(f"fcc-server unreachable at {url}: {exc}") from exc


def _build_subprocess_env(*, fcc_server_url: str, auth_token: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = str(fcc_server_url).rstrip("/")
    env["ANTHROPIC_AUTH_TOKEN"] = auth_token
    env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    env["TERM"] = "dumb"
    return env


async def run_turn(
    *,
    prompt: str,
    fcc_model_label: str,
    correlation_id: str,
    workspace: str,
    fcc_server_url: str,
    auth_token: str,
    claude_bin: str,
    timeout_sec: float,
) -> AsyncIterator[Dict[str, object]]:
    from scripts.settings import settings

    if len(_ACTIVE) >= int(settings.HUB_AGENT_CLAUDE_MAX_CONCURRENT):
        yield {
            "type": "error",
            "error": "Another agent-claude turn is already running",
            "error_code": "fcc_claude_max_concurrent",
        }
        return

    label = str(fcc_model_label or DEFAULT_FCC_MODEL_LABEL).strip() or DEFAULT_FCC_MODEL_LABEL
    try:
        model_id = label_to_claude_model_id(label)
    except ValueError as exc:
        yield {"type": "error", "error": str(exc), "error_code": "fcc_claude_bad_model_label"}
        return

    try:
        _preflight_fcc_server(fcc_server_url)
    except RuntimeError as exc:
        yield {"type": "error", "error": str(exc), "error_code": "fcc_claude_spawn_failed"}
        return

    argv = [
        claude_bin,
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--dangerously-skip-permissions",
        "--verbose",
        "--model",
        model_id,
    ]
    started = time.monotonic()
    proc: Optional[asyncio.subprocess.Process] = None
    accumulated = ""
    claude_session_id: Optional[str] = None
    exit_code = 1

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=workspace,
            env=_build_subprocess_env(fcc_server_url=fcc_server_url, auth_token=auth_token),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _register_process(correlation_id, proc)
        assert proc.stdout is not None

        while True:
            try:
                line_bytes = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                proc.kill()
                yield {
                    "type": "error",
                    "error": f"agent-claude turn timed out after {timeout_sec}s",
                    "error_code": "fcc_claude_timeout",
                }
                return

            if not line_bytes:
                break

            parsed = parse_stream_json_line(line_bytes.decode("utf-8", errors="replace"))
            if parsed is None:
                continue

            step = build_step_frame(parsed)
            yield {"type": "step", "step": step}

            text, sid, _dur = extract_final_from_stream_event(parsed, accumulated=accumulated)
            if text:
                accumulated = text
            if sid:
                claude_session_id = sid

        exit_code = await proc.wait()
    except FileNotFoundError:
        yield {
            "type": "error",
            "error": f"claude binary not found: {claude_bin!r} (check PATH or HUB_AGENT_CLAUDE_BIN)",
            "error_code": "fcc_claude_spawn_failed",
        }
        return
    finally:
        _unregister_process(correlation_id)

    duration_ms = int((time.monotonic() - started) * 1000)
    metadata = {
        "fcc_model_label": label,
        "claude_session_id": claude_session_id,
        "duration_ms": duration_ms,
        "exit_code": exit_code,
    }

    if exit_code != 0:
        yield {
            "type": "error",
            "error": f"claude exited with code {exit_code}",
            "error_code": "fcc_claude_nonzero_exit",
            "metadata": metadata,
            "llm_response": accumulated,
        }
        return

    yield {
        "type": "final",
        "llm_response": accumulated,
        "metadata": metadata,
    }


async def cancel_turn(correlation_id: str) -> bool:
    proc = _ACTIVE.get(str(correlation_id))
    if proc is None:
        return False
    proc.kill()
    _unregister_process(correlation_id)
    return True
```

Add helper used by websocket for settings-backed turn:

```python
async def run_turn_from_settings(
    *,
    prompt: str,
    fcc_model_label: str,
    correlation_id: str,
) -> AsyncIterator[Dict[str, object]]:
    from scripts.fcc_env_catalog import expand_env_path, load_fcc_env, resolve_auth_token
    from scripts.settings import settings

    env = load_fcc_env(expand_env_path(settings.HUB_FCC_ENV_PATH))
    token = resolve_auth_token(env, override=settings.HUB_FCC_AUTH_TOKEN)
    async for event in run_turn(
        prompt=prompt,
        fcc_model_label=fcc_model_label,
        correlation_id=correlation_id,
        workspace=settings.HUB_AGENT_CLAUDE_WORKSPACE,
        fcc_server_url=settings.HUB_FCC_SERVER_URL,
        auth_token=token,
        claude_bin=settings.HUB_AGENT_CLAUDE_BIN,
        timeout_sec=float(settings.HUB_AGENT_CLAUDE_TIMEOUT_SEC),
    ):
        yield event
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_fcc_claude_bridge_run.py services/orion-hub/tests/test_fcc_claude_bridge_parse.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/fcc_claude_bridge.py services/orion-hub/tests/test_fcc_claude_bridge_run.py
git commit -m "feat(hub): add FCC Claude subprocess bridge with cancel registry"
```

---

### Task 7: FCC model labels API

**Files:**
- Modify: `services/orion-hub/scripts/api_routes.py`
- Create: `services/orion-hub/tests/test_fcc_model_labels_api.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-hub/tests/test_fcc_model_labels_api.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_fcc_model_labels_disabled_by_default(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.settings import settings

    monkeypatch.setattr(settings, "HUB_AGENT_CLAUDE_ENABLED", False, raising=False)
    resp = client.get("/api/fcc-model-labels")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert "labels" in body


def test_fcc_model_labels_reads_fixture_env(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.settings import settings

    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_HAIKU=llamacpp/quick\n", encoding="utf-8")
    monkeypatch.setattr(settings, "HUB_AGENT_CLAUDE_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "HUB_FCC_ENV_PATH", str(env_file), raising=False)
    monkeypatch.setattr(settings, "HUB_FCC_SERVER_URL", "http://127.0.0.1:59999", raising=False)

    resp = client.get("/api/fcc-model-labels")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["labels"] == ["MODEL_HAIKU"]
    assert body["fcc_server_ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_fcc_model_labels_api.py -v
```

Expected: FAIL — 404

- [ ] **Step 3: Add route**

In `services/orion-hub/scripts/api_routes.py`, near `@router.get("/api/llm-routes")`:

```python
@router.get("/api/fcc-model-labels")
async def api_fcc_model_labels():
    from scripts.fcc_env_catalog import catalog_from_settings
    from scripts.fcc_claude_bridge import _preflight_fcc_server
    from scripts.settings import settings

    payload = catalog_from_settings(
        env_path=settings.HUB_FCC_ENV_PATH,
        auth_override=settings.HUB_FCC_AUTH_TOKEN,
    )
    fcc_server_ok = False
    try:
        _preflight_fcc_server(settings.HUB_FCC_SERVER_URL, timeout_sec=2.0)
        fcc_server_ok = True
    except RuntimeError:
        fcc_server_ok = False
    payload["fcc_server_ok"] = fcc_server_ok
    return payload
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_fcc_model_labels_api.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/api_routes.py services/orion-hub/tests/test_fcc_model_labels_api.py
git commit -m "feat(hub): expose FCC model label catalog API"
```

---

### Task 8: WebSocket routing for agent-claude

**Files:**
- Modify: `services/orion-hub/scripts/websocket_handler.py`
- Create: `services/orion-hub/tests/test_websocket_agent_claude_routing.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-hub/tests/test_websocket_agent_claude_routing.py`:

```python
"""WebSocket agent-claude mode must branch to FCC bridge, not context-exec."""
from __future__ import annotations

from pathlib import Path


HUB_ROOT = Path(__file__).resolve().parents[1]
WS_PATH = HUB_ROOT / "scripts" / "websocket_handler.py"


def test_websocket_handler_imports_agent_claude_bridge() -> None:
    source = WS_PATH.read_text(encoding="utf-8")
    assert "run_turn_from_settings" in source
    assert "prepare_agent_claude_input" in source
    assert 'mode == "agent-claude"' in source or "agent-claude" in source


def test_websocket_handler_emits_claude_step_kind() -> None:
    source = WS_PATH.read_text(encoding="utf-8")
    assert '"claude_step"' in source or "'claude_step'" in source
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_websocket_agent_claude_routing.py -v
```

Expected: FAIL

- [ ] **Step 3: Add routing branch**

At top of `websocket_handler.py`, add imports:

```python
from scripts.agent_claude_input import prepare_agent_claude_input
from scripts.fcc_claude_bridge import run_turn_from_settings
from scripts.fcc_env_catalog import catalog_from_settings
from scripts.fcc_model_mapping import DEFAULT_FCC_MODEL_LABEL
```

Add helper before `websocket_endpoint`:

```python
def _agent_claude_enabled() -> bool:
    return bool(getattr(settings, "HUB_AGENT_CLAUDE_ENABLED", False))


async def _run_agent_claude_turn_ws(
    *,
    websocket: WebSocket,
    data: Dict[str, Any],
    transcript: str,
    trace_id: str,
    biometrics_cache: Any,
) -> Optional[Dict[str, Any]]:
    """Run FCC harness turn; stream claude_step frames. Returns final payload or None if error sent."""
    if not _agent_claude_enabled():
        await websocket.send_json(
            await _with_biometrics(
                {
                    "error": "Agent Claude mode is disabled",
                    "error_code": "agent_claude_disabled",
                    "mode": "agent-claude",
                    "correlation_id": trace_id,
                },
                cache=biometrics_cache,
            )
        )
        return None

    turn = prepare_agent_claude_input(transcript)
    catalog = catalog_from_settings(env_path=settings.HUB_FCC_ENV_PATH, auth_override=settings.HUB_FCC_AUTH_TOKEN)
    fcc_label = str(data.get("fcc_model_label") or catalog.get("default_label") or DEFAULT_FCC_MODEL_LABEL)

    final_text = ""
    final_meta: Dict[str, Any] = {}
    async for event in run_turn_from_settings(
        prompt=turn.prompt,
        fcc_model_label=fcc_label,
        correlation_id=trace_id,
    ):
        etype = str(event.get("type") or "")
        if etype == "step":
            step = event.get("step") if isinstance(event.get("step"), dict) else {}
            await websocket.send_json(
                await _with_biometrics(
                    {
                        "kind": "claude_step",
                        "correlation_id": trace_id,
                        "mode": "agent-claude",
                        "step": step,
                    },
                    cache=biometrics_cache,
                )
            )
        elif etype == "error":
            partial = str(event.get("llm_response") or "")
            await websocket.send_json(
                await _with_biometrics(
                    {
                        "error": str(event.get("error") or "agent-claude failed"),
                        "error_code": str(event.get("error_code") or "fcc_claude_nonzero_exit"),
                        "mode": "agent-claude",
                        "correlation_id": trace_id,
                        "llm_response": partial or None,
                        "metadata": event.get("metadata"),
                    },
                    cache=biometrics_cache,
                )
            )
            return None
        elif etype == "final":
            final_text = str(event.get("llm_response") or "")
            final_meta = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}

    return {"llm_response": final_text, "metadata": final_meta, "fcc_model_label": fcc_label}
```

Replace the block starting at `used_context_exec_lane = should_use_context_exec_agent_lane(chat_req)` (~line 1010) with:

```python
                used_context_exec_lane = False
                resp = None
                if str(mode or "").strip().lower() == "agent-claude":
                    agent_claude_out = await _run_agent_claude_turn_ws(
                        websocket=websocket,
                        data=data,
                        transcript=transcript or "",
                        trace_id=trace_id,
                        biometrics_cache=biometrics_cache,
                    )
                    if agent_claude_out is None:
                        continue
                    orion_response_text = str(agent_claude_out.get("llm_response") or "")
                    agent_trace = None
                    cortex_result_dump = {}
                    route_debug = route_debug if isinstance(route_debug, dict) else {}
                    route_debug["agent_claude"] = {
                        "fcc_model_label": agent_claude_out.get("fcc_model_label"),
                        **(agent_claude_out.get("metadata") or {}),
                    }
                else:
                    used_context_exec_lane = should_use_context_exec_agent_lane(chat_req)
                    if used_context_exec_lane:
                        # existing context-exec block unchanged (step_queue, drain_task, run_hub_agent_via_context_exec, …)
                        ...
                    else:
                        resp = await cortex_client.chat(chat_req, correlation_id=trace_id)
                        # existing cortex response extraction unchanged
                        ...
```

Keep the existing `if used_context_exec_lane:` / `else:` cortex branches inside the `else` arm above; only the outer `agent-claude` check is new.

Also map trace verb (~line 573):

```python
            elif mode == "agent-claude":
                trace_verb = "task_execution"
```

Ensure final WS response includes `"mode": "agent-claude"` and metadata — locate the existing `llm_response` send block and confirm `mode` variable flows through (it already comes from client payload).

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_websocket_agent_claude_routing.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/websocket_handler.py services/orion-hub/tests/test_websocket_agent_claude_routing.py
git commit -m "feat(hub): route agent-claude WebSocket turns through FCC bridge"
```

---

### Task 9: HTTP chat parity for agent-claude

**Files:**
- Modify: `services/orion-hub/scripts/api_routes.py`
- Modify: `services/orion-hub/tests/test_fcc_model_labels_api.py` (or new `test_http_agent_claude.py`)

- [ ] **Step 1: Write the failing test**

Create `services/orion-hub/tests/test_http_agent_claude.py`:

```python
from __future__ import annotations

import pytest

from scripts.agent_claude_input import prepare_agent_claude_input


@pytest.mark.asyncio
async def test_http_agent_claude_collects_events(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import api_routes

    async def fake_run_turn_from_settings(**kwargs):
        yield {"type": "step", "step": {"type": "assistant", "raw": {"type": "assistant"}}}
        yield {"type": "final", "llm_response": "HTTP done.", "metadata": {"exit_code": 0}}

    monkeypatch.setattr(api_routes, "run_turn_from_settings", fake_run_turn_from_settings)
    monkeypatch.setattr(api_routes.settings, "HUB_AGENT_CLAUDE_ENABLED", True, raising=False)

    result = await api_routes._run_agent_claude_http(
        prompt="hello",
        fcc_model_label="MODEL_HAIKU",
        correlation_id="corr-http",
    )
    assert result["llm_response"] == "HTTP done."
    assert len(result["claude_steps"]) == 1
    assert prepare_agent_claude_input("x").prompt == "x"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_http_agent_claude.py -v
```

Expected: FAIL

- [ ] **Step 3: Add HTTP helper + branch**

In `api_routes.py` imports:

```python
from scripts.agent_claude_input import prepare_agent_claude_input
from scripts.fcc_claude_bridge import run_turn_from_settings
from scripts.fcc_env_catalog import catalog_from_settings
from scripts.fcc_model_mapping import DEFAULT_FCC_MODEL_LABEL
```

Add helper:

```python
async def _run_agent_claude_http(
    *,
    prompt: str,
    fcc_model_label: str,
    correlation_id: str,
) -> dict:
    if not bool(settings.HUB_AGENT_CLAUDE_ENABLED):
        return {
            "error": "Agent Claude mode is disabled",
            "error_code": "agent_claude_disabled",
            "mode": "agent-claude",
            "correlation_id": correlation_id,
        }

    turn = prepare_agent_claude_input(prompt)
    steps = []
    async for event in run_turn_from_settings(
        prompt=turn.prompt,
        fcc_model_label=fcc_model_label,
        correlation_id=correlation_id,
    ):
        etype = str(event.get("type") or "")
        if etype == "step":
            steps.append(event.get("step"))
        elif etype == "error":
            return {
                "error": str(event.get("error") or "agent-claude failed"),
                "error_code": str(event.get("error_code") or "fcc_claude_nonzero_exit"),
                "mode": "agent-claude",
                "correlation_id": correlation_id,
                "claude_steps": steps,
                "llm_response": str(event.get("llm_response") or ""),
                "metadata": event.get("metadata"),
            }
        elif etype == "final":
            return {
                "mode": "agent-claude",
                "correlation_id": correlation_id,
                "llm_response": str(event.get("llm_response") or ""),
                "text": str(event.get("llm_response") or ""),
                "claude_steps": steps,
                "metadata": event.get("metadata"),
                "fcc_model_label": fcc_model_label,
            }
    return {
        "error": "agent-claude produced no final frame",
        "error_code": "fcc_claude_nonzero_exit",
        "mode": "agent-claude",
        "correlation_id": correlation_id,
        "claude_steps": steps,
    }
```

In shared chat core (before `if should_use_context_exec_agent_lane(req):` ~line 2240):

```python
    if str(mode or "").strip().lower() == "agent-claude":
        catalog = catalog_from_settings(
            env_path=settings.HUB_FCC_ENV_PATH,
            auth_override=settings.HUB_FCC_AUTH_TOKEN,
        )
        fcc_label = str(
            payload.get("fcc_model_label")
            or catalog.get("default_label")
            or DEFAULT_FCC_MODEL_LABEL
        )
        return await _run_agent_claude_http(
            prompt=user_prompt,
            fcc_model_label=fcc_label,
            correlation_id=corr_id,
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_http_agent_claude.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/api_routes.py services/orion-hub/tests/test_http_agent_claude.py
git commit -m "feat(hub): add HTTP parity for agent-claude mode"
```

---

### Task 10: Hub UI — mode + FCC model dropdown

**Files:**
- Modify: `services/orion-hub/templates/index.html`
- Modify: `services/orion-hub/static/js/app.js`

- [ ] **Step 1: Add HTML controls**

In `templates/index.html`, after the Agent mode option (~line 250):

```html
              <option value="agent_claude" id="hubModeAgentClaudeOption" hidden>Agent Claude</option>
```

After the Compute dropdown block (~line 264), add:

```html
          <div id="hubFccModelRow" class="flex items-center gap-2 pt-1 border-t border-gray-700/80 hidden">
            <label for="hubFccModelSelect" class="uppercase text-gray-500 font-bold text-[10px] shrink-0">FCC Model</label>
            <select
              id="hubFccModelSelect"
              class="flex-1 min-w-0 bg-gray-900/80 text-gray-100 text-xs rounded border border-gray-700 px-2 py-1.5 focus:outline-none focus:border-indigo-500 font-mono"
              aria-label="FCC model env label"
              title="FCC ~/.fcc/.env key label (MODEL, MODEL_HAIKU, …)"
            >
              <option value="">loading…</option>
            </select>
          </div>
```

In the script tags section where `agent-trace.js` is loaded, add:

```html
    <script src="/static/js/agent-claude-trace.js?v={{ build_id }}"></script>
```

- [ ] **Step 2: Extend `app.js` mode specs**

In `HUB_MODE_SPECS` (~line 9368):

```javascript
    agent_claude: { mode: 'agent-claude', verb: null, lane: null, label: 'Agent Claude', skipComputeConfirm: true },
```

In `applyHubModeSelection`, after setting `currentMode`:

```javascript
    const fccRow = document.getElementById('hubFccModelRow');
    if (fccRow) {
      fccRow.classList.toggle('hidden', key !== 'agent_claude');
    }
```

Add bootstrap loader (near compute route poll init):

```javascript
  let fccModelLabels = [];
  let fccDefaultLabel = null;
  let agentClaudeEnabled = false;

  async function loadFccModelLabels() {
    const opt = document.getElementById('hubModeAgentClaudeOption');
    const row = document.getElementById('hubFccModelRow');
    const sel = document.getElementById('hubFccModelSelect');
    try {
      const r = await fetch(`${API_BASE_URL}/api/fcc-model-labels`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const body = await r.json();
      agentClaudeEnabled = Boolean(body.enabled);
      fccModelLabels = Array.isArray(body.labels) ? body.labels : [];
      fccDefaultLabel = body.default_label || fccModelLabels[0] || null;
      if (opt) opt.hidden = !agentClaudeEnabled;
      if (sel) {
        sel.innerHTML = '';
        fccModelLabels.forEach((label) => {
          const o = document.createElement('option');
          o.value = label;
          o.textContent = label;
          sel.appendChild(o);
        });
        if (fccDefaultLabel) sel.value = fccDefaultLabel;
      }
      if (row && hubModeSelect && hubModeSelect.value === 'agent_claude') {
        row.classList.remove('hidden');
      }
    } catch (err) {
      console.warn('fcc model labels load failed', err);
      if (opt) opt.hidden = true;
    }
  }
  loadFccModelLabels();
```

In text send payload construction (~line 10798), after `payload.mode = requestMode`:

```javascript
    if (requestMode === 'agent-claude') {
      const fccSel = document.getElementById('hubFccModelSelect');
      payload.fcc_model_label = (fccSel && fccSel.value) ? fccSel.value : (fccDefaultLabel || 'MODEL');
      payload.use_recall = false;
      payload.llm_route = null;
    }
```

In `confirmDownRouteOrProceed` call sites for text send (~10760), skip when agent-claude:

```javascript
    const skipCompute = requestMode === 'agent-claude' || (HUB_MODE_SPECS[hubModeSelect?.value]?.skipComputeConfirm);
    if (!skipCompute) {
      // existing confirmDownRouteOrProceed logic
    }
```

Mirror the same `fcc_model_label` + skip compute logic in the voice `audioPayload` block (~11158).

- [ ] **Step 3: Manual UI check (operator)**

With `HUB_AGENT_CLAUDE_ENABLED=true` and Hub running on host:

1. Reload Hub — Mode dropdown shows **Agent Claude**
2. Selecting it reveals **FCC Model** dropdown populated from `~/.fcc/.env`
3. Compute confirm dialog does not appear on send

- [ ] **Step 4: Commit**

```bash
git add services/orion-hub/templates/index.html services/orion-hub/static/js/app.js
git commit -m "feat(hub): add agent-claude mode and FCC model dropdown UI"
```

---

### Task 11: Live Claude step renderer

**Files:**
- Create: `services/orion-hub/static/js/agent-claude-trace.js`
- Modify: `services/orion-hub/static/js/app.js` (WS handler)

- [ ] **Step 1: Create renderer**

Create `services/orion-hub/static/js/agent-claude-trace.js`:

```javascript
(function (global) {
  const _liveClaudeSteps = new Map();
  const LIVE_ANCHOR_ID = 'conversation';

  function resolveAnchor(doc) {
    const root = doc || (typeof document !== 'undefined' ? document : null);
    return root && root.getElementById ? root.getElementById(LIVE_ANCHOR_ID) : null;
  }

  function ensurePanel(correlationId, doc) {
    const anchor = resolveAnchor(doc);
    if (!anchor) return null;
    const panelId = `claude-live-${correlationId}`;
    let panel = (doc || document).getElementById(panelId);
    if (panel) return panel;
    panel = (doc || document).createElement('div');
    panel.id = panelId;
    panel.className = 'agent-live-trace claude-live-trace';
    const heading = (doc || document).createElement('div');
    heading.className = 'agent-live-trace__heading';
    heading.textContent = 'Claude harness (live)';
    panel.appendChild(heading);
    const steps = (doc || document).createElement('div');
    steps.className = 'agent-live-trace__steps';
    panel.appendChild(steps);
    anchor.appendChild(panel);
    return panel;
  }

  function summarizeStep(step) {
    if (!step || typeof step !== 'object') return 'step';
    const raw = step.raw && typeof step.raw === 'object' ? step.raw : step;
    const type = String(step.type || raw.type || 'event');
    if (type === 'assistant') {
      const msg = raw.message && raw.message.content;
      if (Array.isArray(msg)) {
        const text = msg.filter((b) => b && b.type === 'text').map((b) => b.text).join('');
        if (text) return text.slice(0, 240);
      }
    }
    if (type === 'result') return String(raw.result || 'result').slice(0, 240);
    return type;
  }

  function appendLiveClaudeStep(correlationId, step, doc) {
    if (!correlationId || !step) return;
    const list = _liveClaudeSteps.get(correlationId) || [];
    list.push(step);
    _liveClaudeSteps.set(correlationId, list);
    const panel = ensurePanel(correlationId, doc);
    if (!panel) return;
    const host = panel.querySelector('.agent-live-trace__steps');
    if (!host) return;
    const row = (doc || document).createElement('div');
    row.className = 'agent-live-trace__step';
    row.textContent = `#${list.length - 1} ${summarizeStep(step)}`;
    host.appendChild(row);
    host.scrollTop = host.scrollHeight;
  }

  global.appendLiveClaudeStep = appendLiveClaudeStep;
  global.OrionClaudeTrace = { appendLiveClaudeStep };
})(typeof window !== 'undefined' ? window : globalThis);
```

- [ ] **Step 2: Wire WS handler in `app.js`**

After the `agent_step` block (~line 10559):

```javascript
          if (d.kind === 'claude_step' && d.step) {
            try { appendLiveClaudeStep(d.correlation_id, d.step); } catch (err) { console.warn('claude_step render failed', err); }
            return;
          }
```

- [ ] **Step 3: Commit**

```bash
git add services/orion-hub/static/js/agent-claude-trace.js services/orion-hub/static/js/app.js services/orion-hub/templates/index.html
git commit -m "feat(hub): render live claude_step frames in chat UI"
```

---

### Task 12: Live smoke script

**Files:**
- Create: `services/orion-hub/scripts/verify_agent_claude_stream_live.py`

- [ ] **Step 1: Create script**

Create `services/orion-hub/scripts/verify_agent_claude_stream_live.py`:

```python
"""Live smoke: Hub agent-claude WS streams claude_step + final llm_response.

Usage:
  PYTHONPATH=services/orion-hub:. python services/orion-hub/scripts/verify_agent_claude_stream_live.py \\
      --ws ws://127.0.0.1:8080/ws \\
      --text "list files in services/orion-hub/scripts" \\
      --fcc-model-label MODEL_HAIKU
Exit 0 iff: >=1 claude_step frames AND final llm_response non-empty.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

import websockets


async def run(ws_url: str, text: str, fcc_label: str, timeout: float) -> int:
    steps = 0
    final = ""
    session_id = f"agent-claude-{uuid.uuid4().hex[:8]}"
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({
            "mode": "agent-claude",
            "text": text,
            "session_id": session_id,
            "fcc_model_label": fcc_label,
            "claude_session_id": None,
            "resume": False,
        }))
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                d = json.loads(raw)
                if d.get("kind") == "claude_step":
                    steps += 1
                    step = d.get("step") or {}
                    print(f"  step type={step.get('type')}")
                elif d.get("llm_response"):
                    final = str(d.get("llm_response"))
                    break
                elif d.get("error"):
                    print("error:", d.get("error"), d.get("error_code"))
                    break
        except asyncio.TimeoutError:
            print("timeout waiting for frames")

    print(f"steps={steps} final_len={len(final)}")
    print("final_text:", final[:500])
    ok = steps >= 1 and bool(final.strip())
    print("AGENT_CLAUDE_PASS" if ok else "AGENT_CLAUDE_FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="ws://127.0.0.1:8080/ws")
    ap.add_argument("--text", default="list files in services/orion-hub/scripts")
    ap.add_argument("--fcc-model-label", default="MODEL_HAIKU")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()
    return asyncio.run(run(args.ws, args.text, args.fcc_model_label, args.timeout))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Document in README** (Task 13 covers full README; note script path there)

- [ ] **Step 3: Commit**

```bash
git add services/orion-hub/scripts/verify_agent_claude_stream_live.py
git commit -m "test(hub): add agent-claude live WebSocket smoke script"
```

---

### Task 13: README + gate run

**Files:**
- Modify: `services/orion-hub/README.md`

- [ ] **Step 1: Add README section**

After the Agent mode → context-exec section (~line 514), add:

```markdown
### Agent Claude mode (FCC harness)

When `HUB_AGENT_CLAUDE_ENABLED=true`, Hub exposes **Agent Claude** mode. Each message spawns one `claude -p … --output-format stream-json` turn with `ANTHROPIC_BASE_URL` set to `HUB_FCC_SERVER_URL` (default `http://127.0.0.1:8082`). The **FCC Model** dropdown lists env key labels from `HUB_FCC_ENV_PATH` (`MODEL`, `MODEL_HAIKU`, …), not resolved gateway values. **Compute** lane is ignored for this mode.

**Requirements (v1):** Hub process on host (or container with `claude` on PATH + mounted repo + mounted `~/.fcc/.env`). `fcc-server` running. Default Docker compose keeps `HUB_AGENT_CLAUDE_ENABLED=false`.

**Live smoke:**

```bash
HUB_AGENT_CLAUDE_ENABLED=true fcc-server  # separate terminal
PYTHONPATH=services/orion-hub:. python services/orion-hub/scripts/verify_agent_claude_stream_live.py \
  --ws ws://127.0.0.1:8080/ws \
  --text "list files in services/orion-hub/scripts" \
  --fcc-model-label MODEL_HAIKU
```
```

- [ ] **Step 2: Run full gate tests**

```bash
cd /mnt/scripts/Orion-Sapienform
python scripts/sync_local_env_from_example.py
python scripts/check_env_template_parity.py
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_fcc_env_catalog.py \
  services/orion-hub/tests/test_fcc_model_label_mapping.py \
  services/orion-hub/tests/test_fcc_claude_bridge_parse.py \
  services/orion-hub/tests/test_fcc_claude_bridge_run.py \
  services/orion-hub/tests/test_agent_claude_input.py \
  services/orion-hub/tests/test_fcc_model_labels_api.py \
  services/orion-hub/tests/test_http_agent_claude.py \
  services/orion-hub/tests/test_websocket_agent_claude_routing.py -q
```

Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add services/orion-hub/README.md
git commit -m "docs(hub): document agent-claude FCC harness mode"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Mode `agent-claude` in dropdown when enabled | Task 10 |
| Single-turn per message (no `--resume`) | Task 6 (no resume argv) |
| Stream `claude_step` frames | Tasks 6, 8, 11 |
| FCC env key labels in dropdown | Tasks 2, 7, 10 |
| Fixed workspace `HUB_AGENT_CLAUDE_WORKSPACE` | Tasks 1, 6 |
| Compute dropdown unchanged / not used | Task 10 (`skipComputeConfirm`) |
| No FCC package changes | All tasks (subprocess only) |
| `GET /api/fcc-model-labels` + `fcc_server_ok` | Task 7 |
| Error codes from spec §8 | Tasks 6, 8, 9 |
| Gate tests from spec §9 | Tasks 2–9 |
| Live smoke script | Task 12 |
| v2-ready seams (`cancel_turn`, `prepare_agent_claude_input`) | Tasks 4, 6 |

---

## Restart required (after full implementation)

Hub on host (recommended for v1):

```bash
# ensure fcc-server is up, then restart Hub with agent-claude enabled
export HUB_AGENT_CLAUDE_ENABLED=true
# restart your existing Hub process (systemd, manual uvicorn, etc.)
```

Docker (agent-claude stays disabled unless operator opts in):

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build orion-hub
```

---

## Risks (from spec — no code change)

- Hub in Docker without `claude` on PATH → keep default `HUB_AGENT_CLAUDE_ENABLED=false`
- Long tool turns → timeout + v2 `/stop` via `cancel_turn()`
- `~/.fcc/.env` not visible in container → mount or override `HUB_FCC_ENV_PATH`
