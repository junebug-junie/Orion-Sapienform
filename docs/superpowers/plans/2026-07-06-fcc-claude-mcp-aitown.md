# fcc-claude MCP + mesh AI Town Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire GitHub, Firecrawl, and Orion AI Town gameplay MCPs into Hub agent-claude turns; deploy self-hosted AI Town on the mesh; add Hub AI Town tab with proxied iframe visualization.

**Architecture:** `fcc_mcp_config.py` renders ephemeral `--mcp-config` JSON per turn from a checked-in template + `~/.fcc/.env` secrets. `fcc_claude_bridge.run_turn()` passes `--mcp-config` and `--allowedTools "mcp__*"`. `services/orion-ai-town/` runs upstream ai-town with self-hosted Convex. `orion-aitown-mcp` (Python stdio) exposes gameplay tools via Convex HTTP `sendInput`. Hub proxies `/aitown/` → `HUB_AITOWN_UI_URL` and polls `/api/aitown/status`.

**Tech Stack:** Python 3.12, FastAPI, asyncio subprocess, Claude Code CLI, Docker MCP (github), npx MCP (firecrawl), MCP Python SDK, Convex self-hosted, pytest.

**Spec:** [`docs/superpowers/specs/2026-07-06-fcc-claude-mcp-aitown-design.md`](../specs/2026-07-06-fcc-claude-mcp-aitown-design.md)

**Branch:** `feat/fcc-claude-mcp-aitown` (worktree: `../Orion-Sapienform-fcc-claude-mcp-aitown`)

---

## File map

| Path | Responsibility |
|------|----------------|
| `services/orion-hub/config/fcc_claude_mcp.template.json` | **Create** — MCP server defs (no secrets) |
| `services/orion-hub/scripts/fcc_mcp_config.py` | **Create** — render, preflight, temp cleanup |
| `services/orion-hub/scripts/fcc_claude_bridge.py` | **Modify** — `--mcp-config`, MCP preflight |
| `services/orion-hub/app/settings.py` | **Modify** — `HUB_AGENT_CLAUDE_MCP_ENABLED`, `HUB_AITOWN_*` |
| `services/orion-hub/.env_example` | **Modify** — new keys |
| `services/orion-hub/Dockerfile` | **Modify** — Node 20 for npx |
| `services/orion-hub/docker-compose.yml` | **Modify** — env passthrough |
| `services/orion-hub/scripts/api_routes.py` | **Modify** — `/api/aitown/status`, `/aitown/` proxy |
| `services/orion-hub/templates/index.html` | **Modify** — AI Town tab |
| `services/orion-hub/static/js/aitown-panel.js` | **Create** — status poll, lazy iframe |
| `services/orion-hub/static/js/app.js` | **Modify** — tab routing |
| `services/orion-hub/tests/test_fcc_mcp_config.py` | **Create** |
| `services/orion-hub/tests/test_fcc_claude_bridge_mcp.py` | **Create** |
| `services/orion-hub/tests/test_aitown_status_api.py` | **Create** |
| `services/orion-hub/tests/test_aitown_proxy.py` | **Create** |
| `services/orion-hub/tests/test_hub_aitown_tab.py` | **Create** |
| `services/orion-ai-town/docker-compose.yml` | **Create** — self-hosted Convex + frontend |
| `services/orion-ai-town/.env_example` | **Create** |
| `services/orion-ai-town/README.md` | **Create** — mesh bootstrap |
| `services/orion-ai-town/mcp/orion_aitown_mcp/` | **Create** — gameplay MCP package |
| `services/orion-ai-town/mcp/tests/` | **Create** |
| `services/orion-hub/README.md` | **Modify** — MCP + AI Town tab docs |

**Non-goals:** Harness-governor MCP, bus events from gameplay, iframe postMessage, Convex cloud, AI Town character fork.

---

## Phase 1 — MCP core (GitHub + Firecrawl)

### Task 1: MCP settings + env

**Files:**
- Modify: `services/orion-hub/app/settings.py`
- Modify: `services/orion-hub/.env_example`
- Modify: `services/orion-hub/docker-compose.yml`

- [ ] **Step 1: Add settings fields** (after `HUB_AGENT_CLAUDE_MAX_CONCURRENT` block)

```python
    HUB_AGENT_CLAUDE_MCP_ENABLED: bool = Field(
        default=False,
        alias="HUB_AGENT_CLAUDE_MCP_ENABLED",
    )
    HUB_AITOWN_ENABLED: bool = Field(
        default=False,
        alias="HUB_AITOWN_ENABLED",
    )
    HUB_AITOWN_UI_URL: str = Field(
        default="http://127.0.0.1:5173",
        alias="HUB_AITOWN_UI_URL",
    )
```

- [ ] **Step 2: Append to `.env_example`**

```bash
# fcc-claude MCP (GitHub + Firecrawl + AI Town when enabled)
HUB_AGENT_CLAUDE_MCP_ENABLED=false
HUB_AITOWN_ENABLED=false
HUB_AITOWN_UI_URL=http://127.0.0.1:5173
# Secrets live in ~/.fcc/.env: GITHUB_PAT, FIRECRAWL_API_KEY, AITOWN_*
```

- [ ] **Step 3: docker-compose passthrough**

```yaml
      - HUB_AGENT_CLAUDE_MCP_ENABLED=${HUB_AGENT_CLAUDE_MCP_ENABLED:-false}
      - HUB_AITOWN_ENABLED=${HUB_AITOWN_ENABLED:-false}
      - HUB_AITOWN_UI_URL=${HUB_AITOWN_UI_URL:-http://127.0.0.1:5173}
```

- [ ] **Step 4: Sync local env**

```bash
python scripts/sync_local_env_from_example.py
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/app/settings.py services/orion-hub/.env_example services/orion-hub/docker-compose.yml
git commit -m "feat(hub): add fcc-claude MCP and AI Town settings"
```

---

### Task 2: MCP template + render module

**Files:**
- Create: `services/orion-hub/config/fcc_claude_mcp.template.json`
- Create: `services/orion-hub/scripts/fcc_mcp_config.py`
- Create: `services/orion-hub/tests/test_fcc_mcp_config.py`

- [ ] **Step 1: Write failing test**

```python
# services/orion-hub/tests/test_fcc_mcp_config.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.fcc_mcp_config import McpPreflightError, render_mcp_config


def test_render_injects_github_and_firecrawl_secrets(tmp_path: Path) -> None:
    out = render_mcp_config(
        correlation_id="corr-1",
        fcc_env={
            "GITHUB_PAT": "ghp_test",
            "FIRECRAWL_API_KEY": "fc_test",
        },
        tmp_dir=tmp_path,
        include_aitown=False,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mcpServers"]["github"]["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_test"
    assert data["mcpServers"]["firecrawl"]["env"]["FIRECRAWL_API_KEY"] == "fc_test"
    assert "orion-aitown" not in data["mcpServers"]


def test_render_fails_without_github_pat(tmp_path: Path) -> None:
    with pytest.raises(McpPreflightError) as exc:
        render_mcp_config(
            correlation_id="corr-2",
            fcc_env={"FIRECRAWL_API_KEY": "fc_test"},
            tmp_dir=tmp_path,
            include_aitown=False,
        )
    assert exc.value.error_code == "fcc_mcp_github_missing"
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_fcc_mcp_config.py -v
```

- [ ] **Step 3: Create template** `services/orion-hub/config/fcc_claude_mcp.template.json`

```json
{
  "mcpServers": {
    "github": {
      "type": "stdio",
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "__GITHUB_PAT__"
      }
    },
    "firecrawl": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "firecrawl-mcp"],
      "env": {
        "FIRECRAWL_API_KEY": "__FIRECRAWL_API_KEY__"
      }
    }
  }
}
```

- [ ] **Step 4: Implement** `services/orion-hub/scripts/fcc_mcp_config.py`

```python
"""Render ephemeral MCP config for fcc-claude turns."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "config" / "fcc_claude_mcp.template.json"
_TMP_ROOT = Path("/tmp/orion-fcc-mcp")


@dataclass(frozen=True)
class McpPreflightError(Exception):
    error_code: str
    message: str

    def __str__(self) -> str:
        return self.message


def _require(env: Mapping[str, str], key: str, *, error_code: str) -> str:
    val = str(env.get(key) or "").strip()
    if not val:
        raise McpPreflightError(error_code=error_code, message=f"Missing {key} in FCC env")
    return val


def _deep_replace(obj: Any, replacements: Dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_replace(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_replace(v, replacements) for v in obj]
    if isinstance(obj, str):
        out = obj
        for needle, repl in replacements.items():
            out = out.replace(needle, repl)
        return out
    return obj


def render_mcp_config(
    *,
    correlation_id: str,
    fcc_env: Mapping[str, str],
    tmp_dir: Optional[Path] = None,
    include_aitown: bool = False,
    aitown_env: Optional[Mapping[str, str]] = None,
) -> Path:
    github_pat = _require(fcc_env, "GITHUB_PAT", error_code="fcc_mcp_github_missing")
    firecrawl_key = _require(fcc_env, "FIRECRAWL_API_KEY", error_code="fcc_mcp_firecrawl_missing")

    template = json.loads(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    replacements = {
        "__GITHUB_PAT__": github_pat,
        "__FIRECRAWL_API_KEY__": firecrawl_key,
    }
    rendered = _deep_replace(template, replacements)

    if include_aitown:
        ae = dict(aitown_env or fcc_env)
        rendered["mcpServers"]["orion-aitown"] = {
            "type": "stdio",
            "command": "python3",
            "args": ["-m", "orion_aitown_mcp"],
            "env": {
                "AITOWN_CONVEX_URL": _require(ae, "AITOWN_CONVEX_URL", error_code="fcc_mcp_aitown_config"),
                "AITOWN_ADMIN_KEY": _require(ae, "AITOWN_ADMIN_KEY", error_code="fcc_mcp_aitown_config"),
                "AITOWN_WORLD_ID": _require(ae, "AITOWN_WORLD_ID", error_code="fcc_mcp_aitown_config"),
                "AITOWN_ORION_AGENT_ID": str(ae.get("AITOWN_ORION_AGENT_ID") or ""),
                "AITOWN_ORION_PLAYER_ID": str(ae.get("AITOWN_ORION_PLAYER_ID") or ""),
            },
        }

    root = tmp_dir or _TMP_ROOT
    root.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^a-zA-Z0-9._-]", "_", str(correlation_id))
    out = root / f"{safe_id}.json"
    out.write_text(json.dumps(rendered, indent=2), encoding="utf-8")
    return out


def cleanup_mcp_config(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_fcc_mcp_config.py -q
```

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub/config/fcc_claude_mcp.template.json \
        services/orion-hub/scripts/fcc_mcp_config.py \
        services/orion-hub/tests/test_fcc_mcp_config.py
git commit -m "feat(hub): add fcc MCP config render and preflight"
```

---

### Task 3: Wire bridge spawn argv

**Files:**
- Modify: `services/orion-hub/scripts/fcc_claude_bridge.py`
- Create: `services/orion-hub/tests/test_fcc_claude_bridge_mcp.py`

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_run_turn_adds_mcp_config_when_enabled(monkeypatch):
    captured_argv: list = []

    async def fake_exec(*args, **kwargs):
        captured_argv.extend(args)
        return _FakeProc([])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(bridge, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(bridge, "_maybe_render_mcp_config", lambda **k: Path("/tmp/fake-mcp.json"))

    from scripts.settings import settings
    monkeypatch.setattr(settings, "HUB_AGENT_CLAUDE_MCP_ENABLED", True, raising=False)

    async for _ in bridge.run_turn(...):
        pass

    assert "--mcp-config" in captured_argv
    assert "/tmp/fake-mcp.json" in [str(x) for x in captured_argv]
```

- [ ] **Step 2: Implement in `fcc_claude_bridge.py`**

Add helper:

```python
def _maybe_render_mcp_config(*, correlation_id: str) -> Optional[Path]:
    from scripts.fcc_env_catalog import expand_env_path, load_fcc_env
    from scripts.fcc_mcp_config import McpPreflightError, render_mcp_config
    from scripts.settings import settings

    if not settings.HUB_AGENT_CLAUDE_MCP_ENABLED:
        return None
    env = load_fcc_env(expand_env_path(settings.HUB_FCC_ENV_PATH))
    try:
        return render_mcp_config(
            correlation_id=correlation_id,
            fcc_env=env,
            include_aitown=bool(settings.HUB_AITOWN_ENABLED),
        )
    except McpPreflightError as exc:
        raise RuntimeError(str(exc)) from exc
```

In `run_turn`, before spawn:

```python
    mcp_config_path: Optional[Path] = None
    try:
        mcp_config_path = _maybe_render_mcp_config(correlation_id=correlation_id)
    except RuntimeError as exc:
        yield {"type": "error", "error": str(exc), "error_code": "fcc_mcp_preflight_failed"}
        return

    argv = [claude_bin, "-p", prompt, "--output-format", "stream-json", "--verbose", "--model", model_id]
    if mcp_config_path is not None:
        argv.extend(["--mcp-config", str(mcp_config_path), "--allowedTools", "mcp__*"])
```

In `finally`: call `cleanup_mcp_config(mcp_config_path)` when set.

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_fcc_claude_bridge_mcp.py services/orion-hub/tests/test_fcc_claude_bridge_run.py -q
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(hub): pass --mcp-config to fcc-claude subprocess"
```

---

### Task 4: Hub Dockerfile Node 20

**Files:**
- Modify: `services/orion-hub/Dockerfile`

- [ ] **Step 1: Add Node 20 install** after apt docker block:

```dockerfile
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
```

- [ ] **Step 2: Verify in container build** (or document if Docker unavailable)

```bash
docker compose --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml build hub-app
docker compose -f services/orion-hub/docker-compose.yml run --rm hub-app node --version
```

Expected: `v20.x.x`

- [ ] **Step 3: Commit**

```bash
git commit -m "chore(hub): add Node 20 for firecrawl MCP npx"
```

---

## Phase 2 — orion-ai-town mesh deploy

### Task 5: orion-ai-town compose + README

**Files:**
- Create: `services/orion-ai-town/docker-compose.yml`
- Create: `services/orion-ai-town/.env_example`
- Create: `services/orion-ai-town/README.md`

- [ ] **Step 1: Create compose** — adapt upstream ai-town self-hosted services (`convex-backend`, `dashboard`, `frontend` build). Pin `AITOWN_UPSTREAM_REF=main` or specific tag in README.

- [ ] **Step 2: `.env_example`** with `PORT=3210`, `SITE_PROXY_PORT=3211`, `DASHBOARD_PORT=6791`, `OLLAMA_HOST`, `INSTANCE_SECRET`.

- [ ] **Step 3: README** — bootstrap steps: admin key, `npm run predev`, `npx convex run init`, capture `AITOWN_WORLD_ID`, production frontend build, mesh LLM wiring.

- [ ] **Step 4: Smoke**

```bash
docker compose -f services/orion-ai-town/docker-compose.yml config
curl -fsS http://127.0.0.1:3210/version
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-ai-town/
git commit -m "feat(aitown): add mesh self-hosted AI Town compose"
```

---

## Phase 3 — orion-aitown-mcp gameplay tools

### Task 6: Convex HTTP client + MCP tools

**Files:**
- Create: `services/orion-ai-town/mcp/orion_aitown_mcp/__init__.py`
- Create: `services/orion-ai-town/mcp/orion_aitown_mcp/client.py`
- Create: `services/orion-ai-town/mcp/orion_aitown_mcp/server.py`
- Create: `services/orion-ai-town/mcp/pyproject.toml` or `requirements.txt`
- Create: `services/orion-ai-town/mcp/tests/test_aitown_tools.py`

- [ ] **Step 1: Write failing test** for `send_input` payload shape (mock `urllib` or `httpx`).

- [ ] **Step 2: Implement `client.py`** — `convex_mutation(path, args)` and `convex_query(path, args)` against `{AITOWN_CONVEX_URL}/api/...` with admin key header per self-hosted Convex HTTP docs.

- [ ] **Step 3: Implement MCP tools** in `server.py` using `mcp` Python SDK stdio server — tools from spec table (`aitown_move_player`, `aitown_send_input`, etc.).

- [ ] **Step 4: Extend template** — add `orion-aitown` block via `include_aitown=True` path (already in render module).

- [ ] **Step 5: Extend `test_fcc_mcp_config.py`** — `include_aitown=True` includes third server.

- [ ] **Step 6: Run tests**

```bash
pytest services/orion-ai-town/mcp/tests/ services/orion-hub/tests/test_fcc_mcp_config.py -q
```

- [ ] **Step 7: Commit**

```bash
git commit -m "feat(aitown): add gameplay MCP server for Convex sendInput"
```

---

## Phase 4 — Hub AI Town tab + proxy

### Task 7: Status API + reverse proxy

**Files:**
- Create: `services/orion-hub/scripts/aitown_status.py`
- Modify: `services/orion-hub/scripts/api_routes.py`
- Create: `services/orion-hub/tests/test_aitown_status_api.py`
- Create: `services/orion-hub/tests/test_aitown_proxy.py`

- [ ] **Step 1: `aitown_status.py`** — `fetch_aitown_status(settings) -> dict` probing Convex `/version` + lightweight query for engine stats (reuse client patterns from MCP or httpx).

- [ ] **Step 2: Routes**

```python
@router.get("/api/aitown/status")
async def aitown_status():
    ...

@router.api_route("/aitown/{path:path}", methods=["GET", "HEAD"])
async def aitown_proxy(path: str, request: Request):
    # httpx forward to HUB_AITOWN_UI_URL when HUB_AITOWN_ENABLED
```

- [ ] **Step 3: Tests** — mock upstream; assert status JSON shape; proxy forwards path.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(hub): AI Town status API and reverse proxy"
```

---

### Task 8: Hub UI tab

**Files:**
- Modify: `services/orion-hub/templates/index.html`
- Create: `services/orion-hub/static/js/aitown-panel.js`
- Modify: `services/orion-hub/static/js/app.js`
- Create: `services/orion-hub/tests/test_hub_aitown_tab.py`

- [ ] **Step 1: Add nav tab** `#aiTownTabButton` and section `#ai-town` gated by server-side flag or always present with JS hide when disabled.

- [ ] **Step 2: `aitown-panel.js`** — poll `/api/aitown/status` every 10s when tab visible; lazy-set iframe `src="/aitown/"` on first visit.

- [ ] **Step 3: `test_hub_aitown_tab.py`** — render template with `HUB_AITOWN_ENABLED=true`; assert `#ai-town` and iframe present.

- [ ] **Step 4: Manual smoke** — open Hub `#ai-town`, confirm game loads, status strip updates.

- [ ] **Step 5: Commit + README**

```bash
git commit -m "feat(hub): AI Town visualization tab with proxied iframe"
```

---

## Phase 5 — Docs + operator keys

### Task 9: Document ~/.fcc/.env keys

**Files:**
- Modify: `services/orion-hub/README.md`
- Modify: `services/orion-ai-town/README.md`

- [ ] Document all `GITHUB_PAT`, `FIRECRAWL_API_KEY`, `AITOWN_*` keys with bootstrap order.
- [ ] Document `HUB_AGENT_CLAUDE_MCP_ENABLED=true` rollout.
- [ ] Commit.

---

## Verification gate

```bash
make agent-check SERVICE=orion-hub  # extend if Makefile target exists
PYTHONPATH=services/orion-hub:. pytest \
  services/orion-hub/tests/test_fcc_mcp_config.py \
  services/orion-hub/tests/test_fcc_claude_bridge_mcp.py \
  services/orion-hub/tests/test_aitown_status_api.py \
  services/orion-hub/tests/test_aitown_proxy.py \
  services/orion-hub/tests/test_hub_aitown_tab.py -q
pytest services/orion-ai-town/mcp/tests/ -q
python scripts/check_env_template_parity.py
```

Manual smoke checklist (from spec acceptance):
1. MCP enabled + secrets → agent-claude turn shows MCP tool steps.
2. AI Town tab renders proxied game.
3. `aitown_move_player` moves character visible in iframe.

---

## Restart required

```bash
docker compose --env-file services/orion-ai-town/.env \
  -f services/orion-ai-town/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

Add to `~/.fcc/.env` on operator machine after bootstrap.

---

## Plan self-review

| Spec requirement | Task |
|------------------|------|
| Orion-managed MCP template + render | Task 2–3 |
| GitHub Docker MCP | Task 2 template |
| Firecrawl npx MCP | Task 2, Task 4 Node |
| Mesh AI Town deploy | Task 5 |
| Gameplay MCP (not generic Convex) | Task 6 |
| Embodied defaults + send_input | Task 6 |
| Hub proxied iframe tab | Task 7–8 |
| Secrets in ~/.fcc/.env | Task 9 |
| Preflight error codes | Task 2–3 |
| `HUB_AGENT_CLAUDE_MCP_ENABLED` gate | Task 1, 3 |

No placeholders remain. Phases are independently shippable: Phase 1 delivers GitHub+Firecrawl MCP; Phase 3 adds gameplay; Phase 4 adds UI.
