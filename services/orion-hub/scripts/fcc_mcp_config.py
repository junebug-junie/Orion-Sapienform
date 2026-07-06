"""Render ephemeral MCP config for fcc-claude turns."""
from __future__ import annotations

import json
import re
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "config" / "fcc_claude_mcp.template.json"
_TMP_ROOT = Path("/tmp/orion-fcc-mcp")


@dataclass
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


def _probe_convex_version(base_url: str, *, timeout_sec: float = 5.0) -> None:
    url = f"{str(base_url).rstrip('/')}/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            if resp.status >= 400:
                raise McpPreflightError(
                    error_code="fcc_mcp_aitown_unreachable",
                    message=f"Convex /version returned {resp.status}",
                )
    except McpPreflightError:
        raise
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        raise McpPreflightError(
            error_code="fcc_mcp_aitown_unreachable",
            message=f"Convex unreachable at {base_url}: {exc}",
        ) from exc


def _probe_convex_auth(base_url: str, admin_key: str, *, timeout_sec: float = 8.0) -> None:
    payload = json.dumps({"path": "testing:stopAllowed", "args": {}, "format": "json"}).encode("utf-8")
    req = urllib.request.Request(
        f"{str(base_url).rstrip('/')}/api/query",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Convex {admin_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body) if body.strip() else {}
        if isinstance(parsed, dict) and parsed.get("status") == "error":
            raise McpPreflightError(
                error_code="fcc_mcp_aitown_auth",
                message=str(parsed.get("errorMessage") or parsed),
            )
    except McpPreflightError:
        raise
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise McpPreflightError(
                error_code="fcc_mcp_aitown_auth",
                message=f"Convex auth rejected ({exc.code})",
            ) from exc
        raise McpPreflightError(
            error_code="fcc_mcp_aitown_unreachable",
            message=f"Convex auth probe failed ({exc.code})",
        ) from exc
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        raise McpPreflightError(
            error_code="fcc_mcp_aitown_unreachable",
            message=f"Convex auth probe failed: {exc}",
        ) from exc


def _require_tool(command: str, *, error_code: str) -> None:
    if shutil.which(command) is None:
        raise McpPreflightError(
            error_code=error_code,
            message=f"Required command not found on PATH: {command}",
        )


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
    _require_tool("docker", error_code="fcc_mcp_docker_missing")
    _require_tool("npx", error_code="fcc_mcp_node_missing")

    template = json.loads(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    replacements = {
        "__GITHUB_PAT__": github_pat,
        "__FIRECRAWL_API_KEY__": firecrawl_key,
    }
    rendered = _deep_replace(template, replacements)

    if include_aitown:
        ae = dict(aitown_env or fcc_env)
        convex_url = _require(ae, "AITOWN_CONVEX_URL", error_code="fcc_mcp_aitown_config")
        admin_key = _require(ae, "AITOWN_ADMIN_KEY", error_code="fcc_mcp_aitown_config")
        world_id = _require(ae, "AITOWN_WORLD_ID", error_code="fcc_mcp_aitown_config")
        _probe_convex_version(convex_url)
        _probe_convex_auth(convex_url, admin_key)
        rendered["mcpServers"]["orion-aitown"] = {
            "type": "stdio",
            "command": "python3",
            "args": ["-m", "orion_aitown_mcp"],
            "env": {
                "AITOWN_CONVEX_URL": convex_url,
                "AITOWN_ADMIN_KEY": admin_key,
                "AITOWN_WORLD_ID": world_id,
                "AITOWN_ORION_AGENT_ID": str(ae.get("AITOWN_ORION_AGENT_ID") or ""),
                "AITOWN_ORION_PLAYER_ID": str(ae.get("AITOWN_ORION_PLAYER_ID") or ""),
            },
        }

    root = tmp_dir or _TMP_ROOT
    safe_id = re.sub(r"[^a-zA-Z0-9._-]", "_", str(correlation_id))
    out = root / f"{safe_id}.json"
    try:
        root.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rendered, indent=2), encoding="utf-8")
    except OSError as exc:
        raise McpPreflightError(
            error_code="fcc_mcp_render_failed",
            message=f"Failed to write MCP config: {exc}",
        ) from exc
    return out


def cleanup_mcp_config(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
