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
