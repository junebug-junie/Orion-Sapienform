from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .workspace import ContextExecWorkspace


def _workspace_unavailable_message(workspace_info: dict[str, Any] | None) -> str | None:
    if workspace_info is None:
        return "workspace unavailable: no workspace allocated for this run"
    if not workspace_info.get("enabled", True):
        reason = workspace_info.get("reason") or "workspace disabled"
        return f"workspace unavailable: {reason}"
    if not workspace_info.get("allocated", False):
        err = workspace_info.get("error") or workspace_info.get("reason") or "allocation failed"
        return f"workspace unavailable: {err}"
    return None


def _resolve_workspace_path(
    workspace: ContextExecWorkspace,
    path: str,
    *,
    allowed_roots: tuple[Path, ...] | None = None,
) -> tuple[Path, str] | str:
    raw = path.strip()
    if not raw:
        return "path required"
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        return f"path blocked: absolute paths not allowed: {path!r}"
    if ".." in Path(raw).parts:
        return f"path blocked: traversal not allowed: {path!r}"

    ws_root = workspace.root.resolve()
    target = (ws_root / raw).resolve()
    if ws_root not in target.parents and target != ws_root:
        return f"path blocked: escapes workspace: {path!r}"

    if allowed_roots is not None:
        if not any(
            root == target or root in target.parents
            for root in allowed_roots
        ):
            allowed = ", ".join(str(r.relative_to(ws_root)) for r in allowed_roots)
            return f"path blocked: must stay under {allowed}"

    rel = str(target.relative_to(ws_root)).replace("\\", "/")
    return target, rel


def workspace_write(
    workspace_info: dict[str, Any] | None,
    workspace: ContextExecWorkspace | None,
    path: str,
    content: str,
) -> str:
    unavailable = _workspace_unavailable_message(workspace_info)
    if unavailable:
        return unavailable
    if workspace is None:
        return "workspace unavailable: workspace handle missing for this run"
    resolved = _resolve_workspace_path(workspace, path)
    if isinstance(resolved, str):
        return resolved
    target, rel = resolved
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"write failed for {rel}: {exc}"
    return f"wrote {rel} ({len(content)} chars)"


def workspace_read(
    workspace_info: dict[str, Any] | None,
    workspace: ContextExecWorkspace | None,
    path: str,
) -> str:
    unavailable = _workspace_unavailable_message(workspace_info)
    if unavailable:
        return unavailable
    if workspace is None:
        return "workspace unavailable: workspace handle missing for this run"
    resolved = _resolve_workspace_path(workspace, path)
    if isinstance(resolved, str):
        return resolved
    target, rel = resolved
    if not target.is_file():
        if target.exists():
            return f"path is not a file: {rel}"
        return f"path absent: {rel}"
    try:
        return target.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return f"read failed for {rel}: {exc}"


def workspace_list(
    workspace_info: dict[str, Any] | None,
    workspace: ContextExecWorkspace | None,
    path: str = "",
) -> str:
    unavailable = _workspace_unavailable_message(workspace_info)
    if unavailable:
        return unavailable
    if workspace is None:
        return "workspace unavailable: workspace handle missing for this run"
    resolved = _resolve_workspace_path(workspace, path or ".")
    if isinstance(resolved, str):
        return resolved
    target, rel = resolved
    if not target.is_dir():
        return f"path is not a directory: {rel or '.'}"
    entries: list[str] = []
    for fp in sorted(target.iterdir()):
        child_rel = str(fp.relative_to(workspace.root)).replace("\\", "/")
        entries.append(child_rel + ("/" if fp.is_dir() else ""))
    if not entries:
        return f"No entries under: {rel or '.'}"
    return "\n".join(entries)


def workspace_write_patch(
    workspace_info: dict[str, Any] | None,
    workspace: ContextExecWorkspace | None,
    name: str,
    unified_diff: str,
) -> str:
    unavailable = _workspace_unavailable_message(workspace_info)
    if unavailable:
        return unavailable
    if workspace is None:
        return "workspace unavailable: workspace handle missing for this run"
    safe_name = Path(name.strip()).name
    if not safe_name:
        return "name required"
    if not safe_name.endswith(".patch"):
        safe_name = f"{safe_name}.patch"
    target = workspace.patches_dir / safe_name
    try:
        workspace.patches_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(unified_diff, encoding="utf-8")
    except OSError as exc:
        return f"write failed for patches/{safe_name}: {exc}"
    return f"wrote patches/{safe_name} ({len(unified_diff)} chars)"


def workspace_write_report(
    workspace_info: dict[str, Any] | None,
    workspace: ContextExecWorkspace | None,
    name: str,
    markdown: str,
) -> str:
    unavailable = _workspace_unavailable_message(workspace_info)
    if unavailable:
        return unavailable
    if workspace is None:
        return "workspace unavailable: workspace handle missing for this run"
    safe_name = Path(name.strip()).name
    if not safe_name:
        return "name required"
    if not safe_name.endswith(".md"):
        safe_name = f"{safe_name}.md"
    target = workspace.outputs_dir / safe_name
    try:
        workspace.outputs_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown, encoding="utf-8")
    except OSError as exc:
        return f"write failed for outputs/{safe_name}: {exc}"
    return f"wrote outputs/{safe_name} ({len(markdown)} chars)"
