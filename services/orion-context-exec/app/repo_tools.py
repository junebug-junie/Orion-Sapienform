from __future__ import annotations

import ast
import fnmatch
import re
from pathlib import Path

from .schemas import RepoFile, RepoHit


DENY_PATTERNS = (
    r"\.env$",
    r"\.pem$",
    r"\.key$",
    r"\.sqlite$",
    r"\.db$",
    r"node_modules/",
    r"\.venv/",
    r"__pycache__/",
    r"\.git/",
)

ALLOW_PREFIXES = ("orion/", "services/", "app/", "docs/", "tests/", "scripts/")


def _repo_root() -> Path:
    from .settings import settings

    root = settings.context_exec_repo_root or settings.orion_repo_root
    return Path(root).resolve()


def _is_denied(rel: str) -> bool:
    for pat in DENY_PATTERNS:
        if re.search(pat, rel):
            return True
    return False


def _is_allowed(rel: str) -> bool:
    rel = rel.lstrip("/")
    if _is_denied(rel):
        return False
    return any(rel.startswith(p) for p in ALLOW_PREFIXES)


def _normalize_rel_path(path: str) -> str | None:
    rel = path.lstrip("/")
    if not rel and path not in ("", "/"):
        return None
    parts = rel.split("/")
    if any(part == ".." for part in parts):
        return None
    return rel


def _resolve_repo_path(path: str) -> tuple[Path, str] | None:
    rel = _normalize_rel_path(path)
    if rel is None:
        return None
    root = _repo_root()
    target = (root / rel).resolve() if rel else root
    if not str(target).startswith(str(root)):
        return None
    resolved_rel = str(target.relative_to(root)).replace("\\", "/") if rel else ""
    return target, resolved_rel


def repo_grep(
    pattern: str,
    path: str | None = None,
    limit: int = 50,
    *,
    literal: bool = False,
) -> list[RepoHit]:
    root = _repo_root()
    sub = (path or "").lstrip("/")
    base = (root / sub).resolve() if sub else root
    if not str(base).startswith(str(root)):
        return []
    hits: list[RepoHit] = []
    try:
        rx = re.compile(re.escape(pattern) if literal else pattern)
    except re.error:
        return hits
    for fp in base.rglob("*"):
        if not fp.is_file():
            continue
        if len(hits) >= limit:
            break
        rel = str(fp.relative_to(root)).replace("\\", "/")
        if not _is_allowed(rel):
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                hits.append(
                    RepoHit(
                        path=rel,
                        line_start=i,
                        line_end=i,
                        snippet=line[:240],
                        source_ref=f"repo:{rel}:{i}",
                    )
                )
                if len(hits) >= limit:
                    break
    return hits


def repo_read(path: str, max_chars: int | None = None) -> RepoFile | None:
    from .settings import settings

    root = _repo_root()
    cap = max_chars if max_chars is not None else settings.context_exec_repo_max_file_chars
    rel = path.lstrip("/")
    if not _is_allowed(rel):
        return None
    fp = (root / rel).resolve()
    if not str(fp).startswith(str(root)) or not fp.is_file():
        return None
    try:
        content = fp.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    truncated = len(content) > cap
    if truncated:
        content = content[:cap]
    return RepoFile(path=rel, content=content, truncated=truncated, source_ref=f"repo:{rel}")


def repo_read_range(path: str, start_line: int, end_line: int) -> str:
    if start_line < 1 or end_line < start_line:
        return f"invalid range: start_line={start_line}, end_line={end_line}"
    resolved = _resolve_repo_path(path)
    if resolved is None:
        return f"path blocked or invalid traversal: {path!r}"
    fp, rel = resolved
    if not rel:
        return f"path is not a file: {path!r}"
    if not _is_allowed(rel):
        return f"path blocked by repo policy: {rel}"
    if not fp.is_file():
        if fp.exists():
            return f"path is not a file: {rel}"
        return f"path absent: {rel}"
    try:
        lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        return f"read failed for {rel}: {exc}"
    if start_line > len(lines):
        return f"range start {start_line} beyond file length {len(lines)}: {rel}"
    end_line = min(end_line, len(lines))
    numbered = [f"{i}|{lines[i - 1]}" for i in range(start_line, end_line + 1)]
    return "\n".join(numbered)


def repo_find_files(pattern: str, path: str = "", limit: int = 200) -> str:
    resolved = _resolve_repo_path(path)
    if resolved is None:
        return f"path blocked or invalid traversal: {path!r}"
    base, _ = resolved
    if not base.is_dir():
        return f"path is not a directory: {path!r}" if path else "path is not a directory"
    root = _repo_root()
    matches: list[str] = []
    for fp in sorted(base.rglob("*")):
        if not fp.is_file():
            continue
        rel = str(fp.relative_to(root)).replace("\\", "/")
        if not _is_allowed(rel):
            continue
        name = fp.name
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern):
            matches.append(rel)
        if len(matches) >= limit:
            break
    if not matches:
        return f"No files matching {pattern!r} under {path or '.'}"
    if len(matches) >= limit:
        return "\n".join(matches) + f"\n[truncated at limit={limit}]"
    return "\n".join(matches)


def repo_tree(path: str = "", depth: int = 2, limit: int = 300) -> str:
    if depth < 0:
        return f"invalid depth: {depth}"
    resolved = _resolve_repo_path(path)
    if resolved is None:
        return f"path blocked or invalid traversal: {path!r}"
    base, rel_prefix = resolved
    if not base.is_dir():
        return f"path is not a directory: {path!r}" if path else "path is not a directory"
    root = _repo_root()
    lines: list[str] = []
    prefix = rel_prefix or "."

    def _walk(dir_path: Path, rel: str, current_depth: int, indent: str) -> None:
        if len(lines) >= limit:
            return
        if current_depth > depth:
            return
        try:
            children = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for child in children:
            if len(lines) >= limit:
                lines.append(f"{indent}...[truncated at limit={limit}]")
                return
            child_rel = str(child.relative_to(root)).replace("\\", "/")
            if _is_denied(child_rel):
                continue
            if child.is_dir():
                if not _is_allowed(child_rel + "/"):
                    continue
                lines.append(f"{indent}{child.name}/")
                _walk(child, child_rel, current_depth + 1, indent + "  ")
            elif _is_allowed(child_rel):
                lines.append(f"{indent}{child.name}")

    lines.append(f"{prefix}/")
    _walk(base, rel_prefix, 0, "  ")
    return "\n".join(lines)


def repo_outline(path: str) -> str:
    resolved = _resolve_repo_path(path)
    if resolved is None:
        return f"path blocked or invalid traversal: {path!r}"
    fp, rel = resolved
    if not rel:
        return f"path is not a file: {path!r}"
    if not _is_allowed(rel):
        return f"path blocked by repo policy: {rel}"
    if not fp.is_file():
        if fp.exists():
            return f"path is not a file: {rel}"
        return f"path absent: {rel}"
    if not rel.endswith(".py"):
        return f"outline unsupported for non-Python file: {rel}"
    try:
        source = fp.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=rel)
    except SyntaxError as exc:
        return f"outline parse error for {rel}: {exc}"
    except OSError as exc:
        return f"read failed for {rel}: {exc}"

    parts: list[str] = [f"outline: {rel}"]
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts.append(f"L{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            parts.append(f"L{node.lineno}: from {module} import ...")
        elif isinstance(node, ast.ClassDef):
            parts.append(f"L{node.lineno}: class {node.name}")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    kind = "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
                    parts.append(f"L{item.lineno}:   {kind} {item.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            parts.append(f"L{node.lineno}: {kind} {node.name}")
    return "\n".join(parts)


def patch_validate(unified_diff: str) -> str:
    text = unified_diff.strip()
    if not text:
        return "invalid: empty diff"

    lines = unified_diff.splitlines()
    touched: list[str] = []
    saw_hunk = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                return "invalid: missing +++ line after ---"
            old_path = _diff_path_from_header(line[4:])
            new_path = _diff_path_from_header(lines[i + 1][4:])
            for candidate in (new_path, old_path):
                if candidate and candidate != "/dev/null" and candidate not in touched:
                    touched.append(candidate)
            i += 2
            continue
        if line.startswith("@@"):
            saw_hunk = True
        i += 1

    if not touched:
        return "invalid: no file paths found in diff headers"
    if not saw_hunk:
        return "invalid: no diff hunks (@@) found"

    denied: list[str] = []
    blocked: list[str] = []
    for rel in touched:
        norm = _normalize_rel_path(rel)
        if norm is None:
            blocked.append(rel)
            continue
        if not _is_allowed(norm) or _is_denied(norm):
            denied.append(norm)

    if blocked:
        return f"invalid: path traversal in diff: {', '.join(blocked)}"
    if denied:
        return f"invalid: denied paths: {', '.join(denied)}"

    return f"valid: files touched: {', '.join(touched)}"


def _diff_path_from_header(raw: str) -> str:
    path = raw.strip()
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path.lstrip("/")


def repo_list(path: str = "", *, max_entries: int = 200) -> list[str]:
    root = _repo_root()
    sub = path.lstrip("/")
    base = (root / sub).resolve() if sub else root
    if not str(base).startswith(str(root)) or not base.is_dir():
        return []
    entries: list[str] = []
    for fp in sorted(base.iterdir()):
        rel = str(fp.relative_to(root)).replace("\\", "/")
        if _is_denied(rel):
            continue
        if fp.is_dir():
            if _is_allowed(rel + "/"):
                entries.append(rel + "/")
        elif _is_allowed(rel):
            entries.append(rel)
        if len(entries) >= max_entries:
            break
    return entries


def repo_write(*_args: object, **_kwargs: object) -> None:
    raise PermissionError("repo.write blocked by context-exec policy")
