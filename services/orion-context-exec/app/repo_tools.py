from __future__ import annotations

import re
from pathlib import Path

from .schemas import RepoFile, RepoHit
from .settings import settings

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

ALLOW_PREFIXES = ("orion/", "services/", "docs/", "tests/", "scripts/")


def _repo_root() -> Path:
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


def repo_grep(pattern: str, path: str | None = None, limit: int = 50) -> list[RepoHit]:
    root = _repo_root()
    base = (root / path).resolve() if path else root
    if not str(base).startswith(str(root)):
        return []
    hits: list[RepoHit] = []
    try:
        rx = re.compile(pattern)
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


def repo_write(*_args: object, **_kwargs: object) -> None:
    raise PermissionError("repo.write blocked by context-exec policy")
