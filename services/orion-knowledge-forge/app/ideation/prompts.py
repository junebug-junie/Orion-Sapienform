from __future__ import annotations

from pathlib import Path

STATIC_KNOWLEDGE_FORGE_CONTRACT = """You are an Orion Knowledge Forge ideation assistant.

Rules:
- Proposals are NOT canonical truth.
- Do NOT mutate accepted claims, specs, or decisions.
- Disputed and stale claims are excluded from execution advice by default.
- Write outputs as review artifacts only (reviews/pending), never silent canonical edits.

Required sections in your response:
## Current shape
## Arsonist critique
## Missing questions
## Proposed v-next
## Files likely to touch
## Non-goals
## Acceptance checks
"""

_MAX_SNIPPET_BYTES = 32_000
_MAX_FILES_PER_PATH = 12


def build_user_prompt(
    *,
    task: str,
    mode: str,
    input_paths: list[str],
    corpus_root: Path,
    monorepo_root: Path | None,
    status_summary: str,
) -> str:
    sections = [
        f"Mode: {mode}",
        f"Task: {task}",
        "",
        "## Corpus status",
        status_summary,
        "",
        "## Input path snippets",
    ]
    for path_str in input_paths:
        resolved = resolve_input_path(path_str, corpus_root=corpus_root, monorepo_root=monorepo_root)
        if resolved is None:
            sections.append(f"- {path_str}: (not found)")
            continue
        snippets = _collect_snippets(resolved)
        sections.append(f"- {path_str}:")
        if not snippets:
            sections.append("  (no readable files)")
        else:
            sections.extend(snippets)
    return "\n".join(sections)


def resolve_input_path(
    path_str: str,
    *,
    corpus_root: Path,
    monorepo_root: Path | None,
) -> Path | None:
    raw = Path(path_str)
    if raw.is_absolute() and raw.exists():
        return raw.resolve()
    for base in (corpus_root, monorepo_root):
        if base is None:
            continue
        candidate = (base / path_str).resolve()
        if candidate.exists():
            return candidate
    return None


def detect_monorepo_root(corpus_root: Path) -> Path | None:
    if corpus_root.name == "orion-knowledge" and corpus_root.parent.is_dir():
        return corpus_root.parent.resolve()
    for parent in [corpus_root, *corpus_root.parents]:
        if (parent / "services" / "orion-knowledge-forge").is_dir():
            return parent.resolve()
    return None


def _collect_snippets(path: Path) -> list[str]:
    if path.is_file():
        return [_read_snippet(path)]
    if not path.is_dir():
        return []
    files = sorted(p for p in path.rglob("*") if p.is_file())[:_MAX_FILES_PER_PATH]
    return [_read_snippet(p) for p in files]


def _read_snippet(path: Path) -> str:
    try:
        data = path.read_bytes()[:_MAX_SNIPPET_BYTES]
        text = data.decode("utf-8", errors="replace")
    except OSError:
        return f"  - {path}: (unreadable)"
    rel = path.name
    return f"  - {rel}:\n```\n{text}\n```"
