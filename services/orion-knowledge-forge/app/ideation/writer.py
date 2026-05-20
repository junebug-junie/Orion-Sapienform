from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

_CANONICAL_PREFIXES = (
    "claims/accepted/",
    "claims/disputed/",
    "specs/execution_ready/",
    "decisions/",
)


def slugify_task(task: str, *, max_len: int = 48) -> str:
    folded = task.casefold().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")
    return (slug or "ideation")[:max_len]


def build_artifact_path(corpus_root: Path, *, task: str, now: datetime | None = None) -> Path:
    ts = now or datetime.now(timezone.utc)
    stamp = ts.strftime("%Y-%m-%d-%H%M%S")
    slug = slugify_task(task)
    rel = Path("reviews") / "pending" / f"ideation-{stamp}-{slug}.md"
    return (corpus_root / rel).resolve()


def assert_safe_artifact_path(corpus_root: Path, artifact_path: Path) -> None:
    root = corpus_root.resolve()
    resolved = artifact_path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("artifact path escapes corpus root")
    rel = resolved.relative_to(root).as_posix()
    if not rel.startswith("reviews/pending/"):
        raise ValueError(f"ideation may only write under reviews/pending/: {rel}")
    for forbidden in _CANONICAL_PREFIXES:
        if rel.startswith(forbidden):
            raise ValueError(f"ideation must not write canonical path: {rel}")


def write_review_artifact(
    corpus_root: Path,
    *,
    task: str,
    mode: str,
    content: str,
    run_id: str,
) -> Path:
    artifact_path = build_artifact_path(corpus_root, task=task)
    assert_safe_artifact_path(corpus_root, artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "---",
            f"run_id: {run_id}",
            f"mode: {mode}",
            "status: proposed",
            "target: review_artifact",
            "---",
            "",
            "> Proposals are not canonical truth. Pending human review only.",
            "",
        ]
    )
    artifact_path.write_text(header + content.strip() + "\n", encoding="utf-8")
    return artifact_path
