from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from orion.knowledge_forge.yaml_doc import save_yaml_doc

_FRONTMATTER_RE = re.compile(r"^---\n(?P<meta>.*?)\n---\n", re.DOTALL)
_FENCE_RE = re.compile(r"```yaml\n(?P<body>.*?)\n```", re.DOTALL)

EXECUTION_READY_PREFIX = "specs/execution_ready/"


@dataclass
class PendingPatch:
    patch_id: str
    target: str
    action: str
    path: Path


def list_pending_patches(corpus_root: Path) -> list[PendingPatch]:
    pending_dir = corpus_root / "reviews" / "pending"
    if not pending_dir.is_dir():
        return []
    out: list[PendingPatch] = []
    for path in sorted(pending_dir.glob("*.patch.md")):
        meta, _ = _parse_patch_file(path)
        out.append(
            PendingPatch(
                patch_id=str(meta["patch_id"]),
                target=str(meta["target"]),
                action=str(meta["action"]),
                path=path,
            )
        )
    return out


def apply_pending_patch(corpus_root: Path, patch_id: str) -> Path:
    pending_dir = corpus_root / "reviews" / "pending"
    for path in pending_dir.glob("*.patch.md"):
        meta, body = _parse_patch_file(path)
        if meta.get("patch_id") != patch_id:
            continue
        target_rel = str(meta["target"])
        if target_rel.startswith(EXECUTION_READY_PREFIX):
            # v0: execution_ready writes must come from reviewed specs via compile promotion,
            # not direct agent patch — force path through specs/design or specs/plan first.
            raise ValueError(
                "direct patches to specs/execution_ready/ are forbidden; "
                "promote via reviewed spec + human approval"
            )
        doc = yaml.safe_load(body)
        if not isinstance(doc, dict):
            raise ValueError(f"patch body must be YAML mapping: {path}")
        target_path = corpus_root / target_rel
        if meta.get("action") == "create" and target_path.exists():
            raise ValueError(f"refusing to overwrite existing target: {target_path}")
        save_yaml_doc(target_path, doc)
        accepted_dir = corpus_root / "reviews" / "accepted"
        accepted_dir.mkdir(parents=True, exist_ok=True)
        path.replace(accepted_dir / path.name)
        return target_path
    raise FileNotFoundError(f"pending patch not found: {patch_id}")


def _parse_patch_file(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"missing frontmatter: {path}")
    meta = yaml.safe_load(m.group("meta"))
    if not isinstance(meta, dict):
        raise ValueError(f"invalid frontmatter: {path}")
    rest = text[m.end() :]
    fm = _FENCE_RE.search(rest)
    if not fm:
        raise ValueError(f"missing yaml fenced block: {path}")
    return {str(k): str(v) for k, v in meta.items()}, fm.group("body")
