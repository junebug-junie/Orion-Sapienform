from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import re

import yaml

import orion

ORION_PKG_DIR = Path(orion.__file__).resolve().parent
VERBS_DIR = ORION_PKG_DIR / "cognition" / "verbs"
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class VerbInfo:
    name: str
    label: str
    description: str
    category: str
    version: str
    timeout_ms: int
    services: list[str]
    steps: list[str]
    tags: list[str] | None = None


def _tokenize(text: str) -> set[str]:
    return {tok for tok in _TOKEN_RE.findall((text or "").lower()) if tok}


def load_verb_catalog() -> list[VerbInfo]:
    catalog: list[VerbInfo] = []
    for path in sorted(VERBS_DIR.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or path.stem).strip()
        if not name:
            continue
        raw_steps = raw.get("steps") or raw.get("plan") or []
        step_names: list[str] = []
        if isinstance(raw_steps, list):
            for item in raw_steps:
                if isinstance(item, dict):
                    step_name = str(item.get("name") or "").strip()
                    if step_name:
                        step_names.append(step_name)
        tags = raw.get("tags")
        tags_list = [str(t).strip() for t in tags if str(t).strip()] if isinstance(tags, list) else None
        catalog.append(
            VerbInfo(
                name=name,
                label=str(raw.get("label") or name),
                description=str(raw.get("description") or ""),
                category=str(raw.get("category") or "general"),
                version=str(raw.get("version") or ""),
                timeout_ms=int(raw.get("timeout_ms") or 120000),
                services=[str(s) for s in (raw.get("services") or []) if str(s).strip()],
                steps=step_names,
                tags=tags_list,
            )
        )
    return catalog


def filter_allowed(
    catalog: Sequence[VerbInfo],
    allowlist_names: Iterable[str] | None = None,
    allow_categories: Iterable[str] | None = None,
    denylist_names: Iterable[str] | None = None,
) -> list[VerbInfo]:
    allow_names = {str(v).strip().lower() for v in (allowlist_names or []) if str(v).strip()}
    allow_cats = {str(v).strip().lower() for v in (allow_categories or []) if str(v).strip()}
    deny_names = {str(v).strip().lower() for v in (denylist_names or []) if str(v).strip()}

    out: list[VerbInfo] = []
    for verb in catalog:
        name_l = verb.name.lower()
        cat_l = verb.category.lower()
        if deny_names and name_l in deny_names:
            continue
        if allow_names and name_l not in allow_names:
            continue
        if allow_cats and cat_l not in allow_cats:
            continue
        out.append(verb)
    return out


def rank_verbs_for_query(catalog: Sequence[VerbInfo], text: str, k: int = 10) -> list[VerbInfo]:
    query_tokens = _tokenize(text)
    scored: list[tuple[int, str, VerbInfo]] = []
    for verb in catalog:
        name_tokens = _tokenize(verb.name.replace("-", "_").replace(" ", "_"))
        doc_tokens = _tokenize(" ".join([verb.label, verb.description, verb.category]))
        overlap_name = len(query_tokens & name_tokens)
        overlap_doc = len(query_tokens & doc_tokens)
        score = overlap_name * 4 + overlap_doc
        scored.append((score, verb.name, verb))

    scored.sort(key=lambda row: (-row[0], row[1]))
    picked = [v for _, _, v in scored[: max(1, k)]]
    return picked


def serialize_shortlist(verblist: Sequence[VerbInfo], max_chars: int = 2000) -> str:
    lines: list[str] = []
    for v in verblist:
        line = f"- {v.name}: {v.label} [{v.category}] — {v.description.strip()}"
        if len("\n".join(lines + [line])) > max_chars:
            break
        lines.append(line)
    return "\n".join(lines)
