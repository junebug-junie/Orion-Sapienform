from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import dataclass_to_dict


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = dataclass_to_dict(row) if hasattr(row, "__dataclass_fields__") else row
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
