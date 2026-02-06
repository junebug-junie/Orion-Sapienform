from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


DEFAULT_TAXONOMY = [
    "Infra",
    "Product",
    "Planning",
    "Debugging",
    "Communication",
    "Learning",
    "Execution",
    "Decision",
    "Risk",
    "Support",
    "Quality",
    "Performance",
    "Security",
    "Data",
    "Other",
]


def load_taxonomy(path: Optional[str]) -> List[str]:
    if not path:
        return DEFAULT_TAXONOMY
    taxonomy_path = Path(path)
    if not taxonomy_path.exists():
        return DEFAULT_TAXONOMY
    try:
        data = json.loads(taxonomy_path.read_text())
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            return data
    except Exception:
        return DEFAULT_TAXONOMY
    return DEFAULT_TAXONOMY
