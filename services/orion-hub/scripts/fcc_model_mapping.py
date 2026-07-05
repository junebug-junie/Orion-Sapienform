"""Map FCC env key labels to stable Claude tier model ids for claude CLI --model."""
from __future__ import annotations

DEFAULT_FCC_MODEL_LABEL = "MODEL"

_LABEL_TO_CLAUDE_MODEL: dict[str, str] = {
    "MODEL": "claude-sonnet-4-20250514",
    "MODEL_OPUS": "claude-opus-4-20250514",
    "MODEL_SONNET": "claude-sonnet-4-20250514",
    "MODEL_HAIKU": "claude-haiku-4-20250514",
}


def label_to_claude_model_id(label: str) -> str:
    key = str(label or "").strip()
    if key not in _LABEL_TO_CLAUDE_MODEL:
        raise ValueError(f"unknown fcc model label: {label!r}")
    return _LABEL_TO_CLAUDE_MODEL[key]
