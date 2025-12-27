"""Verb loading utilities for Cortex Orch.

This module exists because the orchestrator (full-fidelity version) imports
`VerbLoader` from `app.verbs`.

It intentionally stays small:
- Reads YAML files from a verbs directory (default points at /app/orion/cognition/verbs)
- Normalizes the step list to `steps` sourced from YAML `plan`
- Adds `_path` for observability

The orchestrator handles prompt rendering and step execution via Cortex Exec.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class VerbLoader:
    verbs_dir: str

    def load_verb(self, verb_name: str) -> Dict[str, Any]:
        """Load a verb YAML and return a normalized dict.

        Expected YAML shape (existing Orion format):
        - name: <verb>
        - plan: [ {name, order, services, prompt_template, ...}, ... ]

        We normalize:
        - steps := plan (or steps if already present)
        """
        root = Path(self.verbs_dir)
        path = root / f"{verb_name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Verb YAML not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Normalize steps
        steps = data.get("steps")
        if steps is None:
            steps = data.get("plan") or []

        out: Dict[str, Any] = dict(data)
        out["steps"] = steps
        out["_path"] = str(path)
        return out
