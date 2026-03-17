from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

from orion.core.schemas.concept_induction import ConceptProfile


class LocalProfileStore:
    """Minimal JSON-backed store for latest profiles."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load_raw(self) -> Dict[str, Dict]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {}

    def _save_raw(self, data: Dict[str, Dict]) -> None:
        self.path.write_text(json.dumps(data, indent=2))

    def load(self, subject: str) -> Optional[ConceptProfile]:
        data = self._load_raw()
        profiles = data.get("profiles") or data
        if subject not in profiles:
            return None
        try:
            raw = dict(profiles[subject])
            raw.pop("_hash", None)
            return ConceptProfile.model_validate(raw)
        except Exception:
            return None

    def save(self, subject: str, profile: ConceptProfile, profile_hash: str) -> None:
        data = self._load_raw()
        data.setdefault("profiles", {})
        data.setdefault("_hashes", {})

        serialized = profile.model_dump(mode="json")
        data["profiles"][subject] = serialized
        data["_hashes"][subject] = profile_hash
        self._save_raw(data)

    def load_hash(self, subject: str) -> Optional[str]:
        data = self._load_raw()
        if "_hashes" in data:
            return data.get("_hashes", {}).get(subject)
        if subject in data:
            return data[subject].get("_hash")
        return None

    def load_drive_state(self, subject: str) -> Dict[str, Any]:
        data = self._load_raw()
        states = data.get("drive_states", {})
        if not isinstance(states, dict):
            return {}
        state = states.get(subject)
        return state if isinstance(state, dict) else {}

    def save_drive_state(
        self,
        subject: str,
        *,
        pressures: Dict[str, float],
        activations: Dict[str, bool],
        updated_at: datetime,
    ) -> None:
        data = self._load_raw()
        data.setdefault("drive_states", {})
        data["drive_states"][subject] = {
            "pressures": pressures,
            "activations": activations,
            "updated_at": updated_at.isoformat(),
        }
        self._save_raw(data)
