from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import SecurityState
from .settings import Settings


class SecurityStateStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = Path(settings.SECURITY_STATE_PATH)
        self._state: Optional[SecurityState] = None

    def load(self) -> SecurityState:
        if self._state is not None:
            return self._state

        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text())
                self._state = SecurityState(
                    enabled=self.settings.SECURITY_ENABLED,
                    armed=data.get("armed", self.settings.SECURITY_DEFAULT_ARMED),
                    mode=data.get("mode", self.settings.SECURITY_MODE),
                    updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
                    updated_by=data.get("updated_by"),
                )
                return self._state
            except Exception:
                pass

        # Default
        self._state = SecurityState(
            enabled=self.settings.SECURITY_ENABLED,
            armed=self.settings.SECURITY_DEFAULT_ARMED,
            mode=self.settings.SECURITY_MODE,
            updated_at=None,
            updated_by=None,
        )
        return self._state

    def save(
        self,
        armed: Optional[bool] = None,
        mode: Optional[str] = None,
        updated_by: str = "api",
    ) -> SecurityState:
        current = self.load()

        new_state = SecurityState(
            enabled=self.settings.SECURITY_ENABLED,
            armed=current.armed if armed is None else armed,
            mode=current.mode if mode is None else mode,
            updated_at=datetime.utcnow(),
            updated_by=updated_by,
        )
        self._state = new_state

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {
                        "armed": new_state.armed,
                        "mode": new_state.mode,
                        "updated_at": new_state.updated_at.isoformat() if new_state.updated_at else None,
                        "updated_by": new_state.updated_by,
                    },
                    indent=2,
                )
            )
        except Exception:
            pass

        return new_state
