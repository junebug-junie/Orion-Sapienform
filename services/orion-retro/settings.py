"""Boot configuration; secrets are files, never included in evidence."""
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    enabled: bool = False
    cognition_enabled: bool = False
    catalog: Path = Path('/catalog/catalog.json')
    artifacts: Path = Path('/evidence')
    core: Path = Path('/opt/retro/core.so')
    token_file: Path = Path('/run/secrets/retro_operator')
    hub_token_file: Path = Path('/run/secrets/retro_hub')
    session_bytes: int = 268435456
    total_bytes: int = 2147483648
    # Host bind mount (RETRO_SAVES_DIR). Missing or read-only -> saves are disabled.
    saves: Path = Path('/saves')
    save_max_per_game: int = 20
    save_max_bytes: int = 4194304

    @classmethod
    def from_env(cls):
        return cls(enabled=os.getenv('RETRO_ENABLED', 'false').lower() == 'true',
                   cognition_enabled=os.getenv('RETRO_COGNITION_ENABLED', 'false').lower() == 'true')

    def token(self):
        value = self.token_file.read_text().strip()
        if len(value) < 32:
            raise ValueError('operator credential must contain at least 32 characters')
        return value
