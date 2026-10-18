"""Boot configuration; secrets are files, never included in evidence."""
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    enabled: bool = False
    catalog: Path = Path('/catalog/catalog.json')
    artifacts: Path = Path('/evidence')
    core: Path = Path('/opt/retro/core.so')
    token_file: Path = Path('/run/secrets/retro_operator')
    session_bytes: int = 268435456
    total_bytes: int = 2147483648

    @classmethod
    def from_env(cls):
        return cls(enabled=os.getenv('RETRO_ENABLED', 'false').lower() == 'true')

    def token(self):
        value = self.token_file.read_text().strip()
        if len(value) < 32:
            raise ValueError('operator credential must contain at least 32 characters')
        return value
