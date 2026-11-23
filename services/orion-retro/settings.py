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
    snes_core: Path = Path('/opt/retro/snes9x.so')
    n64_core: Path = Path('/opt/retro/mupen64plus_next.so')
    # Private scratch on the container's tmpfs for the N64 core's system directory
    # (it copies an ini there). Cartridge saves never go here: see battery().
    n64_scratch: Path = Path('/tmp/n64')
    token_file: Path = Path('/run/secrets/retro_operator')
    hub_token_file: Path = Path('/run/secrets/retro_hub')
    session_bytes: int = 268435456
    total_bytes: int = 2147483648
    # Host bind mount (RETRO_SAVES_DIR, required by compose). If /saves is absent or
    # read-only (e.g. a bare `docker run`), save_enabled is false and save/load refuse.
    saves: Path = Path('/saves')
    save_max_per_game: int = 20
    save_max_bytes: int = 4194304
    n64_save_max_bytes: int = 33554432  # an N64 state is ~16.8 MB (8 MB RDRAM + device state)

    @classmethod
    def from_env(cls):
        return cls(enabled=os.getenv('RETRO_ENABLED', 'false').lower() == 'true',
                   cognition_enabled=os.getenv('RETRO_COGNITION_ENABLED', 'true').lower() == 'true')

    def token(self):
        value = self.token_file.read_text().strip()
        if len(value) < 32:
            raise ValueError('operator credential must contain at least 32 characters')
        return value

    def core_for(self, system):
        if system == 'nes': return self.core
        if system == 'snes': return self.snes_core
        if system == 'n64': return self.n64_core
        raise ValueError('unsupported system')

    def core_host(self, system):
        """Extra frontend support one core needs. The service has no GPU or GL context,
        so N64 renders in software: Angrylion RDP + jttl (ParaLLEl) RSP, 16 render
        threads (fits pids_limit; the 'all threads' default spawns one per host CPU).
        GoldenEye's microcode is the heaviest class there is: on 8 render threads it
        missed 3 frames in a row (`deadline_overrun` pauses), and on 16 threads CXD4 --
        an interpreted LLE RSP -- became the serial wall: display-list generation runs
        one-threaded through it while the render threads idle (measured 2026-11-22:
        ~35 fps effective, ~5 cores of a 20-CPU quota, nr_throttled=0). jttl JIT-
        compiles the RSP microcode instead of interpreting it; upstream pairs it with
        Angrylion by default on modern desktop targets. ParaLLEl-RDP is not an option
        here: it requires a Vulkan hw-render context our frontend does not provide."""
        if system != 'n64': return {}
        return dict(scratch=self.n64_scratch, warm_restore=True, options={
            'mupen64plus-rdp-plugin': 'angrylion', 'mupen64plus-rsp-plugin': 'parallel',
            'mupen64plus-angrylion-multithread': '16'})

    def battery(self, system):
        """Persist cartridge save memory per game (<saves>/<game>/battery.srm). N64 save
        states omit it (verified 2026-11-17: a state restored into a fresh core leaves its
        SRAM unformatted-blank, and a restore leaves existing SRAM untouched), so without
        this an in-game save is lost when the session ends."""
        return system == 'n64'

    def save_limit(self, system):
        return self.n64_save_max_bytes if system == 'n64' else self.save_max_bytes
