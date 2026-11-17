"""N64 cartridge saves survive sessions (battery.srm), and warm restore.
Live 2026-11-17: an N64 save state omits cartridge memory, so an in-game save was
lost when the session ended; and a state restored before the first frame failed."""
import importlib.util
from pathlib import Path

import pytest

from app.saves import SaveStore
from app.worker import Battery
from orion.retro.probe import CoreProbe
from settings import Settings


class FakeCore:
    def __init__(self, size=16): self.ram = bytearray(size)
    def save_ram(self): return bytes(self.ram)
    def load_save_ram(self, data): self.ram[:] = data


def test_battery_round_trips_across_sessions(tmp_path):
    store, records = SaveStore(tmp_path), []
    first = FakeCore(); battery = Battery(first, store, 'oot', lambda *a, **k: records.append(k))
    first.ram[3] = 7                     # the game saves in-game
    battery.flush(now=battery.at + 1)    # inside the 5 s period: nothing written yet
    assert store.read_battery('oot') is None
    battery.flush(force=True)
    second = FakeCore(); Battery(second, store, 'oot', lambda *a, **k: records.append(k))
    assert second.ram[3] == 7            # the next session starts with it
    assert {'action': 'loaded', 'bytes': 16} in records


def test_unchanged_memory_is_not_rewritten_and_wrong_size_is_left_alone(tmp_path):
    store = SaveStore(tmp_path); store.write_battery('oot', b'x' * 8)
    records = []
    core = FakeCore(16); battery = Battery(core, store, 'oot', lambda *a, **k: records.append(k))
    assert core.ram == bytearray(16) and records == [{'action': 'size_mismatch', 'bytes': 8}]
    battery.flush(force=True)            # nothing changed: the odd file is not touched
    assert store.read_battery('oot') == b'x' * 8


def test_disk_fault_is_recorded_not_fatal(tmp_path):
    store, records = SaveStore(tmp_path / 'missing'), []
    core = FakeCore(); battery = Battery(core, store, 'oot', lambda *a, **k: records.append(k))
    core.ram[0] = 1
    battery.flush(force=True)
    assert records[-1] == {'action': 'write_failed', 'error': 'saves_unavailable'}


def test_battery_file_never_shows_up_as_a_save_or_gets_pruned(tmp_path):
    store = SaveStore(tmp_path, max_per_game=1)
    store.write_battery('oot', b'b' * 4)
    identity = dict(rom_sha256='r' * 64, core_sha256='c' * 64)
    for _ in range(2): store.write('oot', identity, b'state', frame=1, played_frames=1, session_id='s')
    assert len(store.list('oot')) == 1 and store.read_battery('oot') == b'b' * 4


def test_only_n64_keeps_a_battery_and_warm_restores():
    s = Settings()
    assert s.battery('n64') and not s.battery('snes') and not s.battery('nes')
    assert s.core_host('n64')['warm_restore'] is True


def test_warm_restore_runs_one_frame_first():
    probe = CoreProbe.__new__(CoreProbe)
    probe.loaded, probe.warm_restore, probe.frames_run = True, True, 0
    calls = []
    probe.step = lambda *a, **k: (calls.append('step'), setattr(probe, 'frames_run', 1))
    probe.lib = type('L', (), {'retro_serialize_size': lambda self: 4,
                               'retro_unserialize': lambda self, b, n: calls.append('restore') or True})()
    probe.restore(b'abcd')
    probe.restore(b'abcd')
    assert calls == ['step', 'restore', 'restore']


def test_import_overlays_each_region_at_the_core_offsets():
    spec = importlib.util.spec_from_file_location(
        'import_n64_save', Path(__file__).parents[1]/'evals/import_n64_save.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = bytes([0xEE]) * 0x48800
    image = mod.battery_image(base, {'eep': b'\1' * 512, 'sra': b'\2' * 0x8000, 'fla': b'\3' * 0x20000,
                                     'mpk': b'\4' * 0x20000})
    assert image[:512] == b'\1' * 512 and image[512:0x800] == b'\xee' * (0x800 - 512)  # 4 kbit EEPROM
    assert image[0x800:0x20800] == b'\4' * 0x20000
    assert image[0x20800:0x28800] == b'\2' * 0x8000 and image[0x28800:] == b'\3' * 0x20000
    with pytest.raises(ValueError, match='at most'): mod.battery_image(base, {'eep': b'\0' * 0x801})


@pytest.mark.parametrize('stored,action', [(None, 'load_failed'), (b'x' * 8, 'size_mismatch')])
def test_an_unloadable_battery_is_never_overwritten(tmp_path, stored, action):
    # Review 2026-11-17: a blank image must not replace a file that exists but was not loaded.
    store, records = SaveStore(tmp_path), []
    if stored: store.write_battery('oot', stored)
    else:
        store.read_battery = lambda game_id: (_ for _ in ()).throw(ValueError('battery_io_failed'))
    core = FakeCore(); battery = Battery(core, store, 'oot', lambda *a, **k: records.append(k))
    core.ram[0] = 9
    battery.flush(force=True)
    assert records[0]['action'] == action
    if stored: assert SaveStore(tmp_path).read_battery('oot') == stored


def test_warm_snapshot_runs_one_frame_first():
    # Live 2026-11-17: N64 retro_serialize fails before the first frame too.
    probe = CoreProbe.__new__(CoreProbe)
    probe.loaded, probe.warm_restore, probe.frames_run = True, True, 0
    calls = []
    probe.step = lambda *a, **k: (calls.append('step'), setattr(probe, 'frames_run', 1))
    probe.lib = type('L', (), {'retro_serialize_size': lambda self: 4,
                               'retro_serialize': lambda self, b, n: calls.append('serialize') or True})()
    probe.snapshot()
    assert calls == ['step', 'serialize']
