"""N64 catalog, input (buttons + analog sticks), frontend support and media bound.
No commercial ROM fixtures; the real core runs only in the container smoke."""
from dataclasses import replace
import ctypes as C
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.contracts import Input
from app.control import Arbiter, N64_MASK
from app.media import MediaProbe
from app.session import Sessions
from orion.retro.probe import CoreProbe, Variable, digest
from settings import Settings
from test_authority import catalog, intent, settings


def with_n64(tmp_path, file='kart.z64'):
    cfg = replace(settings(tmp_path), n64_core=tmp_path/'n64.so', saves=tmp_path/'saves')
    cfg.saves.mkdir()
    rows = catalog(cfg, tmp_path)
    cfg.n64_core.write_bytes(b'n64-core')
    rom = tmp_path/file; rom.write_bytes(b'local n64 fixture')
    rows.append(dict(game_id='kart64', title='Mario Kart 64', system='n64', file=rom.name,
                     rom_sha256=digest(rom), ports=[0, 1], provenance='fixture'))
    cfg.catalog.write_text(json.dumps(rows))
    return cfg, rows


@pytest.mark.parametrize('file', ['kart.n64', 'kart.z64', 'kart.V64'])
def test_n64_catalog_uses_its_own_core(tmp_path, file):
    cfg, _ = with_n64(tmp_path, file)
    sessions = Sessions(cfg)
    assert ('Mario Kart 64', 'n64') in [(g['title'], g['system']) for g in sessions.games()]
    assert sessions.identity('kart64')['core_sha256'] == digest(cfg.n64_core)
    # An N64 state (~16.8 MB) fits N64's store; NES keeps its own smaller bound.
    assert sessions.store('kart64').max_bytes == cfg.n64_save_max_bytes
    assert sessions.store('test').max_bytes == cfg.save_max_bytes


def test_missing_n64_core_refuses_startup(tmp_path):
    cfg, _ = with_n64(tmp_path)
    cfg.n64_core.unlink()
    with pytest.raises(ValueError, match='n64 core missing'): Sessions(cfg)


def test_core_host_is_software_render_only_for_n64():
    s = Settings()
    assert s.core_host('nes') == {} and s.core_host('snes') == {}
    host = s.core_host('n64')
    assert host['options']['mupen64plus-rdp-plugin'] == 'angrylion'
    assert host['options']['mupen64plus-rsp-plugin'] == 'cxd4'
    # GoldenEye's microcode is the heaviest class and drops 3 frames in a row (a
    # deadline_overrun pause) at 8 threads, so the count is pinned, not just bounded.
    assert int(host['options']['mupen64plus-angrylion-multithread']) == 16
    assert s.save_limit('n64') > 16_793_412 > s.save_limit('snes')


def test_render_threads_have_cpu_quota_and_pid_room_to_match():
    """Threads without quota are throttled threads: raising the render thread count
    without raising `cpus` leaves GoldenEye pausing exactly as before. Pin the pair."""
    import yaml
    threads = int(Settings().core_host('n64')['options']['mupen64plus-angrylion-multithread'])
    root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load((root/'docker-compose.yml').read_text())['services']['retro']
    # The core and the worker run alongside the render threads, so quota exceeds them.
    assert compose['cpus'] > threads
    # Every render thread is a thread, plus the core's own: pids must not be the limit.
    assert compose['pids_limit'] > threads + 8
    assert compose['memswap_limit'] == compose['mem_limit']  # no silent swap escape


def test_n64_buttons_z_accepted_select_refused():
    n64 = Arbiter([0, 1], 'n64')
    grant = n64.control('one', 0, 'acquire', 0)
    n64.frame(0)
    n64.submit('one', intent(grant, mask=4096 | 1 | 8), .01)  # Z, A, Start
    assert n64.frame(.02)[0] == [4096 | 1 | 8, 0]
    with pytest.raises(ValueError, match='unsupported_buttons'):
        n64.submit('one', intent(grant, seq=2, mask=4), .03)  # Select is not an N64 button
    assert N64_MASK & 4 == 0 and N64_MASK & (1 << 12)


def test_axes_reach_only_the_live_n64_port_and_centre_on_expiry():
    n64 = Arbiter([0, 1], 'n64')
    grant = n64.control('one', 0, 'acquire', 0)
    n64.frame(0)
    assert n64.axes() == ((0, 0, 0, 0), (0, 0, 0, 0))  # neutral frame after acquire
    n64.submit('one', dict(intent(grant, mask=0), axes=[32767, -32767, 0, 100]), .01)
    n64.frame(.02)
    assert n64.axes() == ((32767, -32767, 0, 100), (0, 0, 0, 0))
    n64.submit('one', dict(intent(grant, seq=2, mask=0), axes=None), .03)  # omitted = centred
    n64.frame(.04)
    assert n64.axes()[0] == (0, 0, 0, 0)
    n64.submit('one', dict(intent(grant, seq=3, mask=0), axes=[9, 9, 9, 9]), .05)
    n64.frame(10)  # lease expired
    assert n64.axes() == ((0, 0, 0, 0), (0, 0, 0, 0))


def test_axes_are_refused_for_consoles_without_sticks():
    snes = Arbiter([0], 'snes')
    grant = snes.control('one', 0, 'acquire', 0)
    with pytest.raises(ValueError, match='unsupported_axes'):
        snes.submit('one', dict(intent(grant), axes=[0, 0, 0, 0]), .01)


def test_input_contract_bounds_axes():
    base = dict(version=1, session_id='s'*36, generation='g'*36, lease='l'*32, epoch=1, seq=0, port=0, mask=0)
    assert Input(**base).axes is None
    assert Input(**base, axes=[-32767, 32767, 0, 1]).axes == [-32767, 32767, 0, 1]
    for bad in ([0, 0, 0], [0, 0, 0, 0, 0], [40000, 0, 0, 0], [-32768, 0, 0, 0]):
        with pytest.raises(ValidationError): Input(**base, axes=bad)


def bare(cls=CoreProbe, **extra):
    probe = cls.__new__(cls)
    probe.options, probe.unsupported = {}, set()
    probe.pixel_format = probe.video_frames = 0
    probe.frame_hash = probe.error = None
    probe.buttons = (0, 0); probe.axes = ((1, -2, 3, -4), (0, 0, 0, 0))
    probe.__dict__.update(extra)
    return probe


def test_probe_answers_analog_reads_per_port():
    probe = bare()
    assert [probe._input(0, 5, i, b) for i in (0, 1) for b in (0, 1)] == [1, -2, 3, -4]
    assert probe._input(1, 5, 0, 0) == 0 and probe._input(0, 5, 2, 0) == 0


def test_scratch_and_log_are_given_only_when_configured(tmp_path):
    out = C.c_char_p()
    assert not bare()._environment(9, C.addressof(out))  # NES/SNES keep refusing a directory
    probe = bare(_scratch=C.c_char_p(str(tmp_path).encode()), _log=None)
    from orion.retro.probe import LogCallback, Log
    probe._log = LogCallback(Log(lambda level, fmt: None))
    assert probe._environment(9, C.addressof(out)) and out.value == str(tmp_path).encode()
    assert probe._environment(31, C.addressof(out))
    log = LogCallback()
    assert probe._environment(27, C.addressof(log)) and log.log


def test_option_overrides_replace_declared_defaults():
    probe = bare(overrides={b'core-rdp': b'angrylion'})
    variables = (Variable * 3)(Variable(b'core-rdp', b'RDP; gliden64|angrylion'),
                               Variable(b'core-other', b'Other; a|b'), Variable(None, None))
    assert probe._environment(16, C.addressof(variables))
    assert probe.options == {b'core-rdp': b'angrylion', b'core-other': b'a'}


def test_an_override_the_core_never_declared_fails_loudly():
    # A core bump renaming the renderer key would otherwise fall back to GL and die unexplained.
    probe = bare(overrides={b'core-renamed': b'angrylion'})
    variables = (Variable * 2)(Variable(b'core-rdp', b'RDP; gliden64|angrylion'), Variable(None, None))
    assert not probe._environment(16, C.addressof(variables))
    assert probe.error == 'core does not declare option core-renamed'


def test_media_halves_angrylion_640x480_to_the_native_320x240():
    p = bare(MediaProbe, video=b'', pcm=bytearray(), geometry=(0, 0))
    p.pixel_format = 1
    width, height = 640, 480
    pixels = (C.c_uint32 * (width * height))(*range(width * height))
    p._video(C.addressof(pixels), width, height, width * 4)
    assert p.error is None and p.geometry == (320, 240)
    got = memoryview(p.video).cast('I')
    assert len(got) == 320 * 240
    assert got[0] == 0 and got[1] == 2 and got[320] == 2 * width  # every other pixel and row


def test_media_halves_only_the_wide_axis_of_640x240():
    # Live 2026-11-16: halving both axes turned Angrylion's 640x240 into a squashed 320x120.
    p = bare(MediaProbe, video=b'', pcm=bytearray(), geometry=(0, 0))
    p.pixel_format = 1
    pixels = (C.c_uint32 * (640 * 240))(*range(640 * 240))
    p._video(C.addressof(pixels), 640, 240, 640 * 4)
    assert p.error is None and p.geometry == (320, 240)
    got = memoryview(p.video).cast('I')
    assert len(got) == 320 * 240 and got[320] == 640  # row 1 kept, not skipped


def test_media_still_refuses_pictures_beyond_twice_the_bound():
    p = bare(MediaProbe, video=b'', pcm=bytearray(), geometry=(0, 0))
    p.pixel_format = 1
    pixels = (C.c_uint32 * (1100 * 2))()
    p._video(C.addressof(pixels), 1100, 2, 1100 * 4)
    assert p.error == 'media_geometry_limit'


def test_importer_accepts_n64_rom_images(tmp_path):
    spec = importlib.util.spec_from_file_location('import_rom', Path(__file__).parents[1]/'evals/import_rom.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg, original = with_n64(tmp_path)
    rom = tmp_path/'oot.z64'; rom.write_bytes(b'another n64 fixture')
    row = mod.import_rom(cfg.catalog, rom, 'ocarina', 'The Legend of Zelda: Ocarina of Time', 1)
    assert row['system'] == 'n64' and row['file'] == 'ocarina.z64'
    assert len(Sessions(cfg).games()) == len(original) + 1


def test_media_probe_forwards_core_host_options(monkeypatch, tmp_path):
    # Live 2026-11-16: MediaProbe(*args) dropped scratch/options, so every streamed
    # N64 session failed at start while CoreProbe-only tests passed.
    seen = {}
    monkeypatch.setattr(CoreProbe, '__init__', lambda self, core, rom, **kw: seen.update(kw))
    host = Settings().core_host('n64')
    MediaProbe(tmp_path/'core.so', tmp_path/'game.z64', **host)
    assert seen == host
