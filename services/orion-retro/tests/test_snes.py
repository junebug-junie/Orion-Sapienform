"""Mixed catalog, core identity and input contract; no commercial ROM fixtures."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

from app.control import Arbiter
from app.session import Sessions
from app.saves import SaveStore
from orion.retro.probe import digest
from test_authority import catalog, intent, settings


def mixed(tmp_path):
    cfg = replace(settings(tmp_path), snes_core=tmp_path/'snes.so', saves=tmp_path/'saves')
    cfg.saves.mkdir()
    rows = catalog(cfg, tmp_path)  # legacy NES row deliberately omits system
    cfg.snes_core.write_bytes(b'different-snes-core')
    rom = tmp_path/'kart.sfc'; rom.write_bytes(b'local test fixture')
    rows.append(dict(game_id='kart', title='Super Mario Kart', system='snes', file=rom.name,
                     rom_sha256=digest(rom), ports=[0, 1], provenance='fixture'))
    cfg.catalog.write_text(json.dumps(rows))
    return cfg, rows


def test_mixed_catalog_uses_each_games_core_for_save_compatibility(tmp_path):
    cfg, _ = mixed(tmp_path)
    sessions = Sessions(cfg)
    assert [(g['title'], g['system']) for g in sessions.games()] == [
        ('Test', 'nes'), ('Super Mario Kart', 'snes')]
    assert sessions.identity('test')['core_sha256'] == digest(cfg.core)
    snes = sessions.identity('kart')
    assert snes['core_sha256'] == digest(cfg.snes_core)
    store = SaveStore(cfg.saves)
    for identity in (snes, dict(snes, core_sha256=digest(cfg.core))):
        store.write('kart', identity, b'state', frame=1, played_frames=1, session_id='s')
    assert sorted(s['compatible'] for s in sessions.list_saves('kart')) == [False, True]
    assert all('path' not in g and 'file' not in g for g in sessions.games())


@pytest.mark.parametrize('system,file', [('nes', 'kart.sfc'), ('snes', 'game.nes'), ('n64', 'kart.sfc')])
def test_system_and_extension_must_match(tmp_path, system, file):
    cfg, rows = mixed(tmp_path)
    rows[1].update(system=system, file=file)
    cfg.catalog.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='system'): Sessions(cfg)


def test_missing_snes_core_refuses_startup_without_loading_native_code(tmp_path):
    cfg, _ = mixed(tmp_path)
    cfg.snes_core.unlink()
    with pytest.raises(ValueError, match='snes core missing'): Sessions(cfg)


@pytest.mark.parametrize('button', [2, 512, 1024, 2048])
def test_snes_extra_buttons_reach_both_ports_and_nes_still_rejects_them(button):
    snes = Arbiter([0, 1], 'snes')
    a = snes.control('one', 0, 'acquire', 0)
    b = snes.control('two', 1, 'acquire', 0)
    snes.frame(0)
    snes.submit('one', intent(a, mask=button), .01)
    snes.submit('two', intent(b, mask=button | 256), .01)
    assert snes.frame(.02)[0] == [button, button | 256]
    nes = Arbiter([0])
    grant = nes.control('one', 0, 'acquire', 0)
    with pytest.raises(ValueError, match='unsupported_buttons'):
        nes.submit('one', intent(grant, mask=button), .01)


@pytest.mark.parametrize('mask', [4096, 8192, 48, 192])
def test_snes_rejects_triggers_sticks_and_opposing_directions(mask):
    arbiter = Arbiter([0], 'snes')
    grant = arbiter.control('one', 0, 'acquire', 0)
    with pytest.raises(ValueError): arbiter.submit('one', intent(grant, mask=mask), .01)
    assert arbiter.frame(.02)[0] == [0, 0]


def test_import_is_additive_idempotent_and_refuses_replacement(tmp_path):
    spec = importlib.util.spec_from_file_location('import_rom', Path(__file__).parents[1]/'evals/import_rom.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg, original = mixed(tmp_path)
    rom = tmp_path/'new.smc'; rom.write_bytes(b'another fixture')
    row = mod.import_rom(cfg.catalog, rom, 'world', 'Super Mario World', 2)
    after = cfg.catalog.read_bytes()
    assert json.loads(after) == original + [row]
    mod.import_rom(cfg.catalog, rom, 'world', 'Super Mario World', 2)
    assert cfg.catalog.read_bytes() == after
    assert len(Sessions(cfg).games()) == 3
    rom.write_bytes(b'changed')
    with pytest.raises(ValueError, match='replace existing game'):
        mod.import_rom(cfg.catalog, rom, 'world', 'Super Mario World', 2)
    assert cfg.catalog.read_bytes() == after


def test_base_installer_keeps_operator_imports(tmp_path):
    spec = importlib.util.spec_from_file_location('install_catalog', Path(__file__).parents[1]/'evals/install_catalog.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    root = tmp_path/'service'; (root/'catalog').mkdir(parents=True)
    rows = [dict(game_id='world', system='snes', file='world.sfc', title='World',
                 rom_sha256='a'*64, ports=[0, 1], provenance='operator')]
    path = root/'catalog/catalog.json'; path.write_text(json.dumps(rows))
    mod.install(tmp_path/'empty-kit', root, tmp_path/'local.env')
    assert json.loads(path.read_text()) == rows


def test_base_installer_refuses_65th_game_without_replacing_catalog(tmp_path):
    spec = importlib.util.spec_from_file_location('install_catalog', Path(__file__).parents[1]/'evals/install_catalog.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    root = tmp_path/'service'; (root/'catalog').mkdir(parents=True)
    rom = root/'catalog/world.sfc'; rom.write_bytes(b'operator fixture')
    rows = [dict(game_id=f'game-{n}', system='snes', file=rom.name, title=f'Game {n}',
                 rom_sha256=digest(rom), ports=[0], provenance='operator') for n in range(64)]
    path = root/'catalog/catalog.json'; path.write_text(json.dumps(rows))
    before = path.read_bytes()
    kit = tmp_path/'kit'; kit.mkdir()
    pad = kit/'pad-lab.nes'; pad.write_bytes(b'authored fixture')
    (kit/'manifest.json').write_text(json.dumps(dict(rom=pad.name, rom_sha256=digest(pad))))
    with pytest.raises(ValueError, match='64 games'):
        mod.install(kit, root, tmp_path/'local.env')
    assert path.read_bytes() == before
