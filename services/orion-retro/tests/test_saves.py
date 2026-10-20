"""Persistent save states: store, real native worker round trip, API refusal."""
import hashlib
import json
import queue
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import uuid

import pytest
from fastapi.testclient import TestClient

from app import worker
from app.main import create_app
from app.saves import SaveStore
from app.session import Sessions
from orion.retro.probe import CoreProbe, digest
from settings import Settings

IDENTITY = dict(rom_sha256='r'*64, core_sha256='c'*64)


def test_store_round_trip_lists_newest_first_and_refuses_mismatch(tmp_path):
    store = SaveStore(tmp_path)
    first = store.write('game', IDENTITY, b'one', frame=10, played_frames=10, session_id='s')
    second = store.write('game', IDENTITY, b'two', frame=20, played_frames=99, session_id='s')
    assert [m['save_id'] for m in store.list('game')] == [second['save_id'], first['save_id']]
    assert store.read('game', first['save_id'], IDENTITY) == (first, b'one')
    for wrong in (dict(IDENTITY, rom_sha256='x'*64), dict(IDENTITY, core_sha256='x'*64)):
        with pytest.raises(ValueError, match='save_identity_mismatch'):
            store.read('game', first['save_id'], wrong)
    with pytest.raises(ValueError, match='unknown_save'): store.read('game', 'f'*32, IDENTITY)
    with pytest.raises(ValueError, match='invalid_save'): store.read('game', '../x', IDENTITY)
    with pytest.raises(ValueError, match='invalid_game'): store.list('../etc')
    (tmp_path/'game'/f"{first['save_id']}.state").write_bytes(b'ONE')
    with pytest.raises(ValueError, match='save_corrupt'): store.read('game', first['save_id'], IDENTITY)


def test_store_bounds_size_and_prunes_oldest_and_orphans(tmp_path):
    store = SaveStore(tmp_path, max_per_game=3, max_bytes=8)
    with pytest.raises(ValueError, match='save_too_large'):
        store.write('game', IDENTITY, b'x'*9, frame=0, played_frames=0, session_id='s')
    (tmp_path/'game').mkdir(exist_ok=True)
    (tmp_path/'game'/('a'*32+'.state')).write_bytes(b'orphan')  # crash between renames
    saved = [store.write('game', IDENTITY, bytes([i]), frame=i, played_frames=i, session_id='s')['save_id']
             for i in range(5)]
    assert [m['save_id'] for m in store.list('game')] == saved[:1:-1]
    assert sorted(p.name for p in (tmp_path/'game').iterdir()) == sorted(
        f'{s}{x}' for s in saved[2:] for x in ('.json', '.state'))


def test_unwritable_or_missing_root_disables_saves(tmp_path):
    store = SaveStore(tmp_path/'missing')
    with pytest.raises(ValueError, match='saves_unavailable'):
        store.write('game', IDENTITY, b'x', frame=0, played_frames=0, session_id='s')
    assert store.list('game') == []


@pytest.fixture
def native(tmp_path):
    if not shutil.which('cc'):
        pytest.fail('native save-state gate requires a C compiler')
    library = tmp_path/'fixture.so'
    source = Path(__file__).resolve().parents[3]/'tests/retro/fixture_core.c'
    subprocess.run(['cc', '-shared', '-fPIC', '-o', str(library), str(source)], check=True)
    rom = tmp_path/'fixture.nes'; rom.write_bytes(b'fixture')
    return library, rom


def run_worker(monkeypatch, tmp_path, native, scenario, *, save_id=None, loops=60, session='session'):
    """The real worker loop and the real native ABI; only the clock is simulated."""
    commands = queue.Queue(); urgent = queue.Queue(); rows = []; hashes = []
    clock = SimpleNamespace(now=0., loops=0)
    class Stop:
        def is_set(self): return clock.loops >= loops
    def sleep(seconds): clock.now += max(seconds, .001); clock.loops += 1
    monkeypatch.setattr(worker, 'time', SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    class Recording(CoreProbe):
        def step(self, p1=0, p2=0):
            hashes.append(super().step(p1, p2)); return hashes[-1]
    monkeypatch.setattr(worker, 'CoreProbe', Recording)
    def enqueue(op, data=None):
        urgent.put_nowait(dict(op=op, id=op+str(len(rows)), connection='', data=data or {}, received_at=clock.now))
    class Replies:
        def put_nowait(self, row):
            rows.append(dict(row, steps=len(hashes))); scenario(rows[-1], enqueue)
    cfg = Settings(artifacts=tmp_path/'evidence', saves=tmp_path/'saves', core=native[0])
    (tmp_path/'saves').mkdir(exist_ok=True)
    worker.run(commands, urgent, Replies(), Stop(), cfg,
               dict(path=native[1], ports=[0], game_id='fixture', save_id=save_id), session, 'generation')
    events = [json.loads(line) for line in (tmp_path/'evidence'/f'{session}.jsonl').read_text().splitlines()]
    return rows, hashes, events


def test_native_save_load_in_place_and_fresh_session_replay_same_frames(monkeypatch, tmp_path, native):
    sent = set()
    def scenario(row, enqueue):
        if row['kind'] == 'status' and row['frame'] >= 6 and 'save' not in sent:
            sent.add('save'); enqueue('save')
        elif row.get('result', {}).get('saved') and 'load' not in sent:
            sent.add('load'); scenario.saved = row
        elif row['kind'] == 'status' and 'load' in sent and 'loaded' not in sent and row['steps'] >= scenario.saved['steps']+8:
            sent.add('loaded'); enqueue('load', {'save_id': scenario.saved['result']['saved']['save_id']})
    rows, hashes, events = run_worker(monkeypatch, tmp_path, native, scenario)
    assert not any(r['kind'] == 'fatal' for r in rows), rows
    saved = scenario.saved; meta = saved['result']['saved']
    loaded = next(r for r in rows if r.get('result', {}).get('loaded'))
    # Same state + same (neutral) inputs => the same frames, frame for frame.
    at_save, at_load = saved['steps'], loaded['steps']
    assert at_load > at_save+4
    assert hashes[at_load:at_load+4] == hashes[at_save:at_save+4]
    assert hashes[at_load] != hashes[at_load-1]
    store = SaveStore(tmp_path/'saves')
    identity = dict(rom_sha256=digest(native[1]), core_sha256=digest(native[0]))
    assert store.read('fixture', meta['save_id'], identity)[0] == meta
    assert meta['frame'] == meta['played_frames'] == at_save
    kinds = {e['kind']: e for e in events}
    assert kinds['save']['save_id'] == meta['save_id'] and kinds['save']['frame'] == at_save
    assert kinds['load']['save_id'] == meta['save_id'] and kinds['load']['frame'] == at_load
    assert kinds['load']['save_frame'] == at_save

    # A fresh worker (e.g. after a service restart) starts from the save.
    rows2, hashes2, events2 = run_worker(monkeypatch, tmp_path, native, lambda *a: None,
                                         save_id=meta['save_id'], loops=8, session='session2')
    assert not any(r['kind'] == 'fatal' for r in rows2), rows2
    assert hashes2[:4] == hashes[at_save:at_save+4]
    assert [e['kind'] for e in events2][:2] == ['start', 'load']


def test_worker_refuses_mismatched_save_without_dying(monkeypatch, tmp_path, native):
    foreign = SaveStore(tmp_path/'saves')
    (tmp_path/'saves').mkdir()
    bad = foreign.write('fixture', dict(rom_sha256=digest(native[1]), core_sha256='0'*64), b'\0'*4,
                        frame=1, played_frames=1, session_id='other')
    def scenario(row, enqueue):
        if row['kind'] == 'ready': enqueue('load', {'save_id': bad['save_id']})
    rows, hashes, events = run_worker(monkeypatch, tmp_path, native, scenario, loops=10)
    reply = next(r for r in rows if r['kind'] == 'reply')
    assert reply['error'] == 'save_identity_mismatch'
    assert not any(r['kind'] == 'fatal' for r in rows) and len(hashes) >= 8
    assert 'load' not in {e['kind'] for e in events}


def api_fixture(tmp_path):
    token = tmp_path/'token'; token.write_text('a'*48)
    rom = tmp_path/'game.nes'; rom.write_bytes(b'NES\x1a'+b'\0'*16)
    core = tmp_path/'core.so'; core.write_bytes(b'not-a-real-core')
    (tmp_path/'saves').mkdir()
    catalog = tmp_path/'catalog.json'
    catalog.write_text(json.dumps([dict(game_id='test', title='Test', file=rom.name,
        rom_sha256=hashlib.sha256(rom.read_bytes()).hexdigest(), ports=[0], provenance='fixture')]))
    cfg = Settings(enabled=True, token_file=token, catalog=catalog, artifacts=tmp_path/'evidence',
                   core=core, saves=tmp_path/'saves')
    identity = dict(rom_sha256=digest(rom), core_sha256=digest(core))
    return cfg, identity


def test_api_lists_saves_and_refuses_mismatch_before_the_worker(tmp_path):
    import asyncio
    cfg, identity = api_fixture(tmp_path)
    store = SaveStore(cfg.saves)
    good = store.write('test', identity, b'x', frame=5, played_frames=7, session_id='s')
    bad = store.write('test', dict(identity, core_sha256='0'*64), b'y', frame=6, played_frames=8, session_id='s')
    m = Sessions(cfg)
    assert m.games()[0]['save_enabled'] is True
    listed = m.list_saves('test')
    assert [(s['save_id'], s['compatible']) for s in listed] == [(bad['save_id'], False), (good['save_id'], True)]
    with pytest.raises(ValueError, match='save_identity_mismatch'):
        asyncio.run(m.start('test', 'k', save_id=bad['save_id']))
    assert m.process is None
    app = create_app(cfg)
    h = {'Authorization': 'Bearer '+'a'*48}
    with TestClient(app) as c:
        manager = app.state.manager
        calls = []
        async def command(op, connection='', data=None):
            calls.append((op, data)); return {'loaded': good} if op == 'load' else {'saved': good}
        sid = str(uuid.uuid4())
        manager.session = dict(session_id=sid, owner='operator', game_id='test', state='running')
        manager.process = object(); manager.command = command
        assert [s['save_id'] for s in c.get('/games/test/saves', headers=h).json()] == [bad['save_id'], good['save_id']]
        r = c.post(f'/sessions/{sid}/load', json={'save_id': bad['save_id']}, headers=h)
        assert r.status_code == 409 and r.json()['detail'] == 'save_identity_mismatch' and not calls
        assert c.post(f'/sessions/{sid}/load', json={'save_id': '../x'}, headers=h).status_code == 422
        assert c.post(f'/sessions/{sid}/load', json={'save_id': good['save_id']}, headers=h).status_code == 200
        assert c.post(f'/sessions/{sid}/save', headers=h).status_code == 200
        assert calls == [('load', {'save_id': good['save_id']}), ('save', None)]
        assert c.post(f'/sessions/{sid}/save', headers={'Authorization': 'Bearer '+'b'*48}).status_code == 401
        manager.process = None


def test_compose_mounts_saves_from_a_required_host_dir():
    import yaml
    root = Path(__file__).resolve().parents[1]
    volumes = yaml.safe_load((root/'docker-compose.yml').read_text())['services']['retro']['volumes']
    assert '${RETRO_SAVES_DIR:?save-state directory required}:/saves' in volumes
    assert 'RETRO_SAVES_DIR=' in (root/'.env_example').read_text()
    assert Settings().saves == Path('/saves')
