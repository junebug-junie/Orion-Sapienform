"""Browser eval for Orion's play panel (spec 2026-10-22: Hub view, acceptance check 5 UI half).

Real template (templates/retro.html), real static asset (static/js/retro-panel.js), real Hub
router (scripts/retro_routes.py) and a real headless browser. Retro itself is a FIXTURE on a
private Unix socket: it serves a Zelda-shaped session (one leisure policy with a decision,
an accepted goal and verdict history) and an atlas. It is not the native worker, so this eval
proves the page and its wiring, not Lane A's arbiter.

Checks (evals/retro_play_panel.mjs): the leisure session is visible live on open (media
socket only, no controller taken); the atlas grid shows known rooms, the current room, used
exits, caves and deaths; goal, rationale, prediction and verdicts render; a heat shed shows
"Orion is waiting: cabinet hot"; Take this controller acquires a controller (the preempt
request) and the relay records a human action.

    python services/orion-hub/evals/retro_play_panel.py --output /tmp/retro-play-panel
"""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
from types import SimpleNamespace

HUB = Path(__file__).resolve().parents[1]
SID = '5e55105e-0000-4000-8000-00000000c0de'
GEN = '9e9e9e9e-0000-4000-8000-000000000001'


def zelda_session():
    attempt = dict(decision_id='3:7', port=0, expected_epoch=3, supported_goals=['exit', 'fight', 'wait'],
                   supported_predictions=['reach_room', 'survive'],
                   observation=dict(status='observed', profile='zelda-1-ram-v1', mode='play', level=0, room=119,
                                    link=dict(x=120, y=141, tile=[15, 11], controllable=True), enemies=[], walls_hash='h'))
    goal = dict(decision_id='3:7', goal=dict(kind='exit', dir='N'), prediction=dict(kind='reach_room', value=103),
                rationale='North of the start is unexplored and no enemy blocks it.', budget_seconds=12,
                planner_id='llm-gateway:metacog_background:fixture')
    return dict(session_id=SID, generation=GEN, state='running', game_id='the-legend-of-zelda', frame=4000,
                owner='hub', save_enabled=False, loaded_save=None, game_profile='zelda-1-ram-v1',
                scripted=[dict(port=0, policy_id='zelda-navigator-v1', epoch=3, duration_seconds=1800, started_frame=10,
                               cognition=dict(policy_id='zelda-navigator-v1', attempt=attempt, goal=goal,
                                              last_finished=None))])


def finished_row(i):
    return dict(decision_id=f'3:{i}', goal=dict(kind='exit', dir='E'), prediction=dict(kind='reach_room', value=120),
                planner_id='fixture', outcome='prediction_met' if i % 2 else 'prediction_refuted',
                verdict='met' if i % 2 else 'refuted')


# Shape of Retro's GET /sessions/{id}/atlas (services/orion-retro/app/zelda/atlas.py neighbourhood()).
ATLAS = dict(lineage_id='lineage-fixture', level=0, room=119, rooms_known=3,
             current=dict(visits=3, deaths=0, exits_seen=dict(N=103, E=120, S=None, W=None)),
             caves=[dict(room='0:119', tile=[8, 2], entered=True)], deaths=[dict(room='0:120', deaths=2)],
             map={'0:119': dict(visits=3, exits=dict(N=103, E=120, S=None, W=None), caves=1, deaths=0),
                  '0:120': dict(visits=1, exits=dict(W=119, N=None, S=None, E=None), caves=0, deaths=2),
                  '0:103': dict(visits=1, exits=dict(S=119, N=None, E=None, W=None), caves=0, deaths=0)})


async def fake_retro(path, log):
    from aiohttp import web, WSMsgType
    app = web.Application()
    session = zelda_session()
    async def ready(_): return web.json_response({'status': 'ready', 'session': session})
    async def games(_): return web.json_response([dict(game_id='the-legend-of-zelda', title='The Legend of Zelda', ports=[0, 1], save_enabled=False)])
    finished = iter(range(1, 7))
    async def status(_):
        # Retro shows only the newest goal_finished row; advance one per poll so the page must keep history.
        i = next(finished, None) if session['scripted'] else None
        if i is not None: session['scripted'][0]['cognition']['last_finished'] = finished_row(i)
        return web.json_response(session)
    async def atlas(_): return web.json_response(ATLAS)
    async def op(request):
        body = await request.text()
        log.append(dict(op=request.match_info['op'], body=json.loads(body) if body else None))
        if request.match_info['op'] == 'control':
            return web.json_response(dict(lease='l' * 40, epoch=9, port=0))
        data = json.loads(body) if body else {}
        if request.match_info['op'] == 'scripted' and data.get('action') == 'start':
            # Like Retro: a started hand-over is a new grant (new epoch) with the asked duration.
            base = zelda_session()['scripted'][0]
            n = sum(1 for r in log if r['op'] == 'scripted')
            session['scripted'] = [dict(base, epoch=100 + n, duration_seconds=data.get('duration_seconds', 60),
                                        started_frame=session.get('frame', 0))]
        return web.json_response({'state': 'running'})
    async def ws(request):
        sock = web.WebSocketResponse(); await sock.prepare(request)
        log.append(dict(op='ws', channel=request.match_info['channel']))
        if request.match_info['channel'] == 'input':
            await sock.send_json({'connection_id': 'c0c0c0c0-0000-4000-8000-000000000003', 'generation': GEN})
        async for msg in sock:
            if msg.type != WSMsgType.TEXT: break
        log.append(dict(op='ws_closed', channel=request.match_info['channel']))
        return sock
    app.add_routes([web.get('/health/ready', ready), web.get('/games', games), web.get(f'/sessions/{SID}', status),
                    web.get(f'/sessions/{SID}/atlas', atlas), web.post(f'/sessions/{SID}/{{op}}', op),
                    web.get(f'/sessions/{SID}/{{channel}}', ws)])
    runner = web.AppRunner(app); await runner.setup()
    await web.UnixSite(runner, path).start()
    return session


class MemoryStore:
    def __init__(self): self.human = 0; self.browsers = set(); self.value = {}
    async def mark_human(self): self.human += 1
    async def beat_browser(self, sid): self.browsers.add(sid)
    async def drop_browser(self, sid): self.browsers.discard(sid)
    async def status(self): return dict(self.value, updated_at=__import__('time').time())
    async def journal(self, n): return []


async def serve(output: Path, port: int, sock: str):
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    sys.path.insert(0, str(HUB))
    spec = importlib.util.spec_from_file_location('retro_play_routes', HUB / 'scripts/retro_routes.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    log = []
    session = await fake_retro(sock, log)
    store = MemoryStore()
    store.value = dict(state='playing', session_id=SID, game_id='the-legend-of-zelda', started_at=__import__('time').time() - 300,
                       decisions=6, waiting=None, last_goal=zelda_session()['scripted'][0]['cognition']['goal'])
    config = SimpleNamespace(HUB_RETRO_ENABLED=True, HUB_RETRO_COGNITION_ENABLED=True,
                             HUB_RETRO_ORIGIN=f'http://127.0.0.1:{port}', HUB_RETRO_AUTH_DIR=str(output),
                             HUB_RETRO_SOCKET=sock)
    app = FastAPI(); app.include_router(mod.create_router(config, presence=store))
    app.mount('/static', StaticFiles(directory=HUB / 'static'))
    @app.post('/__fixture/waiting')
    async def waiting():
        store.value = dict(store.value, state='waiting', waiting=dict(reason='shed:cabinet_hot', retry_after_sec=60))
        return {}
    @app.post('/__fixture/end_grant')
    async def end_grant(expired: int = 0):
        # The handed-off grant is gone while the game keeps running: taken by another controller,
        # or (expired=1) because its time ran out (the session frame passes its budget).
        if expired and session['scripted']:
            g = session['scripted'][0]
            session['frame'] = g.get('started_frame', 0) + int((g.get('duration_seconds', 60) + 1) * 60.0988)
        session['scripted'] = []
        return {}
    @app.get('/__fixture/log')
    async def fixture_log():
        return dict(retro=log, human=store.human, browsers=len(store.browsers))
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
    await server.serve()


def run(args):
    import time
    import httpx
    output = args.output.resolve(); output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / 'hub.token').write_text(secrets.token_urlsafe(48))
    # AF_UNIX paths are limited to ~107 bytes; keep the fixture socket in a short private dir.
    import tempfile
    sock_dir = Path(tempfile.mkdtemp(prefix='orion-rpp-', dir='/tmp' if len(str(output)) > 80 else None))
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, __file__, '--serve', '--output', str(output), '--port', str(port),
                             '--socket', str(sock_dir / 'retro.sock')],
                            stdout=(output / 'server.log').open('w'), stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            try:
                if httpx.get(f'http://127.0.0.1:{port}/retro').status_code == 200: break
            except httpx.TransportError: pass
            time.sleep(.1)
        env = dict(os.environ, RETRO_EVAL_URL=f'http://127.0.0.1:{port}', RETRO_EVAL_OUTPUT=str(output))
        subprocess.run(['node', str(Path(__file__).with_suffix('.mjs'))], env=env, check=True)
        print('PASS', output / 'play-panel.json')
    finally:
        proc.terminate(); proc.wait(timeout=5)
        for f in sock_dir.iterdir(): f.unlink()
        sock_dir.rmdir()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--serve', action='store_true'); p.add_argument('--port', type=int)
    p.add_argument('--socket')
    a = p.parse_args()
    if a.serve: asyncio.run(serve(a.output, a.port, a.socket))
    else: run(a)
