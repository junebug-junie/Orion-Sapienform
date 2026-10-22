"""Live smoke: real HTTP app + spawned native worker + real Zelda core. Real input only."""
import json, secrets, sys, tempfile, time
from pathlib import Path
ROOT = Path('/mnt/scripts/Orion-Sapienform-zelda-retro-core')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'services/orion-retro'))
from fastapi.testclient import TestClient
from settings import Settings
from app.main import create_app
def main():
    tmp = Path(tempfile.mkdtemp(prefix='zelda-http-'))
    token = secrets.token_hex(24); (tmp/'tok').write_text(token)
    (tmp/'saves').mkdir()
    st = Settings(enabled=True, cognition_enabled=True,
                  catalog=Path('/home/athena/.local/share/orion/retro/service/catalog/catalog.json'),
                  artifacts=tmp/'evidence', core=Path('/home/athena/.local/share/orion/retro/games/nova-the-squirrel-v1.0.6a/cores/fceumm-ac0d00a7.so'),
                  token_file=tmp/'tok', hub_token_file=tmp/'none', saves=tmp/'saves')
    H = {'authorization': 'Bearer '+token}
    out = {}
    with TestClient(create_app(st)) as c:
        sid = c.post('/sessions', json={'game_id': 'the-legend-of-zelda', 'idempotency_key': 'smoke'}, headers=H).json()['session_id']
        with c.websocket_connect(f'/sessions/{sid}/input', headers=H) as ws:
            hello = ws.receive_json(); conn, gen = hello['connection_id'], hello['generation']
            g = c.post(f'/sessions/{sid}/control', json={'connection_id': conn, 'port': 0}, headers=H).json()
            seq = [0]
            def send(mask):
                seq[0] += 1
                ws.send_text(json.dumps(dict(version=1, session_id=sid, generation=gen, lease=g['lease'], epoch=g['epoch'], seq=seq[0], port=0, mask=mask)))
                ws.receive_json()
            def hold(mask, seconds):
                end = time.monotonic()+seconds
                while time.monotonic() < end: send(mask); time.sleep(.05)
            hold(0, 2.2); hold(8, .1); hold(0, 1.2); hold(8, .1); hold(0, 1.2); hold(256, .1); hold(0, .3)
            for _ in range(3): hold(4, .1); hold(0, .25)
            hold(8, .1); hold(0, 1.2); hold(8, .1); hold(0, 2.5)
            s = c.get(f'/sessions/{sid}', headers=H).json()
            out['profile'] = s.get('game_profile'); out['obs_at_boot'] = s['zelda']['observation']
            assert s['zelda']['observation']['mode'] == 'play', s['zelda']['observation']
            r = c.post(f'/sessions/{sid}/scripted', json=dict(cognition=True, connection_id=conn, generation=gen, port=0, action='start', duration_seconds=120, expected_epoch=g['epoch']), headers=H)
            out['scripted_start'] = r.status_code; print('SCRIPTED', r.status_code, r.text[:300], c.get(f'/sessions/{sid}', headers=H).json().get('state'))
            def decision():
                for _ in range(200):
                    sc = c.get(f'/sessions/{sid}', headers=H).json()['scripted']
                    cog = sc[0]['cognition'] if sc else {}
                    if cog.get('attempt') and not cog.get('goal'): return cog['attempt']
                    time.sleep(.05)
            def run_goal(goal, prediction, budget=20):
                a = decision()
                body = dict(connection_id=conn, generation=gen, port=0, expected_epoch=a['expected_epoch'], decision_id=a['decision_id'],
                            goal=goal, prediction=prediction, rationale='http smoke', planner_id='http-smoke', budget_seconds=budget)
                r = c.post(f'/sessions/{sid}/goal', json=body, headers=H); assert r.status_code == 200, r.text
                dup = c.post(f'/sessions/{sid}/goal', json=body, headers=H).status_code  # idempotent retry
                for _ in range(budget*25):
                    sc = c.get(f'/sessions/{sid}', headers=H).json()['scripted']
                    lf = sc[0]['cognition']['last_finished'] if sc else None
                    if lf and lf['decision_id'] == a['decision_id']: return dict(lf, retry_status=dup)
                    time.sleep(.04)
            out['enter'] = run_goal({'kind': 'enter', 'tile': {'x': 8, 'y': 2}}, {'kind': 'enter_cave'})
            out['sword'] = run_goal({'kind': 'collect', 'tile': {'x': 15, 'y': 12}}, {'kind': 'item_gained'})
            out['exit_s'] = run_goal({'kind': 'exit', 'dir': 'S'}, {'kind': 'reach_room', 'value': 119})
            out['bad_goal'] = c.post(f'/sessions/{sid}/goal', json={'goal': 'walk_right'}, headers=H).status_code
            time.sleep(2.5)
            atlas = c.get(f'/sessions/{sid}/atlas', headers=H)
            out['atlas'] = (atlas.status_code, atlas.json())
            save = c.post(f'/sessions/{sid}/save', headers=H).json()['saved']
            out['save_lineage'] = save.get('lineage_id'); out['atlas_lineage'] = atlas.json().get('lineage_id')
            # Leisure preemption: stop the script, release, leisure acquires + starts, human acquires.
            c.post(f'/sessions/{sid}/scripted', json=dict(cognition=True, connection_id=conn, generation=gen, port=0, action='stop', duration_seconds=120,
                   expected_epoch=c.get(f'/sessions/{sid}', headers=H).json()['scripted'][0]['epoch']), headers=H)
            out['state_after_stop'] = c.get(f'/sessions/{sid}', headers=H).json()['state']
            c.post(f'/sessions/{sid}/resume', headers=H)
            lz = c.post(f'/sessions/{sid}/control', json={'leisure_id': 'smoke-1', 'port': 0}, headers=H)
            print('LEISURE', lz.status_code, lz.text[:200])
            out['leisure_acquire'] = (lz.status_code, {k: v for k, v in lz.json().items() if k != 'lease'})
            ls = c.post(f'/sessions/{sid}/scripted', json=dict(leisure_id='smoke-1', generation=gen, port=0, action='start', duration_seconds=60, expected_epoch=lz.json()['epoch']), headers=H)
            out['leisure_script'] = ls.status_code
            time.sleep(.5)
            hu = c.post(f'/sessions/{sid}/control', json={'connection_id': conn, 'port': 0}, headers=H)
            out['human_acquire_over_leisure'] = (hu.status_code, {k: v for k, v in hu.json().items() if k != 'lease'})
            time.sleep(.3)
            out['scripted_after_human'] = c.get(f'/sessions/{sid}', headers=H).json()['scripted']
            out['leisure_takeover_refused'] = c.post(f'/sessions/{sid}/control', json={'leisure_id': 'smoke-1', 'port': 0}, headers=H).json()
        c.post(f'/sessions/{sid}/stop', headers=H)
    ev = list((tmp/'evidence').glob('*.jsonl'))[0].read_text().splitlines()
    kinds = [json.loads(l)['kind'] for l in ev]
    out['journal_kinds'] = {k: kinds.count(k) for k in set(kinds)}
    out['stops'] = [json.loads(l) for l in ev if json.loads(l)['kind'] in ('policy_stopped', 'goal_finished') and 'stopped' in json.dumps(json.loads(l))][-3:]
    Path('/tmp/zelda-play/http_smoke.json').write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: (v if k not in ('atlas', 'obs_at_boot') else '...') for k, v in out.items()}, indent=1, default=str)[:5000])

if __name__ == '__main__':
    main()
