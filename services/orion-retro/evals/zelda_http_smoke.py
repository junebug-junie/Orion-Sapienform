"""Live smoke: real HTTP app + spawned native worker + real Zelda core, at real time.

Boots with real controller input over the input WebSocket, hands off to the GoalV2
motor, runs enter -> collect sword -> exit S over POST /goal (with an idempotent retry),
reads GET /atlas, saves (the save must carry the atlas lineage), then checks that a
human acquire preempts a leisure grant at once. Temp dirs only; no live session touched.

Run: python services/orion-retro/evals/zelda_http_smoke.py --core <so> --catalog <catalog.json>
"""
import argparse, json, secrets, sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'services/orion-retro'))
from fastapi.testclient import TestClient
from settings import Settings
from app.main import create_app
def main(args):
    tmp = Path(tempfile.mkdtemp(prefix='zelda-http-'))
    token = secrets.token_hex(24); (tmp/'tok').write_text(token)
    (tmp/'saves').mkdir()
    st = Settings(enabled=True, cognition_enabled=True,
                  catalog=Path(args.catalog),
                  artifacts=tmp/'evidence', core=Path(args.core),
                  token_file=tmp/'tok', hub_token_file=tmp/'none', saves=tmp/'saves')
    H = {'authorization': 'Bearer '+token}
    out = {}
    with TestClient(create_app(st)) as c:
        sid = c.post('/sessions', json={'game_id': args.game_id, 'idempotency_key': 'smoke'}, headers=H).json()['session_id']
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
            out['scripted_start'] = r.status_code
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
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(out, indent=1, default=str))
    ok = (out['profile'] == 'zelda-1-ram-v1' and all(out[k]['verdict'] == 'met' and out[k]['retry_status'] == 200
          for k in ('enter', 'sword', 'exit_s')) and out['bad_goal'] == 422 and out['atlas'][0] == 200
          and out['save_lineage'] == out['atlas_lineage'] and out['leisure_acquire'][0] == 200
          and out['human_acquire_over_leisure'][1].get('preempted_leisure') is True and not out['scripted_after_human'])
    print(json.dumps({k: out[k] for k in ('profile', 'bad_goal', 'save_lineage', 'leisure_acquire',
                      'human_acquire_over_leisure')}, default=str))
    print('OK' if ok else 'FAILED', args.output)
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--core', required=True)
    p.add_argument('--catalog', required=True, help='catalog.json listing the Zelda ROM')
    p.add_argument('--game-id', default='the-legend-of-zelda')
    p.add_argument('--output', default='/tmp/zelda-play/http_smoke.json')
    main(p.parse_args())
