import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.control import Arbiter
from app.evidence import Evidence
from app.main import create_app
from app.session import Sessions
from settings import Settings


def intent(grant, seq=1, mask=256):
    return dict(port=grant['port'], lease=grant['lease'], epoch=grant['epoch'], seq=seq, mask=mask)


def test_takeover_neutralizes_before_new_owner_and_rejects_old_epoch():
    a = Arbiter([0,1]); old = a.control('one',0,'acquire',0)
    a.submit('one',intent(old),0)
    assert a.frame(0)[0] == [0,0]
    assert a.frame(.01)[0] == [256,0]
    new = a.control('two',0,'takeover',.02)
    with pytest.raises(ValueError): a.submit('one',intent(old,2),.02)
    a.submit('two',intent(new,mask=128),.02)
    assert a.frame(.02)[0] == [0,0]
    assert a.frame(.03)[0] == [128,0]


def test_expiry_cannot_be_revived_and_ports_are_independent():
    from app.control import LEASE_TTL as T
    a = Arbiter([0,1]); p1=a.control('one',0,'acquire',0); p2=a.control('two',1,'acquire',0)
    a.frame(0); a.submit('one',intent(p1),0); a.submit('two',intent(p2,mask=128),.1)
    assert a.frame(.2)[0] == [256,128]
    assert a.frame(T+.01)[0] == [0,128]
    with pytest.raises(ValueError): a.submit('one',intent(p1,2),T+.01)
    a.disconnect('two'); assert a.frame(T+.02)[0] == [0,0]


@pytest.mark.parametrize('change', [{'mask':2}, {'mask':48}, {'mask':192}, {'epoch':99}, {'lease':'wrong'}, {'seq':0}])
def test_invalid_or_stale_inputs_cannot_change_mask(change):
    a = Arbiter([0]); g=a.control('one',0,'acquire',0); a.submit('one',intent(g),0)
    data=intent(g,2);data.update(change)
    with pytest.raises(ValueError): a.submit('one',data,.01)
    a.frame(.01); assert a.frame(.02)[0] == [256,0]


def test_wrong_connection_port_and_release_rejected():
    a=Arbiter([0]);g=a.control('one',0,'acquire',0)
    with pytest.raises(ValueError):a.submit('two',intent(g),0)
    with pytest.raises(ValueError):a.control('two',0,'release',0)
    with pytest.raises(ValueError):a.control('one',1,'acquire',0)
    with pytest.raises(ValueError):a.control('two',0,'acquire',0)
    a.clear()
    with pytest.raises(ValueError):a.submit('one',intent(g),0)


def test_evidence_quota_fails_without_writing_partial_row(tmp_path):
    e=Evidence(tmp_path,'one',200,400);e.write({'kind':'start'})
    previous=e.path.read_bytes()
    with pytest.raises(OSError): e.write({'data':'x'*201})
    assert e.path.read_bytes()==previous
    e.close()
    with pytest.raises(FileExistsError):Evidence(tmp_path,'one',200,400)


def test_total_quota_includes_existing_files(tmp_path):
    (tmp_path/'prior').write_bytes(b'x'*200)
    e=Evidence(tmp_path,'new',200,210)
    with pytest.raises(OSError):e.write({'kind':'start'})
    e.close()


def settings(tmp_path, enabled=False):
    token=tmp_path/'token'; token.write_text('a'*48)
    return Settings(enabled=enabled,token_file=token,catalog=tmp_path/'catalog.json',
                    artifacts=tmp_path/'evidence',core=tmp_path/'core.so')


def test_api_auth_origin_and_disabled_health(tmp_path):
    cfg=settings(tmp_path)
    with TestClient(create_app(cfg)) as c:
        assert c.get('/health/live').status_code==401
        h={'Authorization':'Bearer '+'a'*48}
        assert c.get('/health/live',headers=h).json()=={'status':'alive','enabled':False}
        assert c.get('/health/ready',headers=h).status_code==503
        assert c.get('/health/live',headers=dict(h,Origin='https://attacker.example')).status_code==403
        assert c.post('/sessions',content='x'*4097,headers=h).status_code==413


def catalog(cfg, tmp_path):
    import hashlib
    rom=tmp_path/'game.nes';rom.write_bytes(b'NES\x1a'+b'\0'*16)
    cfg.core.write_bytes(b'not-a-real-core')
    data=[dict(game_id='test',title='Test',file=rom.name,rom_sha256=hashlib.sha256(rom.read_bytes()).hexdigest(),ports=[0],provenance='fixture')]
    cfg.catalog.write_text(json.dumps(data));return data


def test_catalog_rejects_hash_and_path_escape(tmp_path):
    cfg=settings(tmp_path);data=catalog(cfg,tmp_path)
    m=Sessions(cfg);assert m.games()[0]['save_enabled'] is False
    data[0]['rom_sha256']='0'*64;cfg.catalog.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='digest'):Sessions(cfg)
    data[0]['file']='../outside.nes';(tmp_path.parent/'outside.nes').write_bytes(b'x')
    cfg.catalog.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='inside'):Sessions(cfg)


def test_restart_marks_active_session_failed_without_spawning(tmp_path):
    cfg=settings(tmp_path);catalog(cfg,tmp_path);cfg.artifacts.mkdir()
    (cfg.artifacts/'status.json').write_text(json.dumps({'state':'running','session_id':'prior'}))
    m=Sessions(cfg)
    assert m.process is None
    assert m.session['state']=='failed' and m.session['reason']=='service_restart'


def test_catalog_missing_dependencies_fails_startup(tmp_path):
    cfg=settings(tmp_path,True)
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(cfg)): pass


def test_non_ascii_lease_is_rejected_not_a_worker_crash():
    a=Arbiter([0]);g=a.control('one',0,'acquire',0)
    data=intent(g);data['lease']='é'*32
    with pytest.raises(ValueError):a.submit('one',data,0)


def test_start_rejected_during_stop_cleanup(tmp_path):
    import asyncio
    import threading
    cfg=settings(tmp_path);catalog(cfg,tmp_path);m=Sessions(cfg)
    entered=threading.Event();release=threading.Event()
    class Process:
        def join(self, timeout):entered.set();release.wait(1)
        def is_alive(self):return False
    class Queue:
        def cancel_join_thread(self):pass
        def close(self):pass
    m.process=Process();m.stop_event=threading.Event()
    m.commands=m.urgent=m.replies=Queue();m.session={'session_id':'old','state':'running'}
    async def scenario():
        task=asyncio.create_task(m.stop())
        await asyncio.to_thread(entered.wait,1)
        try:
            with pytest.raises(ValueError,match='session_active'):await m.start('test','new')
        finally:
            release.set();await task
        assert m.process is None and m.session['session_id']=='old'
    asyncio.run(scenario())


def test_old_connection_cleanup_cannot_pause_new_session(tmp_path):
    import asyncio
    cfg=settings(tmp_path);catalog(cfg,tmp_path);m=Sessions(cfg)
    m.process=object();m.session={'session_id':'new','state':'running'}
    calls=[]
    async def command(*args):calls.append(args)
    m.command=command
    async def scenario():
        # Old WS is still closing after stop cleared its registration.
        await m.disconnect('old-connection','old',m.generation)
        # Even stale membership must not cross the session boundary.
        m.connections.add('old-connection')
        await m.disconnect('old-connection','old',m.generation)
        assert not calls
        m.connections.add('new-connection')
        await m.disconnect('new-connection','new',m.generation)
        assert calls==[('disconnect','new-connection')]
    asyncio.run(scenario())


def test_service_is_selected_for_rebuild():
    import subprocess
    root=Path(__file__).resolve().parents[3]
    completed=subprocess.run(['python3','scripts/rebuild_affected_services.py','--paths',
                             'services/orion-retro/app/main.py','--plan'],cwd=root,
                            capture_output=True,text=True,check=True)
    assert 'orion-retro\tbuild' in completed.stdout


def test_packaging_preserves_offline_boundary_and_disabled_default():
    import yaml
    root=Path(__file__).resolve().parents[1]
    compose=yaml.safe_load((root/'docker-compose.yml').read_text())['services']['retro']
    assert compose['network_mode']=='none' and 'ports' not in compose
    assert compose['read_only'] is True and compose['cap_drop']==['ALL']
    assert compose['logging']['driver']=='journald'
    assert all('docker.sock' not in mount for mount in compose['volumes'])
    assert compose['environment']['RETRO_ENABLED']=='${RETRO_ENABLED:-false}'
    assert 'RETRO_ENABLED=false' in (root/'.env_example').read_text()
    assert Settings().enabled is False
    assert 'COPY services/orion-retro /app' not in (root/'Dockerfile').read_text()


def test_lost_worker_ack_is_unknown_not_a_definite_rejection(tmp_path):
    import uuid
    cfg=settings(tmp_path)
    sid=str(uuid.uuid4());generation=str(uuid.uuid4())
    class Manager:
        session={'session_id':sid,'owner':'operator'};process=True;connections=set()
        async def command(self,*args):
            # The native frame may have run; its acknowledgment was lost.
            raise ValueError('worker_unavailable')
        async def disconnect(self,*args):pass
        async def stop(self,**kwargs):pass
    m=Manager();m.generation=generation
    app=create_app(cfg)
    with TestClient(app) as c:
        app.state.manager=m
        with c.websocket_connect('/sessions/'+sid+'/input',headers={'Authorization':'Bearer '+'a'*48}) as ws:
            ws.receive_json()
            ws.send_json(dict(version=1,session_id=sid,generation=generation,lease='a'*48,epoch=1,seq=1,port=0,mask=256))
            assert ws.receive_json()=={'unknown':'worker_unavailable','seq':1,'epoch':1,'port':0}


def test_hub_credential_cannot_access_operator_session(tmp_path):
    from dataclasses import replace
    cfg=settings(tmp_path);hub=tmp_path/'hub.token';hub.write_text('b'*48)
    app=create_app(replace(cfg,hub_token_file=hub))
    class Manager:
        session={'session_id':'a'*36,'owner':'operator'}
        def require(self,sid):pass
        async def stop(self,**kwargs):pass
    with TestClient(app) as c:
        app.state.manager=Manager()
        assert c.get('/sessions/'+'a'*36,headers={'Authorization':'Bearer '+'b'*48}).status_code==403
        assert c.get('/sessions/'+'a'*36,headers={'Authorization':'Bearer '+'a'*48}).status_code==200
        assert c.get('/health/ready',headers={'Authorization':'Bearer '+'b'*48}).json()['session'] is None


def test_script_api_requires_owner_live_connection_generation_and_valid_budget(tmp_path):
    from dataclasses import replace
    cfg=settings(tmp_path);hub=tmp_path/'hub.token';hub.write_text('b'*48)
    app=create_app(replace(cfg,hub_token_file=hub))
    calls=[]
    class Manager:
        session={'session_id':'a'*36,'owner':'operator'}
        generation='g'*36
        connections={'c'*36}
        def require(self,sid):
            if sid!=self.session['session_id']:raise ValueError('unknown_session')
        async def command(self,*args):
            calls.append(args);return {'scripted':[]}
        async def stop(self,**kwargs):pass
    body=dict(connection_id='c'*36,generation='g'*36,port=0,action='start',expected_epoch=1)
    path='/sessions/'+'a'*36+'/scripted'
    with TestClient(app) as c:
        app.state.manager=Manager()
        h={'Authorization':'Bearer '+'a'*48}
        assert c.post(path,json=body).status_code==401
        assert c.post(path,json=body,headers={'Authorization':'Bearer '+'b'*48}).status_code==403
        for change in ({'connection_id':'d'*36},{'generation':'h'*36}):
            assert c.post(path,json=body|change,headers=h).status_code==409
        assert c.post(path,json=body|{'duration_seconds':1801},headers=h).status_code==422
        assert not calls
        assert c.post(path,json=body,headers=h).status_code==200
        assert calls==[('scripted','c'*36,dict(port=0,action='start',expected_epoch=1,duration_seconds=60))]


def test_paused_viewer_disconnect_releases_capacity(tmp_path):
    import time
    cfg=settings(tmp_path)
    class Manager:
        session={'session_id':'a'*36,'owner':'operator'};process=True
        latest_media=None;media_version=0;viewers=0
        def require(self,sid):pass
        async def stop(self,**kwargs):pass
    m=Manager();app=create_app(cfg)
    with TestClient(app) as c:
        app.state.manager=m
        for _ in range(3):
            with c.websocket_connect('/sessions/'+'a'*36+'/media',headers={'Authorization':'Bearer '+'a'*48}):
                assert m.viewers==1
            for _ in range(20):
                if m.viewers==0:break
                time.sleep(.01)
            assert m.viewers==0


def test_slow_ack_does_not_gate_reading_the_next_input(tmp_path):
    # Live 2026-10-20: one slow acknowledgment held the next input in the socket,
    # the 250 ms lease expired mid-play, and the game paused under the player.
    import asyncio, uuid
    cfg=settings(tmp_path)
    sid=str(uuid.uuid4());generation=str(uuid.uuid4())
    class Manager:
        session={'session_id':sid,'owner':'operator'};process=True;connections=set()
        async def command(self,op,connection,data):
            if data['seq']==1: await asyncio.sleep(.4)
            return {'port':0,'epoch':1,'seq':data['seq'],'mask':0,'frame':data['seq']}
        async def disconnect(self,*args):pass
        async def stop(self,**kwargs):pass
    m=Manager();m.generation=generation
    app=create_app(cfg)
    with TestClient(app) as c:
        app.state.manager=m
        with c.websocket_connect('/sessions/'+sid+'/input',headers={'Authorization':'Bearer '+'a'*48}) as ws:
            ws.receive_json()
            for seq in (1,2):
                ws.send_json(dict(version=1,session_id=sid,generation=generation,lease='a'*48,epoch=1,seq=seq,port=0,mask=0))
            assert [ws.receive_json()['applied']['seq'] for _ in range(2)]==[2,1]


def test_cancelled_input_cannot_abandon_timeout_cleanup():
    # Review 2026-10-20: input tasks are cancelled when their socket closes. A
    # cancel during command-timeout cleanup left stopping=True and the worker alive.
    import asyncio, queue
    class Fake:
        process=True;stopping=False;pending={};commands=queue.Queue();done=False
        async def stop(self,*args):
            await asyncio.sleep(.2);self.done=True
    async def scenario():
        fake=Fake(); fake.ready=asyncio.Event(); fake.ready.set()
        task=asyncio.create_task(Sessions.command(fake,'input','c',{}))
        await asyncio.sleep(.8);task.cancel()
        await asyncio.gather(task,return_exceptions=True)
        await asyncio.sleep(.3)
        return fake.done
    assert asyncio.run(scenario())


def test_lease_survives_throttled_browser_heartbeat():
    # Live 2026-10-20: the browser's 50 ms timer ran at about 1 Hz, so a 250 ms
    # lease expired before the second input; the first arrived ~500 ms after grant.
    from app.control import LEASE_TTL as T
    a = Arbiter([0]); g=a.control('one',0,'acquire',0)
    a.frame(0); assert a.frame(.6)[0] == [0,0] and 0 in a.leases
    a.submit('one',intent(g),.6); assert a.frame(.61)[0] == [256,0]
    a.submit('one',intent(g,2),1.6); assert a.frame(1.61)[0] == [256,0]  # 1 s heartbeat
    assert a.frame(1.6+T)[0] == [0,0] and 0 not in a.leases  # silence still expires


def test_tap_inside_one_frame_still_reaches_the_game():
    # Live 2026-10-20: Enter's press and release arrived before the same frame; the
    # release replaced the press and Crystalis never saw Start.
    a = Arbiter([0]); g=a.control('one',0,'acquire',0); a.frame(0)
    a.submit('one',intent(g,1,mask=8),.01); a.submit('one',intent(g,2,mask=0),.011)
    masks, applied, _ = a.frame(.02)
    assert masks == [8,0] and applied[0]['mask'] == 8
    assert a.frame(.03)[0] == [0,0]  # held for exactly one frame


def test_latched_direction_yields_to_the_current_opposite():
    a = Arbiter([0]); g=a.control('one',0,'acquire',0); a.frame(0)
    a.submit('one',intent(g,1,mask=64),.01); a.submit('one',intent(g,2,mask=128),.011)  # left then right
    assert a.frame(.02)[0] == [128,0]


def test_flick_across_both_directions_keeps_the_newest():
    # Review 2026-10-20: left, right, release inside one frame cleared the axis.
    a = Arbiter([0]); g=a.control('one',0,'acquire',0); a.frame(0)
    for seq,mask in ((1,64),(2,128),(3,0)): a.submit('one',intent(g,seq,mask=mask),.01+seq/1000)
    assert a.frame(.02)[0] == [128,0]


def test_latch_does_not_follow_a_takeover():
    a = Arbiter([0]); g=a.control('one',0,'acquire',0); a.frame(0)
    a.submit('one',intent(g,1,mask=8),.01); a.submit('one',intent(g,2,mask=0),.011)
    t = a.control('two',0,'takeover',.012)
    assert a.frame(.02)[0] == [0,0]  # neutral handoff frame, no inherited Start
    a.submit('two',intent(t,1,mask=0),.021); assert a.frame(.03)[0] == [0,0]


def test_command_waits_for_a_starting_worker_instead_of_failing_the_session():
    # Live 2026-10-20: New game's resume arrived while the core was still loading
    # (~0.5 s); with Nova it passed the 0.75 s command timeout and the session failed.
    import asyncio, queue
    class Fake:
        process=True;stopping=False;pending={};commands=queue.Queue();urgent=queue.Queue();session={}
        stopped=False
        async def stop(self,*a): self.stopped=True
        def persist(self): pass
    async def scenario():
        fake=Fake(); fake.ready=asyncio.Event()
        async def worker():
            await asyncio.sleep(1.0); fake.ready.set()  # slower than the 0.75 s budget
            while True:
                try: cmd=fake.urgent.get_nowait(); break
                except queue.Empty: await asyncio.sleep(.01)
            fake.pending[cmd['id']].set_result({'result':{'state':'running'}})
        task=asyncio.create_task(worker())
        result=await Sessions.command(fake,'resume'); await task
        return result, fake.stopped
    assert asyncio.run(scenario())==({'state':'running'},False)


def test_waiters_released_together_still_respect_the_pending_cap():
    # Review 2026-10-20: inputs waiting on startup were not counted; all woke at once
    # and could overflow the 32-slot queue, failing the new session.
    import asyncio, queue
    class Fake:
        process=True;stopping=False;pending={};commands=queue.Queue(32);urgent=queue.Queue(8)
        session={'session_id':'s1'};stopped=False
        async def stop(self,*a): self.stopped=True
    async def scenario():
        fake=Fake(); fake.ready=asyncio.Event()
        tasks=[asyncio.create_task(Sessions.command(fake,'input','c',{})) for _ in range(40)]
        await asyncio.sleep(.05); fake.ready.set(); await asyncio.sleep(.05)
        for f in list(fake.pending.values()): f.set_result({'result':{}})
        out=await asyncio.gather(*tasks,return_exceptions=True)
        return fake.stopped, sum(isinstance(o,ValueError) and str(o)=='unavailable_or_busy' for o in out), fake.commands.qsize()
    stopped, refused, queued = asyncio.run(scenario())
    assert not stopped and refused == 16 and queued == 24
