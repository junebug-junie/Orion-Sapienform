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
    a = Arbiter([0,1]); p1=a.control('one',0,'acquire',0); p2=a.control('two',1,'acquire',0)
    a.frame(0); a.submit('one',intent(p1),0); a.submit('two',intent(p2,mask=128),.1)
    assert a.frame(.2)[0] == [256,128]
    assert a.frame(.26)[0] == [0,128]
    with pytest.raises(ValueError): a.submit('one',intent(p1,2),.26)
    a.disconnect('two'); assert a.frame(.27)[0] == [0,0]


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
