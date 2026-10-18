"""Disposable service smoke and failure injection. Never starts a live deployment."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
import uuid
import httpx
from websockets.asyncio.client import unix_connect
from orion.retro.pad_lab import build_rom
from operator_smoke import smoke


def docker(*args):
    return subprocess.check_output(['docker',*args],text=True).strip()


async def faults(socket, token, name, evidence):
    h={'Authorization':'Bearer '+token}
    with httpx.Client(transport=httpx.HTTPTransport(uds=str(socket)),base_url='http://localhost',headers=h) as client:
        def call(path,body=None):
            r=client.get(path) if body is None else client.post(path,json=body)
            r.raise_for_status();return r.json()
        async def start():
            s=call('/sessions',dict(game_id='pad-lab',idempotency_key=str(uuid.uuid4())))
            path='/sessions/'+s['session_id']
            for _ in range(100):
                s=call(path)
                if s['state']=='running':return path,s
                assert s['state']!='failed',s
                await asyncio.sleep(.02)
            raise AssertionError('worker did not start')
        old,s=await start()
        old_ws=await unix_connect(str(socket),uri='ws://localhost'+old+'/input',additional_headers=h)
        await old_ws.recv()
        call(old+'/stop',{})
        current,s=await start()
        await old_ws.close();await asyncio.sleep(.15)
        assert call(current)['state']=='running','old socket paused new session'
        async with unix_connect(str(socket),uri='ws://localhost'+current+'/input',additional_headers=h) as ws:
            hello=json.loads(await ws.recv())
            g=call(current+'/control',dict(connection_id=hello['connection_id'],port=0))
            call(current+'/resume',{})  # retry must not revoke lease
            data=dict(version=1,session_id=s['session_id'],generation=hello['generation'],
                      lease=g['lease'],epoch=g['epoch'],seq=1,port=0,mask=256)
            await ws.send(json.dumps(dict(data,lease='é'*32)))
            assert 'rejected' in json.loads(await ws.recv())
            await ws.send(json.dumps(data));assert 'applied' in json.loads(await ws.recv())
            await asyncio.sleep(.4)
            assert call(current)['state']=='paused','expired lease kept running'
        call(current+'/stop',{})
        current,s=await start()
        # Kill only this disposable container's spawned emulator child.
        kill_script="""import os,signal
from pathlib import Path
pids=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes()
  if b'multiprocessing.spawn' in cmd and int(p.name)!=os.getpid():pids.append(int(p.name))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert len(pids)==1,pids
os.kill(pids[0],signal.SIGKILL)
"""
        docker('exec',name,'python','-c',kill_script)
        for _ in range(100):
            status=call(current)
            if status['state']=='failed':break
            await asyncio.sleep(.02)
        assert status['state']=='failed',status
        current,s=await start()
        await asyncio.sleep(5.3)
        assert call(current)['state']=='paused','unclaimed start continued past grace'
        call(current+'/stop',{})
        quota=evidence/'quota-fixture'
        with quota.open('xb') as f:f.truncate(2147483648)
        try:
            s=call('/sessions',dict(game_id='pad-lab',idempotency_key=str(uuid.uuid4())))
            path='/sessions/'+s['session_id']
            for _ in range(100):
                status=call(path)
                if status['state']=='failed':break
                await asyncio.sleep(.02)
            assert status['state']=='failed' and 'evidence_quota' in status['reason'],status
        finally:
            quota.unlink()
        assert client.get('/health/live',headers={'Authorization':'invalid'}).status_code==401
        assert client.get('/health/live',headers=dict(h,Origin='https://example.invalid')).status_code==403
    return {'cross_session_disconnect':'PASS','duplicate_resume':'PASS','malformed_token':'PASS',
            'lease_expiry':'PASS','worker_kill':'PASS','initial_grace':'PASS','authorization':'PASS','live_storage_quota':'PASS'}


def run(image, output):
    if os.getuid()==0:raise ValueError('run evaluations as a non-root operator')
    output.mkdir(parents=True,mode=0o700,exist_ok=False)
    for directory in ('catalog','evidence','ipc'):(output/directory).mkdir(mode=0o700)
    rom=build_rom();(output/'catalog/pad-lab.nes').write_bytes(rom)
    (output/'catalog/catalog.json').write_text(json.dumps([dict(game_id='pad-lab',title='Pad Lab',file='pad-lab.nes',
        rom_sha256=hashlib.sha256(rom).hexdigest(),ports=[0,1],provenance='original authored diagnostic')]))
    token=secrets.token_urlsafe(48);token_path=output/'token';token_path.write_text(token);token_path.chmod(0o600)
    name='orion-retro-eval-'+uuid.uuid4().hex[:12]
    args=['run','-d','--name',name,'--label','orion.purpose=retro-eval','--network','none','--read-only',
          '--user',f'{os.getuid()}:{os.getgid()}','--cap-drop','ALL','--security-opt','no-new-privileges',
          '--pids-limit','64','--memory','512m','--memory-swap','512m','--cpus','2',
          '--tmpfs','/tmp:rw,noexec,nosuid,size=16m','-e','RETRO_ENABLED=true']
    for source,dest,ro in [(output/'catalog','/catalog',True),(output/'evidence','/evidence',False),
                            (output/'ipc','/ipc',False),(token_path,'/run/secrets/retro_operator',True)]:
        args += ['--mount',f'type=bind,src={source.resolve()},dst={dest}'+(',readonly' if ro else '')]
    try:
        docker(*args,image)
        (output/'container.json').write_text(docker('inspect',name)+'\n')
        socket=output/'ipc/retro.sock'
        for _ in range(100):
            if socket.exists():break
            time.sleep(.1)
        basic=asyncio.run(smoke(socket,token,'pad-lab',3))
        fault_results=asyncio.run(faults(socket,token,name,output/'evidence'))
        network=docker('exec',name,'python','-c',"import socket; s=socket.socket(); s.settimeout(1)\ntry: s.connect(('1.1.1.1',443))\nexcept OSError: print('DENIED')\nelse: raise RuntimeError('unexpected egress')\nfinally: s.close()")
        assert network=='DENIED'
        result={'status':'PASS','api':basic,'faults':fault_results,'egress':network}
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print('PASS',output/'result.json')
    finally:
        subprocess.run(['docker','stop','--timeout','2',name],capture_output=True)
        subprocess.run(['docker','rm',name],capture_output=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image',default='orion-retro-retro')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.image,a.output.resolve())
