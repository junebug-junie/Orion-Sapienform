"""Real browser/native-core smoke on a disposable isolated container and local Hub router.

Uses authored Pad Lab by default. --catalog explicitly selects an operator-owned catalog.
No production service or Hub process is changed. Browser credentials stay in private files.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace


def proxy(output, port):
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    hub=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('retro_proxy',hub/'scripts/retro_routes.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    config=SimpleNamespace(HUB_RETRO_ENABLED=True,HUB_RETRO_ORIGIN=f'http://127.0.0.1:{port}',
        HUB_RETRO_AUTH_DIR=str(output/'auth'),HUB_RETRO_SOCKET=str(output/'ipc/retro.sock'))
    sys.path.insert(0, str(hub))
    from scripts import retro_planner
    async def fixture_choice(config, attempt):
        return dict(goal='wait', rationale='Synthetic browser fixture: unsupported Pad Lab observation.',
                    planner_id='browser-fixture')
    retro_planner.choose = fixture_choice
    config.HUB_RETRO_COGNITION_ENABLED = True
    app=FastAPI();app.include_router(mod.create_router(config));app.mount('/static',StaticFiles(directory=hub/'static'))
    @app.get('/__retro_tab_test')
    async def tab_fixture():
        # Use the actual template fragments and tab asset; omit unrelated telemetry.
        import re
        from fastapi.responses import HTMLResponse
        html=(hub/'templates/index.html').read_text()
        retro=re.search(r'<a id="retroTabButton".*?</a>',html,re.S).group()
        home=re.search(r'<a\s+id="hubTabButton".*?</a>',html,re.S).group()
        panel=re.search(r'<section id="retro".*?</section>',html,re.S).group()
        return HTMLResponse('<!doctype html><style>.hidden{display:none}iframe{width:100%;height:1200px}</style>'+retro+home+
            '<div id="appPanels">'+panel+'</div><script src="/static/js/retro-tab.js" defer></script>')
    uvicorn.run(app,host='127.0.0.1',port=port,log_level='warning')


def run(args):
    import httpx
    from orion.retro.pad_lab import build_rom
    output=args.output.resolve();output.mkdir(mode=0o700,exist_ok=False,parents=True)
    for d in ('auth','catalog','ipc','evidence'):(output/d).mkdir(mode=0o700)
    for name in ('hub.token','operator.token'):
        p=output/'auth'/name;p.write_text(secrets.token_urlsafe(48));p.chmod(0o600)
    if args.catalog: catalog=args.catalog.resolve()
    else:
        catalog=output/'catalog';rom=build_rom();(catalog/'pad-lab.nes').write_bytes(rom)
        (catalog/'catalog.json').write_text(json.dumps([dict(game_id='pad-lab',title='Pad Lab',file='pad-lab.nes',rom_sha256=hashlib.sha256(rom).hexdigest(),ports=[0,1],provenance='authored')]))
    name='orion-retro-browser-eval-'+uuid.uuid4().hex[:8]
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    cmd=['docker','run','-d','--name',name,'--network','none','--read-only','--init',
        '--user',f'{os.getuid()}:{os.getgid()}','--cap-drop','ALL','--security-opt','no-new-privileges',
        '--pids-limit','64','--memory','512m','--memory-swap','512m','--cpus','2','--tmpfs','/tmp:rw,noexec,nosuid,size=16m','-e','RETRO_ENABLED=true','-e','RETRO_COGNITION_ENABLED=true']
    for src,dst,ro in [(catalog,'/catalog',True),(output/'ipc','/ipc',False),(output/'evidence','/evidence',False),(output/'auth/operator.token','/run/secrets/retro_operator',True),(output/'auth/hub.token','/run/secrets/retro_hub',True)]:
        cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    process=None
    try:
        subprocess.run(cmd+[args.image],check=True,stdout=subprocess.DEVNULL)
        for _ in range(100):
            if (output/'ipc/retro.sock').exists():break
            time.sleep(.1)
        process=subprocess.Popen([sys.executable,__file__,'--proxy','--output',str(output),'--port',str(port)],stdout=(output/'proxy.log').open('w'),stderr=subprocess.STDOUT)
        for _ in range(100):
            try:
                if httpx.get(f'http://127.0.0.1:{port}/retro').status_code==200:break
            except httpx.TransportError:pass
            time.sleep(.1)
        env=dict(os.environ,RETRO_EVAL_URL=f'http://127.0.0.1:{port}',RETRO_EVAL_OUTPUT=str(output),RETRO_EVAL_GAME=args.game,RETRO_EVAL_SECONDS=str(args.seconds),RETRO_ACK_DELAY_MS=str(args.ack_delay_ms))
        subprocess.run(['node',str(Path(__file__).with_suffix('.mjs'))],env=env,check=True)
        print('PASS',output/'browser.json')
    finally:
        if process:process.terminate();process.wait(timeout=5)
        subprocess.run(['docker','stop','--timeout','2',name],capture_output=True)
        subprocess.run(['docker','rm',name],capture_output=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image',default='orion-retro-retro');p.add_argument('--catalog',type=Path)
    p.add_argument('--ack-delay-ms',type=int,default=0)
    p.add_argument('--game',default='pad-lab');p.add_argument('--seconds',type=int,default=8)
    p.add_argument('--proxy',action='store_true');p.add_argument('--port',type=int)
    a=p.parse_args()
    if a.proxy:proxy(a.output,a.port)
    else:run(a)
