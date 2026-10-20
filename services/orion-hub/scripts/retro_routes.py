"""Hub network-boundary Retro access and same-origin UDS proxy; no chat/bus path."""
import asyncio
from pathlib import Path
import time

import aiohttp
from fastapi import APIRouter, Request, WebSocket, HTTPException
from fastapi.responses import FileResponse, JSONResponse


def create_router(config):
    router = APIRouter()
    sockets = set()
    planning = False
    last_plan = float("-inf")

    def enabled():
        if not config.HUB_RETRO_ENABLED: raise HTTPException(503, 'retro_disabled')

    def origin(headers):
        expected = config.HUB_RETRO_ORIGIN.rstrip('/')
        return bool(expected) and headers.get('origin') == expected

    def credentials():
        value = Path(config.HUB_RETRO_AUTH_DIR, 'hub.token').read_text().strip()
        if len(value)<32: raise HTTPException(503, 'retro_credential_missing')
        return {'Authorization':'Bearer '+value, 'Content-Type':'application/json'}

    def client():
        return aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=config.HUB_RETRO_SOCKET),
                                     timeout=aiohttp.ClientTimeout(total=5))

    async def authorize(request):
        enabled()
        if request.method != 'GET' and not origin(request.headers): raise HTTPException(403, 'wrong_origin')

    async def body(request):
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data)>4096: raise HTTPException(413, 'message_too_large')
        return bytes(data)

    @router.get('/retro')
    async def page():
        enabled()
        return FileResponse(Path(__file__).resolve().parents[1]/'templates/retro.html',
                            headers={'Cache-Control':'no-store', 'Referrer-Policy':'no-referrer'})

    @router.api_route('/api/retro/{path:path}', methods=['GET','POST'])
    async def proxy(path: str, request: Request):
        nonlocal planning, last_plan
        import re
        await authorize(request)
        if request.method == 'POST' and re.fullmatch(r'sessions/[0-9a-f-]{36}/plan', path):
            if not getattr(config, 'HUB_RETRO_COGNITION_ENABLED', False):
                raise HTTPException(503, 'retro_cognition_disabled')
            if planning: raise HTTPException(429, 'planner_busy')
            import json
            try:
                data = json.loads(await body(request))
                if (not isinstance(data, dict) or set(data) != {'connection_id', 'generation', 'port', 'expected_epoch'}
                        or type(data['port']) is not int or data['port'] not in (0, 1)
                        or type(data['expected_epoch']) is not int
                        or not all(isinstance(data[k], str) and len(data[k]) == 36
                                   for k in ('connection_id', 'generation'))):
                    raise ValueError()
            except (ValueError, TypeError): raise HTTPException(422, 'invalid_plan_request')
            planning = True
            try:
                from scripts.retro_planner import choose
                async with client() as c:
                    async with c.get('http://retro/'+path.rsplit('/', 1)[0], headers=credentials()) as r:
                        state = await r.json()
                        if r.status != 200: return JSONResponse(state, status_code=r.status)
                    policy = next((p for p in state.get('scripted', []) if p['port'] == data['port']
                                   and p['epoch'] == data['expected_epoch'] and p.get('cognition')), None)
                    if not policy or state['generation'] != data['generation']:
                        raise HTTPException(409, 'stale_controller')
                    attempt = policy['cognition']['attempt']
                    if not attempt: raise HTTPException(409, 'observation_pending')
                    accepted = policy['cognition'].get('goal')
                    if accepted and accepted['attempt_id'] == attempt['attempt_id']:
                        return JSONResponse(accepted)
                    if time.monotonic()-last_plan < 5:
                        raise HTTPException(429, 'planner_rate_limited')
                    last_plan = time.monotonic()
                    choice = await choose(config, attempt)
                    goal = data | choice | {'attempt_id': attempt['attempt_id']}
                    async with c.post('http://retro/'+path.rsplit('/', 1)[0]+'/goal',
                                      headers=credentials(), json=goal) as r:
                        return JSONResponse(await r.json(), status_code=r.status)
            except HTTPException: raise
            except Exception: raise HTTPException(503, 'planner_unavailable_or_invalid')
            finally: planning = False
        valid = ((request.method=='GET' and path in ('games','health/ready')) or
                 (request.method=='POST' and path=='sessions') or
                 re.fullmatch(r'sessions/[0-9a-f-]{36}'+('(?:/evidence)?' if request.method=='GET' else '/(?:control|pause|resume|stop|scripted|goal)'),path))
        if not valid: raise HTTPException(404)
        try:
            async with client() as c:
                async with c.request(request.method,'http://retro/'+path,headers=credentials(),data=await body(request)) as r:
                    result=await r.json()
                    return JSONResponse(result,status_code=r.status,headers={'Cache-Control':'no-store'})
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError): raise HTTPException(503, 'retro_unavailable')

    @router.websocket('/api/retro/sessions/{sid}/{channel}')
    async def relay(ws: WebSocket, sid: str, channel: str):
        import re
        if (not config.HUB_RETRO_ENABLED or not origin(ws.headers)
                or channel not in ('input','media') or not re.fullmatch(r'[0-9a-f-]{36}',sid) or len(sockets)>=8):
            await ws.close(code=1008);return
        sockets.add(ws)
        tasks=[]
        try:
            async with client() as c:
                async with c.ws_connect('http://retro/sessions/'+sid+'/'+channel,headers=credentials(),max_msg_size=1200000) as upstream:
                    await ws.accept()
                    async def receive():
                        count=0;start=time.monotonic()
                        while True:
                            try: msg=await asyncio.wait_for(ws.receive_text(),.5)
                            except asyncio.TimeoutError: continue
                            if time.monotonic()-start>=1: start=time.monotonic();count=0
                            count+=1
                            if channel=='media' or len(msg.encode())>4096 or count>120: return
                            await asyncio.wait_for(upstream.send_str(msg),.25)
                    async def send():
                        while True:
                            try: msg=await asyncio.wait_for(upstream.receive(),.5)
                            except asyncio.TimeoutError: continue
                            if msg.type==aiohttp.WSMsgType.BINARY:
                                await asyncio.wait_for(ws.send_bytes(msg.data),.25)
                            elif msg.type==aiohttp.WSMsgType.TEXT:
                                await asyncio.wait_for(ws.send_text(msg.data),.25)
                            else: return
                    tasks=[asyncio.create_task(receive()),asyncio.create_task(send())]
                    await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
        except (aiohttp.ClientError, OSError, asyncio.TimeoutError, RuntimeError): pass
        finally:
            sockets.discard(ws)
            for task in tasks: task.cancel()
            if tasks: await asyncio.gather(*tasks,return_exceptions=True)
            try: await ws.close()
            except RuntimeError: pass
    return router


def configured_router():
    from app.settings import settings
    return create_router(settings)
