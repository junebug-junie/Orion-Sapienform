"""Scoped Retro operator login and same-origin UDS proxy; no chat/bus path."""
import asyncio
from collections import deque
import hmac
import json
from pathlib import Path
import secrets
import time
from urllib.parse import urlsplit

import aiohttp
from fastapi import APIRouter, Request, WebSocket, HTTPException
from fastapi.responses import FileResponse, JSONResponse


def create_router(config):
    router = APIRouter()
    cookies = {}; attempts = deque(maxlen=20); sockets = set()
    cookie_name = 'orion_retro'

    def enabled():
        if not config.HUB_RETRO_ENABLED: raise HTTPException(503, 'retro_disabled')

    def origin(headers):
        expected = config.HUB_RETRO_ORIGIN.rstrip('/')
        return bool(expected) and headers.get('origin') == expected

    def principal(cookie):
        now = time.monotonic()
        for key in list(cookies):
            if cookies[key] <= now: del cookies[key]
        return cookie in cookies

    def credentials():
        value = Path(config.HUB_RETRO_AUTH_DIR, 'hub.token').read_text().strip()
        if len(value)<32: raise HTTPException(503, 'retro_credential_missing')
        return {'Authorization':'Bearer '+value, 'Content-Type':'application/json'}

    def client():
        return aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=config.HUB_RETRO_SOCKET),
                                     timeout=aiohttp.ClientTimeout(total=5))

    async def authorize(request):
        enabled()
        if not principal(request.cookies.get(cookie_name)): raise HTTPException(401, 'retro_login_required')
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

    @router.post('/api/retro/login')
    async def login(request: Request):
        enabled()
        if not origin(request.headers): raise HTTPException(403, 'wrong_origin')
        now = time.monotonic()
        while attempts and now-attempts[0]>60: attempts.popleft()
        if len(attempts)>=20: raise HTTPException(429, 'try_later')
        attempts.append(now)
        try: supplied = json.loads(await body(request)).get('code', '')
        except (ValueError, AttributeError): raise HTTPException(400, 'invalid_login')
        try: expected = Path(config.HUB_RETRO_AUTH_DIR, 'browser.code').read_text().strip()
        except OSError: raise HTTPException(503, 'retro_login_not_configured')
        if len(expected)<32 or not isinstance(supplied,str) or not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(401, 'invalid_login')
        principal(None)
        if len(cookies)>=8: raise HTTPException(429, 'login_capacity')
        key=secrets.token_urlsafe(32); cookies[key]=now+8*3600
        response=JSONResponse({'authenticated':True})
        response.set_cookie(cookie_name,key,httponly=True,samesite='strict',max_age=8*3600,
                            secure=urlsplit(config.HUB_RETRO_ORIGIN).scheme=='https',path='/api/retro')
        response.headers['Cache-Control']='no-store'
        return response

    @router.post('/api/retro/logout')
    async def logout(request: Request):
        await authorize(request)
        cookies.pop(request.cookies.get(cookie_name), None)
        response=JSONResponse({'authenticated':False});response.delete_cookie(cookie_name,path='/api/retro')
        return response

    @router.api_route('/api/retro/{path:path}', methods=['GET','POST'])
    async def proxy(path: str, request: Request):
        import re
        await authorize(request)
        valid = ((request.method=='GET' and path in ('games','health/ready')) or
                 (request.method=='POST' and path=='sessions') or
                 re.fullmatch(r'sessions/[0-9a-f-]{36}'+('(?:/evidence)?' if request.method=='GET' else '/(?:control|pause|resume|stop)'),path))
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
        cookie=ws.cookies.get(cookie_name)
        if (not config.HUB_RETRO_ENABLED or not principal(cookie) or not origin(ws.headers)
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
                        while principal(cookie):
                            try: msg=await asyncio.wait_for(ws.receive_text(),.5)
                            except asyncio.TimeoutError: continue
                            if time.monotonic()-start>=1: start=time.monotonic();count=0
                            count+=1
                            if channel=='media' or len(msg.encode())>4096 or count>120: return
                            await asyncio.wait_for(upstream.send_str(msg),.25)
                    async def send():
                        while principal(cookie):
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
