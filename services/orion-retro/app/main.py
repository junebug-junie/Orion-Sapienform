"""Owner-scoped CLI/Hub authority. Browser authentication remains in Hub."""
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import time
import uuid
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from settings import Settings
from .contracts import Start, Control, Input, ScriptControl, Goal
from .session import Sessions


def create_app(settings=None):
    settings = settings or Settings.from_env()
    @asynccontextmanager
    async def lifespan(app):
        app.state.manager = None
        app.state.token = settings.token()
        app.state.hub_token = settings.hub_token_file.read_text().strip() if settings.hub_token_file.is_file() else ""
        if app.state.hub_token and (len(app.state.hub_token)<32 or app.state.hub_token == app.state.token):
            raise ValueError("Hub requires a distinct credential")
        if settings.enabled: app.state.manager = Sessions(settings)
        yield
        if app.state.manager: await app.state.manager.stop(reason='service_shutdown')
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def auth(headers):
        supplied = headers.get('authorization', '')
        if hmac.compare_digest(supplied.encode(), ('Bearer '+app.state.token).encode()): return 'operator'
        if app.state.hub_token and hmac.compare_digest(supplied.encode(), ('Bearer '+app.state.hub_token).encode()): return 'hub_operator'
        return None

    @app.middleware('http')
    async def boundary(request, call_next):
        from starlette.responses import JSONResponse
        if not auth(request.headers): return JSONResponse({'detail':'unauthorized'}, status_code=401)
        # P1 is CLI-only. Browser origins are deliberately rejected until Hub auth lands.
        if request.headers.get('origin'): return JSONResponse({'detail':'browser_not_enabled'}, status_code=403)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 4096: return JSONResponse({'detail':'message_too_large'}, status_code=413)
        request._body = bytes(body)
        return await call_next(request)

    def manager():
        m = app.state.manager
        if m is None: raise HTTPException(503, 'retro_disabled')
        return m

    def owned(sid, headers):
        m = manager(); m.require(sid)
        if m.session['owner'] != auth(headers): raise HTTPException(403, 'wrong_owner')
        return m

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        from starlette.responses import JSONResponse
        return JSONResponse({'detail':str(exc)}, status_code=409)

    @app.get('/health/live')
    async def live(): return {'status':'alive', 'enabled':settings.enabled}

    @app.get('/health/ready')
    async def ready(request: Request):
        m = manager()
        return {'status':'ready', 'session':m.session if not m.session or m.session['owner']==auth(request.headers) else None}

    @app.get('/games')
    async def games(): return manager().games()

    @app.post('/sessions')
    async def start(body: Start, request: Request):
        return await manager().start(body.game_id, body.idempotency_key, auth(request.headers))

    @app.get('/sessions/{sid}')
    async def status(sid: str, request: Request):
        return owned(sid, request.headers).session

    @app.post('/sessions/{sid}/{operation}')
    async def lifecycle(sid: str, operation: str, request: Request):
        m = owned(sid, request.headers)
        if operation == 'goal':
            if not settings.cognition_enabled: raise HTTPException(503, 'retro_cognition_disabled')
            try: body = Goal.model_validate_json(await request.body())
            except ValidationError: raise HTTPException(422, 'invalid_goal')
            if body.generation != m.generation: raise ValueError('session_generation_mismatch')
            if body.connection_id not in m.connections: raise ValueError('connection_not_live')
            return await m.command('goal', body.connection_id,
                                   body.model_dump(exclude={'connection_id', 'generation'}))
        if operation == 'scripted':
            try: body = ScriptControl.model_validate_json(await request.body())
            except ValidationError: raise HTTPException(422, 'invalid_script_control')
            if body.cognition and not settings.cognition_enabled: raise HTTPException(503, 'retro_cognition_disabled')
            if body.generation != m.generation: raise ValueError('session_generation_mismatch')
            if body.connection_id not in m.connections: raise ValueError('connection_not_live')
            return await m.command('scripted', body.connection_id,
                                   body.model_dump(exclude={'connection_id', 'generation'}))
        if operation == 'control':
            try: body = Control.model_validate_json(await request.body())
            except ValidationError: raise HTTPException(422, 'invalid_control')
            if body.connection_id not in m.connections: raise ValueError('connection_not_live')
            return await m.command('control', body.connection_id,
                                   dict(port=body.port, action=body.action))
        if operation in ('pause','resume'): return await m.command(operation)
        if operation == 'stop':
            await m.stop(); return m.session
        raise HTTPException(404)

    @app.get('/sessions/{sid}/evidence')
    async def evidence(sid: str, request: Request):
        m = owned(sid, request.headers)
        return {'applied':list(m.history), 'complete_history':False,
                'artifact_id':sid+'.jsonl'}

    @app.websocket('/sessions/{sid}/input')
    async def inputs(ws: WebSocket, sid: str):
        if not auth(ws.headers) or ws.headers.get('origin'):
            await ws.close(code=1008); return
        m = app.state.manager
        if not m or not m.session or sid != m.session['session_id'] or m.session['owner'] != auth(ws.headers) or not m.process or len(m.connections)>=2:
            await ws.close(code=1008); return
        connection = str(uuid.uuid4()); m.connections.add(connection)
        await ws.accept()
        lock = asyncio.Lock(); inflight = set(); failed = asyncio.Event()
        async def send(row):
            async with lock:
                await asyncio.wait_for(ws.send_json(row), .25)
        async def apply(data):
            # Each input waits for its native frame off the read loop. A slow
            # acknowledgment must not hold newer input until the lease expires.
            try:
                try:
                    result = await m.command('input', connection, data)
                    await send({'applied':result})
                except ValueError as exc:
                    reason = str(exc)[:200]
                    # A lost reply cannot prove that the native frame never ran.
                    verdict = 'unknown' if reason in ('worker_unavailable', 'session_stopped') else 'rejected'
                    await send({verdict:reason, 'seq':data['seq'], 'epoch':data['epoch'], 'port':data['port']})
            except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
                failed.set()
        try:
            await send({'connection_id':connection, 'generation':m.generation})
            window = time.monotonic(); count = 0
            while connection in m.connections and not failed.is_set():
                try: raw = await asyncio.wait_for(ws.receive_text(), .1)
                except asyncio.TimeoutError: continue
                now = time.monotonic()
                if now-window >= 1: window=now; count=0
                count += 1
                if len(raw.encode())>4096 or count>120:
                    await ws.close(code=1008); break
                try:
                    data = Input.model_validate_json(raw).model_dump()
                except ValidationError:
                    await send({'rejected':'invalid_input'}); continue
                if data['session_id'] != sid or data['generation'] != m.generation:
                    await send({'rejected':'session_generation_mismatch', 'seq':data['seq'],
                                'epoch':data['epoch'], 'port':data['port']}); continue
                # Tasks start in creation order, so inputs reach the worker in arrival order.
                task = asyncio.create_task(apply(data))
                inflight.add(task); task.add_done_callback(inflight.discard)
        except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
            pass
        finally:
            for task in list(inflight): task.cancel()
            await asyncio.gather(*inflight, return_exceptions=True)
            await m.disconnect(connection, sid, m.generation)
            try: await ws.close()
            except RuntimeError: pass
    @app.websocket('/sessions/{sid}/media')
    async def media(ws: WebSocket, sid: str):
        if not auth(ws.headers) or ws.headers.get('origin'):
            await ws.close(code=1008); return
        try: m = owned(sid, ws.headers)
        except (ValueError, HTTPException):
            await ws.close(code=1008); return
        if getattr(m, 'viewers', 0) >= 2:
            await ws.close(code=1008); return
        m.viewers = getattr(m, 'viewers', 0)+1
        disconnected = None; version = -1
        try:
            await ws.accept()
            disconnected = asyncio.create_task(ws.receive())
            while m.process and m.session['session_id'] == sid and not disconnected.done():
                if m.latest_media is not None and m.media_version != version:
                    version = m.media_version
                    await asyncio.wait_for(ws.send_bytes(m.latest_media), .25)
                await asyncio.sleep(.005)
        except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError): pass
        finally:
            m.viewers -= 1  # Reclaim before any cancellable cleanup await.
            if disconnected:
                disconnected.cancel()
                await asyncio.gather(disconnected, return_exceptions=True)
            try: await ws.close()
            except RuntimeError: pass
    return app


app = create_app()
