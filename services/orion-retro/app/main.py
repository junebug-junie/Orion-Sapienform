"""Operator-only P1 API. No public/browser identity or Hub integration yet."""
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import time
import uuid
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from settings import Settings
from .contracts import Start, Control, Input
from .session import Sessions


def create_app(settings=None):
    settings = settings or Settings.from_env()
    @asynccontextmanager
    async def lifespan(app):
        app.state.manager = None
        app.state.token = settings.token()
        if settings.enabled: app.state.manager = Sessions(settings)
        yield
        if app.state.manager: await app.state.manager.stop(reason='service_shutdown')
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def auth(headers):
        supplied = headers.get('authorization', '')
        return hmac.compare_digest(supplied.encode(), ('Bearer '+app.state.token).encode())

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

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        from starlette.responses import JSONResponse
        return JSONResponse({'detail':str(exc)}, status_code=409)

    @app.get('/health/live')
    async def live(): return {'status':'alive', 'enabled':settings.enabled}

    @app.get('/health/ready')
    async def ready():
        m = manager()
        return {'status':'ready', 'session':m.session}

    @app.get('/games')
    async def games(): return manager().games()

    @app.post('/sessions')
    async def start(body: Start):
        return await manager().start(body.game_id, body.idempotency_key)

    @app.get('/sessions/{sid}')
    async def status(sid: str):
        m = manager(); m.require(sid); return m.session

    @app.post('/sessions/{sid}/{operation}')
    async def lifecycle(sid: str, operation: str, request: Request):
        m = manager(); m.require(sid)
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
    async def evidence(sid: str):
        m = manager(); m.require(sid)
        return {'applied':list(m.history), 'complete_history':False,
                'artifact_id':sid+'.jsonl'}

    @app.websocket('/sessions/{sid}/input')
    async def inputs(ws: WebSocket, sid: str):
        if not auth(ws.headers) or ws.headers.get('origin'):
            await ws.close(code=1008); return
        m = app.state.manager
        if not m or not m.session or sid != m.session['session_id'] or not m.process or len(m.connections)>=2:
            await ws.close(code=1008); return
        connection = str(uuid.uuid4()); m.connections.add(connection)
        await ws.accept()
        async def send(row):
            await asyncio.wait_for(ws.send_json(row), .25)
        try:
            await send({'connection_id':connection, 'generation':m.generation})
            window = time.monotonic(); count = 0
            while connection in m.connections:
                try: raw = await asyncio.wait_for(ws.receive_text(), .1)
                except asyncio.TimeoutError: continue
                now = time.monotonic()
                if now-window >= 1: window=now; count=0
                count += 1
                if len(raw.encode())>4096 or count>120:
                    await ws.close(code=1008); break
                try:
                    data = Input.model_validate_json(raw).model_dump()
                    if data['session_id'] != sid or data['generation'] != m.generation:
                        raise ValueError('session_generation_mismatch')
                    result = await m.command('input', connection, data)
                    await send({'applied':result})
                except ValidationError:
                    await send({'rejected':'invalid_input'})
                except ValueError as exc:
                    await send({'rejected':str(exc)[:200]})
        except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
            pass
        finally:
            await m.disconnect(connection, sid, m.generation)
            try: await ws.close()
            except RuntimeError: pass
    return app


app = create_app()
