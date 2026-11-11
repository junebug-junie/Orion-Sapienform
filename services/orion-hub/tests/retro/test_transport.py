"""Exercise backpressure and cleanup without touching a live game or bus."""
import asyncio
import logging
import subprocess
import sys
from types import SimpleNamespace

import aiohttp
from starlette.websockets import WebSocketDisconnect

from scripts import retro_transport as transport


class Browser:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def receive_text(self):
        item = await self.incoming.get()
        if isinstance(item, Exception): raise item
        return item

    async def send_bytes(self, data):
        self.started.set()
        await self.release.wait()
        self.sent.append(data)

    async def send_text(self, data):
        await self.send_bytes(data)


class Native:
    def __init__(self): self.incoming = asyncio.Queue(); self.sent = []
    async def receive(self): return await self.incoming.get()
    async def send_str(self, msg): self.sent.append(msg)
    def frame(self, data, kind=aiohttp.WSMsgType.BINARY):
        self.incoming.put_nowait(SimpleNamespace(type=kind, data=data))


def test_slow_media_keeps_only_latest_packet_and_other_routes_run():
    async def check():
        ws, native = Browser(), Native()
        before = set(asyncio.all_tasks())
        task = asyncio.create_task(transport.relay_messages(ws, native, 'media', 'fixture'))
        native.frame(b'first')
        await asyncio.wait_for(ws.started.wait(), 1)
        for i in range(100): native.frame(str(i).encode())
        # The loop stays responsive while this viewer is blocked. Drain producer.
        for _ in range(105): await asyncio.sleep(0)
        assert native.incoming.empty()
        ws.release.set()
        for _ in range(10): await asyncio.sleep(0)
        assert ws.sent == [b'first', b'99']
        ws.incoming.put_nowait(WebSocketDisconnect(1000))
        assert await asyncio.wait_for(task, 1) == 'browser_disconnected'
        assert set(asyncio.all_tasks()) == before
    asyncio.run(check())


def test_send_over_250ms_survives_but_stuck_send_closes_and_logs(monkeypatch, caplog):
    async def check():
        ws, native = Browser(), Native()
        task = asyncio.create_task(transport.relay_messages(ws, native, 'media', 'fixture'))
        native.frame(b'picture')
        await ws.started.wait()
        await asyncio.sleep(.3)
        assert not task.done()
        ws.release.set()
        for _ in range(5): await asyncio.sleep(0)
        assert ws.sent == [b'picture']
        ws.incoming.put_nowait(WebSocketDisconnect(1000))
        await task
        monkeypatch.setattr(transport, 'SEND_TIMEOUT_SEC', .01)
        ws, native = Browser(), Native()
        native.frame(b'picture')
        reason = await asyncio.wait_for(transport.relay_messages(ws, native, 'media', 'fixture'), 1)
        assert reason == 'send_timeout'
    with caplog.at_level(logging.INFO, logger=transport.logger.name): asyncio.run(check())
    assert 'sid=fixture channel=media reason=send_timeout' in caplog.text


def test_input_replies_are_ordered_and_never_dropped():
    async def check():
        ws, native = Browser(), Native(); ws.release.set()
        task = asyncio.create_task(transport.relay_messages(ws, native, 'input', 'fixture'))
        for i in range(24): native.frame(str(i), aiohttp.WSMsgType.TEXT)
        ws.incoming.put_nowait('buttons')
        for _ in range(150): await asyncio.sleep(0)
        assert ws.sent == [str(i) for i in range(24)]
        assert native.sent == ['buttons']
        native.frame(None, aiohttp.WSMsgType.CLOSE)
        assert await asyncio.wait_for(task, 1) == 'upstream_closed'
    asyncio.run(check())


def test_cancel_reclaims_all_child_tasks_and_ingress_limits_remain():
    async def check():
        for channel, message in [('media', 'forbidden'), ('input', 'x' * 4097)]:
            ws, native = Browser(), Native(); ws.incoming.put_nowait(message)
            assert await transport.relay_messages(ws, native, channel, 'fixture') == 'invalid_browser_message'
            assert native.sent == []
        before = set(asyncio.all_tasks())
        task = asyncio.create_task(transport.relay_messages(Browser(), Native(), 'media', 'fixture'))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert set(asyncio.all_tasks()) == before
    asyncio.run(check())


def test_close_diagnostics_are_enabled_by_actual_uvicorn_logging():
    code = ('import logging,uvicorn; uvicorn.Config("unused:app",log_level="info").configure_logging(); '
            'log=logging.getLogger("uvicorn.error.retro_transport"); '
            'assert log.isEnabledFor(logging.INFO); log.info("retro_relay_closed fixture")')
    run = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr
    assert 'retro_relay_closed fixture' in run.stderr


def test_shared_router_keeps_other_http_routes_responsive(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from scripts import retro_routes
    from starlette.websockets import WebSocket
    import threading
    blocked = threading.Event()
    async def blocked_send(self, data):
        blocked.set()
        await asyncio.Future()  # genuinely blocked viewer write, cancelled on close
    monkeypatch.setattr(WebSocket, 'send_bytes', blocked_send)
    native = Native(); native.frame(b'picture')
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        @asynccontextmanager
        async def ws_connect(self, *args, **kwargs): yield native
    monkeypatch.setattr(retro_routes.aiohttp, 'ClientSession', lambda **kwargs: Client())
    (tmp_path/'hub.token').write_text('x'*48)
    config = SimpleNamespace(HUB_RETRO_ENABLED=True, HUB_RETRO_ORIGIN='http://testserver',
                             HUB_RETRO_AUTH_DIR=str(tmp_path), HUB_RETRO_SOCKET='/unused')
    app = FastAPI(); app.include_router(retro_routes.create_router(config))
    @app.get('/unrelated/status')
    async def status(): return {'ok': True}
    @app.post('/unrelated/chat')
    async def chat(): return {'reply': 'fixture'}
    with TestClient(app) as client:
        with client.websocket_connect('/api/retro/sessions/'+'a'*36+'/media',
                                      headers={'Origin': 'http://testserver'}) as ws:
            assert blocked.wait(1), 'media write never entered the blocked state'
            assert client.get('/unrelated/status').json() == {'ok': True}
            assert client.post('/unrelated/chat').json() == {'reply': 'fixture'}
