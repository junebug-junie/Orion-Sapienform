import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

spec=importlib.util.spec_from_file_location('retro_routes',Path(__file__).resolve().parents[2]/'scripts/retro_routes.py')
routes=importlib.util.module_from_spec(spec);spec.loader.exec_module(routes)


def test_cookie_origin_and_explicit_route_boundary(tmp_path):
    (tmp_path/'browser.code').write_text('x'*48)
    (tmp_path/'hub.token').write_text('y'*48)
    cfg=SimpleNamespace(HUB_RETRO_ENABLED=True,HUB_RETRO_ORIGIN='http://testserver',HUB_RETRO_AUTH_DIR=str(tmp_path),HUB_RETRO_SOCKET='/nonexistent')
    app=FastAPI();app.include_router(routes.create_router(cfg))
    with TestClient(app) as c:
        assert c.get('/api/retro/games').status_code==401
        assert c.post('/api/retro/login',json={'code':'x'*48}).status_code==403
        assert c.post('/api/retro/login',json={'code':'x'*48},headers={'Origin':'https://evil.example'}).status_code==403
        good={'Origin':'http://testserver'}
        assert c.post('/api/retro/login',json={'code':'wrong'},headers=good).status_code==401
        r=c.post('/api/retro/login',json={'code':'x'*48},headers=good)
        assert r.status_code==200 and 'httponly' in r.headers['set-cookie'].lower()
        assert 'y'*48 not in r.text and 'x'*48 not in r.text
        assert c.post('/api/retro/sessions',json={}).status_code==403
        assert c.get('/api/retro/arbitrary').status_code==404
        assert c.get('/api/retro/games').status_code==503
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect('/api/retro/sessions/'+'a'*36+'/input',headers={'Origin':'https://evil.example'}):pass
        assert c.post('/api/retro/logout',json={},headers=good).status_code==200
        assert c.get('/api/retro/games').status_code==401
        assert '/static/js/retro-panel.js' in c.get('/retro').text
