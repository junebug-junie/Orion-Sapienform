import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

spec=importlib.util.spec_from_file_location('retro_routes',Path(__file__).resolve().parents[2]/'scripts/retro_routes.py')
routes=importlib.util.module_from_spec(spec);spec.loader.exec_module(routes)


def test_network_boundary_without_access_code(tmp_path):
    (tmp_path/'hub.token').write_text('y'*48)
    cfg=SimpleNamespace(HUB_RETRO_ENABLED=True,HUB_RETRO_ORIGIN='http://testserver',HUB_RETRO_AUTH_DIR=str(tmp_path),HUB_RETRO_SOCKET='/nonexistent')
    app=FastAPI();app.include_router(routes.create_router(cfg))
    with TestClient(app) as c:
        # No browser credential file or cookie: reaches upstream (unavailable in this test).
        assert c.get('/api/retro/games').status_code==503
        assert c.post('/api/retro/sessions',json={}).status_code==403
        assert c.post('/api/retro/sessions',json={},headers={'Origin':'https://evil.example'}).status_code==403
        good={'Origin':'http://testserver'}
        assert c.post('/api/retro/sessions',json={},headers=good).status_code==503
        assert c.post('/api/retro/login',json={},headers=good).status_code==404
        assert c.post('/api/retro/logout',json={},headers=good).status_code==404
        assert c.get('/api/retro/arbitrary').status_code==404
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect('/api/retro/sessions/'+'a'*36+'/input',headers={'Origin':'https://evil.example'}):pass
        html=c.get('/retro').text
        assert '/static/js/retro-panel.js' in html
        assert 'id="login"' not in html and 'id="code"' not in html
        assert 'y'*48 not in html
        cfg.HUB_RETRO_ENABLED=False
        assert c.get('/api/retro/games').status_code==503
