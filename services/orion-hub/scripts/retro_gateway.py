"""Dedicated Retro relay process: realtime game traffic off Hub's shared event loop.

Live 2026-10-20 (evals/retro_play.mjs A/B, a Hub page open in another tab): via the
Hub process, input acknowledgments took 720 ms p50 and play dropped 4 times; via this
process, 17 ms and no drops. Same router, origin check and credentials as Hub's own
/api/retro routes; tailscale serve sends /api/retro here (docs/runbooks/retro-browser.md).
"""
from fastapi import FastAPI

from scripts.retro_routes import configured_router

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(configured_router())


@app.get('/api/retro-gateway/health')
async def health():
    return {'status': 'ok'}
