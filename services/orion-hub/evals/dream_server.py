#!/usr/bin/env python3
"""Isolated Dream browser smoke server. No Hub workers or database writes.

Without --env-file, use browser fixtures. With it, read real stores through
only the new GET endpoints. Bind loopback; never deploy this eval server.
"""
from pathlib import Path
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--port', type=int, default=18091)
    args = parser.parse_args()
    hub = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(hub.parents[1]))
    sys.path.insert(0, str(hub))
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file)
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from jinja2 import Environment, FileSystemLoader
    from scripts.dream_routes import router
    app = FastAPI()
    app.include_router(router)
    app.mount('/static', StaticFiles(directory=hub / 'static'))

    @app.get('/', response_class=HTMLResponse)
    def index():
        return Environment(loader=FileSystemLoader(hub / 'templates')).get_template('index.html').render(
            HUB_UI_ASSET_VERSION='dream-eval', NOTIFY_TOAST_SECONDS='8',
            HUB_CFG='{"apiBaseOverride":"","wsBaseOverride":""}',
            HUB_AUTONOMY_SUBJECT_DISPLAY='Orion', HUB_MEMORY_STORE_BANNER_CLASS='hidden',
            HUB_MEMORY_STORE_READY='false',
        )

    uvicorn.run(app, host='127.0.0.1', port=args.port, access_log=False)


if __name__ == '__main__':
    main()
