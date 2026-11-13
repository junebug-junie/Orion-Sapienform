"""Loopback-only rendered Hub/static smoke; no workers, config, or stores."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

HUB = Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            content = Environment(loader=FileSystemLoader(HUB / 'templates')).get_template('index.html').render(
                HUB_UI_ASSET_VERSION='cabinet-eval', HUB_CFG='{}', NOTIFY_TOAST_SECONDS='8',
                HUB_MEMORY_STORE_READY='false', HUB_MEMORY_STORE_BANNER_CLASS='hidden',
            ).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(content)
        else:
            super().do_GET()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18097)
    args = parser.parse_args()
    ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(HUB))).serve_forever()
