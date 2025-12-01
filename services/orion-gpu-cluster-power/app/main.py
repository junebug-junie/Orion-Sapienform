# services/orion-psu-proxy/app/main.py

from .api import create_app

app = create_app()
