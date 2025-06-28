import requests
from api.collapse import CollapseMirrorEntry
from api.utils import get_current_timestamp, get_environment_info
import os
from dotenv import load_dotenv

load_dotenv()

class ConjourneyClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("CHRONICLE_API_BASE_URL", "http://localhost:8086")


    def log_collapse(self, **kwargs):
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = get_current_timestamp()
        if "environment" not in kwargs:
            kwargs["environment"] = get_environment_info()

        entry = CollapseMirrorEntry(**kwargs)
        response = requests.post(
            f"{self.base_url}/api/log/collapse",
            json=entry.dict()
        )
        response.raise_for_status()
        return response.json()

    def query_collapse(self, prompt: str, **filters):
        params = {"prompt": prompt, **filters}
        response = requests.get(f"{self.base_url}/api/log/query", params=params)
        response.raise_for_status()
        return response.json()


