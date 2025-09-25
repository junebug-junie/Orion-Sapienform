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
        # Always inject defaults if missing
        kwargs.setdefault("timestamp", get_current_timestamp())
        kwargs.setdefault("environment", get_environment_info())

        # Build entry (with validation) and auto-dump
        entry = CollapseMirrorEntry(**kwargs)
        try:
            response = requests.post(
                f"{self.base_url}/api/log/collapse",
                json=entry.model_dump(),   # <- use model_dump() (Pydantic v2)
                timeout=10
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to log collapse: {e}") from e
        return response.json()

    def query_collapse(self, prompt: str, **filters):
        params = {"prompt": prompt, **filters}
        response = requests.get(f"{self.base_url}/api/log/query", params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def __repr__(self):
        return f"<ConjourneyClient base_url={self.base_url}>"
