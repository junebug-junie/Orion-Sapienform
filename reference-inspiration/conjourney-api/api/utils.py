from datetime import datetime
import socket
import os
import platform

def get_current_timestamp() -> str:
    """Returns ISO-formatted current timestamp with timezone offset."""
    return datetime.now().astimezone().isoformat()

def get_environment_info() -> str:
    """Returns a simple description of the current environment."""
    hostname = socket.gethostname()
    user = os.getenv("USER") or os.getenv("USERNAME") or "unknown"
    system = platform.system()
    node = platform.node()
    return f"{system} host '{hostname}' ({node}) under user '{user}'"
