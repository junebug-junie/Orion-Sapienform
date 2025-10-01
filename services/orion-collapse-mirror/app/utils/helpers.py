from datetime import datetime, timezone
import socket
import os
import platform


def get_current_timestamp() -> str:
    """Returns ISO-formatted current timestamp in UTC with offset."""
    return datetime.now(timezone.utc).isoformat()


def get_environment_info() -> str:
    """Returns a description of the current runtime environment."""
    uname = platform.uname()
    hostname = socket.gethostname()
    user = os.getenv("USER") or os.getenv("USERNAME") or "unknown"
    return f"{uname.system} {uname.release} on host '{hostname}' ({uname.node}) under user '{user}'"
