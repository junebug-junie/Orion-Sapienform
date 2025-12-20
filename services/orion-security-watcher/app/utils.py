from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def redact_url(url: str) -> str:
    """
    Remove `user:pass@` credentials from URLs for safe logging / email.

    Examples:
      http://user:pass@10.0.0.2:7100/snapshot.jpg -> http://10.0.0.2:7100/snapshot.jpg
      rtsp://admin:pw@cam.local:554/stream1       -> rtsp://cam.local:554/stream1

    If the input is empty, not a URL, or has no credentials, returns it unchanged.
    """
    if not url:
        return url

    try:
        parts = urlsplit(url)
        netloc = parts.netloc

        # netloc can be like "user:pass@host:port"
        if "@" in netloc:
            netloc = netloc.split("@", 1)[1]

        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        return url
