#!/usr/bin/env python3
"""
§8.1-style check: POST minimal OTLP/HTTP JSON to the collector, then confirm Tempo serves the trace.

Requires the observability stack running (same defaults as smoke_otel_phase1.sh):
  OTLP_HTTP=http://127.0.0.1:4318  TEMPO_HTTP=http://127.0.0.1:3200

Uses stdlib only. OTLP/HTTP JSON uses lowercase hex strings for ``traceId`` / ``spanId``
(as accepted by opentelemetry-collector-contrib HTTP receiver).
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    otlp_http = os.environ.get("OTLP_HTTP_URL", "http://127.0.0.1:4318").rstrip("/")
    tempo_http = os.environ.get("TEMPO_HTTP_URL", os.environ.get("TEMPO_HTTP", "http://127.0.0.1:3200")).rstrip("/")

    trace_hex = secrets.token_hex(16)
    span_hex = secrets.token_hex(8)

    body = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "e2e-otel-phase1"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {},
                        "spans": [
                            {
                                "traceId": trace_hex,
                                "spanId": span_hex,
                                "name": "e2e-smoke-span",
                                "kind": 1,
                                "startTimeUnixNano": "1000",
                                "endTimeUnixNano": "2000",
                            }
                        ],
                    }
                ],
            }
        ]
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{otlp_http}/v1/traces",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status not in (200, 202):
                print(f"e2e_otel_phase1: OTLP POST unexpected status {resp.status}", file=sys.stderr)
                return 1
    except urllib.error.HTTPError as exc:
        print(f"e2e_otel_phase1: OTLP POST failed: {exc}", file=sys.stderr)
        try:
            print(exc.read().decode("utf-8", errors="replace")[:2000], file=sys.stderr)
        except Exception:
            pass
        return 1
    except urllib.error.URLError as exc:
        print(f"e2e_otel_phase1: OTLP POST unreachable ({otlp_http}): {exc}", file=sys.stderr)
        return 1

    trace_url = f"{tempo_http}/api/traces/{trace_hex}"
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(trace_url, timeout=5) as resp:
                if resp.status == 200:
                    raw = resp.read().decode("utf-8", errors="replace")
                    if resp.status == 200 and ("e2e-smoke-span" in raw or trace_hex.lower() in raw.lower()):
                        print(f"e2e_otel_phase1: ok trace_id={trace_hex}")
                        return 0
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                time.sleep(0.5)
                continue
            print(f"e2e_otel_phase1: Tempo query error {exc}", file=sys.stderr)
            return 1
        except urllib.error.URLError:
            time.sleep(0.5)
            continue

    print(f"e2e_otel_phase1: timeout waiting for trace in Tempo ({trace_url})", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
