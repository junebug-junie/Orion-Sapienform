#!/usr/bin/env bash
set -euo pipefail

HUB_URL="${HUB_URL:-http://localhost:8080}"
WS_URL="${WS_URL:-ws://localhost:8080/ws}"
TIMEOUT="${TIMEOUT:-20}"
FAIL=0

pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; FAIL=1; }

echo "[1/2] /api/chat council mode includes council artifacts"
chat_payload='{"messages":[{"role":"user","content":"Give me a short council deliberation."}],"mode":"council","use_recall":false}'
chat_json="$(curl -fsS --max-time "$TIMEOUT" -H 'content-type: application/json' -d "$chat_payload" "$HUB_URL/api/chat")" || { fail "Council /api/chat request failed"; chat_json=''; }
if [[ -n "$chat_json" ]]; then
  has_council="$(printf '%s' "$chat_json" | python - <<'PY'
import json,sys
obj=json.load(sys.stdin)
raw=obj.get('raw') or {}
steps=raw.get('steps') or []
ok=False
for step in steps:
    res=step.get('result') if isinstance(step,dict) else None
    if isinstance(res,dict) and isinstance(res.get('CouncilService'),dict):
        ok=True
        break
print('1' if ok else '0')
PY
)"
  if [[ "$has_council" == "1" ]]; then pass "/api/chat raw includes CouncilService step"; else fail "No CouncilService payload found in /api/chat raw.steps"; fi
fi

echo "[2/2] WebSocket council mode emits council_debug when available"
ws_output="$(python - <<'PY'
import asyncio, json, os, sys
ws_url = os.environ.get('WS_URL', 'ws://localhost:8080/ws')
try:
    import websockets
except Exception:
    print('NO_WEBSOCKETS_LIB')
    raise SystemExit(0)

async def main():
    try:
        async with websockets.connect(ws_url, open_timeout=10, close_timeout=5) as ws:
            await ws.send(json.dumps({
                'mode': 'council',
                'session_id': 'smoke-council-debug',
                'text_input': 'Provide a quick council response with rationale.',
                'use_recall': False,
            }))
            for _ in range(8):
                msg = await asyncio.wait_for(ws.recv(), timeout=12)
                payload = json.loads(msg)
                if payload.get('llm_response'):
                    print('HAS_DEBUG' if isinstance(payload.get('council_debug'), dict) else 'NO_DEBUG')
                    return
            print('NO_RESPONSE')
    except Exception:
        print('WS_ERROR')

asyncio.run(main())
PY
)"
case "$ws_output" in
  HAS_DEBUG)
    pass "WS emitted council_debug"
    ;;
  NO_WEBSOCKETS_LIB)
    fail "Python websockets library unavailable"
    ;;
  NO_DEBUG)
    fail "WS response missing council_debug"
    ;;
  NO_RESPONSE)
    fail "WS did not return llm_response in time"
    ;;
  *)
    fail "WS council debug check failed (${ws_output})"
    ;;
esac

if [[ "$FAIL" -ne 0 ]]; then
  echo "SMOKE RESULT: FAIL"
  exit 1
fi

echo "SMOKE RESULT: PASS"
