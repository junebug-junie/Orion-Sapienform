"""Hub network-boundary Retro access and same-origin UDS proxy; no chat/bus path."""
import asyncio
from pathlib import Path
import time
import uuid

import aiohttp
from fastapi import APIRouter, Request, WebSocket, HTTPException
from fastapi.responses import FileResponse, JSONResponse


COPLAY_POLL_SEC = 2.0


def create_router(config, presence=None, experience=None):
    """``presence``: a scripts.retro_presence.PresenceStore (tests/evals); default is the
    process's Redis store from ORION_BUS_URL, or none (human play never depends on it).
    ``experience``: a scripts.retro_experience.ZeldaExperience (action bets); default from config
    (HUB_ACTION_BETS_ENABLED / HUB_PREDICTION_URL)."""
    router = APIRouter()
    if experience is None:
        from scripts.retro_experience import ZeldaExperience
        experience = ZeldaExperience.from_config(config)
    sockets = set()
    planning = False
    last_plan = float("-inf")
    background = set()

    def store():
        if presence is not None: return presence
        from scripts.retro_presence import store_for
        try: return store_for(config)
        except Exception: return None

    def soon(coro_fn, *args):
        # Presence writes ride beside human play, never in front of it.
        s = store()
        if s is None: return
        task = asyncio.create_task(getattr(s, coro_fn)(*args))
        background.add(task); task.add_done_callback(background.discard)

    def enabled():
        if not config.HUB_RETRO_ENABLED: raise HTTPException(503, 'retro_disabled')

    def origin(headers):
        expected = config.HUB_RETRO_ORIGIN.rstrip('/')
        return bool(expected) and headers.get('origin') == expected

    def credentials():
        value = Path(config.HUB_RETRO_AUTH_DIR, 'hub.token').read_text().strip()
        if len(value)<32: raise HTTPException(503, 'retro_credential_missing')
        return {'Authorization':'Bearer '+value, 'Content-Type':'application/json'}

    def client():
        return aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=config.HUB_RETRO_SOCKET),
                                     timeout=aiohttp.ClientTimeout(total=5))

    async def authorize(request):
        enabled()
        if request.method != 'GET' and not origin(request.headers): raise HTTPException(403, 'wrong_origin')

    async def body(request):
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data)>4096: raise HTTPException(413, 'message_too_large')
        return bytes(data)

    @router.get('/retro')
    async def page():
        enabled()
        return FileResponse(Path(__file__).resolve().parents[1]/'templates/retro.html',
                            headers={'Cache-Control':'no-store', 'Referrer-Policy':'no-referrer'})

    def handoff(app, sid, raw, result):
        # A browser handed its port to Orion's goal motor: watch that grant to its end.
        import json
        try: data = json.loads(raw)
        except ValueError: return
        if not (isinstance(data, dict) and data.get('action') == 'start' and data.get('cognition') is True
                and data.get('connection_id')): return
        policy = next((p for p in (result or {}).get('scripted') or []
                       if isinstance(p, dict) and p.get('port') == data.get('port')), None)
        if not policy or not isinstance(policy.get('epoch'), int): return
        key = (sid, policy['port'], policy['epoch'])
        if key in watching: return
        watching.add(key)
        coplay_counts[key] = {}
        task = asyncio.create_task(watch_coplay(app, sid, policy['port'], policy['epoch'], time.time()))
        background.add(task); task.add_done_callback(background.discard)

    @router.get('/api/retro/leisure')
    async def leisure():
        enabled()
        s = store()
        status = await s.status() if s is not None else None
        journal = await s.journal(10) if s is not None else None
        if status is None:
            return JSONResponse({'state': 'unknown', 'reason': 'presence_store_unavailable', 'journal': []},
                                headers={'Cache-Control': 'no-store'})
        status = dict(status or {'state': 'idle'})
        # The runner rewrites its status every poll (1 s). A live state that stopped updating
        # is a crashed or hung runner, not a session to watch.
        if status.get('state') in ('starting', 'playing', 'waiting') and \
                time.time() - float(status.get('updated_at') or 0) > 15:
            status = dict(status, state='stale', stale_state=status.get('state'))
        return JSONResponse(dict(status, journal=journal or []), headers={'Cache-Control': 'no-store'})

    histories = {}  # session id -> last goal_finished rows (Retro shows only the newest)
    coplay_counts = {}  # session id -> {'sheds': n, 'rejections': n} from plan_v2
    watching = set()    # (sid, port, epoch) co-play handoffs being watched to their end

    async def emit_session_end(app, ended):
        sink = getattr(getattr(app, 'state', None), 'retro_session_end_sink', None)
        if sink is None:
            return  # no exporter registered: nothing is written (integration seam, not a fallback)
        try:
            result = sink(ended)
            if asyncio.iscoroutine(result): await result
        except Exception:
            import logging
            logging.getLogger('orion-hub.retro').warning('retro_session_end_sink_failed', exc_info=True)

    async def watch_coplay(app, sid, port, epoch, started_at):
        """Follow one browser cognition handoff until Retro ends it (stop, takeover, expiry,
        disconnect), then hand a RetroSessionEnded to the registered sink. Zelda only."""
        from scripts import retro_planner
        from scripts.retro_session_end import (RetroSessionEnded, atlas_session_fields, human_held_frames,
                                               record_delta)
        key = (sid, port, epoch)
        history, goals, last, first, failures, reason = [], set(), None, None, 0, 'policy_ended'
        last_decision = None
        try:
          try:
            while True:
                await asyncio.sleep(COPLAY_POLL_SEC)
                try:
                    async with client() as c:
                        async with c.get('http://retro/sessions/'+sid, headers=credentials()) as r:
                            status, state = r.status, await r.json()
                except Exception:
                    failures += 1
                    if failures * COPLAY_POLL_SEC >= 30: reason = 'retro_unreachable'; break
                    continue
                failures = 0
                if status != 200 or not isinstance(state, dict) or state.get('state') not in ('running', 'paused'):
                    reason = 'session_ended'; break
                last = state
                if first is None: first = atlas_session_fields(state)   # Retro's record at handoff
                policy = next((p for p in state.get('scripted') or []
                               if isinstance(p, dict) and p.get('port') == port and p.get('epoch') == epoch), None)
                if policy is None:
                    done = history[-1].get('outcome', '') if history else ''
                    reason = done[len('stopped:'):] if done.startswith('stopped:') else 'policy_ended'
                    break
                cognition = policy.get('cognition') or {}
                if not retro_planner.is_zelda_cognition(cognition):
                    return  # Nova goal experiment: no session record in this arc
                retro_planner.remember_finished(history, cognition)
                if isinstance(cognition.get('attempt'), dict):
                    last_decision = cognition['attempt']
                if isinstance(cognition.get('goal'), dict) and cognition['goal'].get('decision_id'):
                    goals.add(cognition['goal']['decision_id'])
            if last is None:
                return  # never saw a Zelda cognition policy: nothing observed, nothing to record
            # Action bets: nothing of this handoff stays open; then read back what was learned.
            await experience.end_session(sid, history, last_decision)
            learned = tuple(await experience.learned_lines(sid, since=started_at))
            # Retro's counts are session-wide; report only what happened during this handoff.
            record = record_delta(first, atlas_session_fields(last))
            held = human_held_frames(record['controller_frames'])
            verdicts = {'met': 0, 'refuted': 0, 'unknown': 0}
            for h in history:
                v = h.get('verdict'); verdicts[v if v in verdicts else 'unknown'] += 1
            counts = coplay_counts.get(key, {})
            atlas = ((last.get('zelda') or {}).get('atlas') or {}) if isinstance(last.get('zelda'), dict) else {}
            ended = RetroSessionEnded(
                session_id=sid, lineage_id=atlas.get('lineage_id') if isinstance(atlas, dict) else None,
                leisure_id=None, game_id=str(last.get('game_id') or ''), started_at=started_at,
                ended_at=time.time(), end_reason=reason, controller='co_play',
                loaded_save_id=last.get('loaded_save'), decisions_planned=len(goals),
                planner_rejections=counts.get('rejections', 0), sheds=counts.get('sheds', 0),
                # A human handed off and can take back at any time: unknown counts mean "maybe".
                human_may_be_in_control=held is not False, verdicts=verdicts, learned_lines=learned, **record)
            await emit_session_end(app, ended)
          except Exception:   # never die silently: an unretrieved task exception logs nothing
            import logging
            logging.getLogger('orion-hub.retro').warning('retro_coplay_watch_failed sid=%s', sid, exc_info=True)
        finally:
            watching.discard(key)
            coplay_counts.pop(key, None)

    async def plan_v2(path, data, state, policy, c):
        """Zelda co-play decision (spec D). No goal on any failure; Retro's motor holds still."""
        from collections import deque
        from scripts import retro_planner
        cognition = policy['cognition']
        sid = path.rsplit('/', 1)[0]
        handoff_key = (sid.rsplit('/', 1)[-1], data['port'], data['expected_epoch'])
        def bump(name):
            # Only for a handoff being watched; keyed per grant so a newer handoff never mixes in.
            if handoff_key in coplay_counts:
                coplay_counts[handoff_key][name] = coplay_counts[handoff_key].get(name, 0) + 1
        if sid not in histories:
            histories.clear()  # one game at a time: keep only the current session's history
            histories[sid] = deque(maxlen=retro_planner.HISTORY_LIMIT)
        history = retro_planner.remember_finished(histories[sid], cognition)
        decision = cognition.get('attempt')
        accepted = cognition.get('goal')
        if isinstance(decision, dict) and isinstance(accepted, dict) and accepted.get('decision_id') == decision.get('decision_id'):
            return JSONResponse(accepted)
        atlas = None
        async with c.get('http://retro/'+sid+'/atlas', headers=credentials()) as r:
            if r.status == 200: atlas = await r.json()
        retro_sid = sid.rsplit('/', 1)[-1]
        # Action bets: settle the previous goal's bet, quote this decision's options (None when
        # off or the prediction service is unreachable: the plain #2870 prompt, and no bet).
        menu = await experience.prepare(retro_sid, decision, atlas)
        try:
            goal = await retro_planner.plan_zelda(config, decision, atlas, list(history),
                                                  **({'menu': menu.for_planner()} if menu else {}))
        except retro_planner.PlannerShed as exc:
            bump('sheds')
            return JSONResponse(exc.public(), status_code=503,
                                headers={'Retry-After': str(int(max(1, exc.details.get('retry_after_sec') or 30)))})
        except retro_planner.PlannerBusy as exc:
            return JSONResponse(exc.public(), status_code=429)
        except retro_planner.PlannerRejected as exc:
            if exc.reason not in ('decision_pending', 'observation_unknown'):
                bump('rejections')
                import logging
                logging.getLogger('orion-hub.retro').info('retro_plan_rejected sid=%s decision_id=%s reason=%s details=%s',
                                                          sid, (decision or {}).get('decision_id'), exc.reason, exc.details)
            return JSONResponse(exc.public(), status_code=409 if exc.reason in ('decision_pending', 'observation_unknown') else 422)
        except retro_planner.PlannerError as exc:
            return JSONResponse(exc.public(), status_code=503)
        body = data | retro_planner.goal_v2_body(goal)
        async with c.post('http://retro/'+sid+'/goal', headers=credentials(), json=body) as r:
            result, status = await r.json(), r.status
        if status == 200:
            await experience.placed(retro_sid, decision, atlas, goal, menu)
        return JSONResponse(result, status_code=status)

    @router.api_route('/api/retro/{path:path}', methods=['GET','POST'])
    async def proxy(path: str, request: Request):
        nonlocal planning, last_plan
        import re
        await authorize(request)
        # Every proxied POST is a human action (the leisure runner talks to Retro directly).
        if request.method == 'POST': soon('mark_human')
        if request.method == 'POST' and re.fullmatch(r'sessions/[0-9a-f-]{36}/plan', path):
            if not getattr(config, 'HUB_RETRO_COGNITION_ENABLED', False):
                raise HTTPException(503, 'retro_cognition_disabled')
            if planning: raise HTTPException(429, 'planner_busy')
            import json
            try:
                data = json.loads(await body(request))
                if (not isinstance(data, dict) or set(data) != {'connection_id', 'generation', 'port', 'expected_epoch'}
                        or type(data['port']) is not int or data['port'] not in (0, 1)
                        or type(data['expected_epoch']) is not int
                        or not all(isinstance(data[k], str) and len(data[k]) == 36
                                   for k in ('connection_id', 'generation'))):
                    raise ValueError()
            except (ValueError, TypeError): raise HTTPException(422, 'invalid_plan_request')
            planning = True
            try:
                from scripts.retro_planner import choose
                async with client() as c:
                    async with c.get('http://retro/'+path.rsplit('/', 1)[0], headers=credentials()) as r:
                        state = await r.json()
                        if r.status != 200: return JSONResponse(state, status_code=r.status)
                    policy = next((p for p in state.get('scripted', []) if p['port'] == data['port']
                                   and p['epoch'] == data['expected_epoch'] and p.get('cognition')), None)
                    if not policy or state['generation'] != data['generation']:
                        raise HTTPException(409, 'stale_controller')
                    from scripts.retro_planner import is_zelda_cognition
                    if is_zelda_cognition(policy['cognition']):
                        return await plan_v2(path, data, state, policy, c)
                    attempt = policy['cognition']['attempt']
                    if not attempt: raise HTTPException(409, 'observation_pending')
                    accepted = policy['cognition'].get('goal')
                    if accepted and accepted['attempt_id'] == attempt['attempt_id']:
                        return JSONResponse(accepted)
                    if time.monotonic()-last_plan < 5:
                        raise HTTPException(429, 'planner_rate_limited')
                    last_plan = time.monotonic()
                    choice = await choose(config, attempt)
                    goal = data | choice | {'attempt_id': attempt['attempt_id']}
                    async with c.post('http://retro/'+path.rsplit('/', 1)[0]+'/goal',
                                      headers=credentials(), json=goal) as r:
                        return JSONResponse(await r.json(), status_code=r.status)
            except HTTPException: raise
            except Exception: raise HTTPException(503, 'planner_unavailable_or_invalid')
            finally: planning = False
        valid = ((request.method=='GET' and path in ('games','health/ready')) or
                 (request.method=='GET' and re.fullmatch(r'games/[a-z0-9][a-z0-9-]{0,63}/saves',path)) or
                 (request.method=='POST' and path=='sessions') or
                 re.fullmatch(r'sessions/[0-9a-f-]{36}'+('(?:/evidence|/atlas)?' if request.method=='GET' else '/(?:control|pause|resume|stop|scripted|goal|save|load)'),path))
        if not valid: raise HTTPException(404)
        raw = await body(request)
        try:
            async with client() as c:
                async with c.request(request.method,'http://retro/'+path,headers=credentials(),data=raw) as r:
                    result=await r.json()
                    if request.method=='POST' and r.status==200 and path.endswith('/scripted'):
                        handoff(request.app, path.split('/')[1], raw, result)
                    return JSONResponse(result,status_code=r.status,headers={'Cache-Control':'no-store'})
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError): raise HTTPException(503, 'retro_unavailable')

    @router.websocket('/api/retro/sessions/{sid}/{channel}')
    async def relay(ws: WebSocket, sid: str, channel: str):
        import re
        if (not config.HUB_RETRO_ENABLED or not origin(ws.headers)
                or channel not in ('input','media') or not re.fullmatch(r'[0-9a-f-]{36}',sid) or len(sockets)>=8):
            await ws.close(code=1008);return
        sockets.add(ws)
        tasks=[]
        socket_id=str(uuid.uuid4())
        try:
            async with client() as c:
                async with c.ws_connect('http://retro/sessions/'+sid+'/'+channel,headers=credentials(),max_msg_size=1200000) as upstream:
                    await ws.accept()
                    async def receive():
                        count=0;start=time.monotonic()
                        while True:
                            try: msg=await asyncio.wait_for(ws.receive_text(),.5)
                            except asyncio.TimeoutError: continue
                            if time.monotonic()-start>=1: start=time.monotonic();count=0
                            count+=1
                            if channel=='media' or len(msg.encode())>4096 or count>120: return
                            await asyncio.wait_for(upstream.send_str(msg),.25)
                    async def send():
                        while True:
                            try: msg=await asyncio.wait_for(upstream.receive(),.5)
                            except asyncio.TimeoutError: continue
                            if msg.type==aiohttp.WSMsgType.BINARY:
                                await asyncio.wait_for(ws.send_bytes(msg.data),.25)
                            elif msg.type==aiohttp.WSMsgType.TEXT:
                                await asyncio.wait_for(ws.send_text(msg.data),.25)
                            else: return
                    if channel=='input': soon('mark_human')
                    async def beat():
                        # Presence heartbeat for the leisure gate; never ends the relay.
                        while True:
                            s=store()
                            if s is not None:
                                try: await s.beat_browser(socket_id)
                                except Exception: pass
                            await asyncio.sleep(5)
                    tasks=[asyncio.create_task(receive()),asyncio.create_task(send())]
                    beating=asyncio.create_task(beat())
                    try: await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        beating.cancel();await asyncio.gather(beating,return_exceptions=True)
        except (aiohttp.ClientError, OSError, asyncio.TimeoutError, RuntimeError): pass
        finally:
            sockets.discard(ws)
            soon('drop_browser',socket_id)
            for task in tasks: task.cancel()
            if tasks: await asyncio.gather(*tasks,return_exceptions=True)
            try: await ws.close()
            except RuntimeError: pass
    return router


def configured_router():
    from app.settings import settings
    return create_router(settings)
